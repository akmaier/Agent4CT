"""PYRO-NN backend for 2D fan-beam (flat detector) projection / FBP.

Reconciled against PYRO-NN master (https://github.com/csyben/PYRO-NN, v1.1.0,
torch backend). The torch layers live in
``pyronn.ct_reconstruction.layers.torch.{projection_2d, backprojection_2d}``
and ultimately call into the ``pyronn_layers_torch`` CUDA extension.

Real public API (observed in the upstream sources):

  * ``pyronn.ct_reconstruction.geometry.geometry_base.GeometryFan2D``
        Constructor:
          ``GeometryFan2D(volume_shape, volume_spacing,
                          detector_shape, detector_spacing,
                          number_of_projections, angular_range,
                          source_detector_distance, source_isocenter_distance)``
        After construction, ``geometry.set_trajectory(circular_trajectory_2d(...))``
        is required — the trajectory is the per-view central-ray unit vector.

  * ``pyronn.ct_reconstruction.helpers.trajectories.circular_trajectory.circular_trajectory_2d``
        ``circular_trajectory_2d(number_of_projections, angular_range, swap_detector_axis)``
        Returns ``np.ndarray`` of shape ``(A, 2)`` with ``[cos β, ±sin β]`` per view.

  * ``pyronn.ct_reconstruction.layers.torch.projection_2d.FanProjection2D``
        ``nn.Module`` whose ``.forward(input, **geometry_dict)`` calls
        ``FanProjection2DFunction.apply`` with the named CUDA-tensor kwargs:
            ``sinogram_shape, volume_origin, detector_origin, volume_spacing,
             detector_spacing, source_isocenter_distance, source_detector_distance,
             trajectory``.
        Expects ``input`` of shape ``(B, H, W)`` (no channel dim) on CUDA,
        returns ``(B, A, D)``.

  * ``pyronn.ct_reconstruction.layers.torch.backprojection_2d.FanBackProjection2D``
        Same convention; takes ``volume_shape`` instead of ``sinogram_shape``.

  * ``pyronn.ct_reconstruction.helpers.filters.filters``
        Provides ``ram_lak``, ``hann``, ``shepp_logan``, ``hamming``, ``cosine``
        and their ``*_2D`` tiled counterparts. The 2D variants take
        ``(detector_shape, detector_spacing, number_of_projections)`` and return
        an ``(A, D)`` filter in the frequency domain.

  * ``pyronn.ct_reconstruction.helpers.misc.general_utils.fft_and_ifft``
        Applies the frequency-domain filter (``torch.fft.fft`` along ``dim=-1``,
        multiply, ``ifft`` and take the real part). Uses ``norm='ortho'``.

Our training loop wants ``(B, 1, H, W) → (B, 1, A, D)``, so this wrapper inserts
and removes the singleton channel dimension and keeps everything on CUDA.
"""
from __future__ import annotations
import math
from typing import Optional

import numpy as np
import torch

from .geometry import FanBeamGeometry


# Lazy global to avoid importing CUDA modules at package import time.
_PYRONN_IMPORTED = False


def _import_pyronn():
    """Eager import of PYRO-NN. Raises ImportError off-CUDA."""
    global _PYRONN_IMPORTED
    if _PYRONN_IMPORTED:
        return
    try:
        import pyronn  # noqa: F401  (triggers backend selection / CONFIG.json read)
        # Force torch backend before pyronn_layers gets imported via filters etc.
        try:
            if pyronn.read_backend() != "torch":
                pyronn.set_backend("torch")
        except Exception:
            pass
        import pyronn_layers  # noqa: F401
        from pyronn.ct_reconstruction.geometry.geometry_base import GeometryFan2D  # noqa: F401
        from pyronn.ct_reconstruction.layers.torch.projection_2d import FanProjection2D  # noqa: F401
        from pyronn.ct_reconstruction.layers.torch.backprojection_2d import FanBackProjection2D  # noqa: F401
        from pyronn.ct_reconstruction.helpers.trajectories.circular_trajectory import circular_trajectory_2d  # noqa: F401
        from pyronn.ct_reconstruction.helpers.filters import filters as pyronn_filters  # noqa: F401
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "PYRO-NN is not installed or its CUDA extension cannot load. "
            "Build pyronn from https://github.com/csyben/PYRO-NN on a CUDA "
            "compute node (see cluster/setup.sh). The pure-PyTorch fallback "
            "has been removed — this codebase targets PYRO-NN only."
        ) from e
    _PYRONN_IMPORTED = True


class PyronnFanBeamProjector(torch.nn.Module):
    """Thin wrapper around PYRO-NN's fan-beam projector / back-projector / filters.

    The geometry is constructed at module init; the geometry's per-view
    trajectory array is built via the upstream ``circular_trajectory_2d`` helper
    so we stay byte-for-byte compatible with PYRO-NN's example.

    Geometry tensors are stored as a buffer dict (CUDA tensors, contiguous,
    float32) so the autograd ``Function.apply`` calls do not re-upload them on
    every step.
    """

    def __init__(self, geometry: FanBeamGeometry):
        super().__init__()
        _import_pyronn()
        from pyronn.ct_reconstruction.geometry.geometry_base import GeometryFan2D
        from pyronn.ct_reconstruction.layers.torch.projection_2d import FanProjection2D
        from pyronn.ct_reconstruction.layers.torch.backprojection_2d import FanBackProjection2D
        from pyronn.ct_reconstruction.helpers.trajectories.circular_trajectory import (
            circular_trajectory_2d,
        )

        self.geom = geometry
        g = geometry

        # Build the upstream geometry. Note constructor order:
        #   (volume_shape, volume_spacing, detector_shape, detector_spacing,
        #    number_of_projections, angular_range, sdd, sid)
        py_geom = GeometryFan2D(
            volume_shape=[g.image_size, g.image_size],
            volume_spacing=[g.pixel_spacing, g.pixel_spacing],
            detector_shape=[g.n_det],
            detector_spacing=[g.det_spacing],
            number_of_projections=g.n_angles,
            angular_range=[g.angle_start, g.angle_end],
            source_detector_distance=g.sdd,
            source_isocenter_distance=g.sod,
        )
        # circular trajectory matches the upstream 2d fan example; the third
        # argument flips the detector axis convention — keep it on (True) to
        # match the reference example_fan_2d.py.
        py_geom.set_trajectory(
            circular_trajectory_2d(
                py_geom.number_of_projections,
                py_geom.angular_range,
                True,
            )
        )
        self._py_geom = py_geom

        # Cache the geometry as a tensor-dict on CUDA. Mirrors the
        # high-level FanProjectionFor2D wrapper (which rebuilds them on every
        # call) but does it once at construction.
        tensor_geometry = {}
        for k, v in vars(py_geom).items():
            try:
                if hasattr(v, "__len__"):
                    t = torch.tensor(np.asarray(v), dtype=torch.float32)
                else:
                    t = torch.tensor([v], dtype=torch.float32)
            except (TypeError, ValueError):
                continue
            tensor_geometry[k] = t.cuda().contiguous()
        # Keep references on self so they live as long as the module.
        self._tensor_geom = tensor_geometry

        self._fp = FanProjection2D()
        self._bp = FanBackProjection2D()

        # Parker redundancy weights — the upstream example_fan_2d.py shows
        # this is mandatory for a full-scan fan-beam FBP to come out at the
        # right intensity. For 2π the scale_factor inside parker_weights_2d
        # is ≈ 2; without it the FBP is ~half the correct amplitude.
        #
        # NOTE: parker_weights_2d iterates beta starting at 0 and rising by
        # angular_increment per view (CCW), but ``circular_trajectory_2d(...,
        # swap_detector_axis=True)`` (the upstream `example_fan_2d.py` choice
        # we follow) sweeps the projector in the opposite direction. We
        # therefore flip the Parker matrix along the angle axis to bring the
        # weighting back into register — without this, the FBP shows a
        # rotation-dependent intensity asymmetry that an expert eye spots
        # immediately even though PSNR looks "fine".
        from pyronn.ct_reconstruction.helpers.filters.weights import (
            parker_weights_2d,
        )
        pw = parker_weights_2d(py_geom).astype(np.float32)
        pw = np.ascontiguousarray(pw[::-1, :])
        self.register_buffer(
            "_parker_weights",
            torch.from_numpy(pw).cuda().contiguous(),
            persistent=False,
        )

    # ---- helpers -----------------------------------------------------------

    def _ensure_3d(self, x: torch.Tensor) -> tuple[torch.Tensor, bool]:
        """Convert ``(B,1,H,W)`` → ``(B,H,W)`` (PYRO-NN convention).

        Returns the squeezed tensor and a flag noting whether the channel dim
        was present so we can restore it after the layer call.
        """
        had_channel = False
        if x.dim() == 4:
            assert x.shape[1] == 1, "PYRO-NN 2D layers expect a single-channel input"
            x = x.squeeze(1)
            had_channel = True
        elif x.dim() == 2:
            x = x[None]  # add batch
        return x.contiguous(), had_channel

    @staticmethod
    def _restore_channel(x: torch.Tensor, had_channel: bool) -> torch.Tensor:
        return x.unsqueeze(1) if had_channel else x

    def _geom_kwargs(self) -> dict[str, torch.Tensor]:
        # Filter to just the keys the layers consume (see FanProjection2D /
        # FanBackProjection2D forward signatures in PYRO-NN torch sources).
        keys = (
            "sinogram_shape",
            "volume_shape",
            "volume_origin",
            "detector_origin",
            "volume_spacing",
            "detector_spacing",
            "source_isocenter_distance",
            "source_detector_distance",
            "trajectory",
        )
        return {k: self._tensor_geom[k] for k in keys if k in self._tensor_geom}

    # ---- forward / back ----------------------------------------------------

    def forward_project(self, image: torch.Tensor) -> torch.Tensor:
        """``(B, 1, H, W) → (B, 1, A, D)``.

        ``image`` must be on CUDA and float32. Output keeps the channel dim.
        """
        if not image.is_cuda:
            image = image.cuda()
        if image.dtype != torch.float32:
            image = image.float()
        x, had_c = self._ensure_3d(image)
        sino = self._fp.forward(x, **self._geom_kwargs())
        return self._restore_channel(sino, had_c)

    def back_project(self, sino: torch.Tensor) -> torch.Tensor:
        """``(B, 1, A, D) → (B, 1, H, W)``."""
        if not sino.is_cuda:
            sino = sino.cuda()
        if sino.dtype != torch.float32:
            sino = sino.float()
        x, had_c = self._ensure_3d(sino)
        reco = self._bp.forward(x, **self._geom_kwargs())
        return self._restore_channel(reco, had_c)

    # ---- filter & FBP ------------------------------------------------------

    def filter_sino(self, sino: torch.Tensor, filter_name: str = "hann") -> torch.Tensor:
        """Frequency-domain ramp / Hann / etc. filter, applied along detector axis.

        Sino shape ``(B, 1, A, D)`` or ``(B, A, D)``. Filter shape ``(A, D)`` is
        broadcast across the batch (and channel) dimension.
        """
        from pyronn.ct_reconstruction.helpers.filters import filters as pyronn_filters

        g = self.geom
        # PYRO-NN's *_2D filters tile per-view; we apply per-batch so we only
        # need the 1D filter, but the existing helpers return a 2D (A,D) array
        # — fine, broadcast over batch axis.
        det_shape = [g.n_det]
        det_spacing = [g.det_spacing]
        if filter_name == "ramlak":
            f_np = pyronn_filters.ram_lak_2D(det_shape, det_spacing, g.n_angles)
        elif filter_name == "hann":
            f_np = pyronn_filters.hann_2D(det_shape, det_spacing, g.n_angles)
        elif filter_name == "shepp-logan":
            f_np = pyronn_filters.shepp_logan_2D(det_shape, det_spacing, g.n_angles)
        elif filter_name == "hamming":
            f_np = pyronn_filters.hamming_2D(det_shape, det_spacing, g.n_angles)
        elif filter_name == "cosine":
            f_np = pyronn_filters.cosine_2D(det_shape, det_spacing, g.n_angles)
        else:
            raise ValueError(f"unknown filter {filter_name!r}")

        # PYRO-NN's pre-built filters use the number of projections of the full
        # geometry. If the sino has a different angle count (e.g. half-set),
        # rebuild with that count.
        A_sino = sino.shape[-2]
        if f_np.shape[0] != A_sino:
            if filter_name == "ramlak":
                f_np = pyronn_filters.ram_lak_2D(det_shape, det_spacing, A_sino)
            elif filter_name == "hann":
                f_np = pyronn_filters.hann_2D(det_shape, det_spacing, A_sino)
            elif filter_name == "shepp-logan":
                f_np = pyronn_filters.shepp_logan_2D(det_shape, det_spacing, A_sino)
            elif filter_name == "hamming":
                f_np = pyronn_filters.hamming_2D(det_shape, det_spacing, A_sino)
            elif filter_name == "cosine":
                f_np = pyronn_filters.cosine_2D(det_shape, det_spacing, A_sino)

        f = torch.as_tensor(f_np, dtype=sino.dtype, device=sino.device)

        # Apply per-projection 1D FFT along the detector axis.
        spec = torch.fft.fft(sino, dim=-1, norm="ortho")
        spec = spec * f  # broadcast over batch / channel
        return torch.fft.ifft(spec, dim=-1, norm="ortho").real

    def fbp(self, sino: torch.Tensor, filter_name: str = "hann") -> torch.Tensor:
        """Filtered back-projection. Returns ``(B, 1, H, W)``.

        Order: Parker redundancy weights → ramp/Hann filter → back-project,
        per the upstream ``example_fan_2d.py`` recipe. Parker is required even
        for full 2π scans (its trailing ``scale_factor ≈ 2`` is what makes the
        FBP intensity match the phantom).
        """
        # Parker weights are (A, D) for the *full* geometry. If the input
        # sinogram has half the views (the Noise2Inverse split), recompute
        # them lazily for that view count.
        A = sino.shape[-2]
        if A == self._parker_weights.shape[0]:
            pw = self._parker_weights
        else:
            from pyronn.ct_reconstruction.geometry.geometry_base import GeometryFan2D
            from pyronn.ct_reconstruction.helpers.trajectories.circular_trajectory import (
                circular_trajectory_2d,
            )
            from pyronn.ct_reconstruction.helpers.filters.weights import (
                parker_weights_2d,
            )
            g = self.geom
            half = GeometryFan2D(
                volume_shape=[g.image_size, g.image_size],
                volume_spacing=[g.pixel_spacing, g.pixel_spacing],
                detector_shape=[g.n_det],
                detector_spacing=[g.det_spacing],
                number_of_projections=A,
                angular_range=[g.angle_start, g.angle_end],
                source_detector_distance=g.sdd,
                source_isocenter_distance=g.sod,
            )
            half.set_trajectory(circular_trajectory_2d(A, half.angular_range, True))
            pw_np = parker_weights_2d(half).astype(np.float32)
            # Same flip as in __init__ — keep the half-set Parker weights in
            # register with the swept rotation direction.
            pw_np = np.ascontiguousarray(pw_np[::-1, :])
            pw = torch.as_tensor(
                pw_np, dtype=sino.dtype, device=sino.device,
            )
        sino_w = sino * pw  # broadcast over (B, [1,] A, D)
        return self.back_project(self.filter_sino(sino_w, filter_name=filter_name))

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.forward_project(image)
