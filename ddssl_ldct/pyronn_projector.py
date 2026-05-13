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

    def __init__(self, geometry: FanBeamGeometry, redundancy: str = "auto"):
        """
        redundancy:
            "auto"      Pick by angular_range. ≥ 2π → "full_scan"; otherwise
                        Parker (short scan).
            "full_scan" Uniform constant = angular_range / π across all
                        (angle, detector) pairs. Correct for 2π scans —
                        every ray is sampled twice, no per-view tapering
                        is needed and Parker's endpoint taper would
                        wrongly down-weight the first / last few views to
                        zero.
            "parker"    Use ``pyronn.ct_reconstruction.helpers.filters.
                        weights.parker_weights_2d`` — the Parker-style
                        redundancy weighting for short scans (angular
                        range ≈ π + fan_angle).
            "none"      No redundancy weighting at all (debug / scaling
                        sweeps). FBP magnitude will be off by a constant
                        factor.
        """
        super().__init__()
        _import_pyronn()
        from pyronn.ct_reconstruction.geometry.geometry_base import GeometryFan2D
        from pyronn.ct_reconstruction.layers.torch.projection_2d import FanProjection2D
        from pyronn.ct_reconstruction.layers.torch.backprojection_2d import FanBackProjection2D
        from pyronn.ct_reconstruction.helpers.trajectories.circular_trajectory import (
            circular_trajectory_2d,
        )

        self.geom = geometry
        self.redundancy = redundancy
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

        # Redundancy weights for FBP. See the class docstring's `redundancy`
        # parameter for what each mode means. For a 2π full scan (the Wagner
        # default + every Pentathlon challenge today), the right thing is a
        # uniform constant — Parker's endpoint taper would wrongly zero the
        # first/last few views, even though its trailing `scale_factor` of
        # ~2 gets the global magnitude right. A flat angular_range/π
        # produces the same global scaling without that boundary defect.
        ang_range = float(g.angle_end - g.angle_start)
        mode = self.redundancy
        if mode == "auto":
            mode = "full_scan" if ang_range >= 2 * math.pi - 1e-3 else "parker"
        self._redundancy_mode = mode
        n_a, n_d = g.n_angles, g.n_det
        if mode == "full_scan":
            const = ang_range / math.pi
            rw = np.full((n_a, n_d), const, dtype=np.float32)
        elif mode == "parker":
            from pyronn.ct_reconstruction.helpers.filters.weights import (
                parker_weights_2d,
            )
            rw = parker_weights_2d(py_geom).astype(np.float32)
        elif mode == "none":
            rw = np.ones((n_a, n_d), dtype=np.float32)
        else:
            raise ValueError(f"unknown redundancy mode {mode!r}")
        self.register_buffer(
            "_redundancy_weights",
            torch.from_numpy(rw).cuda().contiguous(),
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

        Order: redundancy weights → ramp/Hann filter → back-project, per the
        upstream ``example_fan_2d.py`` recipe. See class docstring for the
        choice of redundancy weighting (default is mode-aware: a flat
        ``angular_range / π`` for 2π full scan, Parker otherwise).
        """
        A = sino.shape[-2]
        if A == self._redundancy_weights.shape[0]:
            rw = self._redundancy_weights
        else:
            # Half-set or otherwise non-full angle count — rebuild for this A.
            g = self.geom
            ang_range = float(g.angle_end - g.angle_start)
            mode = self._redundancy_mode
            if mode == "full_scan":
                const = ang_range / math.pi
                rw_np = np.full((A, g.n_det), const, dtype=np.float32)
            elif mode == "parker":
                from pyronn.ct_reconstruction.geometry.geometry_base import GeometryFan2D
                from pyronn.ct_reconstruction.helpers.trajectories.circular_trajectory import (
                    circular_trajectory_2d,
                )
                from pyronn.ct_reconstruction.helpers.filters.weights import (
                    parker_weights_2d,
                )
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
                rw_np = parker_weights_2d(half).astype(np.float32)
            elif mode == "none":
                rw_np = np.ones((A, g.n_det), dtype=np.float32)
            else:
                raise ValueError(f"unknown redundancy mode {mode!r}")
            rw = torch.as_tensor(rw_np, dtype=sino.dtype, device=sino.device)
        sino_w = sino * rw  # broadcast over (B, [1,] A, D)
        return self.back_project(self.filter_sino(sino_w, filter_name=filter_name))

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.forward_project(image)
