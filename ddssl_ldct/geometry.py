"""2D fan-beam geometry matching Wagner et al. 2023 (arXiv:2211.01111).

Defaults reproduce the rebinned helical Mayo LDCT-and-Projection-data geometry
(Siemens SOMATOM Definition AS in the DICOM-CT-PD format), as used by the
`helix2fan` repository's reference reconstruction (`reco_example_fan_beam.py`):

  - image: 512 × 512 at 0.7 mm voxel pitch
  - detector: 736 channels, 1.2858 mm spacing (flat detector after rebinning)
  - source-isocentre distance (SOD / dso): 595.0 mm
  - source-detector distance (SDD / dsd):   1085.6 mm
  - 1152 views per full rotation (one-rotation full scan, 2π)

For the AAPM 2016 Low Dose CT Grand Challenge the slices are reconstructed at
1 mm thickness with these in-plane parameters (cf. Wagner 2022 Med. Phys.).
"""
from __future__ import annotations
from dataclasses import dataclass
import math
import torch


@dataclass
class FanBeamGeometry:
    image_size: int = 512
    pixel_spacing: float = 0.7              # mm per voxel (in-slice)
    n_angles: int = 1152                    # one full rotation
    n_det: int = 736                        # Siemens AS channels after rebinning
    det_spacing: float = 1.2858             # mm per detector pixel
    sod: float = 595.0                      # source-object distance, mm
    sdd: float = 1085.6                     # source-detector distance, mm
    angle_start: float = 0.0
    angle_end: float = 2 * math.pi          # full-scan

    @property
    def odd(self) -> float:
        return self.sdd - self.sod

    def angles(self, device=None, dtype=torch.float32) -> torch.Tensor:
        return torch.linspace(self.angle_start, self.angle_end, self.n_angles + 1,
                              device=device, dtype=dtype)[:-1]

    def det_u(self, device=None, dtype=torch.float32) -> torch.Tensor:
        n = self.n_det
        u = (torch.arange(n, device=device, dtype=dtype) - (n - 1) / 2.0) * self.det_spacing
        return u

    def split_angles(self) -> tuple["FanBeamGeometry", "FanBeamGeometry"]:
        """Even / odd angular subsets, used for Noise2Inverse-style splits."""
        even = FanBeamGeometry(**{**self.__dict__, "n_angles": self.n_angles // 2})
        odd = FanBeamGeometry(**{**self.__dict__, "n_angles": self.n_angles // 2})
        return even, odd

    # -- Mayo LDCT presets ----------------------------------------------------
    # The Mayo `LDCT-and-Projection-data` DICOM-CT-PD header reports a nominal
    # geometry (pixel_spacing=0.703125, sod=595.0, sdd=1085.6, det_spacing=
    # 1.285839). On L014 fulldose, a data-driven 5-parameter fit (scripts/
    # fit_fbp_geometry_L014.py, job 762284, 2026-05-26) found a measurable
    # mismatch: the actual recon geometry is ~0.32 % off in pixel_spacing
    # and ~0.1 % off in sdd. Using the fitted values in the FBP gives
    # +3.26 dB PSNR and -31 % RMSE at the peak GT slice (SSIM 0.94 → 0.9466).
    # The mismatch is most likely between Mayo's nominal ReconstructionDiameter
    # (360 mm rounded) and the actual scanner FoV (~358.84 mm).
    #
    # Use `FanBeamGeometry.mayo_ldct_fitted(...)` for the recommended Mayo
    # FBP geometry. Keep `mayo_ldct_nominal(...)` available for diagnostic
    # comparison against the DICOM header.

    @classmethod
    def mayo_ldct_fitted(cls, *, n_angles: int, n_det: int = 736,
                          angle_start: float = 0.0,
                          angle_end: float = 2 * math.pi) -> "FanBeamGeometry":
        """Mayo LDCT FBP geometry from the data-driven L2 fit on L014.

        These are the recommended defaults for all Mayo-LDCT FBP / FBP-based
        solver pipelines as of 2026-05-26. The fitted parameters minimise
        the calibrated L2 between FBP-cal and truth on the peak GT slice
        (job 762284).

        Fitted values:
          - pixel_spacing = 0.700857 mm (DICOM nominal: 0.703125, Δ = -0.32 %)
          - det_spacing   = 1.285044 mm (DICOM nominal: 1.285839, Δ = -0.06 %)
          - sod           = 595.362 mm  (DICOM nominal: 595.000, Δ = +0.06 %)
          - sdd           = 1086.803 mm (DICOM nominal: 1085.600, Δ = +0.11 %)
        Note: an additional sub-pixel detector_origin offset of −0.040 mm
        was found by the fit; that needs to be applied OUTSIDE this dataclass
        on PyronnFanBeamProjector._tensor_geom['detector_origin'] because
        PYRO-NN doesn't expose it as a constructor argument.
        """
        return cls(
            image_size=512,
            pixel_spacing=0.700857,
            n_angles=n_angles,
            n_det=n_det,
            det_spacing=1.285044,
            sod=595.362,
            sdd=1086.803,
            angle_start=angle_start,
            angle_end=angle_end,
        )

    # Sub-pixel detector centre offset (mm) recovered by the fit alongside
    # the four scalars above. Apply via
    #    proj = PyronnFanBeamProjector(FanBeamGeometry.mayo_ldct_fitted(...))
    #    proj._tensor_geom["detector_origin"] = (
    #        proj._tensor_geom["detector_origin"] + MAYO_LDCT_DET_OFFSET
    #    )
    # (The classmethod can't bake this in because PYRO-NN sets detector_origin
    # internally from volume_shape*spacing/2 in its constructor.)
    @classmethod
    def mayo_ldct_nominal(cls, *, n_angles: int, n_det: int = 736,
                           angle_start: float = 0.0,
                           angle_end: float = 2 * math.pi) -> "FanBeamGeometry":
        """Mayo LDCT FBP geometry using the DICOM-CT-PD header values.

        Kept for diagnostic comparison against `mayo_ldct_fitted`. Going
        forward, prefer the fitted variant — the nominal one is ~0.5 px
        off in the radial direction (pin-cushion residual visible as
        table-edge banding in the diff against truth).
        """
        return cls(
            image_size=512,
            pixel_spacing=0.703125,
            n_angles=n_angles,
            n_det=n_det,
            det_spacing=1.285839,
            sod=595.000,
            sdd=1085.600,
            angle_start=angle_start,
            angle_end=angle_end,
        )


# Sub-pixel detector-origin offset (mm) — recovered alongside the fitted
# geometry above. Apply at the projector level (see `mayo_ldct_fitted`
# docstring for the recipe).
MAYO_LDCT_DET_OFFSET = -0.0397
