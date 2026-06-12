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
    # 1.285839). On L014 fulldose, a data-driven 5-parameter Powell fit
    # (scripts/fit_fbp_geometry_L014.py, job 762284, 2026-05-26) found a
    # measurable mismatch — the actual FBP geometry is ~0.32 % off in
    # pixel_spacing and ~0.1 % off in sdd. Using the fitted values in the
    # FBP gives +3.26 dB PSNR and -31 % RMSE at the peak GT slice (SSIM
    # 0.94 → 0.9466).
    #
    # **Important — SSR step has DIFFERENT defaults than FBP step:**
    # The helical→fan rebin (SSR) and the back-projection (FBP) are two
    # SEPARATE stages with two SEPARATE (sod, sdd) pairs that the joint
    # fit can move independently. The multi-GT joint Adam fit (SLURM
    # 762369, 2026-05-27) holds the FBP geometry FIXED at the Powell
    # values below and optimises ONLY the SSR sod/sdd, finding the SSR
    # optimum at (593.461, 1086.831). Empirically validated by ablation
    # (SLURM 762403 / 762404, 2026-05-27): replacing the FBP Powell
    # values with the SSR multi-GT values DEGRADES SSIM by ~0.018 at
    # the same pixel_sp and shifts the optimum off the Powell value.
    #
    # See `MAYO_LDCT_SSR_DEFAULTS` below for the SSR-step values used by
    # `scripts/fit_rebin_end2end_L014.py`, `cache_proj_flat_L014.py`, and
    # the ablation scripts. Keep these distinct.
    #
    # Use `FanBeamGeometry.mayo_ldct_fitted(...)` for the recommended Mayo
    # FBP geometry. Keep `mayo_ldct_nominal(...)` available for diagnostic
    # comparison against the DICOM header.

    @classmethod
    def mayo_ldct_fitted(cls, *, n_angles: int, n_det: int = 736,
                          angle_start: float = 0.0,
                          angle_end: float = 2 * math.pi) -> "FanBeamGeometry":
        """Mayo LDCT FBP geometry from the 5-parameter Powell fit on L014.

        Recommended defaults for all Mayo-LDCT FBP / FBP-based solver
        pipelines (job 762284). Re-validated by ablation 2026-05-27
        (SLURM 762403): at these FBP values + SSR-step values from
        `MAYO_LDCT_SSR_DEFAULTS`, pixel_sp=0.700857 hits SSIM 0.9622 /
        PSNR 42.27 dB on the central L014 GT.

        Fitted values (all DICOM-nominal-minus-fit deltas are sub-percent):
          - pixel_spacing = 0.700857 mm (DICOM nominal: 0.703125, Δ = -0.32 %)
          - det_spacing   = 1.285044 mm (DICOM nominal: 1.285839, Δ = -0.06 %)
          - sod           = 595.362 mm  (DICOM nominal: 595.000,  Δ = +0.06 %)
          - sdd           = 1086.803 mm (DICOM nominal: 1085.600, Δ = +0.11 %)

        Note: an additional sub-pixel `detector_origin` offset of −0.040 mm
        was found alongside the four scalars above. It must be applied
        OUTSIDE this dataclass on PyronnFanBeamProjector._tensor_geom
        ['detector_origin'] (see MAYO_LDCT_DET_OFFSET below) because
        PYRO-NN doesn't expose detector_origin as a constructor argument.

        Do NOT replace these values with the SSR-step optimum — see
        `MAYO_LDCT_SSR_DEFAULTS` for that. The two are independent.
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


# ---------------------------------------------------------------------------
# Mayo LDCT SSR-step defaults
# ---------------------------------------------------------------------------
#
# The SSR (single-slice rebinning, helix→fan, Noo 1999 Eq. 1/2) step has
# its OWN (sod, sdd) pair, distinct from the FBP back-projection step
# above. They parameterise different operators:
#
#   * SSR sod/sdd: govern v_precise = dZ · (u² + sdd²) / (sod · sdd)
#       — i.e. how helical rays at offset dZ get mapped to the v-row of
#       the virtual fan-beam detector. This is a geometric correction
#       term that scales how aggressively z-offset rays get pulled
#       toward the central slice.
#
#   * FBP sod/sdd: govern the back-projection trajectory inside PYRO-NN's
#       parallel/fan FBP operator (Hann-/RamLak-filtered, then
#       integration over angles).
#
# These two pairs were assumed to be one and the same prior to 2026-05-26.
# The multi-GT joint Adam fit (SLURM 762369) showed that the LOSS GRADIENT
# pushes them in different directions when the model also has Δz / slab /
# post-FBP-scale knobs — the SSR sod is pulled to a different value while
# the FBP sod stays at the Powell value 595.362 mm. Ablation confirms this
# is the correct way to interpret the result (SLURM 762403 / 762404).
#
# **v3 fit (SLURM 763384, 2026-06-12)** — the current production values
# below — added a `s_z` (z-axis scaling) parameter that the previous
# multi-GT fit lacked. The 1-parameter sweep (SLURM 763383, 21 points
# in s_z ∈ [0.995, 1.005] holding everything else at the v2 optimum)
# confirmed a clear interior minimum at s_z ≈ 1.0005 (PSNR +0.08 dB);
# the joint Adam fit relaxed the coupled sod / sdd / Δz and converged
# at s_z = 1.001665 (PSNR +0.26 dB vs v2, +0.5 dB at the patient z
# extremes — see docs/findings.md 2026-06-12 entry).
#
# `s_z` multiplies the per-readout source-z BEFORE the α_dz FFS shift:
#     z_eff = s_z * z_positions + α_dz * ffs_dz
# Physically it corresponds to a ~0.17 % under-estimate of the helical
# pitch_mm (mechanism A) or, equivalently, the effective dv / sdd in
# the curved-to-flat step (mechanism B). The two are observationally
# identical at leading order; we absorb them into one scalar.
#
# These are the SSR-step defaults that go into the cached
# `L0XX_proj_flat_*.pt` blobs via `scripts/cache_proj_flat_L014.py` /
# `cache_proj_flat_L014_full.py`, and into the bulk-rebin pipeline
# via `data/fetch_mayo_ldct.py` (env var HELIX2FAN_SSR_FITTED=1):
MAYO_LDCT_SSR_DEFAULTS = {
    # SSR rebin geometry (v3 joint Adam fit + s_z, SLURM 763384)
    "sod": 592.829,    # mm — DICOM nominal 595.000 (Δ = -0.37 %)
    "sdd": 1087.268,   # mm — DICOM nominal 1085.600 (Δ = +0.15 %)
    # Detector pitch — held FIXED at DICOM-CT-PD private tags during the
    # multi-GT and v3 fits (detector pitch is hardware).
    "du":  1.285839,   # mm — DICOM tag (0x7029, 0x1002)
    "dv":  1.094723,   # mm — DICOM tag (0x7029, 0x1006)
    # NEW in v3: per-readout z-axis scaling. Equivalent to (pitch_mm * s_z)
    # or to a scaled effective dv. See docs/findings.md 2026-06-12.
    "s_z": 1.001665,   # multiplicative correction on z_positions
    # Slab / z-shift / post-FBP (also from SLURM 763384)
    "delta_z_mm": -0.159,
    "alpha_dz":   +1.0,     # FFS-z sign (ablation winner 762363)
    "slab_offsets_mm": (-3, -2, -1, 0, 1, 2, 3),
    "w_slab":     (0.049, 0.205, 0.141, 0.195, 0.148, 0.204, 0.058),
    "post_fbp_a":  0.809,
    "post_fbp_bg": -0.000304,
    "post_fbp_hi": 0.0519,
    # 64-bin radial filter H(ρ) at unity ρ-step dr=0.05 (also v3).
    # Loaded as numpy array — too long to embed here; lives at
    # data/mayo_ldct_geom/L014_h_radial_v3.npy. Helper:
    #     from ddssl_ldct.geometry import load_mayo_ssr_h_radial
    "h_radial_path": "ddssl_ldct/mayo_geom/L014_h_radial_v3.npy",
    "h_radial_n_bins": 64,
    "h_radial_dr": 0.05,
}


# Pure DICOM-CT-PD-nominal SSR config — the "what would you get if you
# trusted the DICOM tags only" fallback. Same schema as MAYO_LDCT_SSR_DEFAULTS,
# but every value comes straight from the DICOM header (or is set to the
# no-op identity for knobs DICOM cannot express). Useful as the comparison
# arm against the fitted recon, and as a "challenge-reported" baseline
# for any agent that wants to reproduce the DICOM-nominal SSIM 0.874 /
# PSNR 33.0 dB (SLURM 762409).
MAYO_LDCT_SSR_NOMINAL = {
    "sod": 595.000,    # mm — DICOM tag (0x7031, 0x1003)
    "sdd": 1085.600,   # mm — DICOM tag (0x7031, 0x1031)
    "du":  1.285839,   # mm — DICOM tag (0x7029, 0x1002)
    "dv":  1.094723,   # mm — DICOM tag (0x7029, 0x1006)
    # Knobs DICOM cannot express → identity / no-op:
    "delta_z_mm": 0.0,                       # no sub-mm anchor offset
    "alpha_dz":   0.0,                       # no FFS-z correction
    "slab_offsets_mm": (0,),                 # single-slice (no slab averaging)
    "w_slab":     (1.0,),                    # ditto
    "post_fbp_a":  1.0,                      # identity (rely on intensity_calibrate)
    "post_fbp_bg": 0.0,
    "post_fbp_hi": float("inf"),             # no upper clip
}


def load_mayo_ssr_h_radial(repo_root=None) -> "np.ndarray":
    """Load the 64-bin radial filter H(ρ) co-fitted with the v3 SSR
    defaults (SLURM 763384, 2026-06-12).

    Returns a (64,) float32 numpy array of multiplicative gains applied
    in Fourier-radial space at dr=0.05 (ρ range [0, 64·0.05) ).
    Production usage is

        from ddssl_ldct.geometry import load_mayo_ssr_h_radial
        h = load_mayo_ssr_h_radial()
        # build a 2D radial gain map on the FBP image grid, then
        # multiply the FFT2 of the FBP image by it.

    Path is read from MAYO_LDCT_SSR_DEFAULTS["h_radial_path"]. If
    ``repo_root`` is None we anchor to the geometry.py file's repo
    root so callers from any cwd resolve correctly.
    """
    import numpy as _np
    from pathlib import Path as _Path
    if repo_root is None:
        repo_root = _Path(__file__).resolve().parents[1]
    p = _Path(repo_root) / MAYO_LDCT_SSR_DEFAULTS["h_radial_path"]
    return _np.load(p).astype(_np.float32)


def mayo_ldct_ssr_config(name: str = "fitted") -> dict:
    """One-line switch between the v3-fitted SSR defaults and the
    DICOM-nominal fallback.

    Pass ``name="fitted"`` (default, recommended — SSIM 0.957 / PSNR
    40.8 dB mean across 10 GTs spanning the full L014 patient range at
    s_z=1.001665, sod=592.829, sdd=1087.268; +0.5 dB at the patient z
    extremes vs the multi-GT-only fit) or ``name="nominal"`` (SSIM 0.874
    / PSNR 33.0 dB — what you'd get from trusting DICOM tags alone).
    """
    if name == "fitted":
        return dict(MAYO_LDCT_SSR_DEFAULTS)
    if name == "nominal":
        return dict(MAYO_LDCT_SSR_NOMINAL)
    raise ValueError(f"unknown ssr config name: {name!r}; "
                      f"expected 'fitted' or 'nominal'")
