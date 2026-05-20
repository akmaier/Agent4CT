"""Demo-DL (synthetic Wagner-LDCT-style phantom) geometry + solver defaults.

This is the "always-available" no-staged-data fallback dataset that every
solver in `pentathlon/demo_dl_reference/` runs against unless
`AGENT4CT_DATASET` selects a real one. Geometry mirrors a single 2-D slice
of a Siemens SOMATOM Definition AS after helical → fan rebinning (Wagner
et al., the codebase's default `FanBeamGeometry`). Phantoms are randomised
ellipse mixtures and the noise model is Poisson(I0) + AWGN(σ_e) on the
detector intensities, then `−log(I/I0)` to recover line integrals.

Solvers should import `DEFAULTS` and merge it into their CONFIG:

    from challenges.demo_dl.geometry import DEFAULTS as DEMO_DL_DEFAULTS
    CONFIG = {**DEMO_DL_DEFAULTS, ...solver_specific_keys}
"""
from __future__ import annotations
from challenges._common import DatasetInfo


GEOMETRY = DatasetInfo(
    image_size=512, pixel_spacing=0.7,
    n_angles=128, n_det=736, det_spacing=1.2858,
    sod=595.0, sdd=1085.6,
    display_min=0.0, display_max=0.05,
    has_real_sino=False,
)


# Full solver-CONFIG-shaped defaults: geometry + simulation knobs + splits.
# Solvers can `CONFIG = {**DEFAULTS, ...overrides}` and stay consistent
# across the demo-dl test suite without re-declaring constants.
DEFAULTS = dict(
    image_size=GEOMETRY.image_size, pixel_spacing=GEOMETRY.pixel_spacing,
    n_angles=GEOMETRY.n_angles, n_det=GEOMETRY.n_det,
    det_spacing=GEOMETRY.det_spacing,
    sod=GEOMETRY.sod, sdd=GEOMETRY.sdd,
    display_min=GEOMETRY.display_min, display_max=GEOMETRY.display_max,
    train_n=400, val_n=100,
    noise_i0=1e5, noise_sigma_e=10.0,
    seed=42,
)
