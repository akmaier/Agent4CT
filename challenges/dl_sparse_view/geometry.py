"""DL-Sparse-View CT (Sidky 2021 breast challenge) geometry + solver defaults.

This is the canonical location of the breast_ct geometry. The
`ddssl_ldct.staged_dataset.GEOMETRIES["breast_ct"]` entry re-exports
this module so all consumers see the same single source of truth.

Solvers running against the real Sidky data (set `AGENT4CT_DATASET=breast_ct`)
get this geometry automatically via `ddssl_ldct.staged_dataset.geometry_overrides()`.
"""
from __future__ import annotations
import os as _os
from pathlib import Path
from challenges._common import DatasetInfo


_DEFAULT_DATA_ROOT = Path(_os.environ.get(
    "AGENT4CT_DATA", "/cluster/maier/Agent4CT/data"))


# Sidky 2022 Med Phys (DL-Sparse-View challenge) §II.B:
#   "source-to-center-of-rotation distance of 50 cm,
#    source-to-detector distance of 100 cm,
#    1024 detector elements,
#    [reconstruction] 512x512 pixels covering an area (18cm)²"
#
# Pixel spacing: 180 mm / 512 = 0.3516 mm/pixel ✓
#
# Detector pitch: NOT stated in the paper. Pinned down empirically by
# minimising the forward-projection L2 of the matched-pair Siddon
# projector against the released val_sinograms[0] over a fine c-sweep
# (see scripts/debug_breast_ct_detspacing_sweep.py, job 761480):
#     best c = 1.017  ->  det_spacing = 0.35156 · 1.017 = 0.35754 mm
# Forward L2_k = 7.9e-4 at that value (essentially exact match);
# FBP cal-SSIM vs Sidky's val_fbp128 jumped 0.61 -> 0.76 vs the earlier
# FOV-coverage assumption (0.3516 mm). The detector iso-projection
# covers 0.35754·1024·0.5 = 18.31 cm — a 1.7% margin over the 18 cm
# image FOV (a sensible CT design choice).
#
# Phantom radius is 8 cm (Sidky paper §II.B, line 110). Linear (flat)
# detector. Source rotates CCW from gantry-zero; sino_angle_shift=+32
# (= +90° in 128-view sino-time) aligns the sino with pyronn's
# (angle 0 = source at +x, CCW positive) convention. Sidky masks all
# pixels outside the inscribed circle (radius 256 px = 9 cm) of the
# 512x512 grid; our metrics apply the same mask via
# `ddssl_ldct.metrics.fov_mask`.

GEOMETRY = DatasetInfo(
    image_size=512, pixel_spacing=0.3516,
    n_angles=128, n_det=1024, det_spacing=0.35754,
    sod=500.0, sdd=1000.0,
    display_min=0.0, display_max=0.5,
    has_real_sino=True,
    staged_dir=_DEFAULT_DATA_ROOT / "dl_sparse_view" / "staged",
    sino_angle_shift=32,
)


# Full solver-CONFIG-shaped defaults. Note display_max here matches the
# demo-dl default's *intent* (highlight the breast tissue band) — μ runs
# 0..0.33 1/cm in truth.
DEFAULTS = dict(
    image_size=GEOMETRY.image_size, pixel_spacing=GEOMETRY.pixel_spacing,
    n_angles=GEOMETRY.n_angles, n_det=GEOMETRY.n_det,
    det_spacing=GEOMETRY.det_spacing,
    sod=GEOMETRY.sod, sdd=GEOMETRY.sdd,
    display_min=GEOMETRY.display_min, display_max=GEOMETRY.display_max,
)
