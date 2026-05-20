# Demo-DL — synthetic phantom test suite

The "always-on" no-staged-data fallback dataset that every solver in
`pentathlon/demo_dl_reference/` runs against unless `AGENT4CT_DATASET`
points it at a real challenge. Used for fast (≤ 5 min) iteration cycles
where staged H5 reads + checkpoint loading would dominate wall time.

## Overview

| | |
|---|---|
| Purpose | Fast self-contained sandbox for the autoresearch loop |
| Phantom | Randomised ellipse mixtures (`ddssl_ldct.phantoms.random_ellipses_phantom`) |
| Forward model | `PyronnFanBeamProjector` fan-beam line integrals |
| Noise model | Poisson(I0) + Gaussian(σ_e) on detector intensities, then `-log(I/I0)` |
| Metric | RMSE / PSNR / SSIM vs the clean phantom, calibrated via `intensity_calibrate` |
| Geometry source | Wagner et al. helical-rebinned Mayo LDCT (1.2858 mm pitch, 736 channels) |
| Units | μ in 1/mm; display range [0, 0.05] (= adipose/water typical) |

## Geometry

Single source of truth: [`challenges/demo_dl/geometry.py`](geometry.py).
`GEOMETRY` is a `DatasetInfo`; `DEFAULTS` is a solver-CONFIG-shaped dict
that bundles the geometry with the simulation knobs (noise, splits, seed).

```python
GEOMETRY = DatasetInfo(
    image_size=512, pixel_spacing=0.7,
    n_angles=128, n_det=736, det_spacing=1.2858,
    sod=595.0, sdd=1085.6,
    display_min=0.0, display_max=0.05,
    has_real_sino=False,
)
```

## Why these numbers

| Knob | Value | Source |
|---|---|---|
| image_size | 512 | Wagner LDCT recon grid |
| pixel_spacing | 0.7 mm | Wagner — Siemens AS in-plane |
| n_angles | 128 | DL-Sparse-View challenge sparsity convention |
| n_det | 736 | Siemens AS channels after rebinning |
| det_spacing | 1.2858 mm | Siemens AS post-rebin pitch |
| sod | 595 mm | Siemens AS source-to-iso |
| sdd | 1085.6 mm | Siemens AS source-to-detector |
| noise_i0 | 1e5 | Wagner low-dose convention |
| noise_sigma_e | 10.0 | Wagner electronic-noise floor (counts) |

## Solver usage

```python
from challenges.demo_dl.geometry import DEFAULTS as DEMO_DL_DEFAULTS

CONFIG = {**DEMO_DL_DEFAULTS, "your_solver_key": ...}
```

`CONFIG` then has every geometry + simulation field a solver needs to
build its `FanBeamGeometry` and call `build_dataset()`. When
`AGENT4CT_DATASET=breast_ct` (or similar) is set,
`ddssl_ldct.staged_dataset.geometry_overrides()` rewrites the geometry
keys in CONFIG to match the real challenge — so the same solver code
runs against both `demo_dl` and `breast_ct` without source edits.

## Related code

- [`ddssl_ldct/phantoms.py`](../../ddssl_ldct/phantoms.py) — phantom generator
- [`ddssl_ldct/simulate.py`](../../ddssl_ldct/simulate.py) — Poisson+Gaussian noise
- [`ddssl_ldct/staged_dataset.py`](../../ddssl_ldct/staged_dataset.py) — dispatch + staged-data loader
- [`ddssl_ldct/pyronn_projector.py`](../../ddssl_ldct/pyronn_projector.py) — fan-beam projector
