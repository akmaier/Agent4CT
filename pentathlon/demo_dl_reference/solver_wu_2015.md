# `solver_wu_2015.py` — Wu 2015 non-trainable (closed-form filter-band modulation)

Companion design doc. For the trainable variant (10 learnable
scalars) see `solver_wu_2015_trainable.py`.

The Wu 2015 algorithm decomposes a noisy FBP into filter-band
sub-images (triangular bands in the Ram-Lak filter spectrum),
estimates a noise-suppressed residual via motion-windowed L1
matching, and reconstructs. **Zero trainable parameters** — every
knob is a structural choice (n_bands, n_outer, motion_range,
motion_window, soft_thresh).

## When Wu 2015 wins

Wu 2015 is designed for **motion-blurred or low-photon scans** —
the filter-band decomposition lets it estimate noise in each
frequency band independently. On data with that noise structure,
it can recover detail other classical solvers smooth away.

## Cross-dataset record

| Dataset | hr | Source | Notes |
|---|---:|---|---|
| `demo_dl` | **0.2295** | TPE `demo-intensity-calibrated-tpe-wu-search-20260521-01` (n_bands=8, n_outer=1, range=3, thresh=1.2e-3, train_n=0) | rank 18 on demo-DL. Beaten by every learned solver but clears baseline. |
| `breast_ct` | **0.0425** | TPE `breast-ct-calibrated-tpe-wu-search-20260521-01` (n_bands=4, n_outer=1, range=8, thresh=1.1e-3, train_n=0) | rank 16 on breast-CT. Just above the noise floor — same low-capacity-ceiling pattern as the trainable variant. |
| `mayo_ldct` | **0** | Mayo Step-2 iter-1/2 (n_bands=4→8, ep 3→6 plateau confirmed) | **STOP** — closed-form 10-coefficient filter cannot reach Mayo's dynamic range. Same structural verdict as the trainable variant — doubling `n_bands` only inched SSIM from 0.350 to 0.357. |

## 2026-06-07 — Mayo verdict

Mayo Step-2 agentic ran Wu 2015 non-trainable for 2 iters:
- iter-1: n_bands=4 → hr=0 SSIM 0.350 PSNR 12.35 dB
- iter-2: n_bands=8 → hr=0 SSIM 0.357 PSNR 12.37 dB

Both below the baseline PSNR 12.59 dB. Doubling n_bands lifted SSIM
by +0.007 only — the 10-coefficient filter family doesn't have the
capacity to match Mayo's noise structure. **STOP filed.**

The trainable variant (`solver_wu_2015_trainable.py`) is also STOP'd
on Mayo at hr=0. The 10 trainable scalars + the closed-form Wu 2015
band-decomposition both hit the same low-capacity ceiling.

## CONFIG defaults

```python
CONFIG = {
    "wu_n_bands":      4,        # paper uses 8 triangular bands; 4 keeps cost down
    "wu_n_outer":      2,        # restoration iterations (paper recommends 2-3)
    "wu_motion_range": 5,        # ±pixels for symmetric motion search
    "wu_motion_window": 2,       # ±pixels for L1 windowed patch
    "wu_soft_thresh":  0.0015,   # soft threshold on the residual reco (mu mm^-1)
}
```

## Trainable vs non-trainable Wu 2015

| Dataset | Non-trainable hr | Trainable hr | Δ |
|---|---:|---:|---:|
| `demo_dl` | 0.2295 | 0.2288 | −0.0007 (tied) |
| `breast_ct` | 0.0425 | **0.3170** (TPE 2026-06-09) | **+0.2745** |
| `mayo_ldct` | 0 | 0 | 0 |

**On breast-CT the TRAINABLE Wu variant lifts hr by +0.27** — the
10 learnable scalars find an asymmetric weighting across the 4-8
filter bands that hand-tuning + TPE-of-hand-knobs can't. This is
the cleanest case of "trainable >> non-trainable" in the pentathlon.

But on Mayo and demo-DL the trainable variant lands within noise of
the non-trainable. The 10-scalar ceiling is real where the data
doesn't reward asymmetric band weighting.

## Hints for the next autoresearch agent

- **Use Wu 2015 non-trainable as a zero-param baseline** on new
  datasets. If it gets hr > 0, the trainable variant is worth
  searching (see breast-CT's +0.27 lift).
- If it gets hr = 0 (Mayo), don't bother with the trainable variant
  either — the structural ceiling is below baseline.
- The agentic loop converges fast on Wu's 5 knobs (~10 iters). TPE
  doesn't add much over agentic for the non-trainable variant —
  the trainable variant is where TPE pays off (TPE found 10× lower
  lr corner on breast-CT, +45% over agentic).
