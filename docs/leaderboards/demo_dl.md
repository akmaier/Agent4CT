---
title: Demo-DL leaderboard
description: Calibrated-SSIM-headroom ranking, one row per solver family. See solver_plan.md for methodology.
---

# Demo-DL leaderboard

128-view 2-D fan-beam sparse synthetic phantoms (Sidky-style random
ellipse phantoms — the "demo" track used as a fast iteration substrate).
Smaller and faster than `breast_ct`; useful for prototyping. **The
post-2026-05-19 calibrated-scoring convention is the canonical metric**
(legacy uncalibrated runs reported higher numbers that don't compare
like-for-like with the rest of the pentathlon).

All metrics through `evaluate_calibrated`: linear intensity calibration
on the foreground inside an inscribed-circle FOV mask, then PSNR/SSIM
on the calibrated pred. `hr = max(0, 1 − rmse/baseline_rmse)`.

## Calibrated leaderboard (canonical)

Slug prefix `demo-intensity-calibrated-tpe-*`.

| Rank | Solver | SSIM | hr | Source slug / iter |
|---:|---|---:|---:|---|
| 1 | **ITNet v3** | 0.9178 | 0.4676 | `demo-intensity-calibrated-tpe-itnet-v3-search-20260520-01` / iter-9 |
| 2 | USwin | 0.8722 | 0.4655 | `demo-intensity-calibrated-tpe-uswin-search-20260520-01` / iter-11 |
| 3 | **RAM zero-shot** (pretrained) | 0.9181 | 0.4648 | `demo-intensity-calibrated-tpe-ram-zeroshot-search-20260521-01` / iter-16 |
| 4 | ITNet v2 | 0.9178 | 0.4567 | `demo-intensity-calibrated-tpe-itnet-v2-search-20260520-01` / iter-20 |
| 5 | **Diff-recon DC-step — DDPM unconstrained** | 0.8251 | 0.4530 | `demo-intensity-calibrated-tpe-diff-recon-dcstep-unconstrained-search-20260521-01` / iter-17 |
| 6 | **Diff-recon DC-step — DDPM constrained** | 0.8090 | 0.4418 | `demo-intensity-calibrated-tpe-diff-recon-dcstep-constrained-search-20260521-01` / iter-18 |
| 7 | NAF | 0.8534 | 0.4160 | `demo-intensity-calibrated-tpe-naf-search-20260521-01` / iter-19 |
| 8 | TV-iterative | 0.8706 | 0.4056 | `demo-intensity-calibrated-tpe-tv-search-20260520-01` / iter-13 |
| 9 | DD-UNet N2I | 0.6854 | 0.3811 | `demo-intensity-calibrated-tpe-dual-domain-search-20260520-01` / iter-17 |
| 10 | Hammernik 2017 | 0.7890 | 0.3622 | `demo-intensity-calibrated-tpe-hammernik-search-20260520-01` / iter-6 |
| 11 | Hammernik VN | 0.7722 | 0.3621 | `demo-intensity-calibrated-tpe-hammernik-vn-search-20260520-01` / iter-11 |
| 12 | DD-BF N2I | 0.7605 | 0.3611 | `demo-intensity-calibrated-tpe-dual-domain-bf-search-20260520-01` / iter-1 |
| 13 | R2Gaussian | 0.8324 | 0.2999 | `demo-intensity-calibrated-tpe-r2gaussian-search-20260521-01` / iter-14 |
| 14 | Wu 2015 (non-trainable) | 0.5495 | 0.2295 | `demo-intensity-calibrated-tpe-wu-search-20260521-01` / iter-18 |

## Constrained-vs-unconstrained DDPM (Step 4 of solver_plan.md)

The Demo-DL DDPM was trained two ways:
- `ddpm_constrained_final.pt` — train_n=200 (the same 200 phantoms the
  supervised solvers train on; no test-set distribution leakage).
- `ddpm_unconstrained_final.pt` — train_n=2000 (different seed range
  from training/test, larger sample → richer prior).

The unconstrained variant scored hr=0.4530 vs constrained's 0.4418,
**+0.0112 hr from "seeing more (random-seed-disjoint) ellipse phantoms"**.
The gap is the empirical answer to "how much does the DDPM prior benefit
from a larger / test-distribution-overlapping training set?" for this
benchmark. Modest but real.

## Earlier uncalibrated runs (not directly comparable)

Slug prefix `demo-dl-*` (pre-2026-05-19 convention). Kept for historical
context but **not used in the canonical leaderboard above**. The reported
hr there comes from a different scoring rule and is systematically
higher than the calibrated equivalent.

## Methodology

See [`/solver_plan.md`](../../solver_plan.md).
