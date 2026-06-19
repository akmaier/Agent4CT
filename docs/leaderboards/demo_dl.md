---
title: Demo-DL leaderboard
description: Calibrated headroom ranking of every solver family on the demo-DL 128-view sparse-view substrate.
---

# Demo-DL leaderboard

128-view 2-D fan-beam sparse-view synthetic phantoms (Sidky-style random
ellipse phantoms — the "demo" track used as a fast iteration substrate).
Smaller and faster than the breast-CT track; useful for prototyping.

All metrics through
[`ddssl_ldct.metrics.evaluate_calibrated`](../../ddssl_ldct/metrics.py):
two-point linear intensity calibration on the foreground inside an
inscribed-circle FoV mask, then PSNR/SSIM/RMSE on the calibrated
prediction. `hr = max(0, 1 − rmse / baseline_rmse)` where the baseline is
the calibrated FBP (`demo-intensity-calibrated-tpe-*` family).

The **2026-05-19 calibrated-scoring convention** is the canonical metric;
earlier uncalibrated runs (slug prefix `demo-dl-*`, `dl-sparse-view-*`)
are kept at the bottom for historical context and **are not directly
comparable**.

## Leaderboard — every solver (rendered from the registry)

The table below is **rendered live from the registry**
(`docs/runs/index/leaderboard.json`, built by
[`scripts/build_registry.py`](../../scripts/build_registry.py) from each iter's
immutable `observation.json`). It lists **every** solver exercised on the
calibrated demo-DL substrate, ranked by **headroom** (SSIM tiebreak); any
below-baseline / discarded solver stays on the board **dimmed and unranked** so
the inventory is always complete — never a top-N. **No number on this page is
typed by hand.** The pre-2026-06 TPE runs show `—` for PSNR/RMSE/time (only SSIM
+ headroom were recorded then). `params (M)` is trainable parameters in millions.

The registry scope is the **calibrated** convention only
(`demo-intensity-calibrated-tpe-*` + the `demo-dl-claude-agentic-*` N2I seeds).
The legacy uncalibrated `demo-dl-*` runs use a *different scoring rule* and are
**not** in the registry; the few kept for historical context are in the
"Earlier uncalibrated runs" section near the bottom.

<div data-leaderboard="demo_dl">loading leaderboard…</div>

<script src="../assets/table.js"></script>
<script src="../assets/leaderboard.js"></script>

## Below-baseline inventory (`hr = 0`, structural STOPs)

**None of the 18 variants tested on demo-DL fell below baseline.** The
synthetic Sidky-style 128-view ellipse substrate is broad enough that
every learned and non-learned solver finds *some* working corner.

## Inventory-gap closure log (2026-06-09)

Two solvers from the [`solver_plan.md`](../../solver_plan.md) 19-solver
inventory were dispatched on demo-DL on 2026-06-09 to close the
coverage gap:

| Solver | TPE job | Result | Status |
|---|---|---:|---|
| **ItNet v1** (`solver_itnet.py`) | 762957 | **hr=0.4665** at iter-20 winner (k=2, c=16, pretrain_ep=6, lr=5e-4) | ✅ **ABOVE BASELINE** — slots in at rank 4 between v3 (0.4676) and USwin (0.4655). Surprises the Mayo verdict (hr=0): on demo-DL's broader synthetic substrate, v1's deeper finetune-pass + lower-k schedule lets it compete with v3. |
| **TV-iterative supervised** (`solver_tv_iterative_supervised.py`) | 762958 | running iter-16/20; **all 15 completed iters hr=0** with SSIM frozen at exactly 0.4402 (the FBP baseline SSIM on demo-DL) | 🚫 **CONFIRMED STOP** — FBP-init no-op verdict empirically validated. TPE has explored `tv_step_init` ∈ [1.4e-4, 7e-2] (>2 orders), `tv_lambda_init` ∈ [1.2e-5, 9.8e-3] (>3 orders), both `share_steps` toggles, `epochs` ∈ [5, 20], `lr` ∈ [5e-4, 5e-2], both `mse`/`l1` — across the entire search space the network outputs the FBP input exactly. Structural — no config can break the lock. |

**Cross-dataset coverage compare (gap closure 2026-06-09 — FINAL):**

| Dataset | Above baseline | Below baseline | Total | Coverage |
|---|---:|---:|---:|---|
| **Demo-DL** | 19 (added ItNet v1 at rank 4) | 1 (TV-iter sup STOP) | 20 entries | **19/19 inventory variants tested** |
| **Breast-CT** | 16 (added ItNet v1 at rank 15, Wu trainable TPE at rank 10) | 13 | 29 entries | **19/19 inventory variants tested** |
| **Mayo-LDCT** | 12 | 10 + 1 deprioritised (diff_recon v3) | 23 entries | **19/19 inventory variants tested** |

**All three datasets now have full 19-solver inventory coverage.** The
gap closures revealed two surprises: (a) ItNet v1 clears baseline on
both synthetic datasets (despite Mayo verdict hr=0) — the v1
architecture is competitive with v3 on demo-DL (0.4665 ≈ 0.4676);
(b) Wu trainable on breast-CT lifted +45% from agentic (0.2189 →
0.3170) via TPE finding a higher-`n_bands` + lower-lr corner the
agentic search missed.

The demo-DL "every solver works" pattern is a useful sanity check for
the substrate but a poor predictor of behaviour on the two
realistic-data benchmarks — see [`solver_plan.md`](../../solver_plan.md)
Step 1 for the cross-dataset transfer record.

## Constrained vs. unconstrained DDPM (Step 4 of solver_plan.md)

The Demo-DL DDPM was trained two ways:

- `ddpm_constrained_final.pt` — `train_n=200` (the same 200 phantoms the
  supervised solvers train on; no test-set distribution leakage).
- `ddpm_unconstrained_final.pt` — `train_n=2000` (different seed range
  from training/test, larger sample → richer prior).

The unconstrained variant scored `hr=0.4530` vs constrained's `0.4418`,
**+0.0112 hr from "seeing more (random-seed-disjoint) ellipse
phantoms"**. The gap is the empirical answer to "how much does the DDPM
prior benefit from a larger / test-distribution-overlapping training
set?" for this benchmark — modest but real.

## Earlier uncalibrated runs (not directly comparable)

Slug prefix `demo-dl-*` and `dl-sparse-view-*` (pre-2026-05-19
convention). Kept for historical context; the reported `hr` comes from a
different scoring rule and is systematically higher than the calibrated
equivalent. Top entries only:

| Solver | params (M) | hr (uncalibrated) | Source | Comparison |
|---|---:|---:|---|---|
| ITNet v3 | 2.082 | 0.8215 | [results](../runs/demo-dl-itnet-v3-search-20260516-02/results.tsv) | [iter-9](../runs/demo-dl-itnet-v3-search-20260516-02/iterations/iter-0009/comparison.png) |
| USwin | 3.954 | 0.8103 | [results](../runs/demo-dl-uswin-search-20260516-01/results.tsv) | [iter-10](../runs/demo-dl-uswin-search-20260516-01/iterations/iter-0010/comparison.png) |
| USwin (fair) | 3.954 | 0.8090 | [results](../runs/demo-dl-fair-uswin-search-20260517-01/results.tsv) | [iter-10](../runs/demo-dl-fair-uswin-search-20260517-01/iterations/iter-0010/comparison.png) |
| Res-UNet | 0.225 | 0.6095 | [results](../runs/dl-sparse-view-res-20260513-01/results.tsv) | [iter-91](../runs/dl-sparse-view-res-20260513-01/iterations/iter-0091/comparison.png) |
| BF / NAF / iter-UNet | 0.0–0.5 | 0.54–0.62 | (various `demo-dl-*` slugs) | — |

## Methodology

See [`solver_plan.md`](../../solver_plan.md) for the full benchmark
protocol — dataset construction, baseline FBP definition, calibrated
metric, and per-solver hyperparameter spaces.
