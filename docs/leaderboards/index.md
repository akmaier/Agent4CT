---
title: Leaderboards
description: Calibrated headroom rankings across the three CT reconstruction benchmarks — full 19-solver inventory exercised on each.
---

# Leaderboards

Best-of-best per solver per dataset under the canonical
**calibrated-SSIM-headroom** scoring convention
([`evaluate_calibrated`](https://github.com/akmaier/Agent4CT/blob/main/ddssl_ldct/metrics.py)).
All 3 leaderboards have **full 19-solver inventory coverage**. **Mayo-LDCT is
being re-run live** (`search-20260619-01`) under the corrected metric — the Mayo
champion row below auto-updates each wave from the run data; the breast/demo
boards are stable. The cross-dataset counts further down are a 2026-06-09
snapshot.

## Champions — rendered from the registry

The per-dataset champion (single canonical ranking = **headroom**, SSIM
tiebreak) is rendered live from the registry below, so it can never drift from
the dataset boards. Each panel is that dataset's **full** all-solver board
(below-baseline solvers dimmed, never dropped). For the prose write-ups and the
baseline tables, open the dataset board linked under "Full standings".

<h3>Mayo-LDCT</h3>
<div data-leaderboard="mayo_ldct">loading leaderboard…</div>
<h3>Breast-CT</h3>
<div data-leaderboard="breast_ct">loading leaderboard…</div>
<h3>BreastCT-Noise <small>(high-dose robustness re-eval, I0 = 100k photons)</small></h3>
<div data-leaderboard="breast_ct_noise">loading leaderboard…</div>
<h3>BreastCT-Noise-Retrained <small>(retrained on noisy train data, I0 = 100k photons)</small></h3>
<div data-leaderboard="breast_ct_noise_retrain">loading leaderboard…</div>
<h3>Demo-DL</h3>
<div data-leaderboard="demo_dl">loading leaderboard…</div>

<script src="../assets/table.js"></script>
<script src="../assets/leaderboard.js"></script>

## Full standings — every solver

The complete per-dataset rankings — **all 19 solvers**, with every column
(params, SSIM, hr, PSNR, RMSE, time) — are on the dataset boards below. These
are the single source of truth and list **every** solver (no truncated summary):

- **[Mayo-LDCT — all 19 solvers →](mayo_ldct.html)** &nbsp; _(live `search-20260619-01`, auto-regenerated every wave)_
- **[Breast-CT — all solvers →](breast_ct.html)**
- **[BreastCT-Noise — no-retrain robustness re-eval →](breast_ct_noise.html)** &nbsp; _(same models, Poisson-noised inputs)_
- **[BreastCT-Noise-Retrained — matched-noise retraining →](breast_ct_noise_retrain.html)** &nbsp; _(clean ranking largely returns; ρ 0.16→0.65)_
- **[Demo-DL — all 19 solvers →](demo_dl.html)**

## Methodology

See [`solver_plan.md`](https://github.com/akmaier/Agent4CT/blob/main/solver_plan.md)
for the canonical methodology — calibrated metric, per-solver
hyperparameter spaces, autoresearch+TPE protocol.

Per-solver design docs and cross-dataset transfer records:
[`pentathlon/demo_dl_reference/`](https://github.com/akmaier/Agent4CT/tree/main/pentathlon/demo_dl_reference/).
Every canonical-19 solver has a dedicated `.md` design doc with
cross-dataset hr record, CONFIG defaults, and "hints for the next
autoresearch agent".

Cross-cutting findings (DDPM training quality NOT predictive of DPS
performance, FBP-init 1st-step-no-op mechanism, etc.) live in
[`docs/findings.md`](../findings.html).
