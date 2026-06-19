---
title: Mayo-LDCT leaderboard
description: Real-helical-data leaderboard (Wagner split). LIVE search-20260619-01 — all 19 solvers onboarded and iterating under the corrected metric (214 L277 slices, 321px geometry FOV). Current standings in the auto-generated solver table.
---

# Mayo-LDCT leaderboard (Wagner split)

> 🟢 **LIVE — the fresh `search-20260619-01` campaign is running now.** All 19
> solvers are onboarded and iterating (iter-1 → iter-20) under the corrected
> metric (all 214 L277 slices + 321 px detector-geometry FOV). **Current
> standings, including the headroom champion, are in the auto-generated solver
> table further down** (regenerated every wave, ranked by headroom). The reset
> notice immediately below is *historical context* for why the prior campaign
> was discarded — it is not the current status.

> 🛑 **2026-06-19 — SECOND full reset (`search-20260614-01` discarded).**
> The entire 2026-06-14 rebuild (`search-20260614-01`, 19 solvers) was scored on
> an **invalid validation metric** and has been discarded (runs purged):
> - **Val scored the first `val_n` (4–30) slices of L277** — the top-of-volume
>   boundary slices (near-empty, near-identical), not representative anatomy.
> - **Wrong FOV mask** — the 256px Sidky inscribed circle instead of the
>   detector-geometry measurement FOV (237.54 mm → 321 px), discarding the valid
>   256→321 px annulus (commit `5ced1ec9`).
> - **Figures rendered upside-down** and repeated the same boundary slice.
>
> The whole search trajectory was driven by this bad signal, so the dashboard,
> leaderboard, and search are being **redone from scratch** with the corrected
> metric (all 214 L277 slices + geometry FOV), a real **20-min per-iteration
> budget** (val figure excluded), and the orientation fix. New run-id:
> **`search-20260619-01`**. See [README](../../README.md) +
> [`solver_plan.md`](../../solver_plan.md) + [`mayo_campaign_state.md`](../runs/mayo_campaign_state.md)
> for the corrected protocol.
>
> 🧹 **2026-06-14 — leaderboard reset. Starting over.**
> Every Mayo result produced before this date has been **discarded** and
> the dashboard run directories removed. Two compounding problems made the
> old numbers untrustworthy:
> 1. **`intensity_calibrate` background-offset bug** — all calibrated
>    SSIM/PSNR were scored mapping the recon background to 0, but Mayo
>    truth background is ~+0.0005 μ (≈0.01 SSIM low on average, more for
>    low-overlap slices). Fixed via opt-in `bg_target="truth"`
>    (commit `bcfa2720`); end-to-end validated 2026-06-13.
> 2. **Geometry + FBP path not fully hard-wired** — the old agentic/TPE
>    runs scored on a subset of slices, did not all use the final v3 SSR
>    geometry, and the solver FBP path did not yet carry the detector
>    offset + water-cylinder truncation correction.
>
> The rebuild starts from a clean **HD vs LD FBP baseline computed over
> every truth slice** of all 10 Wagner patients, using the frozen
> production path (v3 SSR rebin + Powell FBP geometry + `MAYO_LDCT_DET_OFFSET`
> + `MAYO_LDCT_TRUNCATION` + `bg_target="truth"`). That baseline defines the
> headroom-scoring endpoints (LD FBP = score 0, HD FBP = score 1) for every
> solver added from here on. See [`docs/findings.md`](../findings.md) (2026-06-14).

AAPM 2016 Low-Dose CT Grand Challenge data — helical Siemens SOMATOM
AS+, rebinned to 2-D fan-beam via the in-house
[`helix2fan`](../../ddssl_ldct/helix2fan.py) SSR pipeline. Wagner split:

```
Train: L145, L186, L209, L219      (4 patients)
Val:   L277                         (1 patient)
Test:  L014, L056, L058, L075, L123 (5 patients)
```

**All splits are evaluated on _every_ reconstructed slice** (≈155 full-dose
truth-image slices per patient), not a subsampled subset.

## Baseline — HD vs LD FBP (all slices)

Computed 2026-06-14 (SLURM 763659, `scripts/compare_hd_ld_fbp_allslices.py`)
over **every** full-dose truth slice of all 10 Wagner patients (**1538 slices**),
both doses, on the frozen production path. These define the headroom-scoring
endpoints: **`score = (SSIM − LD_FBP) / (HD_FBP − LD_FBP)`** per split.
z-registration residual ≤ 0.38 mm for every patient; HD-FBP L277 0.9331
matches the validated central-slice number (0.9315).

| Split | Patients | n slices | HD-FBP SSIM (oracle) | LD-FBP SSIM (baseline) | HD PSNR | LD PSNR | headroom gap |
|---|---|---:|---:|---:|---:|---:|---:|
| **train** | L145 L186 L209 L219 | 579 | 0.9501 | 0.8659 | 36.88 | 34.95 | 0.0843 |
| **val** | L277 | 214 | 0.9331 | 0.8078 | 37.65 | 34.45 | 0.1252 |
| **test** | L014 L056 L058 L075 L123 | 745 | 0.9528 | 0.8848 | 37.99 | 36.15 | 0.0680 |
| **overall** | all 10 | 1538 | 0.9491 | 0.8670 | 37.52 | 35.46 | 0.0821 |

Per-patient (calibrated SSIM, all slices):

| Patient | split | n | truth ps (mm) | HD-FBP | LD-FBP | gap |
|---|---|---:|---:|---:|---:|---:|
| L145 | train | 160 | 0.781 | 0.9469 | 0.8571 | 0.0898 |
| L186 | train | 167 | 0.781 | 0.9482 | 0.8588 | 0.0894 |
| L209 | train | 98 | 0.742 | 0.9453 | 0.8534 | 0.0919 |
| L219 | train | 154 | 0.664 | 0.9587 | 0.8905 | 0.0682 |
| L277 | val | 214 | 0.742 | 0.9331 | 0.8078 | 0.1252 |
| L014 | test | 154 | 0.703 | 0.9564 | 0.9078 | 0.0487 |
| L056 | test | 93 | 0.703 | 0.9605 | 0.9017 | 0.0588 |
| L058 | test | 210 | 0.742 | 0.9330 | 0.8272 | 0.1058 |
| L075 | test | 137 | 0.664 | 0.9626 | 0.9149 | 0.0477 |
| L123 | test | 151 | 0.664 | 0.9631 | 0.9039 | 0.0591 |

![HD vs LD FBP per-patient + per-split SSIM](baseline_2026-06-14/summary.png)

Per-patient representative slices (min / median / max LD-SSIM; GT | HD | LD | LD−GT)
and per-slice SSIM-vs-z curves are in
[`baseline_2026-06-14/`](baseline_2026-06-14/) — e.g. val patient
[L277 montage](baseline_2026-06-14/L277_montage.png) ·
[L277 SSIM vs z](baseline_2026-06-14/L277_ssim_vs_z.png), test patient
[L014 montage](baseline_2026-06-14/L014_montage.png).

## Solver leaderboard

> **`search-20260619-01` — in progress (2026-06-19).** Agentic autoresearch loop
> (Step 2 of [`solver_plan.md`](../../solver_plan.md)) on the corrected per-sample
> path: training-NaN grad-clip fix (`metrics.clip_and_step`) + per-sample `ps_eff`
> reconstruction + `bg_target="truth"` calibration. The val metric is the
> **corrected** one — calibrated full-image SSIM scored on **all 214 L277 slices**
> inside the **321 px detector-geometry FOV** (`train_n=200` stratified across the
> 4 train patients). The headroom `hr = (SSIM − LD_FBP)/(oracle − LD_FBP)` uses
> the val row of the baseline table above (LD-FBP 0.8078, HD-FBP oracle 0.9331).
> Each solver runs under a **hard 20-min train+score budget** (val figure
> excluded), driving toward the iter-20 hard stop.

The table below is **rendered live from the registry** (`docs/runs/index/leaderboard.json`,
built by [`scripts/build_registry.py`](../../scripts/build_registry.py) from each
iter's immutable `observation.json`). It lists **every** solver in the campaign,
ranked by **headroom** (SSIM tiebreak); below-baseline / discarded solvers stay
on the board (dimmed, unranked) so the inventory is always complete — never a
top-N. **No number on this page is typed by hand**, so nothing can go stale.
Each comparison thumbnail is the solver's own 4-panel val figure at its best iter
(**GT │ LD-FBP │ recon │ recon − GT diff** on the first L277 slices), regenerated
at run time so it tracks the metric.

<div data-leaderboard="mayo_ldct">loading leaderboard…</div>

<script src="../assets/table.js"></script>
<script src="../assets/leaderboard.js"></script>

## Methodology

See [`solver_plan.md`](../../solver_plan.md) for the corrected Mayo protocol
(214 L277 val slices, 321 px detector-geometry FOV, 20-min per-iter budget,
orientation fix) and [`docs/findings.md`](../findings.md) for the cross-cutting
substrate facts. The ranking, params resolution, and below-baseline handling are
the same canonical logic used across all three dataset boards (one ranking,
every solver, no top-N).
