---
title: Mayo-LDCT leaderboard
description: Real-helical-data leaderboard (Wagner split). REBUILDING 2026-06-14 — all prior results were discarded (scored with the bg→0 calibration bug). See solver_plan.md for methodology.
---

# Mayo-LDCT leaderboard (Wagner split)

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

_Being computed (SLURM job, 2026-06-14). This section will hold the per-split
oracle (HD FBP) and baseline (LD FBP) calibrated SSIM/PSNR over all slices,
with per-patient images. Solver rows are added below it as the rebuild
proceeds._

| Split | Patients | n slices | HD-FBP SSIM (oracle) | LD-FBP SSIM (baseline) |
|---|---|---:|---:|---:|
| train | L145 L186 L209 L219 | — | — | — |
| val | L277 | — | — | — |
| test | L014 L056 L058 L075 L123 | — | — | — |

## Solver leaderboard

_Empty — the rebuild has not added any solver yet._
