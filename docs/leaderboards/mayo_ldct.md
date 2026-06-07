---
title: Mayo-LDCT leaderboard
description: Real-helical-data leaderboard. Geometry validation complete; solver autoresearch pending. See solver_plan.md for methodology.
---

# Mayo-LDCT leaderboard (Wagner split)

AAPM 2016 Low-Dose CT Grand Challenge data — helical Siemens SOMATOM
AS+, rebinned to 2-D fan-beam via the in-house
[`helix2fan`](../../ddssl_ldct/helix2fan.py) SSR pipeline. Wagner split:

```
Train: L145, L186, L209, L219      (4 patients)
Val:   L277                         (1 patient)
Test:  L014, L056, L058, L075, L123 (5 patients)
```

## Status

**Geometry calibration complete (2026-05-26); solver autoresearch pending.**

The helical-to-fan rebin pipeline has been calibrated end-to-end against
the L014 B30f reference recon. Best L014 multi-GT joint fit (10 central
slices sharing all parameters, with `pixel_spacing` consistent between
FBP geometry and FoV mask; SLURM 762369):

- **SSIM mean = 0.9676, PSNR mean = 42.92 dB, RMSE mean = 0.00036**
- range over 10 slices: SSIM [0.9646, 0.9704], PSNR [42.39, 43.35] dB
- vs. the original (uncalibrated) pipeline: SSIM = 0.882, PSNR = 34.5 dB
- Δ ≈ **+8.7 dB PSNR / +0.083 SSIM / −63 % RMSE** from end-to-end joint
  geometry fit.

The remaining ~3 % SSIM gap to truth is architectural (2-D SSR vs. full
3-D cone-beam; B30f kernel MTF mismatch; slab-profile shape) and is not
expected to close further without a 3-D recon model.

### Key calibration findings

Documented in [`findings.md`](../findings.md):

- **Fitted pixel spacing = 0.700857 mm** beats Mayo's DICOM-nominal
  `PixelSpacing = 0.703125 mm = 360/512` by **+8.4 dB PSNR** at the
  joint-fit optimum (`ssim 0.9622 vs 0.9444` at fixed other params).
  The −0.32 % deviation corresponds to a sub-percent detector-pitch /
  effective-sdd correction that no DICOM tag exposes.
- **FFS-z (flying-focal-spot, axial) sign**: α_dz = +1 (additive
  convention, `ffs_dz` is added to source-z in the rebin step). Verified
  by 3-point ablation `α_dz ∈ {−1, 0, +1}`. Effect is small (Δz
  parameter absorbs most of it) but consistent.
- **FFS-ρ (radial)**: no effect at the calibrated metric — two-point
  linear FoV-masked calibration absorbs the 0.92 % alternate-readout
  magnification.
- **FFS-φ (in-plane)**: confirmed zero / no-op in the Mayo SOMATOM
  AS+ data (no in-plane FFS programmed for these scans).
- **B30f kernel mismatch**: PYRO-NN's `hann` filter is the closest
  PYRO-NN approximation but is not identical MTF; shows up as faint
  smoothing differences in the diff panel.

L014 calibrated FBP-vs-truth, top configurations (single-anchor slab
averaging at truth `SliceThickness = 5 mm`):

| Config | SSIM | PSNR (dB) | RMSE | Source |
|---|---:|---:|---:|---|
| Joint multi-GT fit (10 slices, consistent `pixel_spacing`) | **0.9676** | **42.92** | 0.00036 | `scripts/fit_rebin_end2end_L014.py` (SLURM 762369) |
| Pixel-spacing ablation `ps=0.700857` | 0.9622 | 42.27 | — | `scripts/pixel_spacing_ablation_L014.py` |
| Mayo DICOM nominal `ps=0.703125` | 0.9444 | 33.91 | — | `scripts/pixel_spacing_ablation_L014.py` |
| Baseline rebin + intensity calibrate | 0.8819 | 34.47 | 0.00094 | (baseline in fit script) |

## Solver leaderboard

🟢 **Autoresearch loop is now live** (2026-06-03). First Mayo solver
above baseline: **Learned Primal-Dual** at LPD-iter-2 (shrunk to
hidden=32 to fit Q6000 vs the breast-CT champion's hidden=96 which
OOMed on Mayo's 2304-angle sino). Loop continuing per `solver_plan.md`
Step 2 — see `docs/runs/mayo-ldct-claude-agentic-*-search-20260603-01/`.

| Rank | Solver | Variant | params (M) | SSIM | hr | Source | Comparison |
|---:|---|---|---:|---:|---:|---|---|
| 1 | **Learned Primal-Dual** | I=4, hidden=48, n_p=n_d=3, ep=3, lr=3.2e-4, train_n=100 (iter-3) | 0.193 | 0.4681 | **0.2445** | [results](../runs/mayo-ldct-claude-agentic-learned-primal-dual-search-20260603-01/results.tsv) | [iter-3](../runs/mayo-ldct-claude-agentic-learned-primal-dual-search-20260603-01/iterations/iter-0003/comparison.png) |
| 2 | **USwin** | c=16, win=8, heads=8, ep=3, train_n=50 (iter-2) | — | 0.3747 | **0.1425** | [results](../runs/mayo-ldct-claude-agentic-uswin-search-20260603-01/results.tsv) | [iter-2](../runs/mayo-ldct-claude-agentic-uswin-search-20260603-01/iterations/iter-0002/comparison.png) |
| 3 | **DD-UNet supervised L2** | c=24, ep=3, lr=5e-4, train_n=100 (iter-3) | — | 0.4200 | **0.1337** | [results](../runs/mayo-ldct-claude-agentic-dual-domain-supervised-search-20260603-01/results.tsv) | [iter-3](../runs/mayo-ldct-claude-agentic-dual-domain-supervised-search-20260603-01/iterations/iter-0003/comparison.png) |
| 4 | **NAF** (per-scene MLP) | n_freqs=6, hidden=192, layers=5, n_iter=2000 (iter-1) | 0.143 | 0.5395 | **0.0202** | [results](../runs/mayo-ldct-claude-agentic-naf-search-20260603-01/results.tsv) | [iter-1](../runs/mayo-ldct-claude-agentic-naf-search-20260603-01/iterations/iter-0001/comparison.png) |
| 5 | **DD-BF N2I** | proj/img_n_bf=3, ep=3, lr=5e-4, train_n=50 (iter-1) | 0.000018 | 0.4868 | **0.0047** | [results](../runs/mayo-ldct-claude-agentic-dual-domain-bilateral-n2i-search-20260603-01/results.tsv) | [iter-1](../runs/mayo-ldct-claude-agentic-dual-domain-bilateral-n2i-search-20260603-01/iterations/iter-0001/comparison.png) |

### Structural deal-breakers + plateaued (filed 2026-06-03)

| Solver | Final state | Why deprioritised |
|---|---|---|
| **DD-BF supervised L2** | iter-3 hr=0 (2 consecutive hr=0) | Loss stuck — 18-parameter BF too low-capacity for Mayo. The breast-CT hr=0.26 variant cannot transfer. |
| **RAM zero-shot (pretrained)** | iter-3 hr=0 (3 consecutive hr=0) | SSIM crept 0.40→0.48 but PSNR ceiling 12.45 < baseline 12.59. `ram.pth.tar` (natural images) cannot bridge to Mayo μ-range. |
| **Learned Primal-Dual** | iter-3 winner hr=**0.2445**; iter-4/5 regressed (loss explosion at hidden=64 + lpd_iters=6) | Capacity scaling exhausted. Both width-up and depth-up broke the loss landscape at train_n=100. iter-3 stays — Step 3 TPE next. |
| **DD-UNet supervised L2** | iter-3 winner hr=**0.1337**; iter-4/5 regressed (c=32 and ep=6 both worse) | Plateaued at c=24, ep=3. iter-3 stays — Step 3 TPE next. |
| **USwin** | iter-2 winner hr=**0.1425**; iter-3/4 OOMed, iter-5 (ep=6) regressed to 0.107 | Plateaued at c=16, win=8, ep=3, train_n=50. iter-2 stays — Step 3 TPE next. |
| **ITNet v3** | iter-1+2 both OOM (FBP inside the unrolled body at 5 GB even with train_n=50) | Structural OOM: each unrolled iter calls FBP, so memory scales with model.itnet_k. Needs chunked-FBP in solver_itnet_v3.py before Mayo retry. |
| **Hammernik VN** | iter-2/3 hr=0, SSIM 0.27→0.29 (2 consecutive hr=0) | Loss decreasing but val PSNR ceiling 12.14 < baseline 12.59. T=3 and T=5 both undercooked vs the sino complexity; larger T would risk OOM. |
| **TV-iterative supervised** | iter-1 hr=0 SSIM 0.30 (loss STUCK at 0.001, step/λ scalars stuck at init) | Same structural verdict as breast-CT: FBP init makes 1st GD step a no-op (data-fidelity gradient ≈ 0 at FBP), so step/λ scalars never get a useful gradient. |
| **Wu 2015 trainable** | iter-1 hr=0 SSIM 0.34, iter-2 hr=0 SSIM 0.34 (ep 3→6, no improvement) | Loss converged at 0.00013 by ep-1; the 10 trainable scalars *do* move (blend 0.97→0.74) but final SSIM stays at 0.34. The 10-param image-domain BF-like solver hits the same low-capacity ceiling here as it did on breast-CT (which got hr=0.22 only because that dataset has a much narrower dynamic range). |
| **ITNet v2** | iter-1 OOM in `filter_sino` (2.53 GiB FFT pad) | Two-bug combo: (a) `solver_itnet_v2.py` ignores the agentic JSON cfg and silently keeps its hardcoded `train_n=400`, `val_n=100`, `itnet_k=5` defaults → FBP scratch blows past 24 GB. Fix is structural in `solver_itnet_v2.py` (read cfg from `CFG_JSON` env like solver_itnet_v3 does) before any retry. |
| **Hammernik 2017** | iter-1 hr=0 SSIM 0.27, PSNR 11.55 < baseline 12.59 (λ_t stuck within 0.8 % of init) | Same Hammernik family failure as VN: the per-step λ regulariser barely budges over 3 epochs and the final recon is biased darker than baseline. Sino-complexity ceiling, not a config knob away from working. |
| **DD-UNet N2I** | iter-1 hr=0 SSIM 0.46, PSNR 12.52 < baseline 12.59 (loss 0.00001 from epoch 1) | Same N2I noise-floor over-smoothing as breast-CT: loss is already at the N2I supervision floor by ep-1, so the network has nowhere to climb. Supervised L2 (rank 3, hr=0.13) is the right DD-UNet variant for Mayo. |

### Known infrastructure cap: Mayo FBP requires train_n ≤ 50 (Q6000, 24 GB)

`PyronnFanBeamProjector.fbp` on Mayo's 2304-angle sino allocates 2.5–5 GB of FFT scratch with `train_n=100`; combined with model+gradient memory this exceeds Q6000. **USwin iter-3/4, ITNet v3 iter-1, Hammernik VN iter-1 all OOMed at this exact line.** All Mayo iter-N+1 configs now use `train_n=50` (was the silent default in iter-2 USwin which fit). If a solver needs more data, the fix is chunked FBP in `solver_*.py`, not just bumping the GPU class.

**Autoresearch loop active — 3 jobs in flight as of 2026-06-07 ~12:42:**

| Solver | Latest result | Next hypothesis (short budget, 30-min wall) |
|---|---|---|
| **NAF** | iter-2 hr=0 SSIM 0.5277 at val_n=3 (INCONCLUSIVE — I changed val_n 5→3 and the baseline jumped 12.59→13.98) | iter-3 (**762810**): revert val_n 3→5 (apples-to-apples with iter-1), keep `naf_n_iter=4000`. If hr ≥ 0.03 the extra iters helped; if hr ~0.02 plateaued and we file. |
| **R²-Gaussian** | iter-1 still running (23 min in) | (await iter-1 result) |
| **DD-BF N2I** | iter-1 hr=**0.0047** SSIM 0.487 (PSNR 12.63 vs baseline 12.59; just above plateau threshold) | iter-2 (**762811**): ep 3 → 6 (the 18 BF scalars only moved ~1 % over 3 epochs). If hr < 0.01 at ep=6, file plateau. |

The previous batch outcome:
- 762802 NAF iter-2 → inconclusive (val_n change), retest as iter-3 above.
- 762806 R²-G iter-1 → still running (per-scene, 23 min into 30-min wall).
- 762807 DD-UNet N2I iter-1 → STOP (N2I noise-floor over-smoothing — verdict filed in the table above).
- 762808 DD-BF N2I iter-1 → marginal hit hr=0.0047, retried as iter-2 above.

## Plan

Once the rebin is fully blessed (job 762369 verification + bulk
re-rebin):

1. **Re-rebin the remaining 9 patients** (low-dose + 9 missing
   full-dose) with the fitted geometry + FFS-z correction.
2. **Per-solver autoresearch + TPE refinement** on the Wagner
   train/val split. Solvers in order of expected promise (from
   breast-CT leaderboard):
   - Learned Primal-Dual (current breast-CT champion at `hr` = 0.91)
   - DD-UNet supervised L2 (`hr` = 0.84)
   - ITNet v3, USwin
   - RAM zero-shot (pretrained — distribution match on Mayo TBD)
   - Hammernik VN
   - DD-BF supervised L2
3. **DDPM training**: two variants per
   [`solver_plan.md`](../../solver_plan.md) Step 4. Constrained uses
   `Train: L145/186/209/219` labels; unconstrained uses all 10 patients.
4. **Diff-recon TPE** on both DDPM variants.

## Methodology

See [`solver_plan.md`](../../solver_plan.md). Geometry-calibration
methodology and the full pixel-spacing / FFS-sign ablation history are
recorded in [`findings.md`](../findings.md) (newest entries first).
