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

### Structural deal-breakers (filed 2026-06-03)

| Solver | Final state | Why deprioritised |
|---|---|---|
| **DD-BF supervised L2** | iter-3 hr=0 (2 consecutive hr=0) | Loss stuck at 0.00016 across both λ_neg=5 and λ_neg=2 — the 18-parameter BF stack is too low-capacity for Mayo's complexity. The variant that worked on breast-CT at hr=0.26 cannot transfer. |
| **RAM zero-shot (pretrained)** | iter-3 hr=0 (3 consecutive hr=0) | SSIM monotonically increased 0.40 → 0.48 → 0.48 across blend sweep, but PSNR ceiling 12.45 dB < baseline 12.59 dB. The pretrained `ram.pth.tar` (natural images) cannot bridge to Mayo's μ-intensity range. |
| **Learned Primal-Dual — agentic loop plateaued** | iter-3 hr=0.2445 is the winner; iter-4 (hidden 64) and iter-5 (lpd_iters=6) both regressed (gradient instability with deeper / wider nets at train_n=100). | iter-3 stays as Mayo LPD agentic winner. Both width-up (iter-4) and depth-up (iter-5) attempts broke the loss landscape — short-budget train_n=100 isn't enough to support more capacity. Step 3 (TPE around iter-3) is the next move. |

### Known infrastructure cap: Mayo FBP requires train_n ≤ 50 (Q6000, 24 GB)

`PyronnFanBeamProjector.fbp` on Mayo's 2304-angle sino allocates 2.5–5 GB of FFT scratch with `train_n=100`; combined with model+gradient memory this exceeds Q6000. **USwin iter-3/4, ITNet v3 iter-1, Hammernik VN iter-1 all OOMed at this exact line.** All Mayo iter-N+1 configs now use `train_n=50` (was the silent default in iter-2 USwin which fit). If a solver needs more data, the fix is chunked FBP in `solver_*.py`, not just bumping the GPU class.

**Autoresearch loop active — iter-5+ in flight as of 2026-06-03 ~11:45:**

| Solver | Latest result | iter-N+1 hypothesis (short budget, 30-min wall) |
|---|---|---|
| DD-UNet sup | iter-3 hr=**0.1337** (c=24); iter-4 (c=32) regressed to 0.089 | iter-5: ep 3 → 6 at c=24 (more training of best config; capacity scaling exhausted) |
| USwin | iter-2 hr=**0.1425** (c=16, train_n=50); iter-3/4 OOMed at train_n=100 | iter-5: ep 3 → 6 at iter-2 config + train_n=50 (FBP fits, more training) |
| ITNet v3 | iter-1 ❌ FBP-OOM at train_n=100 | iter-2: train_n 100 → 50 (feasibility re-test, then iter-3 moves on real knob) |
| Hammernik VN | iter-1 ❌ FBP-OOM at train_n=100 | iter-2: train_n 100 → 50 (same fix) |

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
