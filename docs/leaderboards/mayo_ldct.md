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

🟢 **Autoresearch loop ran 2026-06-03 → 2026-06-07.** Eight solvers
beat baseline; four structural surprises stand out:
1. **NAF** worked on Mayo despite the breast-CT structural verdict (per-scene per-voxel implicit field finds enough signal at 2304 angles).
2. **R²-Gaussian** still fails on Mayo (per-scene optimisation too expensive for the 30-min autoresearch wall).
3. **DDPM-prior diff-recon works** at hr=0.06+ — the breast-CT verdict that "DDPM-prior diff-recon doesn't beat baseline" does not transfer; the Mayo dataset is more strongly aligned with the DDPM training prior.
4. **TV-iterative (non-trainable)** is still climbing past iter-5 — pure classical TV proximal gradient on FBP init is a real positive on the helical dataset.

Loop continuing per `solver_plan.md` Step 2 — see `docs/runs/mayo-ldct-claude-agentic-*-search-20260603-01/`.

| Rank | Solver | Variant | params (M) | SSIM | hr | Source | Comparison |
|---:|---|---|---:|---:|---:|---|---|
| 1 | **Learned Primal-Dual** | I=4, hidden=48, n_p=n_d=3, ep=3, lr=3.2e-4, train_n=100 (iter-3) | 0.193 | 0.4681 | **0.2445** | [results](../runs/mayo-ldct-claude-agentic-learned-primal-dual-search-20260603-01/results.tsv) | [iter-3](../runs/mayo-ldct-claude-agentic-learned-primal-dual-search-20260603-01/iterations/iter-0003/comparison.png) |
| 2 | **USwin** | c=16, win=8, heads=8, ep=3, train_n=50 (iter-2) | — | 0.3747 | **0.1425** | [results](../runs/mayo-ldct-claude-agentic-uswin-search-20260603-01/results.tsv) | [iter-2](../runs/mayo-ldct-claude-agentic-uswin-search-20260603-01/iterations/iter-0002/comparison.png) |
| 3 | **DD-UNet supervised L2** | c=24, ep=3, lr=5e-4, train_n=100 (iter-3) | — | 0.4200 | **0.1337** | [results](../runs/mayo-ldct-claude-agentic-dual-domain-supervised-search-20260603-01/results.tsv) | [iter-3](../runs/mayo-ldct-claude-agentic-dual-domain-supervised-search-20260603-01/iterations/iter-0003/comparison.png) |
| 4 | **diff_recon DCstep unconstrained** (DDPM v2 prior) | DPS, sample_steps=200, eta=30, dcstep_n_cg=10, FBP init (iter-1) | 3.823 | 0.5309 | **0.0647** | [results](../runs/mayo-ldct-claude-agentic-diff-recon-dcstep-unconstrained-mayo-v2-search-20260603-01/results.tsv) | [iter-1](../runs/mayo-ldct-claude-agentic-diff-recon-dcstep-unconstrained-mayo-v2-search-20260603-01/iterations/iter-0001/comparison.png) |
| 5 | **diff_recon DCstep constrained** (DDPM v2 prior) | DPS, sample_steps=200, eta=30, dcstep_n_cg=10, FBP init (iter-1) | 3.823 | 0.4892 | **0.0560** | [results](../runs/mayo-ldct-claude-agentic-diff-recon-dcstep-constrained-mayo-v2-search-20260603-01/results.tsv) | [iter-1](../runs/mayo-ldct-claude-agentic-diff-recon-dcstep-constrained-mayo-v2-search-20260603-01/iterations/iter-0001/comparison.png) |
| 6 | **TV-iterative** (non-trainable) | tv_iterations=3200, tv_lambda=0.01, tv_step=0.5 (iter-6) | 0 | 0.5387 | **0.0433** | [results](../runs/mayo-ldct-claude-agentic-tv-iterative-search-20260603-01/results.tsv) | [iter-6](../runs/mayo-ldct-claude-agentic-tv-iterative-search-20260603-01/iterations/iter-0006/comparison.png) |
| 7 | **NAF** (per-scene MLP) | n_freqs=6, hidden=192, layers=5, n_iter=2000 (iter-1) | 0.143 | 0.5395 | **0.0202** | [results](../runs/mayo-ldct-claude-agentic-naf-search-20260603-01/results.tsv) | [iter-1](../runs/mayo-ldct-claude-agentic-naf-search-20260603-01/iterations/iter-0001/comparison.png) |
| 8 | **DD-BF N2I** | proj/img_n_bf=3, ep=3, lr=5e-4, train_n=50 (iter-1) | 0.000018 | 0.4868 | **0.0047** | [results](../runs/mayo-ldct-claude-agentic-dual-domain-bilateral-n2i-search-20260603-01/results.tsv) | [iter-1](../runs/mayo-ldct-claude-agentic-dual-domain-bilateral-n2i-search-20260603-01/iterations/iter-0001/comparison.png) |

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
| **DD-BF N2I** | iter-1 hr=0.0047, iter-2 (ep 3→6) hr=0.0035 — Δhr = −0.0012 (plateau) | The 18 BF scalars kept moving (σ_y 1.973→1.953 from ep-3 to ep-6) but val SSIM stayed flat at 0.485 and hr actually slipped. iter-1 stays as rank 5; cap is structural (18 trainable params can't beat the FBP baseline by more than the noise floor). |
| **R²-Gaussian** | iter-1 TIMEOUT at 30 min (per-scene fit takes longer than the autoresearch sbatch wall allows) | gs_n_iter=3000 + FBP² init at train_n=2, val_n=2 still didn't complete a single per-scene fit within the 30-min cluster_guide §4.6 budget. Same breast-CT structural verdict applies — R²-Gaussian's per-scene optimisation is too expensive for the autoresearch loop on Mayo's 2304-angle, 512² scenes. Would need a TPE-scale 4-h job to test seriously. |
| **NAF** plateau verdict | iter-1 hr=**0.0202** (the rank-4 entry); iter-4 (naf_n_iter 2000 → 3000 at val_n=5) regressed to hr=0.0010, SSIM 0.5395 → 0.4843 | NAF **overshoots** beyond 2000 iters on Mayo: the implicit-field MLP starts hallucinating high-frequency detail that does not match the truth slab, pulling SSIM down by ~5.5 pts. iter-1's `naf_n_iter=2000` is the optimum. iter-1 stays as rank 4 — Step 3 TPE would need to bracket below 2000 iters, not above. |

### Known infrastructure cap: Mayo FBP requires train_n ≤ 50 (Q6000, 24 GB)

`PyronnFanBeamProjector.fbp` on Mayo's 2304-angle sino allocates 2.5–5 GB of FFT scratch with `train_n=100`; combined with model+gradient memory this exceeds Q6000. **USwin iter-3/4, ITNet v3 iter-1, Hammernik VN iter-1 all OOMed at this exact line.** All Mayo iter-N+1 configs now use `train_n=50` (was the silent default in iter-2 USwin which fit). If a solver needs more data, the fix is chunked FBP in `solver_*.py`, not just bumping the GPU class.

**Autoresearch loop active — 3 jobs queued as of 2026-06-07 ~16:07:**

| Solver | Latest result | Next hypothesis |
|---|---|---|
| **TV-iterative** (non-trainable) | iter-6 hr=**0.0433** (was 0.0362 at iter-5). Δhr: +0.011 → +0.009 → +0.008 → +0.007 — converging. Loss 3.25 at iter-3200, still falling. | iter-7 (**762839**, 60-min wall): `tv_iterations` 3200 → 6400. Hyp: hr lifts above 0.05 if loss keeps falling; if Δhr<0.005, plateau finally lands. |
| **diff_recon UNCON** | iter-3 (n_cg 10→20) hr=0.0544 — also **REGRESSED** from iter-1's 0.0647. Both compute-axis knobs (sample_steps, n_cg) hurt. | iter-4 (**762840**): try lower step size `recon_eta` 30 → 10. Hyp: more-conservative DPS updates = less hallucination = hr ≥ 0.07. If Δhr<0.005, the iter-1 corner of the space is the optimum and we plateau. |
| **diff_recon CON** | iter-3 (n_cg 10→20) hr=0.0526 — marginal regression from iter-1's 0.0560 (Δhr=-0.003 already within plateau threshold). | iter-4 (**762841**): same `recon_eta` 30 → 10 test as uncon iter-4. If marginal regression, file plateau at iter-1's 0.0560. |

The previous batch outcome:
- 762833 TV-iter iter-6 → hr=0.0433 (rank 6, climbing). iter-7 dispatched.
- 762834 diff_recon UNCON iter-3 → hr=0.0544 (knob-axis #2 regression). iter-1 still rank 4.
- 762835 diff_recon CON iter-3 → hr=0.0526 (marginal regression). iter-1 still rank 5.

**Diff-recon space map so far:** iter-1 config (`sample_steps=200`, `n_cg=10`, `eta=30`, `fbp init`) is at a local optimum. Two compute-up directions (more sample_steps, more n_cg) both hurt → the system is **compute-saturated** at iter-1. iter-4 tests the orthogonal axis (smaller `eta` = different dynamics, not more compute) before filing plateau.

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
