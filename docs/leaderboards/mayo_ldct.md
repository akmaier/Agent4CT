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
| 5 | **TV-iterative** (non-trainable) | tv_iterations=400, tv_lambda=0.01, tv_step=0.5 (iter-3) | 0 | 0.5214 | **0.0197** | [results](../runs/mayo-ldct-claude-agentic-tv-iterative-search-20260603-01/results.tsv) | [iter-3](../runs/mayo-ldct-claude-agentic-tv-iterative-search-20260603-01/iterations/iter-0003/comparison.png) |
| 6 | **DD-BF N2I** | proj/img_n_bf=3, ep=3, lr=5e-4, train_n=50 (iter-1) | 0.000018 | 0.4868 | **0.0047** | [results](../runs/mayo-ldct-claude-agentic-dual-domain-bilateral-n2i-search-20260603-01/results.tsv) | [iter-1](../runs/mayo-ldct-claude-agentic-dual-domain-bilateral-n2i-search-20260603-01/iterations/iter-0001/comparison.png) |

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

**Autoresearch loop active — 3 jobs queued as of 2026-06-07 ~14:43 (Step-2 finishing + diff-recon unlocked):**

| Solver | Latest result | Next hypothesis |
|---|---|---|
| **TV-iterative** (non-trainable) | iter-3 hr=**0.0197** SSIM 0.5214 (with cfg-fix; 400 actual iters). Loss 4.86→4.15 from iter-200→400 — still falling. | iter-4 (**762821**): `tv_iterations` 400 → 800. Hyp: if loss keeps falling, hr pushes into 0.025–0.035. If Δhr < 0.005 vs iter-3, plateau and iter-3 stays. |
| **diff_recon_dcstep_unconstrained_mayo_v2** | (just started) | iter-1 (**762822**) — DPS against the new uncon DDPM v2 ckpt. Smallest budget from breast_v3 TPE space (sample_steps=200, dcstep_n_cg=10, eta=30, fbp init). |
| **diff_recon_dcstep_constrained_mayo_v2** | (just started) | iter-1 (**762823**) — same DPS budget, against the 50-sample constrained DDPM v2 ckpt (the variant that typically wins for diff-recon on breast). |

**Both Mayo DDPM v2 ckpts READY:**
- `/cluster/maier/Agent4CT/checkpoints/ddpm_mayo_unconstrained_v2.pt` (3.823 M params, 200 train, best val ε-loss=0.0049)
- `/cluster/maier/Agent4CT/checkpoints/ddpm_mayo_constrained_v2.pt` (3.823 M params, 50 train, best val ε-loss=0.0087)

The `scripts/claude_agentic_one_iter.py` SOLVER_MAP picked up two new keys: `diffusion_recon_dcstep_unconstrained_mayo_v2` and `diffusion_recon_dcstep_constrained_mayo_v2`. The ckpt path is passed inside the per-iter CFG_JSON as `recon_ckpt`, so no solver-side patching was needed.

The previous batch outcome:
- 762819 Mayo DDPM v2 constrained → **COMPLETED** in 4:11 (50-sample dataset → fast).
- 762820 TV-iter iter-3 (cfg-fix landed) → hr=0.0197 ▲ vs iter-1's hr=0.0108. iter-4 dispatched above.

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
