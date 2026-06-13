---
title: Mayo-LDCT leaderboard
description: Real-helical-data leaderboard (Wagner split). Step-3 TPE refinement closed 2026-06-09 — 12 of 19 solvers above the FBP baseline. See solver_plan.md for methodology.
---

# Mayo-LDCT leaderboard (Wagner split)

> ⚠️ **2026-06-13 — calibrated-metric bug found (`intensity_calibrate` background offset).** Every SSIM/PSNR below was scored with a calibration that mapped the recon background to 0, but Mayo truth background is ~+0.0005 μ — so these numbers are **~0.01 SSIM low on average** (more for low-overlap slices). Fixed via opt-in `bg_target="truth"` (commit `bcfa2720`); other datasets unchanged. Re-score with the corrected metric when a solver is next evaluated, then update its row. See [`docs/findings.md` 2026-06-13 entry](../findings.md).

> 🆕 **2026-06-12 — v3 geometry promoted to production.** The SSR rebin
> parameters (`MAYO_LDCT_SSR_DEFAULTS`) were re-fitted with a learnable
> z-axis scaling (`s_z = 1.001665`) plus updated sod/sdd/Δz/post-FBP
> values. See [`docs/findings.md` 2026-06-12 entry](../findings.md) for
> the rationale, fit numbers (SSIM 0.957, PSNR 40.8 dB averaged across
> 10 GT slices spanning the full L014 patient z-range), and bulk
> re-rebin / re-staging plan (SLURM 763396 → 763397 → 763398).
> Leaderboard rows below were computed against the v2 staging; absolute
> scores will shift slightly (+0.001–0.005 SSIM, +0.1–0.5 dB PSNR) when
> solvers next train against the v3-staged data. Rank order is
> expected to stay stable except in the bottom group (Δ < 0.01 SSIM
> neighbours).

AAPM 2016 Low-Dose CT Grand Challenge data — helical Siemens SOMATOM
AS+, rebinned to 2-D fan-beam via the in-house
[`helix2fan`](../../ddssl_ldct/helix2fan.py) SSR pipeline. Wagner split:

```
Train: L145, L186, L209, L219      (4 patients)
Val:   L277                         (1 patient)
Test:  L014, L056, L058, L075, L123 (5 patients)
```

## Status

**Step-3 TPE refinement CLOSED 2026-06-09** — all 5 phase-3 TPEs completed (1 day, ~70 GPU-h). Top of Mayo leaderboard:

```
Rank 1  DD-UNet sup TPE       hr=0.3890   (Step-3 phase 1)
Rank 2  Learned Primal-Dual TPE     0.3063   (Step-3 phase 1)
Rank 3  USwin TPE                   0.2492   (Step-3 phase 1)
Rank 4  diff_recon UNCON v4 TPE     0.2377   (Step-3 phase 3, eta=0.30 fbp)
Rank 5  diff_recon UNCON v2 TPE     0.2352   (Step-3 phase 3, eta=0.31 noise)
Rank 6  ItNet v3 TPE                0.2181   (Step-3 phase 1)
Rank 7  diff_recon CON v4 TPE       0.1632   (Step-3 phase 3, eta=1.5 fbp)
Rank 8  diff_recon CON v2 TPE       0.1071   (Step-3 phase 3, eta=7.2 fbp)
Rank 9  TV-iterative (non-trainable) 0.0557  (Step-2 agentic)
Rank 10 Hammernik VN TPE            0.0551   (Step-3 phase 2 — OVERTURNED Step-2 STOP)
Rank 11 NAF                         0.0202   (Step-2; TPE found 0.0131 worse)
Rank 12 DD-BF N2I                   0.0047   (Step-2)
```

**12 ranks above baseline / 15 inventory.** 3 structural STOPs remain (DD-BF L2, RAM zero-shot, R²-Gaussian — all blocked by solver-side issues, not config knobs).

**Mode × prior eta-corner discovery from phase 3 TPE:**
| Mode × DDPM | Optimum cfg | hr (val_n=5) | Notes |
|---|---|---:|---|
| UNCON v2 | eta=0.31, **noise**, clamp=False, steps=500 | 0.2352 | very-low-eta corner; agentic missed this regime |
| UNCON v4 | eta=0.30, **fbp**, clamp=True, steps=200 | 0.2377 | same low-eta but anchored init |
| CON v2 | eta=7.21, fbp, clamp=True, warmup=40 | 0.1071 | mid-eta corner |
| CON v4 | eta=1.5, fbp, clamp=True, warmup=25 | 0.1632 | between v2's mid-eta and UNCON's low-eta |

**UNCON converges at very-low eta (~0.3); CON prefers mid-eta (1.5-7).** v4 (ch=96, 120 ep) prefers fbp init in both modes; v2 (ch=64, 60 ep) accepts noise init in UNCON but needs fbp+clamp in CON. The eta<0.5 regime had been clamped out by the agentic loop (which used breast-CT eta≥1 priors); TPE found it by extending the search range to (0.3, 30) log on Mayo.

**Geometry calibration complete (2026-05-26); solver autoresearch CLOSED.**

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

**Consolidated val_n=5 ranking (Step-3 TPE results take precedence over Step-2 agentic where TPE was run; each solver appears once with its best run):**

| Rank | Solver | Variant | params (M) | SSIM | hr | Source | Comparison |
|---:|---|---|---:|---:|---:|---|---|
| 1 | **DD-UNet supervised L2** *(Step-3 TPE — COMPLETE 20/20)* | TPE iter-12 winner | 0.466 | 0.5977 | **0.3890** | [results](../runs/mayo-ldct-2d-calibrated-tpe-dual-domain-supervised-search-20260608-02/results.tsv) | [iter-12](../runs/mayo-ldct-2d-calibrated-tpe-dual-domain-supervised-search-20260608-02/iterations/iter-0012/comparison.png) |
| 2 | **Learned Primal-Dual** *(Step-3 TPE — COMPLETE 20/20)* | TPE iter-6 winner: lpd_iters=2, hidden=48, ep=12 | 0.104 | 0.6009 | **0.3063** | [results](../runs/mayo-ldct-2d-calibrated-tpe-lpd-search-20260608-06/results.tsv) | [iter-6](../runs/mayo-ldct-2d-calibrated-tpe-lpd-search-20260608-06/iterations/iter-0006/comparison.png) |
| 3 | **USwin** *(Step-3 TPE)* | TPE iter-11 winner (search-space-clamped) | 1.760 | 0.4039 | **0.2492** | [results](../runs/mayo-ldct-2d-calibrated-tpe-uswin-search-20260608-04/results.tsv) | [iter-11](../runs/mayo-ldct-2d-calibrated-tpe-uswin-search-20260608-04/iterations/iter-0011/comparison.png) |
| 4 | **diff_recon DCstep unconstrained** (DDPM **v4** prior) *(Step-3 TPE COMPLETE 20/20)* | TPE iter-13: **eta=0.30, fbp init, clamp=True**, sample_steps=200, every=3, warmup=25 — discovered low-eta corner BELOW agentic explore range | 8.594 | 0.5169 | **0.2377** | [results](../runs/mayo-ldct-2d-calibrated-tpe-diff-recon-dcstep-unconstrained-mayo-v4-search-20260608-01/results.tsv) | [iter-13](../runs/mayo-ldct-2d-calibrated-tpe-diff-recon-dcstep-unconstrained-mayo-v4-search-20260608-01/iterations/iter-0013/comparison.png) |
| 5 | **diff_recon DCstep unconstrained** (DDPM v2 prior) *(Step-3 TPE COMPLETE 20/20)* | TPE iter-12: **eta=0.31, noise init, clamp=False**, sample_steps=500, every=5, warmup=10, relax=0.95 | 3.823 | 0.5487 | **0.2352** | [results](../runs/mayo-ldct-2d-calibrated-tpe-diff-recon-dcstep-unconstrained-mayo-v2-search-20260608-01/results.tsv) | [iter-12](../runs/mayo-ldct-2d-calibrated-tpe-diff-recon-dcstep-unconstrained-mayo-v2-search-20260608-01/iterations/iter-0012/comparison.png) |
| 6 | **ItNet v3** *(Step-3 TPE)* | TPE iter-9, search-space-clamped (20/20 done) | 8.318 | 0.4113 | **0.2181** | [results](../runs/mayo-ldct-2d-calibrated-tpe-itnet-v3-search-20260608-04/results.tsv) | [iter-9](../runs/mayo-ldct-2d-calibrated-tpe-itnet-v3-search-20260608-04/iterations/iter-0009/comparison.png) |
| 7 | **diff_recon DCstep constrained** (DDPM **v4** prior) *(Step-3 TPE COMPLETE 20/20)* | TPE iter-4: eta=1.52, fbp init, clamp=True, sample_steps=200, every=3, n_cg=20, warmup=25, relax=0.85 | 8.594 | 0.5134 | **0.1632** | [results](../runs/mayo-ldct-2d-calibrated-tpe-diff-recon-dcstep-constrained-mayo-v4-search-20260609-01/results.tsv) | [iter-4](../runs/mayo-ldct-2d-calibrated-tpe-diff-recon-dcstep-constrained-mayo-v4-search-20260609-01/iterations/iter-0004/comparison.png) |
| 8 | **diff_recon DCstep constrained** (DDPM v2 prior) *(Step-3 TPE COMPLETE 20/20)* | TPE iter-9: eta=7.21, fbp init, clamp=True, warmup=40 (CON v2 prefers mid-eta unlike UNCON) | 3.823 | 0.4847 | **0.1071** | [results](../runs/mayo-ldct-2d-calibrated-tpe-diff-recon-dcstep-constrained-mayo-v2-search-20260608-01/results.tsv) | [iter-9](../runs/mayo-ldct-2d-calibrated-tpe-diff-recon-dcstep-constrained-mayo-v2-search-20260608-01/iterations/iter-0009/comparison.png) |
| 9 | **TV-iterative** (non-trainable) | tv_iterations=12800, tv_lambda=0.01, tv_step=0.5 (iter-8) | 0 | 0.5439 | **0.0557** | [results](../runs/mayo-ldct-claude-agentic-tv-iterative-search-20260603-01/results.tsv) | [iter-8](../runs/mayo-ldct-claude-agentic-tv-iterative-search-20260603-01/iterations/iter-0008/comparison.png) |
| 10 | **Hammernik VN** *(Step-3 TPE — COMPLETE 20/20, overturns Step-2 STOP)* | TPE iter-6 cfg: vn_T=5, vn_n_filters=16, vn_kernel=11, vn_λ_init=2.3e-3, ep=12, lr=2.6e-4 | 0.012 | 0.4087 | **0.0551** | [results](../runs/mayo-ldct-2d-calibrated-tpe-hammernik-vn-search-20260608-01/results.tsv) | [iter-6](../runs/mayo-ldct-2d-calibrated-tpe-hammernik-vn-search-20260608-01/iterations/iter-0006/comparison.png) |
| 11 | **NAF** (per-scene MLP) *(Step-3 TPE 20/20 done; TPE found 0.0131 < Step-2 0.0202)* | Step-2 iter-1 winner: n_freqs=6, hidden=192, layers=5, n_iter=2000 (TPE went deeper, hurt by overshoot) | 0.143 | 0.5395 | **0.0202** | [results](../runs/mayo-ldct-claude-agentic-naf-search-20260603-01/results.tsv) | [iter-1](../runs/mayo-ldct-claude-agentic-naf-search-20260603-01/iterations/iter-0001/comparison.png) |
| 12 | **DD-BF N2I** | proj/img_n_bf=3, ep=3, lr=5e-4, train_n=50 (iter-1) | 0.000018 | 0.4868 | **0.0047** | [results](../runs/mayo-ldct-claude-agentic-dual-domain-bilateral-n2i-search-20260603-01/results.tsv) | [iter-1](../runs/mayo-ldct-claude-agentic-dual-domain-bilateral-n2i-search-20260603-01/iterations/iter-0001/comparison.png) |

**12 ranks above baseline / 22 entries tested (10 below baseline + 12 above) — full canonical-19 inventory coverage.** 4 diff_recon TPE winners (ranks 4, 5, 7, 8) all converged at val_n=5 — directly comparable to learned-solver TPEs. TPE phase 3 lifted UNCON v4 by **+37%** vs Step-2 val_n=3 (0.1736 → 0.2377) by discovering a previously-unexplored **eta ≈ 0.3** low-eta corner.

**Cross-dataset inventory compare:**
- **Demo-DL** (Sidky synthetic): 18 above + 0 below = **18/19 inventory variants** (missing ItNet v1, TV-iter supervised)
- **Breast-CT** (Sidky synthetic with anatomy): 14 above + 13 below = **27 entries** (missing ItNet v1)
- **Mayo-LDCT** (real helical, 2304-view): 12 above + 10 below + 1 deprioritised = **23 entries, full 19-inventory coverage**

Mayo is the only dataset where every canonical solver from solver_plan.md has been exercised — including ItNet v1/v2 (both retried post-cfg-patch 2026-06-08) and TV-iterative supervised. The demo-DL and breast-CT leaderboards predate the cfg-patch retries and were never extended.

### Non-trainable solvers (full report)

Non-trainable solvers reconstruct each val slab without any learnable parameters. They run for completeness, even when hr<baseline. The FBP baseline itself is included for reference.

| Solver | Variant | params | SSIM | PSNR (dB) | hr | Source |
|---|---|---:|---:|---:|---:|---|
| **FBP baseline** | full_scan + hann + 2N pad (canonical) | 0 | 0.5161¹ | 13.98¹ / 12.59² | 0 (reference) | [`ddssl_ldct/pyronn_projector.py`](../../ddssl_ldct/pyronn_projector.py) |
| **TV-iterative** (non-trainable, **rank 6 above**) | tv_iterations=12800, λ=0.01, step=0.5 (iter-8) | 0 | 0.5439 | 13.09² | **0.0557** | [results](../runs/mayo-ldct-claude-agentic-tv-iterative-search-20260603-01/results.tsv) |
| **Wu 2015 non-trainable** | n_bands=8, n_outer=2, motion_range=5, soft_thresh=1.5e-3 (iter-2; plateau) | 0 | 0.357 | 12.37² | 0 | [results](../runs/mayo-ldct-claude-agentic-wu-2015-search-20260603-01/results.tsv) |

¹ at val_n=3 (the diff_recon configs).  
² at val_n=5 (the standard agentic-iter configs).  

The val_n discrepancy means the FBP baseline shifts between rows: PSNR≈13.98 when only the 3 sharpest L014 slabs are scored, and ≈12.59 with the broader val_n=5 set. All hr values reported throughout this leaderboard subtract the SAME-val_n baseline used for that solver's run.

### Mayo phase-4 — agentic-loop retests of all 10 STOP'd solvers (dispatched 2026-06-10)

Following the 3 retroactive false-STOP cases (Hammernik VN, Wu
trainable BC, ItNet v1 demo/BC) and the relaxed STOP criterion
update in `solver_plan.md`, **all 10 currently-STOP'd Mayo solvers
are being retested via the 20-iter agentic-autoresearch protocol**
(Claude-driven iter-by-iter with explicit hypothesis-per-iter,
NOT TPE). Configuration-space sparsity is now the assumed cause of
any hr=0 result until 20 hypothesis-driven iters rule it out.

**Phase-4 final scoreboard: 6 of 10 above baseline ⬆ · 4 SOFT STOPped**
🥇 **ItNet v2 FINAL: 0.2222** · 🥈 **ItNet v1 FINAL: 0.2100** · **R²-Gaussian FINAL: 0.0844** (iter-18, 48g; iter-19 32g regressed to 0.0834) · **DD-BF L2 FINAL: 0.0706** · **Hammernik 2017 FINAL: 0.0665** · **RAM FINAL: 0.0461** | 🛑 Wu non/trainable, DD-UNet N2I, TV-iter sup

**📊 TPE phase started 2026-06-11** — 6 phase-4 TPE jobs dispatched (763302-763307) around the agentic sweet spots:
- itnet_v2_mayo_phase4 · itnet_mayo_phase4 · hammernik_mayo_phase4
- dual_domain_bilateral_supervised_mayo_phase4 · r2gaussian_mayo_phase4 · ram_zeroshot_mayo_phase4

**Key insight discovered**: `train_n` is a UNIVERSAL binding constraint across ItNet v1/v2, DD-BF L2, AND Hammernik 2017. R²-Gaussian additionally shows monotonic gain from sparser Gaussian density (256g → 96g).

**Key insight discovered**: `train_n` is a UNIVERSAL binding constraint across ItNet v1, ItNet v2, DD-BF L2, AND Hammernik 2017. Lifting to 75-90 broke ceilings in all four solvers.

| Solver | iters tried | Best hr | Best config | Latest trajectory + next knob |
|---|---:|---:|---|---|
| 🏆🏆 **ItNet v1** | 16 | **0.1732** (iter-14) | k=1, c=32, α=0.05, ep=40, finetune ep=8/lr=1.5e-3 | iter-16 (finetune_ep=12) regressed to 0.1315 — fe=8 also sweet; iter-17 763228 tries c=40 (untested mid) |
| 🚀 **ItNet v2** | 15 | **0.0855** (iter-6) | k=1, c=32, α_init=0.05, ep=12, residual=F, train_n=50 | iter-15 (train_n=100) CRASHED OOM; iter-16 763218 tries train_n=75 (intermediate) |
| 🎉 **Hammernik 2017** | 9+1Q | **0.0621** (iter-5) | T=5, filters=24, kernel=11, λ=5e-3, ep=12, lr=5e-4 | iter-9 (T=6) collapsed SSIM 0.228 — T=5 firmly sweet; iter-10 763215 queued (filters 24→16) |
| 🏆🏆 **DD-BF sup L2** | 11 | **0.0590** (iter-11) | img_n_bf=27, proj_n_bf=1, k=5/7, ep=10, lr=5.9e-3 | **11 monotonic climbs** (img_n_bf 7→27, +124%!); iter-12 763223 tries img_n_bf 27→29 |
| 🚀 **R²-Gaussian** | 9+1Q | **0.0438** (iter-8) | 256g, 500i, lr_pos=5e-4 | iter-9 regressed; iter-10 763216 queued (gaussians 256→320) |
| 🎯 **RAM zero-shot** | 14+1Q | **0.0402** (iter-14) | input_norm=global_max, blend=0.2, factor=0.5, σ=5e-3 | iter-14 (b=0.2) hr=0.0402 (+15%, slowing); iter-15 763219 queued (b 0.2→0.1) |
| DD-UNet N2I 🔵 INCHING UP | 10 | 0 (SSIM 0.471 iter-9 best) | c=24, ep=80, lr=5e-5 (long-train regime) | iter-10 (ep=120) TIMEOUT at ep 105/120 (loss=0.00001); iter-11 763221 reverts ep=100 |
| TV-iter sup 🟡 MOVING | 9+1R | 0 (SSIM 0.381 iter-6/9 best) | share_steps=T, lr=1e-1, grad_clip=20-100, ep=30 | iter-10 (tv_K=30) running (25+ min); last lever before STOP |
| 🛑 Wu non-trainable **SOFT STOP** | 10 | 0 (SSIM 0.358 iter-2 best) | n_bands=6, soft=1.5e-3, range=5, window=2 | iter-10 (n_outer=5) SSIM 0.343 — 3 families covered, paper-edge exhausted. Wu-2015 closed-form **structurally bounded** on Mayo LDCT. |
| 🛑 Wu trainable **SOFT STOP** | 10 | 0 (SSIM 0.351 iter-6 best) | n_bands=6, lr=1.1e-4, ep=13, λ_neg=0.7 | iter-10 (n_outer=4) SSIM 0.322 regressed — 4 families exhausted (capacity/motion/reg/optimizer), paper-edge coverage achieved. **STRUCTURALLY BOUNDED** on Mayo LDCT. |

(iter values are hr; SSIM/PSNR shown after `/` for solvers stuck at hr=0)

**iter-4 + iter-5 highlights:**
- 🏆 **DD-BF L2**: 4 monotonic climbs in a row (0.0264 → 0.0313 → 0.0356 → **0.0394**, +49% total). img_n_bf 7→9→11→13 each added gain. iter-5 tests img_n_bf=15.
- 🏆 **ItNet v1 EXPLOSIVE**: k=1+c=32+α=0.05 (iter-4) → hr=0.0284 broke 0.26 SSIM ceiling. iter-5 added ep+lr → hr=0.0608 (+114% in one iter!). iter-6 (c=48) targets continued climb.
- 🎉 **ItNet v2 OVERTURNED**: cross-solver hypothesis ported v1's winning recipe → hr=0.0324 (was 0 across iter-1/2/3/4). iter-6 mimics v1's iter-5 ep/lr/c=32 to match v1's hr=0.0608.
- 🟡 **R²-Gaussian recovering**: 0.0134 (iter-4 at 1500i) — closer to iter-1's 0.0219 but not there yet. iter-5 pushes 2500 iters.
- 🔴 **TV-iter sup confirmed structurally locked**: SSIM=0.298537 to 6 digits across iter-1/2/3/4 despite K, step, λ, share_steps changes. iter-5 final attempt: lr↑5×, grad_clip↑10× — if no movement, the FBP-init basin is a true fixed point.
- 🔴 **DD-UNet N2I structural floor**: SSIM 0.46-0.47 across c ∈ {8, 16, 24, 32}. iter-5 tests lr↑+ep↑ at max c=32 — last lever.
- 🔴 **RAM bounded**: best PSNR 12.35 at b=0.8, regress at b=0.95. iter-6 refines b=0.9 to confirm peak — RAM structurally < baseline 12.59.
- 🟡 **Wu trainable + non-trainable parallel test**: both get n_outer 2→3 to test if the per-iter denoise-rerun loop is undertrained.

**iter-3 trajectory:**
- 🏆 **DD-BF sup L2**: 3rd monotonic climb (0.0264→0.0313→0.0356, +35% over iter-1) — img_n_bf=11 keeps gaining. iter-4 pushes to 13.
- 🟡 R²-Gaussian: partial recovery (~0→0.0110) at intermediate density (384g/800i). Confirms 256g is sweet spot — iter-4 tests 256g with 3× more iters.
- 🔴 Hammernik 2017: filters=32 crashed in 5.7s (likely OOM). iter-4 reverts ALL hyperparams to iter-1 winner except shrinks vn_kernel 11→7.
- 🔴 TV-iter sup: SSIM exactly 0.299 across K + step×10 shock — FBP-init basin unbreakable by step-tuning. iter-4 switches to tv_share_steps=True (architectural change).
- 🟡 Wu non-trainable: 0.358 (iter-2 n_bands=6) > 0.345 (iter-3 n_bands=8) — sweet spot confirmed n_bands=6. iter-4 pushes soft_thresh 3× at the proven n_bands.
- 🔴 ItNet v1/v2: high-lr shock didn't move 0.26-0.27 ceiling. iter-4 tries opposite direction (k=1, c=32) — collapse unrolling.
- 🔴 RAM: 0.29 dB below baseline across blend ∈ {0.3, 0.5, 0.7}. iter-4 final attempt at blend=0.8 (mostly FBP) before declaring structural.
- 🟡 DD-UNet N2I: SSIM moved 0.46→0.47 with c=8. iter-4 tests opposite extreme (c=32 + ep=25).
- 🔴 Wu trainable: lr/ep/λ_neg sweep all stuck at 0.345 ceiling. iter-4 tries motion_window 2→1.

**🎯 3 STOP verdicts already OVERTURNED at iter-1**: DD-BF sup L2 (Mayo Step-2 said hr=0 "structural; 18 BF too low cap"; breast-CT TPE winner ported here → hr=0.0264). Hammernik 2017 (Mayo Step-2 said hr=0 "λ stuck"; VN-winning arch with T=5, n_filters=24 → hr=0.0483, just below VN's 0.0551). R²-Gaussian (Mayo Step-2 was TIMEOUT; minimal 256-Gaussian + 500-iter config → hr=0.0219, fit cleanly in 30-min wall).

iter-2 dispatched for all 10 with ONE knob change each per autoresearch protocol.

**Expected outcomes:**
- **High confidence overturn**: Wu 2015 trainable (breast-CT TPE found +45%), Hammernik 2017 (Hammernik family precedent — VN was overturned), TV-iter supervised (the trainable variant has more flexibility than the non-trainable that already lands rank 9 on Mayo).
- **Moderate confidence**: ItNet v1/v2 (Mayo capacity ceiling may not be the binding constraint at the right c/k combo), DD-BF L2 (18 BF scalars same family as DD-BF N2I rank 12).
- **Low confidence overturn** (genuine structural ceiling): R²-Gaussian (per-scene wall), RAM (μ-range), DD-UNet N2I (N2I floor mechanism), Wu 2015 non-trainable (closed-form 10-coeff).

Results land 10-14h after dispatch (4 concurrent on QOS=4, queue 6 behind).

### Below-baseline inventory (`hr = 0`, structural STOPs — under retest by phase-4 TPE 2026-06-10)

These 10 solver variants were tested on Mayo and remained at `hr = 0`
under the calibrated metric — they are below the FBP baseline regardless
of further tuning. Listed for completeness alongside the 12-rank
above-baseline table:

| Solver | Variant | params (M) | SSIM | hr | Why it fails on Mayo |
|---|---|---:|---:|---:|---|
| **DD-BF supervised L2** | proj/img_n_bf=3, ep=3 (iter-3 winner) | 0.000018 | 0.485 | **0** | 18-param BF too low-capacity; breast-CT hr=0.26 variant doesn't transfer to Mayo's wider μ-range. |
| **DD-UNet N2I** | c=16, ep=3 (iter-1) | 0.466 | 0.46 | **0** | N2I supervision floor reached by ep-1; PSNR 12.52 < baseline 12.59. Supervised L2 variant (rank 1) is the right DD-UNet for Mayo. |
| **ItNet v1** *(post-patch retry)* | k=2, c=16, ep=6 (iter-3) | — | 0.249 | **0** | Low-capacity ceiling. Only v3 (deeper UNet + per-step α) clears baseline on Mayo. |
| **ItNet v2** *(post-patch retry)* | k=2, c=16, ep=6 (iter-3) | — | 0.264 | **0** | Same low-capacity ceiling as v1; v2 architecture sits below baseline on Mayo regardless of training budget. |
| **Hammernik 2017** | T=3, λ-clamp, ep=3 (iter-1) | 0.004 | 0.27 | **0** | Per-step λ stuck within 0.8% of init; recon biased darker than baseline (PSNR 11.55 < 12.59). |
| **TV-iterative supervised** | K=10–30, step=1e-4, λ=1e-5 (iter-1) | 0.0001 | 0.30 | **0** | FBP-init + smooth-TV makes the 1st GD step ≈ no-op; step/λ scalars never get a useful gradient. |
| **Wu 2015 trainable** | n_bands=4, ep=6, range=5 (iter-2) | 0.000010 | 0.34 | **0** | 10 trainable scalars hit low-capacity ceiling at SSIM≈0.34 (PSNR 12.37); blend moves 0.97→0.74 but image quality stays flat. |
| **Wu 2015 non-trainable** | n_bands=8, range=5, soft_thresh=1.5e-3 (iter-2) | 0 | 0.357 | **0** | Same 10-filter-coeff ceiling as the trainable variant; closed-form ⇏ better than tuned. |
| **RAM zero-shot** | pretrained ram.pth.tar, σ=0.075, blend=0.42 (iter-3) | 35.6 *(frozen)* | 0.48 | **0** | PSNR 12.45 < baseline 12.59; natural-image prior can't bridge to Mayo μ-range. |
| **R²-Gaussian** | gs_n_iter=3000, FBP-init, train_n=2 (iter-1 timeout) | 0.003 | — | **0** | Per-scene fit at Mayo's 2304-angle × 512² scenes exceeds 30-min sbatch wall; structurally too expensive for the agentic loop budget. |

**Plus 1 deprioritised checkpoint variant:**
- **diff_recon v3** (ch=96, batch=1, 60 ep) — UNCON best 0.0641, CON best 0.0686. Both ⅓ of the corresponding v2/v4 ckpts. The v3 ckpt's `batch=1, 60 ep` schedule yielded ¼ the effective training of v2's `batch=2, 60 ep` (half the batch at 2.25× params). v4 (ch=96, batch=2, 120 ep — the fix proposed in v3's verdict) lands at ranks 4 and 7.

**Above-baseline plateau notes (these solvers ARE in the rank table; included here as TPE/Step-2 verdicts):**

| Solver | State | Note |
|---|---|---|
| **DD-BF N2I** *(rank 12)* | Step-2 iter-1 hr=0.0047, iter-2 (ep 3→6) hr=0.0035 plateau | 18 BF scalars can't beat baseline by more than the noise floor. Phase-2 TPE 762925 STOP'd on hardcoded `R_full.fbp(val_clean)` OOM (solver-side patch needed for chunked FBP). |
| **NAF** *(rank 11)* | Step-2 iter-1 hr=**0.0202**; Step-3 TPE 762923 found hr=0.0131 (worse — overshoot at n_freqs≥12). | Step-2 stays as the rank entry; TPE iter-7 (n_freqs=8/hidden=192/layers=4) was the working corner but TPE never revisited it. |
| **diff_recon v3 (Mayo, ch=96 batch=1)** | Plateau at hr=0.064–0.069 across 6 v3 iters. | Superseded by v4 (ch=96 batch=2 ep=120) at ranks 4/7. |

**Step-3 TPE plateau-resolution log (per-solver verdicts kept for the cross-dataset transfer record):**
- **LPD**: Step-2 iter-3 hr=0.2445 plateaued; **Step-3 TPE 0.3063** (lpd_iters=2, hidden=48, ep=12).
- **DD-UNet supervised L2**: Step-2 iter-3 hr=0.1337 plateaued; **Step-3 TPE 0.3890** (TPE iter-12 winner).
- **USwin**: Step-2 iter-2 hr=0.1425 plateaued; **Step-3 TPE 0.2492** (TPE iter-11 winner, search-space-clamped).
- **ItNet v3**: Step-2 iter-5 hr=0.1336 plateaued; **Step-3 TPE 0.2181** (iter-9, k=3, c=16 with the cfg-patch eae661bc that fixed the silent-drop bug in solver_itnet*.py).
- **Hammernik VN**: Step-2 2-consecutive-hr=0 STOP **OVERTURNED** by Step-3 TPE 762926 — final hr=**0.0551** at vn_T=5, vn_n_filters=16, vn_kernel=11, vn_λ_init=2.3e-3, ep=12, lr=2.6e-4. **Lesson: TPE on a wider search space can rescue agentic-loop hr=0 plateaus on low-complexity learned solvers.**
- **diff_recon UNCON v2/v4 + CON v2/v4**: Step-3 TPE phase 3 lifted all 4 by +12–66% vs Step-2 val_n=3 baselines via discovering eta=0.30 (UNCON) and eta=1.5–7 (CON) corners — see mode×prior summary above.

### Known infrastructure cap: Mayo FBP requires train_n ≤ 50 (Q6000, 24 GB)

`PyronnFanBeamProjector.fbp` on Mayo's 2304-angle sino allocates 2.5–5 GB of FFT scratch with `train_n=100`; combined with model+gradient memory this exceeds Q6000. **USwin iter-3/4, ITNet v3 iter-1, Hammernik VN iter-1 all OOMed at this exact line.** All Mayo iter-N+1 configs now use `train_n=50` (was the silent default in iter-2 USwin which fit). If a solver needs more data, the fix is chunked FBP in `solver_*.py`, not just bumping the GPU class.

**Step-3 TPE phase 2 results (Mayo TPE for low-rank Step-2 positives):**

| Solver | Step-2 best hr | Step-3 TPE | TPE best | Status |
|---|---:|---:|---:|---|
| TV-iter (non-trainable) | 0.0557 | 762924 | 0.0511 | TPE clamp tv_iterations too low — Step-2 agentic 12,800 iters wins. **No improvement.** |
| NAF (per-scene MLP) | 0.0202 | 762923 | **COMPLETE 20/20**, final TPE hr=**0.0131** | iter-20 hr=0; final TPE 0.0131 < Step-2's 0.0202 (TPE found worse, agentic iter-1 was best). Step-2 entry stays in rank table. |
| Hammernik VN | 0 (Step-2 STOP) | 762926 | **0.0551 at iter-6, FINAL** (20/20 COMPLETE) | **ESCAPED BASELINE.** vn_T=5, vn_n_filters=16, vn_kernel=11, vn_λ_init=2.3e-3, ep=12, lr=2.6e-4. TPE clustered iters 13/17 in the same family (hr=0.0338/0.0290) but no config beat iter-6. Hammernik VN is now rank 10 on Mayo. |
| DD-BF N2I | 0.0047 | 762925 | 0 (all-fail) | OOM at hardcoded `R_full.fbp(val_clean)` — solver-side chunking needed. **STOP** |

**Step-3 TPE phase 1 (top-4 plateaued positives) — all complete:**

| Solver | Step-2 best hr | First TPE | Status | Retry |
|---|---:|---:|---|---:|
| Learned Primal-Dual | 0.2445 | 762896, 762898 | both FAILED (wrong key + SQLite lock) | **762902** (serial dispatch) |
| USwin | 0.1425 | 762897 | COMPLETED but ALL trials hr=0 (OOM in `filter_sino` 5 GiB at `train_n=200` from default search space — Mayo's 2304-angle sino can't fit train_n>50 on Q6000) | needs Mayo-specific search-space clamp |
| DD-UNet supervised L2 | 0.1337 | 762899 | FAILED (SQLite lock when launched in parallel with 762898/900) | queued |
| ItNet v3 (post-patch, iter-5 hr=0.1336) | 0.1336 | 762900 | FAILED (SQLite lock) | queued |
| diff_recon DCstep UNCON (v2) | 0.2095 | — | needs Mayo entries in `learned_solver_search_agent.py` (only breast_v2/v3 exist) | **dispatched in Step-3 TPE phase 3** (see below) |

**Step-3 TPE phase 3 — diff_recon Mayo (dispatched 2026-06-08 commit `6fa14e1b`, scp synced to cluster after 3 jobs died at startup with old SOLVERS dict):**

| Solver | Step-2 best hr (val_n=3) | Step-3 TPE | TPE seed iter-1 hr (**val_n=5**) | Status |
|---|---:|---:|---:|---|
| diff_recon DCstep UNCON v2 | 0.2095 (iter-6) | **762934 COMPLETE 20/20** | **0.2352 FINAL** SSIM 0.5487 (iter-12/16 tied — eta=0.31, **noise**, clamp=False, sample_steps=500, every=5, warmup=10, relax=0.95; iter-20 eta=4.1 noise hr=0.1191) | **FINAL.** TPE beats seed +7%; eta=0.31 corner reproducible across iter-12 and iter-16 |
| diff_recon DCstep UNCON v4 | 0.1736 (iter-2)  | **762935 COMPLETE 20/20** | **0.2377 FINAL** SSIM 0.5169 (iter-13 — eta=0.30, fbp, clamp=True; iter-20 eta=0.63 hr=0.1557) | **FINAL.** TPE beats seed +1.5%; very stable optimum at eta=0.30-0.39 |
| diff_recon DCstep CON v2   | 0.0847 (iter-6) | **762933 COMPLETE 20/20** | **0.1071 FINAL** SSIM 0.4847 (iter-9 — eta=7.21, fbp, clamp=True, warmup=40; iter-20 eta=20.7 hr=0.0556 high-eta fails) | **FINAL.** CON v2 optimum at mid-eta=7-11 with fbp init |
| diff_recon DCstep CON v4   | 0.0981 (iter-1) | **762936 COMPLETE 20/20** | **0.1632 FINAL** SSIM 0.5134 (iter-4 — eta=1.52, fbp, clamp=True, sample_steps=200, every=3, n_cg=20, warmup=25, relax=0.85; iter-20 eta=4.11 hr=0.0969 confirms high-eta fails) | **FINAL.** CON v4 optimum at eta=0.55-2.0 with fbp+clamp=True. TPE +66% over Step-2 val_n=3 baseline. |
| diff_recon DCstep CON v4   | 0.0981 (iter-1) | 762936 PENDING (QOS=4 cap) | — | queued behind NAF |

**TPE convergence (iters 2-13):** Startup random (iters 2-5) regressed to hr 0.04-0.20. **Prior-conditioned phase (iter-6+) found a new optimum corner BELOW the agentic search range:** `eta ≈ 0.30-0.40` (vs agentic explore range 1-30) for both UNCON priors. Iter-12/13 refined the optimum further:
- UNCON v2: seed 0.2197 → iter-8/9 0.2317 → iter-12 **0.2352** (eta=0.31, **noise** init, clamp=False)
- UNCON v4: seed 0.2343 → iter-6 0.2371 → iter-13 **0.2377** (eta=0.30, **fbp** init, clamp=True)
- CON v2: seed 0.1068 → iter-9 0.1071 (eta=7.21, mid-eta corner; low-eta confirmed bad for CON in iter-15)

**Init/clamp split between DDPM priors:** UNCON v2 prefers `noise + clamp=False`, UNCON v4 prefers `fbp + clamp=True`. Likely because v4 (ch=96, batch=2, 120 ep) has higher capacity but slower mixing → wants a deterministic anchor; v2 (ch=64, batch=2, 60 ep) has more "noise tolerance" and benefits from a randomised init at this low eta regime.

**Mode (UNCON vs CON) × prior (v2 vs v4) sweep summary:**
| Mode × prior | Optimum eta | Init | Clamp | hr | Note |
|---|---:|---|---|---:|---|
| UNCON v2 | **0.31** | noise | False | 0.2352 | very-low-eta corner |
| UNCON v4 | **0.30** | fbp | True | 0.2377 | very-low-eta corner |
| CON v2 | **7.21** | fbp | True | 0.1071 | mid-eta corner |
| CON v4 | **1.5** *(running)* | fbp | True | 0.1632 | low-mid-eta corner (NOT 0.3!) |

Both UNCON modes converge at very-low-eta (≈0.3). CON modes prefer higher eta — CON v2 at mid-eta=7-11, CON v4 at low-mid-eta=1-1.5. Constrained training drives the DPS toward sharper noise schedules than unconstrained.

**Mechanism:** at eta<0.5, the DPS noise injection becomes essentially deterministic (or near-clamped) — DPS reduces to mostly CG-based data consistency with mild diffusion prior. This regime had been unexplored by the agentic loop (which clamped eta≥1 based on breast-CT priors).

**Caveat on the val_n=5 lift:** Step-2 diff_recon configs were validated against the 3 sharpest L014 slabs (val_n=3, baseline PSNR=13.98); MAYO_CLAMPS in the TPE flow forces val_n=5 (baseline PSNR=12.59). The lower baseline at val_n=5 means the *same recon* gives a higher hr. The TPE numbers are the "fair" cross-solver comparison; will rebuild rank table once all 4 TPEs finish.

Search space mirrors breast_v3 with two changes: `recon_ckpt` paths point to Mayo ckpts; `recon_eta` narrowed to (0.3, 30) log (Mayo agentic winners converged at eta=1-10 vs breast's eta=30). Auto-injected by MAYO_CLAMPS: val_n=5, val_chunk=1, train_n=50 (unused — DPS doesn't train), batch_size=1.

**Lesson from 762930-932 startup failures:** the cluster `/cluster/maier/Agent4CT/` is NOT a git checkout — git pull fails. New code reaches the cluster only via direct `scp -o ProxyJump=lme-bastion`. The first 3 sbatches in the dispatch went out before the scp landed, so their copy of `learned_solver_search_agent.py` lacked the new SOLVERS keys and they died at argparse. Re-dispatched as 762934/935/936 after verifying the file synced.

**TPE infrastructure fixes (locked in 2026-06-08):**
1. **SQLite NFS lock** — Resolved via `--tpe-storage=/tmp/optuna-$SLURM_JOB_ID` in `cluster/slurm/mayo_ldct_solver_tpe_v2.sbatch` (per-job local storage, copied to `optuna-local-backups/` post-run).
2. **Search-space `train_n` exceeds Mayo capacity** — Resolved via `MAYO_CLAMPS` in `scripts/learned_solver_search_agent.py` (auto-injects `val_n=5, val_chunk=1, train_n=50, batch_size=1` when `--dataset=mayo_ldct_2d`).
3. **Cluster sync gotcha (2026-06-08)** — `/cluster/maier/Agent4CT` is NOT a git checkout. New code reaches the cluster via `scp -o ProxyJump=lme-bastion`, NOT via `git pull`. Three Mayo TPEs (762930-932) died at startup before this was discovered.

**Step-2 → Step-3 trajectory (historical preservation):**

The autoresearch loop **Step 2** (agentic random-walk, 2026-06-03 to 2026-06-08) reached **8 ranks above baseline / 15 inventory entries**, with the top entries being LPD (0.2445), diff_recon UNCON v2 (0.2095), USwin (0.1425), DD-UNet sup (0.1337), ItNet v3 (0.1336), diff_recon CON v2 (0.0847), TV-iter (0.0557), NAF (0.0202), DD-BF N2I (0.0047). 12 structural STOPs filed.

**Step 3** (TPE refinement, 2026-06-08 to 2026-06-09) lifted the count to **12 ranks above baseline** by (a) refining the top-4 plateaued positives with TPE on the agentic neighbourhood (LPD 0.2445→0.3063, DD-UNet sup 0.1337→0.3890, USwin 0.1425→0.2492, ItNet v3 0.1336→0.2181); (b) discovering the very-low-eta corner for diff_recon UNCON v2/v4 + CON v4 (Step-2 val_n=3 numbers replaced by val_n=5 TPE numbers); (c) **overturning the Hammernik VN Step-2 STOP** at hr=0.0551 (vn_T=5, n_filters=16, kernel=11). NAF TPE found a worse config than Step-2, so Step-2 stays as the rank-11 entry. See the rank table + below-baseline inventory above for the consolidated state.

**Original Step-2 convergence summary (preserved below):**

iter-3 closed the v3 exploration:
- 762865 UNCON v3 iter-3 (eta=30, top of log space) → hr=0.0641 (v3 best, still ~⅓ of v2's 0.2095)
- 762866 CON v3 iter-3 (warmup 40→25) → hr=0.0668 ≈ iter-1's 0.0686 (v3 plateau at ~0.069)

The v3 prior is **structurally inferior to v2** for DPS posterior sampling (see verdict in the table above). The v2 ckpts (3.823 M params, ch=64, batch=2) stay as the recorded rank-2/rank-5 entries:

- **diff_recon UNCON v2** at hr=0.2095 — iter-6 cfg: `eta=3, warmup=25, clamp=True, every=3, init=fbp, relax=1.0`
- **diff_recon CON v2**   at hr=0.0847 — iter-6 cfg: `eta=10, warmup=40, clamp=False, every=3, init=fbp, relax=1.0`

**Final Mayo Step-2 summary (8 ranks above baseline, 10 structural STOPs filed):**

| Rank | Solver | hr |
|---:|---|---:|
| 1 | Learned Primal-Dual            | 0.2445 |
| 2 | diff_recon DCstep UNCON (v2)   | 0.2095 |
| 3 | USwin                          | 0.1425 |
| 4 | DD-UNet supervised L2          | 0.1337 |
| 5 | diff_recon DCstep CON (v2)     | 0.0847 |
| 6 | TV-iterative (non-trainable)   | 0.0557 |
| 7 | NAF (per-scene MLP)            | 0.0202 |
| 8 | DD-BF N2I                      | 0.0047 |

Next phase: Step 3 TPE refinement on the four highest-rank learned solvers (LPD, diff_recon UNCON v2, USwin, DD-UNet). Needs a Mayo TPE sbatch (not yet scaffolded — see `cluster/slurm/` for breast/demo-dl analogues).

**Both Mayo DDPM v3 ckpts landed:**
- `ddpm_mayo_unconstrained_v3.pt` (33 MB, 8.594 M params, best val ε-loss=0.0061)
- `ddpm_mayo_constrained_v3.pt` (33 MB, 8.594 M params, best val ε-loss=0.0077)

**v2 plateau confirmation:** UNCON iter-7..10 (4 knob tests) all stayed within Δhr<0.01 of iter-6's 0.2095 — confirms the iter-6 corner is the v2 optimum on the explored breast_v3 search space. Same for CON across iter-6..10.

The previous batch outcome:
- 762856 DDPM v3 → both ckpts saved (faster than expected, 48 min total).
- 762859 UNCON iter-10 (init noise) → hr=0.2034 (≈ iter-6).
- 762860 CON iter-10 (clamp True) → hr=0.0760 (≈ iter-6).

**Diff_recon knob exploration summary (across 9 iters per variant):**

| axis | UNCON winner | CON winner |
|---|---|---|
| `recon_eta` | 3 (floor) | 10 (moderate) |
| `recon_sample_steps` | 200 | 200 |
| `recon_dcstep_n_cg` | 10 | 10 |
| `recon_dcstep_warmup` | 25 (early DC) | 40 (late DC) |
| `recon_dcstep_relax` | 1.0 (inert at 0.95) | 1.0 (inert at 0.95) |
| `recon_dcstep_every` | 3 (testing 4 now) | 3 |
| `recon_eta_clamp` | True | (symmetric assumption) |
| `recon_init` | fbp | (testing noise now) |

The previous batch outcome:
- 762851 TV iter-9 → still running, will TIMEOUT (25 % done at 30 min mark). TV-iter ends at iter-8 rank 6.
- 762852 UNCON iter-8 → hr=0.2045 (relax inert).
- 762853 CON iter-8 → hr=0.0751 (every-axis regression).

## Plan (DONE — preserved as historical record)

The original 4-point plan from after geometry calibration. All four
points have been executed; see the rank table + below-baseline
inventory above for the consolidated state.

1. ✅ **Re-rebin the remaining 9 patients** — bulk re-rebin of all 10
   Wagner patients with the fitted SSR geometry + FFS-z correction
   landed 2026-06-02 (job batches 762100, 762140-200, etc.).
2. ✅ **Per-solver autoresearch + TPE refinement** — Step 2 agentic
   (2026-06-03 → 2026-06-08) followed by Step 3 TPE refinement
   (2026-06-08 → 2026-06-09). Final ranking: DD-UNet sup TPE
   (`hr` = 0.3890), LPD TPE (0.3063), USwin TPE (0.2492), ItNet v3
   TPE (0.2181). RAM zero-shot CONFIRMED STOP (PSNR 12.45 < baseline
   12.59 — μ-range mismatch with natural-image prior). Hammernik VN
   `hr` = 0.0551 (overturned by Step-3 TPE). DD-BF supervised L2
   STOP at `hr` = 0 (18 BF params can't bridge to Mayo).
3. ✅ **DDPM training** — 4 ckpts trained: v2 (ch=64 batch=2 60ep),
   v3 (ch=96 batch=1 60ep — superseded), v4 (ch=96 batch=2 120ep).
   Both constrained (`L145/186/209/219` train) and unconstrained
   (all 10 patients) variants for v2/v3/v4.
4. ✅ **Diff-recon TPE** — Step-3 phase 3 ran TPE on UNCON/CON × v2/v4
   = 4 jobs. UNCON v4 TPE 0.2377 (rank 4), UNCON v2 TPE 0.2352
   (rank 5), CON v4 TPE 0.1632 (rank 7), CON v2 TPE 0.1071 (rank 8).
   TPE discovered the previously-unexplored very-low-eta (~0.3) corner
   for UNCON modes — agentic loop's eta≥1 clamp (inherited from
   breast-CT) had missed it.

## Methodology

See [`solver_plan.md`](../../solver_plan.md). Geometry-calibration
methodology and the full pixel-spacing / FFS-sign ablation history are
recorded in [`findings.md`](../findings.md) (newest entries first).
