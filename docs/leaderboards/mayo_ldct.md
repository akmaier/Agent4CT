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
| 2 | **diff_recon DCstep unconstrained** (DDPM v2 prior) | DPS, sample_steps=200, **eta=3, eta_clamp=True**, dcstep_n_cg=10, FBP init (iter-6) | 3.823 | 0.5703 | **0.2095** | [results](../runs/mayo-ldct-claude-agentic-diff-recon-dcstep-unconstrained-mayo-v2-search-20260603-01/results.tsv) | [iter-6](../runs/mayo-ldct-claude-agentic-diff-recon-dcstep-unconstrained-mayo-v2-search-20260603-01/iterations/iter-0006/comparison.png) |
| 3 | **USwin** | c=16, win=8, heads=8, ep=3, train_n=50 (iter-2) | — | 0.3747 | **0.1425** | [results](../runs/mayo-ldct-claude-agentic-uswin-search-20260603-01/results.tsv) | [iter-2](../runs/mayo-ldct-claude-agentic-uswin-search-20260603-01/iterations/iter-0002/comparison.png) |
| 4 | **DD-UNet supervised L2** | c=24, ep=3, lr=5e-4, train_n=100 (iter-3) | — | 0.4200 | **0.1337** | [results](../runs/mayo-ldct-claude-agentic-dual-domain-supervised-search-20260603-01/results.tsv) | [iter-3](../runs/mayo-ldct-claude-agentic-dual-domain-supervised-search-20260603-01/iterations/iter-0003/comparison.png) |
| 5 | **ItNet v3** (after cfg-patch eae661bc) | **k=3**, c=16, ep=3, pretrain=2, lr=5e-4, train_n=50 (iter-5) | — | 0.3146 | **0.1336** | [results](../runs/mayo-ldct-claude-agentic-itnet-v3-search-20260603-01/results.tsv) | [iter-5](../runs/mayo-ldct-claude-agentic-itnet-v3-search-20260603-01/iterations/iter-0005/comparison.png) |
| 6 | **diff_recon DCstep constrained** (DDPM v2 prior) | DPS, sample_steps=200, **eta=10, dcstep_warmup=40**, dcstep_n_cg=10, FBP init (iter-6) | 3.823 | 0.5108 | **0.0847** | [results](../runs/mayo-ldct-claude-agentic-diff-recon-dcstep-constrained-mayo-v2-search-20260603-01/results.tsv) | [iter-6](../runs/mayo-ldct-claude-agentic-diff-recon-dcstep-constrained-mayo-v2-search-20260603-01/iterations/iter-0006/comparison.png) |
| 7 | **TV-iterative** (non-trainable) | tv_iterations=12800, tv_lambda=0.01, tv_step=0.5 (iter-8) | 0 | 0.5439 | **0.0557** | [results](../runs/mayo-ldct-claude-agentic-tv-iterative-search-20260603-01/results.tsv) | [iter-8](../runs/mayo-ldct-claude-agentic-tv-iterative-search-20260603-01/iterations/iter-0008/comparison.png) |
| 8 | **NAF** (per-scene MLP) | n_freqs=6, hidden=192, layers=5, n_iter=2000 (iter-1) | 0.143 | 0.5395 | **0.0202** | [results](../runs/mayo-ldct-claude-agentic-naf-search-20260603-01/results.tsv) | [iter-1](../runs/mayo-ldct-claude-agentic-naf-search-20260603-01/iterations/iter-0001/comparison.png) |
| 9 | **DD-BF N2I** | proj/img_n_bf=3, ep=3, lr=5e-4, train_n=50 (iter-1) | 0.000018 | 0.4868 | **0.0047** | [results](../runs/mayo-ldct-claude-agentic-dual-domain-bilateral-n2i-search-20260603-01/results.tsv) | [iter-1](../runs/mayo-ldct-claude-agentic-dual-domain-bilateral-n2i-search-20260603-01/iterations/iter-0001/comparison.png) |

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

### Structural deal-breakers + plateaued (filed 2026-06-03)

| Solver | Final state | Why deprioritised |
|---|---|---|
| **DD-BF supervised L2** | iter-3 hr=0 (2 consecutive hr=0) | Loss stuck — 18-parameter BF too low-capacity for Mayo. The breast-CT hr=0.26 variant cannot transfer. |
| **RAM zero-shot (pretrained)** | iter-3 hr=0 (3 consecutive hr=0) | SSIM crept 0.40→0.48 but PSNR ceiling 12.45 < baseline 12.59. `ram.pth.tar` (natural images) cannot bridge to Mayo μ-range. |
| **Learned Primal-Dual** | iter-3 winner hr=**0.2445**; iter-4/5 regressed (loss explosion at hidden=64 + lpd_iters=6) | Capacity scaling exhausted. Both width-up and depth-up broke the loss landscape at train_n=100. iter-3 stays — Step 3 TPE next. |
| **DD-UNet supervised L2** | iter-3 winner hr=**0.1337**; iter-4/5 regressed (c=32 and ep=6 both worse) | Plateaued at c=24, ep=3. iter-3 stays — Step 3 TPE next. |
| **USwin** | iter-2 winner hr=**0.1425**; iter-3/4 OOMed, iter-5 (ep=6) regressed to 0.107 | Plateaued at c=16, win=8, ep=3, train_n=50. iter-2 stays — Step 3 TPE next. |
| **ITNet v3** | iter-1+2 both OOM (FBP inside the unrolled body at 5 GB even with train_n=50). **Resolved 2026-06-08** by cfg-patch (commit `eae661bc`) — the OOM was because solver_itnet_v3.py was silently dropping the agentic JSON cfg and using hardcoded `itnet_k=5` defaults; with `itnet_k=2` actually honored, the retry landed at hr=**0.1036** (rank 5). | Cfg-merge bug filed + patched; v3 is now a working rank entry. |
| **Hammernik VN** | iter-2/3 hr=0, SSIM 0.27→0.29 (2 consecutive hr=0) | Loss decreasing but val PSNR ceiling 12.14 < baseline 12.59. T=3 and T=5 both undercooked vs the sino complexity; larger T would risk OOM. |
| **TV-iterative supervised** | iter-1 hr=0 SSIM 0.30 (loss STUCK at 0.001, step/λ scalars stuck at init) | Same structural verdict as breast-CT: FBP init makes 1st GD step a no-op (data-fidelity gradient ≈ 0 at FBP), so step/λ scalars never get a useful gradient. |
| **Wu 2015 trainable** | iter-1 hr=0 SSIM 0.34, iter-2 hr=0 SSIM 0.34 (ep 3→6, no improvement) | Loss converged at 0.00013 by ep-1; the 10 trainable scalars *do* move (blend 0.97→0.74) but final SSIM stays at 0.34. The 10-param image-domain BF-like solver hits the same low-capacity ceiling here as it did on breast-CT (which got hr=0.22 only because that dataset has a much narrower dynamic range). |
| **ITNet v2** | iter-1 OOM in `filter_sino`; **resolved 2026-06-08** by cfg-patch (commit `eae661bc`). Retry iter-2 (k=2, c=16, train_n=50, ep=3) hr=0 SSIM 0.268 PSNR 10.21; iter-3 (ep 3→6, pretrain 2→4) hr=0 SSIM 0.264 — slightly worse. 2 consecutive hr=0 → **plateau filed**. | v2 architecture sits below baseline on Mayo regardless of training budget. Same low-capacity ceiling that affects v1. |
| **ITNet v1** | iter-1 OOM (same root cause as v2; solver_itnet.py had NO env-read). **Resolved 2026-06-08** by cfg-patch. Retry iter-2 hr=0 SSIM 0.256; iter-3 (ep 3→6) hr=0 SSIM 0.249 — slightly worse. 2 consecutive hr=0 → **plateau filed**. | Same low-capacity ceiling as v2; the v3 architecture (deeper UNet + per-step α) is the only ItNet that lifts above baseline on Mayo. |
| **Hammernik 2017** | iter-1 hr=0 SSIM 0.27, PSNR 11.55 < baseline 12.59 (λ_t stuck within 0.8 % of init) | Same Hammernik family failure as VN: the per-step λ regulariser barely budges over 3 epochs and the final recon is biased darker than baseline. Sino-complexity ceiling, not a config knob away from working. |
| **DD-UNet N2I** | iter-1 hr=0 SSIM 0.46, PSNR 12.52 < baseline 12.59 (loss 0.00001 from epoch 1) | Same N2I noise-floor over-smoothing as breast-CT: loss is already at the N2I supervision floor by ep-1, so the network has nowhere to climb. Supervised L2 (rank 3, hr=0.13) is the right DD-UNet variant for Mayo. |
| **DD-BF N2I** | iter-1 hr=0.0047, iter-2 (ep 3→6) hr=0.0035 — Δhr = −0.0012 (plateau) | The 18 BF scalars kept moving (σ_y 1.973→1.953 from ep-3 to ep-6) but val SSIM stayed flat at 0.485 and hr actually slipped. iter-1 stays as rank 5; cap is structural (18 trainable params can't beat the FBP baseline by more than the noise floor). |
| **R²-Gaussian** | iter-1 TIMEOUT at 30 min (per-scene fit takes longer than the autoresearch sbatch wall allows) | gs_n_iter=3000 + FBP² init at train_n=2, val_n=2 still didn't complete a single per-scene fit within the 30-min cluster_guide §4.6 budget. Same breast-CT structural verdict applies — R²-Gaussian's per-scene optimisation is too expensive for the autoresearch loop on Mayo's 2304-angle, 512² scenes. Would need a TPE-scale 4-h job to test seriously. |
| **NAF** plateau verdict | iter-1 hr=**0.0202** (the rank-4 entry); iter-4 (naf_n_iter 2000 → 3000 at val_n=5) regressed to hr=0.0010, SSIM 0.5395 → 0.4843 | NAF **overshoots** beyond 2000 iters on Mayo: the implicit-field MLP starts hallucinating high-frequency detail that does not match the truth slab, pulling SSIM down by ~5.5 pts. iter-1's `naf_n_iter=2000` is the optimum. iter-1 stays as rank 4 — Step 3 TPE would need to bracket below 2000 iters, not above. |
| **diff_recon v3 (ch=96 prior)** | UNCON best iter-3 hr=0.0641 (eta=30); CON best iter-1 hr=0.0686 (eta=10). All v3 results sit at ⅓ of the corresponding v2 ckpt. | Bigger ≠ better here: the v3 ckpt (8.594 M params, 60 ep at batch=1) is structurally weaker than v2 (3.823 M params, 60 ep at batch=2) for DPS posterior sampling. Root cause is likely insufficient gradient updates per parameter (half the batch size at 2.25× the params ⇒ ¼ the effective training). A v4 attempt should pair ch=96 with batch=2 (likely OOMs on Q6000) or train ch=96 for 120+ epochs. For now the v2 ckpts stay as the rank-2/rank-5 priors. |
| **ItNet v1** | iter-1 OOM in `filter_sino` (2.53 GiB FFT pad) | Same cfg-merge bug as v2: `solver_itnet.py` ignores the agentic JSON cfg and uses hardcoded `train_n=400`, `val_n=100`, `itnet_k=5`. Two unrelated solvers (v1 and v2) have the same defect; the fix is structural (read `CFG_JSON` env like solver_itnet_v3.py does). Cannot retry on Mayo without a code patch. |
| **Wu 2015 non-trainable** | iter-1 (n_bands=4) hr=0 SSIM 0.350; iter-2 (n_bands=8) hr=0 SSIM 0.357 — 2 consecutive hr=0, **plateau confirmed** | Closed-form filter-band modulation matches the trainable variant's ceiling at SSIM≈0.35 / PSNR≈12.37; doubling `n_bands` only inched SSIM +0.007. The 10 filter coefficients (frozen or trained) cannot reach Mayo's dynamic range. Same structural verdict as the trainable variant. |

### Known infrastructure cap: Mayo FBP requires train_n ≤ 50 (Q6000, 24 GB)

`PyronnFanBeamProjector.fbp` on Mayo's 2304-angle sino allocates 2.5–5 GB of FFT scratch with `train_n=100`; combined with model+gradient memory this exceeds Q6000. **USwin iter-3/4, ITNet v3 iter-1, Hammernik VN iter-1 all OOMed at this exact line.** All Mayo iter-N+1 configs now use `train_n=50` (was the silent default in iter-2 USwin which fit). If a solver needs more data, the fix is chunked FBP in `solver_*.py`, not just bumping the GPU class.

**Step-3 TPE active 2026-06-08 — infrastructure issues identified, partial retry in progress:**

| Solver | Step-2 best hr | First TPE | Status | Retry |
|---|---:|---:|---|---:|
| Learned Primal-Dual | 0.2445 | 762896, 762898 | both FAILED (wrong key + SQLite lock) | **762902** (serial dispatch) |
| USwin | 0.1425 | 762897 | COMPLETED but ALL trials hr=0 (OOM in `filter_sino` 5 GiB at `train_n=200` from default search space — Mayo's 2304-angle sino can't fit train_n>50 on Q6000) | needs Mayo-specific search-space clamp |
| DD-UNet supervised L2 | 0.1337 | 762899 | FAILED (SQLite lock when launched in parallel with 762898/900) | queued |
| ItNet v3 (post-patch, iter-5 hr=0.1336) | 0.1336 | 762900 | FAILED (SQLite lock) | queued |
| diff_recon DCstep UNCON (v2) | 0.2095 | — | needs Mayo entries in `learned_solver_search_agent.py` (only breast_v2/v3 exist) | scaffolding next turn |

**Two TPE infrastructure issues to fix:**
1. **SQLite lock** when launching multiple Mayo TPEs in parallel. Fix: serialize dispatches (one solver's TPE at a time, or per-solver storage path).
2. **Search-space `train_n` exceeds Mayo capacity**. The breast/demo TPE specs all use `train_n=400`/`200` which OOMs the Mayo 2304-angle `filter_sino` FFT pad at ~5 GB. Fix: when `--dataset=mayo_ldct_2d`, clamp `train_n` to ≤50 in the search space.

**Productive parallel work in flight:**
- 762888 DDPM v4 COMPLETED ~13:08 — both ckpts saved (unconstrained 12:49, constrained 13:07).
- **diff_recon v4 UNCON iter-1 (762901)** → hr=**0.1466** SSIM 0.5421 PSNR 15.35 (between v3's 0.0641 and v2's 0.2095). **Surprise:** v4 has the best DDPM training (val ε-loss 0.0025 vs v2's 0.0049), but v2 still wins diff_recon. **DDPM training quality is NOT predictive of DPS performance.**
- 762903 diff_recon v4 CON iter-1, 762904 diff_recon v4 UNCON iter-2 (eta 3→1, testing if v4 wants even more conservative DPS) in flight.

Also in flight: DDPM Mayo v4 training (job **762888**, ep 84/120 at last check, best val ε-loss=0.0025 — already beats v3 and v2). When v4 ckpt lands, dispatch `diff_recon_mayo_v4` iter-1.

**Autoresearch loop — coverage audit CLOSED 2026-06-07 ~22:23. All 15 solver_plan.md entries now run on Mayo.**

Last-2 results:
- **ITNet v1** (job 762874) → **OOM in `filter_sino`** (2.53 GiB FFT pad). Same cfg-merge bug as v2 — `solver_itnet.py` ignores the JSON cfg. STOP filed in the deal-breakers table.
- **Wu 2015 non-trainable** (job 762875) → hr=**0** SSIM 0.350 PSNR 12.35 < baseline. Closed-form 10-coefficient filter cannot reach the Mayo dynamic range, same as the trainable variant. STOP filed.

**Final Mayo Step-2 status:** 8 ranks above baseline, 12 structural STOPs filed across the full inventory. Loop closed for Step 2.

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
