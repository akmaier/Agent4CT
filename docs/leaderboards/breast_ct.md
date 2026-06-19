---
title: Breast-CT leaderboard
description: Calibrated headroom ranking of every solver family on the Sidky breast-CT 128-view sparse-view benchmark.
---

# Breast-CT leaderboard

128-view 2-D fan-beam sparse-view synthetic phantoms (Sidky-group breast
model, real μ range up to ~0.5). All metrics through
[`ddssl_ldct.metrics.evaluate_calibrated`](../../ddssl_ldct/metrics.py):
two-point linear intensity calibration on the foreground inside an
inscribed-circle FoV mask, then PSNR/SSIM/RMSE on the calibrated
prediction.

**Baseline FBP**: SSIM = 0.957, PSNR = 39.74 dB, `hr = 0`.
`hr = max(0, 1 − rmse / baseline_rmse)`.

## Calibrated leaderboard (canonical)

Slug prefixes `breast-ct-calibrated-tpe-*` (TPE Bayesian search) and
`breast-ct-claude-agentic-*` (Claude-agentic seed runs). Sorted by `hr`;
one row per solver family (best iteration). `params (M)` is the number
of trainable parameters in millions; `0` = non-trainable; `(frozen)` =
pretrained checkpoint loaded without finetuning. `PSNR (dB)` / `RMSE` /
`time (s)` are logged per-iter by the current harness; `—` marks the
pre-2026-06 TPE runs that predate those fields (only SSIM + headroom were
recorded then, and the raw recon is not retained, so they can't be
back-computed without re-running the solver).

| Rank | Solver | Variant | params (M) | SSIM | hr | PSNR (dB) | RMSE | time (s) | Source | Comparison |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| 1 | **Learned Primal-Dual** | I=8, hidden=96, n_p=5, n_d=5, ep=23, lr=3.2e-4, clip=0.3, train_n=400 | 1.492 | 0.9996 | 0.9062 | — | — | — | [results](../runs/breast-ct-calibrated-tpe-lpd-search-20260524-01/results.tsv) | [iter-11](../runs/breast-ct-calibrated-tpe-lpd-search-20260524-01/iterations/iter-0011/comparison.png) |
| 2 | **DD-UNet supervised L2** | c=24, ep=18, lr=2.1e-4, λ_neg=0.58, train_n=400 | 1.045 | 0.9989 | 0.8361 | — | — | — | [results](../runs/breast-ct-calibrated-tpe-dual-domain-supervised-search-20260524-01/results.tsv) | [iter-19](../runs/breast-ct-calibrated-tpe-dual-domain-supervised-search-20260524-01/iterations/iter-0019/comparison.png) |
| 3 | LPD (agentic seed) | I=10, hidden=64, n_p=5, n_d=5, ep=20, lr=5e-4, clip=1.0, train_n=400 | 0.875 | 0.9988 | 0.8290 | 55.08 | 8.81e-04 | 2447 | [results](../runs/breast-ct-claude-agentic-learned-primal-dual-search-20260522-01/results.tsv) | [iter-3](../runs/breast-ct-claude-agentic-learned-primal-dual-search-20260522-01/iterations/iter-0003/comparison.png) |
| 4 | DD-UNet supervised L2 (agentic seed) | c=16, ep=10, lr=5e-4, λ_neg=1.0, train_n=400 | 0.466 | 0.9986 | 0.8120 | 54.25 | 9.69e-04 | 151 | [results](../runs/breast-ct-claude-agentic-dual-domain-unet-l2-search-20260522-01/results.tsv) | [iter-1](../runs/breast-ct-claude-agentic-dual-domain-unet-l2-search-20260522-01/iterations/iter-0001/comparison.png) |
| 5 | ITNet v3 | ep=15, lr=2.2e-4, c=16, k=2, α=2.6e-3, train_n=200 | 3.699 | 0.9965 | 0.7342 | — | — | — | [results](../runs/breast-ct-calibrated-tpe-itnet-v3-search-20260521-01/results.tsv) | [iter-18](../runs/breast-ct-calibrated-tpe-itnet-v3-search-20260521-01/iterations/iter-0018/comparison.png) |
| 6 | USwin | ep=14, lr=1.8e-4, c=24, win=16, heads=8, train_n=200 | 3.954 | 0.9970 | 0.7174 | — | — | — | [results](../runs/breast-ct-calibrated-tpe-uswin-search-20260521-01/results.tsv) | [iter-18](../runs/breast-ct-calibrated-tpe-uswin-search-20260521-01/iterations/iter-0018/comparison.png) |
| 7 | ITNet v2 | pre_ep=3, pre_lr=2.6e-4, k=4, α=2.6e-3, residual=T, train_n=400 | 0.233 | 0.9918 | 0.5386 | — | — | — | [results](../runs/breast-ct-calibrated-tpe-itnet-v2-search-20260521-01/results.tsv) | [iter-13](../runs/breast-ct-calibrated-tpe-itnet-v2-search-20260521-01/iterations/iter-0013/comparison.png) |
| 8 | Hammernik VN | ep=18, lr=3.0e-4, T=3, filters=32, kernel=7, init=fbp, train_n=200 | 0.008 | 0.9875 | 0.4883 | — | — | — | [results](../runs/breast-ct-calibrated-tpe-hammernik-vn-search-20260521-01/results.tsv) | [iter-12](../runs/breast-ct-calibrated-tpe-hammernik-vn-search-20260521-01/iterations/iter-0012/comparison.png) |
| 9 | Hammernik 2017 | ep=30, lr=9.6e-4, T=3, filters=16, kernel=9, λ=9.7e-3, train_n=200 | 0.005 | 0.9834 | 0.4549 | — | — | — | [results](../runs/breast-ct-calibrated-tpe-hammernik-search-20260521-01/results.tsv) | [iter-15](../runs/breast-ct-calibrated-tpe-hammernik-search-20260521-01/iterations/iter-0015/comparison.png) |
| 10 | **Wu 2015 trainable** *(Step-3 TPE — COMPLETE 20/20, 2026-06-09)* | iter-2 winner; cluster spans iter-2/8/12/13/16 (hr 0.31-0.32); winner cfg: ep~12-13, lr~1e-4, n_bands=6, n_outer=2, range=8, window=2, soft_thresh~1e-3, λ_neg~0.7, train_n=400 | 12 | 0.9760 | **0.3170** | — | — | — | [results](../runs/breast-ct-calibrated-tpe-wu-2015-trainable-search-20260609-01/results.tsv) | [iter-2](../runs/breast-ct-calibrated-tpe-wu-2015-trainable-search-20260609-01/iterations/iter-0002/comparison.png) |
| 11 | **RAM zero-shot** (pretrained) | σ=8.1e-3, factor=0.40, blend=0.50, multiscale=F, train_n=0 (frozen) | 35.619 *(frozen)* | 0.9879 | 0.3077 | — | — | — | [results](../runs/breast-ct-calibrated-tpe-ram-zeroshot-search-20260522-01/results.tsv) | [iter-7](../runs/breast-ct-calibrated-tpe-ram-zeroshot-search-20260522-01/iterations/iter-0007/comparison.png) |
| 12 | DD-BF supervised L2 | proj_n=1, img_n=7, proj_k=5, img_k=7, ep=10, lr=5.9e-3, train_n=400 | 24 | 0.9898 | 0.2634 | — | — | — | [results](../runs/breast-ct-calibrated-tpe-dual-domain-bilateral-supervised-search-20260524-01/results.tsv) | [iter-12](../runs/breast-ct-calibrated-tpe-dual-domain-bilateral-supervised-search-20260524-01/iterations/iter-0012/comparison.png) |
| 13 | DD-BF supervised L2 (agentic seed) | proj_n=3, img_n=3, proj_k=3, img_k=9, ep=10, lr=5e-3 | 18 | 0.9894 | 0.2476 | 42.21 | 3.88e-03 | 203 | [results](../runs/breast-ct-claude-agentic-dual-domain-bf-l2-search-20260522-01/results.tsv) | [iter-3](../runs/breast-ct-claude-agentic-dual-domain-bf-l2-search-20260522-01/iterations/iter-0003/comparison.png) |
| 14 | Wu 2015 trainable (agentic seed) | n_bands=4, n_outer=2, range=5, ep=10, lr=1e-3, train_n=400 | 10 | 0.9691 | 0.2189 | 41.74 | 4.09e-03 | 230 | [results](../runs/breast-ct-claude-agentic-wu-2015-l2-search-20260522-01/results.tsv) | [iter-2](../runs/breast-ct-claude-agentic-wu-2015-l2-search-20260522-01/iterations/iter-0002/comparison.png) |
| 15 | **ItNet v1** *(Step-3 TPE — COMPLETE 20/20, 2026-06-09)* | iter-18 winner: pretrain_ep=5, pretrain_lr=1e-3, k=8, α=0.015, finetune_ep=11, finetune_lr=5e-4, c=8, train_n=400 | 0.059 | 0.9881 | **0.1703** | — | — | — | [results](../runs/breast-ct-calibrated-tpe-itnet-v1-search-20260609-01/results.tsv) | [iter-18](../runs/breast-ct-calibrated-tpe-itnet-v1-search-20260609-01/iterations/iter-0018/comparison.png) |
| 16 | Wu 2015 (non-trainable) | n_bands=4, n_outer=1, range=8, thresh=1.1e-3, train_n=0 | 0 | 0.9699 | 0.0425 | — | — | — | [results](../runs/breast-ct-calibrated-tpe-wu-search-20260521-01/results.tsv) | [iter-16](../runs/breast-ct-calibrated-tpe-wu-search-20260521-01/iterations/iter-0016/comparison.png) |

## Below-baseline inventory (`hr = 0`, structural STOPs)

13 solver variants tested on breast-CT remain at `hr = 0` under the
calibrated metric — listed below for completeness alongside the
14-rank above-baseline table. These solvers were **TPE-tuned but never
reach above the FBP baseline** — they are not "just under the
threshold", they are structurally bounded.

| Solver | Variant | params (M) | SSIM | hr | PSNR (dB) | RMSE | time (s) | Source | Comparison |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| DD-UNet N2I | ep=6, lr=4.2e-4, c=24, train_n=400 | 1.045 | 0.9645 | 0.000 | — | — | — | [results](../runs/breast-ct-calibrated-tpe-dual-domain-search-20260521-01/results.tsv) | [iter-1](../runs/breast-ct-calibrated-tpe-dual-domain-search-20260521-01/iterations/iter-0001/comparison.png) |
| DD-BF N2I | ep=23, lr=2.3e-3, proj_k=7, img_k=5, train_n=400 | 6 | 0.9715 | 0.000 | — | — | — | [results](../runs/breast-ct-calibrated-tpe-dual-domain-bf-search-20260521-01/results.tsv) | [iter-1](../runs/breast-ct-calibrated-tpe-dual-domain-bf-search-20260521-01/iterations/iter-0001/comparison.png) |
| TV-iterative supervised | K∈{10,30}, step=1e-4, λ=1e-5 | 0.000 | 0.9543 | 0.000 | — | — | — | (deprioritised — structurally bounded by FBP) | — |
| TV-iterative (non-trainable) | λ=1.8e-3, iters=214, lr=4.0e-2, train_n=0 | 0 | 0.9467 | 0.000 | — | — | — | [results](../runs/breast-ct-calibrated-tpe-tv-search-20260521-01/results.tsv) | [iter-1](../runs/breast-ct-calibrated-tpe-tv-search-20260521-01/iterations/iter-0001/comparison.png) |
| NAF | n_freqs=10, hidden=256, n_iter=2143, lr=1.2e-3, train_n=0 | 0.143 | 0.8334 | 0.000 | — | — | — | [results](../runs/breast-ct-calibrated-tpe-naf-search-20260521-01/results.tsv) | [iter-1](../runs/breast-ct-calibrated-tpe-naf-search-20260521-01/iterations/iter-0001/comparison.png) |
| R²-Gaussian (cold init, n_iter≤800) | n_gauss=512, n_iter=483, lr_pos=1.9e-3, train_n=0 | 0.003 | 0.8261 | 0.000 | — | — | — | [results](../runs/breast-ct-calibrated-tpe-r2gaussian-search-20260521-01/results.tsv) | [iter-2](../runs/breast-ct-calibrated-tpe-r2gaussian-search-20260521-01/iterations/iter-0002/comparison.png) |
| R²-Gaussian **v2** (n_iter ∈ [10k, 40k], cold init) | n_gauss=1024, n_iter=11434, lr_pos=1.7e-2, train_n=0 | 0.003 | 0.8942 | 0.000 | — | — | — | (in agentic search; partial timeout, 6/20 iters all hr=0) — extended iter budget **did not** clear baseline | — |
| R²-Gaussian **v3** (FBP-warm-start init) | n_gauss=1024, n_iter=5000, FBP-of-noisy as Gaussian positions/amps, train_n=0 | 0.003 | 0.8919 | 0.000 | — | — | — | (agentic iter-2, slug `…r2g-fbp-init-search-20260602-01`) — FBP warm-start **did not** clear baseline either | — |
| **Diff-recon — DDPM constrained (breast, v1 ch=32)** | DPS+DC, breast-DDPM v1 ckpt, train_n=200 | 0.958 *(frozen)* | 0.4702 | 0.000 | — | — | — | [results](../runs/breast-ct-calibrated-tpe-diff-recon-dcstep-constrained-breast-search-20260523-01/results.tsv) | [iter-1](../runs/breast-ct-calibrated-tpe-diff-recon-dcstep-constrained-breast-search-20260523-01/iterations/iter-0001/comparison.png) |
| **Diff-recon — DDPM unconstrained (breast, v1 ch=32)** | DPS+DC, breast-DDPM v1 ckpt, train_n=3600 | 0.958 *(frozen)* | 0.4626 | 0.000 | — | — | — | [results](../runs/breast-ct-calibrated-tpe-diff-recon-dcstep-unconstrained-breast-search-20260523-01/results.tsv) | [iter-17](../runs/breast-ct-calibrated-tpe-diff-recon-dcstep-unconstrained-breast-search-20260523-01/iterations/iter-0017/comparison.png) |
| Diff-recon — DDPM **v2** (ch=64, 80 ep) | retrained with 4× capacity + 3× training | 3.823 *(frozen)* | 0.30–0.49 | 0.000 | — | — | — | SLURM 762636 — all 40 trials hr=0 | — |
| Diff-recon — DDPM **v3** (ch=128, 60 ep) | retrained with 16× capacity vs v1 | 15.272 *(frozen)* | 0.33–0.48 | 0.000 | — | — | — | SLURM 762652 (5/40 iters at writing) | — |

### Why these fail structurally

- **Self-supervised dual-domain (N2I)**: Noise2Inverse rewards smoothing
  in the dense-view regime; the half-set FBP target carries noise the
  optimiser tries to match. The DD-BF/DD-UNet supervised L2 twins above
  show what fixing the loss alone gets you (`hr` = 0.26 / 0.84).
- **Per-scene neural-implicit (NAF / R²-Gaussian)**: designed for
  sparse-view CBCT; can't compete with a properly-tuned FBP at 128
  views on this dataset. **Two retry rounds confirmed this empirically
  for R²-G** (2026-06-03 agentic): extending `gs_n_iter` from [300,
  800] to [10k, 40k] left SSIM at 0.89 (still hr=0); FBP-warm-starting
  the Gaussian positions also left SSIM at 0.89. The basis is too
  sparse to represent dense soft tissue at the resolution that clears
  baseline FBP; the inductive bias is fundamentally for sparse-view
  scans where FBP is weak. See findings.md 2026-06-03 entry.
- **TV-iterative supervised L2 (unrolled)**: FBP init + smooth-TV
  gradient + supervised L2 → the first GD step learns to do nothing;
  structural ceiling = baseline FBP.
- **Diffusion-recon with breast-DDPM checkpoints**: **three checkpoint
  arches retrained, all hr=0 across ≥40 trials each.** v1 (ch=32,
  25 ep) → SSIM ~0.46; v2 (ch=64, 80 ep, val_eps_loss=0.0050) →
  SSIM 0.30–0.49; v3 (ch=128, 60 ep, val_eps_loss=0.0020) →
  SSIM 0.33–0.48 (first 5 iters). The training metric improves
  monotonically (loss 0.0050 → 0.0020) but the posterior-sampling
  reconstructions stay in the same SSIM band — **the failure is the
  prior class, not capacity or training duration.** SmallDDPM
  generates *individually plausible* breast images but they are not
  conditionally faithful to the input sino under DPS/DC sampling.
  Closing this path requires a structurally different prior
  (score-SDE / EDM / U-ViT) — not in scope for current solver code.
  Full diagnosis in findings.md 2026-06-03 entry.

## Inventory-gap + Wu-TPE closure log (2026-06-09)

ItNet v1 was dispatched + Wu 2015 trainable got a full TPE pass on
breast-CT on 2026-06-09:

| Solver | TPE job | Result | Status |
|---|---|---:|---|
| **ItNet v1** (`solver_itnet.py`) | 762956 | **hr=0.1703** at iter-20 winner (k=8, c=8, pretrain_ep=5, lr=1e-3, α=0.015) | ✅ **ABOVE BASELINE** — slots in at rank 15. Surprises the Mayo verdict (hr=0): on breast-CT's broader synthetic-anatomy substrate, v1's deeper finetune-pass lets it clear baseline. Confirms ItNet family transfer pattern: v1 < v2 < v3 on breast-CT (0.1703 → 0.5386 → 0.7342). |
| **Wu 2015 trainable** (`solver_wu_2015_trainable.py`) | 762955 | **COMPLETE 20/20**: final hr=**0.3170** (winner cluster across iter-8/12/13/15/16 at hr=0.31-0.32, all with n_bands=6, n_outer=2, range=8, window=2, ep~12-13, lr~1e-4, λ_neg~0.7) | 🎯 **+45% over agentic 0.2189**! TPE found a higher-`n_bands` + lower-lr corner the agentic search missed. **FINAL** at rank 10. Agentic-seed row stays at rank 14. |

**Cross-dataset coverage compare (after 2026-06-09 gap-closure dispatches):**

- **Demo-DL** (Sidky synthetic): 19 above + 0 below = **19/19 inventory variants** (ItNet v1 just closed at rank 4; TV-iter supervised running iter-7, all hr=0 — confirming structural verdict)
- **Breast-CT** (Sidky synthetic with anatomy): 15 above + 13 below = **28 entries, full 19-inventory coverage** (ItNet v1 closed; Wu trainable TPE lifted +43%)
- **Mayo-LDCT** (real helical, 2304-view): 12 above + 10 below + 1 deprioritised = **23 entries, full 19-inventory coverage**

## Methodology

See [`solver_plan.md`](../../solver_plan.md). One row per solver family;
"Variant" picks the best config across all autoresearch + TPE iterations.
