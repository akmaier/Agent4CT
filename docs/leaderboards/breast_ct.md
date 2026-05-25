---
title: Breast-CT leaderboard
description: Calibrated-SSIM-headroom ranking, one row per solver family. See solver_plan.md for methodology.
---

# Breast-CT leaderboard

128-view 2-D fan-beam sparse synthetic phantoms (Sidky-group breast model,
real μ range up to ~0.5). All metrics through
`ddssl_ldct.metrics.evaluate_calibrated`: linear intensity calibration on
the foreground inside an inscribed-circle FOV mask, then PSNR/SSIM/RMSE
on the calibrated pred. `hr = max(0, 1 − rmse/baseline_rmse)` where
baseline is FBP at SSIM 0.957 / PSNR 39.74 dB.

| Rank | Solver | Best config | params | SSIM | PSNR (dB) | hr | Source slug / iter |
|---:|---|---|---:|---:|---:|---:|---|
| 1 | **Learned Primal-Dual** | I=8, hidden=96, ep=23, lr=3.2e-4, grad_clip=0.3 | 1.49 M | 0.9996 | ~56 | **0.9062** | `breast-ct-calibrated-tpe-lpd-search-20260524-01` / trial 11 |
| 2 | LPD (agentic seed) | I=10, hidden=64, ep=20, lr=5e-4, grad_clip=1.0 | 0.88 M | 0.9985 | 55.08 | 0.8290 | `breast-ct-claude-agentic-learned-primal-dual-search-20260522-01` / iter-3 |
| 3 | **DD-UNet supervised L2** | c=24, ep=18, lr=2.1e-4, λ_neg=0.58 (TPE) | 1.04 M | 0.999 | — | **0.8361** | `breast-ct-calibrated-tpe-dual-domain-supervised-search-20260524-01` |
| 4 | ITNet v3 | (TPE iter-18) | ~2.5 M | 0.9965 | — | 0.7342 | `breast-ct-calibrated-tpe-itnet-v3-search-20260521-01` / iter-18 |
| 5 | USwin | (TPE iter-18) | — | 0.9970 | — | 0.7174 | `breast-ct-calibrated-tpe-uswin-search-20260521-01` / iter-18 |
| 6 | ITNet v2 | (TPE iter-13) | — | 0.9918 | — | 0.5386 | `breast-ct-calibrated-tpe-itnet-v2-search-20260521-01` / iter-13 |
| 7 | Hammernik VN | (TPE iter-12) | — | 0.9875 | — | 0.4883 | `breast-ct-calibrated-tpe-hammernik-vn-search-20260521-01` / iter-12 |
| 8 | Hammernik 2017 | (TPE iter-15) | — | 0.9834 | — | 0.4549 | `breast-ct-calibrated-tpe-hammernik-search-20260521-01` / iter-15 |
| 9 | **RAM zero-shot** (pre-trained) | sigma=0.008, blend=0.42 (TPE iter-7) | (frozen) | 0.9879 | — | 0.3077 | `breast-ct-calibrated-tpe-ram-zeroshot-search-20260522-01` / iter-7 |
| 10 | DD-BF supervised L2 | proj_n=1, img_n=7, img_kernel=7 (TPE) | ~24 | — | — | **0.2634** | `breast-ct-calibrated-tpe-dual-domain-bilateral-supervised-search-20260524-01` |
| 11 | Wu 2015 trainable | lr=1e-3, ep=10, n_bands=4 | 10 | 0.9691 | 41.74 | 0.2189 | `breast-ct-claude-agentic-wu-2015-l2-search-20260522-01` / iter-2 |
| 12 | Wu 2015 (non-trainable) | TPE (iter-16) | 0 | 0.9699 | — | 0.0425 | `breast-ct-calibrated-tpe-wu-search-20260521-01` / iter-16 |
| — | DD-UNet N2I | (best of TPE) | — | 0.9645 | — | 0.000 | `breast-ct-calibrated-tpe-dual-domain-search-20260521-01` (N2I loss bottleneck) |
| — | DD-BF N2I | (best of TPE) | — | 0.9715 | — | 0.000 | `breast-ct-calibrated-tpe-dual-domain-bf-search-20260521-01` (N2I loss bottleneck) |
| — | **TV-iterative supervised** | K∈{10,30}, step=1e-4, λ=1e-5 | 20 | 0.9543 | 32.05 | 0.000 | `breast-ct-claude-agentic-tv-iterative-supervised-search-20260523-01` (structurally bounded by FBP) |
| — | TV-iterative (non-trainable) | TPE | 0 | 0.9620 | — | 0.000 | `breast-ct-calibrated-tpe-tv-search-20260521-01` |
| — | NAF | n_iter=12000, lr=1e-3 | — | 0.7914 | 16.90 | 0.000 | `breast-ct-claude-agentic-naf-search-20260523-01` / iter-2 (wrong inductive bias for dense view) |
| — | R2Gaussian | n_gauss=1024, n_iter=600 | — | 0.8861 | 26.61 | 0.000 | `breast-ct-claude-agentic-r2gaussian-search-20260523-01` / iter-4 (wrong inductive bias) |
| — | **Diffusion recon — DDPM constrained** | DPS+DC, breast-DDPM ckpt | (frozen) | 0.4702 | — | 0.000 | `breast-ct-calibrated-tpe-diff-recon-dcstep-constrained-breast-search-20260523-01` (under-trained ckpt) |
| — | **Diffusion recon — DDPM unconstrained** | DPS+DC, breast-DDPM ckpt | (frozen) | 0.4626 | — | 0.000 | `breast-ct-calibrated-tpe-diff-recon-dcstep-unconstrained-breast-search-20260523-01` (under-trained ckpt) |

**Baseline FBP**: SSIM 0.957, PSNR 39.74 dB, hr 0.

**hr = 0 entries are structural deal-breakers**, not "just under the
threshold". They mean the recon does not improve on baseline FBP under
the calibrated metric:
- *Self-supervised dual-domain*: Noise2Inverse rewards smoothing in
  dense-view regime; the half-set FBP target carries noise the
  optimiser tries to match. The DD-BF/DD-UNet supervised twins above
  show what fixing the loss alone gets you (0.21 / 0.83).
- *Per-scene neural-implicit (NAF / R2Gaussian)*: designed for sparse-view CBCT; can't compete with a properly-tuned FBP at 128 views.
- *TV-iterative supervised L2 (unrolled)*: FBP init + smooth-TV gradient + supervised L2 → first GD step learns to do nothing; structural ceiling = baseline FBP.
- *Diffusion-recon with breast-DDPM checkpoints*: the existing
  `ddpm_breast_*_final.pt` checkpoints (n_train=3600 unconstrained /
  n_train=200 constrained) produce SSIM ~0.46 — well below baseline
  0.957. **The checkpoints themselves are weak**; needs retraining.
  See `solver_diffusion_recon.md` for the diagnosis.

## Methodology

Methodology of how this leaderboard is generated is in
[`/solver_plan.md`](../../solver_plan.md). One row per solver family;
"variant" picks the best config across all autoresearch + TPE iterations.
