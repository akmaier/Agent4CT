# `solver_dual_ddomain_bilateral_supervised.py` — Dual-domain BF, supervised L2 (full 128 views)

Companion design doc. For the self-supervised version of the same
architecture see `solver_dual_ddomain_bilateral_n2i.md`. For the
U-Net supervised variant see `solver_dual_ddomain_supervised.md`.

## What it is

The supervised-loss twin of `solver_dual_ddomain_bilateral_n2i.py`.
Two `BilateralFilterStack` denoisers (each is N chained
`TrainableBilateralFilter2d` layers) in projection and image domain,
trained on the **full 128-view forward pass** with **MSE against the
clean phantom + non-negativity penalty**.

```
pred = img_dn( R_full.fbp( proj_dn(sino_full) ) )
loss = MSE(pred, clean_phantom) + 1.0 · negativity_penalty(pred)
```

The pipeline class is `FullViewBilateralPipeline` (defined inline).

## Design considerations

- **Built 2026-05-22** as the lowest-parameter test of "is the N2I
  loss the bottleneck on dense-view breast-CT?". Same 6-param BF
  pipeline as the N2I variant — only the loss + the half-vs-full
  view split change. Result: hr 0 → 0.21.
- **`proj_n_bf` / `img_n_bf` (Wagner §3.2)** — same chaining as the
  N2I variant. The supervised loss removes the σ-runaway pressure,
  so the BF chain can train past identity without collapsing.
  Empirically on breast-CT, `img_n_bf=3` lifted hr from 0.21 to
  0.248; `proj_n_bf=3` made no difference (proj stack is symmetric
  when proj filters are near-identity — see Weaknesses of the N2I
  doc).
- **Defaults baked in are physics-informed**: `proj_sx=0.01,
  proj_sy=0.3, proj_sr=0.0005` — these are the values the agentic
  loop converged to (per `findings.md`); they keep projection-domain
  smoothing near-identity to avoid introducing radial blur.

## Strengths

- **Ultra-low parameter count** with respectable headroom. 6 params
  (n=1×1) → hr 0.21; 18 params (n=3×3) → hr 0.25. Comparable to
  classical-iterative TV with random / TPE searched λ.
- **Interpretable σ-trajectories**. Per-epoch printout shows σ_x σ_y
  σ_r per BF in each stack — you can diagnose whether the model is
  smoothing too hard, where the multi-scale cascade is settling, etc.
- **Symmetric / asymmetric stack init** is a single CONFIG flag away
  (currently single-init; per-BF init would be a small extension).
- **Compiles to a single forward kernel** at inference — fast and
  small enough for edge deployment.

## Weaknesses

- **Capacity-limited**. 18 params can't capture the fine-grained
  noise patterns a 466k-param U-Net can. Ceiling on breast-CT ≈ hr
  0.25 — far below `solver_dual_ddomain_supervised.py`'s 0.81.
- **Image-domain σ_x still grows during training** (0.5 → 8.3 in 10
  epochs on iter-1 with kernel=5). At very large σ the kernel
  saturates → the BF degenerates to a uniform box filter at that
  layer. Mitigated by **using a larger kernel** (kernel=9 leaves
  room for σ ≈ 2 to learn) or limiting training time.
- **Projection-domain BF stays near identity** in practice on
  breast-CT (proj_sr=0.0005 + small proj_sx). Adding `proj_n_bf` >
  1 doesn't help when the proj BFs share gradients (symmetric
  collapse).
- **Same supervised-only caveat** as the U-Net supervised variant:
  needs paired clean data at train time.

## When to prefer this solver

- **Parameter-budget constrained** + a clean target is available.
- As a **baseline / sanity check** before scaling to the U-Net
  supervised variant — establishes "what's the cheapest model that
  beats baseline FBP?".
- For **interpretability-required** medical-pipeline integrations
  where every parameter must be defensible.

## When to **not** prefer this solver

- **Headroom hunting on dense-view supervised tasks**. The U-Net
  variant is +3.5 dB PSNR / +0.56 hr ahead at < 0.01% of the
  parameter cost difference.
- **Sparse-view + no clean target.** Use the N2I bilateral variant.

## Knobs (in `CONFIG`)

| Knob | Default | Effect |
|---|---:|---|
| `proj_n_bf` | 1 | Chained BFs in projection domain (Wagner §3.2). |
| `img_n_bf` | 1 | Chained BFs in image domain. **n=3 gave +0.018 hr on breast-CT.** |
| `proj_kernel`, `img_kernel` | 3, 5 | Spatial kernel size (odd). Larger img_kernel gives room for σ to grow. |
| `proj_sx`, `proj_sy`, `proj_sr` | 0.01, 0.3, 0.0005 | Physics-informed init. Tiny along-detector to avoid radial blur. |
| `img_sx`, `img_sy`, `img_sr` | 0.5, 0.5, 0.02 | Image-domain init. |
| `epochs` | 10 | Convergence is fast (σ stabilises by epoch 1-3 with kernel=9). |
| `lr` | 5e-3 | Wagner's BF lr. |
| `lambda_neg` | 1.0 | Non-negativity penalty weight. |

## Hints for the next autoresearch agent

- **`img_n_bf` chain works** — the supervised loss removes the
  noise-floor over-smoothing pressure that locks the N2I variant.
  Test `img_n_bf ∈ {3, 5, 7}` to find where returns saturate.
- **`proj_n_bf` chain does NOT help** with the default symmetric
  init. To make it useful, add per-BF init perturbation (small
  random σ offset across the stack) — would be a one-line solver
  edit.
- **`img_kernel` sweep**: kernel=5 saturates by σ≈3 (kernel radius
  2); kernel=9 lets σ≈4 do useful work. Try kernel=11 / 13 to see
  if longer-range smoothing helps.
- The **`proj_sr` ceiling is set by the sinogram intensity
  variations.** On other datasets, recalibrate this — for Mayo (mu
  scale 0..0.05) proj_sr should be ~10× smaller than for breast-CT.

## Cross-dataset observations

| Dataset | Best hr | Config | Notes |
|---|---:|---|---|
| `demo_dl` | — | not run as supervised | The N2I twin (`solver_dual_ddomain_bilateral_n2i.py`) hits 0.3611 on demo_dl. Supervised L2 should improve on that. |
| `breast_ct` | **0.2476** | proj_n=3, img_n=3, img_kernel=9 (18 params) | DD-BF supervised L2 hits 0.25 with just **18 trainable params** — within 0.05 hr of RAM zero-shot (0.30) which uses a frozen 10M-param pretrained net. Best parameter-efficiency reachable without a learned CNN denoiser. |
| `mayo_ldct` | — | not yet run | |

**Pattern**: useful as an **interpretable / ultra-low-parameter
baseline** before scaling to a U-Net. On breast_ct, the supervised L2
variant is a 6× improvement over the N2I variant (loss = bottleneck,
same as for the U-Net twin).

## Empirical results on breast-CT (128 views, intensity-calibrated)

| Source | Config | params | val_psnr | val_ssim | hr |
|---|---|---:|---:|---:|---:|
| baseline FBP | — | 0 | 39.74 dB | 0.957 | 0 |
| `claude-agentic-dual-domain-bf-l2-search-20260522-01/iter-1` | n=1×1, img_k=5 | 6 | 41.83 dB | 0.986 | 0.214 |
| `claude-agentic-dual-domain-bf-l2-search-20260522-01/iter-2` | n=1×1, img_k=9 | 6 | 42.01 dB | 0.987 | 0.230 |
| `claude-agentic-dual-domain-bf-l2-search-20260522-01/iter-3` | n=3×3, img_k=9 | 18 | 42.21 dB | 0.989 | 0.248 |

For reference, same architecture with N2I loss
(`solver_dual_ddomain_bilateral_n2i.py`): val_psnr 37.5 dB,
SSIM 0.957, hr=0.
