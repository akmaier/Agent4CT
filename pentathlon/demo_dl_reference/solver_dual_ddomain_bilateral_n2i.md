# `solver_dual_ddomain_bilateral_n2i.py` — Dual-domain bilateral filters, Noise2Inverse self-supervised

Companion design doc. For the supervised-L2 variant of the same
architecture see `solver_dual_ddomain_bilateral_supervised.md`. For the
U-Net dual-domain variant see `solver_dual_ddomain_n2i.md`.

## What it is

The **ultra-low-parameter** alternative to dual-domain U-Net denoising.
Each "denoiser" is a `BilateralFilterStack` — N chained
`TrainableBilateralFilter2d` layers (3 trainable scalars each:
σ_x, σ_y, σ_r). With `proj_n_bf = img_n_bf = 1` (default) the entire
network has 6 trainable parameters. Trained via the same
Noise2Inverse self-supervision as the U-Net variant
(`DualDomainPipeline.training_step`).

```
sino_full ─→ split half-sets (64 angles each) ─→ pipeline(x_a), pipeline(x_b)

pipeline(x) = img_dn( R_half.fbp( proj_dn(x) ) )
            = BF_img( FBP_64( BF_proj(x) ) )

loss = ½ MSE(y_hat_a, R_half.fbp(x_b)) + ½ MSE(y_hat_b, R_half.fbp(x_a))
     + 1.0 · neg_pen(y_hat)
```

## Design considerations

- **Wagner et al. 2022 (`literature/2201.10345_*.md`)**: 97% of dual
  U-Net SSIM with only 8 parameters on abdomen CT — *if* the
  parameters are initialised and trained sensibly.
- **σ_x / σ_y are spatial sigmas; σ_r is the range (intensity) sigma.**
  In the projection-domain BF, σ_x indexes the *detector* direction
  (after FBP this becomes radial direction → radial blur if σ_x is
  large) and σ_y indexes the *angle* direction (→ angular
  interpolation, useful for sparse views). σ_r ≪ data range preserves
  edges; σ_r ≫ data range degenerates to a Gaussian.
- **`proj_n_bf` / `img_n_bf` (Wagner §3.2)**: stacking N BFs in
  series in a domain. Each BF contributes 3 trainable params; n=3
  per domain = 18 params total. The image-domain stack readily
  differentiates into a multi-scale cascade (BF1 mild + tight σ_r,
  BF2 wider, BF3 widest); the projection-domain stack often stays
  locked-symmetric when proj filters are near-identity (see
  Weaknesses).
- **`proj_sx` should be small** on sparse-view scans to avoid
  introducing radial blur on top of the existing streak artefacts.
- **`proj_sr` should be very small** — sinogram values are line
  integrals, and the projection-domain BF should preserve attenuation
  edges (e.g. the breast/air boundary in the sinogram) tightly.

## Strengths

- **6–18 parameters total.** Trains and runs on a CPU.
- **Interpretable.** σ values are physically meaningful filter widths
  — you can read them from the per-epoch log and reason about what
  the model is doing.
- **Cannot memorise** the training set. Whatever it learns has to
  generalise via the bilateral-filter inductive bias.
- **The BF stack adds capacity at a fraction of a U-Net's parameter
  budget.** 3 BFs/domain = 18 params vs SmallUNet(c=16) ≈ 466 k.

## Weaknesses

- **N2I + dense-view = over-smoothing.** Same root cause as the U-Net
  variant: half-view FBP targets carry a noise floor; MSE against
  noisy targets rewards smoothing. Image-domain σ_x grew 0.5 → 1.08 in
  2 epochs on breast-CT iter-1 — saturating the kernel.
- **Projection-domain symmetry trap.** When proj σ values are near
  identity (proj_sx ≤ 0.05 on a kernel=3 grid → kernel is effectively
  identity), the gradients on the proj-stack are tiny *and all
  identical across the stack*. So stacked proj BFs initialised
  symmetrically stay symmetric — chaining them adds parameters but no
  expressivity in that regime.
- **Tied to fan-beam (PYRO-NN PyTorch backend)**; cone-beam would
  need projector swap and a 3D BF.

## When to prefer this solver

- **Parameter-budget constrained** challenges or systems
  (interpretable medical pipelines, edge deployment).
- When you want a **baseline before the U-Net** — gets you 70-80% of
  the way at 0.001% of the parameter count.
- **Sparse-view abdomen CT** (Wagner's original setting) — N2I works
  well there.

## When to **not** prefer this solver

- **Dense-view dense-tissue** like breast-CT — N2I bottleneck plus
  the BF's tendency to smooth across fibroglandular detail. Use
  `solver_dual_ddomain_bilateral_supervised.py`.
- When you need to recover **fine high-frequency detail** that's
  buried in noise — the BF will smooth it away.

## Knobs (in `CONFIG`)

| Knob | Default | Effect |
|---|---:|---|
| `proj_n_bf` | 1 | Number of chained BFs in projection domain. |
| `img_n_bf` | 1 | Number of chained BFs in image domain. |
| `proj_kernel`, `img_kernel` | 5, 7 | Spatial kernel size (odd). |
| `proj_sx`, `proj_sy`, `proj_sr` | 1.0, 2.0, 0.02 | Initial proj σ values. Wagner abdomen defaults. |
| `img_sx`, `img_sy`, `img_sr` | 1.5, 1.5, 0.02 | Initial img σ values. |
| `epochs` | 20 | Wagner used 20+. More epochs → more over-smoothing on dense scans. |
| `lr` | 5e-3 | Wagner's BF lr (vs 5e-5 for U-Nets). |

## Hints for the next autoresearch agent

- The proj sigmas in the Wagner defaults (proj_sx=1.0, proj_sy=2.0)
  are **too aggressive for sparse-view breast-CT** — they produce
  radial blur. On breast-CT, start with `proj_sx=0.01,
  proj_sy=0.3, proj_sr=0.0005` (proven sensible from iters in
  `breast-ct-claude-agentic-dual-domain-bf-search-20260521-01`).
- The **`proj_n_bf` stack only helps if the proj filter has gradient
  signal** (i.e. proj σ values that move during training). If proj
  is near-identity, increase `proj_n_bf` is wasted parameters; spend
  the budget on `img_n_bf` instead.
- To **break proj-stack symmetry**, initialise each BF differently
  (currently the stack initialises all BFs from the same scalars —
  this is a known limitation; future work could add per-BF init
  config).
- If the goal is to beat baseline on a dense-view challenge, prefer
  `solver_dual_ddomain_bilateral_supervised.py` immediately. Don't
  waste iters here.

## Empirical results on breast-CT (128 views, intensity-calibrated)

| Source | Config | val_psnr | val_ssim | hr |
|---|---|---:|---:|---:|
| baseline FBP | — | 39.74 dB | 0.957 | 0 |
| `claude-agentic-dual-domain-bf-search-20260521-01/iter-1` | n=1, ep=2 | 37.50 dB | 0.957 | 0 |
| `claude-agentic-dual-domain-bf-search-20260521-01/iter-2` | n=1, ep=5 | 37.24 dB | 0.948 | 0 |
| 20-iter TPE search | various | — | — | 0 |

Same architecture with supervised L2 (see
`solver_dual_ddomain_bilateral_supervised.py`): hr=0.248 (n=3×3
chain, 18 params).
