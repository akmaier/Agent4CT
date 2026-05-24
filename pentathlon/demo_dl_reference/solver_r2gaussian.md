# `solver_r2gaussian.py` — R²-Gaussian-lite (Zha et al. 2024, NeurIPS)

Per-scene reconstruction via 2-D Gaussian splatting: `N` anisotropic
Gaussian primitives rasterised into the μ image, then projected through
`PyronnFanBeamProjector` and matched against the noisy sinogram.

Origin: Zha R., Cheng L., Han L., Gao C., Zhang Y. *"R²-Gaussian:
Rectifying Radiative Gaussian Splatting for Tomographic
Reconstruction"*, NeurIPS 2024 (arXiv:2405.20693).

## What it is

A trainable bag of `gs_n_gaussians` anisotropic 2-D Gaussian primitives
defined by (position, log-scale_x, log-scale_y, rotation, amplitude).
Each iteration:

1. Rasterise the Gaussian sum into the (H, W) μ image (chunked over
   gaussians for memory).
2. Forward-project through PYRO-NN → predicted sinogram.
3. MSE against the noisy sino + TV penalty on the μ image +
   non-negativity penalty.
4. Adam step over (positions, scales, rotation, amplitudes) with
   separate lr's per group.

Per-scene optimisation: nothing transfers between scans.

## Design considerations

- **Lite variant** — no tiled splatting kernels; just direct
  per-pixel Gaussian evaluation, chunked along the n-gaussians axis
  to control peak memory.
- **Chunk size 64** (default) balances forward memory vs autograd
  graph length. **Chunk size also limits effective n_gaussians** —
  too many chunks (≥30) accumulate enough autograd activations to
  OOM on a 16-GB Q5000.
- **Per-scene wall** via `gs_outer_wall_s` (added 2026-05-23); default
  600 s was too tight for n_iter=600 at n_gauss=1024.

## Strengths

- **Memory-bounded forward pass** via chunked rasterisation.
- **Explicit primitives** — interpretable; you can visualise the
  Gaussians.
- **Per-scene optimisation** — no training data needed.

## Weaknesses

- **Wrong inductive bias for dense-view fan-beam.** Same root cause as
  NAF: at 128 angles the FBP is already strong, so a from-scratch
  per-scene fit can't beat it. The Gaussian primitives can express
  any smooth μ image, but the fit converges to something away from
  truth in the supervised L2 sense.
- **Autograd memory blowup** when `gs_n_gaussians > ~1500` and chunk
  size is small — 32 chunks × per-chunk activation graph saturates
  the GPU.
- **Per-scene wall is hard-coded by config** — old solver had the
  outer wall hardcoded at 600 s; patched 2026-05-23 to be cfg-driven.

## When to prefer this solver

- **Sparse-view CT** with limited iterations available — the Gaussian
  primitives can express low-frequency structure efficiently.
- **As an interpretable / no-training baseline**.

## When to **not** prefer this solver

- **Dense-view CT** — see Empirical results below; structurally
  beaten by FBP.
- **Pentathlon all-rounder** — per-scene optimisation doesn't share.

## Knobs (in `CONFIG`)

| Knob | Default | Effect |
|---|---:|---|
| `gs_n_gaussians` | 1024 | Primitive count. **2048 OOMs on Q5000** with the default chunk size; stay ≤ 1024 or rewrite the rasterisation to use checkpointing. |
| `gs_n_iter` | 600 | Per-scene Adam iters. Paper default. |
| `gs_lr_pos` / `_scale` / `_amp` / `_rot` | 5e-3 / 1e-2 / 1e-2 / 1e-2 | Param-group lr's. Adam. |
| `gs_amp_init` | 0.01 | Initial amplitude (softplus-parametrised). |
| `gs_scale_init` | 0.04 | Initial spatial scale in normalised [-1,1] coords. |
| `gs_tv_weight` | 1e-4 | TV penalty weight on μ image. |
| `gs_n_clip` | 0.05 | μ clamp upper bound. |
| `gs_outer_wall_s` | 600 | Total wall across all val scenes. **Bump to 3600** when val_n=5 with default `n_iter=600` — per-scene cost is ~440 s on Q5000 = 2200 s total. |

## Hints for the next autoresearch agent

- **Deprioritised on dense-view breast-CT.** Four agentic iters tried:
  - iter-1: n_gauss=2048, n_iter=3000 → CUDA OOM.
  - iter-2: n_gauss=1024, n_iter=3000 → 10-min wall at scene 1.
  - iter-3: n_gauss=1024, n_iter=600 → 10-min wall at scene 2.
  - iter-4: n_gauss=1024, n_iter=600, gs_outer_wall_s=3600 → all 5
    scenes fit but hr=0, PSNR 26.6 dB vs baseline 39.6 dB.
  **Don't burn iters here.**
- **Memory fix for n_gauss > 1024**: wrap each chunk in
  `torch.utils.checkpoint.checkpoint(...)` so the autograd graph
  recomputes the per-chunk rasterisation during backward instead of
  retaining activations. 2× compute, ~30× less activation memory.
  Not yet implemented; if you need more gaussians, do this first.
- **For sparse-view tracks**, this is still a reasonable baseline —
  the architecture is unchanged.

## Cross-dataset observations

| Dataset | Best hr | Config | Notes |
|---|---:|---|---|
| `demo_dl` | 0.2999 | n_gauss=1024, n_iter=600 (TPE iter-14) | Bottom-pack on demo_dl — still beats baseline FBP modestly. |
| `breast_ct` | **0.000** | n_gauss=1024, n_iter=600, gs_outer_wall_s=3600 | Structural mismatch — 13 dB below baseline FBP. |
| `mayo_ldct` | — | not yet run | Likely same outcome (dense-view, real anatomy). |

**Pattern**: same as NAF — Gaussian primitives can compete on `demo_dl`
(simple ellipse phantoms) but lose on `breast_ct` and any other
dataset where baseline FBP is already strong. Wrong architectural
family for dense-view tomography.

## Empirical results on breast-CT (128 views, intensity-calibrated)

| Source | Config | scenes fit | val_ssim | val_psnr | hr | notes |
|---|---|---:|---:|---:|---:|---|
| baseline FBP | — | 5 | 0.957 | 39.61 dB | 0 | reference |
| `calibrated-tpe-r2gaussian-search-20260521-01` | best of 12 (TPE) | — | — | — | 0 | |
| `claude-agentic-r2gaussian-search-20260523-01/iter-1` | n_gauss=2048, n_iter=3000 | — | — | — | — | OOM |
| `claude-agentic-r2gaussian-search-20260523-01/iter-2` | n_gauss=1024, n_iter=3000 | 1/5 | — | — | 0 | wall |
| `claude-agentic-r2gaussian-search-20260523-01/iter-3` | n_gauss=1024, n_iter=600 | 2/5 | 0.884 | 26.4 dB | 0 | wall |
| `claude-agentic-r2gaussian-search-20260523-01/iter-4` | n_gauss=1024, n_iter=600, gs_outer_wall_s=3600 | **5/5** | 0.886 | 26.6 dB | 0 | clean run |

All variants are **13+ dB below baseline FBP**. Confirmed structural
mismatch; expect to stay at hr=0 at any configuration on dense-view
breast-CT.
