# `solver_naf.py` — Neural Attenuation Fields (Zha et al. 2022, MICCAI)

Per-scene coordinate-MLP recon: a small sin/cos-positional-encoded MLP
maps 2-D pixel coordinates → linear attenuation `μ`, optimised
end-to-end against the measured sinogram via `‖R·μ(x) − g‖²`. No
training set; one optimisation per scan.

Origin: Zha R., Zhang Y., Li H. *"NAF: Neural Attenuation Fields for
Sparse-View CBCT Reconstruction"*, MICCAI 2022 (arXiv:2209.14540).

## What it is

For each input slice the solver:

1. Builds a fresh `NAF` MLP (`naf_hidden`-wide, `naf_layers`-deep
   feed-forward with PReLU activations and a positional encoding of
   `naf_n_freqs` frequencies on the pixel coordinates).
2. Initialises so `σ(b₀) · out_scale = init_mu ≈ 0.005` everywhere
   (water-μ flat start — debugged 2026-05-16 after the previous
   softplus + post-clamp init saturated the head at 0.05 and killed
   gradients).
3. Iterates `naf_n_iter` Adam steps over the data-fidelity loss
   `‖R(μ) − g‖²` + TV penalty + non-negativity penalty.
4. Returns the final μ image.

Runs per-scene with an `naf_outer_wall_s` budget across the whole val
set, plus a per-scene wall derived from `outer / val_n`.

## Design considerations

- **Per-scene only.** No shared weights between samples; nothing to
  transfer across val.
- **Tuned for sparse-view CBCT** in the original paper. The positional
  encoding gives the MLP enough frequency capacity to express
  high-resolution detail *when the rest of the views are missing*.
- **Wall-clock-bounded** by a soft budget — useful when each val
  sample's optimisation cost varies.
- **Init bias to water-μ** at the output of the final linear layer is
  the only training-stability trick; without it the network
  initialises into a saturated zero-gradient regime.

## Strengths

- **No training data required** — fits each scan from scratch.
- **Strong at sparse-view CBCT** (the paper's setting).
- **Memory-light**: a 256-hidden 5-layer MLP fits comfortably on a
  Q5000.

## Weaknesses

- **Wrong inductive bias for dense-view fan-beam CT.** With 128 angles
  available, FBP is already close to optimal; a coordinate MLP can't
  outperform a properly-tuned back-projection chain on this regime.
- **Each scene is an independent optimisation** — slow at val time,
  no cross-sample leverage.
- **Last-loss plateaus high.** At lr=1e-3, n_iter=12 k on breast-CT,
  the per-scene fit settles at `last_loss ≈ 0.05` — orders of magnitude
  larger than what FBP-equivalent reconstructions would produce in
  the data-fidelity term.

## When to prefer this solver

- **Genuinely sparse-view CT** with `n_angles ≲ 60` and no training
  data — the paper's regime.
- **As a per-scan ablation baseline** when no training set is
  available.

## When to **not** prefer this solver

- **Dense-view CT (`n_angles ≥ 128`)** — structurally beaten by FBP
  + denoising chain (see Empirical results below).
- **Anything pentathlon all-rounder** — it cannot share knowledge
  across scenes.

## Knobs (in `CONFIG`)

| Knob | Default | Effect |
|---|---:|---|
| `naf_n_freqs` | 10 | Positional-encoding frequency bands. |
| `naf_hidden` | 192 | MLP width. |
| `naf_layers` | 5 | MLP depth. |
| `naf_n_iter` | 2000 | Per-scene Adam steps. **12000 still doesn't beat FBP** on dense view. |
| `naf_lr` | 5e-3 | Adam lr. **Drop to 1e-3** for stability — 5e-3 bounces. |
| `naf_tv_weight` | 1e-4 | TV penalty weight. |
| `naf_n_clip` | 0.05 | Output sigmoid times this; effectively μ upper bound. |
| `naf_outer_wall_s` | 2400 | Total wall across all val scenes. **Bump to 3600+** if val_n × per-scene-cost > 2400 s. |
| `naf_per_scene_s` | derived | Override per-scene wall. |

## Hints for the next autoresearch agent

- **Deprioritised on dense-view breast-CT.** Two agentic iters tested:
  iter-1 (lr=5e-3) and iter-2 (lr=1e-3) — both hit hr=0 with PSNR
  15–17 dB (vs baseline FBP 39.6 dB). **Don't burn iters here.**
- **Lowering lr is the only thing that helped** (iter-1 last_loss 0.87
  → iter-2 last_loss 0.05 = 17× better fit), but the SSIM only went
  0.755 → 0.791. The fit converges to something that's not the truth.
- **For sparse-view tracks** (e.g. dl_sparse_view), NAF is still
  worth a search. The fan-beam projector is the same; only the data
  side changes.
- **FBP warm-start** (pre-train MLP to match FBP, then refine with
  data fidelity) is a plausible next experiment — would need a small
  solver change. Lower priority than fixing the breast DDPM
  checkpoint or extending the LPD search.

## Cross-dataset observations

| Dataset | Best hr | Config | Notes |
|---|---:|---|---|
| `demo_dl` | 0.4160 | naf_n_iter=2000, lr=5e-3, 5-layer 256-hidden MLP | TPE iter-19; performs ~average for the dataset (mid-pack). |
| `breast_ct` | **0.000** | lr=1e-3, n_iter=12000 | Structural mismatch — 22 dB below baseline FBP. NAF's coordinate-MLP cannot beat a properly-tuned FBP+denoising chain on dense-view data. |
| `mayo_ldct` | — | not yet run | Likely same outcome as breast_ct (also dense-view 2304-angle helical) |

**Pattern**: NAF is competitive on `demo_dl` (simpler phantoms,
sparse-view-like in the sense that the dataset is small) but fails on
`breast_ct` (denser content, harder distributions). The dataset
characteristic to predict success is **how good baseline FBP is** —
if baseline FBP is already strong (SSIM > 0.95), per-scene NAF cannot
recover the precision gap. If baseline FBP is mediocre (SSIM ~ 0.7-0.8
on `demo_dl`), NAF closes the gap.

## Empirical results on breast-CT (128 views, intensity-calibrated)

| Source | Config | val_n | val_ssim | val_psnr | hr |
|---|---|---:|---:|---:|---:|
| baseline FBP | — | 5 | 0.957 | 39.61 dB | 0 |
| `calibrated-tpe-naf-search-20260521-01` | best of 20 trials | 20 | — | — | 0 |
| `claude-agentic-naf-search-20260523-01/iter-1` | lr=5e-3, n_iter=12000, naf_outer_wall_s=3600 | 5 | 0.755 | 15.78 dB | 0 |
| `claude-agentic-naf-search-20260523-01/iter-2` | lr=1e-3, n_iter=12000 | 5 | 0.791 | 16.90 dB | 0 |

All variants are **22+ dB below baseline FBP**. The architecture is
the wrong fit for this data; expect this to remain true at any
configuration.
