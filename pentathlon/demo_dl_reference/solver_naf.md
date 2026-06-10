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
| `mayo_ldct` | **0.0202** | Step-2 iter-1: n_freqs=6, hidden=192, layers=5, n_iter=2000, train_n=50 | **Surprise — NAF clears Mayo baseline!** Per-scene MLP finds enough signal at 2304 angles. Step-3 TPE found a worse config (0.0131); Step-2 iter-1 stays as the rank-11 entry. |

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

## 2026-06-08/09 — Mayo: NAF clears baseline (surprise positive)

NAF surprised the breast-CT "wrong inductive bias for dense view"
verdict on Mayo-LDCT: **iter-1 of the Mayo Step-2 agentic loop
landed hr=0.0202** (slug `mayo-ldct-claude-agentic-naf-search-20260603-01`,
n_freqs=6, hidden=192, layers=5, n_iter=2000, train_n=50, val_n=5,
0.143 M params, SSIM 0.5395, PSNR 13.98 dB vs baseline 12.59 dB).

The 2304-angle Mayo sino has enough redundant view coverage that the
per-scene MLP can fit the noisy projections coherently, even though
breast-CT's 128 dense views were too few. The dataset characteristic
to predict NAF success refines from breast-CT verdict: it's not
just "how good is baseline FBP" — it's **"how many independent view
constraints per voxel"**. Mayo's 18× more angles than breast-CT means
NAF has 18× more constraints per scene to fit, and clears baseline
despite a low absolute SSIM.

### NAF plateau verdict (Step-2)

Mayo iter-2/3/4 explored deeper configs:
- iter-2 (n_iter 2000 → 3000): hr=0.0010, SSIM dropped 0.5395 → 0.4843.
- iter-3+: similar regressions.

NAF **overshoots** beyond 2000 iters on Mayo — the implicit-field MLP
starts hallucinating high-frequency detail that doesn't match the
truth slab. iter-1's `naf_n_iter=2000` is the local optimum.

### Mayo Step-3 TPE — found WORSE config than Step-2 iter-1

Job 762923 (`mayo-ldct-2d-calibrated-tpe-naf-search-20260608-01`) ran
20-trial TPE with Mayo clamps. **Final best hr=0.0131** (TPE iter-6,
n_freqs=8, hidden=192, layers=4, n_iter=3909, lr=0.009) — **WORSE
than Step-2's 0.0202**.

TPE explored 12 configs deeper than iter-6 (layers=5-6, n_freqs=12-14)
expecting them to improve; all regressed. Then explored the
(8/192/4) corner (matching the iter-6 config family) but only hit
hr=0.0073/0.0059/0.0027 — TPE's random-search startup never landed
the exact Step-2 winner config.

**Verdict on Mayo**: Step-2 iter-1 stays as the rank-11 entry at
hr=0.0202. NAF on Mayo has a **narrow working corner** that TPE's
broader exploration didn't reproduce. Lesson: when an agentic
iter-1 lands a positive result and subsequent iters all regress,
the winner config may be on a knife-edge — TPE's exhaustive
exploration may not find it back.

### Cross-dataset NAF record (final)

| Dataset | hr | n_iter (best) | params (M) | Verdict |
|---|---:|---:|---:|---|
| `demo_dl`   | 0.4160 | 2216 | 0.270 | rank 11. Strong on simpler synthetic substrate. |
| `breast_ct` | 0.000 | 12000 | — | **STOP** — 22+ dB below baseline FBP on dense-view 128-angle. |
| `mayo_ldct` | **0.0202** | 2000 | 0.143 | **rank 11** (surprise positive). 2304-angle redundancy carries NAF over baseline by a thin margin. |
