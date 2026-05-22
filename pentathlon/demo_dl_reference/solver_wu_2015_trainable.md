# `solver_wu_2015_trainable.py` — Wu 2015 with end-to-end trainable scalar hyperparams

Companion design doc. For the original non-trainable algorithm see
`solver_wu_2015.py` (still the recommended baseline if no clean
target is available — the trainable version overfits on small data).

## What it is

The classical **Wu et al. 2015** aliasing-free reconstruction wrapped
with `nn.Parameter` on its previously hand-tuned scalar knobs. Same
4-band radial-frequency split + symmetric motion-compensated
interpolation + soft-thresholded residual refinement, but the band
weights, sigmoid roll-off, soft thresholds, and per-iter residual
blends are now learnt end-to-end with **supervised L2 against the
clean phantom + non-negativity penalty** on the full 128-view sino.

Trainable parameters (defaults `n_bands=4`, `n_outer=2` → 10
scalars):

| Parameter | Shape | Role | Init |
|---|---|---|---|
| `log_band_scale` | (n_bands,) | per-band multiplicative scale before sum-norm | `log(1.0)` |
| `sigmoid_slope` | () | band roll-off steepness around Nyquist | 10.0 |
| `sigmoid_offset` | () | Nyquist threshold (f·ΔR == offset) | 1.0 |
| `log_soft_thresh` | (n_outer,) | per-iter soft-threshold on residual reco | `log(0.0015)` |
| `residual_blend` | (n_outer,) | per-iter weight on the residual that's added back | 1.0 |

Hard clamps (added in iter-10 of the autoresearch loop, see below)
keep each scalar inside a sane range and are configurable via
`wu_band_scale_{min,max}` etc.

## Design considerations

- **Built 2026-05-22** as the autoresearch experiment: "can we learn
  the scalar hyperparameters of a classical CT algorithm
  end-to-end with backprop through the differentiable forward
  projector?". Yes; with the right lr and epoch budget it lands on
  hr ≈ 0.22 on breast-CT, comparable to a 6-param dual-domain BF.
- **PYRO-NN PyTorch backend** is fully differentiable, so
  `proj.forward_project(g)` inside the outer iteration sends
  gradients back to all trainable scalars (including the band
  weights and sigmoid params used in `_aliasing_free_fbp`).
- **`feature_preserving_interp` shift selection is
  piecewise-constant** — the gradient through the `torch.where`-based
  per-pixel best-shift pick is zero in places. The selected
  midpoint values themselves carry gradient through their linear
  combination of inputs, so the algorithm still has a useful
  gradient signal end-to-end (empirically the optimizer converges).
- **Structural integers stay fixed** (`wu_n_bands`, `wu_n_outer`,
  `wu_motion_range`, `wu_motion_window`) — they would need an
  outer search to tune; the trainable scalars are the continuous
  knobs only.
- **AdamW with `weight_decay`**: added as a knob to pull scalars
  toward 0. Doesn't help on this problem (wd=1e-3, 1e-2 both too
  weak; larger wd would pull scalars away from sane init).

## Strengths

- **Truly tiny — 10 trainable parameters total** on the default
  config. Trains in 100 s for 5 epochs on a single RTX 6000.
- **Fully interpretable.** The per-epoch log prints every scalar; you
  can read the algorithm's internal state directly. Watching
  `band_scale = [1, 1, 1, 1] → [0.8, 2.3, 0.08, 0.06]` tells you
  "the optimiser decided to amplify the low-mid band and kill the two
  highest-frequency bands" — that's a physically meaningful
  interpretation, not a black-box weight tensor.
- **Inherits Wu 2015's physical priors** (radial-band Nyquist
  threshold, motion-compensated interpolation, soft-thresholded
  residual refinement) — these are domain-knowledge constraints that
  a fresh neural net would have to rediscover.

## Weaknesses

- **Structural overfit.** The 10-parameter optimisation landscape on
  400 phantoms has a *very* attractive bad-minimum where
  `band_scale[1]` runs away to 3-9× and bands 3,4 collapse toward
  zero. This appears in every run that trains past ~12 epochs (see
  iter-3, iter-6/7/8, iter-9, iter-10 in the autoresearch loop).
- **More data makes it worse.** train_n=2000 (5× more breast
  phantoms) pushed `band_scale[0]` to **53×** — the optimizer
  becomes more confident in the overfit pattern with more data.
- **Hard parameter clamps don't help.** When `band_scale` is capped
  at [0.1, 3.0], the optimizer pins it at the corners (iter-10).
  Just shifts the failure mode.
- **L1 loss doesn't change the trajectory.** Same band-runaway
  pattern (iter-8) as MSE.
- **Discrete-component gradient cliffs.** The motion-compensated
  shift selection blocks gradient at the cost-comparison step. Not
  proven to be a blocker, but it's a known non-smoothness.
- **Ceiling at hr ≈ 0.22** on breast-CT (iter-2 best). The
  autoresearch loop ran 10 configurations and could not break past
  this.

## When to prefer this solver

- **Algorithm-grounding studies** — "how much can you push a
  classical algorithm with end-to-end-learnable knobs?"
- **Interpretable medical pipelines** where each learnt parameter
  must be inspectable and traceable to a physical meaning.
- **Tiny-data regimes** — works with 10s of train pairs (though
  ceiling is also lower).

## When to **not** prefer this solver

- **Headroom hunting**. The U-Net supervised variant hits hr=0.81
  with 466 k params; the BF supervised hits hr=0.25 with 18 params.
  This solver hits hr=0.22 with 10 params — *worse per parameter
  than the BF variant* because the discrete components limit what
  gradients can do.
- **As a baseline against itself** without supervised data — the
  non-trainable `solver_wu_2015.py` is the better default for
  unsupervised settings.

## Knobs (in `CONFIG`)

| Knob | Default | Effect |
|---|---:|---|
| `wu_n_bands` | 4 | Number of trainable band scales. Paper uses 8 — try this for higher ceiling. |
| `wu_n_outer` | 2 | Outer iterations. **n_outer=3 caused complete collapse** in iter-5. |
| `wu_motion_range`, `wu_motion_window` | 5, 2 | Integer, hand-tuned, not trainable. |
| `epochs` | 10 | **Sweet spot is exactly 10** at lr=1e-3. Past 12 → overfit. |
| `lr` | 1e-3 | **Critical**: lr=1e-2 (iter-1) collapses the high-freq bands. |
| `weight_decay` | 0.0 | AdamW wd. wd=1e-2 was too weak; larger probably pulls too hard. |
| `loss_base` | "mse" | "mse" or "l1". L1 doesn't change qualitative behaviour. |
| `wu_{band_scale,blend,soft_thresh}_{min,max}` | sane caps | Hard clamps. Pin corners of failure rather than fix it. |

## Hints for the next autoresearch agent

The 10-iter loop in `breast-ct-claude-agentic-wu-2015-l2-search-20260522-01`
has already tested:
- lr sweep (1e-2, 1e-3, 5e-4)
- epoch sweep (5, 8, 10, 15, 20)
- n_outer sweep (2, 3)
- weight_decay (0, 1e-3, 1e-2)
- L1 vs MSE
- train_n=400 vs 2000
- hard clamps

**Don't redo these.** They all converge to or below iter-2's
hr=0.219.

**What might break the ceiling** (not yet tested):
1. **Validation-based early stopping** with best-checkpoint
   restoration. iter-2 won because epochs=10 happened to land on the
   sweet spot — a proper early-stop generalises that to other
   datasets. Requires a small solver edit: track best val every K
   batches, save state_dict, restore at end.
2. **Softmax-parametrised band weights** with sum constrained to a
   fixed total. Removes the band-runaway DoF *by construction*.
   `weights = softmax(logits) * n_bands` keeps the total at
   `n_bands` and forces band reallocation rather than amplification.
3. **Hybrid `Wu + image-domain BF tail`**. The Wu does aliasing-free
   + residual refinement; chain a 6-param `BilateralFilterStack` at
   the output. Probable headroom: ~hr 0.30 (BF tail alone gets 0.21;
   the two might compose).
4. **`wu_n_bands=8`** (Wagner's value) for finer spectral control.
   Costs +4 params; might or might not help given the overfit
   pattern.
5. **Per-pixel soft threshold via tiny conv prior** (CNN that
   outputs a per-pixel threshold map). Adds parameters but in a
   physics-meaningful place.
6. **Curriculum from non-trainable Wu**: warm-start parameters at
   paper defaults, freeze sigmoid + band_scale for the first 5
   epochs, then unfreeze. Constrains where the optimization can
   start.

## Empirical results on breast-CT (128 views, intensity-calibrated)

| Iter | Change vs default | val_psnr | val_ssim | hr |
|---|---|---:|---:|---:|
| 1 | lr=1e-2, ep=5 (init) | 36.65 dB | 0.958 | 0 |
| **2** | **lr=1e-3, ep=10** | **41.74 dB** | **0.969** | **0.219** |
| 3 | ep=20 | 39.72 dB | 0.958 | 0.015 |
| 4 | ep=8 | 40.64 dB | 0.971 | 0.114 |
| 5 | n_outer=3 | 25.14 dB | 0.656 | 0 |
| 6 | wd=1e-3 | 40.81 dB | 0.963 | 0.131 |
| 7 | wd=1e-2 | 40.65 dB | 0.963 | 0.114 |
| 8 | L1 loss | 40.81 dB | 0.964 | 0.131 |
| 9 | train_n=2000 | 38.63 dB | 0.944 | 0 |
| 10 | hard clamps | 36.55 dB | 0.940 | 0 |

Non-trainable `solver_wu_2015.py` (TPE-searched): hr 0.042 (best of
20 trials, slug `breast-ct-calibrated-tpe-wu-search-20260521-01`).
So end-to-end-trainable Wu is a 5× hr improvement over hand-/
TPE-tuned Wu, validating the machinery — but the ceiling is set by
the algorithm's expressivity, not by the optimiser.
