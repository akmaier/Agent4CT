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

## Cross-dataset observations

| Dataset | Best hr | Config | Notes |
|---|---:|---|---|
| `demo_dl` | 0.2295 | TPE iter-18; non-trainable Wu | The non-trainable `solver_wu_2015.py` (calibrated TPE) reaches 0.2295 — Wu's classical algorithm with hand-tuned params. |
| `breast_ct` | 0.2189 | lr=1e-3, ep=10, n_bands=4 (10 params trainable) | iter-2 of the 10-iter agentic loop. Non-trainable Wu only reaches 0.0425 here; **trainable variant is a 5× improvement** but hits a hard ceiling at 0.22. |
| `mayo_ldct` | — | not yet run | Wu's algorithm is sparse-view-oriented; expected to be modest on dense-view Mayo. |

**Pattern**: trainable scalars on top of a classical algorithm gives a
small but real boost on `breast_ct` (5× improvement vs the
non-trainable version). On `demo_dl` the non-trainable version is
already at the ceiling — adding trainable scalars wouldn't help. The
algorithm's expressivity, not the optimisation, is the limit.

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

## 2026-06-09 — breast-CT TPE +45% lift over agentic

Job 762955 (slug `breast-ct-calibrated-tpe-wu-2015-trainable-search-20260609-01`)
ran a full 20-trial Optuna TPE seeded from the agentic iter-2 winner.
**Final best hr=0.3170** — a **+45% lift over the agentic 0.2189**.

Winner cluster (TPE rediscovered the optimum 5+ times across
iter-8/12/13/15/16, all in the hr=0.31-0.32 band):

| Knob | Agentic best | TPE best | Δ |
|---|---|---|---|
| `wu_n_bands` | 4 | **6** | +2 (bigger filter bank) |
| `wu_n_outer` | 2 | 2 | unchanged |
| `wu_motion_range` | 5 | **8** | +3 (wider motion search) |
| `wu_motion_window` | 1 | **2** | +1 |
| `epochs` | 10 | 12-13 | +2-3 |
| `lr` | 1e-3 | **1.1e-4** | **10× smaller** |
| `soft_thresh` | 1e-3 | ~1e-3 | unchanged |
| `lambda_neg` | 1.0 (default) | **0.7** | -0.3 |
| `loss_base` | mse | mse | unchanged |

**Key TPE finding**: the agentic search clamped `lr` to log(1e-4, 5e-3)
based on early demo-DL/breast results; the working corner is at the
BOTTOM of that range. The agentic-loop neighbourhood walk visited
`lr=1e-3` and never re-explored toward 1e-4. **TPE's exhaustive
exploration of the lr axis found a 10× lower-lr regime that lifts hr
by +45%.**

The structural knobs (`wu_n_bands=6`, `wu_motion_range=8`,
`wu_motion_window=2`) also moved by +50% in TPE — bigger filter bank
+ wider motion estimation. Suggests the algorithm has more capacity
to absorb than the agentic seed gave it.

### Cross-dataset Wu 2015 trainable record (updated 2026-06-09)

| Dataset | hr (best) | Source | Notes |
|---|---:|---|---|
| `breast_ct` | **0.3170** | TPE 762955 (2026-06-09) | **+45% over agentic.** New rank 10 on breast-CT (was rank 13). Robust cluster across 5 TPE iters. |
| `demo_dl` | 0.2288 | TPE `demo-intensity-calibrated-tpe-wu-2015-trainable-search-20260601-01` | rank 19 on demo-DL. Wu trainable is mid-pack on the easier synthetic dataset. |
| `mayo_ldct` | **0** | Mayo Step-2 iter-1/2 (n_bands=4, ep 3→6 plateau) | **STOP**. 10 trainable scalars hit low-capacity ceiling at SSIM≈0.34 (PSNR 12.37). Mayo's wider dynamic range can't be matched by 10 scalars. |

**Lesson for the autoresearch loop**: when agentic finds an
above-baseline plateau, ALWAYS run a TPE refinement. Even on
"already-done" solvers, TPE consistently finds corners the agentic
random walk missed — Wu trainable's +45% lift is now the third
biggest TPE-vs-agentic gain logged (after DD-UNet sup Mayo +191% and
USwin Mayo +75%).
