---
title: BreastCT-Noise-Retrained leaderboard
description: Retrained-noisy board — each trainable Breast-CT solver's clean-best config retrained on Poisson-noised (I0=100k) training data, scored on the noisy test set. The clean ranking largely returns.
---

# BreastCT-Noise-Retrained leaderboard — matched-noise retraining

The [BreastCT-Noise board](breast_ct_noise.html) is a **no-retrain** robustness
test: clean-optimal models are re-evaluated on noisy inputs they never trained on,
and the ideal-data ranking nearly inverts. This third board asks the converse
question: **if the noise is known at training time, does the clean ranking return?**

**Setup (matched-noise retraining).** For each *trainable* solver we take its
noiseless best-iteration configuration — unchanged architecture and
hyper-parameters — and **retrain it from scratch** on the same Breast-CT training
set with matched Poisson noise added (**I0 = 100 000 photons/pixel**, deterministic
seed), select the checkpoint on the noisy validation split, and score it on the
**same 200 noisy held-out test cases** as the no-retrain board. The ground truth
stays clean and the FBP baseline is recomputed from the noisy sinogram, so the
retrained and no-retrain boards share the same `hr` scale and are directly
comparable. Classical / per-scene / zero-shot solvers carry no trainable weights
and are carried over unchanged from the no-retrain board. Only the frozen
clean-best config was retrained (no hyper-parameter re-search), so these scores are
a conservative lower bound.

## Leaderboard — every solver (rendered from the registry)

Rendered live from `docs/runs/index/leaderboard.json`
([`build_registry.py`](../../scripts/build_registry.py) /
[`build_breast_retrain_board.py`](../../scripts/build_breast_retrain_board.py)),
same run-ids as Breast-CT but scored after matched-noise retraining.

<div data-leaderboard="breast_ct_noise_retrain">loading leaderboard…</div>

<script src="../assets/table.js"></script>
<script src="../assets/leaderboard.js"></script>

## The finding — the collapse is a distribution mismatch, not a deficit

Retraining on matched noise **largely restores the clean ranking**. Its Spearman
correlation with the noiseless board rises from **ρ ≈ 0.16** (no retrain, ~3 % of
the variance) to **ρ ≈ 0.65** (retrained, ~42 %).

- **The collapsed noiseless champion recovers to #1.** `dual-domain-supervised`
  goes **hr 0.00 → 0.958** (SSIM 0.35 → 0.996) and reclaims rank 1 — above its own
  noiseless score, because `hr` is measured against the worse noisy FBP baseline.
  Its collapse was a train/test distribution mismatch, not an architecture limit.
- **The supervised unrolled family returns to the top tier** (ITNet v1/v2/v3,
  U-Swin, hr ≈ 0.94).
- **`learned-primal-dual` barely moves** (0.935 → 0.936): already robust because it
  keeps the forward operator in the loop, it had nothing to regain.
- **Two groups do not recover:** the *constrained* fast-diffusion variants stay at
  hr 0 (their hard fit to the clean forward model does not survive noisy training),
  and a few solvers dip slightly (e.g. Hammernik-2017, 0.70 → 0.61) because only the
  fixed clean-best config was retrained, not re-searched.

**Take-away for the paper:** a single-distribution leaderboard does not predict
robustness to *unseen* noise — but that brittleness is largely repairable once the
noise is known and trained for. A complete benchmark should report both conditions:
transfer to an unseen perturbation, and performance after matched retraining.

## Methodology

Noise staging (train + val + test):
[`data/stage_breast_noise.py`](../../data/stage_breast_noise.py) (deterministic,
seed 20260709). Retraining + scoring:
[`scripts/score_breast_noise_retrain.py`](../../scripts/score_breast_noise_retrain.py)
→ `score_breast_testset.py --worker --noise-i0 100000 --retrain`. Board assembly +
rank correlation: [`scripts/build_breast_retrain_board.py`](../../scripts/build_breast_retrain_board.py)
and [`scripts/breast_board_rank_corr.py`](../../scripts/breast_board_rank_corr.py).
