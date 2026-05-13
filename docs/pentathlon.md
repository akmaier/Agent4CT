---
title: Pentathlon
description: Five CT challenges, headroom-recovered scoring, train / val / test budget.
---

## Why five

A single challenge tells you whether a method works in that narrow setting. Five tells you whether the method **generalises**. The Agent4CT autoresearch loop runs on each challenge independently first; then a final pass searches for an **all-rounder configuration** that maximises the unweighted mean of per-challenge scores.

## Headroom-recovered scoring

The five metrics aren't directly commensurable (SSIM ∈ [0, 1]; PSNR in dB; RMSE in HU vs normalised intensity; CT-MAR's 8-metric IQ aggregate). To average them we use the fraction of the gap between a trivial baseline and an oracle that a solver closes:

```
score_c   = (metric_c − baseline_c) / (oracle_c − baseline_c)   ∈ [0, 1]
pentathlon = mean(score_c for c in challenges)
```

| Challenge | Baseline (score 0) | Oracle (score 1) |
|---|---|---|
| Mayo LDCT | Low-dose FBP, no denoising | High-dose FBP (clean sinogram) |
| DL-Sparse-View | Sparse-view FBP, no learning | RMSE = 0 against exact truth |
| TrueCT | Uncorrected FBP | Mono-energetic phantom truth |
| CT-MAR | Uncorrected FBP (metal in) | Metal-free phantom recon |
| DL-Spectral | Per-energy FBP composite | Exact tissue maps |

A negative score means the solver is worse than doing nothing on that challenge; a score above 1 means the solver beats the oracle on that test split (possible with noisy oracles). The Pentathlon score is the unweighted mean — five equal-weight events, just like decathlon.

## Subset sizes (5-minute iteration budget)

Sized for ~ 10–20 epochs in 5 minutes on a single A6000-class GPU. Test sets are **never** read during the loop.

| Challenge | Train | Val (in-loop) | Test (held) |
|---|---:|---:|---:|
| Mayo LDCT | 200 slices (Wagner 4 scans × 50) | 50 slices | 250 slices |
| DL-Sparse-View | 400 phantoms | 100 | 100 |
| TrueCT | 200 slices (stratified by pathology) | 50 | 100 |
| CT-MAR | 400 cases (stratified by anatomy) | 100 | 100 |
| DL-Spectral | 200 cases × 2 kVp | 50 | 100 |

## Three budgets, three journals

| Budget | When | Subset | Wall-time | Where it lands |
|---|---|---|---|---|
| **Iteration** | every step | the per-challenge subset above | 5 min on 1 GPU | `docs/runs/<run>/results.tsv` |
| **Stage** | every 30 iter | 3× the iteration subset | ~ 1 h on 1 GPU | `docs/runs/<run>/stages.tsv` |
| **Final** | once at end | full test set | unbounded | `docs/runs/<run>/final.json` |

## Anti-overfit rules (every agent sees these)

1. **Parameter budget vs. training-set size.** Keep total trainable parameters under roughly `10 × train_n × pixels_per_sample` unless you have a clear regularisation story.
2. **Stage gap is the overfitting signal.** If `stage_score ≪ iter_score` on the last stage, the next iteration *must* shrink the model, increase weight decay, or augment more.
3. **Augmentation is cheap.** Random crop, intensity jitter, flip, small affine warps — try these before bigger architectures.
4. **Self-supervised first.** Noise2Inverse-style consistency losses extend the effective training-set size without new labels.
5. **Don't game the val set.** Improvements under 1 % across three iterations are noise.
6. **One change per iteration.** If two things change and val moves, you don't know which one did it.
7. **Cite the iteration that inspired you** in the commit message.

## Why this design

- **5-minute iteration budget** forces the agent to trade architecture against throughput. No "just train longer" cheats.
- **30-iteration stage** catches overfitting cheaply: if the iteration metric kept rising but stage stalls, the agent has memorised its 200–400-sample subset.
- **Headroom recovered** lets the five challenges sit on the same scale; the Pentathlon average is then meaningful instead of accidentally weighted by the metric with the largest dynamic range.
- **Five agents in parallel + shared scratch pad** is the bit beyond Karpathy: if "AdamW with weight_decay=1e-4 beats Adam" appears on three challenges, the remaining two agents see that in their context window before their next edit.
