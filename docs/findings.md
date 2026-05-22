---
title: Cross-cutting findings
description: Substantive learnings from the autoresearch loop that span multiple iterations, agents, or sessions. Newest first.
---

This file is the cross-agent handoff log. Per-iteration rationale belongs in
`docs/runs/<slug>/iterations/iter-NNNN/observation.json`. Stage-check
verdicts belong in `docs/runs/<slug>/stages.tsv`. Things that belong here:
**facts about the substrate or methodology that the next agent should not
have to re-discover.**

## 2026-05-22 — Breast-CT DD-UNet supervised L2 reaches PSNR 54.25 dB (hr 0.81)

Following the DD-BF supervised-L2 finding (next entry), ported the same
loss change to the U-Net dual-domain solver
(`solver_dual_ddomain_supervised.py`, 466 k params at c=16):

| width | params | val_ssim | val_psnr | hr |
|---|---:|---:|---:|---:|
| c=8  | ≈120 k | 0.9979 | 52.53 dB | 0.771 |
| c=16 | 466 k  | 0.9986 | 54.25 dB | 0.812 |

This is 14.5 dB above baseline FBP — saturation level for the val set.
The previous N2I-trained version of the same architecture at c=16
peaked at val_ssim 0.967 / val_psnr 38.0 dB / hr=0. **Loss is the
dominant lever; doubling U-Net width gains only +1.7 dB on top of the
supervised-L2 change.** Mark this for the pentathlon all-rounder: any
dual-domain solver competing on breast-CT or other dense-scan
challenges should default to MSE-vs-clean if clean targets are
available; reserve N2I for sparse-view or no-truth regimes.

### Naming convention (2026-05-22)

The N2I solvers now carry the training scheme explicitly in their
filename:

| Old name | New name | Loss |
|---|---|---|
| `solver_dual_ddomain.py`           | `solver_dual_ddomain_n2i.py`           | Noise2Inverse (self-sup) |
| `solver_dual_ddomain_bilateral.py` | `solver_dual_ddomain_bilateral_n2i.py` | Noise2Inverse (self-sup) |
| —                                  | `solver_dual_ddomain_supervised.py`           | Supervised L2 vs clean phantom |
| —                                  | `solver_dual_ddomain_bilateral_supervised.py` | Supervised L2 vs clean phantom |

The bilateral solvers also gained `proj_n_bf` / `img_n_bf` config keys
(Wagner §3.2 BF chain). On the supervised-L2 BF variant, n=3 BFs per
domain pushed hr from 0.214 (n=1) → 0.230 (n=1, bigger kernel) →
0.248 (n=3). The image-domain cascade differentiated into a
multi-scale stack (BF1 σ_x≈1.10, BF2 σ_x≈1.18, BF3 σ_x≈1.33); the
projection-domain stack stayed locked-symmetric (gradients too small
to break the identical init).

### Trainable Wu 2015 (`solver_wu_2015_trainable.py`) — 10-iter autoresearch loop

Built the variant 2026-05-22 and ran a 10-iteration agentic search
(slug `breast-ct-claude-agentic-wu-2015-l2-search-20260522-01`):

| iter | change vs prior | val_psnr | hr | observation |
|---|---|---:|---:|---|
| 1  | baseline cfg (lr=1e-2, ep=5)        | 36.65 | 0.000 | lr too high, killed high-freq bands |
| 2  | **lr=1e-3, ep=10**                   | **41.74** | **0.219** | **sweet spot — best of the search** |
| 3  | ep=20                                | 39.72 | 0.015 | overfit; train loss 0.016→1e-4 but val drops |
| 4  | ep=8                                 | 40.64 | 0.114 | undertrained |
| 5  | n_outer=3                            | 25.14 | 0.000 | collapse — blend went [1.87, 2.04, 1.03] |
| 6  | wd=1e-3 (AdamW)                      | 40.81 | 0.131 | wd too weak; trajectory ≈ no-wd |
| 7  | wd=1e-2                              | 40.65 | 0.114 | still too weak |
| 8  | L1 loss (`loss_base="l1"`)           | 40.81 | 0.131 | same band-runaway pattern |
| 9  | train_n=2000 (5× more data)          | 38.63 | 0.000 | data made overfit *worse* — band_scale[0]→53× |
| 10 | hard clamps on band_scale / blend / soft_thresh | 36.55 | 0.000 | optimizer pinned to clamp corners — new pathology |

**Ceiling**: hr ≈ 0.22 (iter-2). No further config or
architectural-clamp move broke past it. The pattern across iters:
**the optimisation landscape consistently pushes band_scale[1]
upward and bands 3-4 toward zero** — a textbook noise-suppression
solution that does well on the 400-phantom train set but fails on val
because the per-band amplification it learns is dataset-specific
overfit. Hard clamps don't help (corners become the new attractor);
more data makes it worse (the optimizer becomes more confident);
weight decay at reasonable magnitudes is too weak; L1 doesn't change
the trajectory.

What would likely break past hr=0.22:
- **Validation-based early stopping** with best-checkpoint restoration.
  Iter-2 happened to land in the sweet spot because epochs=10 was
  exactly right; a proper early-stop would generalise that.
- **Softmax-parametrised band weights** (sum constrained to a fixed
  total) instead of independent log-scales — removes the band-runaway
  degree of freedom by construction.
- **Hybrid Wu + BF tail** — Wu does aliasing-free + residual cleanup,
  a 6-param image-domain BF tail (supervised L2) cleans up remaining
  streaks. Likely lands near hr ≈ 0.30 based on the 6-param BF
  reference (hr=0.21).
- **Bigger algorithmic capacity** — make `wu_n_bands` a config knob
  that goes to 8 (the paper's value); learn per-pixel soft thresholds
  via a tiny CNN-prior; etc.

Even with the ceiling, the trainable Wu (10 params, hr=0.22) matches
the 6-param DD-BF supervised (hr=0.23) and the 18-param 3×3 BF stack
(hr=0.25). For a *classical* CT algorithm with end-to-end trainable
knobs and no neural net, that's a respectable result and proves the
machinery works.

## 2026-05-22 — Breast-CT DD-BF: Noise2Inverse is the bottleneck; supervised L2 + full 128 views unlocks hr ≈ 0.21

The dual-domain bilateral-filter (DD-BF) solver
(`solver_dual_ddomain_bilateral_n2i.py`, formerly
`solver_dual_ddomain_bilateral.py` — renamed 2026-05-22 to make the
training scheme explicit) had been stuck at hr=0 across all TPE and
agentic iterations on breast-CT (best val_ssim ≈ 0.957 ≤ baseline FBP
0.957). The U-Net dual-domain variant (`solver_dual_ddomain_n2i.py`,
formerly `solver_dual_ddomain.py`) had the same problem at every U-Net
width
tested (c=4, c=8, c=16): SSIM stuck just above baseline FBP, PSNR
*below* baseline by ~1.7 dB.

Root cause confirmed empirically (iters 1–5 on
`breast-ct-claude-agentic-dual-domain-{,bf-}search-20260521-01`): the
**Noise2Inverse split-view MSE** (`DualDomainPipeline.training_step`,
`ddssl_ldct/training.py`) feeds the model the FBP of one 64-view
half-set as a "target" for the other half-set. On dense breast scans
the half-view FBP has an irreducible noise floor, so minimising MSE
against it *encourages over-smoothing*. The bilateral filter's image-
domain spatial sigma grew 0.5 → 1.08 in 2 epochs (iter-1) and 0.3 →
0.87 in 5 epochs (iter-2), saturating the kernel and over-blurring
fibroglandular detail. Increasing U-Net width didn't help — the loss
was pulling all architectures the same way.

Fix: new `solver_dual_ddomain_bilateral_supervised.py` — full 128-view
forward pass, plain MSE against the *clean* phantom + non-negativity
penalty (using the existing `supervised_recon_loss`). Same 6-parameter
bilateral filter, same intensity calibration, same comparison panel.

Iter-1 result (slug
`breast-ct-claude-agentic-dual-domain-bf-l2-search-20260522-01`,
job 761594, 70 s, lme221):

| solver | params | val_ssim | val_psnr | headroom |
|---|---:|---:|---:|---:|
| baseline FBP | 0 | 0.957 | 39.74 dB | 0 |
| N2I DD-BF (best of 2 iters) | 6 | 0.957 | 37.50 dB | 0 |
| N2I DD-UNet (c=16, 5 ep) | 466 k | 0.967 | 38.01 dB | 0 |
| **Supervised-L2 DD-BF (this iter)** | **6** | **0.986** | **41.83 dB** | **0.21** |

A 6-parameter bilateral filter trained with the right loss beats a
466 k-parameter U-Net trained with the wrong loss by 3.8 dB PSNR on
the same dataset. Lesson: **loss formulation, not model capacity, was
the bottleneck for breast-CT dense-view denoising.** Mark this for the
all-rounder phase — the N2I assumption is good for sparse-view (where
the half-view FBP is so degraded the noise floor is below the signal)
but a poor fit for dense scans where FBP itself is already strong.

Caveat: supervised-L2 needs the clean phantom at train time, so this
variant is only fair against other supervised baselines (it cannot
play the "self-supervised" tag). Track separately in the dashboard.

## 2026-05-22 — Random / TPE search is NOT autoresearch (terminology fix)

Previous Agent4CT iterations conflated three distinct things by writing all
of them into `docs/runs/<slug>/...` with the same on-disk shape:

1. **Autoresearch (Karpathy-style)** — an *LLM* reads prior observations
   + literature, forms a hypothesis about *why* the last iter
   under/over-performed, proposes a *qualitatively different* next config
   (architecture / loss / optimiser / preprocessing) and writes a
   `rationale` that names the hypothesis. The slug carries
   `claude-agentic` (or another LLM family tag).
2. **TPE / Bayesian hyperparameter optimisation** — Optuna or similar
   samples the *same* parametric solver inside a *fixed* numeric bounding
   box, learning a surrogate over the box. No hypothesis, no architecture
   change, no qualitative jump. Slug should carry `tpe`, model field
   should be `optuna-tpe-*`.
3. **Random search** — uniform / log-uniform sampling inside the same
   box. Slug should carry `random-search`.

All three are *hyperparameter-tuning aids* for the agentic loop, but
**only #1 is the autoresearch loop itself**. Earlier scripts
(`scripts/tv_search_agent.py`, `scripts/learned_solver_search_agent.py`,
`scripts/tv_search_agent_standalone.py`, `scripts/record_demo_refs.py`)
called themselves "agentic" or "autoresearch" because they wrote into the
same `docs/runs/` schema; that wording is wrong and has been corrected.
Rule of thumb when reading any prior run: **if `agent` ends in
`-search` and `model` is `random-search` or starts with `optuna-`, the
run was a parameter sampler, not an autoresearch iteration.** The
dashboard groups them next to each other for comparison, which is fine
— just don't claim that "TPE found x" was an "agentic finding".

What an autoresearch iteration *must* include:
- A specific testable hypothesis in the `rationale` field, naming the
  expected mechanism (e.g. "img-domain BF σ ran away from 0.5 → 1.08 in
  iter 1, causing over-smoothing; cap with smaller kernel + lower lr").
- A change that the sampler could not have proposed by itself (a new
  loss term, a new architecture, an asymmetric hyperparameter constraint,
  a literature-cited prior).
- Inspection of the previous iter's `comparison.png` *by the agent*
  before the proposal (the agent should be able to name the artefact:
  radial blur, ringing, oversmoothing, streak, etc.).

If the proposal is "the same solver with the next sample in a box", it
should be filed as a TPE / random-search iteration, not an autoresearch
one.

## 2026-05-16 — B epochs=10 confirmed at stage: new high-water mark 0.6248

Followed up on B's "underfits at stage" finding by raising iter-base
epochs 8 → 10 (iter-147 at epochs=12 dropped to 0.5864 at iter scale,
so 10 is at or above the iter sweet spot). Ran a fresh B stage on
iter-146 (epochs=10). Stage hr=**0.6248**, up +0.46pp from the previous
B stage (0.6202 at epochs=8). Highest stage headroom on any agent.

Caveat: the stage sbatch auto-scales epochs to `max(base*2, 16)`, so
base=10 → 20 stage epochs vs base=8 → 16 stage epochs. The gain mixes
"iter-base config matters" and "more stage epochs". Operational
takeaway is clear: keep `epochs=10` for B going forward.

## 2026-05-16 — A and main iter scores are insensitive to weight_decay

After the iter-95 capacity-down failure, the next hypothesis was that
regularisation (wd-up) might close A and main's iter→stage gap. Tested:

| Agent | wd values tested | Samples | Iter hr range |
|---|---|---|---|
| A | 3e-5, 1e-4, 2e-4 | 5 | 0.6162–0.6167 (0.05pp) |
| main | 1e-5, 5e-5, 1e-4 | 5 | 0.6001–0.6005 (0.04pp) |

**Iter is essentially insensitive to wd** on both substrates. Doesn't
mean wd is neutral at stage scale — only iter. main stage v2 (wd=1e-4)
and a future A stage will test whether wd helps at scale.

## 2026-05-15 — capacity-down does NOT close A's iter-stage gap

Hypothesis from the first round of stages: A's -6.56pp gap was overfit due
to capacity (BF tail + 5 NAFNet blocks memorise the 400-case iter subset).
Tested by cutting capacity: `naf_n_bf` 10→6 + `naf_blocks` 5→4 (model
shrinks 0.050M → 0.040M params). Iter score dropped -1.39pp (expected if
overfit shrunk). **Then ran a second stage check on the capacity-down
config: stage hr DROPPED 0.5506 → 0.4991, gap GREW from -6.56pp to
-10.32pp.** Hypothesis falsified.

What we know now:
- The iter-stage gap on A is real but it is **not** "overfit due to
  capacity". Cutting capacity hurts both iter and stage.
- The gap must come from something else: maybe training-dynamics
  mismatch (12 epochs vs 6, larger val set), maybe optimiser/LR not
  tuned for longer training, maybe the BF tail's α parameters benefit
  disproportionately from longer training, maybe the iter-phase val
  set is genuinely easier than the stage val set.

Reverted A's solver to iter-86 KEEP base. Next probes for closing the
gap should look like **regularisation** (dropout, weight decay, schedule)
or **training-time fixes** (longer LR warmup, lower lr), not capacity.

## 2026-05-15 — iter vs stage gap signs differ by architecture

First four stage checks ran on DL-Sparse-View (with synthetic phantom data,
since real data isn't staged yet). Results:

| Agent | Architecture | Iter best | Stage hr (1h, 3× data, 2× epochs) | Gap | Direction |
|---|---|---:|---:|---:|---|
| main | NAFNet + SWA + BF tail | 0.6144 | 0.5928 | -2.16pp | overfits |
| A | NAFNet + BF tail | 0.6162 | 0.5506 | -6.56pp | severely overfits |
| **B** | resnet + AdamW + wd=5e-5 + batchnorm | 0.6120 | **0.6202** | **+0.82pp** | **underfits in iter, scales positively** |
| C | resnet + Adam + wd=0 + batchnorm + aug | 0.6102 | 0.5787 | -3.15pp | overfits |

**Real capability ranking at stage scale: B > main > C > A** (opposite of
the iter-best ranking). The iter phase's 400-case subset is small enough
to be memorised by NAFNet+BF families; the narrower resnet is undertrained
at iter scale.

**B vs C divergence is informative** — both resnet+batchnorm, but B
(AdamW + wd=5e-5) underfits while C (Adam + wd=0 + augs) overfits. The
augmentations in C don't add data diversity for sparse-view CT in a useful
way; Adam wd=0 removes the only weight regularisation. Cross-port `B`'s
optimiser to C (iter-98 in flight as of writing).

**Strategy implications:**
- Stop probing sub-pp KEEPs on B/C's 5-min iters — that's under-fitting
  noise, not signal.
- For A/main: iter probes should *reduce* capacity or add regularisation,
  not push capacity up.
- For B/C: iter probes should test things that benefit from more data
  (augmentations are not those things on this dataset).
- Stages are the only reliable signal until staged HDF5 data exists.

## 2026-05-15 — same-config variance on the 5-min iter substrate

Repeated same-config runs to characterise noise:

- **B (resnet)**: 6 same-config samples spanned 0.5774–0.6120, mean ≈0.600,
  3.5pp range. The iter-135 +0.14pp "KEEP" and iter-88 +0.22pp "KEEP" both
  fall inside this noise window.
- **main (NAFNet)**: 3 same-config samples spanned 0.6001–0.6144, mean ≈0.605,
  1.4pp range. The iter-102 best 0.6144 is the upper tail.

Likely cause: BatchNorm with `batch_size=1` + CUDA non-determinism.
`torch.manual_seed(42)` is fixed but BN running stats accumulate
differently across reorderings. Adding `torch.use_deterministic_algorithms(True)`
+ `CUBLAS_WORKSPACE_CONFIG=:4096:8` would reduce this; not enabled yet.

**Action:** treat any iter-phase KEEP < 0.5pp on B as variance. On
main/A/C the noise is tighter (probably <0.5pp) but still relevant.

## 2026-05-15 — silent substrate drift from un-reverted DISCARDs

When a DISCARD iter changes a CONFIG knob but the next iter doesn't
explicitly revert it, the substrate drifts. The "best so far" baseline is
only the headroom at the recorded KEEP iter, but the **current solver
state** can quietly be worse, masking opportunities.

Confirmed case: C iter-76 found a +0.29pp KEEP just by reverting
lr 8.5e-5→8.0e-5 — a change introduced in iter-70's DISCARD that was
never reverted. The solver had silently run below its iter-46 KEEP base
for ~30 iters.

**Action:** every ~15–20 iters without a KEEP, `diff` the current CONFIG
against the last KEEP iter's solver snapshot in
`docs/runs/<slug>/iterations/iter-NNNN/solver.py.txt`. Any knob that
drifted across a DISCARD without explicit revert is a candidate for
restoration. The current iter loop does not auto-revert.

## 2026-05-15 — data download status (see `data/README.md` for current state)

Live state of which challenge data is on the cluster lives in
[`data/README.md`](../data/README.md) §"Where the raw data lives on the
cluster". Summary at time of writing: CT-MAR + DL-Spectral are on disk,
Mayo Wagner subset is downloading slowly, DL-Sparse-View and TrueCT are
CodaLab-gated. **`stage_h5()` is not implemented for any fetcher**, so
the harness still trains on synthetic phantoms (`ddssl_ldct/phantoms.py`).
