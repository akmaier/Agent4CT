---
title: Cross-cutting findings
description: Substantive learnings from the autoresearch loop that span multiple iterations, agents, or sessions. Newest first.
---

This file is the cross-agent handoff log. Per-iteration rationale belongs in
`docs/runs/<slug>/iterations/iter-NNNN/observation.json`. Stage-check
verdicts belong in `docs/runs/<slug>/stages.tsv`. Things that belong here:
**facts about the substrate or methodology that the next agent should not
have to re-discover.**

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
