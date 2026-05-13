---
title: Agents
description: The autoresearch loop, the shared scratch pad, and the run/iteration versioning.
---

## The loop

```
┌─────────────────────────────────────────────────────────────────┐
│  Five agents (one per challenge), running concurrently          │
│                                                                 │
│    each iteration ≡ one 5-minute Slurm job:                     │
│      1. agent reads docs/runs/observations.jsonl (last ~30)     │
│      2. agent edits  pentathlon/<challenge>/solver.py           │
│      3. sbatch  <challenge>_5min.sbatch  → 5-min training       │
│      4. compute val score on the held val subset                │
│      5. decide   keep (git commit) | discard (git reset)        │
│      6. write    docs/runs/<run>/iterations/iter-NNNN/{...}     │
│      7. append   docs/runs/observations.jsonl                   │
│                                                                 │
│    every 30 iterations: stage job (1-hour wall, 3× larger set)  │
│      — checks generalisation; if iter ≫ stage, agent told to    │
│        regularise or shrink on the next iteration.              │
└─────────────────────────────────────────────────────────────────┘
```

## Run versioning

A **run** is one sequence of iterations targeting one challenge (or in the Pentathlon all-rounder phase, all five). Runs are addressed by a slug:

```
<challenge-slug>-YYYYMMDD-NN
```

Examples:

| Slug | What it is |
|---|---|
| `dl-sparse-view-20260513-01` | First DL-Sparse-View run started on 2026-05-13 |
| `mayo-ldct-20260514-02` | Second Mayo run started on 2026-05-14 |
| `pentathlon-allrounder-20260601-01` | All-rounder, joint over five challenges |

Numbering is per-day (zero-padded). The `NN` lets multiple agents start the same day without colliding.

## Iteration versioning

Within a run, iterations are zero-padded four-digit indices, starting at `iter-0001`. Each iteration writes a directory containing:

```
docs/runs/<run>/iterations/iter-NNNN/
  observation.json     # full scratch-pad entry (see below)
  comparison.png       # side-by-side reference / FBP / pred / phantom
  solver.py.txt        # snapshot of pentathlon/<challenge>/solver.py
  stdout.log           # tail of the training log
```

Iteration commits look like:

```
iter 0017 dl-sparse-view-20260513-01: val=0.62 headroom=0.41 keep

Switched Adam -> AdamW with weight_decay=1e-4. Inspired by iter
mayo-ldct-20260513-02/iter-0014 (advice: "weight_decay helps when
model > 10x dataset size").
```

so the git log doubles as a navigable journal.

## The shared scratch pad

Every iteration emits one line into the global append-only log at
`docs/runs/observations.jsonl`. The next agent — on any challenge — reads
the last ~ 30 lines before its own edit.

Fixed schema, so agents can parse it cheaply:

```json
{
  "ts": "2026-05-13T20:51:20Z",
  "run_id": "dl-sparse-view-20260513-01",
  "iter": 17,
  "challenge": "dl_sparse_view",
  "agent": "claude-sonnet-4.5",
  "change_class": "optimizer",
  "rationale": "Switched Adam -> AdamW; weight_decay=1e-4 to combat overfit on 400-sample train.",
  "val_score": 0.62,
  "headroom": 0.41,
  "delta_vs_best": 0.04,
  "kept": true,
  "params_M": 1.8,
  "train_n": 400,
  "comparison_image": "runs/dl-sparse-view-20260513-01/iterations/iter-0017/comparison.png",
  "advice_for_others": "weight_decay helps when params >> training-set size"
}
```

The scratch pad is rendered on the live [dashboard](dashboard.html) — every entry shows up as a card with the comparison image, the score row, and the rationale.

## The harness writer

The Python helper in [`ddssl_ldct/harness.py`](https://github.com/akmaier/Agent4CT/blob/main/ddssl_ldct/harness.py) makes all of the file-shape conventions concrete:

```python
from ddssl_ldct.harness import open_run

with open_run(challenge="dl_sparse_view", slug_prefix="dl-sparse-view") as run:
    for iter_n in run.iterations(start_at=1):
        # ... the agent edited pentathlon/dl_sparse_view/solver.py ...
        val_score, headroom = train_and_score(...)
        run.record(
            iter_n=iter_n,
            val_score=val_score,
            headroom=headroom,
            rationale="switched Adam -> AdamW",
            change_class="optimizer",
            comparison_png="runs/local/iter.png",
            solver_path="pentathlon/dl_sparse_view/solver.py",
            kept=val_score > run.best_score,
        )
```

The helper handles: building the slug, creating per-iteration directories, writing `observation.json`, appending to `results.tsv` and the global `observations.jsonl`, and updating `runs-index.json` so the dashboard sees the new iteration on its next refresh.

## How to start (and run) a run

The CLI auto-commits and auto-pushes after each `record` (and after `finalize`) by default — pass `--no-commit` / `--no-push` if you want to batch up changes.

```bash
# Once, at the start of a run:
SLUG=$(python scripts/agent4ct_record.py new-run \
    --challenge dl_sparse_view --slug-prefix dl-sparse-view \
    --notes "first run, baseline U-Net")

# After each 5-minute Slurm iteration:
python scripts/agent4ct_record.py record \
    --slug "$SLUG" --iter 1 \
    --val-score 0.58 --headroom 0.37 \
    --change-class architecture \
    --rationale "baseline U-Net c=24, Adam lr=1e-3" \
    --advice "U-Net c=24 is a sane starting point for ~400 train samples" \
    --kept true --commit "$(git rev-parse --short HEAD)" \
    --comparison runs/local/comparison.png \
    --solver pentathlon/dl_sparse_view/solver.py
# → writes docs/runs/$SLUG/iterations/iter-0001/{observation.json,comparison.png,…},
#   appends a row to results.tsv, appends a line to docs/runs/observations.jsonl,
#   commits the journal additions, pulls --rebase (in case another agent
#   pushed during the 5-min Slurm job), and pushes.

# Every 30 iterations (1-hour Slurm job, 3× larger subset):
python scripts/agent4ct_record.py stage \
    --slug "$SLUG" --iter 30 \
    --stage-val-score 0.55 --stage-headroom 0.30 \
    --iter-val-score 0.62 --verdict overfit \
    --notes "stage val ≪ iter val — shrink model on next iter"
```

For the *discard* case, the agent reverts its solver edit before calling `record`:

```bash
git checkout HEAD -- pentathlon/$CHALLENGE/solver.py
python scripts/agent4ct_record.py record --slug "$SLUG" --iter N \
    --kept false --status discard ...
```

— so the next iteration starts from the last *kept* solver, and the journal still records the failed attempt.

## Stopping a run

A run ends when **any** of these is true:

| Condition | Default |
|---|---|
| Iteration budget exhausted | 100 iterations |
| No improvement (no new `keep`) in last 30 iterations | always on |
| Three consecutive **stage** checks return `overfit` with no intervening `ok` | always on |
| Manual stop (operator's call) | — |

The agent decides which applies and calls `finalize` with the matching `--stop-reason`. The default budget of **100 iterations** is a sensible upper bound for a single challenge — five challenges × 100 iters × 5 min ≈ 40 GPU-hours, modest on a small lab cluster.

## Final test-set evaluation (once per run)

The iteration phase **never** touches the test set. After the iteration phase ends:

1. Identify the best iteration from `results.tsv` (highest `val_score`).
2. Optionally re-train its solver on the **full** challenge train set (not the 200–400-sample iteration subset) — typically a 1-to-2-hour Slurm job mirroring the stage job's budget.
3. Run that retrained solver on the **held test set** (also via a Slurm job).
4. Record the result:

```bash
python scripts/agent4ct_record.py finalize \
    --slug "$SLUG" --stop-reason budget \
    --final-test-score 0.71 --final-test-headroom 0.55 \
    --final-test-comparison runs/local/final_test_comparison.png \
    --notes "Retrained best iter (iter 64, AdamW c=24) on full 4000-phantom train set; tested on 100 held-out phantoms."
```

→ Writes `docs/runs/$SLUG/final.json` and updates the manifest's status to `done`. The dashboard renders the result as a banner on the run-detail page.

## Preparing for the Pentathlon final

Once each of the five challenges has a finalised run, the **Pentathlon all-rounder** phase searches for a single configuration that maximises the unweighted mean of headrooms across all five test sets.

1. Start an all-rounder run: `--slug-prefix pentathlon-allrounder --challenge pentathlon`.
2. Each iteration evaluates on a *small subset* of each challenge's val set (still 5-min budget). The reported `val_score` is the mean headroom across challenges.
3. Stage checks every 30 iter still trigger (1-hour, 3× larger subsets).
4. When the all-rounder run ends, the final-test eval evaluates on the **test set of all five challenges** in one job and reports the Pentathlon score (mean test headroom).
