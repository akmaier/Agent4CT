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

## How to start a run

```bash
# from the laptop, one terminal per challenge
agent4ct-record --challenge dl_sparse_view --new-run
# then submit iterations in a loop:
agent4ct-record --challenge dl_sparse_view --iter 1 --comparison comparison.png \
    --val-score 0.58 --headroom 0.37 --rationale "baseline U-Net"
```

The dashboard updates as soon as you `git push` — the JSON/TSV files are served as static assets out of `docs/runs/`.
