---
title: Runs
description: Per-run data served as static assets to the dashboard.
---

This folder is written by [`ddssl_ldct/harness.py`](https://github.com/akmaier/Agent4CT/blob/main/ddssl_ldct/harness.py) (via [`scripts/agent4ct_record.py`](https://github.com/akmaier/Agent4CT/blob/main/scripts/agent4ct_record.py)).
The live [dashboard](../dashboard.html) reads everything from here.

```
runs-index.json                   # auto-maintained list of all runs
observations.jsonl                # cross-run append-only scratch pad
<run-slug>/
  manifest.json                   # one per run
  results.tsv                     # iteration journal (one row per iter)
  stages.tsv                      # stage-check journal (every 30 iter)
  iterations/
    iter-NNNN/
      observation.json
      comparison.png
      solver.py.txt
      stdout.log
```

`<run-slug>` ≡ `<challenge-slug>-YYYYMMDD-NN`. See
[Agents](../agents.html) for the convention details.

You should not commit anything in this tree by hand — every file lands
here as a side-effect of the harness recording an iteration.
