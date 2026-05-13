# DL-Sparse-View CT — agent program

Your job: minimise the validation RMSE of a 2D sparse-view fan-beam CT
reconstruction by editing `pentathlon/dl_sparse_view/solver.py` and running
short Slurm iterations.

## Hard contract

You may edit **only**:
- `pentathlon/dl_sparse_view/solver.py` (any block, including `CONFIG`
  and `build_denoisers`)

You may **not** edit:
- `ddssl_ldct/` (the recon backbone, simulation, harness, metrics)
- `cluster/slurm/dl_sparse_view_5min.sbatch` (the time budget + entry
  point are fixed by the harness)
- the scoring formula (`headroom = 1 − val_rmse / baseline_rmse`)
- the validation subset size or seed (`val_n=100`, `seed=42`)

## Scoring

```
val_score = val_ssim                                   # primary single number
headroom  = 1 − val_rmse / baseline_sparse_view_RMSE   # in [0, 1]
```

`headroom=0` means "same as a plain sparse-view FBP", `headroom=1` means
"reconstruction error is zero against the truth". Aim to push headroom up.

## Time budget per iteration

5 minutes wall on a single GPU. The solver should finish well inside that,
so you have headroom for setup + the comparison-image dump.

## Stage check (every 30 iter)

A 1-hour stage Slurm job runs on a **3× larger** train + val subset. If
the stage val_score is much lower than the iter val_score, the agent has
overfit the small subset and the next edit must regularise (smaller model,
more weight decay, augmentation) — see `docs/agents.md` for the rules.

## Required reading before any non-trivial change

In rough priority order:

1. `literature/sidky_2022_dl_sparse_view_2109.09640.md` — the challenge
   report. Section 5 enumerates the **top-five teams' approaches**. Skim
   for what worked: most successful teams used sinogram-domain priors +
   image-domain refinement.
2. `literature/2211.01111_Wagner_DualDomainDenoising_LDCT.md` — our recon
   backbone. The dual-domain split-projection N2I idea is already wired in.
3. `literature/artifact_gallery.md` — for *naming* what's wrong in your
   comparison image. The DL-Sparse-View regime is exactly the
   "limited-angle / sparse-view" entry's family — radial streak artefacts
   from under-sampling.
4. `docs/runs/observations.jsonl` — the last ~ 30 entries across all
   agents. Worth scanning even though you're the only DL-Sparse-View
   agent for now; future cross-challenge agents will benefit.

## Anti-overfit rules (every iteration)

1. **Parameter budget vs. training-set size.** Keep total trainable
   parameters under roughly `10 × train_n × pixels_per_sample` unless
   you have a clear regularisation story. With `train_n=400` and
   `512²` pixels, the soft cap is ≈ 1 G parameters per network — but
   in practice anything beyond ~ 5 M starts to overfit our subset.
2. **One change per iteration.** If two things change and val moves,
   you don't know which one did it.
3. **Augmentation is cheap.** Random flip / intensity jitter / small
   affine warps consistently lift small-data scores. Try before scaling
   architecture.
4. **Don't game val.** Improvements under 1 % across three iterations
   are noise — discard.
5. **Cite the iteration that inspired you** in `--rationale` so the
   journal stays auditable.

## Per-iteration protocol

1. Read `docs/runs/observations.jsonl` (last 30 entries).
2. Read the literature folder if you're proposing a non-trivial change.
3. Edit `pentathlon/dl_sparse_view/solver.py`.
4. Submit `sbatch cluster/slurm/dl_sparse_view_5min.sbatch`.
5. Read `result.json` from the run dir.
6. Decide keep / discard. For discards, run
   `git checkout HEAD -- pentathlon/dl_sparse_view/solver.py` first.
7. Call `agent4ct_record record …` with the result + a one-sentence
   rationale + a one-sentence `--advice` for other agents.

## Stopping

The run stops when **any** of these holds (per `docs/agents.md`):

- 150 iterations completed,
- no improvement (no new `keep`) in the last 30 iterations,
- 3 consecutive stage checks return `overfit` with no recovery,
- operator stops manually.

Then `finalize` with the matching `--stop-reason`, after retraining the
best iteration's solver on the full DL-Sparse-View train set and
evaluating on the held test set.

## What "real DL-Sparse-View data" means here

The real challenge data (4 000 simulated breast phantoms with 128-view
sinograms + perfectly-known truth) is **not yet on the cluster**. The
solver synthesises random-ellipse phantoms as a stand-in with the
correct geometry. Replace `build_dataset(...)` in solver.py with a real
loader once `data/dl_sparse_view/staged/` is populated — that swap is
explicitly *allowed* (the agent can edit data-loading in solver.py).
Note in the iteration's `--rationale` when this happens.
