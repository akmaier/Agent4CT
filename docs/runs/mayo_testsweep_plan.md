# Mayo-LDCT test-selection sweep — plan

**Goal.** Make the Mayo leaderboard select **each solver's board iteration by
TEST mean `hr`** (mean over the 5 held-out Wagner test patients), instead of the
current "pick the val-best iter, then test-score only that one iter." Today the
board is *ranked* across solvers by test `hr`, but **within** each solver the iter
is chosen by **val** (L277) `hr` — so a solver whose val-best iter is not its
test-best iter is misrepresented (param-efficient is the clear case: val-best
iter-32 scores **0.194** on test, but its test-best iter-33/36 scores **~0.324**).

Frozen references: README "Evaluation paradigm (canonical, frozen)";
metric = `evaluate_calibrated` (no upper clamp, `data_range`=truth range,
FOV-masked, `hr = max(0, 1 − rmse/LD-FBP-rmse)`).

## Decisions (2026-07-01, user-confirmed)

1. **Scope = full per-iter sweep** — every iteration of every solver gets a test
   score; each solver's board iter is then the max-`test_hr_mean` iter.
2. **Trained solvers: train each config exactly ONCE, save the checkpoint, reuse
   it** for all 5 test patients (and for any future metric re-score). No saved
   weights exist (verified), so one training per config is unavoidable — but it is
   done once, not 5× per patient and not per iteration-run twice.
3. **Keep the test reconstructions** (metric-off safety net). See "Disk" for how
   this is done without a ~700 GB blow-up.
4. Per-image / diffusion solvers: reconstruction only, no training (diffusion
   priors `ddpm_mayo_*_v{2,3,4}.pt` already exist on the cluster).

---

## Do we have to retrain?  — VERIFIED 2026-07-01

**The trained model weights DO NOT EXIST.** Verified on both the laptop and the
cluster: every Mayo per-iter dir contains only `comparison.png` +
`result.json`/`observation.json` — extension histogram across all
`mayo-…-iter-*` dirs is **png + json only**; `find … -name '*.pt/*.pth/*.ckpt'`
returns **0** checkpoints in the whole runs tree. The agentic loop trained a model
each iteration, scored it, saved the figure + metrics, and **discarded the
weights**. So there are **no existing trained models to reuse** — the premise of
"reuse the trained models / no new training" cannot be met for the trained
solvers as-is.

Consequence, by solver family:

- **Trained solvers** (itnet / itnet-v2 / itnet-v3, uswin, dual-domain
  supervised + N2I + bilateral variants, hammernik-2017 / hammernik-vn,
  learned-primal-dual, wu-2015-trainable, tv-iterative-supervised): a per-iter
  test score is **impossible without running the config = training it once**.
  There is no checkpoint to load. This is *train once on the fixed 4 train
  patients, then infer* (the canonical paradigm — NOT a per-patient hold-out),
  but it **is** training. param-efficient is in this family but is **already
  fully test-scored** (39/40), so it needs no new work.
- **Per-image / iterative / neural-field / diffusion solvers** (tv-iterative,
  manhart-pwls-tv, manduca-bilateral, naf, r2gaussian, ram, diff-recon-* [fixed
  pretrained prior], fastdiff-* [fixed pretrained prior]): **no model is trained
  on the train set** — each test slice is reconstructed / per-scene-optimised at
  eval time. Scoring these per-iter is **reconstruction only, no training** —
  fully consistent with "no new training" (it is still compute, e.g. NAF/R2G
  ~18 min/patient).

**Safety net for the CURRENT board already exists.** The val-best-iter TEST
reconstructions are saved on the cluster: **156 `recon_raw.npz` across 29
`-testset` dirs, ~47 GB** (raw pred/truth/baseline per test patient). If the
metric is ever wrong again, the current board re-scores **offline from these,
with zero recompute**. The sweep will save the same `recon_raw.npz` for every
iter it runs (and we will **not delete** them).

**Implementation caveat (why the proven harness runs the solver 5× per iter).**
PYRO-NN throws `invalid resource handle` if one process cycles through multiple
patients' pixel-spacings, so every eval pass handles **exactly one patient**.
Because no checkpoints are saved, the current worker (`score_mayo_testset.py`)
re-invokes the solver's `main()` once per patient — so a trained solver trains 5×
per iter. Training is **deterministic** (fixed seed) + patient-independent, so
those 5 runs yield the **identical** model (paradigm-correct, just 5× compute).
The fix (train once, save the ckpt, 5 inference-only passes) needs a per-solver
ckpt save/load path across the trained solvers — engineering, but it removes the
5× waste **and** makes the weights reusable for any future metric re-score.

## On which data do we evaluate?

- **Score on the 5 held-out Wagner TEST patients: L014, L056, L058, L075, L123**
  (745 slices; per-patient 154 / 93 / 210 / 137 / 151). Report **mean ± std over
  the 5 patients** for `hr`, SSIM, PSNR, RMSE.
- **Train (trained solvers) on L145 / L186 / L209 / L219** only.
- **L277 (val) is no longer used for board selection** — it was the search-time
  signal. After this change it does not choose the board iter either.

---

## Inventory (what actually has to run)

- **26 Mayo solvers**, ~20 iters each = **539 iterations** total, all carrying
  `cfg_full` in their per-iter `observation.json`.
- **param-efficient is already fully test-scored** (39/40 iters, all 5 patients,
  in `docs/runs/pe-iter-testeval/trajectory_test.json`). Test-best there:
  iter-33 `0.3241±0.063` (969 p) / iter-36 `0.3237±0.063` (497 p).
- So the sweep covers the **other 25 solvers × ~20 iters ≈ 500 iterations**, each
  = **5 patient eval passes**.
- Only param-efficient snapshots per-iter source (`solver_src.py`); the other 25
  use the fixed `SOLVER_MAP` solver file + the iter's `cfg_full`, so dispatch only
  varies the cfg per iter.
- Median original per-iter wall (train + val-eval + figure): ~8–21 min.

## Cost

Per-iter test cost ≈ 5 × (that iter's train+eval), i.e. roughly 5 × the original
per-iter wall for retrain-per-patient.

| Scope | iters scored | GPU-h (approx) | Wall @ ~6 turbo slots |
|---|---:|---:|---:|
| **Full sweep, retrain-per-patient** (25 solvers × 20 iters × 5 passes) | ~500 | ~750 | **~5 days** |
| **Top-K by val (K=5) × 25 solvers × 5 passes** | ~125 | ~190 | **~1.3 days** |
| Full sweep, train-once-infer-5 (needs per-solver ckpt path) | ~500 | ~250 | ~2 days + build time |

---

## Checkpoint-cache mechanism (train once, reuse)

Verified solver shape: each trained solver's `main()` builds ONE top-level
`nn.Module` and trains it in exactly one place before eval (itnet_v2 → `ItNetV2`
wrapping the pretrained `SmallUNet`; dual_domain_supervised → `FullViewUNetPipeline`).
So a `state_dict()` save/load hook is clean. Minimal per-solver edit, gated by an
env var so normal agentic runs are unaffected:

```python
_CKPT = os.environ.get("AGENT4CT_MODEL_CKPT")          # set only by the sweep worker
if _CKPT and Path(_CKPT).exists():
    <build the model modules (architecture only, NO training)>
    model.load_state_dict(torch.load(_CKPT, map_location=device)); model.eval()
else:
    <existing training>                                # unchanged
    <build the model>
    if _CKPT:
        Path(_CKPT).parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), _CKPT)
```

**Worker change** (`score_mayo_testset.py`): set `AGENT4CT_MODEL_CKPT` to ONE
path per iter (shared across its 5 patient passes). Patient-1 (L014) trains +
saves; patients L056…L123 load → **one training per iter, four inference-only
passes.** Correctness rests on training being patient-independent (train set is
always L145/186/209/219; `AGENT4CT_EVAL_PATIENT` must only affect the eval set —
**verify this per solver**) and deterministic (fixed seed).

## Disk strategy (keep recons without a 700 GB blow-up)

Full recons for all ~500 iters × 5 patients ≈ **~700 GB** — infeasible. Instead:

- **Per iter, the sweep saves the model checkpoint** (small — a few to tens of MB;
  ~10 GB total for ~500 iters) + the scalar test metrics (`final.json`). The ckpt
  is the durable, reusable artifact: any recon can be regenerated by inference
  with zero retraining, and it re-scores under any future metric.
- **Full `recon_raw.npz` are kept only for each solver's SELECTED board iter**
  (~26 × 5 patients ≈ **~47 GB**, same magnitude as the current safety net),
  produced by one inference pass from the cached ckpt after selection.
- The existing 47 GB of current-board recons stay untouched.

(Trade-off flagged to the user: this keeps *checkpoints* for every iter and full
*reconstructions* for the board iters, rather than raw recons for all 500 iters.
It satisfies "re-score if the metric is off" with less disk, since a ckpt
regenerates any recon deterministically.)

## Machinery

The per-iter test-scorer already exists for one solver: `rescore_pe_iters.py
--test`. The sweep **generalizes that pattern to any run-id**:

1. `scripts/score_mayo_alliters.py` (new) — for each run-id, for each iter dir,
   write the iter's `cfg_full` to a tmp cfg and submit **one** sbatch that runs
   the `score_mayo_testset.py` **worker** loop (5 patients, one per eval pass,
   `AGENT4CT_MODEL_CKPT` shared) against the iter's cfg (+ `solver_src.py` for
   param-efficient). Output → `docs/runs/<slug>-itertest/iter-NNNN/final.json`.
   Idempotent: skip an iter whose final.json already has all 5 patients.
   param-efficient reuses its existing `pe-iter-testeval` results.
2. **Collect** → per solver, pick max-`test_hr_mean` iter (SSIM tiebreak) = the
   solver's **test-best iter**; regenerate + keep its 5 recons from the cached ckpt.
3. **`build_registry.py`**: change `best_iter_row` so a **test dataset** picks the
   within-run iter by **max per-iter `test_hr_mean`** (reading the sweep's per-iter
   final.jsons), instead of the current val-headroom pick over one final.json.
4. **README** "Evaluation paradigm" block: selection + ranking both by **test mean
   `hr`**; val (L277) not used for board selection.
5. Rebuild registry → `validate_registry.py` must **PASS** → commit + push.

Until the sweep finishes the board stays as-is (val-selected iters). Nothing on
the live board changes mid-sweep.

---

## Status

- [x] scope + retrain-mode confirmed (full sweep; train-once-save-ckpt; keep recons)
- [x] verified: no saved solver weights exist (laptop + cluster); diffusion priors do
- [x] **ckpt-cache hook reference impl** on itnet_v2 + dual_domain_supervised
      + worker `AGENT4CT_MODEL_CKPT` change — SMOKE-TESTED on cluster
      (jobs 771015/771016): P1 trains+saves, P2–P5 load+skip; reproduced the
      board within ~0.002 (itnet-v2 0.3689 vs 0.3707; dd-sup 0.3593 vs 0.3567);
      ckpt + 5 recons saved. ~18 min train + 4×~1 min infer ≈ 22 min/iter.
- [x] fan out the ckpt hook to the other 11 trained solvers — all py_compile;
      independently re-verified by an 11-agent adversarial pass (all PASS, 0 issues:
      skip-branch skips all training + builds full arch, saved==eval object, eval in
      both branches, save-guard correct, N2I per-image refit unconditional).
- [x] `score_mayo_alliters.py` (`--build`/`--collect`) + `mayo_alliters_array.sbatch`
      (SLURM array, `%8` throttle, `AGENT4CT_SWEEP_NORECON`); `--build` wrote 500
      work items (25 solvers × 20). Worker: `AGENT4CT_SWEEP_NORECON` keeps ckpt, skips recons.
- [x] **PILOT** array 771018 PASSED — 3 shapes reproduced the board: uswin 0.3503
      (board 0.3524, ckpt train-once-reuse), tv-iter 0.0746 (board 0.0746 exact,
      per-image no-hook), dd-n2i 0.0056 (board 0.0076, warm-start cached + refit
      ran); recon_raw.npz=0 everywhere (no-recon mode), ckpts kept.
- [x] full sweep LAUNCHED via self-topping driver `scripts/drive_alliters_sweep.py`
      (nohup on login node; QOS caps 100 submit / 10 run → driver keeps ~90 queued,
      refills as they finish, idempotent + retry-capped 3). 497 iters pending
      (500 − 3 pilot). ETA ~1.5–2 days. ← **running**
- [ ] `--collect` → per-solver test-best iter; regenerate + keep board-iter recons
- [ ] `build_registry.py` test-selection change + README paradigm block
- [ ] rebuild + validate PASS + commit/push
- [ ] collect test-best iter per solver; regenerate + keep board-iter recons
- [ ] `build_registry.py` test-selection change
- [ ] README paradigm block updated
- [ ] rebuild + validate PASS + commit/push
