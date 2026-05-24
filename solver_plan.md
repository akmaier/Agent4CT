# Solver onboarding plan: from new dataset to leaderboard

This document is the **canonical recipe** every agent should follow when
adapting the Agent4CT solver suite to a new CT-reconstruction
benchmark. It tells you exactly what to run, in what order, and where
to write the results down — so the answer to *"how do I add dataset X?"*
is always the same.

The recipe has been used (with retrospective alignment) for the
existing three datasets:

| Dataset | Source | Status |
|---|---|---|
| `demo_dl` | Synthetic Sidky-style sparse-view ellipse phantoms, 128 angles | ✅ leaderboard frozen as of 2026-05-22 |
| `breast_ct` | Synthetic breast phantoms (Sidky group), 128 angles, real-tissue μ range | ✅ leaderboard frozen as of 2026-05-23, LPD-TPE running |
| `mayo_ldct` | AAPM 2016 Low-Dose CT challenge, helical Siemens AS+, **Wagner split**: train = L145/186/209/219, val = L277, test = L014/056/058/075/123 | 🚧 rebin geometry validated 2026-05-24, autoresearch starting |

When a new dataset lands, work through the five steps below in order.

---

## Step 1 — Confirm the data is reconstructable (FBP investigation)

Before training a single network, **prove the baseline FBP is anatomically
correct.** No solver can do better than FBP if the data is broken.

Checklist for any new dataset:

1. **Read one raw sinogram + ground-truth slice pair.** Render side by
   side. The FBP must show recognisable anatomy at the same z position
   as the truth.
2. **For helical data**: walk the rebinning pipeline step by step. We
   have a documented bug list in
   `literature/wagner_helix2fan_algorithm.md` and a step-by-step
   reading + filter / redundancy / padding analysis in
   `docs/findings.md`'s helix2fan entries. Reuse those debug scripts
   (`scripts/debug_l014_*.py`) — they're parameterised on patient ID.
3. **Sanity checks for any 2-D fan sino**:
   - rotational range = 2π (not Parker'd by accident).
   - filter = Hann, padding = 2N, half-DC truncation correction —
     all baked into `PyronnFanBeamProjector.fbp` with `redundancy="full_scan"`
     for 2π scans (see `ddssl_ldct/pyronn_projector.py`).
   - intensity range matches what `dataset_kind`'s `geometry_overrides`
     declares for `display_max`.
   - the **sino has no gaps** — every (s_angle, z_out) cell must be
     populated; check the per-row stats.
4. **Intensity-calibrated FBP-vs-truth SSIM**. Run
   `scripts/validate_*.py` (or equivalent for the new dataset). The
   metric is *calibrated* SSIM/PSNR via `evaluate_calibrated` — same
   formula the solvers use during training, so the numbers compare
   like-for-like.

**Done-criteria**: a single comparison PNG showing anatomy in both
panels, calibrated SSIM ≥ 0.65 (a usable baseline), no obvious flat
bands / shadow ghosts. If you can't get there, **the data is the
problem; do not waste compute on solvers yet.**

What this looked like in practice (for `mayo_ldct`):

- The previous-agent's note flagged "Bug 1–6 in the helix2fan port."
- That turned out to be ~10 % of the story; the dominant bugs were
  found through *this step's* validator + sino inspection + pitch
  arithmetic: alphabetic-sort file order, median-not-mean pitch
  formula, missing intensity calibration. See findings.md
  2026-05-23/24 entries.

---

## Step 2 — Agentic autoresearch per solver

For every solver in `pentathlon/demo_dl_reference/solver_*.py`, run a
**Claude-driven autoresearch loop**:

```bash
# One iter at a time; the agent reads the prior iters' results from
# docs/runs/<slug>/, edits hyperparams (or solver code), and submits.
SLUG=<dataset>-claude-agentic-<solver>-search-YYYYMMDD-NN \
ITER_N=<n> SOLVER=<solver_key> \
CFG_JSON=/cluster/maier/Agent4CT/agentic_cfgs/<solver>_iter_NN.json \
AGENT4CT_DATASET=<dataset> \
sbatch --export=ALL,... cluster/slurm/claude_agentic_one_iter.sbatch
```

**Allowed changes per iter**: anything that the literature suggests
might help on this dataset class — hyperparams (lr, epochs, batch,
weight_decay, lr_schedule, …), capacity (channels, layers, unrolled
iterations), loss-formulation (supervised L2 vs Noise2Inverse, L1,
perceptual…), training-set size, init scheme.

**Rule of thumb**: 10–15 iterations is enough to plateau. Stop when
two consecutive iters fail to improve hr by > 0.005.

**Slug convention**: `<dataset-prefix>-claude-agentic-<solver>-search-YYYYMMDD-NN`
e.g. `breast-ct-claude-agentic-learned-primal-dual-search-20260522-01`.

**Inputs the agent reads each iter**:
- `docs/runs/<slug>/results.tsv` — full prior iter history.
- `docs/runs/<slug>/iterations/iter-NNNN/observation.json` — per-iter
  rationale + scores.
- `docs/runs/<slug>/iterations/iter-NNNN/comparison.png` — for visual
  diagnosis of artefacts.
- The matching solver design doc in `pentathlon/demo_dl_reference/solver_<name>.md`
  — "Hints for the next autoresearch agent" section.

**Outputs each iter writes**: the same paths the previous agent reads,
plus a new `solver.py` snapshot if the code was edited.

---

## Step 3 — TPE hyperparameter refinement

After the autoresearch plateau, **lock the architecture** at the
agentic winner and run a focused TPE search around that point:

```bash
sbatch cluster/slurm/demo_<solver>_search_tpe.sbatch
```

The search space is defined in
`scripts/learned_solver_search_agent.py`'s `SOLVERS` dict. Per-solver
entries describe:
- The numerical / categorical search box for each knob.
- A `tpe_seed_trial` set to the agentic winner — TPE starts from a
  known-good point rather than cold.

Slug convention: `<dataset-prefix>-calibrated-tpe-<solver>-search-YYYYMMDD-NN`
(the `--calibrated --dataset <ds>` CLI flags produce this prefix
automatically).

**Default budget**: 20 trials, 5 random startup + 15 TPE-adaptive.
Each trial inherits the calibrated-scoring infrastructure
(`evaluate_calibrated`).

**When to stop**: TPE's 20-trial default plateau is reasonable. Look
for the trial that beats the seed by > 0.01 hr; if no improvement,
the agentic winner is already the optimum.

---

## Step 4 — Diffusion-prior special case

`solver_diffusion_recon.py` (DPS / MCG / DC-step posterior sampling)
needs a **trained DDPM prior** as its source of inductive bias. The
prior is dataset-specific.

For every new dataset, train **two** DDPM variants:

| Variant | Train data | Purpose |
|---|---|---|
| **constrained** | TRAINING split labels only | Honest LD/HD prior — no test-set leakage. Reported as "DDPM constrained". |
| **unconstrained** | ALL data (training + val + test labels) | Probe how much "having seen the test set" inflates the prior. Reported as "DDPM unconstrained". |

Both train with the same architecture (`SmallDDPM`, default
`ddpm_ch=32`, `ddpm_n_steps=1000`); the dataset's `display_max` is
auto-detected as `ddpm_out_scale` so the intensity normalisation
matches downstream sampling.

Workflow:

1. **Train both DDPM variants** via `solver_ddpm.py`:
   ```bash
   # constrained — uses cfg["seed"] same as supervised solvers
   AGENT4CT_DATASET=<dataset> \
   sbatch --export=ALL,VARIANT=constrained cluster/slurm/ddpm_train.sbatch
   # unconstrained — different seed range, larger n_train
   sbatch --export=ALL,VARIANT=unconstrained cluster/slurm/ddpm_train.sbatch
   ```
   The checkpoint path lands at
   `/cluster/maier/Agent4CT/checkpoints/ddpm_<dataset>_<variant>_final.pt`.
2. **Two separate diff-recon TPE searches** (one per checkpoint),
   each producing its own leaderboard entry. The user explicitly
   wants both so the constrained-vs-unconstrained gap is visible
   per dataset.

**Caveat**: for synthetic-phantom datasets (`demo_dl`, `breast_ct`)
the distinction is "trained on test-distribution seeds" vs "trained
on disjoint seeds". For real datasets (`mayo_ldct`) it's clean:
constrained sees only `L145/186/209/219`, unconstrained sees all 10
Wagner patients.

**RAM (`solver_ram.py`)** is the other model with a pretrained
checkpoint — `ram.pth.tar` (Terris 2025). We don't know what RAM
was pretrained on; document this caveat in the per-solver markdown
(`solver_ram.md` if/when written).

---

## Step 5 — Leaderboard + per-solver insights

Two writing targets after every dataset's autoresearch + TPE round:

### 5.1 — Per-dataset leaderboard

`docs/leaderboards/<dataset>.md` (auto-generated section the agent
keeps fresh). Schema:

```markdown
| Rank | Solver | Variant | SSIM | PSNR | RMSE | hr  | Source slug |
|---:|---|---|---:|---:|---:|---:|---|
| 1 | Learned Primal-Dual | I=8, hidden=96, ep=23, lr=3.2e-4 | 0.9996 | 56.0 | 0.0007 | 0.906 | breast-ct-calibrated-tpe-lpd-search-20260524-01 / trial 11 |
| 2 | DD-UNet supervised L2 | c=32, ep=10, lr=5e-4 | 0.9987 | 54.9 | 0.0009 | 0.826 | breast-ct-claude-agentic-dual-domain-unet-l2-search-20260522-01 / iter-3 |
| … | … | … | … | … | … | … | … |
```

Only the **best version of each solver family** per dataset gets a row.
Multiple variants of the same solver (e.g. DD-UNet L2 vs DD-UNet N2I)
do get separate rows because they're distinct algorithmic choices.

### 5.2 — Per-solver markdown (cross-dataset insights)

Each `pentathlon/demo_dl_reference/solver_<name>.md` carries a section
**"Cross-dataset observations"** that grows over time as the solver is
applied to new benchmarks:

```markdown
## Cross-dataset observations

| Dataset | Best hr | Best config | Notes |
|---|---:|---|---|
| demo_dl   | 0.62 | … | … |
| breast_ct | 0.83 | … | DD-BF supervised hits 0.21 at 6 params here; better than at demo_dl. |
| mayo_ldct | …   | … | … |
```

Followed by **strengths / weaknesses / hints** sections — same
template as the existing docs (e.g. `solver_dual_ddomain_supervised.md`).

---

## Practical conventions

- **Train_n discipline**: when reporting a "best" result, **use the
  full training set** for the final number, not a sub-sample. The
  per-iter agentic loop runs at `train_n=400` for speed; the
  leaderboard entry should be the same config rerun at the
  dataset's full `train_n` (typically 1000–4000 phantoms).
- **Val_n discipline**: keep `val_n=20` during search to make the
  per-iter loop fast, but **stage-check the winner at val_n=60** (or
  higher) to confirm the metric isn't val-set-noise-limited. Several
  solvers have hit the val_n=20 noise floor (≈ ±0.01 hr).
- **Calibration**: every metric in the leaderboard goes through
  `evaluate_calibrated` (FOV mask + linear intensity calibration).
  Same as the training-time metric, so the headroom field is
  comparable across solvers.
- **Reproducibility**: don't re-run configs known to fail. Each
  per-solver design doc carries a "known dead ends" section — read it
  first.

---

## Where to record what

| File | Purpose |
|---|---|
| `solver_plan.md` (this file) | The recipe. Update only to revise the methodology itself. |
| `pentathlon/demo_dl_reference/solver_<name>.md` | Per-solver design + cross-dataset table + hints. |
| `docs/leaderboards/<dataset>.md` | Per-dataset best-of-best leaderboard (one row per solver family). |
| `docs/findings.md` | Cross-cutting insights that span solvers OR datasets — substrate / methodology learnings that no single solver doc owns. Newest first. |
| `docs/runs/<slug>/` | Per-run dashboard data (manifests + results.tsv + per-iter PNGs + observations). Touched by the agent every iter; you generally don't edit by hand. |
| `agentic_cfgs/<solver>_iter_NN.json` | Per-iter agentic config (cluster-only, gitignored). |

---

## Reading order for a new agent

1. **`README.md`** — repo orientation.
2. **`solver_plan.md`** ← you are here.
3. **`docs/findings.md`** — top 10 entries, newest first. Substrate
   facts you shouldn't have to re-discover.
4. **`pentathlon/demo_dl_reference/solver_<name>.md`** — for whichever
   solver you're touching.
5. **`literature/wagner_helix2fan_algorithm.md`** — only if working
   on `mayo_ldct` or any helical-rebinning project.

Then check `docs/runs/runs-index.json` for the current leaderboard
state, pick a solver that's behind, and start at Step 1 (data
investigation) or Step 2 (agentic) as appropriate for your dataset.
