# Agent4CT — Autoresearch for CT Reconstruction

📊 **Live dashboard & docs:** <https://akmaier.github.io/Agent4CT/>

🛑 **Mayo-LDCT was RESET on 2026-06-19 — read before running anything Mayo.**
The prior campaign (`search-20260614-01`) was **discarded and purged**: the
validation metric scored unrepresentative top-of-volume **boundary slices** of
L277 under the **wrong (256px Sidky) FOV mask**, and figures rendered
**upside-down** — so the dashboard, leaderboard, and the whole search trajectory
were built on a bad signal. The corrected protocol (val = **all 214 L277 slices**
+ **detector-geometry FOV 321px**, a **hard 20-min per-iter budget** with the
figure excluded, orientation fix) and a **ready-to-run loop-tick template** live
in [`docs/runs/mayo_campaign_state.md`](docs/runs/mayo_campaign_state.md) —
**START THERE** to launch the fresh `search-20260619-01`. The breast-CT / demo-DL
per-image N2I search (`search-20260618-01`) is unaffected and still finishing to
iter-20.

📋 **Working on this repo?** → start at [`solver_plan.md`](solver_plan.md)
**before** touching any solver. It's the canonical recipe for adapting
solvers to a new dataset (FBP-investigate the data, then agentic
autoresearch + TPE refinement + DDPM constrained/unconstrained variants
+ leaderboard + per-solver cross-dataset insights). Then read
[`docs/findings.md`](docs/findings.md) top-down for cross-cutting
substrate facts.

> ⚠️ **Autoresearch ≠ ablation.** If you're running an agentic loop,
> Step 2 of `solver_plan.md` lists the **six-box checklist** you MUST
> tick before each dispatch (read previous result, name failure mode,
> change ONE knob, name hypothesis, ≤ 15 min wall on iter-2+, do not
> return control to the user between iters). Previous agents have
> stumbled on this; the table of historical anti-patterns is preserved
> in Step 2 so the next agent can avoid them.

> 🛑 **Vision-module caveat (added 2026-06-11).** The agent's built-in
> image-understanding module is **NOT trained on CT / sinogram / medical
> imagery** and produces unreliable "visual" judgments on this domain
> (e.g., reading noise levels off FBPs, judging artifact severity,
> claiming two near-identical reconstructions "look different"). Multiple
> wrong conclusions have been drawn this way during user-guided debug
> sessions. **Rules**:
> - During autonomous agentic autoresearch loops the vision module is
>   acceptable as a quick coarse sanity check (the loop reports a number
>   anyway and the agent can be redirected).
> - When the **user is personally guiding** a debug session, the agent
>   MUST NOT lean on visual inspection to draw conclusions. Report
>   quantitative statistics (means, stds, RMSE, SSIM, profiles) and let
>   the user inspect the images. Do not editorialise from the pixels.
> - All visual judgments are off until a CT-specific
>   calibration/validation has been done — and that work has not
>   happened.

## 🏆 Current leaderboards

Best-of-best per solver per dataset, all metrics through the
calibrated-SSIM-headroom scoring convention
([`evaluate_calibrated`](ddssl_ldct/metrics.py)). Each leaderboard has
unified columns (`Rank | Solver | Variant | params (M) | SSIM | hr |
Source | Comparison`) with per-iteration comparison images linked
inline.

| Dataset | Top solver | SSIM | hr | Leaderboard |
|---|---|---:|---:|---|
| **Breast-CT** (128-view sparse) | Learned Primal-Dual (I=8, hidden=96, 1.49 M) | 0.9996 | **0.9062** | [`docs/leaderboards/breast_ct.md`](docs/leaderboards/breast_ct.md) |
| **Demo-DL** (Sidky ellipse, 128-view sparse) | ITNet v3 (3.7 M) | 0.9178 | 0.4676 | [`docs/leaderboards/demo_dl.md`](docs/leaderboards/demo_dl.md) |
| **Mayo-LDCT** (Wagner split, real helical) | DD-UNet supervised L2 _(live `search-20260619-01`, iter-2/20)_ | 0.9098 | 0.3390 | [`docs/leaderboards/mayo_ldct.md`](docs/leaderboards/mayo_ldct.md) |

Mayo-LDCT was **RESET on 2026-06-19** (the second reset). The 2026-06-14
rebuild (`search-20260614-01`, 19 solvers) was **discarded and purged**: its
validation metric scored the first `val_n` top-of-volume **boundary slices** of
L277 under the wrong **256px Sidky FOV mask**, and figures rendered
**upside-down**, so the search trajectory + leaderboard were built on a bad
signal. The corrected protocol (val = **all 214 L277 slices** + **321px
detector-geometry FOV**, a **hard 20-min per-iter budget** with the figure
excluded, orientation fix) + a ready-to-run loop-tick are in
[`docs/runs/mayo_campaign_state.md`](docs/runs/mayo_campaign_state.md). The fresh
`search-20260619-01` starts from a clean **HD vs LD FBP baseline**. See the
[Mayo-LDCT leaderboard](docs/leaderboards/mayo_ldct.md) for status.

[![Breast-CT champion (LPD)](docs/runs/breast-ct-calibrated-tpe-lpd-search-20260524-01/iterations/iter-0011/comparison.png)](docs/leaderboards/breast_ct.md)

*Current breast-CT champion (Learned Primal-Dual, TPE iter-11 — hr
0.9062). Click for the full leaderboard. (Mayo-LDCT is being rebuilt —
see its leaderboard for the HD/LD FBP baseline as it lands.)*

---


A continuously-running LLM agent that improves a CT reconstruction codebase
by editing it, running short experiments on the LME GPU cluster, and keeping
or discarding changes based on the resulting metrics. Modeled after
[karpathy/autoresearch](https://github.com/karpathy/autoresearch) but
generalised to **five** CT-imaging benchmarks instead of one LLM training
script, with five agents running in parallel and sharing a common scratch
pad. The CT reconstruction backbone is
[PYRO-NN](https://github.com/csyben/PYRO-NN) (Syben et al., Med. Phys. 2019).

### Datasets currently wired in

| Dataset key | Source | Split convention |
|---|---|---|
| `demo_dl` | Synthetic Sidky-style sparse-view ellipse phantoms (128 angles) | random-seed splits |
| `breast_ct` | Synthetic breast phantoms (Sidky group, 128 angles, real μ range) | random-seed splits |
| `mayo_ldct` | AAPM 2016 LDCT challenge, helical Siemens AS+, rebinned 2D fan-beam | **Wagner split** (below) |

The **Wagner split** for `mayo_ldct` (mirrors the Wagner et al. 2023 ISBI
paper's experimental setup):

```
Train: L145, L186, L209, L219     (4 patients)
Val:   L277                        (1 patient)
Test:  L014, L056, L058, L075, L123 (5 patients)
```

This split is defined once in `data/fetch_mayo_ldct.py`'s
`WAGNER_SPLITS` constant and consumed by every Mayo-touching script
(rebin, validator, autoresearch).

The five benchmarks are the **Pentathlon**:

| Folder | Challenge | Year | Where the data comes from |
|---|---|:---:|---|
| [`pentathlon/mayo_ldct/`](challenges/mayo_ldct/) | AAPM Low-Dose CT | 2016 | TCIA `LDCT-and-Projection-Data` |
| [`pentathlon/dl_sparse_view/`](challenges/dl_sparse_view/) | AAPM DL-Sparse-View CT | 2021 | AAPM/Zenodo (some files gated) |
| [`pentathlon/truect/`](challenges/truect/) | AAPM Truth-based CT | 2022 | CVIT Duke |
| [`pentathlon/ct_mar/`](challenges/ct_mar/) | AAPM CT Metal Artifact Reduction | 2024 | XCIST mirror (Box) |
| [`pentathlon/dl_spectral/`](challenges/dl_spectral/) | AAPM DL-Spectral CT | 2022 | Zenodo |

Per-challenge specs (data layout, train/val/test split, download commands)
live in [`challenges/<name>/README.md`](challenges/). Cluster operating
notes are in [`config/cluster_agent_guide.template.md`](config/cluster_agent_guide.template.md) — copy it to `config/<your-site>_cluster_agent_guide.md` and fill in the hostnames locally (that file is gitignored).
Runtime / file-IO performance notes are in
[`docs/performance.md`](docs/performance.md).

---

## How the autoresearch loop works

```
   ┌─────────────────────────────────────────────────────────────────┐
   │  5 agents (one per challenge), running concurrently             │
   │                                                                 │
   │   each iteration ≡ one 5-minute Slurm job:                      │
   │                                                                 │
   │     1.  agent reads pentathlon/shared/observations.jsonl        │
   │         (what every other agent observed in their last runs)    │
   │     2.  agent edits  pentathlon/<challenge>/solver.py           │
   │     3.  sbatch pentathlon_5min.sbatch  → 5-min training         │
   │     4.  on completion: compute val score on the held val subset │
   │     5.  decide:  keep (git commit) | discard (git reset)        │
   │     6.  append one row to pentathlon/<challenge>/results.tsv    │
   │     7.  append one observation to shared/observations.jsonl     │
   │                                                                 │
   │   every 30 iterations: stage job (1-hour wall, larger subset)   │
   │     -- checks generalisation; if iter ≫ stage, agent is told    │
   │        to regularise / shrink the model on the next iteration.  │
   └─────────────────────────────────────────────────────────────────┘
```

Two budgets, two journals:

| Budget | When | Subset | Wall-time | Journal |
|---|---|---|---|---|
| **Iteration** | every step | ~200–400 train cases | 5 min on one GPU | `pentathlon/<challenge>/results.tsv` |
| **Stage**     | every 30 iter | 3× the iteration subset | ~1 h on one GPU | `pentathlon/<challenge>/stages.tsv` |
| **Final**     | once at end | full **test** set | unbounded | `pentathlon/shared/final.tsv` |

**Test sets are never read during the loop.** Anything the agents see during
optimisation is either the iteration's val subset or the stage's val subset.
The test set is used exactly once, when you run the Pentathlon final.

### Why this design

- **5-minute iteration budget** mirrors Karpathy: it forces the agent to
  trade architecture against throughput, instead of just "train longer".
- **30-iteration stage check** catches overfitting to the small subset.
  If the iteration val keeps rising but stage val stalls or drops, the
  agent has memorised the subset — the program.md instructs it to react
  by regularising or shrinking.
- **Shared scratch pad** is the bit that goes beyond Karpathy: five agents
  on five different problems pool what they learn. If "AdamW with weight
  decay 1e-4 always beats Adam" appears on three challenges, the
  remaining two agents see that in their context window before their next
  edit.
- **Git as memory** keeps the journal auditable. Every kept change is a
  commit on a per-challenge branch; every discarded change is `git reset
  --hard` plus a `discard` row in the TSV.

---

## The Pentathlon score

The five challenges report different metrics in different units (SSIM, PSNR,
RMSE in HU, multi-metric IQ scores). To average them we use **headroom
recovered**:

```
score_c   = (metric_c − baseline_c) / (oracle_c − baseline_c)   in [0, 1]
pentathlon = mean(score_c for c in challenges)
```

| Challenge | Baseline (score 0) | Oracle (score 1) |
|---|---|---|
| Mayo LDCT | Low-dose FBP, no denoising | High-dose FBP (clean sinogram) |
| DL-Sparse-View | Sparse-view FBP, no learning | RMSE = 0 against exact truth |
| TrueCT | Uncorrected FBP | Mono-energetic phantom truth |
| CT-MAR | Uncorrected FBP (metal in) | Metal-free phantom recon |
| DL-Spectral | Per-energy FBP composite | Exact tissue maps |

Scores ∈ [0, 1] = "fraction of the gap that was closed". A negative score
means the solver is worse than doing nothing; > 1 means it beat the oracle
on that test split (possible for noisy oracles). The Pentathlon score is
the unweighted mean.

---

## The shared scratch pad — `pentathlon/shared/`

Files:

```
pentathlon/shared/
  observations.jsonl    # append-only, one line per iteration across all agents
  leaderboard.tsv       # current per-challenge best + Pentathlon score
  advice.md             # rules-of-thumb the agents accumulate
  final.tsv             # filled in by `agent4ct pentathlon final`
```

`observations.jsonl` — every iteration emits one line. The format is fixed
so other agents can parse it cheaply:

```json
{
  "ts": "2026-05-13T20:51:20Z",
  "challenge": "dl_sparse_view",
  "iter": 17,
  "agent": "claude-sonnet-4.5",
  "change_class": "optimizer",
  "rationale": "switched Adam -> AdamW; weight_decay=1e-4 to combat overfit on 400-sample train",
  "val_score": 0.62,
  "headroom": 0.41,
  "delta_vs_best": 0.04,
  "kept": true,
  "params_M": 1.8,
  "train_n": 400,
  "advice_for_others": "weight_decay helps when model > 10x dataset size"
}
```

Each agent's `program.md` instructs it to:

1. Read the **last 30 entries** from `observations.jsonl` before editing.
2. Read `advice.md` (curated by humans + agents).
3. Pick a change that is (a) not already tried recently across challenges
   and (b) consistent with what other agents found generalised.
4. Write its `advice_for_others` succinctly — one sentence, generalisable.

`advice.md` is human-bootstrapped (initial rules-of-thumb below) and grows
during the run.

---

## Anti-overfit rules — `pentathlon/shared/advice.md` (initial)

These are part of every agent's `program.md` from iteration 1:

1. **Parameter budget vs. training-set size.** Keep total trainable
   parameters under roughly `10 × train_n × pixels_per_sample` unless you
   have a clear regularisation story. A 30M-param U-Net trained on 400
   abdomen slices will overfit catastrophically; the stage check will
   surface it.
2. **Stage gap is the overfitting signal.** If `stage_score ≪ iter_score`
   on the last stage, the next iteration *must* shrink the model, increase
   weight decay, or augment more. Do not propose a bigger model.
3. **Augmentation is cheap.** Random crop, intensity jitter, flip, and
   small affine warps cost almost nothing and consistently lift small-data
   scores. Try these before bigger architectures.
4. **Self-supervised first.** Noise2Inverse-style consistency losses use
   the data we already have — they extend the effective training-set size
   without any new labels.
5. **Don't game the validation set.** The val subset is fixed; if a change
   improves val by < 1% in three iterations, that's noise, not signal —
   discard.
6. **One change per iteration.** If two things change and val moves, you
   don't know which one did it. The journal becomes useless.
7. **Cite the iteration that inspired you** in the commit message. Makes
   the journal navigable.

---

## How to run

### Prerequisites (one-time, done)

- SSH key auth to `maier@cluster.i5.informatik.uni-erlangen.de` is set up.
- `/cluster/maier/Agent4CT/` is the project root on the cluster.
- The venv + PYRO-NN are built — see `cluster/setup.sh`.

### Pull data for one or more challenges

Each subfolder has a `README.md` with the concrete download commands. The
smallest / quickest to start is **DL-Sparse-View** (~15 GB):

```bash
# Follow challenges/dl_sparse_view/README.md
```

### Start an autoresearch loop for one challenge

```bash
# From your laptop:
agent4ct pentathlon start \
    --challenge dl_sparse_view \
    --iterations 150 \
    --agent claude
```

This kicks off the iteration loop:

- Submits one 5-min Slurm job per iteration.
- The agent reads `pentathlon/shared/observations.jsonl` before each
  edit, then edits `pentathlon/dl_sparse_view/solver.py`.
- Every 30th iteration triggers a 1-hour stage job in parallel; the
  iteration loop continues with the smaller budget while the stage runs.

### Watch progress

```bash
agent4ct pentathlon tail --challenge dl_sparse_view     # live journal
agent4ct pentathlon board                               # all 5, sorted
```

### Run the five challenges in parallel

Each in its own terminal (or tmux pane / persistent agent):

```bash
agent4ct pentathlon start --challenge mayo_ldct       --iterations 150
agent4ct pentathlon start --challenge dl_sparse_view  --iterations 150
agent4ct pentathlon start --challenge truect          --iterations 150
agent4ct pentathlon start --challenge ct_mar          --iterations 150
agent4ct pentathlon start --challenge dl_spectral     --iterations 150
```

Slurm schedules them on whatever GPUs are free. No coordination needed at
the Slurm level — coordination happens through the scratch pad.

### Final Pentathlon — uses test sets exactly once

```bash
agent4ct pentathlon final
```

This:

1. Reads each challenge's current best `solver.py` (last `keep` commit).
2. Runs each on its held-out **test set** at the stage budget.
3. Computes the headroom score for each.
4. Writes `pentathlon/shared/final.tsv` with the per-challenge scores and
   the mean Pentathlon score.
5. Optionally trains an "all-rounder" — a single configuration tuned to
   maximise the Pentathlon mean rather than any one challenge.

---

## File layout

```
Agent4CT/
  README.md                         this file
  ddssl_ldct/                       reusable building blocks (PyTorch package)
    geometry.py                       FanBeamGeometry (Wagner / Siemens AS defaults)
    pyronn_projector.py               PYRO-NN-backed differentiable FBP
    models.py                         SmallUNet, TrainableBilateralFilter2d
    simulate.py                       Poisson + Gaussian LDCT noise, view split
    phantoms.py                       Random-ellipse + Shepp-Logan stand-ins
    metrics.py                        PSNR / SSIM
    training.py                       DualDomainPipeline (Wagner et al. 2023)
    harness.py                        Run / Iteration writer for docs/runs/
  challenges/                       per-challenge documentation
    mayo_ldct/README.md
    dl_sparse_view/README.md
    truect/README.md
    ct_mar/README.md
    dl_spectral/README.md
  pentathlon/                       agent-editable working area (created at first run)
    shared/
      observations.jsonl              cross-agent journal
      leaderboard.tsv                 current best per challenge
      advice.md                       rules of thumb
      final.tsv                       Pentathlon final scores
    <challenge>/
      solver.py                       the file the agent edits
      program.md                      the rules for this challenge
      results.tsv                     iteration journal
      stages.tsv                      stage journal
      runs/<iter>/                    per-iteration outputs (logs, PNGs, ckpt)
  scripts/
    run_experiment.py                 single-pass DDSSL reproduction runner
    sanity_pyronn.py                  projector / FBP sanity test
    agent4ct_record.py                CLI wrapper around ddssl_ldct.harness
    build_literature.py               PDFs + tutorials → literature/*.md
  cluster/
    setup.sh                          one-time build on a GPU compute node
    slurm/
      ddssl_smoke.sbatch              short smoke test for the recon backbone
      ddssl_train.sbatch              full DDSSL training run
      pentathlon_5min.sbatch          (planned) per-iteration job template
      pentathlon_stage.sbatch         (planned) per-stage job template
  config/
    llm_api.example.toml              copy to llm_api.toml and fill in (gitignored)
    cluster_agent_guide.template.md   cluster operating manual (committed)
    <site>_cluster_agent_guide.md     your filled-in copy (gitignored)
  docs/                             GitHub Pages site (served at akmaier.github.io/Agent4CT/)
    _config.yml                       Jekyll config (Cayman theme + overrides)
    index.md                          landing page
    setup.md  pentathlon.md  agents.md  performance.md   sub-pages
    dashboard.html                    live dashboard (fetches docs/runs/)
    assets/dashboard.{js,css}, site.css                  dashboard JS + styling
    runs/                             written by the harness, served as static JSON / TSV / PNG
      runs-index.json                 list of all runs (auto-maintained)
      observations.jsonl              cross-run append-only scratch pad
      <slug>/manifest.json + results.tsv + iterations/iter-NNNN/...
  literature/                       offline markdown copies of papers + tutorials
  papers/                             reference PDFs (gitignored)
```

---

## Run / iteration versioning + dashboard

The autoresearch loop writes its results into `docs/runs/`, which is served
by GitHub Pages as the live dashboard at
<https://akmaier.github.io/Agent4CT/dashboard.html>.

- **Run slug**: `<challenge-slug>-YYYYMMDD-NN`, e.g. `dl-sparse-view-20260513-01`.
- **Iteration**: `iter-NNNN`, four-digit zero-padded.
- Each iteration writes `observation.json`, `comparison.png`, and a
  snapshot of the solver source into its own dir. Every iteration also
  appends one line to the cross-run scratch pad at
  `docs/runs/observations.jsonl`, which the dashboard renders as cards
  with the comparison image, score, and rationale.

The Python helper that owns these conventions is
[`ddssl_ldct/harness.py`](ddssl_ldct/harness.py); the CLI wrapper agents
call from a Slurm job is
[`scripts/agent4ct_record.py`](scripts/agent4ct_record.py):

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
# → auto-commits + auto-pushes (pull --rebase --autostash before push to
#   tolerate the other 4 agents racing on the shared scratch pad). Pass
#   --no-commit / --no-push to opt out.

# Every 30 iterations (1-hour Slurm job on 3x larger subset):
python scripts/agent4ct_record.py stage \
    --slug "$SLUG" --iter 30 \
    --stage-val-score 0.55 --stage-headroom 0.30 \
    --iter-val-score 0.62 --verdict overfit \
    --notes "stage val ≪ iter val — shrink model on next iter"

# When the iteration phase ends (budget hit / no improvement / overfit /
# manual): retrain the best solver on the full train set + eval on the
# held test set in a separate Slurm job, then call finalize once.
python scripts/agent4ct_record.py finalize --slug "$SLUG" \
    --stop-reason budget \
    --final-test-score 0.71 --final-test-headroom 0.55 \
    --final-test-comparison runs/local/final_test_comparison.png \
    --notes "Retrained iter 64 on full 4000-phantom train set."
```

### Stopping a run

A run ends when **any** of these holds:

- iteration budget exhausted (default **150** — about one day per run: 150 × 5 min ≈ 12.5 GPU-h plus 5 × 1-h stage checks ≈ 17.5 h wall-clock),
- no improvement (no new `keep`) in the last 30 iterations,
- three consecutive **stage** checks return `overfit` with no recovery,
- or the operator stops it manually.

The agent picks the matching `--stop-reason` for `finalize`.

### Final test evaluation

Test sets are **never** read during the iteration phase. After the run
ends, the agent reruns the best iteration's solver on the full train
subset, evaluates on the held test set, and records the result via
`finalize`. The dashboard surfaces this as a banner on the run-detail
page.

## Result data flow — single source of truth (refactor in progress)

**Why this is being refactored.** Result numbers historically lived in **6+
places that each dropped fields and drifted**: `results.tsv` (no
PSNR/RMSE/time/params), the per-iter `observation.json`, `index/*.json`,
`runs-index.json`, the three leaderboard `.md` boards (numbers typed in by hand or
by the agent), a separate `solver_params.json`, and `datasets.json`. Two
generators even ranked champions by *different* metrics — the dashboard by **SSIM**,
the leaderboards by **headroom** — so they crowned **different "winners" from the
same data**. Any markdown/HTML that restates numbers inevitably goes stale. The
fix deletes every hand-maintained copy and lets numbers exist in exactly one place.

**The new workflow — one record → one builder → JS renders everything:**

```
observation.json   per-iter, written ONCE by the cluster job, never edited
   │               (the immutable source of truth: SSIM, headroom, PSNR, RMSE,
   │                params_M, elapsed_s, cfg_full, image paths)
   ▼   build_registry.py   (the ONLY aggregator — deterministic, no LLM)
docs/runs/index/  registry.jsonl + <challenge>.json + leaderboard.json + datasets.json
   │               ONE ranking everywhere: headroom (SSIM tiebreak); below-baseline
   │               runs still render (dimmed) so EVERY solver shows — never top-N
   ▼   dashboard.js / leaderboard.js   (render tables on the fly in the browser)
the live site — markdown carries prose only, never numbers, so nothing can drift
```

- **Production stays scripted, not agentic.** The cluster job writes the immutable
  `observation.json`; one `publish.sh` does rsync → build → validate → commit →
  push. The agent's only remaining jobs are *choosing the next config* and
  *calling `publish.sh`* — it never edits a table or hand-writes a metric.
- **A content-hash staleness gate** (`validate_registry.py`, run as a pre-commit
  hook **and** a GitHub Action) fails any commit/PR whose rendered views disagree
  with a fresh build, whose dashboard and leaderboard champions differ, or that
  links a missing image.

**Status:** the design is **adopted** (decisions fixed: full staleness gate;
archive-tag before cleanup), but the **rollout has not started** — the current
`harness.py` / `agent4ct_record.py` / `runs-index.json` path above is still live.
Full design + phased, non-disruptive migration:
[`result_register_refactor_plan.md`](result_register_refactor_plan.md).

## Status today

What works:

- PYRO-NN built on the cluster (Quadro RTX 8000 nodes, CUDA 11.8); forward
  + back-projection + Hann FBP at the full Wagner geometry, sub-second
  per slice.
- `DualDomainPipeline` (the recon backbone) is implemented and structurally
  matches Wagner et al. 2023.
- All five challenges have documented per-folder READMEs with concrete
  download instructions.
- The cluster operating guide, secret-handling rules, and runtime / I-O
  notes are documented.

What is still **design**, not yet implemented:

- The `agent4ct pentathlon` CLI driver. The shape it would take is
  documented above; the actual wrapper around `sbatch`, the journal
  writer, and the shared-scratchpad reader are not written yet.
- Per-challenge `solver.py` and `program.md` templates.
- Real data — only synthetic random-ellipse phantoms have been used so
  far. Replace `build_dataset()` in `scripts/run_experiment.py` with a
  real-data loader once the challenge data is on `/cluster/maier`.
- All-rounder training that maximises the Pentathlon mean across the five
  challenges' test sets.

---

## References

- [karpathy/autoresearch](https://github.com/karpathy/autoresearch) — the
  pattern this project generalises.
- Wagner *et al.*, **On the Benefit of Dual-domain Denoising in a
  Self-Supervised Low-dose CT Setting**, ISBI 2023.
  [arXiv:2211.01111](https://arxiv.org/abs/2211.01111) ·
  [faebstn96/helix2fan](https://github.com/faebstn96/helix2fan)
- Wagner *et al.*, **Ultra-low Parameter Denoising: Trainable Bilateral
  Filter Layers in CT**, Med. Phys. 2022.
  [arXiv:2201.10345](https://arxiv.org/abs/2201.10345) ·
  [faebstn96/trainable-bilateral-filter-source](https://github.com/faebstn96/trainable-bilateral-filter-source)
- Syben *et al.*, **PYRO-NN: Python reconstruction operators in neural
  networks**, Med. Phys. 2019.
- Hendriksen *et al.*, **Noise2Inverse**, IEEE TCI 2020.
- Zaiss, Aly, … Maier, **Agentic MR sequence development (Agent4MR)**, 2026.
  [arXiv:2604.13282](https://arxiv.org/abs/2604.13282)
