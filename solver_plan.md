# Solver onboarding plan: from new dataset to leaderboard

This document is the **canonical recipe** every agent should follow when
adapting the Agent4CT solver suite to a new CT-reconstruction
benchmark. It tells you exactly what to run, in what order, and where
to write the results down — so the answer to *"how do I add dataset X?"*
is always the same.

When a new dataset lands, work through the five steps below in order.

---

## Datasets in the repo (current state of fitness)

Each row is a benchmark that ships with a `data/fetch_<name>.py`
staging script. "Fitness" is whether the dataset is currently ready
to run solvers on (geometry validated, train/val/test split frozen,
baseline FBP confirmed).

| Dataset | Source | Geometry | Fitness | Leaderboard | Notes |
|---|---|---|---|---|---|
| **`demo_dl`** | Synthetic Sidky-style sparse-view 2-D phantoms (128 angles, ~256×256, fan-beam) | n_angles = 128, fan-beam, dense FBP ground truth | ✅ Mature — leaderboard frozen 2026-05-22 | [`docs/leaderboards/demo_dl.md`](docs/leaderboards/demo_dl.md) | Fast iteration substrate. Top hr = 0.4676 (ITNet v3). |
| **`breast_ct`** | Synthetic Sidky breast phantoms (128 angles, real-tissue μ range up to 0.05 / mm) | n_angles = 128, fan-beam, breast μ-distribution | ✅ Mature — leaderboard frozen 2026-05-23 | [`docs/leaderboards/breast_ct.md`](docs/leaderboards/breast_ct.md) | Top hr = 0.9062 (LPD, I=8 hidden=96). |
| **`mayo_ldct`** | AAPM 2016 Low-Dose CT challenge — real Siemens SOMATOM AS+ helical scans (Mayo) | 736 channels × 64 rows curved detector, helical 60-rotations, B30f truth recon, 5-mm thick slices @ 3-mm spacing. **Wagner split**: train = L145/186/209/219, val = L277, test = L014/056/058/075/123 | ✅ **Geometry data-driven-fitted 2026-05-26** — `pixel_spacing=0.700857`, `sod=595.362`, `sdd=1086.803`, `du=1.285044`, `det_offset=-0.040 mm` (vs DICOM nominal: −0.32 % px, +0.11 % sdd, +0.06 % sod; det centre essentially nominal). FBP at peak GT slice: SSIM 0.9466 / PSNR 38.49 dB / RMSE 0.00059 (was 0.94 / 35.23 dB / 0.00087 with nominal DICOM values, +3.26 dB / −31 % RMSE / −35 % diff_max from fit). Ready for solver tests. | [`docs/leaderboards/mayo_ldct.md`](docs/leaderboards/mayo_ldct.md) (skeleton) | Geometry-fit values saved at `results/breast_debug/L014_fbp_geometry_fitted.json`. Use these as the default Mayo FBP geometry going forward. |
| `dl_sparse_view` | AAPM 2021 DL-Sparse-View Grand Challenge (Sidky & Pan, Med. Phys. 2022) | n_angles = 128, fan-beam | 📦 Staged but not benchmarked in current run | — | Earlier 150-iter Claude campaign (May 2026) — see `pentathlon/dl_sparse_view*/` and `docs/runs/dl-sparse-view-*-20260513-*/`. Superseded by `demo_dl` for fast iteration. |
| `dl_spectral` | AAPM 2022 DL-Spectral CT Grand Challenge | 2-material decomposition, dual-energy | 📦 Staged via `fetch_dl_spectral.py`, not yet onboarded as a benchmark | — | Future work — needs spectral-decomposition solver class. |
| `ct_mar` | AAPM 2024 CT Metal Artifact Reduction Grand Challenge | Metal-contaminated sinograms + ground truth | 📦 Staged via `fetch_ct_mar.py`, not yet onboarded | — | Future work — needs MAR-aware solver wrappers. |

Legacy in-repo benchmark (`truect`, AAPM 2022 TrueCT Challenge) is
documented in `data/INVESTIGATE_truect.md` but no staging script is
wired up yet.

---

## Solvers in the repo

Reference / agentic solvers live in
[`pentathlon/demo_dl_reference/`](pentathlon/demo_dl_reference/). Each
has a matching `solver_<name>.md` design doc with hyperparam ranges,
cross-dataset table, and "hints for next agent" notes.

### Sparse-view / dual-domain learned solvers

| Solver | Family | Description | Trainable? |
|---|---|---|---|
| **`solver_learned_primal_dual.py`** | Unrolled iterative | Adler & Öktem 2018 LPD — alternating primal/dual updates with shared-weight conv blocks. Top breast-CT hr=0.9062. | ✅ |
| **`solver_dual_ddomain_supervised.py`** | Dual-domain (sino + image) | DD-UNet with L2 supervised loss. Strong on dense-view sparse-view CT. | ✅ |
| **`solver_dual_ddomain_n2i.py`** | Dual-domain N2I | Same architecture, Noise2Inverse self-supervised loss. Structurally bounded by FBP on hr metric. | ✅ |
| **`solver_dual_ddomain_bilateral_supervised.py`** | DD + trainable BF | Wagner 2022 bilateral filter in image domain + 1-iter sino-net. Few parameters (≈24). | ✅ |
| **`solver_dual_ddomain_bilateral_n2i.py`** | DD + BF + N2I | N2I variant of the BF stack. | ✅ |
| **`solver_itnet.py`** | Iterative-net (v1) | Original ItNet 5-iter unrolled scheme. Superseded by v2/v3. | ✅ |
| **`solver_itnet_v2.py`** | Iterative-net (v2) | More stable v2 (gradient clipping + LR schedule). | ✅ |
| **`solver_itnet_v3.py`** | Iterative-net (v3) | Current canonical ItNet. Top demo_dl hr=0.4676. | ✅ |
| **`solver_uswin.py`** | Swin transformer | UNet-Swin hybrid. Competitive on demo_dl (hr=0.4655). | ✅ |
| **`solver_hammernik_2017.py`** | Variational network | Hammernik et al. 2017 (limited-angle CT). | ✅ |
| **`solver_hammernik_vn.py`** | Variational network | Hammernik MRI VN ported to CT (2018). | ✅ |

### Classical iterative

| Solver | Family | Description | Trainable? |
|---|---|---|---|
| **`solver_tv_iterative.py`** | TV + gradient descent | Total-variation regulariser, scipy / hand-rolled. Non-trainable. | ❌ |
| **`solver_tv_iterative_supervised.py`** | TV + learnable | Learnable step size + λ schedule. | ✅ |
| **`solver_tv_search.py`** | TV search wrapper | Hyperparameter search over the classical TV variant. | ❌ |
| **`solver_wu_2015.py`** | Wu 2015 FBP | Novel FBP for sparse-view CT (filter-band modulation). No params. | ❌ |
| **`solver_wu_2015_trainable.py`** | Wu 2015 trainable | Same but with learnable band coefficients. 10 params. | ✅ |
| **`solver_fbp_baseline.py`** | FBP baseline | Plain Hann-windowed FBP. Used as the `hr` baseline in all leaderboards. | ❌ |

### Diffusion / score-based

| Solver | Family | Description | Trainable? |
|---|---|---|---|
| **`solver_ddpm.py`** | DDPM training | Trains a per-dataset DDPM prior (constrained / unconstrained). Architecture: `SmallDDPM`, default ch=32. | ✅ |
| **`solver_diffusion_recon.py`** | DPS / MCG / DC-step | Posterior sampling with a pretrained DDPM prior + sinogram data-consistency steps. Used as solver, **needs** a `solver_ddpm.py` checkpoint. | (frozen prior) |
| **`solver_diffusion.py`** | Legacy stub | Earlier prototype, superseded by `solver_diffusion_recon.py`. | — |

### Implicit-neural / per-scan optimisation

| Solver | Family | Description | Trainable? |
|---|---|---|---|
| **`solver_naf.py`** | NeRF-style INR | Neural Attenuation Field — per-scan MLP fit. Best for sparse-view sparse-angle CT, **wrong inductive bias** on dense-view benchmarks. | ✅ (per-scan) |
| **`solver_r2gaussian.py`** | Gaussian splatting | R²-Gaussian (CT volume as anisotropic 3-D Gaussians). Same wrong-bias caveat as NAF. | ✅ (per-scan) |

### Foundation / pretrained zero-shot

| Solver | Family | Description | Trainable? |
|---|---|---|---|
| **`solver_ram.py`** | Foundation model | RAM (Terris 2025) zero-shot reconstruction with `ram.pth.tar`. Inference-only — no per-dataset training. Top demo_dl hr=0.4648. | ❌ (frozen) |

### Status legend per leaderboard column

- **hr (calibrated headroom) ≥ 0.5**: solver is competitive on the dataset.
- **0.1 ≤ hr < 0.5**: solver works but isn't top-tier; revisit if it's a small-params interpretability win (e.g. DD-BF, Wu 2015).
- **hr ≈ 0**: structural deal-breaker — wrong inductive bias for the dataset (NAF / R²-Gaussian on dense-view), under-trained checkpoint (DDPM on breast), or loss-formulation issue (DD-* with N2I on dense-view).
- **No entry**: not yet evaluated on this dataset.

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
**Claude-driven autoresearch loop**. *This is a loop, not a one-shot
ablation* — Claude is the agent inside the loop and is responsible
for proposing every iter after iter 1 based on what was observed.

### The loop (Claude's per-iter responsibility)

```
        ┌── start: propose iter-1 config (seed from cross-dataset leader)
        │
        ▼
   1.  Dispatch ONE iter to SLURM (claude_agentic_one_iter.sbatch).
   2.  Wait for the cluster job to finish (~5–60 min, solver-dependent).
   3.  READ docs/runs/<slug>/iterations/iter-NN/{observation.json,
       comparison.png, solver.py.txt} and results.tsv.
   4.  DIAGNOSE: what does the result + image tell us about the
       solver's failure mode? Smoothing? Hallucinations? Convergence?
       Capacity? Loss landscape?
   5.  PROPOSE iter-(N+1): pick a SINGLE knob to change, name the
       hypothesis ("if I do X, I expect Y because Z").
   6.  Goto 1.

   STOP condition: two consecutive iters with Δhr < 0.005, OR
                   iter-15 reached, OR a clear architectural ceiling
                   has been demonstrated (then file as deprioritised).
```

**Crucial**: control does NOT return to the user between iters. Claude
runs the full loop end-to-end and only reports when the plateau /
stop condition is met. A single dispatch + "I'll report when it
lands" is NOT autoresearch — it is an ablation. Autoresearch is the
WHOLE chain of dispatch → review → propose → dispatch.

### Per-iter wall budget — 5–15 minutes (NOT hours)

Karpathy's original autoresearch loop fits one iter in **5 minutes**
on purpose: the LLM must run many iters to be in the loop. Each iter
tests ONE hypothesis; full convergence is for Step 3 TPE.

| Phase | Wall budget | Purpose | Typical config |
|---|---|---|---|
| **iter-1** (feasibility) | 30–60 min | "Does this solver work on this dataset at all?" Seed from cross-dataset champion, full epochs / val_n. | epochs=10, val_n=20, train_n=400 |
| **iter-2..N** (hypothesis) | **5–15 min** | Single targeted knob change. Need SIGNAL not CONVERGENCE — if the change moves SSIM beyond val-noise, keep; else discard. | epochs=1–3, val_n=5–10, train_n=100 |

For Mayo specifically (2304 angles, 18× bigger sino than DL-Sparse-
View): expect iter-2+ walls to be 10–15 min even at the tight config
above. If a solver's per-epoch wall is so high that 1 epoch on
train_n=100 still takes > 20 min, the solver is too expensive for
the iterative protocol — file the verdict and move on.

Anti-pattern: dispatching iter-2 at the same full config as iter-1.
That gives you ~6 iters/day per solver, which is what slow grid
search looks like — not autoresearch.

### Per-iter dispatch

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
perceptual…), training-set size, init scheme, architectural variants
(EMA, attention, residual blocks, …). One change per iter so the
journal stays interpretable.

**Slug convention**: `<dataset-prefix>-claude-agentic-<solver>-search-YYYYMMDD-NN`
e.g. `breast-ct-claude-agentic-learned-primal-dual-search-20260522-01`.

**Inputs the agent reads each iter** (Claude REQUIRED to inspect ALL):
- `docs/runs/<slug>/results.tsv` — full prior iter history.
- `docs/runs/<slug>/iterations/iter-NNNN/observation.json` — per-iter
  rationale + scores.
- `docs/runs/<slug>/iterations/iter-NNNN/comparison.png` — for visual
  diagnosis of artefacts. Open the image; do not skip this.
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
