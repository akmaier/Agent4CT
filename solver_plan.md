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
| **`mayo_ldct`** | AAPM 2016 Low-Dose CT challenge — real Siemens SOMATOM AS+ helical scans (Mayo) | 736 channels × 64 rows curved detector, helical 60-rotations, B30f truth recon, 5-mm thick slices @ 3-mm spacing. **Wagner split**: train = L145/186/209/219, val = L277, test = L014/056/058/075/123 | ✅ **Geometry: v3-final (2026-06-12).** FBP side (held fixed at Powell 2026-05-26): `pixel_spacing=0.700857`, `sod=595.362`, `sdd=1086.803`, `du=1.285044`, `det_offset=−0.040 mm`. SSR side (v3 fit SLURM 763384, joint Adam on 10 GT slices across the full L014 z-range): `sod=592.829`, `sdd=1087.268`, **`s_z=1.001665`** (NEW — per-readout z-axis stretch), `Δz=−0.159`, `post_fbp_(a, bg, hi)=(0.809, −0.000304, 0.0519)`, w_slab + 64-bin H(ρ) co-fit. L014 mean across 10 GT: SSIM 0.9571 / PSNR 40.79 dB. Bulk re-rebin of all 10 Wagner patients with v3 dispatched 2026-06-12 (SLURM 763396 → 763397 → 763398) — solvers automatically consume v3 staging on the next epoch. | [`docs/leaderboards/mayo_ldct.md`](docs/leaderboards/mayo_ldct.md) | Production constants live in `ddssl_ldct/geometry.py::MAYO_LDCT_SSR_DEFAULTS`; H(ρ) at `ddssl_ldct/mayo_geom/L014_h_radial_v3.npy`. See [`docs/findings.md` 2026-06-12 entry](docs/findings.md) for the full promotion log. |
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

> ### 🛑 STOP — read this checklist before dispatching ANYTHING
>
> Every previous agent has stumbled here. **If you cannot honestly tick
> all six boxes below before dispatching iter-N, you are doing
> something other than autoresearch.**
>
> 1. ☐ Have you READ the previous iter's `observation.json` AND
>    `results.tsv` (the NUMBERS — val_ssim, headroom, the per-iter
>    trend)? **Do NOT draw conclusions from the CT pixels** — the vision
>    module is unreliable on CT/sinogram imagery (README caveat). The
>    figure is a coarse sanity check only; the metric is the source of truth.
> 2. ☐ Can you name the SPECIFIC failure mode you observed
>    ("smoothing", "ringing", "OOM", "loss landscape too sharp",
>    "val curve plateaued", "kept memorising the train subset")?
> 3. ☐ Does your iter-N config change EXACTLY ONE knob? Multi-knob
>    changes make the journal uninterpretable.
> 4. ☐ Have you NAMED THE HYPOTHESIS in the commit message and the
>    config's `rationale` field? Format: "if I do X, I expect Y
>    because Z."
> 5. ☐ Does the iteration fit the **HARD 20-MINUTE compute budget**
>    (train + val-score; the val FIGURE renders after and does NOT
>    count)? Size epochs / n_iter / per-image steps to fit. NO exceptions.
> 6. ☐ Will you DISPATCH iter-(N+1) yourself when iter-N lands,
>    without returning control to the user? (User comes back only on
>    plateau / iter-15 / demonstrated architectural ceiling.)
>
> If all six are ✅: dispatch. If any are ❌: stop and resolve before
> sending the job.
>
> **Common past mistakes filed here so the next agent can avoid them:**
>
> | Date | Anti-pattern | Why it's not autoresearch |
> |---|---|---|
> | 2026-06-02 | Dispatched one `*_agentic_iter1.sbatch` and called it "the agentic round" | Single dispatch is an ablation. Loop = dispatch → review → propose → repeat. |
> | 2026-06-03 | Mayo iter-2 dispatched with 4 h wall (same config as iter-1) | At 4 h/iter on 4 slots, ~6 iters/day per solver = slow grid search, not autoresearch. Every iter MUST fit the 20-min budget. |
> | 2026-06-19 | `search-20260614-01` scored val on the FIRST `val_n` boundary slices of L277 under the 256px Sidky FOV mask | The whole 19-solver search + leaderboard were driven by an unrepresentative, mis-masked metric → ALL discarded + redone. Val now scores ALL 214 L277 slices with the 321px geometry FOV. |
> | 2026-05-20 | "Best hr" reported as the best of two iters in a sweep | A two-point sweep isn't autoresearch either; it's a comparison. Need ≥ 5 hypothesis-driven iters to plateau. |

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

   STOP condition (REVISED 2026-06-10 after Mayo Hammernik VN / Wu
   trainable / ItNet v1 false-STOP retroactives):
     - HARD STOP at iter-20 (was 15).
     - Soft "consider STOPping" if ALL of the below hold:
        * ≥ 10 iters completed
        * NO iter above-baseline yet (hr=0 throughout)
        * ≥ 3 architecturally-distinct config families tried (e.g.,
          for ItNet: lo-k+hi-c, hi-k+lo-c, with-residual vs without;
          for VN-class: lo-T+hi-filters, hi-T+lo-filters, lo-λ_init
          vs hi-λ_init)
        * The explored hyperparam range covers the edges of the
          published-paper recommendations (NOT just centroid sweeps)
        * The agent has explicitly logged "what hypothesis would
          still be untested" and concluded there is none.
     - The OLD "2 consecutive hr=0 = STOP" rule is RETIRED. It
       falsely terminated Hammernik VN on Mayo (Step-3 TPE later
       found hr=0.0551 at vn_T=5, n_filters=16, kernel=11, λ_init=
       2.3e-3 — corner missed by the random walk), Wu trainable on
       breast-CT (Step-3 TPE later found hr=0.3170 at 10× lower lr
       than agentic centroid), and ItNet v1 on demo-DL/breast-CT
       (cleared baseline at the TPE seed trial — Mayo verdict was
       hr=0 but synthetic datasets cleared trivially).
     - **TPE is the recommended way to close any STOP verdict.**
       Even structurally-bounded solvers should be retested with a
       full 20-trial TPE (random+prior-conditioned) on the dataset
       before filing the STOP. Configuration-space sparsity is the
       common cause of false STOPs, not architectural ceilings.
```

**Crucial**: control does NOT return to the user between iters. Claude
runs the full loop end-to-end and only reports when the plateau /
stop condition is met. A single dispatch + "I'll report when it
lands" is NOT autoresearch — it is an ablation. Autoresearch is the
WHOLE chain of dispatch → review → propose → dispatch.

### Per-iter compute budget — HARD 20 minutes (train + score)

Each iteration has a **HARD 20-minute compute budget** for solver
**train + val-score**. The driving agent MUST size epochs / n_iter /
per-image steps so train+score fits ≤ 20 min. `mayo_agentic_iter.sbatch`
sets `--time=00:30:00` (20-min budget + headroom for the figure + I/O);
a runaway is killed at 30 min. **No per-solver exceptions** (the old LPD
5h `--time` override is gone — LPD fits 20 min too, with fewer unrolled
iters / epochs).

**The val FIGURE is NOT part of the 20-min budget.** The per-iter
`comparison.png` renders the already-computed val recon (cheap, after
scoring). The heavy 6-patient leaderboard montage is a SEPARATE,
unbudgeted job (`mayo_showcase.sbatch` → `make_test_showcase.py`).

**Val metric (2026-06-19 corrected — do NOT revert):** score **ALL 214
L277 slices** (the Wagner val patient), evenly-spaced across the volume
if you pass a smaller `val_n`, with the **detector-geometry FOV mask
(237.54 mm → 321 px for L277)** — NOT the first-N boundary slices and NOT
the 256px Sidky inscribed circle. This is the default in
`evaluate_calibrated` (gated on `AGENT4CT_DATASET=mayo_ldct_2d`) +
`staged_dataset.py`. Figures display full-512 unmasked + row-flipped
(Mayo truth is stored z-flipped vs radiological-upright).

| Phase | Budget | Typical config |
|---|---|---|
| **iter-1** (feasibility) | ≤ 20 min | "Does this solver run on Mayo at all?" Seed from a cross-dataset champion; val_n=214 if it fits, else a representative subset. |
| **iter-2..N** (hypothesis) | ≤ 20 min | Single knob change. SIGNAL not CONVERGENCE. Size train/steps to fit. |

Anti-pattern: dispatching iter-2 at the same full config as iter-1, or
exceeding 20 min "just this once."

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

> **Parameter-count reporting (REQUIRED for the leaderboard / paper).**
> Every solver's `main()` result dict **MUST** include
> `"params_M": params_total / 1e6` where
> `params_total = sum(p.numel() for p in model.parameters() if p.requires_grad)`
> (use total, not trainable, for frozen/pretrained models — and mark them
> `(frozen)`). The harness records this into `observation.json` and the
> leaderboards show it in the `params (M)` column (rendered as an integer
> count for sub-0.001 M solvers). Three solvers historically computed
> `params_total` but forgot to emit `params_M`
> (`solver_dual_ddomain_bilateral_{supervised,n2i}.py`,
> `solver_wu_2015_trainable.py`) — fixed 2026-06-19; do not regress this.
> Verified trainable-param formulas for the low-parameter families:
> **bilateral** = `3 × (proj_n_bf + img_n_bf)` (3 per `TrainableBilateralFilter2d`;
> default `proj_n_bf=img_n_bf=1` → 6); **wu_2015_trainable** =
> `wu_n_bands + 2 + 2·wu_n_outer`. The Mayo generator
> (`scripts/gen_mayo_leaderboard.py`) recomputes these live from `cfg_full`;
> authoritative counts for runs whose config was pruned live in
> `docs/leaderboards/solver_params.json`.

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

### 5.1 — Per-dataset leaderboard (ONE canonical board per dataset)

There is exactly **one canonical leaderboard per dataset**:
`docs/leaderboards/<dataset>.md`. Jekyll renders that markdown **directly**
into the website page (`/leaderboards/<dataset>.html`) — the `.md` *is* the
website; there is no second copy to keep in sync. Current schema (all columns
REQUIRED):

```markdown
| Rank | Solver | Variant | params (M) | SSIM | hr | PSNR (dB) | RMSE | time (s) | Source | Comparison |
|---:|---|---|---:|---:|---:|---:|---:|---:|---|---|
| 1 | DD-UNet supervised L2 | epochs=20, lr=1e-4, unet_c=16 | 0.466 | 0.9098 | 0.3390 | 36.88 | 7.16e-4 | 419 | [results](…) | [![…](…)](…) |
```

- **`params (M)`** is mandatory — trainable params / 1e6 (integer count shown for
  sub-0.001 M solvers; `(frozen)` for pretrained-no-finetune). Every solver MUST
  emit `params_M` in its result dict (see the **Parameter-count reporting** note
  in Step 4). If a value is genuinely unrecoverable, put `-`.
- **`PSNR (dB)` / `RMSE` / `time (s)`** come from `observation.json`
  (`val_psnr` / `val_rmse` / `elapsed_s`). `-` only for pre-2026-06 runs that
  predate those fields (raw recon not retained → cannot back-compute).
- **Mayo is AUTO-GENERATED** by `scripts/gen_mayo_leaderboard.py` between the
  `<!-- AGENTIC_TABLE_START/END -->` markers. Mayo has a **held-out test set**,
  so its reported numbers follow the canonical evaluation paradigm
  (README → "Evaluation paradigm (canonical, frozen)"): the leaderboard
  **columns (SSIM, hr, PSNR, RMSE) are the MEAN ± STD over the 5 Wagner TEST
  patients** (L014/L056/L058/L075/L123) from a model **trained once** on the
  train set and run **inference-only** on each test patient — **no retraining,
  no per-patient hold-out, no cross-validation**. **Validation (L277) drives the
  search** (it selects the best iter per solver during the agentic/TPE loop) but
  **is NEVER the reported number** — Mayo is ranked by **test hr** (test-SSIM
  tiebreak), not by val. That same script ALSO syncs the Mayo champion row into
  the homepage summaries (`README.md`, `docs/index.md`) and the leaderboards
  landing (`docs/leaderboards/index.md`) — **do NOT hand-edit those summary
  rows**, they are regenerated every wave (single source of truth = the run
  data). Breast/demo boards (no held-out test set) are hand-curated and
  legitimately report/rank by the single-patient **val** metrics; backfill new
  metrics with `scripts/backfill_leaderboard_metrics.py` +
  `docs/leaderboards/solver_params.json`.

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
  dataset's full `train_n` (typically 1000–4000 phantoms). For a
  dataset with a **held-out test set** (Mayo), the reported number is
  that one fully-trained model run **inference-only on the 5 Wagner test
  patients** and summarised as **mean ± std** — the per-iter val (L277)
  scores selected the config but are **not** the reported result.
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
