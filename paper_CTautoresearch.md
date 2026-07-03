# paper_CTautoresearch.md — plan for the *Agentic Autoresearch for CT Reconstruction* paper

**Target journal:** *Medical Physics* (AAPM / Wiley), Research Article.
**Authors:** Andreas Maier, Lucas Kachelriess, Moritz Zaiss.
**Status:** planning + partial results (Mayo-LDCT complete and test-selected; Breast-CT
campaigns pending). Created 2026-07-03. This is the **single source of truth** for the
paper effort — update it in place.

> **How to use this doc.** Sections 1–4 are done or downloaded (results, journal
> template, Maier style guide, outline). Section 5 is the **to-do for the next
> session** (Breast-CT experiments + slug fix). Section 7 tells you exactly how to
> instruct the next agent.

---

## 0 · Status board (the 6 requested workstreams)

| # | Workstream | State | Where |
|---|---|---|---|
| 1 | Report 1% Mayo significance results | **DONE** | `docs/runs/mayo_significance_stats.md` + 3 figures |
| 2 | Run 9 missing methods on Breast-CT (+ fix slugs to match Mayo) | **TODO — BLOCKED on user decisions (§5.0 check-back)** | §5.1, §5.3 |
| 3 | Download Medical Physics LaTeX template | **DONE** | `paper/WileyDesign.zip` (see §2) |
| 4 | Maier writing-style analysis (gentle-intro + known-operator) | **DONE** | §3 |
| 5 | 40-iteration val-hr agentic search on Breast-CT | **TODO — BLOCKED on user decisions (§5.0)** | §5.2 |
| 6 | Paper outline | **DONE** | §4 |
| 7 | Title selected | **DONE** (2026-07-03) | §4 — *"Can an LLM Agent Do Reconstruction Research? …"* |
| 8 | Breast-CT leaderboard value-completion + mean±std | **NEW — TODO, tied to #2** | §5.0, §5.3 |

**2026-07-03 investigation (before restarting breast work):** established that
Breast-CT is **val-only** (Sidky DL-Sparse-View data, no test set, no patients), had
**no uniform wall budget** (unlike Mayo's 20 min), and that **18 of 25 board rows are
missing SSIM/PSNR/RMSE/time** because legacy TPE runs logged headroom only and kept no
recon. The slug-duplication (§5.3) and the missing-values are one problem. Full detail
+ open decisions in **§5.0**. Breast sweeps are **paused pending user answers**.

**Not launched on purpose:** the two Breast-CT campaigns (§5.1, §5.2) are multi-day
agentic autoresearch loops. They were *not* started unattended before this handoff —
an unsupervised loop with no oversight is this project's known failure mode. They are
written as ready-to-run specs; the next agent starts them *with* oversight.

---

## 1 · Results we already have (Mayo-LDCT) — where they are documented

- **Board (live):** `docs/runs/index/{leaderboard,mayo_ldct,datasets}.json` → GitHub
  Pages dashboard + `docs/leaderboards/mayo_ldct.md`. Champion **ITNet v1, hr 0.3756**
  (test, n=5). Rebuilt test-selected in commit `43bfc7d3`.
- **Per-solver test-best selection:** `docs/runs/mayo_testsweep_selection.json`.
- **Significance analysis (5% AND 1%):** **`docs/runs/mayo_significance_stats.md`** —
  paired t-tests (n=5, same 5 test patients) on hr/SSIM/PSNR/RMSE.
- **Figures:** `docs/runs/mayo_topsolver_significance.png` (forest plot),
  `mayo_significance_matrix.png` (solver×metric), `mayo_effort_timeline.png`
  (active-working-days timeline).
- **Method + sweep code:** `scripts/score_mayo_alliters.py`, `drive_alliters_sweep.py`,
  `score_mayo_testset.py`, `build_registry.py`; per-iter test scores in
  `docs/runs/<slug>-itertest/iter-*/final.json`.

**Headline stats for the paper** (n=5, paired; full tables in the stats doc):
- **5% level:** ITNet v1 / ITNet v2 / U-Swin are a statistical 3-way tie on hr, PSNR,
  RMSE (only SSIM separates ITNet v1 from the other two). DD-UNet-supervised and below
  are significantly different on all four metrics.
- **1% level:** the hr-tie tier grows to **four** — ITNet v1 / v2 / U-Swin **+
  Param-efficient** (param-efficient's p’s are 0.02–0.027, none < 0.01). U-Swin still
  separable on SSIM only.
- **Paradox for the discussion:** at 1% DD-UNet-supervised (0.361) *is* significantly
  worse than the champion but Param-efficient (0.324) is *not* — consistency beats
  raw gap at n=5.
- **Effort:** methods took days each; the helical→fan geometry pipeline took ~24 active
  working days (3 travel gaps dropped: May 28–31, Jun 5–6, Jun 25–27).

---

## 2 · Journal & LaTeX template (Medical Physics / Wiley)

**Downloaded:** official Wiley *New Journal Design* package → **`paper/WileyDesign.zip`**
(7.5 MB, valid). Unzips to `Optimal-Design-layout/` with:
- class file **`USG.cls`**, main example **`Optimal-Design-layout.tex`** (+ compiled
  `.pdf`), bib styles `wileyNJD-Chicago.bst` (bundled).
- Compile with **XeLaTeX** on **TeX Live 2022** (the Overleaf NJD-v5 note says TL 2023
  fails). Overleaf mirror: *Wiley NJD-v5* gallery.

**Medical Physics specifics (verify on the live guidelines — automated fetch was
blocked with 402/403):**
- **Research Article:** ≤ **10 typeset pages**; abstract ≤ **500 words**; over-length
  billed **$200/page**; color figures free (online-only journal).
- **Structured abstract, 5 parts (2021 policy, Sechopoulos, *Med Phys* 48(10),
  doi:10.1002/mp.15235):** **Background · Purpose · Methods · Results · Conclusions.**
- **References:** AMA numbered style, superscript numerals in citation order → use
  **`WileyNJD-AMA.bst`** (swap in for the bundled Chicago bst; get it from the Wiley
  NJD bundle / `github.com/schnorr/wileyorg`).
- Page + **continuous line numbering** required; SI units; math per AIP Style Manual 4th ed.
- **VERIFY on the live Author Guidelines** (blocked this session):
  `https://aapm.onlinelibrary.wiley.com/hub/journal/24734209/about/author-guidelines`
  — confirm exact page/word limits, abstract wording, figure rules, ORCID/authorship.

---

## 3 · Maier writing-style guide (for drafting in the first author's voice)

*Grounded in: "A gentle introduction to deep learning in medical image processing"
(Z Med Phys 2019, arXiv:1810.05401); "Learning with Known Operators reduces Maximum
Error Bounds" (Nature Machine Intelligence 2019, arXiv:1907.01992); "Known operator
learning… a review" (Prog. Biomed. Eng. 2022, arXiv:2108.04543); FAU Lecture Notes.*

**The introduction funnel — copy this arc:**
1. Open wide: DL is transformative and "**obviously**" relevant to (medical) imaging.
2. Pivot on "**Yet,** …": pure data-driven methods **neglect prior knowledge** and risk
   **implausible / hallucinated** results.
3. Resolve with the organizing idea: **embed the known operator** (here: the CT
   forward/back-projection + scanner physics) into the learning problem.
4. Name the **payoff triad**: known operator → **maximum error bound ↓** → **trainable
   parameters ↓** → **training-data need ↓ / synthetic→real transfer**.

**Tone & moves:** didactic first-person plural ("we… let us… note that… hence");
motivate by **breaking the black box first** (show a hallucinated recon, then ask
whether "complete black-box learning on image reconstruction" is wise); aphorisms
("all models are wrong, but some are useful"); engage Sutton's *bitter lesson* and
counter that hybrids *enrich* classical theory. Vocabulary to reuse: *known operators,
prior knowledge, hybrid ML, inductive bias, maximum error bound, precision learning,
plausible/implausible, hallucinate.*

**Rigor:** intuition **before** every equation, plain-English restatement **after**;
lightweight consistent notation (θ, L(θ), η; operators g, u; error terms e_f). Report
the **parameter-efficiency headline as a number** (his style: "<6% of the parameters,
1% inferior AUC").

**Four DO-THIS rules for our paper:**
1. Intro funnel, pivot on "Yet"; frame our differentiable projector as the *known
   operator*.
2. Each method block: intuition → math (light notation) → what it means for the recon.
3. Sell + quantify the triad — our **param-efficient solver** is the natural hook
   ("~1% of the parameters, statistically tied with the champion at the 1% level").
4. Include a figure where an unconstrained network hallucinates vs. the operator-
   constrained recon; keep the lecturer's "we/let us" voice.

---

## 4 · Paper outline (Medical Physics)

**Title (SELECTED 2026-07-03, user):**
> **Can an LLM Agent Do Reconstruction Research? Autonomous Implementation and
> Benchmarking of 26 CT Reconstruction Methods**

*(A refinement of former option 2, with "on Mayo LDCT" dropped since the paper now
spans Mayo-LDCT + Breast-CT.)*

Superseded options (kept for the record):
1. *Agentic Autoresearch for CT Reconstruction: An LLM Agent Adapts 26 Reconstruction
   Methods Under a Calibrated Headroom Metric.*
2. *Can an LLM Agent Do Reconstruction Research? Autonomous Implementation and
   Benchmarking of 26 CT Reconstruction Methods on Mayo LDCT.*
3. *The Bottleneck Is the Geometry, Not the Method: Agentic Autoresearch Across 26 CT
   Reconstruction Algorithms.*

**Structured abstract (Background / Purpose / Methods / Results / Conclusions)** — draft
in the stats doc + outline; champion ITNet v1 hr 0.376; top-tier statistical tie;
geometry pipeline the dominant human cost.

**Sections**
1. **Introduction** — iteration-bound recon-research workflow; karpathy-style autoresearch;
   gap = no CT-specific agentic loop grounded in a differentiable physics projector +
   fair calibrated metric; contribution; the "does agentic autoresearch move the
   bottleneck?" thesis. *(Write in the Maier funnel, §3.)*
2. **Methods** — 2.1 agentic loop (edit `solver.py` → ~5-min SLURM job → metric →
   accept/discard; PYRO-NN projectors; provenance in `docs/runs`); 2.2 the ~26 methods
   by family; 2.3 datasets + Wagner split (train L145/186/209/219, val L277, TEST
   L014/056/058/075/123) + the helical→fan geometry pipeline as a first-class artifact;
   2.4 calibrated-headroom metric hr = max(0,1−RMSE/LD-FBP-RMSE), FOV-masked, no upper
   clamp, mean±std over 5 patients; 2.5 per-iteration **test-selection** (best-by-test-hr;
   state it is an optimistic upper bound); 2.6 statistics (paired t-test, n=5, tie defn).
3. **Results** — 3.1 test-selected leaderboard (champion 0.376); 3.2 significance + top
   tier (3-way tie; param-efficient joins at 1%); 3.3 effort/timeline (methods=days,
   geometry≈24 working days); Breast-CT/Demo-DL generalization once §5 completes.
4. **Discussion** — n=5 optimism; test-set selection caveat (val-champion overfit
   observed); geometry cost dominates (agent didn't remove it); single-agent/single-metric
   limits; seed-fragility (FoE+Manduca seed-123 collapse).
5. **Conclusion** — an LLM agent can implement/adapt/tune ~26 methods under one metric;
   the outcome is a small tier of indistinguishable top solvers, not one winner; the
   bottleneck is geometry/data engineering → where to aim future agentic effort.

**Figures:** (1) agentic-loop schematic; (2) helical→fan rebin + FOV-masked test slice;
(3) per-iter val-vs-test hr trajectory (val-champion overfit); (4) test-selected
leaderboard bar chart (mean±std, top tier highlighted) — regenerate from
`docs/runs/index`; (5) qualitative recon panels top-tier vs LD-FBP vs full-dose +
difference maps; (6) effort/active-days timeline (`mayo_effort_timeline.png`);
(7 optional) accuracy-vs-parameters scatter (param-efficient in the tie).
**Tables:** (1) method inventory (family, params, agent-days); (2) datasets/split;
(3) test-selected leaderboard (best iter, test hr mean±std); (4) pairwise paired-t
matrix for the top tier; (5 optional) Breast-CT/Demo-DL results.

> **Reproducibility rule (project CLAUDE.md):** before submission, regenerate Figs 4/5
> and Table 3 from `docs/runs/…` so every reported hr has a backing figure path.

---

## 5 · TODO — Breast-CT experiments (next session)

**Read first:** `README.md` → `solver_plan.md` (onboarding recipe + six-box checklist)
→ `docs/findings.md`. Breast-CT board = `docs/runs/index/breast_ct.json`; allowlist =
`docs/runs/CURRENT_RUNIDS.json`. Cluster: `ssh lme-bastion`,
`/cluster/maier/Agent4CT`, `source .venv/bin/activate`, submit-cap **≤60** (user
constraint), `%8` array throttle, `HDF5_USE_FILE_LOCKING=FALSE`, `import hdf5plugin`.

### 5.0 Breast-CT facts established 2026-07-03 (answers to the four handoff questions)

Investigated before restarting any breast work. These reshape §5.1–§5.3.

**(Q) Does Breast-CT have a val AND a test set?** — **No. Val only.** `breast_ct`
is wired to the **Sidky 2021 DL-Sparse-View** challenge data
(`GEOMETRIES["breast_ct"] = _DL_SPARSE_VIEW`, `ddssl_ldct/staged_dataset.py:293`),
staged as `train`/`val` HDF5 (`fetch_dl_sparse_view.py`: 3600 train / 400 val; the
challenge's real test set is **not** in the public Zenodo release). The 400 val
samples are **independent synthetic phantom scenes — there is no patient grouping**.
The frozen paradigm therefore correctly ranks breast by **val** headroom (no
held-out test set). Reported spread today is **per-slice std over val scenes**.

**(Q) Wall-time budget vs Mayo's 20 min?** — **Breast had NO uniform budget.** Mayo
agentic iters ran under a disciplined **20-min compute budget** (`mayo_agentic_iter.sbatch`,
`--time=00:30:00` = 20 min + figure/IO headroom, no per-solver exceptions). Breast
ran a **mix**: TPE studies as single **10-hour** jobs (`breast_ct_solver_tpe.sbatch`,
`--time=10:00:00`, one job = ~20 trials) and agentic iters at **30 min–3 h**
allocations (`submit_breast_ct_v2_agentic.sh`) under a **1.5-h subprocess safeguard**
(`SEARCH_AGENT_SUBPROC_TIMEOUT_S=5400`, `learned_solver_search_agent.py:1352`).
Recorded per-iter elapsed on breast ranged **~27 s (RAM) → ~41 min (LPD, 2447 s)**.
→ **DECIDED (user, 2026-07-03): strict 20-min budget; re-do ALL breast-CT
auto-research** under it. Breast geometry (128-view fan) is far cheaper than Mayo's
rebinned helical, so 20 min is ample for all but the heaviest LPD/diffusion trials
(which must fit the budget like everything else, exactly as on Mayo).

**(Q) Leaderboard "missing values."** — Of **25 breast rows, only 7 have
SSIM/PSNR/RMSE/time**; the other 18 (legacy TPE) logged **headroom only** and their
raw recons are **not retained**, so the numbers can only be regenerated by
**re-running the config** (there is no cached recon to re-score, unlike a checkpoint
sweep). The **breast champion itself** — LPD `lpd` hr 0.906 — is one of the
metric-less legacy rows (that is why the registry crown shows SSIM `—`). Its
canonical twin `learned-primal-dual` (hr 0.829) *does* have full metrics but a lower
hr. So "complete the values" ≈ **re-run breast solvers uniformly with full logging**
(the same train-once-per-config discipline as Mayo's `score_mayo_alliters.py`), which
*simultaneously* fixes duplicates (§5.3) and lets every row report mean±std.

**(Q) "In the end replace with patient-split means and std." + user follow-up: "we
need train, test, AND val splits like Mayo — does Emil's paper discuss this?"** —
**YES, Sidky & Pan 2022 defines exactly a three-way split** (Med. Phys. 49(8):4986–
5004; repo note `literature/sidky_2022_dl_sparse_view_2109.09640.md`):

| Challenge phase | Size | Truth released? | Scoring |
|---|---:|---|---|
| **Train** | **4000** cases | ✓ (truth + 128-view sino + FBP) | — |
| **Validation** | **10** cases | ✗ (server-scored, unlimited subs) | RMSE |
| **Test** | **100** cases | ✗ (server-scored, best of 3 subs) | **mean RMSE over 100 test cases** (`s1`); tiebreak = worst-case 25×25-ROI RMSE (`s2`) |

Key facts for us:
- **There are no patients** — each case is an i.i.d. stochastic breast-phantom
  realization ("multiple realizations … for training, validation, and testing", §II.B).
  So Mayo's "mean±std over 5 patients" becomes **mean±std over N held-out test cases**
  (exchangeable, so the std is a clean per-case spread). This IS the "patient-split"
  analogue the user asked for.
- **Only the 4000-case TRAIN set is public** (Zenodo 14173522). The official 10-case
  val + 100-case test have **truth withheld** (never released) — we cannot use them.
- **Sidky ranked by mean RMSE over the 100-case test set** and, in his own Limitations
  (§IV, "Test set size"), *explicitly wishes the test set were larger* and wants to
  "study `s1`/`s2` as the number of test cases increases." → strong source-paper
  precedent for a generous held-out test split reported as a per-case mean±std.

**⇒ Plan: give breast a Mayo-style train/val/test by re-partitioning the 4000 public
cases ourselves** (clean case-level split; no patients to leak). Then breast graduates
from "val-ranked (no test set)" to the **full frozen paradigm** (train once → infer on
held-out test cases → report mean±std), and the champion/leaderboard rebuild uses
**test** numbers.

**Split DECIDED (user, 2026-07-03): `train 3600 / val 200 / test 200`.** Keeps train
identical to the current pipeline (no train-size confound); the existing 400-case
"val" pool is split into **200 val** (the search/early-stop signal, replacing the old
400) + **200 held-out test** (never seen during search; the reported number). Reported
breast metrics become **per-case mean±std over the 200 test cases** (n=200 → a far
tighter spread than Mayo's n=5). The split must be **deterministic + disjoint** (fixed
seed, recorded in the fetch/stage manifest, mirroring Mayo's `WAGNER_SPLITS` constant).

This requires **re-staging** breast on the cluster to add `test_*.h5` (+ shrink val to
200), then re-running ALL breast auto-research under the strict 20-min budget scoring
val=200 during search and reporting test=200 (user approved "re-do all breast-CT
auto-research").

> **⚠ README paradigm impact:** the frozen paradigm currently states "Breast-CT has
> NO held-out test set → val-ranked is correct." Once we stage the test split, that
> line must change to Mayo-style test reporting. **Do not edit the README frozen
> paradigm without explicit user sign-off** — flag it in the same commit that stages
> the split.

### 5.0.1 Execution runbook for the breast redo (ordered; each step gated by oversight)

1. **Re-stage** breast on the cluster: re-partition the 4000 public cases into
   `train 3600 / val 200 / test 200` with a fixed seed; write `train_*.h5`,
   `val_*.h5`, `test_*.h5`; record the split (seed + case indices) in a
   `BREAST_SPLITS` constant + stage manifest (mirror `WAGNER_SPLITS`). **Verify** the
   three splits are disjoint and counts match before any training.
2. **Wire test-set scoring** for breast: reuse the Mayo per-patient test mechanism
   (`AGENT4CT_EVAL_PATIENT`-style override in `load_val_split`) so a solver's `main()`
   infers the 200 test cases with zero solver changes; reuse
   `score_mayo_alliters.py` / `drive_alliters_sweep.py` (rename/generalize for breast).
3. **Onboard iter-1** for each solver on the new breast val=200 under the strict 20-min
   budget; confirm one clean iteration end-to-end before fanning out.
4. **Re-run auto-research** for the full solver inventory (the 9 missing §5.1 + the
   existing families, since all must be 20-min + test-scored) as a QOS-capped (≤60)
   `%8`-throttled SLURM array. Six-box protocol per dispatch.
5. **§5.2** 40-iter val-hr deeper search on the strong families, same budget.
6. **Score every iter on the 200 test cases**, pick each solver's best-by-**test-hr**
   iter (exactly the Mayo rule), then `build_registry.py` → `validate_registry.py`
   PASS → commit. Retire legacy breast run-ids in `CURRENT_RUNIDS.json` in the same
   commit. **Flag the README frozen-paradigm edit for user sign-off** (breast now has a
   test set).

### 5.1 Run the 9 methods missing from Breast-CT (agentic autoresearch, ~20 iters each)

Present on Mayo, absent on Breast-CT (by method name):
1. **Fast-diffusion ×4** — flow-pixel {constrained, unconstrained}, WDM-wavelet
   {constrained, unconstrained}
2. **ITNet v1** (Breast has v2 & v3 only — and v1 is the Mayo champion, so this is the
   highest-value gap)
3. **Manduca proj-bilateral (trainable)**
4. **Manhart PWLS-TV (ray-weighted)**
5. **Param-efficient (evolved architecture search)**
6. **TV-iterative (unrolled / supervised)**

Approach: reuse the Mayo agentic onboarding (each solver's `main()` already runs on
`AGENT4CT_DATASET=breast_ct`); onboard iter-1 with the breast geometry/config, then run
the 20-iter loop per the six-box protocol. Run-id convention: mirror Mayo →
`breast-ct-claude-agentic-<solver>-search-20260703-01`.

### 5.2 40-iteration val-hr agentic search on Breast-CT

Fresh 40-iter val-hr search (double the usual 20) seeded with the best configs learned
on Mayo + Breast so far, focused on the strong families (ITNet, dual-domain, LPD, U-Swin).
Purpose: a deeper single-method push for the paper's "given more budget" result. Val =
the breast val split (no held-out test set on breast → val-selected is legitimate here,
per the frozen paradigm). Run-id: `breast-ct-claude-agentic-<solver>-valhr40-20260703-01`.

### 5.3 Fix Breast-CT method slugs to match Mayo (prerequisite for cross-dataset tables)

The Breast board has legacy TPE-era keys that duplicate the newer agentic keys and
break cross-dataset joins. **The duplication and the "missing values" (§5.0) are the
SAME problem**: in every duplicate pair, the legacy TPE row has the *higher hr but no
SSIM/PSNR/RMSE/time*, and the agentic row has *full metrics but lower hr*. Concrete
pairs from `docs/runs/index/breast_ct.json` (hr / has-full-metrics):

| method | legacy key (hr, metrics?) | agentic key (hr, metrics?) |
|---|---|---|
| Learned Primal-Dual | `lpd` (**0.906**, ✗) — *current champion* | `learned-primal-dual` (0.829, ✓) |
| DD-UNet supervised L2 | `dual-domain-supervised` (**0.836**, ✗) | `dual-domain-unet-l2` (0.826, ✓) |
| DD-BF supervised L2 | `dual-domain-bilateral-supervised` (**0.263**, ✗) | `dual-domain-bf-l2` (0.248, ✓) |
| RAM zero-shot | `ram-zeroshot` (0.295, ✗) | `ram-zeroshot` (**0.304**, ✓) *(same key, two rows!)* |
| Wu 2015 trainable | `wu-2015-l2` (0.219, ✓) | *(agentic seed 0.2189, retired)* |
| TV-iterative | `tv`, `tv-v2` (both 0.0, ✗) | — |
| Hammernik (2017) | `hammernik` (0.455, ✗) | `hammernik-2017` (target key) |

Full canonicalization target (Mayo-matching keys):

| Breast legacy key | → canonical (Mayo-matching) key | method |
|---|---|---|
| `lpd` | `learned-primal-dual` | Learned Primal-Dual |
| `ram-zeroshot` (dup) | `ram` | RAM zero-shot |
| `tv`, `tv-v2` | `tv-iterative` | TV-iterative |
| `hammernik` | `hammernik-2017` | Hammernik VN (2017) |
| `wu-2015-l2` | `wu-2015-trainable` | Wu 2015 trainable |
| `dual-domain-unet-l2` | `dual-domain-supervised` | DD-UNet supervised L2 |
| `dual-domain-bf-l2` | `dual-domain-bilateral-supervised` | DD-BF supervised L2 |

Keep distinct: `hammernik-2017` vs `hammernik-vn` (MRI port); `wu-2015` (non-trainable)
vs `wu-2015-trainable`. Mechanism: edit `docs/runs/CURRENT_RUNIDS.json` (retire the
duplicate run-ids), rebuild registry, `validate_registry.py` must PASS, commit.

**RESOLVED by the full-rebuild decision (2026-07-03).** Because breast is being
re-partitioned (train/val/test) and **all** breast auto-research is being re-run under
the strict 20-min budget with full logging + test-set scoring, the "which twin to
keep" dilemma is moot: **every solver gets exactly one fresh, fully-logged,
test-scored row under its canonical (Mayo-matching) key.** The legacy TPE run-ids and
their duplicates are retired wholesale in `CURRENT_RUNIDS.json` as the new runs land;
the old headroom-only numbers (incl. the LPD 0.906 champion) do not carry forward. The
canonical-key table above is the **naming target** for the rebuilt board.

---

## 6 · Author list

**Andreas Maier · Lucas Kachelriess · Moritz Zaiss.**
(Confirm affiliations, ORCIDs, corresponding author, and author-order/contributions
before submission.)

---

## 7 · Handoff — how to instruct the next agent

Start the next session with a message like:

> *Read `README.md`, then `paper_CTautoresearch.md`. We are writing the Medical Physics
> paper (authors: Maier, Kachelriess, Zaiss). Mayo-LDCT is finished + test-selected;
> the stats are in `docs/runs/mayo_significance_stats.md`. Do the Breast-CT to-dos in
> §5, with oversight: (a) fix the Breast-CT slugs per §5.3 (ask me which duplicate run
> is canonical before retiring any); (b) onboard + run the 9 missing methods (§5.1) as
> a QOS-capped (≤60) SLURM-array sweep, reusing the Mayo `score_mayo_alliters.py` /
> `drive_alliters_sweep.py` tooling; (c) run the 40-iter val-hr search (§5.2). Publish
> per the registry pipeline (build_registry → validate PASS → commit). Then draft the
> paper Introduction in Maier's voice using the §3 style guide and the §4 outline, into
> a Wiley NJD LaTeX skeleton under `paper/` (template already in `paper/WileyDesign.zip`).
> Follow the frozen evaluation paradigm; keep the six-box protocol; don't run
> unattended agentic loops without checking in.*

**Operating constraints to carry over (do not relitigate):**
- Submit-cap **≤ 60** on the cluster (user has other work).
- Frozen evaluation paradigm (README): train-once → infer 5 test patients, mean±std,
  L277 never reported; Breast/Demo are val-ranked (no test set).
- Registry gate (`validate_registry.py`) must PASS before any push; leaderboards are
  prose-only, numbers live in `docs/runs/index`.
- Plans live in the repo (this file), not hidden; no stored memories.
- Vision on CT slices is unreliable — report numbers, let the user view images.

**Current cluster/session state at handoff:** Mayo sweep finished, driver exited, queue
idle. Working tree has the committed Mayo rebuild (`43bfc7d3`); the significance stats
doc, the three figures, `paper/WileyDesign.zip`, and this plan are **new/uncommitted** —
commit them with the next push.
