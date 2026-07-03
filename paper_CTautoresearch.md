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
| 2 | Run 9 missing methods on Breast-CT (+ fix slugs to match Mayo) | **TODO (next agent)** | §5.1, §5.3 |
| 3 | Download Medical Physics LaTeX template | **DONE** | `paper/WileyDesign.zip` (see §2) |
| 4 | Maier writing-style analysis (gentle-intro + known-operator) | **DONE** | §3 |
| 5 | 40-iteration val-hr agentic search on Breast-CT | **TODO (next agent)** | §5.2 |
| 6 | Paper outline | **DONE** | §4 |

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

**Title options**
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
break cross-dataset joins. **Proposed canonicalization (confirm with user which run is
canonical before retiring the other):**

| Breast legacy key | → canonical (Mayo-matching) key | method |
|---|---|---|
| `lpd` | `learned-primal-dual` | Learned Primal-Dual |
| `ram-zeroshot` | `ram` | RAM zero-shot |
| `tv`, `tv-v2` | `tv-iterative` | TV-iterative |
| `hammernik` | `hammernik-2017` | Hammernik VN (2017) |
| `wu-2015-l2` | `wu-2015-trainable` | Wu 2015 trainable |
| `dual-domain-unet-l2` | `dual-domain-supervised` | DD-UNet supervised L2 |
| `dual-domain-bf-l2` | `dual-domain-bilateral-supervised` | DD-BF supervised L2 |

Keep distinct: `hammernik-2017` vs `hammernik-vn` (MRI port); `wu-2015` (non-trainable)
vs `wu-2015-trainable`. Mechanism: edit `docs/runs/CURRENT_RUNIDS.json` (retire the
duplicate run-ids), rebuild registry, `validate_registry.py` must PASS, commit.

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
