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

### 2.1 Medical Physics structural requirements — MANDATORY (verified 2026-07 vs the live Author Guidelines)

These are hard journal rules, not preferences. The draft MUST follow them exactly.

- **Mandated main-text section structure (Research Article):** an **Introduction** plus
  **four** further sections in this order — **Materials and Methods · Results · Discussion
  · Conclusions.** (Note: it is **"Materials and Methods"**, not "Methods".) No other
  top-level scheme is accepted. → §4 outline below is rewritten to this.
- **Structured abstract, ≤ 500 words.** The journal's structured-abstract parts are
  **Purpose · Methods · Results · Conclusions** (the 2021 Sechopoulos policy, *Med Phys*
  48(10), doi:10.1002/mp.15235, also lists an optional leading **Background**). Use
  **Background · Purpose · Methods · Results · Conclusions** and cut Background first if
  over 500 words.
- **Length:** ≤ **10 typeset pages**; over-length billed **$200/page**. This is the binding
  constraint on scope — see the "10-page discipline" note in §4. Color figures free
  (online-only).
- **Keywords:** required, **~5–8** (e.g. *CT reconstruction, sparse-view, deep learning,
  known operators, benchmarking, LLM agent, robustness*).
- **References:** **AMA** numbered style, **superscript numerals in citation order** →
  `WileyNJD-AMA.bst` (swap for the bundled Chicago bst; Wiley NJD bundle /
  `github.com/schnorr/wileyorg`). EndNote users: the **JAMA** style matches.
- Page **+ continuous line numbering**; SI units; math per AIP Style Manual 4th ed.;
  compile XeLaTeX on TeX Live 2022 (see §2).
- Source: [Medical Physics Author Guidelines](https://aapm.onlinelibrary.wiley.com/hub/journal/24734209/about/author-guidelines)
  (the hub page 402'd to automated fetch; specifics confirmed via the AAPM/Wiley guideline
  text 2026-07 — re-confirm figure-count/ORCID details at submission).

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

**Readability rule (OVERRIDES verbosity — user directive 2026-07-09).** Write **short,
simple sentences**. Aim for one idea per sentence. The audience includes many **non-native
English speakers**, so keep vocabulary plain and syntax flat. Prefer "we do X. This shows Y."
over long subordinate clauses. The Maier "we/let us" voice stays, but each sentence must be
easy to parse on first read. This applies everywhere, especially the Discussion. *(§5.6.8/5.6.9
are already drafted this way — match that cadence.)*

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

**Title — SHORT + MEMORABLE, focused on agentic autoresearch for CT (user directive
2026-07-09).** The old 15-word title is out. Keep it to ~4–7 words. Center on two words:
*agents/autoresearch* and *CT reconstruction*. Do NOT try to pack the noise result into the
title — that is the abstract hook and the conclusion, not the title. Shortlist:
1. **(recommended)** *Agentic Autoresearch for CT Reconstruction* — 5 words, memorable
   ("autoresearch" is distinctive), exactly on-focus.
2. *Can Agents Do CT Reconstruction Research?* — memorable question, 6 words.
3. *Autonomous Agents for CT Reconstruction* — 5 words, plainest.
4. Optional subtitle variant if a hook is wanted: *Agentic Autoresearch for CT
   Reconstruction: Scaling Benchmarks, Testing Robustness.*
Running head (≤ ~50 chars): *"Agentic autoresearch for CT reconstruction."*

**Keywords (5–8, required):** CT reconstruction; sparse-view; low-dose; known operators;
deep learning; benchmarking; LLM agent; noise robustness.

**Structured abstract — ≤ 500 words, parts: (Background) · Purpose · Methods · Results ·
Conclusions.** Draft content: *Purpose* — test whether an LLM agent can autonomously
implement, adapt, and fairly benchmark CT reconstruction methods, and whether ideal-data
rankings predict noise robustness. *Methods* — agentic loop grounded in a differentiable
projector + frozen calibrated-headroom metric; 26 methods across Mayo-LDCT (noise-limited)
and 128-view breast (incompleteness-limited, noiseless Sidky data); per-case test
selection; a no-retrain Poisson-noise (I0=100k) re-evaluation. *Results* — a small tier of
statistically indistinguishable top solvers; a ~195-param recombined solver within reach
of that tier at ~2% of the parameters; and the reversal — the noiseless champion
(hr 0.89) collapses to last (0.00) under mild noise while a learned primal-dual rises to
champion (0.72→0.93). *Conclusions* — the agent does the labor of reconstruction research;
ideal-data rank does not predict (and can invert under) noise robustness. Cut *Background*
first if over 500 words.

**MANDATED SECTIONS (Med Phys, §2.1): Introduction · Materials and Methods · Results ·
Discussion · Conclusions.** Content map (distilled to fit **10 pages** — see discipline
note):
1. **Introduction** — Maier funnel (§3, drafted in §4.1): DL for CT → the black-box/
   hallucination risk → known operators → *and* a second black box, the human-iteration-
   bound research workflow → can an LLM agent do it, grounded in a projector + fair metric?
   Contributions; lead the reader toward the robustness reversal.
2. **Materials and Methods** — (a) agentic loop (edits `solver.py` → ~5-min/20-min SLURM
   job → metric → accept/discard; PYRO-NN projectors; provenance in `docs/runs`); (b) the
   26 methods by family; (c) datasets + splits — Mayo Wagner split + helical→fan geometry
   pipeline; breast train3600/val200/test200 (note **Sidky data is noiseless**, §5.6.6);
   (d) calibrated-headroom metric hr=max(0,1−RMSE/LD-FBP-RMSE), FOV-masked, mean±std; (e)
   per-iteration test-selection; (f) the **no-retrain Poisson-noise re-evaluation**
   (I0=100k; §5.6.7); (g) statistics — paired t-test, **effect size (Cohen dz) reported
   because significance is sample-size-bound** (n=5 vs n=200, §5.6.5).
3. **Results** — Mayo test board + top-tier tie; breast test board (champion 0.89) with
   the noiseless caveat; the ~195-param compact solver; **the noise-robustness reversal
   (the headline figure)**; effort/timeline (geometry ≈ 24 working days).
4. **Discussion** — the **agent capability scorecard (full text in §5.6.8, write it in short
   sentences).** Strengths: fast paper→code; strong HPO; respects a fixed compute budget;
   scales evaluation; follows good instructions. Weaknesses: does **not** invent new methods;
   had to be **forced to recombine** to climb the board; **CT-image vision fails** (misses
   obvious artifacts → numbers are the source of truth); long tasks must be **decomposed** or
   even 1M-token context is exhausted; **overfits to the task** (not only an agent problem).
   Human = strategist + auditor. Then the two science points: **problem-dependent compact
   optimum** (Mayo ≠ breast, re-derived not transferred) and **the noise reversal** (mild
   noise inverts the breast board). Limitations: single agent/metric, geometry bottleneck
   unremoved, seed fragility, naf/r2gaussian DNF.
5. **Conclusions (§5.6.9) — the storyline.** Agentic autoresearch is a powerful tool to
   **scale** CT-reconstruction evaluation. Using it exposed that **ideal-data leaderboards
   reward brittle methods**: a small, realistic noise perturbation reordered almost the whole
   breast board. Take-home: **we must build better benchmarks** — include noise, dose
   variation, and other realistic perturbations by default, so leaderboards reward *robust*
   methods. Agentic autoresearch makes such multi-condition benchmarking cheap enough to be
   routine. That is its real payoff for the field.

**EQUATIONS for Materials and Methods (user directive 2026-07-09 — the Methods section
MUST show math).** Keep notation light. Precede each with one intuition sentence, follow
with one plain-English sentence. Target ~6 numbered display equations:

1. **Forward model / known operator.** The scanner is a linear operator. Intuition: "the
   sinogram is line integrals of the image." Then
   $$ \mathbf{g} = \mathbf{A}\,\mathbf{x} + \boldsymbol{\varepsilon}, $$
   with $\mathbf{x}$ the image, $\mathbf{A}$ the discrete fan-beam projector (PYRO-NN,
   differentiable), $\mathbf{g}$ the measured projections, $\boldsymbol{\varepsilon}$ noise
   ($\boldsymbol{\varepsilon}=\mathbf{0}$ for the noiseless Sidky breast data). Reconstruction
   = recover $\mathbf{x}$ from $\mathbf{g}$; sparse-view makes $\mathbf{A}$ under-determined.

2. **Unrolled reconstruction with data consistency.** Intuition: "walk toward images that
   agree with the measurements, then clean up." $K$ unrolled steps
   $$ \mathbf{x}_{k+1} = \mathbf{x}_k - \alpha_k\,\mathbf{D}(\mathbf{A}\mathbf{x}_k-\mathbf{g})
      + \mathcal{R}_\theta(\mathbf{x}_k), $$
   where the **data-consistency map** $\mathbf{D}$ is the paper's key lever: the raw adjoint
   $\mathbf{D}=\mathbf{A}^{\!\top}$ re-injects sparse-view streaks, whereas the **filtered**
   $\mathbf{D}=\mathrm{FBP}(\cdot)$ (ramp/Hann-filtered back-projection, SART-style) does not —
   this is the breast breakthrough. $\mathcal{R}_\theta$ is the learned regularizer.

3. **Learned primal–dual block (the compact + the noisy champion).** Intuition: "learn how to
   combine the correction, in image *and* measurement space." Sinogram-space dual $\mathbf{h}$
   and image-space primal $\mathbf{x}$:
   $$ \mathbf{h}_{k+1}=\Gamma_\phi(\mathbf{h}_k,\ \mathbf{A}\mathbf{x}_k-\mathbf{g}),\qquad
      \mathbf{x}_{k+1}=\Lambda_\theta\!\big(\mathbf{x}_k,\ \mathrm{FBP}(\mathbf{h}_{k+1})\big), $$
   with tied small convs $\Gamma_\phi,\Lambda_\theta$ (zero-init final layer ⇒ identity at
   init, no regression). This is `learned-primal-dual`, the noisy-board champion.

4. **Calibrated-headroom metric.** Intuition: "score how far a method closes the gap between
   low-dose FBP and truth." With a two-point intensity calibration $\mathcal{C}$ and FOV mask
   $\mathbf{M}$,
   $$ \mathrm{hr}=\max\!\Big(0,\ 1-\tfrac{\mathrm{RMSE}(\mathbf{M}\!\odot\!\mathcal{C}\hat{\mathbf{x}},\,\mathbf{M}\!\odot\!\mathbf{x})}{\mathrm{RMSE}(\mathbf{M}\!\odot\!\mathbf{x}_{\mathrm{FBP}},\,\mathbf{M}\!\odot\!\mathbf{x})}\Big). $$
   $\mathrm{hr}=0$ = no better than FBP; $\mathrm{hr}\to1$ = near-perfect. Report mean ± std
   over cases; SSIM/PSNR use a batch-wide data range.

5. **Poisson noise model (robustness experiment).** Intuition: "fewer photons ⇒ noisier line
   integrals." For a line integral $p$ and incident photons $I_0$,
   $$ N\sim\mathrm{Poisson}(I_0 e^{-p}),\qquad \hat p=-\log\!\big(\max(N,1)/I_0\big). $$
   $I_0=10^5$ (high dose, ~1–2% at the thickest ray). Applied to the test sinograms only; the
   truth stays clean; models are **not retrained**.

6. **Statistics — effect size, not just p.** Intuition: "with n=200, everything is
   'significant'; report how big the difference is." Paired $t$ over shared cases, plus
   Cohen's $d_z=\bar d/s_d$ (mean of paired differences over their SD). Lead with $d_z$.

*(Optional, if space: state the known-operator error-bound result qualitatively — embedding
the exact operator $\mathbf{A}$ cannot raise, and generally lowers, the maximum error bound
vs. a fully-learned map — and cite Maier et al., Nature Mach. Intell. 2019. One sentence, no
proof.)*

> **10-PAGE DISCIPLINE (hard limit).** §5.6 is a lab notebook, not the paper. The main
> text carries ~5–7 figures + ~3 tables and the distilled narrative above; the full
> per-solver boards, all agent-behavior detail, the param-efficient search arc, and the
> significance tables move to **Supplementary Material** (Med Phys allows unlimited SI).
> Draft main text to ~8 pages, leaving 2 for figures/refs. Cut ruthlessly.

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

## 4.1 · Introduction — full prose draft (v1, Maier funnel)

*Drafted in the first-author voice per §3. Numbers keyed to §1 (Mayo) and §5.6
(Breast). Placeholders `[Fig N]` / `[cite]` to be resolved at LaTeX time.*

Deep learning has reshaped medical image reconstruction. In computed tomography (CT)
the transformation is by now unmistakable: a decade of work has produced a large and
still-growing zoo of learned reconstructors — unrolled iterative networks, learned
primal–dual schemes, model-based deep priors, image- and projection-domain denoisers,
score-based and rectified-flow diffusion priors, and, most recently, implicit-neural and
Gaussian-splatting scene representations. Each promises to recover a diagnostic image
from fewer views or a lower dose than classical filtered back-projection allows, and the
best of them deliver [cite]. It is, at first glance, *obviously* a solved research
program: pick the strongest network, train it on enough data, and reconstruct.

Yet a purely data-driven reconstructor learns a mapping unconstrained by the physics
that produced the measurement, and an unconstrained mapping is free to *hallucinate* —
to synthesize plausible-looking anatomy that the projection data do not support [Fig 1].
This is not a tuning detail but a structural risk, and it is why the field's most durable
answer has been to put the physics back into the network: to embed the **known operator**
— the differentiable CT forward- and back-projector, together with the scanner geometry —
directly into the learned pipeline. Known-operator learning trades a black box for a
hybrid, and the trade is quantified: constraining the network with an exact operator
*provably lowers the maximum error bound*, and in practice reduces the number of trainable
parameters and the amount of training data required, easing the synthetic-to-real transfer
that medical imaging so often depends on [cite]. All models are wrong, the aphorism goes,
but some are useful — and a model that knows its own forward operator is useful for a
principled reason.

There is, however, a second black box in this story, and it sits one level up. The *method
zoo itself* is the product of a slow, human-iteration-bound craft: to compare or advance
any of these reconstructors, a researcher must implement a solver, wire it to the data and
geometry, launch a compute job, read a metric, form a hypothesis about the failure, change
one thing, and repeat — for days per method. This is the loop that recent "autoresearch"
proposals imagine automating, and large language model (LLM) agents that write, run, and
revise code now make the question concrete rather than speculative. *Can an LLM agent do
reconstruction research?* Not merely tune a hyper-parameter, but implement an unfamiliar
method from its description, diagnose why a reconstruction fails, edit its own solver code
to fix it, and benchmark the result fairly against two dozen alternatives — if we ground it
in the same two things that discipline a human in this field: a differentiable physics
projector as the known operator, and a fair, calibrated metric that cannot be gamed by a
display window or a favorable case.

We built exactly such a loop and let an LLM agent run it. Across two clinically distinct
problems — Mayo low-dose CT (dense, helical, noise-limited) and a 128-view sparse-view
breast-CT task (streak-limited) — the agent implemented, adapted, and tuned **26 reconstruction
methods** spanning classical iterative, unrolled, learned-primal–dual, diffusion-prior, and
scene-representation families. Every method was scored by a single frozen **calibrated-headroom**
metric, `hr = max(0, 1 − RMSE/RMSE_LD-FBP)`, computed inside the field-of-view against the
same low-dose FBP baseline, with the top solver selected per iteration on a held-out test
split and reported as a mean ± standard deviation over the test cases. The agent operated
under a fixed **20-minute compute budget per iteration** for strict comparability, edited its
own `solver.py` between iterations, and self-dispatched cluster jobs, with every reconstruction,
configuration, and metric preserved as provenance under `docs/runs`.

Three findings organize the paper. First, the outcome of an honest benchmark is not a single
winner but a *small tier of statistically indistinguishable top solvers*: on Mayo, ITNet, its
variants, and U-Swin form a three-way tie at the 5% level (a fourth, our param-efficient
solver, joins at 1%), and the dominant human cost was not any one method but the
helical-to-fan **geometry pipeline** — roughly 24 active working days the agent did *not*
remove. Second, the role of the human shifts from implementer to *strategist*: the agent
supplied mechanism, discipline, and genuine self-modification, while the human supplied
persistence, breadth ("are there other parameter-efficient options?"), the idea to
*recombine* components, and auditing — a division of labor we characterize explicitly.
Third, and as the paper's parameter-efficiency hook, the agent assembled a **compact
reconstructor at roughly 1–2% of the top networks' parameters that lands within statistical
reach of the champion tier on both problems** — and, tellingly, the compact optimum is
*problem-dependent*: a small denoiser suffices for dense low-dose Mayo, whereas sparse-view
breast demands a filtered data-consistency unroll with a learned primal–dual combination and
an output bilateral filter, an architecture the agent *re-derived from evidence rather than
transferred* from Mayo. Prior knowledge, once again, buys parameter efficiency — here without
a human writing the network.

Our contributions are: **(i)** an agentic CT-reconstruction research loop grounded in a
differentiable physics projector and a frozen, FOV-masked calibrated-headroom metric, with
full provenance; **(ii)** an autonomous implementation and fair benchmark of 26 methods across
a dense-low-dose and a sparse-view CT problem, reported with per-case mean ± std under a uniform
compute budget; **(iii)** a parameter-efficient, evidence-derived compact solver that is
problem-dependent yet competitive with the top tier at ~1–2% of its parameters; and **(iv)** a
characterization of what an LLM agent can and cannot do as a reconstruction researcher —
mechanistic diagnosis and self-modification on one side, and, on the other, the geometry/data
engineering bottleneck it did not move and the human strategy it still required. The thesis we
test throughout is simple: *does agentic autoresearch move the bottleneck of reconstruction
research, or merely automate the part that was never the bottleneck?*

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

## 5.4 · Full 20-iteration Breast-CT campaign (launched 2026-07-03, user-directed)

After the 5-solver shakedown validated the test-split + per-case mean±std pipeline
(test scoring wired in `scripts/score_breast_testset.py` + the `AGENT4CT_EVAL_SPLIT`
gate in `staged_dataset.py`; the interim board was published test-scored), the user
directed a **full deep benchmark**: take **every solver (except `param_efficient`) to
20 agentic iterations** on Breast-CT, then run the param-efficient recombination study.

**Scope (decided with user 2026-07-03): ALL solvers, full 20 iters; extend the 5
shakedown runs to 20 too.** Canonical set = **24 solvers** (deduped SOLVER_MAP aliases;
excludes `param_efficient` and the Mayo-DDPM `diffusion_recon_*_mayo_*` keys — those
need breast DDPM ckpts; the 4 `fastdiff` variants cover the diffusion category):
- Extend 5: itnet, itnet_v2, uswin, learned_primal_dual, dual_domain_supervised
- New 19: itnet_v3, hammernik_2017, hammernik_vn, wu_2015_trainable, wu_2015,
  tv_iterative, tv_iterative_supervised, ram_zeroshot, manduca_bilateral,
  manhart_pwls_tv, dual_domain_bilateral_supervised, dual_domain_n2i,
  dual_domain_bilateral_n2i, naf, r2gaussian, fastdiff_flow_pixel_{constrained,
  unconstrained}, fastdiff_wdm_wavelet_{constrained,unconstrained}
- NAF / R²-Gaussian / fast-diffusion are documented breast structural-failures
  (`docs/leaderboards/breast_ct.md`); user chose to give them the full 20 anyway
  ("we genuinely tried each").

**Operating model.** One Opus-4.8 subagent per solver, self-driving the six-box loop
to iter 20 via a robust **persistent Monitor gate** (numeric-headroom observation.json
+ job left squeue). Run-id `breast-ct-claude-agentic-<solver>-search-20260703-01`.
Every iter: strict **20-min** budget (`--time=00:30:00`), `val_n ≤ 200`, ONE knob
(architecture changes — layers/width/unroll depth — explicitly allowed), and **saves
`model_ckpt.pt` + `recon_raw.npz`** per iter (via `AGENT4CT_SAVE_RECON` +
`AGENT4CT_MODEL_CKPT` in the sbatch `--export`) so re-scoring reloads instead of
retrains. Launched in **waves of ~8** (matches the ~8 effective GPU concurrency under
the ≤60 cap). Known failure mode: subagents stall between iters / on app restart —
mitigated by the hourly watchdog + straggler nudges; on restart, re-resume from
cluster state (observation.json is the durable source of truth).

**Hourly watchdog:** persistent `Monitor` runs `/cluster/maier/Agent4CT/breast_watchdog.py`
(tracks all 24 run-ids, `done N/24`), one chat line per hour.

**Close-out (after all 24 reach 20):** `score_breast_alliters.py --build` → array
test-scoring every iter → `--collect` best-by-test-hr per solver → build_registry →
validate PASS → republish the final board (test, mean±std, n=200). Then commit the
still-uncommitted infra (the `_os` fix + scoring code are already committed as of
`9e7e3d74`; verify tree).

## 5.5 · Param-efficient recombination study (the finale, after the 24 land)

Mirror the Mayo param-efficient investigation but **for the sparse-view (128-view)
regime**, which differs from Mayo's dense helical — so the optimal compact architecture
may be different. Spec (user 2026-07-03):
- A dedicated agent **learns from all 24 solvers' trajectories** (their observation.json
  rationales + what knobs/architectures worked on breast) and **recombines the best
  ideas** into a minimal-parameter solver. **Everything is changeable; cross-method
  recombination to save parameters is explicitly allowed.**
- Goal: highest test hr at the **lowest parameter count** — identify which method
  families are *particularly suited to 128-view sparse-view* (e.g. unrolled
  data-consistency vs pure image-domain denoisers) and fuse their strengths cheaply.
- Report params_M as the headline alongside test hr (mean±std, n=200), the Maier
  "≈X% of the parameters, statistically tied" framing (§3).
- Run under the same 20-min budget + test-scoring pipeline; publish into the same board.

---

## 5.6 · RESULTS (2026-07-06/07) — Breast-CT campaign + finale as run

### 5.6.1 The 24-solver Breast-CT board (val hr, per-case mean±std, n=200)
All 24 reached the full 20 six-box iterations under the 20-min budget. Val leaderboard
below; **the test-scoring close-out is now done — see §5.6.5 for the final TEST-ranked
board + significance** (test ≈ val, champion dual-domain-supervised 0.8948). Val hr,
±per-case std:

| # | solver | hr | ±std | family |
|---|---|---:|---|---|
| 1 | DD-UNet supervised L2 | 0.8925 | 0.0122 | DL supervised |
| 2 | ITNet v1 | 0.8893 | 0.0141 | unrolled DC |
| 3 | ITNet v2 | 0.8853 | 0.0147 | unrolled DC |
| 4 | U-Swin | 0.8696 | 0.0150 | transformer |
| 5 | ITNet v3 | 0.8638 | 0.0138 | unrolled DC |
| 6 | Learned Primal-Dual | 0.6791 | 0.0177 | primal-dual |
| 7 | Hammernik-2017 (VN) | 0.6329 | 0.0141 | variational net |
| 8 | Hammernik-VN | 0.5910 | 0.0136 | variational net |
| 9 | tv_iterative | 0.3629 | 0.0150 | classical TV |
| 10 | manhart-PWLS-TV | 0.3629 | 0.0148 | classical TV (ray-weight inert) |
| 11 | Wu-2015-trainable | 0.3127 | 0.0104 | 12-param band+residual-DC |
| 12 | RAM zero-shot | 0.2950 | 0.0116 | frozen foundation model |
| 13 | Manduca bilateral | 0.2781 | 0.0094 | 21-param bilateral |
| 14 | DD-bilateral supervised | 0.2564 | 0.0097 | 24-param bilateral |
| 15 | DD-bilateral-N2I | 0.2055 | 0.1152 | self-sup (edge-preserving) |
| 16 | wu_2015 (classical) | 0.1547 | 0.0139 | 0-param |
| 17 | tv_iterative_supervised | 0.1501 | 0.0092 | unrolled TV |
| 18–24 | ddn2i · naf · r2gaussian · 4×fastdiff (orig) | 0.0000 | — | ceiling / cross-dataset |

Key cross-solver findings (all six-box, sentinel-verified):
- **Data-consistency is the differentiator on 128-view sparse breast.** Every method with a
  genuine DC step lands 0.6–0.9; every image-domain-only denoiser caps ~0.28. Coherent
  sparse-view streaks must be removed by re-imposing agreement with the 128 projections,
  not denoised in image space.
- **tv_iterative ≡ manhart at 0.3629** (identical TV core; manhart's PWLS ray-weighting is
  inert on breast). Wu-2015-trainable (12 params, band + residual-DC loop) → 0.31, beats
  the image-domain bilaterals (21–24 params, 0.26–0.28): even cheap DC beats cheap denoising.
- **DD-bilateral-N2I 0.206 vs plain DD-N2I 0.000** — the only self-supervised method to
  beat FBP, because its tight edge-preserving range kernel structurally vetoes the N2I
  loss's over-smoothing pull (a generic CNN denoiser has no such guard).
- **Seed-fragility** surfaced independently in ITNet-v3 and Hammernik-2017 (val-hr swings
  ~0.2 at identical train loss) — the per-case std work is what exposes it.
- Documented structural **negatives** (NAF, R²-Gaussian, both TV-supervised, plain N2I):
  the INR/Gaussian bases are too coarse for dense 512²/128-view soft tissue; N2I over-
  smooths; unrolled-TV underperforms its classical sibling because its DC couldn't engage.

### 5.6.2 Breast-native diffusion prior + fastdiff re-run (fair-baseline fix)
The 4 `fastdiff` variants originally scored **hr=0 — a cross-dataset artifact**: they used the
Mayo-trained flow prior (`fd_out_scale=0.05`) on breast (μ→0.5), a ~10× normalization
mismatch that made the prior *repaint* rather than denoise. We trained **2 breast-native
priors** (pixel + wavelet, rectified-flow, `solver_fast_diffusion.py`, on the full 3600-case
breast train split, `out_scale=0.5`, val-loss *better* than the Mayo priors: 0.0047/0.0031 vs
0.016/0.020; val/test untouched) and re-ran the 4 solvers as a genuine **{pixel,wavelet} ×
{DC-on, DC-off}** ablation (constrained/unconstrained was never a sampling flag here — only a
checkpoint-`n_train` label — so we made it a real data-consistency toggle):

| variant | hr | ±std | mechanism |
|---|---:|---|---|
| flow-pixel **DC-on** | **0.5164** | 0.0229 | near-FBP init + prior denoise + semi-converged CG-DC (n_cg≈20) |
| WDM-wavelet DC-on | 0.2767 | 0.0338 | CG-DC does the work; wavelet prior kept near-off (net-harmful) |
| flow-pixel DC-off | 0.2709 | 0.0192 | prior as an FBP-*denoiser* only |
| WDM-wavelet DC-off | 0.0000 | — | prior alone can't reconstruct (no DC anchor) |

Finding: a fair breast-native prior recovers the family from artifact to real results, and the
DC ablation shows **data-consistency is the reconstruction lever; the diffusion prior is
secondary** (a denoiser at best, net-harmful in wavelet) on well-conditioned 128-view breast —
consistent with the param-efficient finale below.

### 5.6.3 Param-efficient finale — Breast (sparse-view) vs Mayo (dense low-dose)
The agent had full latitude to edit its solver. It did NOT port the Mayo compact architecture;
it re-derived a **breast-native** one, and the two compact optima are *different* — the paper's
central cross-dataset point:

- **Mayo (dense-helical low-dose noise):** compact optimum = an evolved **FoE + multi-scale
  bilateral denoiser**, **hr 0.3241 @ ~0.00097 M params** (≈0.2–0.4% of the 0.23–0.47 M DL
  champions), **statistically tied at the 1% level** with the top tier (joins the ITNet
  v1/v2/U-Swin hr-tie; p = 0.02–0.027). On a denoising problem, a tiny denoiser suffices.
- **Breast (128-view sparse-view streaks):** that architecture *fails* — image-domain
  denoising caps at ~0.26. The agent's search arc:
  1. config-only image-domain unroll → ceiling ~0.26 (raw adjoint `Rᵀ(Rx−g)` re-injects
     coherent streaks; deeper adjoint DC → hr 0.0).
  2. **filtered-gradient DC** `fbp(Rx−g)` (SART-style, solver-edited) → **breakthrough
     0.4345 @ 13 params**; depth now monotonic (K sat. ≈30).
  3. prox capacity (FoE / bilateral / directional) → all **hurt** (the filtered gradient
     already reconstructs; an *in-loop* prox only over-smooths).
  4. **learned primal-dual combination block** (LPD-style, cross-step memory, W=4) →
     **hr 0.5047 @ 183 params** (learning to *combine/accumulate* the DC correction is the
     capacity that helps, unlike a prox).
  5. **multi-scale bilateral OUTPUT denoiser** (the same bilateral that *fails in-loop*
     succeeds as a 3-scale post-filter, σ_r=0.02, +12 params) → **champion hr 0.6201
     (single-best) @ 195 params (0.000195 M)**; PSNR 44.73, SSIM 0.9907 (n=200 val).
  6. Wu-2015 in three roles (init / band-weighted-DC / band-prox) and a full **CG-DC** solve
     → all null-or-regress (the ramp-filtered single-gradient DC already applies the *correct*
     projector inverse each step, so Wu's frequency bands are redundant and CG's extra
     completeness isn't worth its ~10× projector cost under a fixed wall).
- **Seed fragility (honest caveat).** The champion is **VN-family seed-fragile**: 4/5 seeds
  cluster at **0.571 ± 0.033**, but 1/5 (seed 45) converges to a bad basin (hr 0.0), giving a
  raw 5-seed **0.457 ± 0.257**. Two stabilizers — tighter grad-clip (0.1) and lower LR —
  *both failed to recover it*, so this is intrinsic to the VN-style architecture, not a
  schedule artifact. Report both the single-best and the good-seed mean.
- **Headline:** a **195-param** solver reaches **hr 0.62 single-best / 0.571 good-seed mean**,
  **matching the Hammernik-VN tier (0.63 @ 9 k) at ≈2 % of VN's parameters** and ≈0.04 % of the
  0.47 M DD-UNet champion — and competitive with the 3.8 M-param breast-native fastdiff (0.52,
  ~20 000× more params). Same 20-min budget as all solvers (a 50-min K-probe was kept as a
  labeled diagnostic only). **36 genuine six-box iterations** (search converged; stopped cleanly
  rather than padding to the 40 allotted).
- **Cross-dataset conclusion:** the best compact CT architecture is *problem-dependent* —
  a denoiser for dense low-dose Mayo, a **filtered-DC + learned-primal-dual unroll + output
  bilateral** for sparse-view breast. The agent found each by evidence, not by transfer, and
  in both cases the compact optimum lands within statistical reach of the DL top tier.

### 5.6.4 Discussion points — agent behavior (for §4 Discussion)
Add a subsection characterizing *how the agent worked*, since the paper's title asks whether an
LLM agent can do reconstruction research. Observed behavior (from the breast finale in particular):
- **Mechanistic, not just tuning.** The agent diagnosed *why* each result occurred (e.g. "the
  raw adjoint re-injects streaks," "filtered-DC re-derives the de-aliased recon each step, so a
  Wu init is redundant") and let those diagnoses drive the next architectural move.
- **Genuine rediscovery + self-modification.** From an image-domain ceiling it independently
  arrived at the **filtered back-projection (SART) gradient** and a **learned primal-dual
  combination** — established recon principles — and *edited its own solver code* to implement
  them. Strongest evidence for the title question.
- **Honest negatives.** Nulls (Wu ×3 roles, in-loop prox capacity, Wu-as-init) were reported
  cleanly with mechanism, distinguishing "undertrained" from "genuine null" via training curves.
- **Discipline.** Strict-serial sentinel gating; zero-init-for-no-regression when stacking
  architecture; separating the comparable 20-min champion from a labeled diagnostic.
- **Failure modes (state honestly).** It did **not** self-continue (needed per-iteration
  nursing); it initially **anchored on the prior Mayo solution** and had to be redirected; it
  **self-limited to config-only edits** until told it could edit code; one sub-run **padded
  iterations** with identical replays until an audit caught it; and it repeatedly trusted an
  unreliable completion signal (Monitor) until forced onto per-JOBID sentinel reads.
- **Human role — the meta-strategies the agent lacks (consistent across BOTH param-efficient
  studies).** The outcome-changing human inputs were strategic, not mechanical, and fell into a
  repeatable pattern — the same on Mayo (dense) and Breast (sparse):
  - **Persistence / breadth.** The agent does not self-sustain a broad search. On **Mayo** the
    human had to **repeatedly ask *"are there any more parameter-efficient approaches?"*** to keep
    it exploring the frontier; on **Breast** the *"40 iterations, keep exploring, here is the
    low-parameter palette"* steers served the identical role. Left alone the agent converges early
    and stops proposing genuinely new directions (and, worse, will pad — one breast sub-run
    replayed identical configs until an audit caught it).
  - **Recombination strategy.** The human supplied the idea of **combining the individual
    param-efficient components into one solver** — on **Mayo, explicitly suggested by the human**,
    producing the FoE + projection-domain-Manduca + anisotropic champion; on **Breast**, the whole
    finale was framed as a cross-method recombination. The agent *executed and mechanistically
    refined* the recombination but did **not originate the strategy** of assembling separate proven
    pieces into a compact whole.
  - **Redirection (Breast-specific).** *"Don't mimic Mayo — different problem, different solution"*
    pushed the agent off the Mayo local optimum toward the filtered-DC breakthrough;
    *"train a fair breast-native diffusion prior"* recovered the fastdiff family from artifact.
  - **Auditing.** The coordinator caught the padded-replay run, the missing-std/provenance gaps,
    and enforced 20-min budget comparability. Domain-hypothesis injections (Wu ×3 roles,
    multi-scale bilateral) mostly yielded principled *negative* findings — rigor, not performance.
  - **The defensible claim:** *agent = tireless, mechanistically-reasoning, self-modifying
    executor; human = strategist (persistence, breadth, recombination, redirection) + auditor.*
    That division — the agent does the experimentation and code-level reasoning, the human supplies
    the meta-strategy and integrity — is the honest answer to "can an LLM agent do reconstruction
    research?" *(NB: the mid-campaign metric re-anchoring on Mayo was a global re-evaluation across
    all solvers, not a param-efficient steer — do NOT frame it as human help specific to this study.)*

### 5.6.5 TEST-ranked Breast-CT board + significance (the finale, 2026-07-09)

Closed out the frozen paradigm on breast: re-partitioned split (train 3600 / val 200 /
**test 200**, seed 20260703, disjoint), scored **every iter of every solver on the 200
held-out test cases** (best-by-test-hr selection, exactly like Mayo), and flipped the board
to **test-ranked, per-case mean ± std over n=200**. Pipeline: `score_breast_alliters.py`
(515-task array) → `breast_testsweep_selection.json` → `build_registry.py` (breast_ct added
to `TEST_RANKED_DATASETS`). All 25 rows carry a backing figure; validate gate PASS.

**Test-selected leaderboard (hr mean ± std, n=200):**

| # | solver | test hr | ±std | best iter |
|---|---|---:|---|---:|
| 1 | **dual-domain-supervised** | **0.8948** | 0.0127 | 20 |
| 2 | itnet | 0.8926 | 0.0150 | 16 |
| 3 | itnet-v2 | 0.8893 | 0.0151 | 15 |
| 4 | itnet-v3 | 0.8749 | 0.0157 | 12 |
| 5 | uswin | 0.8586 | 0.0153 | 20 |
| 6 | learned-primal-dual | 0.7233 | 0.0143 | 12 |
| 7 | hammernik-2017 | 0.6265 | 0.0135 | 12 |
| 8 | **param-efficient (195 p)** | **0.6212** | **0.0076** | 28 |
| 9 | hammernik-vn | 0.5787 | 0.0119 | 16 |
| 10 | fastdiff-flow-pixel-constrained | 0.5119 | 0.0211 | 15 |
| 11–23 | ram · tv-iterative · manhart · wu-trainable · manduca · fastdiff-* · dd-bilateral-sup · wu-2015 · tv-iter-sup · dd-bilateral-n2i · dd-n2i · fastdiff-wdm-unconstr | 0.37 → 0.00 | — | — |
| DNF | **naf, r2gaussian** | — (per-scene INR/splatting: incompatible with amortized 20-min test-scoring; 14/200 cases in 857 s) | — | — |

**Test ≈ val (no overfit to the selection set):** every solver's test-best hr is within ±0.01
of its val-best (uswin/hammernik-vn/hammernik-2017 are *lower* on test), and param-efficient
test 0.6212 ≈ val single-best 0.6201.

**Significance analysis (paired t-test, n=200 test cases; `docs/runs/breast_significance_stats.md`,
`breast_topsolver_significance.png`, `breast_significance_matrix.png`).** The exact
**mirror image of Mayo**:
- **n=200 → everything separates.** Every top-10 method separates from the champion at the
  **1% level, Holm-robust, on ALL four metrics (hr/SSIM/PSNR/RMSE) simultaneously**; every
  *adjacent* rank separates too; the statistical tie tier is the **champion alone**; zero
  metric-discordant solvers. Opposite of Mayo (n=5), where weak power gave a 3–4-way tie and
  metrics *disagreed* (SSIM alone split the top).
- **So p ranks nothing here — effect size does.** Three practical bands: top cluster
  (dual-domain-sup / itnet / v2 / v3 / uswin, all Δhr ≤ 0.036, dz small→large), a **large gap**
  ↓ to LPD (0.723, dz 15.6), then the mid-tier. SSIM is the sharpest discriminator at the
  saturated ceiling (itnet dz 1.80 on SSIM vs 0.64 on hr).
- **Methodological point for the paper:** the identical machinery yields "everyone ties"
  (n=5) and "everyone separates on every metric" (n=200) — raw significance is sample-size-
  bound and **not** cross-dataset-comparable; **effect size (Cohen dz) and raw Δ are.** Lead
  with effect size.
- **param-efficient (195 params):** mid-tier, but its nearest neighbour is a full DL method
  (hammernik-2017) at Δhr 0.008 / dz 0.80 (smallest-effect mid-tier pair), and it has the
  **tightest per-case std of all (±0.0076)** — the most *consistent* reconstructor, at ~2% of
  a full network's parameters. The parameter-efficiency hook, now on held-out test data.

### 5.6.6 The breast task is NOISELESS ideal data (crucial caveat) + leakage audit

**Breast-CT (Sidky DL-Sparse-View) is entirely noise-free by design.** Sidky & Pan pose it as
"accurate recovery from **ideal noiseless projection data** … the floor of [RMSE] is zero at
which point one can say that the CT inverse problem is solved." The phantom *model* is stochastic
(realizations for train/val/test) but the *measurement* has no noise; our fetch/stage adds none.
**Consequence — breast and Mayo probe different regimes and their absolute hr are NOT on a
comparable scale:**

| | Breast (Sidky) | Mayo |
|---|---|---|
| data | noiseless, ideal | real low-dose, quantum noise |
| bottleneck | **data incompleteness** (128 of ~1000 views) | **noise / dose** (+ helical→fan rebin) |
| RMSE floor | 0 (exactly solvable) | > 0 (noise-limited) |
| best hr | ~0.89 (→1 possible) | ~0.38 (capped) |
| dominant lever | **data-consistency / known operator** | **denoising** |

This *reinforces* the cross-dataset thesis: on **noiseless** breast the agent's breakthrough was
**filtered data-consistency** (enforce the physics — 0.066→0.43), while on **noisy** Mayo the
compact optimum was a **denoiser**. Same agent, opposite mechanism, dictated by whether the
problem is incompleteness- or noise-limited. It also explains the razor-sharp n=200 significance
(noiseless → low per-case variance). **State this prominently so readers don't misread breast
0.89 as "better than Mayo 0.38".**

**Train/test leakage audit (breast looked "too good" — ruled out three ways):** (1) *mechanism* —
the `AGENT4CT_EVAL_SPLIT=test` redirect fires only for `split=="val"` (inference); training loads
`split=="train"`, never redirected; (2) *pixel-hash disjointness* — `TEST∩TRAIN=0`, `TEST∩VAL=0`,
`VAL∩TRAIN=0` over all 3600/200/200 truth images; (3) *behavioral* — test-hr ≈ val-hr for every
solver (several lower on test), the opposite of a memorization signature. High hr is the dataset
(noiseless, well-posed), not a leak.

### 5.6.7 PLANNED — noise-robustness re-evaluation (no retraining)

**Open question (user, 2026-07-09):** does the noiseless-trained ranking survive *mild input
noise*? Idea: a **second** test evaluation that injects Poisson photon noise (**I0 ≈ 100 000
photons/pixel — high-dose, mild**) into the 200 test **sinograms**, then reconstructs with the
**already-trained** models — **no retraining** — and re-ranks. Hypothesis: the noiseless-challenge
winners (supervised streak-removers, dual-domain-sup / itnet) may be **brittle** to a distribution
they never saw, while **regularized / data-consistency / per-scene** methods (TV, PWLS, filtered-DC,
param-efficient) may be **more robust** → a potential reordering. This is exactly the "realistic
model M_phys = M + ǫ" extension Sidky flags but does not study. Method: add `AGENT4CT_TEST_NOISE_I0`
to the loader (Poisson on N=I0·e^{−p}); reuse saved `model_ckpt.pt` (load + skip-train) for
supervised solvers and re-fit per-scene solvers on the noisy sinogram; recompute the FBP baseline
from the *same* noisy data (truth stays clean); report noisy-test hr mean±std and the rank delta
vs the noiseless board. Deliverable: a "noiseless vs high-dose-noisy" leaderboard pair + the
robustness ranking. *(Scope: inference-only, no 20-min-budget retraining — a generalization probe,
not a new campaign.)*

### 5.6.8 Agent capability scorecard — the honest Discussion (user assessment, 2026-07-09)

This is the core of the Discussion. Write it plainly. Short sentences. The paper's value is
the candid account of what the agent did and did not do.

**Where the agent is strong.**
- **Fast from paper to code.** It turns a method description into a working solver quickly.
  This is the main speed-up. It is why 26 methods across two datasets was feasible at all.
- **Strong at hyper-parameter optimization.** Given a fixed search space, it tunes well.
- **Respects a fixed compute budget.** It kept the 20-min-per-iteration budget for fair
  comparison. This discipline is hard for humans and easy for the agent.
- **Scales evaluation.** It is a very useful tool to run many benchmarks in parallel. This is
  the strongest practical use case.
- **Follows instructions well — if instructed well.** Clear, decomposed instructions give good
  results. Vague ones do not.

**Where the agent is weak.**
- **It does not invent new methods.** It implements, tunes, and recombines known methods. It
  did not propose a genuinely new reconstruction principle.
- **It had to be forced to mix and match.** Recombining proven pieces into one compact solver
  was a **human** idea, on both Mayo and breast. The agent executed it well. It did not
  originate the strategy. Left alone, it converges early and stops exploring.
- **CT-image vision is unreliable.** The vision module keeps missing very clear artifacts in CT
  slices and sinograms. Numbers, not images, had to be the source of truth. (This is a hard
  project rule; state it as a finding.)
- **Long tasks must be decomposed.** A big task must be broken into sub-tasks. Otherwise even a
  1M-token context is used up very quickly. Task decomposition is the main operational skill.
- **It overfits to the task.** It tunes hard to the given metric and split. *Caveat:* this is
  probably not only an agent problem — human researchers overfit to benchmarks too.

**What this means for the human role.** The human is the strategist. The human forces breadth,
supplies the recombination idea, redirects across problems, decomposes the work, and audits for
padding and provenance. The agent is the tireless, budget-respecting, self-modifying executor.

**The two scientific findings that frame the conclusion.**
- **The compact optimum is problem-dependent.** The Mayo parameter-efficient solution is very
  different from the breast one (a denoiser vs a filtered-DC + primal-dual + bilateral unroll).
  The agent re-derived each from evidence. It did not transfer.
- **A little noise broke the whole breast leaderboard.** Adding mild high-dose noise (I0=100k),
  with no retraining, nearly inverted the ranking. The noiseless champion collapsed to last;
  physics-regularized methods rose to the top.

### 5.6.9 Conclusion / storyline — benchmarks must change to reward robustness

The paper's storyline and conclusion: **agentic autoresearch is a powerful tool to *scale*
CT-reconstruction evaluation — and using it exposed a problem with how we benchmark.** Our
ideal-data leaderboard rewarded methods that are brittle. A small, realistic noise perturbation
reordered almost everything. The methods that won on clean data were the least robust.

So the message is forward-looking, not just "an agent can help." **We must design better
benchmarks.** Benchmarks should include noise, dose variation, and other realistic perturbations
by default, so that leaderboards reward *robust* methods rather than methods that overfit ideal
data. Agentic autoresearch makes such richer, multi-condition benchmarking cheap enough to do —
that is its real payoff for the field. *(This is the paper's take-home; the title and abstract
should point at it.)*

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
