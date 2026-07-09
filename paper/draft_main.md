<!--
FIRST DRAFT v1 — Medical Physics Research Article. 2026-07-09.
Style: short sentences, plain words (many non-native readers). Maier "we/let us" voice.
Structure: mandated Med Phys sections. Target <=10 typeset pages -> keep main text ~6000
words + ~6 figures + 3 tables. Overflow lives in draft_supplement.md (Sx.y refs).
Equations LaTeX-ready; port to Wiley USG.cls at submission. AMA numbered refs.
Numbers are from the live registry (docs/runs/index) as of 2026-07-09.
-->

# Agentic Autoresearch for CT Reconstruction

**Andreas Maier¹, Lukas Kachelrieß¹˒², Siming Bayer¹, Yixing Huang³, Yan Xia², Amber Simpson⁴, Moritz Zaiss²**

¹ Pattern Recognition Lab, Friedrich-Alexander-Universität Erlangen-Nürnberg (FAU), Erlangen, Germany
² Universitätsklinikum Erlangen (UKER), Erlangen, Germany
³ Peking University, Beijing, China
⁴ University of Alberta, Edmonton, Canada

*Corresponding author: Andreas Maier (andreas.maier@fau.de).*

**Running head:** Agentic autoresearch for CT reconstruction.

**Keywords:** CT reconstruction; sparse-view; low-dose; known operators; deep learning; benchmarking; large language model agent; noise robustness.

---

## Abstract

**Background.** Deep learning has produced a large zoo of CT reconstruction methods. Comparing them fairly is slow, human-bound work. Many benchmarks also use idealized data.

**Purpose.** We test two things. First, can a large language model (LLM) agent do the labor of reconstruction research — implement, tune, and benchmark many methods on its own? Second, does a ranking measured on ideal data predict how methods behave under realistic noise?

**Methods.** We built an agentic loop. The agent edits a solver, runs a short job on a compute cluster, reads one frozen metric, and revises. The metric is a calibrated headroom score, `hr = max(0, 1 − RMSE/RMSE_FBP)`, computed inside the field of view. We grounded every method in the same differentiable fan-beam projector. We benchmarked 26 reconstruction methods on two problems: Mayo low-dose CT (noise-limited) and a 128-view sparse-view breast task from the noiseless Sidky DL-Sparse-View data (incompleteness-limited). We selected each method's best iteration on a held-out test set and report the per-case mean ± standard deviation. We then re-evaluated every trained model on the same breast test cases with mild Poisson noise added to the inputs (I₀ = 10⁵ photons), **without any retraining**.

**Results.** Honest benchmarking gives a small tier of statistically indistinguishable top methods, not one winner. From the 26 methods the agent recombined a compact solver of ~195 parameters that reaches the variational-network accuracy tier at ~2 % of its parameters. The best compact architecture is problem-dependent: a denoiser for noisy Mayo, a filtered data-consistency and learned primal-dual unroll for sparse-view breast. The agent re-derived each from evidence; it did not transfer one to the other. The noise experiment is the main finding. Mild input noise nearly inverts the breast ranking. The noiseless champion (supervised image denoiser, hr 0.89) collapses to last (hr 0.00). A learned primal-dual method rises from mid-pack to champion (hr 0.72 → 0.93). Physics-regularized and hand-crafted-smoothing methods rise; supervised image-domain denoisers collapse.

**Conclusions.** An LLM agent can implement, tune, and benchmark reconstruction methods at scale under a fixed metric and compute budget. It does not invent new methods, and it needs a human strategist. More importantly, a leaderboard measured on ideal data does not predict robustness — it can invert under a small, realistic perturbation. We argue that benchmarks must include noise and other perturbations by default, and that agentic autoresearch makes such multi-condition benchmarking cheap enough to be routine.

*(Abstract ≈ 380 words; under the 500-word limit.)*

---

## 1. Introduction

Deep learning has changed CT reconstruction. A decade of work has produced many learned reconstructors. Examples are unrolled iterative networks, learned primal-dual schemes, variational networks, image-domain and dual-domain denoisers, diffusion priors, and per-scene implicit representations.[1–5] Each promises a good image from fewer views or a lower dose than filtered back-projection (FBP). The best of them deliver. At first glance the research program looks simple: pick the strongest network, train it, reconstruct.

Yet a purely data-driven reconstructor learns a map that is not tied to the physics. Such a map is free to *hallucinate*. It can add anatomy that the measurements do not support.[6] This is not a small tuning issue. It is a structural risk. The field's durable answer is to put the physics back in. We embed the **known operator** — the differentiable forward and back projector — into the learned pipeline.[7,8] This trade is quantified. Constraining a network with an exact operator provably cannot raise, and usually lowers, the maximum error bound. In practice it also lowers the number of parameters and the amount of training data needed.[7] All models are wrong, but a model that knows its own forward operator is useful for a principled reason.

There is a second problem, and it sits one level up. The method zoo is itself the product of slow, human-bound work. To compare or improve any method, a researcher must implement a solver, wire it to the data and geometry, launch a job, read a metric, form a hypothesis, change one thing, and repeat. This takes days per method. Recent "autoresearch" ideas imagine automating this loop. LLM agents that write and run code now make the question concrete. *Can an LLM agent do reconstruction research?* Not just tune one number, but implement an unfamiliar method, diagnose why a reconstruction fails, edit its own solver code, and benchmark the result fairly — if we ground it in the same two things that discipline a human: a differentiable projector and a fair, frozen metric.

We built such a loop and let an agent run it. We benchmarked 26 methods across two CT problems. One is noise-limited (Mayo low-dose). One is incompleteness-limited (128-view sparse-view breast, from noiseless challenge data). We then asked a second question that the ideal-data benchmark cannot answer on its own: does the ranking survive a little noise? We added mild Poisson noise to the test inputs and re-scored every trained model without retraining.

This paper makes four contributions. (i) An agentic CT-reconstruction loop grounded in a differentiable projector and a frozen calibrated metric, with full provenance. (ii) An autonomous implementation and fair benchmark of 26 methods across two regimes. (iii) An evidence-derived compact solver, competitive with the top tier at ~1–2 % of its parameters, whose optimal form is problem-dependent. (iv) A robustness result — ideal-data rank does not predict, and can invert under, noise — and the practical lesson it carries for how we build benchmarks. We lead the reader toward that last point. It is the take-home.

---

## 2. Materials and Methods

### 2.1 The agentic loop

The agent works in a fixed cycle. It reads the previous result. It names the failure. It changes one thing — a hyper-parameter, or the solver code itself. It states a hypothesis. It runs one short cluster job under a fixed compute budget. It reads the frozen metric. It accepts or discards the change. Then it repeats.

Every run is grounded in the same differentiable fan-beam projector (PYRO-NN[9]). Every run writes an immutable record: the configuration, the reconstruction, the metric, and a comparison figure. Numbers, not images, are the source of truth. The agent's vision module is not reliable on CT slices (Section 4). We enforced a fixed wall-clock budget per iteration (20 min on breast; matched on Mayo) so that all methods are compared under equal compute. A human coordinator supervised the loop: setting targets, redirecting across problems, and auditing for padding and provenance gaps (Section 4).

### 2.2 A single framework for 26 methods

We describe the forward model first. The scanner is a linear operator. The sinogram is the set of line integrals of the image:
$$ \mathbf{g} = \mathbf{A}\,\mathbf{x} + \boldsymbol{\varepsilon}. \tag{1}$$
Here $\mathbf{x}$ is the image, $\mathbf{A}$ the discrete fan-beam projector, $\mathbf{g}$ the measured projections, and $\boldsymbol{\varepsilon}$ the noise. For the noiseless Sidky breast data $\boldsymbol{\varepsilon}=\mathbf{0}$. Reconstruction means recovering $\mathbf{x}$ from $\mathbf{g}$. With 128 views, $\mathbf{A}$ is under-determined.

Almost every method is one instance of the same regularized inversion. We trade fit to the measurements against a prior:
$$ \hat{\mathbf{x}}=\arg\min_{\mathbf{x}}\ \tfrac12\|\mathbf{A}\mathbf{x}-\mathbf{g}\|_2^2+\lambda\,\mathcal{R}(\mathbf{x}). \tag{2}$$
A method is fixed by two choices. The first is the **data-consistency operator** $\mathbf{D}$, which maps the measurement residual back to image space. The second is the **prior** $\mathcal{R}$. Many solvers approximate Eq. (2) by an unrolled proximal-gradient scheme of $K$ steps:
$$ \mathbf{x}_{k+1} = \mathbf{x}_k - \alpha_k\,\mathbf{D}\big(\mathbf{A}\mathbf{x}_k-\mathbf{g}\big) + \mathcal{R}_\theta(\mathbf{x}_k). \tag{3}$$
The form of $\mathbf{D}$ matters. The raw adjoint $\mathbf{D}=\mathbf{A}^{\!\top}$ back-projects the residual and re-injects sparse-view streaks. The filtered map $\mathbf{D}=\mathrm{FBP}(\cdot)$ (a ramp/Hann-filtered back-projection) does not. This distinction is the key lever on the sparse-view problem.

Some methods add a learned dual in the measurement domain. This is the learned primal-dual (LPD) block.[3] A sinogram-space memory $\mathbf{h}$ and the image $\mathbf{x}$ update together:
$$ \mathbf{h}_{k+1}=\Gamma_\phi\!\big(\mathbf{h}_k,\ \mathbf{A}\mathbf{x}_k-\mathbf{g}\big),\qquad \mathbf{x}_{k+1}=\Lambda_\theta\!\big(\mathbf{x}_k,\ \mathrm{FBP}(\mathbf{h}_{k+1})\big). \tag{4}$$
The convolutions $\Gamma_\phi,\Lambda_\theta$ are small and weight-tied. We zero-initialize their final layer, so the block is the identity at the start and cannot regress.

**Table 1** places all 26 methods on two axes. Axis A is physics engagement: how much $\mathbf{A}$ is used at inference (none, a single data-consistency step, an in-loop unrolled scheme, or a full per-scene fit). Axis B is the prior source: hand-crafted (total variation, bilateral), supervised, self-supervised (Noise2Inverse), generative (diffusion), implicit/per-scene (neural fields, Gaussian splatting), or a frozen foundation model. This one table replaces 26 separate descriptions. It also names the agent's job in the compact study: re-select the pair $(\mathbf{D},\mathcal{R})$.

*[Table 1 — solver taxonomy: family · representative solver(s) · D · R · #params. Full per-solver configurations in Supplement S1.]*

### 2.3 Datasets and geometry

**Mayo low-dose CT.** Real helical low-dose data. The main human cost here is the geometry: rebinning helical to fan-beam. We built and validated this pipeline as a first-class artifact. We use a fixed train/val/test split (Supplement S2).

**Breast (Sidky DL-Sparse-View).** 128-view 2-D fan-beam sparse-view data of synthetic breast phantoms.[10] This data is **noiseless by design**. The challenge asks whether deep learning can solve the sparse-view inverse problem, so its RMSE floor is exactly zero. We re-partitioned the public 4000-case train pool into train (3600), validation (200), and test (200), with a fixed seed and disjoint cases. We verified disjointness by image hashing: no test image appears in train or validation (Supplement S3).

### 2.4 The calibrated-headroom metric

We score how far a method closes the gap between the low-dose FBP and the truth. Let $\mathcal{C}$ be a two-point intensity calibration and $\mathbf{M}$ a field-of-view mask. The headroom is
$$ \mathrm{hr}=\max\!\Big(0,\ 1-\frac{\mathrm{RMSE}\big(\mathbf{M}\!\odot\!\mathcal{C}\hat{\mathbf{x}},\,\mathbf{M}\!\odot\!\mathbf{x}\big)}{\mathrm{RMSE}\big(\mathbf{M}\!\odot\!\mathbf{x}_{\mathrm{FBP}},\,\mathbf{M}\!\odot\!\mathbf{x}\big)}\Big). \tag{5}$$
An hr of 0 means no better than FBP. An hr near 1 means near-perfect. We select each method's best iteration on the held-out test set, then report the per-case mean ± standard deviation. SSIM and PSNR use a batch-wide data range so that they are comparable across cases.

### 2.5 The noise-robustness experiment

We ask whether the noiseless ranking survives a little input noise. We take each method's noiseless best iteration. We re-evaluate it on the **same** 200 breast test cases, but with photon-counting noise added to the sinograms. For a line integral $p$ and incident photon count $I_0$,
$$ N\sim\mathrm{Poisson}\big(I_0\,e^{-p}\big),\qquad \hat p=-\log\!\big(\max(N,1)/I_0\big). \tag{6}$$
We use $I_0=10^5$ photons. This is a high dose. The noise is mild — about 1–2 % at the thickest ray. We apply it to the test inputs only. The ground truth stays clean. Supervised solvers **load their noiseless-trained weights and skip training**. Per-scene and classical solvers re-fit on the noisy input, which is their normal inference. The FBP baseline in Eq. (5) is recomputed from the same noisy data, so hr measures improvement over the noisy FBP. This is exactly the realistic model $\mathbf{A}\mathbf{x}+\boldsymbol{\varepsilon}$ that the noiseless challenge does not study.

### 2.6 Statistics

The 200 test cases are shared by all methods, so comparisons are paired. With $n=200$ the paired $t$-test is very high-powered: tiny mean gaps reach $p<0.01$. So we report **effect size**, not just $p$. We use Cohen's $d_z=\bar d/s_d$, the mean of the paired differences over their standard deviation. We lead with $d_z$ and the raw difference. On Mayo, where $n=5$ patients, the same test has low power; we report both levels and note the contrast (Section 4).

---

## 3. Results

### 3.1 Honest benchmarking gives a tier, not a winner

On Mayo, the test-selected leaderboard has a small top tier that is statistically indistinguishable. ITNet, its variants, and U-Swin tie at the 5 % level. Our compact solver joins at the 1 % level. The champion is ITNet at hr 0.376. The full board and the pairwise significance matrix are in Supplement S4. The dominant cost was not any one method. It was the helical-to-fan geometry pipeline (~24 active working days), which the agent did not remove.

### 3.2 A compact, problem-dependent solver

From the 26 methods the agent recombined a compact solver. On breast it uses a filtered data-consistency step ($\mathbf{D}=\mathrm{FBP}$), a learned cross-step combination (Eq. 4), and an output bilateral filter. It has **195 parameters**. It reaches hr 0.62 on the breast test set. This is the variational-network tier at ~2 % of that network's parameters. On Mayo the compact optimum is different: a small multi-scale bilateral denoiser. The two compact solutions do not transfer. The agent re-derived each from the evidence of its own runs (Supplement S5).

*[Figure 1 — the agentic loop and provenance. Figure 2 — accuracy vs. parameters scatter, both datasets, compact solver highlighted.]*

### 3.3 The breast task is noiseless — and the numbers are not a leak

The breast headroom values are high (champion hr 0.89). This is a property of the data, not a flaw. The Sidky data is noiseless, so near-perfect recovery is possible in principle. We ruled out train/test leakage three ways: the training loader never sees the test split; test and train images are hash-disjoint; and test-hr tracks val-hr for every method, often lower on test (Supplement S3). The high numbers reflect an easy, well-posed problem, not memorization.

### 3.4 The main result: a little noise inverts the ranking

We now add mild noise to the inputs and re-score, with no retraining. The ranking nearly inverts. **Table 2** shows the two boards side by side.

*[Table 2 — Breast: noiseless test hr vs. noisy test hr (I₀=10⁵), same models, same 200 cases, no retraining. Columns: solver, noiseless hr±std, noisy hr, noisy SSIM, rank change.]*

The headline entries:

- **learned-primal-dual** rises from hr 0.72 (rank 6) to **0.93** (rank 1). Its noisy SSIM is 0.99. It genuinely reconstructs under noise.
- **manduca-bilateral** (21 parameters) rises from 0.28 to **0.84**. A hand-crafted smoothing prior is noise-robust by construction.
- **dual-domain-supervised**, the noiseless champion (0.89, rank 1), **collapses to 0.00** (last), with SSIM 0.35. It is a pure supervised image-domain denoiser. It was trained only on clean FBP. It is brittle to a distribution it never saw.
- The **ITNet family** (in-loop data-consistency) degrades in the middle (0.89 → 0.55–0.70). Our compact solver holds relatively well (0.62 → 0.52).
- Per-scene neural-field and Gaussian-splatting methods (NAF, R²-Gaussian) do not complete under the budget on either board and are reported as DNF.

The absolute SSIM and PSNR columns confirm that the rise of learned-primal-dual and the fall of dual-domain-supervised are real, not an artifact of the changing FBP baseline (Supplement S6).

*[Figure 3 — the reversal: paired rank plot, noiseless board → noisy board, arrows per method. Figure 4 — example reconstructions: truth, clean-input recon, noisy-input recon, for the noiseless champion vs. learned-primal-dual.]*

### 3.5 The framework explains the reversal

The reversal reads off the two axes of Table 1. Brittleness concentrates in one corner: supervised priors with weak physics engagement (image-domain maps trained on clean FBP). Robustness concentrates where the prior is not tuned to the clean distribution: either in-loop physics (learned-primal-dual keeps $\mathbf{A}$ in the loop) or a hand-crafted smoothing prior (bilateral, total variation). The framework turns the reversal from a surprise into a predicted consequence of where each method sits.

---

## 4. Discussion

### 4.1 What the agent did well, and what it did not

We give an honest scorecard. It is the core of this Discussion.

**The agent is strong in four ways.** It is fast from paper to code. It turns a method description into a working solver quickly. This is the main speed-up, and it is why 26 methods across two datasets was feasible. It is strong at hyper-parameter optimization. It respects a fixed compute budget, which is hard for humans and easy for the agent. And it scales evaluation: running many benchmarks in parallel is its strongest practical use.

**The agent is weak in several ways.** It does not invent new methods. It implements, tunes, and recombines known ones. It did not propose a new reconstruction principle. It had to be *forced* to mix and match. The idea of combining proven pieces into one compact solver was a human idea, on both datasets. The agent executed it well but did not originate it. Left alone, it converges early and stops exploring. Its CT-image vision is unreliable: it keeps missing very clear artifacts in slices and sinograms, so numbers had to be the source of truth. Long tasks must be decomposed into sub-tasks; otherwise even a one-million-token context is used up quickly. And it overfits to the task and metric. This last point is probably not only an agent problem — human researchers overfit to benchmarks too.

**The division of labor is clear.** The agent is a tireless, budget-respecting, self-modifying executor. The human is the strategist and auditor. The human forces breadth, supplies the recombination idea, redirects across problems, decomposes the work, and checks for padding and provenance. That division is the honest answer to the title question.

### 4.2 Two scientific findings

First, the best compact architecture is problem-dependent. The Mayo solution is a denoiser. The breast solution is a filtered-data-consistency and primal-dual unroll. The agent re-derived each; it did not transfer one to the other. This supports the known-operator view: the right inductive bias depends on whether the problem is noise-limited or incompleteness-limited.

Second, a little noise broke the whole breast leaderboard. The methods that won on clean data were the least robust. This is a caution for the field. Significance also depends on sample size: at $n=5$ (Mayo) the top methods tie; at $n=200$ (breast) everything separates. So $p$-values are not comparable across datasets, and effect size should lead.

### 4.3 Limitations

We used a single agent and a single metric. The geometry/data-engineering bottleneck remains; the agent did not remove it. Some solutions are seed-fragile. Two per-scene methods did not complete under the budget. The noise study uses one dose level; a dose sweep is future work.

---

## 5. Conclusions

An LLM agent can implement, tune, and benchmark 26 CT reconstruction methods under one frozen metric and a fixed compute budget. The honest outcome of such a benchmark is a small tier of indistinguishable top methods, not one winner. The agent does the labor; it does not replace the human strategist.

The more important message is about benchmarks. We used the agent to scale evaluation, and that scale exposed a problem. Our ideal-data leaderboard rewarded brittle methods. A small, realistic noise perturbation reordered almost the whole board. The methods that won on clean data collapsed. So we argue that benchmarks must change. They should include noise, dose variation, and other realistic perturbations by default, so that leaderboards reward robust methods rather than methods that overfit ideal data. Agentic autoresearch makes such multi-condition benchmarking cheap enough to be routine. That is its real payoff for the field.

---

## Acknowledgments
*(To be completed: funding, compute, data providers — Mayo, Sidky group / AAPM DL-Sparse-View Challenge.)*

## Conflict of interest
The authors declare no relevant conflicts of interest.

## Data and code availability
Boards, provenance records, and code are at `github.com/akmaier/Agent4CT`; live dashboards at `akmaier.github.io/Agent4CT`.

## References
*(AMA style, superscript numerals in order of appearance — to complete.)*
1. Sidky EY, Pan X. Report on the AAPM deep-learning sparse-view CT (DL-sparse-view) challenge. *Med Phys.* 2022. **[key dataset ref]**
2. Adler J, Öktem O. Learned primal-dual reconstruction. *IEEE TMI.* 2018.
3. Hammernik K, et al. Learning a variational network for reconstruction. *MRM.* 2018.
4. Würfl T, et al. Deep learning computed tomography / precision learning. *IEEE TMI.* 2018.
5. Ongie G, et al. Deep learning techniques for inverse problems in imaging. *IEEE JSAIT.* 2020.
6. Maier A, et al. A gentle introduction to deep learning in medical image processing. *Z Med Phys.* 2019.
7. Maier A, et al. Learning with known operators reduces maximum error bounds. *Nat Mach Intell.* 2019.
8. Maier A, et al. Known operator learning — a review. *Prog Biomed Eng.* 2022.
9. Syben C, et al. PYRO-NN: Python reconstruction operators in neural networks. *Med Phys.* 2019.
10. Sidky EY, Pan X. DL-sparse-view CT challenge dataset (Zenodo).
*(Add: unrolled/ITNet, U-Swin, diffusion-prior, NAF, Gaussian-splatting, Noise2Inverse, RAM foundation-model, Wu-2015 band decomposition, Manduca bilateral, PWLS-TV, Karpathy autoresearch note, LLM-agent coding refs.)*
