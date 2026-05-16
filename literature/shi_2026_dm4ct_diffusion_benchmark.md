# Shi 2026 — DM4CT: Benchmarking Diffusion Models for CT Reconstruction

First systematic benchmark of diffusion-based CT reconstruction methods.
Evaluates **10 representative diffusion-prior methods + 7 strong baselines**
(classical FBP/SIRT, MBIR with ADMM-PDTV / FISTA-SBTV, deep image prior,
implicit neural representations, R2Gaussian, and supervised SwinIR) on a
unified pipeline across **three CT datasets** (medical 2016 Low-Dose CT
Grand Challenge, industrial LoDoInd, plus a new high-resolution
synchrotron acquisition released with the paper at Zenodo 15420527).
ICLR 2026 conference paper, code at <https://github.com/DM4CT/DM4CT>.

The paper explicitly *does not* propose a new algorithm — it provides
a unified taxonomy and apples-to-apples evaluation under controlled
sparsity (40 / 20 / 80 angles), noise levels, ring artefacts, and
limited-angle (0–¾π) configurations.

## Citation

```bibtex
@inproceedings{shi2026dm4ct,
  title     = {{DM4CT}: Benchmarking Diffusion Models for Computed Tomography Reconstruction},
  author    = {Shi, Jiayang and Pelt, Dani{\"e}l M. and Batenburg, K. Joost},
  booktitle = {Proceedings of the 14th International Conference on Learning Representations (ICLR)},
  year      = {2026},
  note      = {arXiv:2602.18589}
}
```

PDF: [papers/2602.18589_diffusion_ct.pdf](../papers/2602.18589_diffusion_ct.pdf)

## Unified taxonomy of data-consistency strategies (paper Sect. 3)

Every diffusion CT method has to inject the measurement `y = A x + noise`
into the reverse-time SDE. The paper organises the 10 methods around
**five reconstruction strategies**, each row of Table 1:

| Strategy | What the reverse step does at time t | Representative method(s) |
|---|---|---|
| **DC-grad** (data-consistency gradient) | `xₜ ← xₜ − η · ∇_{xₜ} ℒ(A x̂₀(xₜ), y)` — soft pull toward the data, η is a tunable step | DPS, PSLD (latent), Resample, DMPlug, Reddiff, HybridReg, DiffStateGrad |
| **DC-step** (separate optimisation) | After each (or some) reverse step, run `x*ₜ = argmin_x ℒ(A x, y)` — hard projection onto the data manifold, optionally re-noised back | Resample, DMPlug, Reddiff, DiffStateGrad |
| **Plug-and-play** | Alternate between (i) solve `argmin_x ℒ(A x, y)` and (ii) one reverse-diffusion denoising step as the implicit prior | DMPlug |
| **Pseudoinverse** | `gₜ ← ∇_{xₜ} ℒ(A† A x̂₀ − A† y, …)` — guide reverse process by the residual *in image space* using `A† ≈ FBP / SIRT` | MCG, PGDM, Reddiff |
| **Variational Bayesian** | Parameterise `p(x|y)` by a family (e.g. Gaussian) and fit it via gradient descent — no explicit reverse sampling | Reddiff, HybridReg, DiffStateGrad |

A method can sit in multiple columns (e.g. Reddiff combines DC-grad,
pseudoinverse and variational Bayes). The 10 methods × 5 columns matrix
is the paper's Table 1.

## Most successful approaches (paper Table 2, 6 configurations × 3 datasets)

Best diffusion-based score in each config is **bold** in the paper;
second-best is underlined. Reading across all three datasets and six
configurations:

| Strength | Methods that consistently top the diffusion column |
|---|---|
| Sparse-view + low noise (configs i, iii on medical/industrial) | **Resample**, **PGDM**, **DPS** |
| High noise / ring artefacts (config iv) | **HybridReg**, **DiffStateGrad** (variational Bayes) |
| Limited-angle (config v, 0–¾π) | **PGDM**, **Resample** |
| Real-world synchrotron (200 / 100 / 60 projections) | **DiffStateGrad**, **HybridReg** |
| Latent-space methods | **PSLD**, **Resample** (only Resample has the DC-step refinement that makes latent work) |

Cross-cutting findings from Sect. 4:

1. **Diffusion > classical and MBIR** in PSNR/SSIM across almost every
   configuration — but **supervised SwinIR (transformer trained on
   sparse↔dense pairs) often wins overall**. The diffusion advantage
   over supervised is *qualitative* (preserves uncertainty,
   structures) rather than *quantitative*.
2. **No single diffusion method dominates** all (dataset, noise,
   sparsity) settings. Pixel-space methods are more robust to
   distribution shift than latent-space ones; latent-space methods
   *need* an explicit DC-step or they produce discontinuities even at
   noise-free settings (Fig. 5 in the paper).
3. **DC-step (hard projection) beats DC-grad (soft pull) at low noise**
   but *over-fits* to measurement noise at higher noise — the optimum
   strategy shifts with the noise regime (Fig. 5).
4. **DC-grad's step-size η is brittle** (Fig. 3a): moderate η helps
   both data fidelity and PSNR, but slightly too-large η collapses the
   reverse trajectory.
5. **Pixel diffusion is more memory- and time-efficient than latent
   diffusion** despite being conceptually heavier — the encoder/decoder
   of the VQ-VAE adds substantial overhead (Fig. 7b: latent diffusion
   needs ~25.3 h VQ-VAE training + 15.2 h diffusion).
6. **INR matches diffusion on the real-world synchrotron dataset** —
   training-data quality matters more than the prior parameterisation
   when shift is severe.

## Breakdown by "most successful approach"

### **Resample** (Song et al. 2024) — single best all-rounder for sparse-view
- Latent diffusion + DDIM sampling.
- **Combines DC-grad and DC-step** with a re-noising step that puts the
  data-consistent iterate back onto the reverse trajectory.
- Wins config i medical (PSNR 32.45) and remains top-3 across
  industrial/synchrotron.

### **PGDM** (Song et al. 2023a) — strong pseudoinverse-guided sampler
- Pixel DDIM.
- Uses `A†` (approximated via FBP/SIRT) for **measurement-informed
  guidance from t = T**, so the reverse process starts data-aware.
- Top diffusion score on config v (limited-angle 40 angles) for medical.

### **DPS** (Chung et al. 2023) — canonical DC-grad baseline
- Pixel DDIM with `∇ℒ(A x̂₀ − y)` steering at every step.
- Simple, robust at low/moderate noise; brittle to η tuning.

### **HybridReg** + **DiffStateGrad** (Dou 2025 / Zirvi 2025) — variational Bayes
- Approximates posterior `p(x|y)` with a Gaussian family fit by SGD;
  no explicit reverse sampling (post-training).
- **Most robust at high noise** (config iv ring + noise) and on the
  challenging synchrotron data.
- Cost: longer optimisation (no fixed step count).

### **PSLD** (Rout et al. 2023) — early latent DC-grad
- Showcases the failure mode: producing reconstructions with
  discontinuities at noise-free settings unless an explicit DC-step is
  added.

### Bottom-line recommendation from the paper

> "Pixel diffusion models are generally more memory- and time-efficient
> than latent ones; **the best method depends on resource constraints
> and dataset size**, with diffusion methods offering a flexible
> trade-off between training cost and inference performance."

Three open challenges the paper highlights:
1. Limited CT training data (the synchrotron release helps).
2. Mismatched value ranges between simulated and real CT (no HU
   normalisation, distribution shift).
3. Latent-space diffusion needs measurement decoder-friendly
   reformulations.

## Relevance to Agent4CT

- The DM4CT benchmark **defines the apples-to-apples evaluation
  protocol** any new sparse-view method should aim to match.
- The unified taxonomy (DC-grad / DC-step / plug-and-play /
  pseudoinverse / variational Bayes) maps neatly onto our existing
  unrolled solvers: ItNet v3's per-step `R^T(R x − g)` is DC-grad
  with a *learned* η; Hammernik-VN is also DC-grad (per-step `λ_t`).
- "Resample" and "PGDM" are the closest stylistic neighbours we could
  port — both **pixel-space**, **DDIM-sampled**, with an explicit data
  consistency step around our `PyronnFanBeamProjector`.
- The synchrotron dataset (Zenodo 15420527, 200/100/60 proj
  configurations) is a ready-made benchmark we could add to
  `data/` once the disk budget allows.

Note: the paper finds **supervised SwinIR usually wins on metrics**
even though diffusion methods produce more "natural" reconstructions —
which matches our observed dominance of ItNet v3 (end-to-end supervised)
on the headroom leaderboard.
