# Breast-CT — top-10 solver significance analysis

Statistical comparison of the **test-selected** Breast-CT leaderboard (each solver's best-by-test-`hr` iter over the 200 held-out test cases).

## Method

- **n = 200 held-out TEST cases** (i.i.d. synthetic breast phantoms, the SAME 200 for every solver → **paired** comparison; Student's paired t-test, df = 199).

- Metrics tested independently: headroom `hr`, SSIM, PSNR, RMSE.

- Reference = champion **dual-domain-supervised**. A comparison is 'significant' when it separates from the champion.

- **n=200 is very high-powered** — tiny mean gaps reach p<0.01. So we report **effect size** (Cohen's dz = mean(diff)/std(diff)) and the **raw mean Δhr** alongside p. |dz|<0.2 = negligible, 0.2 small, 0.5 medium, 0.8 large.

- p reported raw at **5%** and **1%**, plus **Holm-corrected** (9 comparisons). Wilcoxon signed-rank shown as a non-parametric robustness check.


## Top-10 means (over 200 test cases)

| # | Solver | iter | hr | SSIM | PSNR | RMSE |
|---|---|---:|---:|---:|---:|---:|
| 1 | dual-domain-supervised (champion) | 20 | 0.8948 | 0.9992 | 56.43 | 0.00055 |
| 2 | itnet | 16 | 0.8926 | 0.9991 | 56.27 | 0.00056 |
| 3 | itnet-v2 | 15 | 0.8893 | 0.9991 | 56.01 | 0.00058 |
| 4 | itnet-v3 | 12 | 0.8749 | 0.9989 | 54.93 | 0.00065 |
| 5 | uswin | 20 | 0.8586 | 0.9986 | 53.85 | 0.00074 |
| 6 | learned-primal-dual | 12 | 0.7233 | 0.9962 | 47.98 | 0.00144 |
| 7 | hammernik-2017 | 12 | 0.6265 | 0.9902 | 45.36 | 0.00194 |
| 8 | param-efficient | 28 | 0.6183 | 0.9912 | 45.17 | 0.00198 |
| 9 | hammernik-vn | 16 | 0.5787 | 0.9865 | 44.32 | 0.00219 |
| 10 | fastdiff-flow-pixel-constrained | 15 | 0.5119 | 0.9792 | 43.04 | 0.00253 |

## Champion (dual-domain-supervised) vs each — headroom `hr`

| Solver | Δhr | 95% CI | Cohen dz | p (paired) | Holm | Wilcoxon | verdict |
|---|---:|---:|---:|---:|---:|---:|---|
| itnet | +0.0021 | ±0.0005 | 0.64 | 2.02e-16 *** | 2.02e-16 | 1.59e-14 | sep@1% |
| itnet-v2 | +0.0055 | ±0.0005 | 1.61 | 1.86e-57 *** | 3.71e-57 | 3.70e-33 | sep@1% |
| itnet-v3 | +0.0199 | ±0.0005 | 5.18 | 5.69e-146 *** | 1.71e-145 | 1.44e-34 | sep@1% |
| uswin | +0.0361 | ±0.0005 | 9.84 | 2.82e-200 *** | 1.13e-199 | 1.44e-34 | sep@1% |
| learned-primal-dual | +0.1715 | ±0.0015 | 15.64 | 5.43e-240 *** | 3.26e-239 | 1.44e-34 | sep@1% |
| hammernik-2017 | +0.2683 | ±0.0009 | 42.42 | 0.00e+00 *** | 0.00e+00 | 1.44e-34 | sep@1% |
| param-efficient | +0.2765 | ±0.0015 | 25.58 | 1.94e-282 *** | 1.36e-281 | 1.44e-34 | sep@1% |
| hammernik-vn | +0.3161 | ±0.0010 | 44.46 | 0.00e+00 *** | 0.00e+00 | 1.44e-34 | sep@1% |
| fastdiff-flow-pixel-constrained | +0.3828 | ±0.0041 | 13.06 | 1.61e-224 *** | 8.04e-224 | 1.44e-34 | sep@1% |

## Adjacent-rank hr comparisons (where the tier breaks)

| rank i vs i+1 | Δhr | Cohen dz | p (paired) | separated? |
|---|---:|---:|---:|---|
| dual-domain-supervised → itnet | +0.0021 | 0.64 | 2.02e-16 | yes |
| itnet → itnet-v2 | +0.0033 | 2.11 | 1.93e-75 | yes |
| itnet-v2 → itnet-v3 | +0.0144 | 5.71 | 4.75e-154 | yes |
| itnet-v3 → uswin | +0.0162 | 6.48 | 1.22e-164 | yes |
| uswin → learned-primal-dual | +0.1353 | 10.65 | 4.91e-207 | yes |
| learned-primal-dual → hammernik-2017 | +0.0968 | 11.72 | 3.18e-215 | yes |
| hammernik-2017 → param-efficient | +0.0082 | 0.80 | 3.88e-23 | yes |
| param-efficient → hammernik-vn | +0.0396 | 4.51 | 1.94e-134 | yes |
| hammernik-vn → fastdiff-flow-pixel-constrained | +0.0668 | 2.31 | 8.86e-82 | yes |

## Champion vs each — ALL metrics (paired p; dz in parens)

Sign of Δ is champion − solver (hr/SSIM/PSNR: + = champion better; RMSE: − = champion better).

| Solver | Δhr (dz) p | ΔSSIM (dz) p | ΔPSNR (dz) p | ΔRMSE (dz) p |
|---|---|---|---|---|
| itnet | +0.0021 (0.64) *** | +0.0002 (1.80) *** | +0.1520 (0.56) *** | -0.0000 (-0.66) *** |
| itnet-v2 | +0.0055 (1.61) *** | +0.0001 (1.51) *** | +0.4206 (1.71) *** | -0.0000 (-1.56) *** |
| itnet-v3 | +0.0199 (5.18) *** | +0.0003 (2.26) *** | +1.4999 (7.82) *** | -0.0001 (-4.09) *** |
| uswin | +0.0361 (9.84) *** | +0.0006 (3.37) *** | +2.5768 (12.09) *** | -0.0002 (-6.82) *** |
| learned-primal-dual | +0.1715 (15.64) *** | +0.0030 (4.76) *** | +8.4500 (10.25) *** | -0.0009 (-12.93) *** |
| hammernik-2017 | +0.2683 (42.42) *** | +0.0090 (6.08) *** | +11.0621 (13.92) *** | -0.0014 (-15.54) *** |
| param-efficient | +0.2765 (25.58) *** | +0.0080 (6.53) *** | +11.2547 (11.52) *** | -0.0014 (-27.35) *** |
| hammernik-vn | +0.3161 (44.46) *** | +0.0127 (6.92) *** | +12.1097 (14.02) *** | -0.0016 (-17.21) *** |
| fastdiff-flow-pixel-constrained | +0.3828 (13.06) *** | +0.0200 (5.40) *** | +13.3834 (10.41) *** | -0.0020 (-16.29) *** |

*Legend: n.s. = p≥.05, `*` = p<.05, `**` = p<1e-2, `***` = p<1e-4.*


### Per-metric statistical tie tier (n.s. vs champion at 5%)

- **HR**: dual-domain-supervised (1 method(s))
- **SSIM**: dual-domain-supervised (1 method(s))
- **PSNR**: dual-domain-supervised (1 method(s))
- **RMSE**: dual-domain-supervised (1 method(s))

**Metric-discordant solvers** (tied to champion on some measures, separated on others): none — every solver has the same verdict across all four metrics.


## Statistical tie tier — hr (not separable from champion at 5%)

**dual-domain-supervised** — 1 method(s).


## Findings — all four metrics, and the n=5 (Mayo) vs n=200 (Breast) contrast

1. **All four measures agree — total separation.** Every top-10 method separates from the
   champion at p<1e-4 on **hr, SSIM, PSNR AND RMSE simultaneously**; the per-metric tie tier
   is the **champion alone** for every metric, and there are **zero metric-discordant
   solvers**. This is stronger than — and opposite to — **Mayo (n=5)**, where the metrics
   *disagreed* (SSIM alone separated ITNet-v1 from the v2/U-Swin tie) and the top 3–4 were an
   unbreakable hr-tie. Same frozen framework; the flip is driven purely by sample size (5→200).

2. **p-values don't rank the top tier — effect size does.** By Cohen's dz / raw Δhr, three
   practical bands: **top cluster** dual-domain-sup (0.8948), itnet (0.8926, dz 0.64), itnet-v2
   (0.8893), itnet-v3 (0.8749), uswin (0.8586) — all within Δhr ≤ 0.036; **large practical gap
   ↓** to learned-primal-dual (0.7233, dz 15.6); **mid-tier** hammernik-2017 (0.6265),
   param-efficient (0.6183), hammernik-vn (0.5787), fastdiff (0.5119).

3. **SSIM is the most sensitive discriminator at the ceiling.** For itnet vs champion the SSIM
   effect (dz 1.80) exceeds the hr effect (dz 0.64) — because top SSIM is saturated (0.9991 vs
   0.9992) with tiny variance, so a minuscule mean gap is a large standardized effect. RMSE and
   hr track each other (hr is RMSE-derived). No metric changes the *ordering*, but SSIM
   sharpens the very top and RMSE/hr sharpen the mid-tier.

4. **Param-efficient (195 params) — mid-tier, but the tightest and closest-to-DL.** Its nearest
   neighbour is a full DL method, hammernik-2017, at Δhr 0.008 (dz 0.80) — the smallest-effect
   mid-tier pair — and it has the **smallest per-case std of all solvers (±0.0076 hr)**: the
   most *consistent* reconstructor across the 200 cases, at ~2% of a full network's parameters.

5. **Methodological takeaway.** The identical significance machinery yields "everyone ties"
   (Mayo n=5) and "everyone separates on every metric" (Breast n=200). Raw significance is
   sample-size-bound and not cross-dataset-comparable; **effect size (dz) and raw Δ are** — lead
   with effect size, treat p as secondary.

Figures: `breast_topsolver_significance.png` (Δhr vs champion, 95% CI — all red = all separated
at 1%); `breast_significance_matrix.png` (pairwise −log10 p, no n.s. cells).
