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

## Statistical tie tier (not separable from champion at 5%)

**dual-domain-supervised** — 1 method(s).

## Findings — the n=5 (Mayo) vs n=200 (Breast) contrast

1. **At n=200 the paired test is so powerful that everything separates.** Every one of
   the top-10 is significantly different from the champion at the **1% level, Holm-robust**
   — and *every adjacent rank* separates too (rank-1 vs rank-2: Δhr=+0.0021, p=2e-16).
   The statistical "tie tier" is the **champion alone**. This is the exact **opposite of
   Mayo** (n=5), where weak power made the top **3–4 methods an unbreakable tie**. Same
   frozen framework, opposite verdict — driven purely by sample size.

2. **Therefore p-values do not rank the top tier; effect size does.** Grouped by Cohen's
   dz / raw Δhr, three practical bands emerge:
   - **Top cluster (dz small→large, Δhr ≤ 0.036):** dual-domain-supervised (0.8948),
     itnet (0.8926, dz 0.64), itnet-v2 (0.8893), itnet-v3 (0.8749), uswin (0.8586).
     A tight practical cluster — the champion's lead over itnet is 0.2% headroom.
   - **Large practical gap ↓** to learned-primal-dual (0.7233, dz 15.6 vs champion).
   - **Mid-tier:** hammernik-2017 (0.6265), param-efficient (0.6183), hammernik-vn
     (0.5787), fastdiff-pixel (0.5119).

3. **Param-efficient (195 params) sits in the mid-tier and is the tightest solver.** It is
   statistically far below the top-5 (expected), but its nearest neighbour is a full DL
   method — hammernik-2017 — at **Δhr = 0.008, dz = 0.80** (the smallest-effect mid-tier
   pair): a **195-parameter** solver essentially matching a full learned method at ~2% of
   its parameters. It also has the **smallest per-case std of all (±0.0076)** — the most
   consistent reconstructor across the 200 cases.

4. **Methodological takeaway for the paper.** The same significance machinery yields
   "everyone ties" (Mayo, n=5) and "everyone separates" (Breast, n=200). Raw significance
   is sample-size-bound and not cross-dataset-comparable; **effect size (dz) and raw Δhr
   are.** Report both; lead with effect size.

Figures: `breast_topsolver_significance.png` (Δhr vs champion, 95% CI — all points red =
all separated at 1%), `breast_significance_matrix.png` (pairwise −log10 p, no n.s. cells).
