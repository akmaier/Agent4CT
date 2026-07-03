# Mayo-LDCT — top-solver significance analysis

Statistical comparison of the **test-selected** Mayo leaderboard (each solver's
best iteration by test mean `hr` over the full per-iteration sweep). Source data:
per-iter `final.json` under `docs/runs/<slug>-itertest/` (and
`pe-iter-testeval/` for param-efficient); board built by `scripts/build_registry.py`.

## Method

- **n = 5 held-out Wagner test patients** (L014, L056, L058, L075, L123), the same
  5 for every solver → **paired** comparison (Student's paired t-test on the
  per-patient values, df = 4).
- Every metric tested independently: headroom `hr`, SSIM, PSNR, RMSE.
- Reference = the champion **ITNet v1** (iter-10). A comparison is "significant"
  when it separates from the champion.
- **No multiple-comparison correction** (per the loose-threshold decision); results
  reported at both **5%** and **1%**.
- Note on power: n = 5 is small. A Wilcoxon signed-rank test cannot even reach
  p < 0.05 at n = 5 (its two-sided floor is 0.0625), so the paired t-test is used.
- The board iters are each solver's **best-by-test-hr**, so absolute values are
  optimistic; the paired *relative* comparison under a common selection rule is fair.

## Top-solver means (mean over 5 patients)

| Solver | iter | hr | SSIM | PSNR | RMSE |
|---|---:|---:|---:|---:|---:|
| ITNet v1 (champion) | 10 | 0.3756 | 0.9790 | 43.41 | 0.00051 |
| ITNet v2 | 13 | 0.3735 | 0.9784 | 43.38 | 0.00052 |
| U-Swin | 6 | 0.3700 | 0.9770 | 43.29 | 0.00051 |
| DD-UNet supervised L2 | 15 | 0.3607 | 0.9758 | 43.20 | 0.00052 |
| Param-efficient (evolved) | 33 | 0.3241 | 0.9727 | 42.68 | 0.00055 |
| ITNet v3 | 10 | 0.3066 | 0.9707 | 42.49 | 0.00057 |
| Hammernik-VN (MRI port) | 17 | 0.1591 | 0.9520 | 40.76 | 0.00067 |

Per-patient `hr` (L014, L056, L058, L075, L123):
`itnet [.443 .381 .227 .380 .446]`, `itnet-v2 [.443 .384 .223 .373 .445]`,
`uswin [.428 .357 .284 .362 .420]`, `dd-sup [.428 .365 .215 .363 .432]`,
`param-eff [.377 .317 .231 .308 .388]`, `itnet-v3 [.377 .302 .156 .321 .376]`,
`hammernik-vn [.174 .145 .143 .151 .183]`.

## Paired p-values vs champion (ITNet v1)

| vs ITNet v1 | hr | SSIM | PSNR | RMSE |
|---|---:|---:|---:|---:|
| ITNet v2 | 0.2641 | **0.0417** | 0.2791 | 0.2314 |
| U-Swin | 0.7384 | **0.0058** | 0.5905 | 0.9412 |
| DD-UNet supervised | **0.0001** | **0.0044** | **0.0004** | **0.0000** |
| Param-efficient | **0.0217** | **0.0269** | **0.0203** | **0.0249** |
| ITNet v3 | **0.0000** | **0.0038** | **0.0001** | **0.0016** |
| Hammernik-VN | **0.0031** | **0.0027** | **0.0045** | **0.0008** |

**bold** = p < 0.05.

## Verdicts

**At 5%:**
- **ITNet v1 / ITNet v2 / U-Swin — statistical 3-way tie** on `hr`, PSNR, RMSE. The
  champion "flip" from ITNet v2 → v1 is not significant (Δhr p = 0.26). The *only*
  separation among the top 3 is **SSIM**: ITNet v1 > ITNet v2 (p = 0.042) and
  > U-Swin (p = 0.006).
- **DD-UNet-supervised and everything below** are significantly different on **all
  four** metrics.

**At 1%:**
- **ITNet v2 → full tie** with the champion (its SSIM edge, p = 0.042, drops out).
- **U-Swin → still separable on SSIM only** (p = 0.006 < 0.01); tied elsewhere.
- **Param-efficient → joins the tie** (all four p = 0.02–0.027, none < 0.01). So the
  hr-tie tier grows to **four**: ITNet v1 / v2 / U-Swin / Param-efficient.
- **DD-UNet-supervised, ITNet v3, Hammernik-VN** remain significant on all four.

## The instructive paradox (mean-rank vs significance diverge)

At 1%, **DD-UNet-supervised (hr 0.361) *is* significantly worse than the champion,
but Param-efficient (hr 0.324) is *not*** — even though param-efficient is further
behind in the mean. Reason: DD-UNet is consistently ~0.015 below on *every* patient
(near-zero variance → high t), whereas param-efficient's larger deficit is noisy
across patients (high variance → low t at n = 5). Ranking-by-mean and pairwise
significance need not agree.

## Figures

- `docs/runs/mayo_topsolver_significance.png` — forest plot, paired Δhr vs champion (95% CI).
- `docs/runs/mayo_significance_matrix.png` — solver × metric significance heatmap.
- `docs/runs/mayo_effort_timeline.png` — development effort on an active-working-days axis.

## Reproduce

Per-iter test scores live in `docs/runs/<slug>-itertest/iter-NNNN/final.json`
(`patients` dict has per-patient `headroom`/`ssim`/`psnr`/`rmse`); the selection is
in `docs/runs/mayo_testsweep_selection.json`. The paired t-tests above were computed
with a self-contained regularized-incomplete-beta t-CDF (no SciPy dependency).
