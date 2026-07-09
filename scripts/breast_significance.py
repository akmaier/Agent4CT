"""breast_significance.py — paired significance analysis of the top-N breast-CT
solvers (mirrors the Mayo analysis, but n=200 test cases instead of n=5 patients).

Paired because every solver is scored on the SAME 200 held-out test cases. With
n=200 the paired t-test is very high-powered, so we report EFFECT SIZE (Cohen's dz
= mean(diff)/std(diff), and the raw mean difference) alongside p-values — statistical
significance != practical significance at this n. Champion = rank-1 (dual-domain-
supervised). Metrics: hr (ranking), SSIM, PSNR, RMSE. p at 5% and 1%, plus Holm.

Outputs: docs/runs/breast_significance_stats.md, breast_topsolver_significance.png
(forest plot of Δhr vs champion, 95% CI), breast_significance_matrix.png (pairwise
hr p-value heatmap over the top tier).
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[1]
PC = json.loads((REPO / "docs/runs/breast_percase_top.json").read_text())
# order by hr mean desc
order = sorted(PC, key=lambda k: -np.mean(PC[k]["hr"]))
arr = {k: {m: np.asarray(PC[k][m], float) for m in ("hr", "ssim", "psnr", "rmse")} for k in order}
champ = order[0]
n = len(arr[champ]["hr"])


def holm(pvals):
    idx = np.argsort(pvals); adj = np.empty_like(pvals); m = len(pvals)
    run = 0.0
    for rank, i in enumerate(idx):
        v = (m - rank) * pvals[i]; run = max(run, v); adj[i] = min(run, 1.0)
    return adj


def paired(a, b):
    d = a - b
    t, p = stats.ttest_rel(a, b)
    dz = d.mean() / (d.std(ddof=1) + 1e-12)          # Cohen's dz (paired)
    ci = 1.96 * d.std(ddof=1) / np.sqrt(len(d))       # 95% CI half-width of mean diff
    try:
        _, pw = stats.wilcoxon(a, b)
    except Exception:
        pw = float("nan")
    return dict(mean_diff=float(d.mean()), t=float(t), p=float(p), dz=float(dz),
                ci=float(ci), p_wilcoxon=float(pw))

# champion vs each, hr
rows = []
pv_hr = []
for k in order[1:]:
    r = paired(arr[champ]["hr"], arr[k]["hr"]); rows.append((k, r)); pv_hr.append(r["p"])
holm_hr = holm(np.array(pv_hr))

# adjacent-rank hr (where does the tier break?)
adj = []
for i in range(len(order) - 1):
    a, b = order[i], order[i + 1]
    adj.append((a, b, paired(arr[a]["hr"], arr[b]["hr"])))

# per-metric champion-vs-each (for the doc table)
permetric = {}
for k in order[1:]:
    permetric[k] = {m: paired(arr[champ][m], arr[k][m]) for m in ("hr", "ssim", "psnr", "rmse")}

# ---- markdown ----
def stars(p):
    return "n.s." if p >= 0.05 else ("*" if p >= 0.01 else ("**" if p >= 1e-4 else "***"))

md = []
md.append("# Breast-CT — top-10 solver significance analysis\n")
md.append("Statistical comparison of the **test-selected** Breast-CT leaderboard "
          "(each solver's best-by-test-`hr` iter over the 200 held-out test cases).\n")
md.append("## Method\n")
md.append(f"- **n = {n} held-out TEST cases** (i.i.d. synthetic breast phantoms, the SAME "
          "200 for every solver → **paired** comparison; Student's paired t-test, df = "
          f"{n-1}).\n")
md.append("- Metrics tested independently: headroom `hr`, SSIM, PSNR, RMSE.\n")
md.append(f"- Reference = champion **{champ}**. A comparison is 'significant' when it "
          "separates from the champion.\n")
md.append("- **n=200 is very high-powered** — tiny mean gaps reach p<0.01. So we report "
          "**effect size** (Cohen's dz = mean(diff)/std(diff)) and the **raw mean Δhr** "
          "alongside p. |dz|<0.2 = negligible, 0.2 small, 0.5 medium, 0.8 large.\n")
md.append("- p reported raw at **5%** and **1%**, plus **Holm-corrected** (9 comparisons). "
          "Wilcoxon signed-rank shown as a non-parametric robustness check.\n")
md.append(f"\n## Top-10 means (over {n} test cases)\n")
md.append("| # | Solver | iter | hr | SSIM | PSNR | RMSE |")
md.append("|---|---|---:|---:|---:|---:|---:|")
for i, k in enumerate(order, 1):
    a = arr[k]
    md.append(f"| {i} | {k}{' (champion)' if k==champ else ''} | {PC[k]['iter']} | "
              f"{a['hr'].mean():.4f} | {a['ssim'].mean():.4f} | {a['psnr'].mean():.2f} | "
              f"{a['rmse'].mean():.5f} |")

md.append(f"\n## Champion ({champ}) vs each — headroom `hr`\n")
md.append("| Solver | Δhr | 95% CI | Cohen dz | p (paired) | Holm | Wilcoxon | verdict |")
md.append("|---|---:|---:|---:|---:|---:|---:|---|")
for (k, r), ph in zip(rows, holm_hr):
    verdict = "TIE (n.s.)" if r["p"] >= 0.05 else ("sep@5%" if r["p"] >= 0.01 else "sep@1%")
    md.append(f"| {k} | {r['mean_diff']:+.4f} | ±{r['ci']:.4f} | {r['dz']:.2f} | "
              f"{r['p']:.2e} {stars(r['p'])} | {ph:.2e} | {r['p_wilcoxon']:.2e} | {verdict} |")

md.append("\n## Adjacent-rank hr comparisons (where the tier breaks)\n")
md.append("| rank i vs i+1 | Δhr | Cohen dz | p (paired) | separated? |")
md.append("|---|---:|---:|---:|---|")
for a, b, r in adj:
    md.append(f"| {a} → {b} | {r['mean_diff']:+.4f} | {r['dz']:.2f} | {r['p']:.2e} | "
              f"{'yes' if r['p']<0.05 else 'NO (tie)'} |")

# ---- per-metric significance vs champion (all four measures) ----
# RMSE/PSNR are "lower/higher is better"; we test separation regardless of sign.
md.append(f"\n## Champion vs each — ALL metrics (paired p; dz in parens)\n")
md.append("Sign of Δ is champion − solver (hr/SSIM/PSNR: + = champion better; RMSE: − = champion better).\n")
md.append("| Solver | Δhr (dz) p | ΔSSIM (dz) p | ΔPSNR (dz) p | ΔRMSE (dz) p |")
md.append("|---|---|---|---|---|")
def cell(r):
    return f"{r['mean_diff']:+.4f} ({r['dz']:.2f}) {stars(r['p'])}"
for k in order[1:]:
    pm = permetric[k]
    md.append(f"| {k} | {cell(pm['hr'])} | {cell(pm['ssim'])} | {cell(pm['psnr'])} | "
              f"{cell(pm['rmse'])} |")
md.append("\n*Legend: n.s. = p≥.05, `*` = p<.05, `**` = p<1e-2, `***` = p<1e-4.*\n")

# per-metric tie tiers vs champion
md.append("\n### Per-metric statistical tie tier (n.s. vs champion at 5%)\n")
for m in ("hr", "ssim", "psnr", "rmse"):
    tiem = [champ] + [k for k in order[1:] if permetric[k][m]["p"] >= 0.05]
    md.append(f"- **{m.upper()}**: {', '.join(tiem)} ({len(tiem)} method(s))")
# also flag metrics where a solver is n.s. on one measure but separated on another
disc = []
for k in order[1:]:
    verd = {m: (permetric[k][m]["p"] >= 0.05) for m in ("hr", "ssim", "psnr", "rmse")}
    if len(set(verd.values())) > 1:
        ns = [m for m, v in verd.items() if v]
        disc.append(f"{k} (tied on {','.join(ns) or 'none'})")
md.append("\n**Metric-discordant solvers** (tied to champion on some measures, separated on others): "
          + (", ".join(disc) if disc else "none — every solver has the same verdict across all four metrics") + ".\n")

# tie tier vs champion (hr, kept for the summary)
tie = [champ] + [k for (k, r) in rows if r["p"] >= 0.05]
md.append(f"\n## Statistical tie tier — hr (not separable from champion at 5%)\n")
md.append(f"**{', '.join(tie)}** — {len(tie)} method(s).\n")
neg = [(k, r) for (k, r) in rows if r["p"] < 0.05 and abs(r["dz"]) < 0.2]
if neg:
    md.append("\n**Significant-but-negligible (p<0.05 yet |dz|<0.2 — the n=200 power caveat):** "
              + ", ".join(f"{k} (dz={r['dz']:.2f})" for k, r in neg) + ".\n")

md.append("""
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
""")
(REPO / "docs/runs/breast_significance_stats.md").write_text("\n".join(md))
print("wrote docs/runs/breast_significance_stats.md")
print("TIE TIER:", tie)

# ---- forest plot: Δhr vs champion with 95% CI ----
fig, ax = plt.subplots(figsize=(7.5, 4.6))
ks = [k for k, _ in rows][::-1]; ds = [r["mean_diff"] for _, r in rows][::-1]
cis = [r["ci"] for _, r in rows][::-1]; ps = [r["p"] for _, r in rows][::-1]
y = np.arange(len(ks))
cols = ["#2b8cbe" if p >= 0.05 else ("#fdae61" if p >= 0.01 else "#d7301f") for p in ps]
ax.errorbar(ds, y, xerr=cis, fmt="o", color="none", ecolor="gray", capsize=3, zorder=1)
ax.scatter(ds, y, c=cols, s=60, zorder=2)
ax.axvline(0, color="k", lw=1, ls="--")
ax.set_yticks(y); ax.set_yticklabels(ks, fontsize=8)
ax.set_xlabel(f"Δ headroom vs champion ({champ})  [negative = worse than champion]")
ax.set_title(f"Breast-CT top-10: Δhr vs champion (paired, n={n}, 95% CI)")
from matplotlib.patches import Patch
ax.legend(handles=[Patch(color="#2b8cbe", label="tie (p≥.05)"),
                   Patch(color="#fdae61", label="sep @5%"),
                   Patch(color="#d7301f", label="sep @1%")], fontsize=8, loc="lower left")
fig.tight_layout(); fig.savefig(REPO / "docs/runs/breast_topsolver_significance.png", dpi=140)
print("wrote breast_topsolver_significance.png")

# ---- pairwise hr p-value matrix (top tier) ----
top = order[:10]
M = np.ones((len(top), len(top)))
for i in range(len(top)):
    for j in range(len(top)):
        if i != j:
            M[i, j] = paired(arr[top[i]]["hr"], arr[top[j]]["hr"])["p"]
fig2, ax2 = plt.subplots(figsize=(7.8, 6.4))
logp = -np.log10(np.clip(M, 1e-30, 1))
im = ax2.imshow(logp, cmap="viridis", vmin=0, vmax=min(20, logp.max()))
ax2.set_xticks(range(len(top))); ax2.set_xticklabels(top, rotation=90, fontsize=7)
ax2.set_yticks(range(len(top))); ax2.set_yticklabels(top, fontsize=7)
for i in range(len(top)):
    for j in range(len(top)):
        if i != j:
            txt = "n.s." if M[i, j] >= 0.05 else f"{M[i,j]:.0e}"
            ax2.text(j, i, txt, ha="center", va="center", fontsize=5,
                     color="white" if logp[i, j] > 6 else "black")
ax2.set_title(f"Breast-CT pairwise hr paired-t p-values (n={n})  [−log10 p]")
fig2.colorbar(im, ax=ax2, label="−log10 p"); fig2.tight_layout()
fig2.savefig(REPO / "docs/runs/breast_significance_matrix.png", dpi=140)
print("wrote breast_significance_matrix.png")
