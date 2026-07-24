#!/usr/bin/env python3
"""Publication figures for the Medical Physics paper
"Agentic Autoresearch for CT Reconstruction".

Single reproducible script. Reads only local JSON under docs/runs/ and
writes vector PDFs into paper/figures/. matplotlib only (>=3.10).

Data sources
------------
- docs/runs/index/breast_ct.json        noiseless breast board
- docs/runs/index/breast_ct_noise.json  noisy (I0=100k) breast board
- docs/runs/index/mayo_ldct.json        Mayo board
- docs/runs/breast_percase_top.json     per-case arrays for top-10 breast solvers

Figures produced
----------------
- fig3_reversal.pdf     headline slopegraph: noiseless -> noisy rank inversion
- fig2_params_vs_hr.pdf  params (log) vs test hr, Mayo & breast
- fig1_agentic_loop.pdf  six-box agentic-loop schematic
- sfig_effectsize.pdf    Cohen's dz vs raw Delta-hr against the champion
- sfig_pareto.pdf        param-efficient climb on breast

NOTE on fig4 (recon panels): SKIPPED here. It needs actual reconstruction
images from the cluster (not available in this local JSON-only export).
Generate it separately once the recon PNGs are on the cluster.
"""

import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

# ---------------------------------------------------------------------------
# Global style: embed fonts (Type 42), small clean single-column defaults.
# ---------------------------------------------------------------------------
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
matplotlib.rcParams["font.size"] = 8
matplotlib.rcParams["axes.titlesize"] = 9
matplotlib.rcParams["axes.labelsize"] = 8.5
matplotlib.rcParams["xtick.labelsize"] = 7.5
matplotlib.rcParams["ytick.labelsize"] = 7.5
matplotlib.rcParams["legend.fontsize"] = 7.5
matplotlib.rcParams["font.family"] = "sans-serif"
matplotlib.rcParams["axes.linewidth"] = 0.6
matplotlib.rcParams["axes.spines.top"] = False
matplotlib.rcParams["axes.spines.right"] = False

# Colorblind-safe (Wong / Okabe-Ito palette)
CB = {
    "red": "#D55E00",     # collapse
    "green": "#009E73",   # rise
    "gray": "#999999",
    "blue": "#0072B2",
    "orange": "#E69F00",
    "purple": "#CC79A7",
    "black": "#000000",
}

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IDX = os.path.join(REPO, "docs", "runs", "index")
OUTDIR = os.path.join(REPO, "paper", "figures")
os.makedirs(OUTDIR, exist_ok=True)

# columns: single ~3.3in, double ~6.9in
SINGLE = 3.3
DOUBLE = 6.9


def load_board(name):
    with open(os.path.join(IDX, f"{name}.json")) as fh:
        return json.load(fh)["leaderboard"]["rows"]


def pretty(key, rows):
    """Human solver name if available, else the key."""
    for r in rows:
        if r["solver_key"] == key:
            return r.get("solver_name") or key
    return key


def savefig(fig, path):
    fig.savefig(path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    size = os.path.getsize(path)
    assert size > 0, f"empty PDF: {path}"
    print(f"  wrote {os.path.basename(path)}  ({size} bytes)")
    return size


# ===========================================================================
# FIG 3 (HEADLINE) — noiseless -> noisy rank reversal slopegraph
# ===========================================================================
def fig3_reversal():
    noiseless = load_board("breast_ct")
    noisy = load_board("breast_ct_noise")

    # Ranked-only ordering, by test_hr_mean desc (== the board rank).
    def ranked(rows):
        rk = [r for r in rows if r.get("rank") is not None
              and r.get("test_hr_mean") is not None]
        rk.sort(key=lambda r: r["test_hr_mean"], reverse=True)
        return {r["solver_key"]: i + 1 for i, r in enumerate(rk)}, rk

    nl_rank, nl_rows = ranked(noiseless)
    nz_rank, nz_rows = ranked(noisy)

    n_noisy_ranked = len(nz_rank)

    # Special case: the headline collapse. dual-domain-supervised is rank 1
    # noiseless but is EXCLUDED (hr<=0) on the noisy board. We must still
    # show its plunge, so we route it to the bottom slot (last + 1).
    COLLAPSE = "dual-domain-supervised"
    RISE = "learned-primal-dual"
    collapse_dest = n_noisy_ranked + 1  # one below the last ranked solver

    # Which solvers to draw: everything ranked on BOTH boards, PLUS the
    # collapse solver (ranked noiseless, DNF noisy -> forced to bottom).
    both = [k for k in nl_rank if k in nz_rank]
    dropped = []
    for k in nl_rank:
        if k not in nz_rank and k != COLLAPSE:
            dropped.append(k)

    fig, ax = plt.subplots(figsize=(SINGLE, 4.7))  # single-column slopegraph
    x_left, x_right = 0.0, 2.4  # wide node gap so the two lists clear >=2cm
    LBL_FS, HDR_FS = 5.6, 6.4   # 20% smaller than before, frees the middle

    def color_for(k):
        if k == COLLAPSE:
            return CB["red"], 2.2, 1.0, 20
        if k == RISE:
            return CB["green"], 2.2, 1.0, 20
        return CB["gray"], 0.9, 0.55, 5

    # Draw connectors + endpoint labels.
    all_keys = both + [COLLAPSE]
    for k in all_keys:
        lr = nl_rank[k]
        rr = nz_rank.get(k, collapse_dest)
        col, lw, alpha, z = color_for(k)
        ax.plot([x_left, x_right], [lr, rr], color=col, lw=lw,
                alpha=alpha, zorder=z, solid_capstyle="round")
        ax.scatter([x_left, x_right], [lr, rr], s=14 if k in (COLLAPSE, RISE)
                   else 7, color=col, alpha=alpha, zorder=z + 1)

        name = pretty(k, noiseless)
        lbl_col = col if k in (COLLAPSE, RISE) else "#555555"
        weight = "bold" if k in (COLLAPSE, RISE) else "normal"
        # left labels: rank number prefixed so ranks are legible without ticks
        ax.text(x_left - 0.08, lr, f"{lr}. {name}", ha="right", va="center",
                fontsize=LBL_FS, color=lbl_col, fontweight=weight, zorder=z + 2)
        # right labels: rank number, or WHY the solver is unranked. The collapse
        # solver FINISHED and scored hr 0 (below the floor) -- it is NOT a DNF;
        # a true DNF (per-scene methods that ran out of wall) has no score at all.
        # Mislabelling it "DNF" would contradict the text and the board tables.
        if k in nz_rank:
            rlabel = f"{name}  {rr}"
        else:
            hrv = next((r.get("test_hr_mean") for r in noisy
                        if r["solver_key"] == k), None)
            rlabel = f"{name}  (hr 0)" if hrv is not None else f"{name}  (DNF)"
        ax.text(x_right + 0.08, rr, rlabel, ha="left", va="center",
                fontsize=LBL_FS, color=lbl_col, fontweight=weight, zorder=z + 2)

    ax.set_xlim(x_left - 0.25, x_right + 0.25)  # tight around nodes; labels
    top = 0.15                                  # overflow and tight-bbox keeps them
    ax.set_ylim(collapse_dest + 0.6, top)  # inverted: rank 1 at top
    ax.set_xticks([])
    ax.set_yticks([])
    # column headers anchored to the node columns, aligned to grow outward.
    ax.text(x_left, top - 0.12, "Noiseless rank", ha="right", va="bottom",
            fontsize=HDR_FS, fontweight="bold")
    ax.text(x_right, top - 0.12, "Noisy (I$_0$=100k) rank", ha="left",
            va="bottom", fontsize=HDR_FS, fontweight="bold")
    for sp in ("left", "bottom"):
        ax.spines[sp].set_visible(False)
    ax.grid(False)

    fig.tight_layout()
    # Verify the whitespace gap between the two lists at the printed
    # \columnwidth (244.7pt = 8.60cm). The gap is node-to-node distance.
    fig.canvas.draw()
    COLW_CM = 8.60
    mid = 0.5 * (top + (collapse_dest + 0.6))
    dl = ax.transData.transform((x_left, mid))
    dr = ax.transData.transform((x_right, mid))
    tb = fig.get_tightbbox(fig.canvas.get_renderer())
    gap_cm = COLW_CM * ((dr[0] - dl[0]) / fig.dpi) / tb.width
    print(f"  fig3 node gap ~ {gap_cm:.2f} cm at columnwidth "
          f"({'OK' if gap_cm >= 2.0 else 'TOO NARROW'})")
    savefig(fig, os.path.join(OUTDIR, "fig3_reversal.pdf"))
    return dropped


# ===========================================================================
# FIG 2 — params (log) vs test hr, Mayo & breast(noiseless)
# ===========================================================================
def fig2_params_vs_hr():
    # Two boards. y is normalized per dataset to "% of the best hr reached
    # on that dataset", so the two very different hr scales overlay cleanly
    # (Mayo ~0.38, breast noiseless ~0.89 both map to 100).
    mayo = load_board("mayo_ldct")
    breast_nl = load_board("breast_ct")
    HILITE = "param-efficient"

    def pts(rows):
        out = []
        for r in rows:
            if r.get("rank") is None:
                continue
            p = r.get("params_M")
            hr = r.get("test_hr_mean")
            if p is None or hr is None:
                continue
            params = p * 1e6  # params_M is in millions; param-efficient is 195
            if params <= 0:
                params = 0.5  # < 1 param -> floor so log-x is defined
            out.append((r["solver_key"], params, hr))
        return out

    def norm(rows):
        """Scale hr to percent of the best hr on this board."""
        data = pts(rows)
        best = max(d[2] for d in data)
        return [(k, x, 100.0 * y / best) for (k, x, y) in data]

    mayo_pts = norm(mayo)
    nl_pts = norm(breast_nl)

    fig, ax = plt.subplots(figsize=(SINGLE, 2.8))

    def scatter(data, color, marker, label):
        xs = [d[1] for d in data]
        ys = [d[2] for d in data]
        ax.scatter(xs, ys, s=22, color=color, marker=marker,
                   edgecolors="white", linewidths=0.4, alpha=0.85,
                   label=label, zorder=3)

    scatter(mayo_pts, CB["blue"], "o", "Mayo (noise-limited)")
    scatter(nl_pts, CB["orange"], "^", "Breast, noiseless")

    def highlight(data, annotate, dxdy=(8, 4), ha="left"):
        """Ring param-efficient; optionally label it with its exact count."""
        for k, x, y in data:
            if k == HILITE:
                ax.scatter([x], [y], s=90, facecolors="none",
                           edgecolors=CB["red"], linewidths=1.6, zorder=5)
                if annotate:
                    ax.annotate(f"param-efficient\n({int(round(x))} params)",
                                (x, y), textcoords="offset points",
                                xytext=dxdy, fontsize=6.6, color=CB["red"],
                                fontweight="bold", ha=ha,
                                arrowprops=dict(arrowstyle="-", color=CB["red"],
                                                lw=0.6))
                return

    # Mayo has 969 params, breast 195; label each with its exact count. Put the
    # breast tag up-left of x=195 and the Mayo tag up-right of x=969 so the two
    # labels sit in separate zones.
    highlight(nl_pts, annotate=True, dxdy=(-10, 14), ha="right")
    highlight(mayo_pts, annotate=True, dxdy=(10, 10), ha="left")

    ax.set_xscale("log")
    ax.set_xlabel("trainable parameters (log scale)")
    ax.set_ylabel("hr (% of best on dataset)")
    ax.set_ylim(-4, 110)
    ax.legend(loc="lower center", frameon=False, handletextpad=0.3,
              fontsize=6.6, borderpad=0.2, labelspacing=0.25)
    ax.grid(True, which="major", axis="x", ls=":", lw=0.4, color="#dddddd")
    fig.tight_layout()
    savefig(fig, os.path.join(OUTDIR, "fig2_params_vs_hr.pdf"))


# ===========================================================================
# FIG 1 — six-box agentic-loop schematic
# ===========================================================================
def fig1_agentic_loop():
    fig, ax = plt.subplots(figsize=(DOUBLE, 3.0))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6)
    ax.axis("off")

    boxes = [
        ("1. Read previous\nresult", 1.7, 4.4),
        ("2. Name the\nfailure mode", 4.5, 4.4),
        ("3. Change ONE knob\n(edit solver.py)", 7.3, 4.4),
        ("4. State the\nhypothesis", 10.1, 4.4),
        ("5. Run 20-min SLURM job\n(diff. projector)", 8.6, 1.6),
        ("6. Read frozen metric\n→ accept / discard", 3.4, 1.6),
    ]
    bw, bh = 2.4, 1.0
    centers = []
    for text, cx, cy in boxes:
        box = FancyBboxPatch((cx - bw / 2, cy - bh / 2), bw, bh,
                             boxstyle="round,pad=0.06,rounding_size=0.12",
                             linewidth=1.0, edgecolor=CB["blue"],
                             facecolor="#EAF3FA", zorder=2)
        ax.add_patch(box)
        ax.text(cx, cy, text, ha="center", va="center", fontsize=7.5,
                zorder=3)
        centers.append((cx, cy))

    def arrow(a, b, rad=0.0):
        (x0, y0), (x1, y1) = centers[a], centers[b]
        ar = FancyArrowPatch((x0, y0), (x1, y1),
                             connectionstyle=f"arc3,rad={rad}",
                             arrowstyle="-|>", mutation_scale=11,
                             shrinkA=26, shrinkB=26, lw=1.0,
                             color="#444444", zorder=1)
        ax.add_patch(ar)

    arrow(0, 1)
    arrow(1, 2)
    arrow(2, 3)
    arrow(3, 4, rad=-0.25)
    arrow(4, 5, rad=-0.15)
    arrow(5, 0, rad=-0.2)  # loop back to the top

    ax.text(6.0, 3.0, "loop", ha="center", va="center", fontsize=8,
            style="italic", color="#444444")

    fig.tight_layout()
    savefig(fig, os.path.join(OUTDIR, "fig1_agentic_loop.pdf"))


# ===========================================================================
# SFIG effectsize — Cohen's dz vs raw Delta-hr against the champion
# ===========================================================================
def sfig_effectsize():
    with open(os.path.join(REPO, "docs", "runs",
                           "breast_percase_top.json")) as fh:
        pc = json.load(fh)

    rows = load_board("breast_ct")  # for pretty names

    # Champion = the max-hr solver among the per-case set.
    champ = max(pc, key=lambda k: pc[k]["hr_mean"])
    ch = pc[champ]["hr"]
    n = len(ch)

    results = []
    for k, v in pc.items():
        if k == champ:
            continue
        other = v["hr"]
        m = min(n, len(other))
        d = [ch[i] - other[i] for i in range(m)]  # paired champion - other
        mean_d = sum(d) / m
        var = sum((x - mean_d) ** 2 for x in d) / (m - 1)
        std = var ** 0.5
        dz = mean_d / std if std > 0 else float("nan")
        results.append((k, dz, mean_d))

    fig, ax = plt.subplots(figsize=(SINGLE, 3.0))
    for k, dz, dhr in results:
        ax.scatter([dz], [dhr], s=26, color=CB["blue"],
                   edgecolors="white", linewidths=0.4, zorder=3)
        ax.annotate(pretty(k, rows), (dz, dhr), textcoords="offset points",
                    xytext=(4, 3), fontsize=6.2, color="#333333")

    ax.axhline(0, color="#bbbbbb", lw=0.5, ls="--")
    ax.set_xlabel("Cohen's $d_z$  (paired, vs champion)")
    ax.set_ylabel(r"raw $\Delta$hr  (champion $-$ solver)")
    ax.grid(True, ls=":", lw=0.4, color="#dddddd")
    fig.tight_layout()
    savefig(fig, os.path.join(OUTDIR, "sfig_effectsize.pdf"))
    return champ, n


# ===========================================================================
# SFIG pareto — param-efficient climb on breast
# ===========================================================================
def sfig_pareto():
    # (params, val hr) climb steps, given by the paper.
    steps = [
        (1930, 0.26, "image-domain ceiling"),
        (13, 0.4345, "filtered-DC"),
        (183, 0.5047, "+LPD-combine"),
        (195, 0.6201, "+bilateral"),
    ]

    fig, ax = plt.subplots(figsize=(SINGLE, 2.8))

    xs = [s[0] for s in steps]
    ys = [s[1] for s in steps]

    # Connect the last three as the climb.
    climb_x = [s[0] for s in steps[1:]]
    climb_y = [s[1] for s in steps[1:]]
    ax.plot(climb_x, climb_y, color=CB["green"], lw=1.4, zorder=2,
            solid_capstyle="round")

    # The first point (image-domain ceiling) sits apart from the climb.
    ax.scatter([steps[0][0]], [steps[0][1]], s=40, color=CB["gray"],
               edgecolors="white", linewidths=0.5, zorder=3)
    ax.scatter(climb_x, climb_y, s=40, color=CB["green"],
               edgecolors="white", linewidths=0.5, zorder=3)

    offsets = [(6, -10), (6, -12), (-4, 8), (6, -4)]
    for (x, y, lbl), off in zip(steps, offsets):
        ax.annotate(lbl, (x, y), textcoords="offset points", xytext=off,
                    fontsize=6.6, color="#333333")

    ax.set_xscale("log")
    ax.set_xlabel("trainable parameters (log scale)")
    ax.set_ylabel("val hr")
    ax.grid(True, which="both", axis="x", ls=":", lw=0.4, color="#cccccc")
    fig.tight_layout()
    savefig(fig, os.path.join(OUTDIR, "sfig_pareto.pdf"))


# ===========================================================================
# SFIG hint-climb — compact-solver search: agent-alone plateau vs the break
# once human hints begin at iteration 20.
# ===========================================================================
def sfig_hint_climb():
    import csv
    HINT = 20  # human hints begin (second half of each ~40-iteration search)
    panels = [
        ("Mayo (low-dose, denoising)",
         "mayo-ldct-claude-agentic-param-efficient-search-20260624-01",
         (-0.02, 0.46), (32, 0.400), "break to hr 0.40"),
        ("Breast (128-view sparse)",
         "breast-ct-claude-agentic-param-efficient-search-20260703-01",
         (-0.04, 0.72), (28, 0.620), "break to hr 0.62"),
    ]

    def load(run):
        tsv = os.path.join(REPO, "docs", "runs", run, "results.tsv")
        data = []
        for r in csv.DictReader(open(tsv), delimiter="\t"):
            try:
                it = int(r["iter"])
                hr = float(r.get("headroom") or 0)
            except (ValueError, TypeError):
                continue
            data.append((it, hr, (r.get("status") or "").strip().lower()))
        data.sort(key=lambda d: d[0])
        return data

    fig, axes = plt.subplots(2, 1, figsize=(SINGLE, 5.0))
    for k, (ax, (title, run, ylim, brk, note)) in enumerate(zip(axes, panels)):
        data = load(run)
        its = [d[0] for d in data]
        best, m = [], 0.0
        for _, h, _s in data:
            m = max(m, h)
            best.append(m)
        kept = [(i, h) for (i, h, s) in data if s != "discard" and h > 0]
        disc = [i for (i, h, s) in data if s == "discard" or h <= 0]
        ax.axvspan(HINT, max(its) + 0.6, color=CB["orange"], alpha=0.09, zorder=0)
        ax.axvline(HINT, color=CB["orange"], lw=1.0, ls="--", zorder=1)
        ax.step(its, best, where="post", color=CB["green"], lw=1.5, zorder=3,
                label="best so far")
        ax.scatter([i for i, _ in kept], [h for _, h in kept], s=16,
                   color=CB["blue"], edgecolors="white", linewidths=0.4,
                   zorder=4, label="kept iteration")
        ax.scatter(disc, [0.0] * len(disc), s=14, color=CB["gray"],
                   marker="x", linewidths=0.9, zorder=4, label="discarded")
        ax.annotate(note, brk, textcoords="offset points", xytext=(6, 5),
                    fontsize=6.4, color=CB["green"], fontweight="bold", ha="left")
        ax.set_ylim(*ylim)
        ax.set_ylabel("val hr")
        ax.set_title(title, fontsize=7.6, loc="left", pad=2)
        if k == 0:
            ax.annotate("human hints begin", (HINT, ylim[1]),
                        textcoords="offset points", xytext=(-4, -1),
                        fontsize=6.2, color=CB["orange"], fontweight="bold",
                        va="top", ha="right")
            ax.legend(loc="lower right", frameon=False, fontsize=6.0,
                      handletextpad=0.3, labelspacing=0.2)
    axes[-1].set_xlabel("autoresearch iteration")
    fig.tight_layout(h_pad=1.4)
    savefig(fig, os.path.join(OUTDIR, "sfig_hint_climb.pdf"))


# ===========================================================================
# SFIG mayo-significance — clean forest plot: paired Delta-hr vs champion.
# Per-patient hr (L014,L056,L058,L075,L123) from docs/runs/mayo_significance_stats.md.
# ===========================================================================
def sfig_mayo_significance():
    champ = [0.443, 0.381, 0.227, 0.380, 0.446]  # ITNet v1 (champion)
    others = [
        ("ITNet v2",               0.374, [0.443, 0.384, 0.223, 0.373, 0.445]),
        ("U-Swin",                 0.370, [0.428, 0.357, 0.284, 0.362, 0.420]),
        ("dual-domain-supervised", 0.361, [0.428, 0.365, 0.215, 0.363, 0.432]),
        ("param-efficient",        0.324, [0.377, 0.317, 0.231, 0.308, 0.388]),
        ("ITNet v3",               0.307, [0.377, 0.302, 0.156, 0.321, 0.376]),
        ("Hammernik-VN",           0.159, [0.174, 0.145, 0.143, 0.151, 0.183]),
    ]
    TCRIT = 2.776  # t_{0.975, df=4}
    fig, ax = plt.subplots(figsize=(SINGLE, 2.9))
    ys = list(range(len(others)))[::-1]  # first entry at top
    for y, (name, hrm, vals) in zip(ys, others):
        d = [champ[i] - vals[i] for i in range(5)]
        m = sum(d) / 5
        sd = (sum((x - m) ** 2 for x in d) / 4) ** 0.5
        ci = TCRIT * sd / (5 ** 0.5)
        sig = (m - ci) > 0                 # CI clears 0 -> significantly worse
        col = CB["red"] if sig else CB["gray"]
        ax.errorbar([m], [y], xerr=[[ci], [ci]], fmt="o", color=col, ecolor=col,
                    elinewidth=1.4, capsize=2.5, ms=5, zorder=3)
        ax.text(m + ci + 0.006, y, f"{name}  (hr {hrm:.3f})", va="center",
                ha="left", fontsize=6.4, color=col)
    ax.axvline(0, color="#888888", lw=0.8, ls="--", zorder=1)
    ax.set_yticks([])
    ax.set_ylim(-0.7, len(others) - 0.3)
    ax.set_xlim(-0.03, 0.56)  # room for the solver labels inside the axes so the
                              # centred x-label is not pushed off the (tight-bbox) edge
    ax.set_xticks([0.0, 0.1, 0.2, 0.3])
    ax.set_xlabel(r"$\Delta$ headroom vs champion ITNet v1")  # CI/n stated in caption
    for sp in ("left", "right", "top"):
        ax.spines[sp].set_visible(False)
    fig.tight_layout()
    # Save with a tight bbox extended 3 mm on the left (the default tight crop
    # was shaving the left edge of the axis label / leftmost marker).
    from matplotlib.transforms import Bbox
    fig.canvas.draw()
    tb = fig.get_tightbbox(fig.canvas.get_renderer())
    pad, left_extra = 0.01, 3.0 / 25.4  # inches (3 mm on the left)
    bb = Bbox.from_extents(tb.x0 - left_extra - pad, tb.y0 - pad,
                           tb.x1 + pad, tb.y1 + pad)
    outp = os.path.join(OUTDIR, "sfig_mayo_significance.pdf")
    fig.savefig(outp, format="pdf", bbox_inches=bb)
    plt.close(fig)
    print(f"  wrote {os.path.basename(outp)}  (+3mm left pad)")


# ===========================================================================
def fig_retrain_slope():
    """Three-column slopegraph: noiseless -> noisy (no retrain) -> noisy (retrained).
    Shows the clean ranking scrambling under unseen noise and largely RETURNING once
    the weights are retrained on matched noise. dd-supervised (collapse & recover) and
    learned-primal-dual (robust throughout) are highlighted."""
    noiseless = load_board("breast_ct")
    noisy = load_board("breast_ct_noise")
    retrain = load_board("breast_ct_noise_retrain")

    def ranked(rows):
        rk = [r for r in rows if r.get("rank") is not None
              and r.get("test_hr_mean") is not None]
        rk.sort(key=lambda r: r["test_hr_mean"], reverse=True)
        return {r["solver_key"]: i + 1 for i, r in enumerate(rk)}

    nl_rank, nz_rank, rt_rank = ranked(noiseless), ranked(noisy), ranked(retrain)
    COLLAPSE, RISE = "dual-domain-supervised", "learned-primal-dual"
    xs = [0.0, 2.2, 4.4]
    keys = [k for k in nl_rank]                       # trace every noiseless-ranked solver

    # Solvers excluded (hr<=0 / DNF) on a board get distinct y-slots BELOW the ranked
    # set (ordered by noiseless rank), so their labels never pile on one line.
    def positions(rank_dict):
        pos = dict(rank_dict)
        excl = sorted((k for k in keys if k not in rank_dict),
                      key=lambda k: nl_rank.get(k, 999))
        for j, k in enumerate(excl):
            pos[k] = len(rank_dict) + 1 + j
        return pos

    nl_pos, nz_pos, rt_pos = positions(nl_rank), positions(nz_rank), positions(rt_rank)
    bottom = max(max(nl_pos.values()), max(nz_pos.values()), max(rt_pos.values()))

    fig, ax = plt.subplots(figsize=(DOUBLE, 5.0))
    LBL_FS, HDR_FS = 6.0, 7.2

    def style(k):
        if k == COLLAPSE:
            return CB["red"], 2.2, 1.0, 20
        if k == RISE:
            return CB["green"], 2.2, 1.0, 20
        return CB["gray"], 0.9, 0.5, 5

    for k in keys:
        rks = [nl_pos[k], nz_pos[k], rt_pos[k]]
        col, lw, alpha, z = style(k)
        for i in range(2):
            ax.plot([xs[i], xs[i + 1]], [rks[i], rks[i + 1]], color=col, lw=lw,
                    alpha=alpha, zorder=z, solid_capstyle="round")
        big = k in (COLLAPSE, RISE)
        ax.scatter(xs, rks, s=14 if big else 7, color=col, alpha=alpha, zorder=z + 1)
        name = pretty(k, noiseless)
        lbl_col = col if big else "#555555"
        wt = "bold" if big else "normal"
        # left names (noiseless), right names (retrained); middle rank numbers only.
        # "--" marks a solver that did not clear the hr floor on that board.
        ax.text(xs[0] - 0.08, rks[0], f"{nl_rank[k]}. {name}", ha="right", va="center",
                fontsize=LBL_FS, color=lbl_col, fontweight=wt, zorder=z + 2)
        mid_lbl = str(nz_rank[k]) if k in nz_rank else "--"
        ax.text(xs[1] - 0.10, rks[1], mid_lbl, ha="right", va="center",
                fontsize=LBL_FS - 0.5, color=lbl_col, fontweight=wt, zorder=z + 2)
        rlabel = f"{rt_rank[k]}  {name}" if k in rt_rank else f"--  {name}"
        ax.text(xs[2] + 0.08, rks[2], rlabel, ha="left", va="center",
                fontsize=LBL_FS, color=lbl_col, fontweight=wt, zorder=z + 2)

    ax.set_xlim(xs[0] - 1.15, xs[2] + 1.35)
    top = 0.15
    ax.set_ylim(bottom + 0.6, top)                    # inverted: rank 1 at top
    ax.set_xticks([]); ax.set_yticks([])
    heads = [(xs[0], "Noiseless", "right"),
             (xs[1], "Noisy\n(no retrain)", "center"),
             (xs[2], "Noisy\n(retrained)", "left")]
    for x, t, ha in heads:
        ax.text(x, top - 0.55, t, ha=ha, va="bottom", fontsize=HDR_FS, fontweight="bold")
    for sp in ("left", "bottom"):
        ax.spines[sp].set_visible(False)
    ax.grid(False)
    fig.tight_layout()
    savefig(fig, os.path.join(OUTDIR, "fig5_retrain_slope.pdf"))


def main():
    print(f"Writing figures into {OUTDIR}")
    dropped = fig3_reversal()
    fig_retrain_slope()
    fig2_params_vs_hr()
    fig1_agentic_loop()
    champ, n = sfig_effectsize()
    sfig_pareto()
    sfig_hint_climb()
    sfig_mayo_significance()

    print("\nSummary")
    print(f"  effect-size champion: {champ}  (n={n} cases)")
    if dropped:
        print("  fig3 dropped (ranked noiseless but not on noisy board, "
              "excluding the forced-to-bottom collapse solver):")
        for k in dropped:
            print(f"    - {k}")
    else:
        print("  fig3: no solvers dropped.")
    print("\n  fig4 (recon panels) SKIPPED — needs cluster recon PNGs.")


if __name__ == "__main__":
    main()
