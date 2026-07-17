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
    x_left, x_right = 0.0, 1.0  # rank lists close enough to keep arrows readable in one column

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
        ax.text(x_left - 0.05, lr, f"{lr}. {name}", ha="right", va="center",
                fontsize=7, color=lbl_col, fontweight=weight, zorder=z + 2)
        # right labels: rank number + DNF marker for the collapse solver
        if k in nz_rank:
            rlabel = f"{name}  {rr}"
        else:
            rlabel = f"{name}  (DNF)"
        ax.text(x_right + 0.05, rr, rlabel, ha="left", va="center",
                fontsize=7, color=lbl_col, fontweight=weight, zorder=z + 2)

    ax.set_xlim(-1.7, 2.7)
    top = 0.15
    ax.set_ylim(collapse_dest + 0.6, top)  # inverted: rank 1 at top
    ax.set_xticks([])
    ax.set_yticks([])
    # column headers anchored to the node columns (x=0 left, x=1 right) but
    # aligned so their text grows outward, away from each other.
    ax.text(x_left, top - 0.12, "Noiseless rank", ha="right", va="bottom",
            fontsize=8, fontweight="bold")
    ax.text(x_right, top - 0.12, "Noisy (I$_0$=100k) rank", ha="left",
            va="bottom", fontsize=8, fontweight="bold")
    for sp in ("left", "bottom"):
        ax.spines[sp].set_visible(False)
    ax.grid(False)

    fig.tight_layout()
    savefig(fig, os.path.join(OUTDIR, "fig3_reversal.pdf"))
    return dropped


# ===========================================================================
# FIG 2 — params (log) vs test hr, Mayo & breast(noiseless)
# ===========================================================================
def fig2_params_vs_hr():
    mayo = load_board("mayo_ldct")
    breast = load_board("breast_ct")
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
            # params_M given in millions; param-efficient is ~195 params.
            # Guard: a genuine 0.0 (TV etc.) can't sit on a log axis; give
            # it a tiny floor so it still plots, but keep highlighted solver
            # at its true count.
            params = p * 1e6
            if params <= 0:
                params = 0.5  # < 1 param -> floor so log-x is defined
            out.append((r["solver_key"], params, hr))
        return out

    mayo_pts = pts(mayo)
    breast_pts = pts(breast)

    fig, ax = plt.subplots(figsize=(SINGLE, 2.8))

    def scatter(data, color, marker, label):
        xs = [d[1] for d in data]
        ys = [d[2] for d in data]
        ax.scatter(xs, ys, s=22, color=color, marker=marker,
                   edgecolors="white", linewidths=0.4, alpha=0.85,
                   label=label, zorder=3)

    scatter(mayo_pts, CB["blue"], "o", "Mayo-LDCT")
    scatter(breast_pts, CB["orange"], "^", "Breast (noiseless)")

    # Highlight param-efficient on both boards.
    for data, color in ((mayo_pts, CB["blue"]), (breast_pts, CB["orange"])):
        for k, x, y in data:
            if k == HILITE:
                ax.scatter([x], [y], s=90, facecolors="none",
                           edgecolors=CB["red"], linewidths=1.6, zorder=5)
                ax.annotate("param-efficient\n(~195 params)",
                            (x, y), textcoords="offset points",
                            xytext=(8, -2), fontsize=6.8, color=CB["red"],
                            fontweight="bold",
                            arrowprops=dict(arrowstyle="-", color=CB["red"],
                                            lw=0.6))

    ax.set_xscale("log")
    ax.set_xlabel("trainable parameters (log scale)")
    ax.set_ylabel("test hr (mean)")
    ax.legend(loc="lower right", frameon=False, handletextpad=0.3)
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
def main():
    print(f"Writing figures into {OUTDIR}")
    dropped = fig3_reversal()
    fig2_params_vs_hr()
    fig1_agentic_loop()
    champ, n = sfig_effectsize()
    sfig_pareto()

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
