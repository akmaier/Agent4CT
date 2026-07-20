"""make_example_slices.py — one example CT slice per challenge for the main paper.

A 2x2 panel: rows = Mayo low-dose (top) and breast 128-view sparse (bottom);
columns = the FBP input the solvers start from and the ground truth. Shows both
the anatomy and the degradation each problem poses (low-dose quantum noise vs
sparse-view streaks). Reads truth/baseline from the same recon_raw.npz arrays the
boards were scored from. Run on the cluster (data lives there).

Usage (cluster):  python scripts/make_example_slices.py
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
matplotlib.rcParams["font.size"] = 8
matplotlib.rcParams["font.family"] = "sans-serif"

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MAYO = os.path.join(ROOT, "runs",
    "mayo-ldct-claude-agentic-itnet-search-20260619-01-testset/L058/recon_raw.npz")
BREAST = os.path.join(ROOT, "runs",
    "breast-ct-claude-agentic-dual-domain-supervised-search-20260703-01-itertest__iter-0020-breasttest/test/recon_raw.npz")
OUT = os.path.join(ROOT, "paper", "figures", "fig_data_examples.pdf")


def fov(h, w):
    yy, xx = np.mgrid[0:h, 0:w]
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    r = min(h, w) / 2.0 - 1
    return ((yy - cy) ** 2 + (xx - cx) ** 2) <= r * r


def load(path, k):
    d = np.load(path)
    truth = d["truth"][:, 0]
    base = d["baseline"][:, 0]
    k = k if k is not None else truth.shape[0] // 2
    return base[k], truth[k]


def calibrate(img, truth, f):
    """Two-point (linear) intensity calibration of img to truth over the FOV, the
    same calibration the scoring metric applies before RMSE, so the FBP input is
    shown on the truth's intensity scale rather than its raw (offset) scale."""
    x = img[f].astype(np.float64)
    y = truth[f].astype(np.float64)
    a, b = np.linalg.lstsq(np.vstack([x, np.ones_like(x)]).T, y, rcond=None)[0]
    return a * img + b


def main():
    m_fbp, m_truth = load(MAYO, None)     # central slice of the patient volume
    b_fbp, b_truth = load(BREAST, 100)    # a mid-index test case
    H, W = m_truth.shape
    f = fov(H, W)

    # Single short row of four panels: Mayo (FBP, truth), breast (FBP, truth).
    groups = [("Mayo (low-dose)", m_fbp, m_truth),
              ("Breast (128-view sparse)", b_fbp, b_truth)]
    fig, axs = plt.subplots(1, 4, figsize=(3.3, 1.28))
    titles = ["FBP", "truth", "FBP", "truth"]
    panels = []
    for lbl, fbp, truth in groups:
        panels.append(("gt", truth, truth))
        vmin = float(np.percentile(truth[f], 1))
        vmax = float(np.percentile(truth[f], 99))
        panels[-1] = (calibrate(fbp, truth, f), truth, (vmin, vmax))
    seq = [(panels[0][0], panels[0][2]), (panels[0][1], panels[0][2]),
           (panels[1][0], panels[1][2]), (panels[1][1], panels[1][2])]
    for k, (img, (vmin, vmax)) in enumerate(seq):
        ax = axs[k]
        m = img.copy(); m[~f] = vmin
        ax.imshow(m, cmap="gray", vmin=vmin, vmax=vmax, interpolation="none")
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_title(titles[k], fontsize=7.5)
    # group labels beneath each pair
    fig.text(0.27, 0.015, "Mayo (low-dose)", ha="center", fontsize=7.5, fontweight="bold")
    fig.text(0.76, 0.015, "Breast (128-view sparse)", ha="center", fontsize=7.5, fontweight="bold")
    fig.tight_layout(rect=[0, 0.08, 1, 1], w_pad=0.3)
    fig.savefig(OUT, bbox_inches="tight", dpi=600)
    print("wrote", OUT, "  Mayo slice window [%.3f,%.3f], breast [%.3f,%.3f]" % (
        np.percentile(m_truth[f], 1), np.percentile(m_truth[f], 99),
        np.percentile(b_truth[f], 1), np.percentile(b_truth[f], 99)))


if __name__ == "__main__":
    main()
