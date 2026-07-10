#!/usr/bin/env python3
"""Figure 4 for the Medical Physics paper: reconstruction panels showing the
noise-robustness reversal.

2 rows (solvers) x 4 columns (Truth | Recon clean-input | Recon noisy-input |
Difference noisy-truth). Row 1 = dual-domain-supervised (noiseless champion
that collapses under noise); Row 2 = learned-primal-dual (noisy champion).

Run on the cluster:
    source .venv/bin/activate
    export HDF5_USE_FILE_LOCKING=FALSE
    python scripts/make_fig4_panels.py

Arrays live under runs/<run-id>.../test/recon_raw.npz with keys pred/truth/baseline.
Truth is identical across all four files (verified). SSIM/hr are computed
FOV-masked vs the clean truth. Numbers are the source of truth.
"""
import os
import numpy as np
import hdf5plugin  # noqa: F401  (registers HDF5 codecs; harmless for npz)

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42   # embed TrueType, editable text
matplotlib.rcParams["ps.fonttype"] = 42
import matplotlib.pyplot as plt
from matplotlib import gridspec
from skimage.metrics import structural_similarity as ssim

ROOT = "/cluster/maier/Agent4CT"
BASE = os.path.join(ROOT, "runs", "breast-ct-claude-agentic-")
PATHS = {
    "dds_clean": BASE + "dual-domain-supervised-search-20260703-01-itertest__iter-0020-breasttest/test/recon_raw.npz",
    "dds_noisy": BASE + "dual-domain-supervised-search-20260703-01-itertest__iter-0020-breasttest-noise100000/test/recon_raw.npz",
    "lpd_clean": BASE + "learned-primal-dual-search-20260703-01-itertest__iter-0012-breasttest/test/recon_raw.npz",
    "lpd_noisy": BASE + "learned-primal-dual-search-20260703-01-itertest__iter-0012-breasttest-noise100000/test/recon_raw.npz",
}

K = 117               # representative test-case index (mid-volume, strong fibroglandular structure)
DISPLAY_MIN = 0.0
DISPLAY_MAX = 0.25    # solid-tissue breast phantom: bulk mu ~0.19-0.23; 0.05 would saturate everything
DIFF_ABS = 0.10       # symmetric diverging scale for (noisy recon - truth)
OUT_PDF = os.path.join(ROOT, "paper", "figures", "fig4_recon_panels.pdf")


def fov_mask(h, w):
    yy, xx = np.mgrid[0:h, 0:w]
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    return r <= (h / 2.0)


def main():
    D = {k: np.load(v) for k, v in PATHS.items()}
    truth_all = D["dds_clean"]["truth"][:, 0]
    baseline_all = D["dds_clean"]["baseline"][:, 0]
    H, W = truth_all.shape[1:]
    fov = fov_mask(H, W)

    truth = truth_all[K] * fov
    baseline = baseline_all[K] * fov
    dr = float(truth.max() - truth.min())  # data_range = truth dynamic range

    def get(key):
        return D[key]["pred"][K, 0] * fov

    recons = {k: get(k) for k in ("dds_clean", "dds_noisy", "lpd_clean", "lpd_noisy")}

    def metrics(pred):
        s = float(ssim(truth, pred, data_range=dr))
        rmse = float(np.sqrt(np.mean((pred[fov] - truth[fov]) ** 2)))
        base_rmse = float(np.sqrt(np.mean((baseline[fov] - truth[fov]) ** 2)))
        hr = max(0.0, 1.0 - rmse / base_rmse)
        return s, hr

    M = {k: metrics(v) for k, v in recons.items()}
    print(f"case k = {K}   FOV-masked, data_range = {dr:.4f}")
    for k in ("dds_clean", "dds_noisy", "lpd_clean", "lpd_noisy"):
        print(f"  {k:10s}  SSIM={M[k][0]:.4f}  hr={M[k][1]:.4f}")

    # ---- figure ----------------------------------------------------------
    fig = plt.figure(figsize=(11.0, 6.0))
    gs = gridspec.GridSpec(
        2, 4, figure=fig,
        left=0.075, right=0.99, top=0.90, bottom=0.02,
        wspace=0.04, hspace=0.06,
    )

    col_headers = ["Truth", "Recon (clean input)", "Recon (noisy input)", "Difference"]
    row_labels = ["dual-domain-supervised", "learned-primal-dual"]
    rows = [("dds_clean", "dds_noisy"), ("lpd_clean", "lpd_noisy")]

    gray_kw = dict(cmap="gray", vmin=DISPLAY_MIN, vmax=DISPLAY_MAX)
    im_gray = None
    im_diff = None

    def annotate(ax, key):
        s, hr = M[key]
        ax.text(0.03, 0.03, f"SSIM {s:.3f}\nhr {hr:.3f}", transform=ax.transAxes,
                fontsize=8.5, va="bottom", ha="left", color="white",
                bbox=dict(boxstyle="round,pad=0.2", fc="black", ec="none", alpha=0.55))

    for r, (clean_key, noisy_key) in enumerate(rows):
        # col 0: Truth (same in both rows)
        ax0 = fig.add_subplot(gs[r, 0])
        im_gray = ax0.imshow(truth, **gray_kw)
        ax0.set_xticks([]); ax0.set_yticks([])
        ax0.set_ylabel(row_labels[r], fontsize=11, labelpad=6)

        # col 1: clean-input recon
        ax1 = fig.add_subplot(gs[r, 1])
        ax1.imshow(recons[clean_key], **gray_kw)
        ax1.set_xticks([]); ax1.set_yticks([])
        annotate(ax1, clean_key)

        # col 2: noisy-input recon
        ax2 = fig.add_subplot(gs[r, 2])
        ax2.imshow(recons[noisy_key], **gray_kw)
        ax2.set_xticks([]); ax2.set_yticks([])
        annotate(ax2, noisy_key)

        # col 3: difference (noisy recon - truth)
        ax3 = fig.add_subplot(gs[r, 3])
        diff = (recons[noisy_key] - truth) * fov
        im_diff = ax3.imshow(diff, cmap="RdBu_r", vmin=-DIFF_ABS, vmax=DIFF_ABS)
        ax3.set_xticks([]); ax3.set_yticks([])

        # column headers on the top row
        if r == 0:
            for ax, htxt in zip((ax0, ax1, ax2, ax3), col_headers):
                ax.set_title(htxt, fontsize=11, pad=4)

    # colorbars
    # grayscale colorbar (thin, right of image columns) under col 0..2 group
    cax_g = fig.add_axes([0.075, 0.005, 0.55, 0.012])
    cb_g = fig.colorbar(im_gray, cax=cax_g, orientation="horizontal")
    cb_g.set_label(r"$\mu$  (display window)", fontsize=8)
    cb_g.ax.tick_params(labelsize=7)

    cax_d = fig.add_axes([0.78, 0.005, 0.20, 0.012])
    cb_d = fig.colorbar(im_diff, cax=cax_d, orientation="horizontal")
    cb_d.set_label(r"noisy recon $-$ truth", fontsize=8)
    cb_d.ax.tick_params(labelsize=7)

    os.makedirs(os.path.dirname(OUT_PDF), exist_ok=True)
    fig.savefig(OUT_PDF, format="pdf", dpi=200)
    print("wrote", OUT_PDF, os.path.getsize(OUT_PDF), "bytes")


if __name__ == "__main__":
    main()
