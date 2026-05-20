"""Sweep a small additive constant subtracted from val_sinograms.h5 to see if
it kills the FBP cupping. If Sidky's data has a constant offset (noise floor
/ BH baseline / `-log(I/I0)` bias), subtracting it should flatten our recon
vs truth.

Sweep c ∈ {-0.05, -0.02, 0, +0.02, +0.05, +0.10, +0.15, +0.20, +0.25, +0.30}.
For each c: FBP((sino - c).clamp_min(0)), intensity-calibrate vs truth, score.

Outputs:
  /cluster/maier/Agent4CT/results/breast_debug/sino_offset_sweep.png   (metric curves + per-case panels at best c)
"""
from __future__ import annotations
import sys
from pathlib import Path
REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path: sys.path.insert(0, str(REPO))

import numpy as np
import torch
import hdf5plugin  # noqa
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ddssl_ldct.geometry import FanBeamGeometry
from ddssl_ldct.pyronn_projector import PyronnFanBeamProjector
from ddssl_ldct.metrics import psnr, ssim, intensity_calibrate

GEOM = dict(image_size=512, pixel_spacing=180/512, n_angles=128,
            n_det=1024, det_spacing=360/1024, sod=500.0, sdd=1000.0)
DISPLAY_MAX = 0.5
SINO_SHIFT = 32
OUT_DIR = Path("/cluster/maier/Agent4CT/results/breast_debug")
C_VALUES = [-0.05, -0.02, 0.0, 0.02, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]


def metrics_v(pred, truth, dmax=DISPLAY_MAX):
    pc = intensity_calibrate(pred.clamp_min(0.0), truth, display_max=dmax)
    ss = float(ssim(pc, truth, data_range=dmax).cpu())
    ps = float(psnr(pc, truth, data_range=dmax).cpu())
    rm = float(((pc - truth) ** 2).mean().sqrt().cpu())
    return ss, ps, rm, pc


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    data = Path("/cluster/maier/Agent4CT/data/dl_sparse_view/staged")
    n = 4
    with h5py.File(data / "val_truth.h5", "r") as f: truth = f["image"][:n]
    with h5py.File(data / "val_sinograms.h5", "r") as f: sino = f["sino"][:n]

    s_all = torch.from_numpy(sino).float().to(device).unsqueeze(1)
    t_all = torch.from_numpy(truth).float().to(device).unsqueeze(1)
    s_all = torch.roll(s_all, shifts=SINO_SHIFT, dims=-2)
    geom = FanBeamGeometry(**GEOM); proj = PyronnFanBeamProjector(geom).to(device)

    # Sweep c on all cases ────────────────────────────────────────────────────
    print(f"\n{'c':>7}  {'SSIM':>7}  {'PSNR':>6}  {'RMSE':>7}    (mean over n=4)")
    ssim_curve, psnr_curve, rmse_curve = [], [], []
    per_c_recs = {}                                # c -> (cal_recs, ss_list)
    for c in C_VALUES:
        s_off = (s_all - c).clamp_min(0.0)        # subtract & non-negative
        with torch.no_grad():
            rec = proj.fbp(s_off, filter_name="hann")
        ss_list, ps_list, rm_list = [], [], []
        cal_recs = []
        for r in range(n):
            ss, ps, rm, rec_cal = metrics_v(rec[r:r+1], t_all[r:r+1])
            ss_list.append(ss); ps_list.append(ps); rm_list.append(rm)
            cal_recs.append(rec_cal[0, 0].cpu().numpy())
        per_c_recs[c] = (cal_recs, ss_list)
        ssim_curve.append(np.mean(ss_list)); psnr_curve.append(np.mean(ps_list)); rmse_curve.append(np.mean(rm_list))
        print(f"{c:>+7.3f}  {ssim_curve[-1]:7.4f}  {psnr_curve[-1]:6.2f}  {rmse_curve[-1]:7.4f}")

    best_idx = int(np.argmax(ssim_curve))
    best_c = C_VALUES[best_idx]
    print(f"\nbest c by SSIM: {best_c:+.3f}  (mean SSIM={ssim_curve[best_idx]:.4f}, "
          f"PSNR={psnr_curve[best_idx]:.2f}, RMSE={rmse_curve[best_idx]:.4f})")

    # Figure: top = sweep curves; bottom = 4 cases at {0, best_c}, recon + diff vs truth
    fig = plt.figure(figsize=(22, 14))
    gs = fig.add_gridspec(3, 8, height_ratios=[1.0, 1.4, 1.4], wspace=0.25, hspace=0.30)

    # Row 0: sweep curves
    ax_s = fig.add_subplot(gs[0, 0:3])
    ax_s.plot(C_VALUES, ssim_curve, "o-", color="C0", label="SSIM")
    ax_s.set_xlabel("subtracted constant c"); ax_s.set_ylabel("SSIM", color="C0")
    ax_s.grid(alpha=0.3); ax_s.set_title("SSIM vs sino-subtract c (4-case mean)", fontsize=10)
    ax_p = ax_s.twinx()
    ax_p.plot(C_VALUES, psnr_curve, "s--", color="C3", label="PSNR")
    ax_p.set_ylabel("PSNR (dB)", color="C3")
    ax_s.axvline(best_c, ls=":", color="k", linewidth=0.8); ax_s.annotate(f"best c={best_c:+.3f}", xy=(best_c, ssim_curve[best_idx]), xytext=(5, 5), textcoords="offset points", fontsize=9)

    # Row 0 cols 3..7: recon at c=0 vs best_c for case #0 (overview), + diff strip
    gt0 = t_all[0:1]; gt0_np = truth[0]
    fig.add_subplot(gs[0, 3]).set_title(f"truth #0", fontsize=9)
    ax = fig.axes[-1]
    ax.imshow(gt0_np, cmap="gray", vmin=0, vmax=DISPLAY_MAX); ax.axis("off")
    for ci, c in enumerate([0.0, best_c], start=4):
        rec_np = per_c_recs[c][0][0]
        ss = per_c_recs[c][1][0]
        ax = fig.add_subplot(gs[0, ci])
        ax.imshow(rec_np, cmap="gray", vmin=0, vmax=DISPLAY_MAX)
        ax.set_title(f"recon #0  c={c:+.3f}\nSSIM={ss:.3f}", fontsize=9); ax.axis("off")
        ax = fig.add_subplot(gs[0, ci + 2])
        diff = rec_np - gt0_np
        lim = DISPLAY_MAX / 4
        ax.imshow(diff, cmap="bwr", vmin=-lim, vmax=lim)
        ax.set_title(f"diff #0  c={c:+.3f}\n|err|max={float(np.abs(diff).max()):.3f}", fontsize=9); ax.axis("off")

    # Rows 1-2: 4 cases × {c=0, c=best_c}: row 1 = recon at c=0 and at best_c
    #                                      row 2 = diff at c=0 and at best_c
    for r in range(n):
        gt_np = truth[r]
        for j, c in enumerate([0.0, best_c]):
            rec_np = per_c_recs[c][0][r]
            ss = per_c_recs[c][1][r]
            # recon panel (row 1)
            ax = fig.add_subplot(gs[1, r * 2 + j])
            ax.imshow(rec_np, cmap="gray", vmin=0, vmax=DISPLAY_MAX)
            ax.set_title(f"recon #{r}  c={c:+.3f}\nSSIM={ss:.3f}", fontsize=9)
            ax.axis("off")
            # diff panel (row 2)
            ax = fig.add_subplot(gs[2, r * 2 + j])
            diff = rec_np - gt_np
            lim = DISPLAY_MAX / 4
            ax.imshow(diff, cmap="bwr", vmin=-lim, vmax=lim)
            ax.set_title(f"diff #{r}  c={c:+.3f}", fontsize=9)
            ax.axis("off")

    plt.suptitle(f"Sino-subtract sweep — does cupping vanish at some c? best c={best_c:+.3f}",
                 fontsize=12, y=1.001)
    out = OUT_DIR / "sino_offset_sweep.png"
    plt.savefig(out, dpi=80, bbox_inches="tight")
    plt.close(fig)
    print(f"\n[saved] {out}")


if __name__ == "__main__":
    main()
