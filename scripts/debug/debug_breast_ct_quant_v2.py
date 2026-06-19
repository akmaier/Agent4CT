"""Quantitative comparison v2: ALL panels use the SAME display range
[0, display_max=0.5] so calibrated recons are visually comparable.

Layout: 4 cases × 7 cols
  truth | Sidky FBP128 (calibrated) | OUR shift=0/+32/-32 (calibrated) | diff(best - truth) | diff(Sidky - truth)
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

DISPLAY_MIN, DISPLAY_MAX = 0.0, 0.5    # breast_ct dataset display range


def metrics_v(pred, truth):
    pc = intensity_calibrate(pred.clamp_min(0.0), truth, display_max=DISPLAY_MAX)
    ss = float(ssim(pc, truth, data_range=DISPLAY_MAX).cpu())
    ps = float(psnr(pc, truth, data_range=DISPLAY_MAX).cpu())
    rm = float(((pc - truth)**2).mean().sqrt().cpu())
    return ss, ps, rm, pc


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data = Path("/cluster/maier/Agent4CT/data/dl_sparse_view/staged")
    with h5py.File(data / "val_truth.h5", "r") as f: truth = f["image"][:4]
    with h5py.File(data / "val_sinograms.h5", "r") as f: sino = f["sino"][:4]
    with h5py.File(data / "val_fbp128.h5", "r") as f: ref = f["image"][:4]

    s_all = torch.from_numpy(sino).float().to(device).unsqueeze(1)
    t_all = torch.from_numpy(truth).float().to(device).unsqueeze(1)
    r_all = torch.from_numpy(ref).float().to(device).unsqueeze(1)
    geom = FanBeamGeometry(**GEOM); proj = PyronnFanBeamProjector(geom).to(device)

    shifts = [0, 32, -32]
    shift_labels = ["shift=0", "shift=+32 (-90°CW)", "shift=-32 (+90°CCW)"]

    fig, axes = plt.subplots(4, 7, figsize=(26, 14))
    for r in range(4):
        gt = t_all[r:r+1]
        gt_np = truth[r]

        # COMMON display window: [0, display_max=0.5]
        def show_gray(ax, img, title):
            ax.imshow(img, cmap="gray", vmin=DISPLAY_MIN, vmax=DISPLAY_MAX)
            ax.set_title(title, fontsize=9); ax.axis("off")

        # Truth — already in display range (no calibration needed; it IS the reference)
        show_gray(axes[r, 0], gt_np,
                   f"truth #{r}\n[{gt_np.min():.3f}, {gt_np.max():.3f}]")

        # Sidky FBP128 — intensity-calibrated against truth
        sk_ss, sk_ps, sk_rm, sk_cal = metrics_v(r_all[r:r+1], gt)
        sk_np = sk_cal[0, 0].cpu().numpy()
        show_gray(axes[r, 1], sk_np,
                   f"Sidky FBP128 (cal.)\nSSIM={sk_ss:.3f} PSNR={sk_ps:.1f}dB\n"
                   f"RMSE={sk_rm:.4f}  range=[{sk_np.min():.3f},{sk_np.max():.3f}]")

        per_shift = []
        for col, (sh, label) in enumerate(zip(shifts, shift_labels), start=2):
            s = torch.roll(s_all[r:r+1], shifts=sh, dims=-2)
            with torch.no_grad():
                rec = proj.fbp(s, filter_name="hann")
            ss, ps, rm, rec_cal = metrics_v(rec, gt)
            per_shift.append((label, rec_cal, ss, ps, rm, sh))
            rec_np = rec_cal[0, 0].cpu().numpy()
            show_gray(axes[r, col], rec_np,
                       f"OUR {label}  (cal.)\nSSIM={ss:.3f} PSNR={ps:.1f}dB\n"
                       f"RMSE={rm:.4f}  range=[{rec_np.min():.3f},{rec_np.max():.3f}]")

        best = max(per_shift, key=lambda x: x[2])
        # Diff with FIXED symmetric range
        diff_lim = DISPLAY_MAX / 4   # consistent across both diff cols
        diff_ours = (best[1] - gt)[0, 0].cpu().numpy()
        axes[r, 5].imshow(diff_ours, cmap="bwr", vmin=-diff_lim, vmax=diff_lim)
        axes[r, 5].set_title(f"OUR - truth (best: {best[0]})\n"
                              f"|err|max={float(np.abs(diff_ours).max()):.3f}",
                              fontsize=9); axes[r, 5].axis("off")
        diff_sk = (sk_cal - gt)[0, 0].cpu().numpy()
        axes[r, 6].imshow(diff_sk, cmap="bwr", vmin=-diff_lim, vmax=diff_lim)
        axes[r, 6].set_title(f"Sidky - truth (cal.)\n"
                              f"|err|max={float(np.abs(diff_sk).max()):.3f}",
                              fontsize=9); axes[r, 6].axis("off")

    plt.suptitle(f"All gray panels at vmin=0, vmax={DISPLAY_MAX} (breast_ct display range). "
                  f"All diffs at ±{diff_lim:.3f}. ALL recons intensity-calibrated.",
                  fontsize=11, y=1.001)
    plt.tight_layout()
    out = Path("/cluster/maier/Agent4CT/results/breast_debug/quant_v2.png")
    plt.savefig(out, dpi=80, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {out}")


if __name__ == "__main__":
    main()
