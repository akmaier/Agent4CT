"""Quantitative version of the start-angle decision. Per case + per
shift, compute SSIM / PSNR / RMSE of OUR FBP against the GROUND TRUTH
(not against Sidky FBP128). Also report the same for Sidky FBP128 vs
truth as the upper-bound reference. Layout:
  4 cases × 7 cols:
    truth | Sidky FBP128 | OUR@0 | OUR@+32 | OUR@-32 | diff(OUR_best - truth) | diff(Sidky - truth)
  Title of each panel: SSIM/PSNR/RMSE vs truth (after intensity calibration).
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


def metrics_vs_truth(pred_t, truth_t, display_max=0.5):
    """SSIM / PSNR / RMSE of pred (after intensity calibration) vs truth."""
    pred_cal = intensity_calibrate(pred_t.clamp_min(0.0), truth_t, display_max=display_max)
    dr = display_max
    ss = float(ssim(pred_cal, truth_t, data_range=dr).cpu())
    ps = float(psnr(pred_cal, truth_t, data_range=dr).cpu())
    rm = float(((pred_cal - truth_t) ** 2).mean().sqrt().cpu())
    return ss, ps, rm, pred_cal


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
    shift_labels = ["shift=0", "shift=+32 (+90°)", "shift=-32 (-90° CW)"]

    fig, axes = plt.subplots(4, 7, figsize=(26, 14))
    summary = {label: {"ssim": [], "psnr": [], "rmse": []} for label in shift_labels}
    sidky_summary = {"ssim": [], "psnr": [], "rmse": []}

    for r in range(4):
        gt = t_all[r:r+1]
        gt_np = truth[r]
        # --- Sidky FBP128 reference ---
        sk_ss, sk_ps, sk_rm, sk_cal = metrics_vs_truth(r_all[r:r+1], gt)
        sidky_summary["ssim"].append(sk_ss); sidky_summary["psnr"].append(sk_ps); sidky_summary["rmse"].append(sk_rm)

        # auto-scale per panel
        def show(ax, img, title):
            lo = float(np.percentile(img, 1)); hi = float(np.percentile(img, 99.5))
            ax.imshow(img, cmap="gray", vmin=lo, vmax=hi)
            ax.set_title(title, fontsize=9); ax.axis("off")

        show(axes[r, 0], gt_np, f"truth #{r}\n[{gt_np.min():.3f}, {gt_np.max():.3f}]")
        show(axes[r, 1], sk_cal[0, 0].cpu().numpy(),
              f"Sidky FBP128 vs truth\nSSIM={sk_ss:.3f} PSNR={sk_ps:.1f}dB\nRMSE={sk_rm:.4f}")

        per_shift_recons = []
        for col, (sh, label) in enumerate(zip(shifts, shift_labels), start=2):
            s = torch.roll(s_all[r:r+1], shifts=sh, dims=-2)
            with torch.no_grad():
                rec = proj.fbp(s)
            ss, ps, rm, rec_cal = metrics_vs_truth(rec, gt)
            per_shift_recons.append((label, rec_cal, ss, ps, rm))
            summary[label]["ssim"].append(ss); summary[label]["psnr"].append(ps); summary[label]["rmse"].append(rm)
            show(axes[r, col], rec_cal[0, 0].cpu().numpy(),
                  f"OUR {label}\nSSIM={ss:.3f} PSNR={ps:.1f}dB\nRMSE={rm:.4f}")

        # Pick the best-SSIM shift for the diff column
        best = max(per_shift_recons, key=lambda x: x[2])
        diff_ours = (best[1] - gt)[0, 0].cpu().numpy()
        diff_sidky = (sk_cal - gt)[0, 0].cpu().numpy()
        lim = float(np.percentile(np.abs(diff_sidky), 99))
        axes[r, 5].imshow(diff_ours, cmap="bwr", vmin=-lim, vmax=lim)
        axes[r, 5].set_title(f"OUR-truth diff\n(best={best[0]})\n|err|99={float(np.percentile(np.abs(diff_ours),99)):.4f}",
                              fontsize=9); axes[r, 5].axis("off")
        axes[r, 6].imshow(diff_sidky, cmap="bwr", vmin=-lim, vmax=lim)
        axes[r, 6].set_title(f"Sidky-truth diff\n|err|99={float(np.percentile(np.abs(diff_sidky),99)):.4f}",
                              fontsize=9); axes[r, 6].axis("off")

    plt.tight_layout()
    out = Path("/cluster/maier/Agent4CT/results/breast_debug/quant.png")
    plt.savefig(out, dpi=80, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {out}")

    print("\n=== SSIM / PSNR / RMSE vs ground truth (n=4 cases, after intensity calibration) ===")
    print(f"{'variant':<24}  {'SSIM':>6}  {'PSNR':>6}  {'RMSE':>7}")
    for label in shift_labels:
        ss = np.mean(summary[label]["ssim"])
        ps = np.mean(summary[label]["psnr"])
        rm = np.mean(summary[label]["rmse"])
        print(f"OUR {label:<18}   {ss:.4f}  {ps:5.2f}   {rm:.4f}")
    ss = np.mean(sidky_summary["ssim"]); ps = np.mean(sidky_summary["psnr"]); rm = np.mean(sidky_summary["rmse"])
    print(f"{'Sidky FBP128 (ref)':<24}   {ss:.4f}  {ps:5.2f}   {rm:.4f}")


if __name__ == "__main__":
    main()
