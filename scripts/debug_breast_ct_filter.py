"""Multi-axis sweep at the verified Sidky geometry + shift=+32:
  - filter:       ram_lak / hann / shepp_logan / cosine / hamming
  - detector axis flip:  on / off
  - fine angle shift around +32:  +30, +31, +32, +33, +34
Each variant scored by SSIM/PSNR/RMSE against ground truth (after
intensity calibration). Report the best combo + a figure of the top-3.
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

FILTERS = ["ramlak", "hann", "shepp-logan", "cosine", "hamming"]
SHIFTS = [30, 31, 32, 33, 34]


def metrics_v(pred, truth, dmax=0.5):
    pc = intensity_calibrate(pred.clamp_min(0.0), truth, display_max=dmax)
    ss = float(ssim(pc, truth, data_range=dmax).cpu())
    ps = float(psnr(pc, truth, data_range=dmax).cpu())
    rm = float(((pc - truth)**2).mean().sqrt().cpu())
    return ss, ps, rm, pc


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data = Path("/cluster/maier/Agent4CT/data/dl_sparse_view/staged")
    n_cases = 8
    with h5py.File(data / "val_truth.h5", "r") as f: truth = f["image"][:n_cases]
    with h5py.File(data / "val_sinograms.h5", "r") as f: sino = f["sino"][:n_cases]
    with h5py.File(data / "val_fbp128.h5", "r") as f: ref = f["image"][:n_cases]

    s_all = torch.from_numpy(sino).float().to(device).unsqueeze(1)
    t_all = torch.from_numpy(truth).float().to(device).unsqueeze(1)
    r_all = torch.from_numpy(ref).float().to(device).unsqueeze(1)
    geom = FanBeamGeometry(**GEOM); proj = PyronnFanBeamProjector(geom).to(device)

    # Sidky FBP128 reference
    sk_ss, sk_ps, sk_rm, _ = metrics_v(r_all, t_all)
    print(f"Sidky FBP128 (ref)    SSIM={sk_ss:.4f}  PSNR={sk_ps:5.2f}  RMSE={sk_rm:.4f}\n")

    # Sweep: filter × flip × shift  (5 × 2 × 5 = 50 combos)
    results = []
    for filt in FILTERS:
        for flip_det in [False, True]:
            for sh in SHIFTS:
                s = torch.roll(s_all, shifts=sh, dims=-2)
                if flip_det:
                    s = torch.flip(s, dims=(-1,))
                with torch.no_grad():
                    rec = proj.fbp(s, filter_name=filt)
                ss, ps, rm, _ = metrics_v(rec, t_all)
                tag = f"filter={filt:<11} flip={int(flip_det)}  shift={sh:+d}"
                results.append((ss, ps, rm, tag, filt, flip_det, sh))
                print(f"  {tag}  SSIM={ss:.4f}  PSNR={ps:5.2f}  RMSE={rm:.4f}")

    results.sort(key=lambda r: -r[0])
    print("\n--- TOP 5 BY SSIM ---")
    for r in results[:5]:
        print(f"  SSIM={r[0]:.4f} PSNR={r[1]:5.2f} RMSE={r[2]:.4f}  {r[3]}")
    print(f"\nSidky FBP128 baseline: SSIM={sk_ss:.4f} PSNR={sk_ps:5.2f}")

    # Visualize top-3 vs truth + Sidky FBP128 (4 cases x 6 cols)
    top3 = results[:3]
    fig, axes = plt.subplots(min(n_cases, 4), 6, figsize=(22, 14))
    for r in range(min(n_cases, 4)):
        # truth + Sidky FBP128 (with intensity calibrate for fairness)
        _, _, _, sk_cal = metrics_v(r_all[r:r+1], t_all[r:r+1])
        def show(ax, im, title):
            lo = float(np.percentile(im, 1)); hi = float(np.percentile(im, 99.5))
            ax.imshow(im, cmap="gray", vmin=lo, vmax=hi); ax.set_title(title, fontsize=9); ax.axis("off")
        show(axes[r, 0], truth[r], f"truth #{r}")
        show(axes[r, 1], sk_cal[0, 0].cpu().numpy(),
              f"Sidky FBP128\nSSIM={sk_ss:.3f}")
        for ci, (ss, ps, rm, tag, filt, fl, sh) in enumerate(top3, start=2):
            s = torch.roll(s_all[r:r+1], shifts=sh, dims=-2)
            if fl: s = torch.flip(s, dims=(-1,))
            with torch.no_grad():
                rec = proj.fbp(s, filter_name=filt)
            _, _, _, rec_cal = metrics_v(rec, t_all[r:r+1])
            show(axes[r, ci], rec_cal[0, 0].cpu().numpy(),
                  f"{filt}  flip={int(fl)}  shift={sh:+d}\nSSIM={ss:.3f} PSNR={ps:.1f}")
        # Diff of top-1
        best = top3[0]
        s = torch.roll(s_all[r:r+1], shifts=best[6], dims=-2)
        if best[5]: s = torch.flip(s, dims=(-1,))
        with torch.no_grad():
            rec = proj.fbp(s, filter_name=best[4])
        _, _, _, rec_cal = metrics_v(rec, t_all[r:r+1])
        diff = (rec_cal - t_all[r:r+1])[0, 0].cpu().numpy()
        lim = float(np.percentile(np.abs(diff), 99))
        axes[r, 5].imshow(diff, cmap="bwr", vmin=-lim, vmax=lim)
        axes[r, 5].set_title(f"top1 - truth"); axes[r, 5].axis("off")

    plt.tight_layout()
    out = Path("/cluster/maier/Agent4CT/results/breast_debug/filter_sweep.png")
    plt.savefig(out, dpi=80, bbox_inches="tight")
    print(f"[saved] {out}")


if __name__ == "__main__":
    main()
