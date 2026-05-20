"""Visual expert-decision comparison: with Sidky geometry locked
(sod=500, sdd=1000, ps=ds=0.3516), apply -90° rotation via sino
start-angle shift (np.roll along angle axis by ±32 views, i.e. ±90°
out of 128 views over 360°). Suppress negatives via clamp_min(0).

4 cases × 5 columns:
  truth | Sidky FBP128 | recon @start-shift=0 | @+32 (=+90°) | @-32 (=-90° CW)

Per panel: auto-scaled to its own [1, 99.5] percentile.
Also reports SSIM and pearson corr against FBP128 after intensity
calibration so the metric is structural, not just "is there a bright
disc".
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
from ddssl_ldct.metrics import ssim as ssim_t, intensity_calibrate

GEOM = dict(image_size=512, pixel_spacing=180/512, n_angles=128,
            n_det=1024, det_spacing=360/1024, sod=500.0, sdd=1000.0)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data = Path("/cluster/maier/Agent4CT/data/dl_sparse_view/staged")
    with h5py.File(data / "val_truth.h5", "r") as f: truth = f["image"][:4]
    with h5py.File(data / "val_sinograms.h5", "r") as f: sino = f["sino"][:4]
    with h5py.File(data / "val_fbp128.h5", "r") as f: ref = f["image"][:4]

    s = torch.from_numpy(sino).float().to(device).unsqueeze(1)
    geom = FanBeamGeometry(**GEOM); proj = PyronnFanBeamProjector(geom).to(device)
    t_truth = torch.from_numpy(truth).float().to(device).unsqueeze(1)

    # Sino-axis start-angle shifts (in view indices). 128 views over 360° → 1 view = 2.8125°
    shifts = [0, 32, -32]
    shift_labels = ["start-shift = 0", "start-shift = +32 (≡ +90°)", "start-shift = -32 (≡ -90° CW)"]

    fig, axes = plt.subplots(4, 5, figsize=(20, 14))
    for r in range(4):
        gt_t = t_truth[r:r+1]
        # truth
        axes[r, 0].imshow(truth[r], cmap="gray",
                          vmin=float(np.percentile(truth[r], 1)),
                          vmax=float(np.percentile(truth[r], 99.5)))
        axes[r, 0].set_title(f"truth #{r}\n[{truth[r].min():.3f}, {truth[r].max():.3f}]",
                              fontsize=10); axes[r, 0].axis("off")
        # FBP128 reference
        axes[r, 1].imshow(ref[r], cmap="gray",
                          vmin=float(np.percentile(ref[r], 1)),
                          vmax=float(np.percentile(ref[r], 99.5)))
        axes[r, 1].set_title(f"Sidky FBP128 #{r}\n[{ref[r].min():.3f}, {ref[r].max():.3f}]",
                              fontsize=10); axes[r, 1].axis("off")
        # 3 shift variants
        for col, (shift, label) in enumerate(zip(shifts, shift_labels), start=2):
            s_shifted = torch.roll(s[r:r+1], shifts=shift, dims=-2)
            with torch.no_grad():
                rec = proj.fbp(s_shifted).clamp_min(0.0)   # suppress negatives
                # Intensity-calibrate against truth so the scale matches
                rec_cal = intensity_calibrate(rec, gt_t, display_max=0.5)
            rec_np = rec_cal[0, 0].cpu().numpy()
            # Structural metrics against FBP128 (also intensity-calibrated for fairness)
            ref_t = torch.from_numpy(ref[r]).float().to(device).view(1, 1, 512, 512)
            with torch.no_grad():
                ref_cal = intensity_calibrate(ref_t, gt_t, display_max=0.5)
            ssim_val = float(ssim_t(rec_cal, ref_cal, data_range=0.5).cpu())
            corr = float(np.corrcoef(rec_np.ravel(), ref[r].ravel())[0, 1])
            axes[r, col].imshow(rec_np, cmap="gray",
                                vmin=float(np.percentile(rec_np, 1)),
                                vmax=float(np.percentile(rec_np, 99.5)))
            axes[r, col].set_title(
                f"OUR  {label}\n"
                f"cal SSIM vs FBP128={ssim_val:.3f}  raw corr={corr:.3f}\n"
                f"[{rec_np.min():.3f}, {rec_np.max():.3f}]",
                fontsize=9)
            axes[r, col].axis("off")

    plt.tight_layout()
    out = Path("/cluster/maier/Agent4CT/results/breast_debug/start_angle.png")
    plt.savefig(out, dpi=80, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {out}")


if __name__ == "__main__":
    main()
