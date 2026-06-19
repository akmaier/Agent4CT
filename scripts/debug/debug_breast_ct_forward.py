"""Forward-project truth through OUR geometry and compare to staged sino.
This is the most direct test of geometry correctness:
  if FP(truth) ≈ staged_sino, the geometry matches Sidky's simulator;
  if FP(truth) differs, the ratio reveals scale & the structural pattern
  reveals rotation/flip.
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

GEOM = dict(image_size=512, pixel_spacing=180/512, n_angles=128, n_det=1024,
            det_spacing=360/1024, sod=500.0, sdd=1000.0)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data = Path("/cluster/maier/Agent4CT/data/dl_sparse_view/staged")
    with h5py.File(data / "val_truth.h5", "r") as f: truth = f["image"][:1]
    with h5py.File(data / "val_sinograms.h5", "r") as f: sino_ref = f["sino"][:1]

    geom = FanBeamGeometry(**GEOM); proj = PyronnFanBeamProjector(geom).to(device)
    t = torch.from_numpy(truth).float().to(device).unsqueeze(1)   # (1,1,512,512)
    with torch.no_grad():
        sino_ours = proj.forward_project(t)[:, 0].cpu().numpy()    # (1,128,1024)

    sr, so = sino_ref[0], sino_ours[0]
    print(f"staged sino  shape={sr.shape} range=[{sr.min():.4f},{sr.max():.4f}] mean={sr.mean():.4f}")
    print(f"our    FP    shape={so.shape} range=[{so.min():.4f},{so.max():.4f}] mean={so.mean():.4f}")
    ratio_max = sr.max() / max(so.max(), 1e-12)
    ratio_mean = sr.mean() / max(so.mean(), 1e-12)
    corr = float(np.corrcoef(sr.ravel(), so.ravel())[0, 1])
    print(f"max ratio (staged/ours) = {ratio_max:.4f}")
    print(f"mean ratio              = {ratio_mean:.4f}")
    print(f"pearson corr            = {corr:.4f}")

    # Also try transpose / flip variants on OUR forward-projection to find the
    # angle-convention alignment.
    print("\n--- sino-axis sweep ---")
    variants = {
        "baseline (ours)":      so,
        "flip angle":           so[::-1],
        "flip det":             so[:, ::-1],
        "flip both":            so[::-1, ::-1],
        "transpose":            so.T,
    }
    best = (None, -np.inf)
    for label, s in variants.items():
        # interp if shape differs after transpose
        if s.shape != sr.shape:
            print(f"  {label:20} SHAPE MISMATCH {s.shape}")
            continue
        c = float(np.corrcoef(sr.ravel(), s.ravel())[0, 1])
        print(f"  {label:20} corr={c:.4f}  range=[{s.min():.3f},{s.max():.3f}]")
        if c > best[1]: best = (label, c)
    print(f"\nbest sino variant: {best[0]} corr={best[1]:.4f}")

    # Visualize: staged sino vs our FP — show side by side and abs(diff)
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    vlo, vhi = float(np.percentile(sr, 1)), float(np.percentile(sr, 99.5))
    axes[0,0].imshow(sr, cmap="gray", aspect="auto", vmin=vlo, vmax=vhi)
    axes[0,0].set_title(f"staged sino #0 (Sidky)\nrange=[{sr.min():.3f},{sr.max():.3f}]"); axes[0,0].axis("off")
    axes[0,1].imshow(so, cmap="gray", aspect="auto",
                      vmin=float(np.percentile(so, 1)),
                      vmax=float(np.percentile(so, 99.5)))
    axes[0,1].set_title(f"our FP(truth)\nrange=[{so.min():.3f},{so.max():.3f}]\nratio={ratio_max:.2f}×"); axes[0,1].axis("off")
    axes[0,2].imshow(sr - so * ratio_mean, cmap="bwr",
                      vmin=-float(np.percentile(np.abs(sr - so*ratio_mean), 99)),
                      vmax=float(np.percentile(np.abs(sr - so*ratio_mean), 99)))
    axes[0,2].set_title(f"staged - (ours × {ratio_mean:.3f})"); axes[0,2].axis("off")
    # second row: vertical line profiles at a fixed angle
    a = 64    # middle angle
    axes[1,0].plot(sr[a], label="staged"); axes[1,0].plot(so[a]*ratio_mean, label="ours·k")
    axes[1,0].set_title(f"line profile at angle {a}"); axes[1,0].legend()
    # column profile
    d = 512
    axes[1,1].plot(sr[:, d], label="staged"); axes[1,1].plot(so[:, d]*ratio_mean, label="ours·k")
    axes[1,1].set_title(f"angle profile at detector {d}"); axes[1,1].legend()
    axes[1,2].imshow(sr - so * ratio_mean, cmap="bwr", aspect="auto",
                      vmin=-float(np.percentile(np.abs(sr - so*ratio_mean), 99)),
                      vmax=float(np.percentile(np.abs(sr - so*ratio_mean), 99)))
    axes[1,2].set_title("same diff (stretched)"); axes[1,2].axis("off")
    plt.tight_layout()
    out = Path("/cluster/maier/Agent4CT/results/breast_debug/forward_check.png")
    plt.savefig(out, dpi=80, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {out}")


if __name__ == "__main__":
    main()
