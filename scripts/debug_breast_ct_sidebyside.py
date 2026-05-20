"""Single canonical side-by-side figure for the breast-CT FBP geometry
debug. 4 cases × 5 columns:
    truth | FBP128 (ref) | OUR FBP @current | OUR FBP @det_spacing=0.6429 | diff(@0.6429 - FBP128)

Saves to /cluster/maier/Agent4CT/results/breast_debug/sidebyside.png so we
can scp it out and inspect visually.
"""
from __future__ import annotations
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import numpy as np
import torch
import hdf5plugin   # noqa
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from ddssl_ldct.geometry import FanBeamGeometry
from ddssl_ldct.pyronn_projector import PyronnFanBeamProjector


def fbp(geom_kw, sino_t, device):
    g = FanBeamGeometry(**geom_kw)
    p = PyronnFanBeamProjector(g).to(device)
    with torch.no_grad():
        return p.fbp(sino_t).clamp_min(0.0)[:, 0].cpu().numpy()


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data = Path("/cluster/maier/Agent4CT/data/dl_sparse_view/staged")
    with h5py.File(data / "val_truth.h5", "r") as f: truth = f["image"][:4]
    with h5py.File(data / "val_sinograms.h5", "r") as f: sino = f["sino"][:4]
    with h5py.File(data / "val_fbp128.h5", "r") as f: fbp_ref = f["image"][:4]

    s = torch.from_numpy(sino).float().to(device).unsqueeze(1)

    fbp_current = fbp(dict(image_size=512, pixel_spacing=0.7, n_angles=128,
                            n_det=1024, det_spacing=1.2858,
                            sod=595.0, sdd=1085.6), s, device)
    fbp_fixed   = fbp(dict(image_size=512, pixel_spacing=0.7, n_angles=128,
                            n_det=1024, det_spacing=0.6429,
                            sod=595.0, sdd=1085.6), s, device)

    out_dir = Path("/cluster/maier/Agent4CT/results/breast_debug")
    out_dir.mkdir(parents=True, exist_ok=True)

    def auto_v(img, p_lo=1, p_hi=99.5):
        return float(np.percentile(img, p_lo)), float(np.percentile(img, p_hi))

    fig, axes = plt.subplots(4, 5, figsize=(18, 13))
    for r in range(4):
        # Auto-scale each panel to its own percentile range so structure
        # is visible regardless of absolute intensity.
        for c, (img, label, cmap) in enumerate([
            (truth[r],              f"truth #{r}\n[{truth[r].min():.3f},{truth[r].max():.3f}]",      "gray"),
            (fbp_ref[r],            f"FBP128 (Sidky ref)\n[{fbp_ref[r].min():.3f},{fbp_ref[r].max():.3f}]",    "gray"),
            (fbp_current[r],        f"OUR @det=1.286\n[{fbp_current[r].min():.4f},{fbp_current[r].max():.4f}]","gray"),
            (fbp_fixed[r],          f"OUR @det=0.643\n[{fbp_fixed[r].min():.4f},{fbp_fixed[r].max():.4f}]",    "gray"),
            (fbp_fixed[r] - fbp_ref[r], "(0.643) - FBP128", "bwr"),
        ]):
            if cmap == "bwr":
                lim = float(np.percentile(np.abs(img), 99))
                vlo, vhi = -lim, lim
            else:
                vlo, vhi = auto_v(img)
            axes[r, c].imshow(img, cmap=cmap, vmin=vlo, vmax=vhi)
            axes[r, c].set_title(label, fontsize=9)
            axes[r, c].axis("off")
    plt.tight_layout()
    plt.savefig(out_dir / "sidebyside.png", dpi=80, bbox_inches="tight")
    plt.close(fig)
    print(f"[done] saved {out_dir/'sidebyside.png'}")

    # Also print key scalars
    print(f"\nFOV check:")
    print(f"  truth phantom occupies ~{(truth[0] > 0.05).sum()} pixels of {truth[0].size}")
    print(f"  FBP128 phantom occupies ~{(fbp_ref[0] > 0.05).sum()} pixels of {fbp_ref[0].size}")
    print(f"  our @1.286 phantom occupies ~{(fbp_current[0] > 0.05).sum()} pixels of {fbp_current[0].size}")
    print(f"  our @0.643 phantom occupies ~{(fbp_fixed[0]   > 0.05).sum()} pixels of {fbp_fixed[0].size}")


if __name__ == "__main__":
    main()
