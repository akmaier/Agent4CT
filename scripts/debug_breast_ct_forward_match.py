"""Smoking-gun forward-operator test.

Hypothesis: our FBP is correct (the disc-phantom test proved that). The gap to
Sidky's FBP128 must come from a difference in the *forward operator* that built
val_sinograms.h5 vs the one pyronn uses.

Test:
  A. Forward-project val_truth through our projector → sino_ours.
  B. FBP(sino_ours)  — should perfectly round-trip val_truth.
  C. FBP(val_sinograms.h5)  — the current breast recon.
  D. Compare sino_ours to val_sinograms.h5 directly (after intensity calibration).

If (B) matches truth but (C) cups, the cupping is forward-operator drift between
Sidky's pipeline and pyronn's, not a filter / pre-weight issue.

Outputs: /cluster/maier/Agent4CT/results/breast_debug/forward_match.png
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
    with h5py.File(data / "val_sinograms.h5", "r") as f: sino_sidky = f["sino"][:n]

    t = torch.from_numpy(truth).float().to(device).unsqueeze(1)         # (n,1,H,W)
    s_sidky = torch.from_numpy(sino_sidky).float().to(device).unsqueeze(1)  # (n,1,A,D)

    geom = FanBeamGeometry(**GEOM); proj = PyronnFanBeamProjector(geom).to(device)

    # ── A. Forward project truth through OUR projector ────────────────────────
    with torch.no_grad():
        s_ours = proj.forward_project(t)                                # (n,1,A,D)

    # Apply same SINO_SHIFT to Sidky sinos for the recon (geometry alignment)
    s_sidky_shifted = torch.roll(s_sidky, shifts=SINO_SHIFT, dims=-2)

    # ── B. FBP(sino_ours): self-consistency round-trip ────────────────────────
    with torch.no_grad():
        rec_ours_from_ours = proj.fbp(s_ours, filter_name="hann")
    # ── C. FBP(sidky_sino): the current breast recon ──────────────────────────
    with torch.no_grad():
        rec_ours_from_sidky = proj.fbp(s_sidky_shifted, filter_name="hann")

    # Per-case figure: 4 rows × 6 cols
    fig, axes = plt.subplots(n, 6, figsize=(24, 14))
    print(f"\n{'case':<4} {'pipeline':<28} {'SSIM':>7} {'PSNR':>7} {'RMSE':>8}")
    for r in range(n):
        gt = t[r:r+1]; gt_np = truth[r]

        ss_b, ps_b, rm_b, rec_b_cal = metrics_v(rec_ours_from_ours[r:r+1], gt)
        ss_c, ps_c, rm_c, rec_c_cal = metrics_v(rec_ours_from_sidky[r:r+1], gt)
        print(f"{r:<4} {'FBP(OUR_forward(truth))':<28} {ss_b:7.4f} {ps_b:7.2f} {rm_b:8.4f}")
        print(f"{r:<4} {'FBP(sidky_sino)':<28}         {ss_c:7.4f} {ps_c:7.2f} {rm_c:8.4f}")

        # Col 0: truth
        axes[r, 0].imshow(gt_np, cmap="gray", vmin=0, vmax=DISPLAY_MAX)
        axes[r, 0].set_title(f"truth #{r}\nrange=[{gt_np.min():.3f}, {gt_np.max():.3f}]",
                              fontsize=9); axes[r, 0].axis("off")
        # Col 1: FBP(OUR_forward(truth))
        im_b = rec_b_cal[0, 0].cpu().numpy()
        axes[r, 1].imshow(im_b, cmap="gray", vmin=0, vmax=DISPLAY_MAX)
        axes[r, 1].set_title(f"FBP(OUR fwd(truth))\nSSIM={ss_b:.3f} PSNR={ps_b:.1f}",
                              fontsize=9); axes[r, 1].axis("off")
        # Col 2: diff (rec_ours_from_ours - truth)
        diff_b = (rec_b_cal - gt)[0, 0].cpu().numpy()
        lim = DISPLAY_MAX / 4
        axes[r, 2].imshow(diff_b, cmap="bwr", vmin=-lim, vmax=lim)
        axes[r, 2].set_title(f"diff (B - truth)\n|err|max={float(np.abs(diff_b).max()):.3f}",
                              fontsize=9); axes[r, 2].axis("off")
        # Col 3: FBP(sidky_sino)
        im_c = rec_c_cal[0, 0].cpu().numpy()
        axes[r, 3].imshow(im_c, cmap="gray", vmin=0, vmax=DISPLAY_MAX)
        axes[r, 3].set_title(f"FBP(sidky sino)\nSSIM={ss_c:.3f} PSNR={ps_c:.1f}",
                              fontsize=9); axes[r, 3].axis("off")
        # Col 4: diff (rec_ours_from_sidky - truth)
        diff_c = (rec_c_cal - gt)[0, 0].cpu().numpy()
        axes[r, 4].imshow(diff_c, cmap="bwr", vmin=-lim, vmax=lim)
        axes[r, 4].set_title(f"diff (C - truth)\n|err|max={float(np.abs(diff_c).max()):.3f}",
                              fontsize=9); axes[r, 4].axis("off")
        # Col 5: sino diff (OUR_forward(truth) - sidky_sino), zero center
        # Scale-match: divide both by max projection thickness so equal-mass rays look equal
        s_o = s_ours[r, 0].cpu().numpy()      # (A, D)  — pyronn convention
        s_k = s_sidky[r, 0].cpu().numpy()     # (A, D)  — sidky convention
        # Quick linear scale-match: solve for k minimizing ||k·s_o - s_k||
        k = float((s_o * s_k).sum() / (s_o * s_o).sum())
        s_diff = (k * s_o - s_k)
        slim = float(np.percentile(np.abs(s_diff), 99))
        axes[r, 5].imshow(s_diff, cmap="bwr", vmin=-slim, vmax=slim,
                          aspect="auto", origin="lower")
        axes[r, 5].set_title(
            f"sino diff: {k:.3g}·OUR_fwd(truth) - sidky_sino\n"
            f"||diff||/||sidky||={float(np.linalg.norm(s_diff)/np.linalg.norm(s_k)):.3f}",
            fontsize=9)

    plt.suptitle(
        f"Forward-operator match. Col 1: FBP of OUR forward(truth) — bound on what "
        f"our FBP can deliver. Col 3: FBP of released sidky sinos — what we actually "
        f"reconstruct. Cols 2/4: diffs at ±{DISPLAY_MAX/4:.3f}.",
        fontsize=11, y=1.001)
    plt.tight_layout()
    out = OUT_DIR / "forward_match.png"
    plt.savefig(out, dpi=80, bbox_inches="tight")
    plt.close(fig)
    print(f"\n[saved] {out}")


if __name__ == "__main__":
    main()
