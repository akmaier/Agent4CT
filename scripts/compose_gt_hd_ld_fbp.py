"""Compose a single GT | HD FBP | LD FBP comparison figure from the two
validator .npz outputs (one per dose). The validator itself runs the
canonical Mayo FBP pipeline (Powell geometry, MAYO_LDCT_DET_OFFSET,
Siemens flip, angle_start_corrected, 5-mm slab averaging, fliplr/flipud,
intensity-calibrated); this script just reads its arrays and arranges
them on one figure.
"""
import argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--hd-png", required=True, type=Path,
                   help="Path to the HD validator PNG (its sibling .npz "
                        "is loaded for the arrays).")
    p.add_argument("--ld-png", required=True, type=Path)
    p.add_argument("--out-png", required=True, type=Path)
    args = p.parse_args()

    hd_npz_path = args.hd_png.with_suffix(".npz")
    ld_npz_path = args.ld_png.with_suffix(".npz")
    if not hd_npz_path.exists():
        raise FileNotFoundError(f"missing {hd_npz_path} — did the HD validator run write it?")
    if not ld_npz_path.exists():
        raise FileNotFoundError(f"missing {ld_npz_path}")

    hd = np.load(hd_npz_path)
    ld = np.load(ld_npz_path)
    truth = hd["truth"]      # truth is dose-independent; both files store the same array
    fbp_hd = hd["fbp_cal"]   # intensity-calibrated against truth
    fbp_ld = ld["fbp_cal"]
    dr = float(hd["display_max"])

    # Top row: the three reconstructions side-by-side.
    # Bottom row: HD-truth diff, LD-truth diff, LD-HD diff.
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    axes[0, 0].imshow(truth, cmap="gray", vmin=0, vmax=dr)
    axes[0, 0].set_title(f"Ground truth (Mayo B30f)\n"
                          f"mean={truth.mean():.4f} μ⁻¹  std={truth.std():.4f}")
    axes[0, 1].imshow(fbp_hd, cmap="gray", vmin=0, vmax=dr)
    axes[0, 1].set_title(f"HD FBP (full-dose, calibrated)\n"
                          f"SSIM={float(hd['ssim_cal']):.4f}  PSNR={float(hd['psnr_cal']):.2f} dB\n"
                          f"RMSE={float(hd['rmse_cal']):.5f}  mean={fbp_hd.mean():.4f}  std={fbp_hd.std():.4f}")
    axes[0, 2].imshow(fbp_ld, cmap="gray", vmin=0, vmax=dr)
    axes[0, 2].set_title(f"LD FBP (low-dose, calibrated)\n"
                          f"SSIM={float(ld['ssim_cal']):.4f}  PSNR={float(ld['psnr_cal']):.2f} dB\n"
                          f"RMSE={float(ld['rmse_cal']):.5f}  mean={fbp_ld.mean():.4f}  std={fbp_ld.std():.4f}")

    diff_hd = fbp_hd - truth
    diff_ld = fbp_ld - truth
    diff_lh = fbp_ld - fbp_hd
    dlim = 0.02

    axes[1, 0].imshow(diff_hd, cmap="seismic", vmin=-dlim, vmax=dlim)
    axes[1, 0].set_title(f"HD − GT\n"
                          f"mean={diff_hd.mean():+.5f}  std={diff_hd.std():.5f}  "
                          f"rmse={np.sqrt((diff_hd**2).mean()):.5f}")
    axes[1, 1].imshow(diff_ld, cmap="seismic", vmin=-dlim, vmax=dlim)
    axes[1, 1].set_title(f"LD − GT\n"
                          f"mean={diff_ld.mean():+.5f}  std={diff_ld.std():.5f}  "
                          f"rmse={np.sqrt((diff_ld**2).mean()):.5f}")
    axes[1, 2].imshow(diff_lh, cmap="seismic", vmin=-dlim, vmax=dlim)
    axes[1, 2].set_title(f"LD − HD\n"
                          f"mean={diff_lh.mean():+.5f}  std={diff_lh.std():.5f}  "
                          f"rmse={np.sqrt((diff_lh**2).mean()):.5f}")

    for ax in axes.ravel():
        ax.set_xticks([]); ax.set_yticks([])

    fig.suptitle(
        f"L014 GT vs HD vs LD FBP — canonical Mayo pipeline\n"
        f"(Powell-fit geometry, MAYO_LDCT_DET_OFFSET=−0.0397 mm, Siemens flip, "
        f"5-mm slab average, fliplr+flipud, intensity-calibrated against GT, no FOV mask)\n"
        f"Display range μ ∈ [0, {dr:g}]; diff range μ ∈ [−{dlim:g}, +{dlim:g}]",
        fontsize=11,
    )
    fig.tight_layout()
    args.out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out_png, dpi=140, bbox_inches="tight")
    print(f"[compose] wrote {args.out_png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
