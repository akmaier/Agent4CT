"""Side-by-side numerical comparison: Sidky reference vs Siddon vs Pyronn,
all three FBPs of the released val_sinograms with the SAME by-the-book filter
configuration (Kak-Slaney ram-lak, zero-padded to 2N, H[0] halved).

Pyronn needs sino_angle_shift=+32 to align with the Sidky world (per earlier
geometry sweep); Siddon needs shift=0 (per its own natural convention).

Prints metrics tables and saves one PNG (4 cases × 6 cols: truth | sidky_fbp128
| Siddon FBP | Siddon − truth | Pyronn FBP | Pyronn − truth).
"""
from __future__ import annotations
import math
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
from ddssl_ldct.siddon_projector import SiddonFanBeamProjector
from ddssl_ldct.pyronn_projector import PyronnFanBeamProjector
from ddssl_ldct.metrics import psnr, ssim, intensity_calibrate

GEOM = dict(image_size=512, pixel_spacing=180/512, n_angles=128,
            n_det=1024, det_spacing=0.35754, sod=500.0, sdd=1000.0)  # det_spacing per job-761480 sweep
DISPLAY_MAX = 0.5
N_CASES = 4
SINO_SHIFT_PYRONN = 32                                # pyronn convention offset
OUT_DIR = Path("/cluster/maier/Agent4CT/results/breast_debug")


def fov_mask(N: int, radius_pix: float, device, dtype) -> torch.Tensor:
    """Circular FOV mask of radius ``radius_pix`` (in pixels), centred."""
    coords = torch.arange(N, device=device, dtype=dtype) - (N - 1) / 2.0
    yy, xx = torch.meshgrid(coords, coords, indexing="ij")
    r2 = xx * xx + yy * yy
    return (r2 <= radius_pix * radius_pix).to(dtype)


def cal_metrics(pred, truth, dmax=DISPLAY_MAX):
    """Original v2 cal_metrics: double-clamp via intensity_calibrate
    (pre-clamp + post-clamp inside the function), per user direction."""
    pc = intensity_calibrate(pred.clamp_min(0.0), truth, display_max=dmax)
    return (
        float(ssim(pc, truth, data_range=dmax).cpu()),
        float(psnr(pc, truth, data_range=dmax).cpu()),
        float(((pc - truth) ** 2).mean().sqrt().cpu()),
        pc,
    )


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    data = Path("/cluster/maier/Agent4CT/data/dl_sparse_view/staged")
    with h5py.File(data / "val_truth.h5", "r") as f: truth_np = f["image"][:N_CASES]
    with h5py.File(data / "val_sinograms.h5", "r") as f: sino_np = f["sino"][:N_CASES]
    with h5py.File(data / "val_fbp128.h5", "r") as f: fbp_np = f["image"][:N_CASES]

    truth = torch.from_numpy(truth_np).float().to(device).unsqueeze(1)
    sino = torch.from_numpy(sino_np).float().to(device).unsqueeze(1)
    fbp_ref = torch.from_numpy(fbp_np).float().to(device).unsqueeze(1)

    geom = FanBeamGeometry(**GEOM)
    siddon = SiddonFanBeamProjector(geom, ray_batch=8192, length_unit_scale=0.1).to(device)
    pyronn = PyronnFanBeamProjector(geom).to(device)

    # Siddon: shift=0
    with torch.no_grad():
        fbp_siddon = siddon.fbp(sino)
    # Pyronn: shift=+32 for sino axis alignment, ramlak filter
    with torch.no_grad():
        sino_pyronn = torch.roll(sino, shifts=SINO_SHIFT_PYRONN, dims=-2)
        fbp_pyronn = pyronn.fbp(sino_pyronn, filter_name="ramlak")

    # Diagnostic: corner pixel values of Sidky's val_fbp128 — if Sidky is
    # masking outside the FOV, corner values should be exactly 0.
    sk0 = fbp_np[0]
    print("\nSidky FBP128[0] corner / edge pixel values (raw):")
    for (ri, ci) in [(0, 0), (0, 255), (0, 511), (255, 0), (255, 511), (511, 0), (511, 511)]:
        print(f"  [{ri:>3},{ci:>3}] = {sk0[ri, ci]:+.6f}")
    print(f"  [256,256] (center) = {sk0[256, 256]:+.6f}")
    # Count pixels exactly at 0 (Sidky's mask sentinel?)
    n_zero = int((sk0 == 0).sum())
    n_total = sk0.size
    print(f"  pixels exactly == 0 : {n_zero} / {n_total}  ({100*n_zero/n_total:.2f} %)")

    # Apply circular FOV mask (radius = N/2 = 256 px = 9 cm) to ALL recons.
    N_img = truth.shape[-1]
    mask = fov_mask(N_img, radius_pix=N_img / 2.0, device=device, dtype=truth.dtype)
    mask_4d = mask[None, None, :, :]                 # (1,1,H,W) for broadcasting
    truth   = truth   * mask_4d
    fbp_ref = fbp_ref * mask_4d
    fbp_siddon = fbp_siddon * mask_4d
    fbp_pyronn = fbp_pyronn * mask_4d
    fbp_np    = fbp_np    * mask.cpu().numpy()
    truth_np  = truth_np  * mask.cpu().numpy()
    print(f"\napplied circular FOV mask, radius = {N_img/2:.1f} px = {N_img/2 * 180/N_img:.2f} mm")

    # --- Numerical tables ----------------------------------------------------
    print(f"\n{'='*80}\nSidky FBP128 vs truth   (gold standard, cal'd)\n{'='*80}")
    print(f"{'case':>4} {'SSIM':>8} {'PSNR':>8} {'RMSE':>8}  {'rawMin':>10} {'rawMax':>10} {'rawMean':>10}")
    sidky_ssim, sidky_psnr, sidky_rmse = [], [], []
    for i in range(N_CASES):
        ss, ps, rm, _ = cal_metrics(fbp_ref[i:i+1], truth[i:i+1])
        sidky_ssim.append(ss); sidky_psnr.append(ps); sidky_rmse.append(rm)
        ref = fbp_np[i]
        print(f"{i:>4} {ss:8.4f} {ps:8.2f} {rm:8.4f}  {ref.min():10.4f} {ref.max():10.4f} {ref.mean():10.4f}")
    print(f"{'mean':>4} {np.mean(sidky_ssim):8.4f} {np.mean(sidky_psnr):8.2f} {np.mean(sidky_rmse):8.4f}")

    for label, recon, sino_used in (
        ("Siddon FBP", fbp_siddon, "shift=0"),
        ("Pyronn FBP (ramlak, zero-pad 2N, H[0]/2)", fbp_pyronn, f"shift=+{SINO_SHIFT_PYRONN}"),
    ):
        print(f"\n{'='*80}\n{label}  ({sino_used})  vs truth   (cal'd)\n{'='*80}")
        print(f"{'case':>4} {'SSIM':>8} {'PSNR':>8} {'RMSE':>8}  {'rawMin':>10} {'rawMax':>10} {'rawMean':>10}  {'mean_ratio_vs_sky':>18}")
        ss_l, ps_l, rm_l = [], [], []
        for i in range(N_CASES):
            ss, ps, rm, _ = cal_metrics(recon[i:i+1], truth[i:i+1])
            ss_l.append(ss); ps_l.append(ps); rm_l.append(rm)
            raw = recon[i, 0].cpu().numpy()
            ref_mean = fbp_np[i].mean()
            print(f"{i:>4} {ss:8.4f} {ps:8.2f} {rm:8.4f}  {raw.min():10.4f} {raw.max():10.4f} {raw.mean():10.4f}  {raw.mean()/ref_mean:18.4f}")
        print(f"{'mean':>4} {np.mean(ss_l):8.4f} {np.mean(ps_l):8.2f} {np.mean(rm_l):8.4f}")

        # Also vs sidky_fbp128
        print(f"\n--- {label}  vs sidky_fbp128 (cal'd) ---")
        print(f"{'case':>4} {'SSIM':>8} {'PSNR':>8} {'RMSE':>8}")
        for i in range(N_CASES):
            ss, ps, rm, _ = cal_metrics(recon[i:i+1], fbp_ref[i:i+1])
            print(f"{i:>4} {ss:8.4f} {ps:8.2f} {rm:8.4f}")

    # --- Image (4×6) ---------------------------------------------------------
    fig, axes = plt.subplots(N_CASES, 6, figsize=(26, 4.6 * N_CASES))
    diff_lim = DISPLAY_MAX / 4.0
    for r in range(N_CASES):
        t = truth_np[r]; sky = fbp_np[r]
        _, _, _, sk_cal = cal_metrics(fbp_ref[r:r+1], truth[r:r+1])
        _, _, _, sd_cal = cal_metrics(fbp_siddon[r:r+1], truth[r:r+1])
        _, _, _, py_cal = cal_metrics(fbp_pyronn[r:r+1], truth[r:r+1])
        sd_np = sd_cal[0, 0].cpu().numpy()
        py_np = py_cal[0, 0].cpu().numpy()

        axes[r, 0].imshow(t, cmap="gray", vmin=0, vmax=DISPLAY_MAX); axes[r, 0].set_title(f"truth #{r}", fontsize=10); axes[r, 0].axis("off")
        axes[r, 1].imshow(sky, cmap="gray", vmin=0, vmax=DISPLAY_MAX); axes[r, 1].set_title(f"sidky FBP128", fontsize=10); axes[r, 1].axis("off")
        axes[r, 2].imshow(sd_np, cmap="gray", vmin=0, vmax=DISPLAY_MAX); axes[r, 2].set_title(f"OUR Siddon FBP (cal)", fontsize=10); axes[r, 2].axis("off")
        axes[r, 3].imshow(sd_np - t, cmap="bwr", vmin=-diff_lim, vmax=diff_lim); axes[r, 3].set_title(f"Siddon − truth", fontsize=10); axes[r, 3].axis("off")
        axes[r, 4].imshow(py_np, cmap="gray", vmin=0, vmax=DISPLAY_MAX); axes[r, 4].set_title(f"OUR Pyronn FBP (cal)", fontsize=10); axes[r, 4].axis("off")
        axes[r, 5].imshow(py_np - t, cmap="bwr", vmin=-diff_lim, vmax=diff_lim); axes[r, 5].set_title(f"Pyronn − truth", fontsize=10); axes[r, 5].axis("off")

    plt.suptitle("All FBPs use Kak-Slaney ram-lak, zero-padded 2N, H[0] halved. "
                 f"Display [0, {DISPLAY_MAX}]. Diffs at ±{diff_lim:.3f}.", fontsize=11, y=1.001)
    plt.tight_layout()
    out = OUT_DIR / "compare_all.png"
    plt.savefig(out, dpi=80, bbox_inches="tight")
    plt.close(fig)
    print(f"\n[saved] {out}")


if __name__ == "__main__":
    main()
