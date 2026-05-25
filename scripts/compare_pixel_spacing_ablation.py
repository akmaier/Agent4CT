"""Ablation: same L014 sino slab, same truth slice, but vary FanBeamGeometry
pixel_spacing between 0.7 mm (legacy Wagner default) and 0.703125 mm (Mayo
truth DICOM's actual PixelSpacing). Produces a 2-row × 4-column figure:

  Row 0 (px=0.7):       truth | FBP_cal | diff | overlay
  Row 1 (px=0.703125):  truth | FBP_cal | diff | overlay

Each panel includes SSIM/PSNR/RMSE in its title.

All else is identical: same 5-mm slab anchored at +3.5 mm offset, same FBP
parameters (Hann filter, full scan), no FOV mask, identical truth-z mapping.

Usage (on cluster):
    python scripts/compare_pixel_spacing_ablation.py
"""
from __future__ import annotations

import math
import sys
import json
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pydicom
import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from ddssl_ldct.geometry import FanBeamGeometry
from ddssl_ldct.pyronn_projector import PyronnFanBeamProjector
from ddssl_ldct.metrics import evaluate_calibrated


def _load_truth_at_z(raw_dir: Path, target_z: float):
    """Return (truth_mu, dicom_meta) for the slice nearest `target_z`."""
    SOP_CT = "1.2.840.10008.5.1.4.1.1.2"
    truth_files = []
    for series_dir in sorted(raw_dir.iterdir()):
        sample = next(series_dir.iterdir(), None)
        if sample is None:
            continue
        try:
            head = pydicom.dcmread(str(sample), stop_before_pixels=True)
        except Exception:
            continue
        if getattr(head, "SOPClassUID", "") != SOP_CT:
            continue
        desc = getattr(head, "SeriesDescription", "").lower()
        if "full" not in desc or "image" not in desc:
            continue
        for fp in series_dir.iterdir():
            try:
                m = pydicom.dcmread(str(fp), stop_before_pixels=True)
                z = float(m.ImagePositionPatient[2])
                truth_files.append((z, fp))
            except Exception:
                continue
        break
    truth_files.sort()
    if not truth_files:
        raise RuntimeError("no truth slices found")
    # Linear-interpolate between the two slices bracketing target_z.
    zs = np.array([t[0] for t in truth_files])
    # idx_above = first slice with z >= target
    idx_above = int(np.searchsorted(zs, target_z, side="left"))
    if idx_above <= 0:
        lo = hi = 0
    elif idx_above >= len(zs):
        lo = hi = len(zs) - 1
    else:
        lo, hi = idx_above - 1, idx_above

    def _mu(fp):
        ds = pydicom.dcmread(str(fp))
        hu = (ds.pixel_array.astype(np.float32) * float(ds.RescaleSlope)
              + float(ds.RescaleIntercept))
        return 0.02 * (1.0 + hu / 1000.0), ds

    if lo == hi:
        mu, ds = _mu(truth_files[lo][1])
        return mu, ds, (truth_files[lo][0], truth_files[lo][0], 1.0)

    z_lo, z_hi = float(truth_files[lo][0]), float(truth_files[hi][0])
    w_lo = (z_hi - target_z) / (z_hi - z_lo)
    w_lo = float(max(0.0, min(1.0, w_lo)))
    mu_lo, ds_lo = _mu(truth_files[lo][1])
    mu_hi, _ = _mu(truth_files[hi][1])
    mu = (w_lo * mu_lo + (1.0 - w_lo) * mu_hi).astype(np.float32)
    return mu, ds_lo, (z_lo, z_hi, w_lo)


def _fbp(sino_slab: list[np.ndarray], geom: FanBeamGeometry, device: str):
    """FBP each slab member individually, average, then flipud+fliplr."""
    proj = PyronnFanBeamProjector(geom).to(device)
    recons = []
    for s in sino_slab:
        s_t = torch.from_numpy(s).to(device).float()[None, None]
        out = proj.fbp(s_t).detach()[0, 0].cpu().numpy()
        recons.append(np.fliplr(np.flipud(out)))
    return np.mean(np.stack(recons, axis=0), axis=0)


def main() -> int:
    import os
    root = Path(os.environ.get("AGENT4CT_DATA",
                                "/cluster/maier/Agent4CT/data")) / "mayo_ldct"
    sino_dir = root / "staged_helix2fan"
    geom_json = json.loads(
        (sino_dir / "L014_sino_fulldose_geometry.json").read_text()
    )
    z_grid = np.load(sino_dir / "L014_sino_fulldose_z_grid.npy")

    nu, rotview, nz = int(geom_json["nu"]), int(geom_json["rotview"]), int(geom_json["nz_rebinned"])
    du = float(geom_json["du"])

    # Same +3.5 mm anchor + 5-mm slab as the validator default.
    nz_center = nz // 2 + 4
    z_center = float(z_grid[nz_center])
    SLAB_HALF = 2
    slab_lo, slab_hi = nz_center - SLAB_HALF, nz_center + SLAB_HALF + 1
    with h5py.File(sino_dir / "L014_sino_fulldose.h5", "r") as f:
        sino_slab = [
            np.ascontiguousarray(np.flip(
                np.asarray(f["sino"][:, :, j], dtype=np.float32),
                axis=-1,
            ))
            for j in range(slab_lo, slab_hi)
        ]

    angle_start = float(geom_json["angle_start_corrected"])
    angle_end = angle_start + 2.0 * math.pi
    sod = float(geom_json.get("sod", 595.0))
    sdd = float(geom_json.get("sdd", 1085.6))

    # Truth at the same z (sign-flipped to patient frame). The Mayo
    # head-first convention is patient_z = -source_z; no offset.
    raw_dir = root / "raw" / "L014"
    target_patient_z = -z_center
    truth_mu, ds, bracket = _load_truth_at_z(raw_dir, target_patient_z)
    print(f"[ablation] target_z (patient)={target_patient_z:.2f} mm  "
          f"truth bracket=({bracket[0]:.1f}, {bracket[1]:.1f})  "
          f"w_lo={bracket[2]:.3f}")
    print(f"[ablation] truth PixelSpacing={float(ds.PixelSpacing[0]):.6f} mm  "
          f"kernel={getattr(ds,'ConvolutionKernel','?')}")

    dr = 0.05  # display range (mu range 0..0.05 1/mm)

    rows = []
    for px in (0.7, 0.703125):
        geom = FanBeamGeometry(
            image_size=512, pixel_spacing=px,
            n_angles=rotview, n_det=nu, det_spacing=du,
            sod=sod, sdd=sdd,
            angle_start=angle_start, angle_end=angle_end,
        )
        fbp_np = _fbp(sino_slab, geom, "cuda")
        fbp_t = torch.from_numpy(fbp_np).to("cuda").float()[None, None].clamp_min(0.0)
        truth_t = torch.from_numpy(truth_mu).to("cuda").float()[None, None]
        metrics = evaluate_calibrated(
            fbp_t, truth_t, baseline=fbp_t,
            display_min=0.0, display_max=dr, fov=False,
        )
        ssim = float(metrics["val_ssim"])
        psnr = float(metrics["val_psnr"])
        rmse = float(metrics["val_rmse"])
        pred_cal = metrics["pred_cal"][0, 0].cpu().numpy()
        diff = pred_cal - truth_mu
        print(f"[ablation] px={px:.6f} mm  SSIM={ssim:.4f}  PSNR={psnr:.2f} dB  "
              f"RMSE={rmse:.5f}  body_FOV_diam={512*px:.2f} mm")
        rows.append({
            "px": px, "truth": truth_mu, "pred_cal": pred_cal, "diff": diff,
            "ssim": ssim, "psnr": psnr, "rmse": rmse, "fbp_raw": fbp_np,
        })

    # 2 rows × 4 cols figure
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    for i, r in enumerate(rows):
        suffix = " (LEGACY)" if r["px"] == 0.7 else " (FIX)"
        axes[i, 0].imshow(r["truth"], cmap="gray", vmin=0, vmax=dr)
        axes[i, 0].set_title(f"truth (px_truth=0.703125 mm)\n"
                              f"z_target={target_patient_z:.1f} mm",
                              fontsize=9)
        axes[i, 1].imshow(r["pred_cal"], cmap="gray", vmin=0, vmax=dr)
        axes[i, 1].set_title(f"FBP_cal (px={r['px']:.6f}{suffix})\n"
                              f"SSIM={r['ssim']:.4f}  PSNR={r['psnr']:.2f} dB  "
                              f"RMSE={r['rmse']:.5f}",
                              fontsize=9)
        axes[i, 2].imshow(r["diff"], cmap="seismic", vmin=-0.02, vmax=0.02)
        axes[i, 2].set_title(f"diff (cal − truth)\n"
                              f"max|·|={np.abs(r['diff']).max():.4f}",
                              fontsize=9)
        # Overlay: truth=cyan, FBP_cal=red
        tn = np.clip(r["truth"] / dr, 0.0, 1.0)
        pn = np.clip(r["pred_cal"] / dr, 0.0, 1.0)
        overlay = np.stack([pn, tn, tn], axis=-1)
        axes[i, 3].imshow(overlay)
        axes[i, 3].set_title("overlay (truth=cyan, FBP=red)\n"
                              "white = pixel-perfect alignment",
                              fontsize=9)
        for ax in axes[i]:
            ax.set_xticks([]); ax.set_yticks([])

    fig.suptitle("L014 fulldose helix2fan ablation: pixel_spacing 0.7 mm (top) vs 0.703125 mm (bottom)  "
                 "— FBP FOV = 358.40 mm vs 360.00 mm (= Mayo ReconDiameter)",
                 fontsize=11)
    fig.tight_layout()
    out_png = Path("/cluster/maier/Agent4CT/results/breast_debug/"
                    "L014_pixel_spacing_ablation.png")
    fig.savefig(out_png, dpi=120)
    print(f"[ablation] wrote {out_png}")

    # Numerical headline
    legacy, fix = rows[0], rows[1]
    print()
    print("=== HEADLINE ===")
    print(f"  px=0.7    (LEGACY)  SSIM={legacy['ssim']:.4f}  PSNR={legacy['psnr']:.2f} dB  RMSE={legacy['rmse']:.5f}")
    print(f"  px=0.703125 (FIX)   SSIM={fix['ssim']:.4f}  PSNR={fix['psnr']:.2f} dB  RMSE={fix['rmse']:.5f}")
    print(f"  Δ                   ΔSSIM={fix['ssim']-legacy['ssim']:+.4f}  "
          f"ΔPSNR={fix['psnr']-legacy['psnr']:+.2f} dB  ΔRMSE={(fix['rmse']-legacy['rmse'])/legacy['rmse']*100:+.1f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
