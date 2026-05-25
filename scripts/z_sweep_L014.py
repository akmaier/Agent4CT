"""Fine z-sweep on L014 fulldose to find the optimal slab anchor.

For each z_offset in [-6, +6] mm (1-mm steps), reconstruct a 5-mm slab
anchored at nz_middle + z_offset, fetch the z-interpolated truth at the
same patient_z, and compute calibrated SSIM/PSNR/RMSE. Also reports the
body centroid row in both truth and FBP so we can detect any consistent
vertical shift.

The whole script uses the corrected validator path:
  * pixel_spacing = truth DICOM's value (0.703125 mm)
  * fov=False (no inscribed-circle mask)
  * sign-flip truth-z mapping (patient_z = -source_z, no offset)
  * 5-slice slab averaging on the FBP side
  * Linear z-interpolation between bracketing truth DICOMs

Writes /cluster/maier/Agent4CT/results/breast_debug/L014_z_sweep.png and
prints the table to stdout.
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


def _load_all_truth(raw_dir: Path):
    SOP_CT = "1.2.840.10008.5.1.4.1.1.2"
    truth_files = []
    for series_dir in sorted(raw_dir.iterdir()):
        sample = next(series_dir.iterdir(), None)
        if sample is None: continue
        try:
            head = pydicom.dcmread(str(sample), stop_before_pixels=True)
        except Exception: continue
        if getattr(head,'SOPClassUID','') != SOP_CT: continue
        desc = getattr(head,'SeriesDescription','').lower()
        if 'full' not in desc or 'image' not in desc: continue
        for fp in series_dir.iterdir():
            try:
                m = pydicom.dcmread(str(fp), stop_before_pixels=True)
                z = float(m.ImagePositionPatient[2])
                truth_files.append((z, fp))
            except Exception:
                continue
        break
    truth_files.sort()
    return truth_files


def _truth_interp_at_z(truth_files, target_z):
    zs = np.array([t[0] for t in truth_files])
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
        return 0.02 * (1.0 + hu / 1000.0)

    if lo == hi:
        return _mu(truth_files[lo][1]), (zs[lo], zs[lo], 1.0)
    z_lo, z_hi = float(zs[lo]), float(zs[hi])
    w_lo = float((z_hi - target_z) / (z_hi - z_lo))
    w_lo = max(0.0, min(1.0, w_lo))
    mu_lo = _mu(truth_files[lo][1])
    mu_hi = _mu(truth_files[hi][1])
    return (w_lo*mu_lo + (1-w_lo)*mu_hi).astype(np.float32), (z_lo, z_hi, w_lo)


def _body_row_centroid(img, thresh=0.005):
    mask = img > thresh
    if not mask.any(): return float("nan")
    rs, cs = np.where(mask)
    return float(rs.mean())


def main() -> int:
    import os
    root = Path(os.environ.get("AGENT4CT_DATA",
                                "/cluster/maier/Agent4CT/data")) / "mayo_ldct"
    sino_dir = root / "staged_helix2fan"
    raw_dir = root / "raw" / "L014"
    geom_json = json.loads((sino_dir / "L014_sino_fulldose_geometry.json").read_text())
    z_grid = np.load(sino_dir / "L014_sino_fulldose_z_grid.npy")
    nu = int(geom_json['nu']); rotview = int(geom_json['rotview']); nz = int(geom_json['nz_rebinned'])
    du = float(geom_json['du'])
    nz_middle = nz // 2

    # Use the truth DICOM's pixel_spacing.
    truth_files = _load_all_truth(raw_dir)
    sample_ds = pydicom.dcmread(str(truth_files[0][1]), stop_before_pixels=True)
    pixel_sp = float(sample_ds.PixelSpacing[0])
    print(f"[z-sweep] truth PixelSpacing={pixel_sp:.6f} mm")
    print(f"[z-sweep] nz_middle={nz_middle}, z_middle={z_grid[nz_middle]:.2f} mm (source)")

    angle_start = float(geom_json['angle_start_corrected'])
    geom = FanBeamGeometry(
        image_size=512, pixel_spacing=pixel_sp,
        n_angles=rotview, n_det=nu, det_spacing=du,
        sod=595.0, sdd=1085.6,
        angle_start=angle_start, angle_end=angle_start+2*math.pi,
    )
    proj = PyronnFanBeamProjector(geom).to("cuda")

    # Sweep z_offset in {-6, -5, ..., +6} mm  (1-mm slices, ±2 slab → 5-mm thick)
    offsets = list(range(-6, 7))
    SLAB_HALF = 2
    dr = 0.05

    rows = []
    for z_off in offsets:
        nz_center = nz_middle + z_off
        if nz_center - SLAB_HALF < 0 or nz_center + SLAB_HALF >= nz:
            continue
        z_center_source = float(z_grid[nz_center])
        patient_z = -z_center_source  # sign-flip

        # Slab FBP
        with h5py.File(sino_dir / "L014_sino_fulldose.h5", "r") as f:
            slab = []
            for j in range(nz_center-SLAB_HALF, nz_center+SLAB_HALF+1):
                s = np.asarray(f["sino"][:, :, j], dtype=np.float32)
                slab.append(np.ascontiguousarray(np.flip(s, axis=-1)))
        recons = []
        for s in slab:
            t = torch.from_numpy(s).to("cuda").float()[None, None]
            out = proj.fbp(t).detach()[0,0].cpu().numpy()
            recons.append(np.fliplr(np.flipud(out)))
        fbp = np.mean(np.stack(recons, axis=0), axis=0)
        fbp_clip = np.clip(fbp, 0.0, None)

        # Truth at patient_z
        truth, (z_lo, z_hi, w_lo) = _truth_interp_at_z(truth_files, patient_z)

        # Calibrated metrics
        fbp_t = torch.from_numpy(fbp_clip).to("cuda").float()[None, None]
        truth_t = torch.from_numpy(truth).to("cuda").float()[None, None]
        m = evaluate_calibrated(
            fbp_t, truth_t, baseline=fbp_t,
            display_min=0.0, display_max=dr, fov=False,
        )
        ssim = float(m['val_ssim'])
        psnr = float(m['val_psnr'])
        rmse = float(m['val_rmse'])
        pred_cal = m['pred_cal'][0,0].cpu().numpy()

        bc_truth = _body_row_centroid(truth)
        bc_fbp = _body_row_centroid(pred_cal)

        print(f"[z-sweep] z_off={z_off:+3d}  source_z={z_center_source:6.2f}  patient_z={patient_z:7.2f}  "
              f"truth_bracket=({z_lo:6.1f},{z_hi:6.1f}) w_lo={w_lo:.3f}  "
              f"SSIM={ssim:.4f}  PSNR={psnr:.2f}dB  RMSE={rmse:.5f}  "
              f"body_row truth={bc_truth:6.2f}  fbp={bc_fbp:6.2f}  Δrow={bc_fbp-bc_truth:+5.2f}")

        rows.append({
            "z_off": z_off, "source_z": z_center_source, "patient_z": patient_z,
            "ssim": ssim, "psnr": psnr, "rmse": rmse,
            "bc_truth": bc_truth, "bc_fbp": bc_fbp,
            "truth_w_lo": w_lo,
        })

    # Find optimum
    best = max(rows, key=lambda r: r["ssim"])
    print()
    print(f"[z-sweep] OPTIMUM:  z_off={best['z_off']:+d} mm  SSIM={best['ssim']:.4f}")

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    z_offs = [r["z_off"] for r in rows]
    ssims  = [r["ssim"]  for r in rows]
    psnrs  = [r["psnr"]  for r in rows]
    rmses  = [r["rmse"]  for r in rows]
    drows  = [r["bc_fbp"] - r["bc_truth"] for r in rows]
    axes[0].plot(z_offs, ssims, "o-", color="C0")
    axes[0].axvline(best['z_off'], color='r', ls='--', lw=1, label=f"peak {best['z_off']:+d}")
    axes[0].set_xlabel("z_offset (mm)"); axes[0].set_ylabel("calibrated SSIM"); axes[0].grid(alpha=0.3)
    axes[0].legend()
    axes[0].set_title("SSIM vs z anchor (5-mm slab @ +z_off)")
    axes[1].plot(z_offs, psnrs, "o-", color="C1", label="PSNR (dB)")
    axes[1].plot(z_offs, np.array(rmses) * 5e4, "s-", color="C2", label="RMSE ×5e4")
    axes[1].axvline(best['z_off'], color='r', ls='--', lw=1)
    axes[1].set_xlabel("z_offset (mm)"); axes[1].grid(alpha=0.3); axes[1].legend()
    axes[1].set_title("PSNR and RMSE")
    axes[2].plot(z_offs, drows, "o-", color="C3")
    axes[2].axhline(0, color='k', ls=':', lw=0.5)
    axes[2].axvline(best['z_off'], color='r', ls='--', lw=1)
    axes[2].set_xlabel("z_offset (mm)"); axes[2].set_ylabel("FBP row - truth row")
    axes[2].set_title("Body centroid Δrow (positive = FBP body lower than truth)")
    axes[2].grid(alpha=0.3)
    fig.suptitle("L014 fulldose z-sweep: 5-mm slab anchored at nz_middle + z_offset")
    fig.tight_layout()
    out_png = Path("/cluster/maier/Agent4CT/results/breast_debug/L014_z_sweep.png")
    fig.savefig(out_png, dpi=120)
    print(f"[z-sweep] wrote {out_png}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
