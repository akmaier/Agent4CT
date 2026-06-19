#!/usr/bin/env python -u
"""Sub-millimetre z-sweep on L014 fulldose around the SSIM peak.

The rebin's native z-spacing is 1 mm (`dv_rebinned = 1`). To get finer
than 1-mm z resolution we linearly interpolate between two adjacent
integer-spaced 5-mm slabs:

    z_off = i0 + frac     (i0 = floor(z_off), frac ∈ [0, 1))
    slab_lo   = average of 5 slices centred at nz_middle + i0
    slab_hi   = average of 5 slices centred at nz_middle + i0 + 1
    fbp_slab  = (1 - frac) * fbp(slab_lo) + frac * fbp(slab_hi)

The truth is z-interpolated between the two bracketing DICOM slices at
the matching patient_z (no extra spacing — Mayo's DICOMs already step
3 mm in z).

For each z_off we save (truth, FBP_cal, diff, overlay) into one row of
a multi-row figure, with calibrated SSIM/PSNR/RMSE in the row title.

Usage:  python -u scripts/z_sweep_L014_fine.py
"""
from __future__ import annotations

import math
import sys
import json
from pathlib import Path

import h5py
import numpy as np
import pydicom
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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
        if getattr(head, 'SOPClassUID', '') != SOP_CT: continue
        desc = getattr(head, 'SeriesDescription', '').lower()
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


def _mu(fp):
    ds = pydicom.dcmread(str(fp))
    hu = (ds.pixel_array.astype(np.float32) * float(ds.RescaleSlope)
          + float(ds.RescaleIntercept))
    return 0.02 * (1.0 + hu / 1000.0)


def _truth_interp_at_z(truth_files, target_z):
    zs = np.array([t[0] for t in truth_files])
    idx_above = int(np.searchsorted(zs, target_z, side="left"))
    if idx_above <= 0:
        lo = hi = 0
    elif idx_above >= len(zs):
        lo = hi = len(zs) - 1
    else:
        lo, hi = idx_above - 1, idx_above
    if lo == hi:
        return _mu(truth_files[lo][1]), (zs[lo], zs[lo], 1.0)
    z_lo, z_hi = float(zs[lo]), float(zs[hi])
    w_lo = float(max(0.0, min(1.0, (z_hi - target_z) / (z_hi - z_lo))))
    mu_lo = _mu(truth_files[lo][1])
    mu_hi = _mu(truth_files[hi][1])
    return (w_lo*mu_lo + (1-w_lo)*mu_hi).astype(np.float32), (z_lo, z_hi, w_lo)


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

    truth_files = _load_all_truth(raw_dir)
    sample_ds = pydicom.dcmread(str(truth_files[0][1]), stop_before_pixels=True)
    pixel_sp = float(sample_ds.PixelSpacing[0])
    print(f"[zfine] truth PixelSpacing={pixel_sp:.6f} mm", flush=True)
    print(f"[zfine] nz_middle={nz_middle}  z_middle_source={z_grid[nz_middle]:.2f} mm", flush=True)
    print(f"[zfine] kernel={getattr(sample_ds, 'ConvolutionKernel', '?')}  "
          f"slice_thickness={getattr(sample_ds, 'SliceThickness', '?')} mm", flush=True)

    angle_start = float(geom_json['angle_start_corrected'])
    geom = FanBeamGeometry(
        image_size=512, pixel_spacing=pixel_sp,
        n_angles=rotview, n_det=nu, det_spacing=du,
        sod=595.0, sdd=1085.6,
        angle_start=angle_start, angle_end=angle_start + 2*math.pi,
    )
    proj = PyronnFanBeamProjector(geom).to("cuda")
    SLAB_HALF = 2  # 5-slice / 5-mm slab
    dr = 0.05

    # FBP one integer-anchored slab
    fbp_cache: dict[int, np.ndarray] = {}
    def fbp_int(i0: int) -> np.ndarray:
        if i0 in fbp_cache:
            return fbp_cache[i0]
        nz_center = nz_middle + i0
        if nz_center - SLAB_HALF < 0 or nz_center + SLAB_HALF >= nz:
            raise RuntimeError(f"out-of-range i0={i0}")
        with h5py.File(sino_dir / "L014_sino_fulldose.h5", "r") as f:
            slab = []
            for j in range(nz_center - SLAB_HALF, nz_center + SLAB_HALF + 1):
                s = np.asarray(f["sino"][:, :, j], dtype=np.float32)
                slab.append(np.ascontiguousarray(np.flip(s, axis=-1)))
        recons = []
        for s in slab:
            t = torch.from_numpy(s).to("cuda").float()[None, None]
            out = proj.fbp(t).detach()[0,0].cpu().numpy()
            recons.append(np.fliplr(np.flipud(out)))
        fbp = np.mean(np.stack(recons, axis=0), axis=0)
        fbp = np.clip(fbp, 0.0, None)
        fbp_cache[i0] = fbp
        return fbp

    # Sub-mm z sweep: 0.5-mm steps from +1 to +6 mm (spans the peak).
    z_offs = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0]
    rows = []
    z_source_mid = float(z_grid[nz_middle])
    for z_off in z_offs:
        i0 = int(math.floor(z_off))
        frac = z_off - i0
        fbp_lo = fbp_int(i0)
        fbp = fbp_lo if frac == 0 else (1 - frac) * fbp_lo + frac * fbp_int(i0 + 1)

        source_z = z_source_mid + z_off          # 1 mm/slice in rebinned sino
        patient_z = -source_z                    # sign-flip mapping
        truth, (zl, zh, w_lo) = _truth_interp_at_z(truth_files, patient_z)

        fbp_t = torch.from_numpy(fbp).to("cuda").float()[None, None]
        truth_t = torch.from_numpy(truth).to("cuda").float()[None, None]
        m = evaluate_calibrated(fbp_t, truth_t, baseline=fbp_t,
                                 display_min=0.0, display_max=dr, fov=False)
        ssim = float(m['val_ssim'])
        psnr = float(m['val_psnr'])
        rmse = float(m['val_rmse'])
        pred_cal = m['pred_cal'][0, 0].cpu().numpy()
        diff = pred_cal - truth

        print(f"[zfine] z_off={z_off:+.2f} mm  patient_z={patient_z:7.2f}  "
              f"truth=({zl:.1f},{zh:.1f}) w_lo={w_lo:.3f}  "
              f"SSIM={ssim:.4f}  PSNR={psnr:.2f} dB  RMSE={rmse:.5f}  "
              f"diff_max={np.abs(diff).max():.4f}", flush=True)
        rows.append({
            'z_off': z_off, 'patient_z': patient_z,
            'truth': truth, 'pred_cal': pred_cal, 'diff': diff,
            'ssim': ssim, 'psnr': psnr, 'rmse': rmse,
            'truth_w_lo': w_lo, 'truth_bracket': (zl, zh),
        })

    # Plot: one row per z_off, columns truth | FBP_cal | diff | overlay
    best = max(rows, key=lambda r: r['ssim'])
    n = len(rows)
    fig, axes = plt.subplots(n, 4, figsize=(16, 3.8 * n))
    if n == 1:
        axes = axes[None, :]
    for i, r in enumerate(rows):
        is_peak = (r['z_off'] == best['z_off'])
        tag = " ← peak" if is_peak else ""
        axes[i, 0].imshow(r['truth'], cmap="gray", vmin=0, vmax=dr)
        axes[i, 0].set_title(f"truth z={r['patient_z']:.2f}  "
                              f"bracket=({r['truth_bracket'][0]:.1f},{r['truth_bracket'][1]:.1f}) "
                              f"w_lo={r['truth_w_lo']:.2f}",
                              fontsize=9)
        axes[i, 1].imshow(r['pred_cal'], cmap="gray", vmin=0, vmax=dr)
        axes[i, 1].set_title(f"FBP_cal  z_off=+{r['z_off']:.1f} mm{tag}\n"
                              f"SSIM={r['ssim']:.4f}  PSNR={r['psnr']:.2f} dB  RMSE={r['rmse']:.5f}",
                              fontsize=9)
        axes[i, 2].imshow(r['diff'], cmap="seismic", vmin=-0.02, vmax=0.02)
        axes[i, 2].set_title(f"diff (cal − truth)\nmax|·|={np.abs(r['diff']).max():.4f}",
                              fontsize=9)
        tn = np.clip(r['truth']    / dr, 0.0, 1.0)
        pn = np.clip(r['pred_cal'] / dr, 0.0, 1.0)
        overlay = np.stack([pn, tn, tn], axis=-1)
        axes[i, 3].imshow(overlay)
        axes[i, 3].set_title("overlay  truth=cyan  FBP=red\n(white = aligned)",
                              fontsize=9)
        for ax in axes[i]:
            ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("L014 fulldose fine z-sweep (0.5 mm steps)  "
                 f"— peak SSIM={best['ssim']:.4f} at z_off=+{best['z_off']:.1f} mm",
                 fontsize=12)
    fig.tight_layout()
    out_png = Path("/cluster/maier/Agent4CT/results/breast_debug/L014_z_sweep_fine.png")
    fig.savefig(out_png, dpi=110)
    print(f"[zfine] wrote {out_png}", flush=True)

    # Also dump a one-line summary table
    print()
    print("=== SUMMARY ===")
    print(f"{'z_off':>6s}  {'SSIM':>6s}  {'PSNR':>6s}  {'RMSE':>8s}  {'diff_max':>8s}")
    for r in rows:
        mark = " ★" if r['z_off'] == best['z_off'] else ""
        print(f"{r['z_off']:+5.1f}   {r['ssim']:.4f}  {r['psnr']:.2f}  {r['rmse']:.5f}  "
              f"{np.abs(r['diff']).max():.4f}{mark}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
