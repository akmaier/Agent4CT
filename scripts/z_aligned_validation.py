#!/usr/bin/env python -u
"""Z-aligned L014 fulldose validation: each comparison is one specific
GT slice, with no GT interpolation. The FBP slab is integrated on the
EXACT same physical z range as Mayo's truth (5-mm slice thickness
centered on the truth's reported z).

Procedure for one truth slice at patient_z = pZ:
  1. truth = pixel array of the DICOM at pZ — no interpolation
  2. truth slab in source frame = [-pZ - 2.5, -pZ + 2.5] mm
  3. For every sino slice j with centre z_j = z_start + j*dv,
     compute weight = overlap of [z_j - dv/2, z_j + dv/2] with the
     slab. FBP each contributing slice, sum weighted recons, divide
     by the slab thickness.
  4. Calibrated SSIM/PSNR/RMSE without FOV mask.

Iterate over a range of truth z's around the cone-beam centre and
emit a CSV-like table + a multi-row image grid (truth | FBP_cal |
diff | overlay) per slice.

Usage:  python -u scripts/z_aligned_validation.py
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


def _list_truth(raw_dir: Path):
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


def _mu_from_dicom(fp: Path):
    ds = pydicom.dcmread(str(fp))
    hu = (ds.pixel_array.astype(np.float32) * float(ds.RescaleSlope)
          + float(ds.RescaleIntercept))
    return 0.02 * (1.0 + hu / 1000.0), ds


def main() -> int:
    import os
    root = Path(os.environ.get("AGENT4CT_DATA",
                                "/cluster/maier/Agent4CT/data")) / "mayo_ldct"
    sino_dir = root / "staged_helix2fan"
    raw_dir = root / "raw" / "L014"
    geom_json = json.loads((sino_dir / "L014_sino_fulldose_geometry.json").read_text())
    z_grid_src = np.load(sino_dir / "L014_sino_fulldose_z_grid.npy")
    nu, rotview, nz = int(geom_json['nu']), int(geom_json['rotview']), int(geom_json['nz_rebinned'])
    du = float(geom_json['du'])
    dv = float(geom_json.get('dv_rebinned', 1.0))
    z_start_src = float(geom_json['z_start'])

    truth_files = _list_truth(raw_dir)
    ds_sample = pydicom.dcmread(str(truth_files[0][1]), stop_before_pixels=True)
    pixel_sp = float(ds_sample.PixelSpacing[0])
    slice_thk = float(ds_sample.SliceThickness)
    print(f"[zalign] truth: PixelSpacing={pixel_sp:.6f} mm  "
          f"SliceThickness={slice_thk} mm  kernel={ds_sample.ConvolutionKernel}",
          flush=True)
    print(f"[zalign] sino:  z_start_src={z_start_src:.3f} mm  dv={dv} mm  "
          f"nz={nz}  (source frame, monotonic increasing)", flush=True)
    print(f"[zalign] truth z range (patient frame): "
          f"{truth_files[0][0]:.2f} → {truth_files[-1][0]:.2f} mm  "
          f"({len(truth_files)} slices @ {truth_files[1][0]-truth_files[0][0]:.2f} mm spacing)",
          flush=True)

    # Pick the cone-beam centre truth slice as the anchor, plus 5 around it
    # (10 truth slices total). They are 3 mm apart, so this spans 27 mm.
    truth_zs = np.array([t[0] for t in truth_files])
    # Source-frame centre of the sino
    sino_centre_src = z_start_src + (nz / 2) * dv
    target_patient_z_centre = -sino_centre_src
    centre_idx = int(np.argmin(np.abs(truth_zs - target_patient_z_centre)))
    span = 5
    idxs = list(range(max(0, centre_idx - span), min(len(truth_files), centre_idx + span + 1)))
    print(f"[zalign] sino source-frame centre = {sino_centre_src:.2f} mm  "
          f"→ patient_z={target_patient_z_centre:.2f}  → truth slice index "
          f"{centre_idx} (z={truth_zs[centre_idx]:.2f})", flush=True)
    print(f"[zalign] sweep over truth indices {idxs[0]}..{idxs[-1]} "
          f"= {len(idxs)} GT slices", flush=True)

    # FBP geometry (truth-matched pixel_spacing)
    angle_start = float(geom_json['angle_start_corrected'])
    geom = FanBeamGeometry(
        image_size=512, pixel_spacing=pixel_sp,
        n_angles=rotview, n_det=nu, det_spacing=du,
        sod=595.0, sdd=1085.6,
        angle_start=angle_start, angle_end=angle_start + 2*math.pi,
    )
    proj = PyronnFanBeamProjector(geom).to("cuda")

    dr = 0.05  # display range
    rows = []
    fbp_cache: dict[int, np.ndarray] = {}

    def fbp_slice(j: int) -> np.ndarray:
        """Return the post-flipud+fliplr FBP of sino slice j."""
        if j in fbp_cache:
            return fbp_cache[j]
        with h5py.File(sino_dir / "L014_sino_fulldose.h5", "r") as f:
            s = np.asarray(f["sino"][:, :, j], dtype=np.float32)
        s = np.ascontiguousarray(np.flip(s, axis=-1))
        t = torch.from_numpy(s).to("cuda").float()[None, None]
        out = proj.fbp(t).detach()[0, 0].cpu().numpy()
        out = np.fliplr(np.flipud(out))
        fbp_cache[j] = out
        return out

    for ti in idxs:
        pZ, fp = truth_files[ti]
        truth_mu, _ds = _mu_from_dicom(fp)

        # Slab in source frame: [-pZ - thk/2, -pZ + thk/2]
        slab_lo_src = -pZ - slice_thk / 2.0
        slab_hi_src = -pZ + slice_thk / 2.0
        # For each sino j: bin centre z_j = z_start_src + j*dv, bin = [z_j - dv/2, z_j + dv/2]
        # weight_j = overlap of bin with slab, normalised by slab thickness
        j_lo = max(0, int(math.floor((slab_lo_src - z_start_src) / dv - 0.5)))
        j_hi = min(nz - 1, int(math.ceil((slab_hi_src - z_start_src) / dv + 0.5)))
        weights = {}
        for j in range(j_lo, j_hi + 1):
            z_j = z_start_src + j * dv
            bin_lo, bin_hi = z_j - dv / 2.0, z_j + dv / 2.0
            ov = max(0.0, min(bin_hi, slab_hi_src) - max(bin_lo, slab_lo_src))
            if ov > 0:
                weights[j] = ov / slice_thk   # normalised so sum(weights) = 1
        wsum = sum(weights.values())
        assert abs(wsum - 1.0) < 1e-6, f"weights don't normalise: {wsum}"

        # Compose the slab FBP
        fbp_slab = np.zeros_like(truth_mu, dtype=np.float64)
        for j, w in weights.items():
            fbp_slab += w * fbp_slice(j)
        fbp_slab = np.clip(fbp_slab.astype(np.float32), 0.0, None)

        # Metric (calibrated, no FOV)
        fbp_t = torch.from_numpy(fbp_slab).to("cuda").float()[None, None]
        truth_t = torch.from_numpy(truth_mu).to("cuda").float()[None, None]
        m = evaluate_calibrated(fbp_t, truth_t, baseline=fbp_t,
                                 display_min=0.0, display_max=dr, fov=False)
        ssim = float(m['val_ssim'])
        psnr = float(m['val_psnr'])
        rmse = float(m['val_rmse'])
        pred_cal = m['pred_cal'][0, 0].cpu().numpy()
        diff = pred_cal - truth_mu

        print(f"[zalign] truth#{ti:3d}  pZ={pZ:7.2f}  slab_src=[{slab_lo_src:.2f},{slab_hi_src:.2f}]  "
              f"weights={ {j: round(w, 3) for j, w in weights.items()} }  "
              f"SSIM={ssim:.4f}  PSNR={psnr:.2f}dB  RMSE={rmse:.5f}  "
              f"diff_max={np.abs(diff).max():.4f}", flush=True)
        rows.append({
            'ti': ti, 'pZ': pZ,
            'slab_lo': slab_lo_src, 'slab_hi': slab_hi_src,
            'truth': truth_mu, 'fbp_cal': pred_cal, 'diff': diff,
            'ssim': ssim, 'psnr': psnr, 'rmse': rmse,
            'weights': weights,
        })

    # Summary + image grid
    best = max(rows, key=lambda r: r['ssim'])
    print()
    print("=== SUMMARY (no GT interpolation, slab integral by physical overlap) ===")
    print(f"{'truth#':>6s}  {'pZ':>8s}  {'SSIM':>6s}  {'PSNR':>6s}  {'RMSE':>8s}  {'diff_max':>8s}")
    for r in rows:
        mark = " ★" if r['ti'] == best['ti'] else ""
        print(f"{r['ti']:6d}  {r['pZ']:+8.2f}  {r['ssim']:.4f}  {r['psnr']:.2f}  "
              f"{r['rmse']:.5f}  {np.abs(r['diff']).max():.4f}{mark}")

    n = len(rows)
    fig, axes = plt.subplots(n, 4, figsize=(16, 3.8 * n))
    if n == 1:
        axes = axes[None, :]
    for i, r in enumerate(rows):
        is_peak = (r['ti'] == best['ti'])
        tag = "  ← peak" if is_peak else ""
        axes[i, 0].imshow(r['truth'], cmap="gray", vmin=0, vmax=dr)
        axes[i, 0].set_title(f"truth #{r['ti']}  pZ={r['pZ']:.2f} mm  "
                              f"(no interp, single DICOM)",
                              fontsize=9)
        axes[i, 1].imshow(r['fbp_cal'], cmap="gray", vmin=0, vmax=dr)
        axes[i, 1].set_title(f"FBP_cal  slab_src=[{r['slab_lo']:.2f}, {r['slab_hi']:.2f}]{tag}\n"
                              f"SSIM={r['ssim']:.4f}  PSNR={r['psnr']:.2f} dB  RMSE={r['rmse']:.5f}",
                              fontsize=9)
        axes[i, 2].imshow(r['diff'], cmap="seismic", vmin=-0.02, vmax=0.02)
        axes[i, 2].set_title(f"diff (cal − truth)\nmax|·|={np.abs(r['diff']).max():.4f}",
                              fontsize=9)
        tn = np.clip(r['truth']   / dr, 0.0, 1.0)
        pn = np.clip(r['fbp_cal'] / dr, 0.0, 1.0)
        overlay = np.stack([pn, tn, tn], axis=-1)
        axes[i, 3].imshow(overlay)
        axes[i, 3].set_title("overlay (truth=cyan, FBP=red)\nwhite = pixel-aligned",
                              fontsize=9)
        for ax in axes[i]:
            ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(f"L014 fulldose z-aligned validation — single GT slice per row, "
                 f"physical-overlap slab integral  (peak SSIM={best['ssim']:.4f} "
                 f"at truth#{best['ti']} z={best['pZ']:.2f})",
                 fontsize=11)
    fig.tight_layout()
    out_png = Path("/cluster/maier/Agent4CT/results/breast_debug/L014_z_aligned.png")
    fig.savefig(out_png, dpi=110)
    print(f"[zalign] wrote {out_png}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
