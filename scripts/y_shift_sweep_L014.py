"""Sweep an integer pixel-row shift of the FBP image against the truth.

At the same +3.5 mm slab anchor + z-interpolated truth, slide the FBP
along the row axis from -8 to +8 pixels and recompute calibrated SSIM /
PSNR / diff max. The shift maximising SSIM is the y-translation between
the FBP recon grid and the Mayo truth recon grid.

If the optimum is non-zero, our FBP `volume_origin` needs an in-plane
y-offset to align with Mayo's recon centre (which is NOT at gantry iso —
Mayo's `ImagePositionPatient[1]` puts its image centre at scanner
y ≈ −136.5 mm, but PYRO-NN puts iso at the image centre).
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
from ddssl_ldct.metrics import evaluate_calibrated, ssim as ssim_fn, psnr as psnr_fn


def main() -> int:
    import os
    root = Path(os.environ.get("AGENT4CT_DATA",
                                "/cluster/maier/Agent4CT/data")) / "mayo_ldct"
    sino_dir = root / "staged_helix2fan"
    raw_dir = root / "raw" / "L014"
    geom_json = json.loads((sino_dir / "L014_sino_fulldose_geometry.json").read_text())
    z_grid = np.load(sino_dir / "L014_sino_fulldose_z_grid.npy")
    nu, rotview, nz = int(geom_json["nu"]), int(geom_json["rotview"]), int(geom_json["nz_rebinned"])
    du = float(geom_json["du"])
    angle_start = float(geom_json["angle_start_corrected"])

    # +3.5 mm anchor, 5-mm slab
    nz_middle = nz // 2
    nz_center = nz_middle + 4
    z_center_source = float(z_grid[nz_center])
    patient_z = -z_center_source

    with h5py.File(sino_dir / "L014_sino_fulldose.h5", "r") as f:
        slab = []
        for j in range(nz_center-2, nz_center+3):
            s = np.asarray(f["sino"][:, :, j], dtype=np.float32)
            slab.append(np.ascontiguousarray(np.flip(s, axis=-1)))

    geom = FanBeamGeometry(
        image_size=512, pixel_spacing=0.703125,
        n_angles=rotview, n_det=nu, det_spacing=du,
        sod=595.0, sdd=1085.6,
        angle_start=angle_start, angle_end=angle_start+2*math.pi,
    )
    proj = PyronnFanBeamProjector(geom).to("cuda")
    recons = []
    for s in slab:
        t = torch.from_numpy(s).to("cuda").float()[None, None]
        out = proj.fbp(t).detach()[0,0].cpu().numpy()
        recons.append(np.fliplr(np.flipud(out)))
    fbp = np.mean(np.stack(recons, axis=0), axis=0)
    fbp = np.clip(fbp, 0.0, None)

    # Truth at patient_z
    SOP_CT = "1.2.840.10008.5.1.4.1.1.2"
    truth_files = []
    for sd in sorted(raw_dir.iterdir()):
        sample = next(sd.iterdir(), None)
        if sample is None: continue
        try:
            head = pydicom.dcmread(str(sample), stop_before_pixels=True)
        except Exception: continue
        if getattr(head,'SOPClassUID','') != SOP_CT: continue
        desc = getattr(head,'SeriesDescription','').lower()
        if 'full' not in desc or 'image' not in desc: continue
        for fp in sd.iterdir():
            try:
                m = pydicom.dcmread(str(fp), stop_before_pixels=True)
                z = float(m.ImagePositionPatient[2])
                truth_files.append((z, fp))
            except Exception: continue
        break
    truth_files.sort()
    zs = np.array([t[0] for t in truth_files])
    idx_above = int(np.searchsorted(zs, patient_z, side="left"))
    lo, hi = (idx_above-1, idx_above) if 0 < idx_above < len(zs) else (idx_above, idx_above)
    def _mu(fp):
        ds = pydicom.dcmread(str(fp))
        hu = (ds.pixel_array.astype(np.float32) * float(ds.RescaleSlope)
              + float(ds.RescaleIntercept))
        return 0.02 * (1.0 + hu / 1000.0)
    z_lo, z_hi = float(zs[lo]), float(zs[hi])
    w_lo = float(max(0.0, min(1.0, (z_hi - patient_z) / (z_hi - z_lo)))) if z_lo != z_hi else 1.0
    truth = (w_lo * _mu(truth_files[lo][1]) + (1-w_lo) * _mu(truth_files[hi][1])).astype(np.float32) if z_lo != z_hi else _mu(truth_files[lo][1])

    print(f"[y-sweep] z_anchor=+4mm, patient_z={patient_z:.2f}, "
          f"truth bracket=({z_lo:.1f},{z_hi:.1f}) w_lo={w_lo:.3f}")

    dr = 0.05
    truth_t = torch.from_numpy(truth).to("cuda").float()[None, None]

    rows = []
    shifts = list(range(-8, 9))
    for s in shifts:
        # Shift FBP image by s pixels in row direction (positive = shift DOWN)
        if s > 0:
            shifted = np.pad(fbp, ((s, 0), (0, 0)), mode="constant")[:512, :]
        elif s < 0:
            shifted = np.pad(fbp, ((0, -s), (0, 0)), mode="constant")[-512:, :]
        else:
            shifted = fbp

        fbp_t = torch.from_numpy(shifted).to("cuda").float()[None, None]
        m = evaluate_calibrated(
            fbp_t, truth_t, baseline=fbp_t,
            display_min=0.0, display_max=dr, fov=False,
        )
        ssim = float(m['val_ssim'])
        psnr = float(m['val_psnr'])
        rmse = float(m['val_rmse'])
        pred_cal = m['pred_cal'][0,0].cpu().numpy()
        dmax = float(np.abs(pred_cal - truth).max())
        rows.append({"shift": s, "ssim": ssim, "psnr": psnr, "rmse": rmse, "dmax": dmax})
        print(f"[y-sweep] shift={s:+3d} px ({s*0.703125:+.2f} mm)  "
              f"SSIM={ssim:.4f}  PSNR={psnr:.2f} dB  RMSE={rmse:.5f}  diff_max={dmax:.4f}")

    best = max(rows, key=lambda r: r["ssim"])
    print()
    print(f"[y-sweep] OPTIMUM:  shift={best['shift']:+d} px  "
          f"({best['shift']*0.703125:+.2f} mm)  SSIM={best['ssim']:.4f}")

    # Plot
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].plot([r["shift"] for r in rows], [r["ssim"] for r in rows], "o-")
    ax[0].axvline(best['shift'], color='r', ls='--', lw=1)
    ax[0].set_xlabel("FBP row shift (px, +ve = shift FBP DOWN)")
    ax[0].set_ylabel("calibrated SSIM")
    ax[0].set_title(f"y-shift sweep on L014 fulldose, +3.5 mm z anchor\n"
                    f"optimum at {best['shift']:+d} px = {best['shift']*0.703125:+.2f} mm")
    ax[0].grid(alpha=0.3)
    ax[1].plot([r["shift"] for r in rows], [r["dmax"] for r in rows], "o-", color="C3")
    ax[1].axvline(best['shift'], color='r', ls='--', lw=1)
    ax[1].set_xlabel("FBP row shift (px)")
    ax[1].set_ylabel("diff max|·|")
    ax[1].set_title("diff max|·| vs y-shift")
    ax[1].grid(alpha=0.3)
    fig.tight_layout()
    out_png = Path("/cluster/maier/Agent4CT/results/breast_debug/L014_y_shift_sweep.png")
    fig.savefig(out_png, dpi=120)
    print(f"[y-sweep] wrote {out_png}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
