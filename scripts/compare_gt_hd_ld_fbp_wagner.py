"""Bulk GT vs HD-FBP vs LD-FBP comparison across all 10 Wagner Mayo patients.

For each patient (Wagner train+val+test = 10 total), pick the central z
slice of the rebinned sinogram, FBP it for HD and LD, load the matching
truth DICOM (z-interpolated to the FBP slab centre), compute metrics,
and assemble a 10×3 panel:

    col 0: truth (GT)
    col 1: HD FBP   + SSIM / PSNR
    col 2: LD FBP   + SSIM / PSNR

The pipeline replicates `scripts/validate_mayo_helix2fan.py` for one
slice per patient, factored to share code via that module. The FBP
geometry is `FanBeamGeometry.mayo_ldct_fitted()` + `MAYO_LDCT_DET_OFFSET`,
the post-FBP intensity calibration uses `evaluate_calibrated`, and
truth alignment uses `_load_truth_slice_for_z`.

Input: rebinned sinograms under `data/mayo_ldct/<STAGED_HELIX2FAN_SUBDIR>/`.
Output: `results/mayo_debug/wagner_gt_hd_ld_fbp_<tag>.png` + per-patient
        metrics JSON.

Usage:
    python scripts/compare_gt_hd_ld_fbp_wagner.py --tag v3
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from ddssl_ldct.geometry import FanBeamGeometry, MAYO_LDCT_DET_OFFSET
from ddssl_ldct.pyronn_projector import PyronnFanBeamProjector
from ddssl_ldct.metrics import (
    ssim as ssim_metric, psnr as psnr_metric, evaluate_calibrated,
)
# Reuse validator's truth-loading + frame-mapping logic.
from scripts.validate_mayo_helix2fan import _load_truth_slice_for_z


WAGNER_ALL = [
    "L145", "L186", "L209", "L219",   # train
    "L277",                            # val
    "L014", "L056", "L058", "L075", "L123",  # test
]
WAGNER_SPLIT_OF = {
    **{p: "train" for p in ["L145", "L186", "L209", "L219"]},
    "L277": "val",
    **{p: "test" for p in ["L014", "L056", "L058", "L075", "L123"]},
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", default=None,
                   help="Override AGENT4CT_DATA root.")
    p.add_argument("--tag", default="v3",
                   help="Suffix on output PNG / JSON. Defaults to 'v3'.")
    p.add_argument("--out-dir", default=None,
                   help="Output dir. Default: results/mayo_debug.")
    p.add_argument("--z-offset-mm", type=float, default=3.5,
                   help="See validate_mayo_helix2fan.py (default +3.5 mm).")
    p.add_argument("--slab-half", type=int, default=2,
                   help="Half-width of FBP slab in 1-mm slices (default 2 → 5 mm).")
    p.add_argument("--device", default="cuda")
    p.add_argument("--display-max", type=float, default=0.05,
                   help="Display window upper bound (μ mm^-1). Default 0.05.")
    return p.parse_args()


def _pick_z_center(geom_json: dict, zgrid: np.ndarray, z_offset_mm: float):
    """Source-frame z of the central rebinned slab (no FBP), so truth can
    be loaded before the FBP grid is built (we need the truth pixel
    spacing to set the FBP FOV)."""
    nz = int(geom_json["nz_rebinned"])
    nz_center = nz // 2 + int(round(z_offset_mm))
    nz_center = max(0, min(nz - 1, nz_center))
    return float(zgrid[nz_center])


def _fbp_central_slab(sino_h5: Path, geom_json: dict, zgrid: np.ndarray,
                       z_offset_mm: float, slab_half: int, device: str,
                       pixel_spacing: float = None):
    """Load central slab, FBP each member with v3-final geometry, average."""
    rotview = int(geom_json["rotview"])
    nu = int(geom_json["nu"])
    nz = int(geom_json["nz_rebinned"])

    nz_middle = nz // 2
    nz_center = nz_middle + int(round(z_offset_mm))
    nz_center = max(0, min(nz - 1, nz_center))
    z_center = float(zgrid[nz_center])
    slab_lo = max(0, nz_center - slab_half)
    slab_hi = min(nz, nz_center + slab_half + 1)

    with h5py.File(sino_h5, "r") as f:
        sino_slab = [
            np.asarray(f["sino"][:, :, j], dtype=np.float32)
            for j in range(slab_lo, slab_hi)
        ]
    sino_slab = [np.ascontiguousarray(np.flip(s, axis=-1)) for s in sino_slab]

    angle_start = float(geom_json.get("angle_start_corrected", 0.0))
    angle_end = angle_start + 2.0 * math.pi
    # FBP grid pixel spacing MUST match this patient's truth display FOV.
    # mayo_ldct_fitted()'s 0.700857 mm was calibrated on L014, whose truth
    # PixelSpacing is 0.703125 mm (360 mm FOV). Mayo patients are
    # reconstructed at varying display FOVs (340/360/380/400 mm →
    # ps 0.6641/0.7031/0.7422/0.7812). Rendering every FBP at 0.700857
    # would put the off-360 patients at the wrong physical scale and the
    # anatomy would not overlap truth (SSIM collapses to ~0.6). We scale
    # the calibrated spacing by (truth_ps / 0.703125) to preserve the
    # sub-pixel fit while matching each patient's FOV. det_spacing / sod /
    # sdd / det_offset are FOV-independent and stay at the fitted values.
    fitted = FanBeamGeometry.mayo_ldct_fitted(
        n_angles=rotview, n_det=nu,
        angle_start=angle_start, angle_end=angle_end,
    )
    if pixel_spacing is None:
        pixel_spacing = fitted.pixel_spacing
    geom = FanBeamGeometry(
        image_size=512, pixel_spacing=pixel_spacing,
        n_angles=rotview, n_det=nu, det_spacing=fitted.det_spacing,
        sod=fitted.sod, sdd=fitted.sdd,
        angle_start=angle_start, angle_end=angle_end,
    )
    proj = PyronnFanBeamProjector(geom).to(device)
    proj._tensor_geom["detector_origin"] = (
        proj._tensor_geom["detector_origin"] + MAYO_LDCT_DET_OFFSET
    )

    fbp_slab_np = []
    for s in sino_slab:
        s_t = torch.from_numpy(s).to(device).float()[None, None]
        fbp_one = proj.fbp(s_t).detach()[0, 0].cpu().numpy()
        fbp_slab_np.append(np.fliplr(np.flipud(fbp_one)))
    fbp_np = np.ascontiguousarray(np.mean(np.stack(fbp_slab_np, axis=0), axis=0))
    return fbp_np, z_center, (slab_lo, slab_hi - 1)


def _run_patient(patient: str, sino_dir: Path, truth_root: Path,
                  z_offset_mm: float, slab_half: int, device: str,
                  display_max: float) -> dict:
    """Return dict with truth/HD-FBP/LD-FBP arrays + metrics for this patient."""
    out = {"patient": patient, "split": WAGNER_SPLIT_OF[patient]}
    t0 = time.time()

    # Use the FULLDOSE sino z_center as the anchor — the matching truth
    # DICOM is the same physical position; the only difference between
    # HD/LD is the projection noise level.
    sino_hd = sino_dir / f"{patient}_sino_fulldose.h5"
    sino_ld = sino_dir / f"{patient}_sino_lowdose.h5"
    geom_hd_path = sino_dir / f"{patient}_sino_fulldose_geometry.json"
    geom_ld_path = sino_dir / f"{patient}_sino_lowdose_geometry.json"
    zgrid_hd_path = sino_dir / f"{patient}_sino_fulldose_z_grid.npy"
    zgrid_ld_path = sino_dir / f"{patient}_sino_lowdose_z_grid.npy"

    for p in [sino_hd, sino_ld, geom_hd_path, geom_ld_path,
              zgrid_hd_path, zgrid_ld_path]:
        if not p.exists():
            return {"patient": patient, "error": f"missing {p.name}",
                    "split": out["split"]}

    geom_hd = json.loads(geom_hd_path.read_text())
    geom_ld = json.loads(geom_ld_path.read_text())
    zgrid_hd = np.load(zgrid_hd_path)
    zgrid_ld = np.load(zgrid_ld_path)

    # Truth FIRST (at the HD central-slab z) so we know this patient's
    # display FOV / pixel spacing before building the FBP grid. HD/LD
    # share the same physical z (alternate dose levels of the same scan).
    z_center_hd0 = _pick_z_center(geom_hd, zgrid_hd, z_offset_mm)
    truth_info = _load_truth_slice_for_z(truth_root / patient, z_center_hd0)
    if truth_info is None:
        return {"patient": patient, "error": "no truth slice",
                "split": out["split"]}
    truth, target_pz, bracket, truth_meta = truth_info
    truth_ps = float(truth_meta["pixel_spacing"])
    # Scale the L014-calibrated FBP spacing to this patient's truth FOV.
    ps_eff = 0.700857 * (truth_ps / 0.703125)

    # HD / LD FBP at central slab, rendered on the FOV-matched grid.
    fbp_hd, z_center_hd, slab_hd = _fbp_central_slab(
        sino_hd, geom_hd, zgrid_hd, z_offset_mm, slab_half, device,
        pixel_spacing=ps_eff)
    fbp_ld, z_center_ld, slab_ld = _fbp_central_slab(
        sino_ld, geom_ld, zgrid_ld, z_offset_mm, slab_half, device,
        pixel_spacing=ps_eff)

    dr = float(display_max)
    truth_t = torch.from_numpy(truth).to(device).float()[None, None]

    metrics_per = {}
    for label, fbp_np in [("hd", fbp_hd), ("ld", fbp_ld)]:
        fbp_t = torch.from_numpy(fbp_np).to(device).float()[None, None]
        fbp_clipped = fbp_t.clamp(min=0.0)
        ssim_raw = float(ssim_metric(fbp_clipped, truth_t, dr).cpu())
        psnr_raw = float(psnr_metric(fbp_clipped, truth_t, dr).cpu())
        cal = evaluate_calibrated(
            fbp_clipped, truth_t,
            baseline=fbp_clipped,
            display_min=0.0, display_max=dr,
            fov=False, bg_target="truth",   # Mayo: truth background != 0
        )
        metrics_per[label] = {
            "ssim_raw": ssim_raw, "psnr_raw": psnr_raw,
            "ssim_cal": float(cal["val_ssim"]),
            "psnr_cal": float(cal["val_psnr"]),
            "rmse_cal": float(cal["val_rmse"]),
        }
        # Stash calibrated array for display.
        out[f"{label}_cal"] = cal["pred_cal"][0, 0].cpu().numpy().astype(np.float32)
        out[f"{label}_raw"] = fbp_np.astype(np.float32)

    out["truth"] = truth.astype(np.float32)
    out["truth_z_patient"] = float(target_pz)
    out["z_center_hd"] = float(z_center_hd)
    out["z_center_ld"] = float(z_center_ld)
    out["truth_pixel_spacing"] = float(truth_meta["pixel_spacing"])
    out["truth_kernel"] = str(truth_meta["recon_kernel"])
    out["truth_thickness"] = float(truth_meta["slice_thickness"])
    out["metrics"] = metrics_per
    out["elapsed_s"] = time.time() - t0
    print(f"[wagner-compare] {patient} ({out['split']:5s}): "
          f"HD SSIM_cal={metrics_per['hd']['ssim_cal']:.4f}/"
          f"PSNR_cal={metrics_per['hd']['psnr_cal']:5.2f}  "
          f"LD SSIM_cal={metrics_per['ld']['ssim_cal']:.4f}/"
          f"PSNR_cal={metrics_per['ld']['psnr_cal']:5.2f}  "
          f"[{out['elapsed_s']:.1f}s]", flush=True)
    return out


def main() -> int:
    args = parse_args()
    root = Path(args.data_root or os.environ.get(
        "AGENT4CT_DATA", "/cluster/maier/Agent4CT/data"))
    challenge = root / "mayo_ldct"
    sino_subdir = os.environ.get("STAGED_HELIX2FAN_SUBDIR",
                                  "staged_helix2fan_v3")
    sino_dir = challenge / sino_subdir
    truth_root = challenge / "raw"
    out_dir = Path(args.out_dir) if args.out_dir else (
        REPO / "results" / "mayo_debug")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[wagner-compare] sino_dir = {sino_dir}", flush=True)
    print(f"[wagner-compare] truth_root = {truth_root}", flush=True)
    print(f"[wagner-compare] out_dir = {out_dir}", flush=True)
    print(f"[wagner-compare] device = {args.device}", flush=True)

    rows = []
    for patient in WAGNER_ALL:
        try:
            row = _run_patient(
                patient, sino_dir, truth_root,
                args.z_offset_mm, args.slab_half,
                args.device, args.display_max,
            )
        except Exception as e:
            print(f"[wagner-compare] {patient} FAILED: {e!r}", flush=True)
            row = {"patient": patient, "split": WAGNER_SPLIT_OF[patient],
                   "error": repr(e)}
        rows.append(row)

    # -- 10×3 panel --
    valid = [r for r in rows if "error" not in r]
    n = len(valid)
    if n == 0:
        print("[wagner-compare] no valid rows; abort", flush=True)
        return 1

    dr = float(args.display_max)
    fig, axes = plt.subplots(n, 3, figsize=(12, 3.0 * n))
    if n == 1:
        axes = axes[None, :]
    for i, r in enumerate(valid):
        ax_gt, ax_hd, ax_ld = axes[i, 0], axes[i, 1], axes[i, 2]
        ax_gt.imshow(r["truth"], cmap="gray", vmin=0, vmax=dr)
        ax_gt.set_title(f"{r['patient']} ({r['split']})\n"
                        f"GT  pZ={r['truth_z_patient']:+.1f}  "
                        f"{r['truth_kernel']}",
                        fontsize=9)
        ax_hd.imshow(r["hd_cal"], cmap="gray", vmin=0, vmax=dr)
        m_hd = r["metrics"]["hd"]
        ax_hd.set_title(f"HD-FBP cal\nSSIM={m_hd['ssim_cal']:.3f}  "
                        f"PSNR={m_hd['psnr_cal']:.2f} dB",
                        fontsize=9)
        ax_ld.imshow(r["ld_cal"], cmap="gray", vmin=0, vmax=dr)
        m_ld = r["metrics"]["ld"]
        ax_ld.set_title(f"LD-FBP cal\nSSIM={m_ld['ssim_cal']:.3f}  "
                        f"PSNR={m_ld['psnr_cal']:.2f} dB  "
                        f"Δ={m_ld['psnr_cal'] - m_hd['psnr_cal']:+.2f} dB",
                        fontsize=9)
        for a in (ax_gt, ax_hd, ax_ld):
            a.set_xticks([]); a.set_yticks([])

    fig.suptitle(f"Wagner Mayo — GT vs HD-FBP vs LD-FBP, central slice  "
                 f"(geometry: v3 SSR + Powell FBP; sino_subdir={sino_subdir})",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.99])
    out_png = out_dir / f"wagner_gt_hd_ld_fbp_{args.tag}.png"
    fig.savefig(out_png, dpi=110, bbox_inches="tight")
    print(f"[wagner-compare] wrote {out_png}", flush=True)

    # -- JSON metrics --
    summary = {
        "tag": args.tag,
        "sino_subdir": sino_subdir,
        "z_offset_mm": args.z_offset_mm,
        "slab_half": args.slab_half,
        "patients": [
            {
                "patient": r["patient"], "split": r["split"],
                **({"error": r["error"]} if "error" in r else {
                    "truth_z_patient": r["truth_z_patient"],
                    "z_center_hd": r["z_center_hd"],
                    "metrics": r["metrics"],
                    "truth_pixel_spacing": r["truth_pixel_spacing"],
                    "truth_kernel": r["truth_kernel"],
                    "truth_thickness": r["truth_thickness"],
                }),
            }
            for r in rows
        ],
    }
    # Compute aggregates over valid rows.
    if valid:
        for label in ("hd", "ld"):
            ssim_arr = np.array([r["metrics"][label]["ssim_cal"] for r in valid])
            psnr_arr = np.array([r["metrics"][label]["psnr_cal"] for r in valid])
            summary[f"{label}_aggregate"] = {
                "ssim_cal_mean": float(ssim_arr.mean()),
                "psnr_cal_mean": float(psnr_arr.mean()),
                "n": len(valid),
            }

    out_json = out_dir / f"wagner_gt_hd_ld_fbp_{args.tag}.json"
    out_json.write_text(json.dumps(summary, indent=2))
    print(f"[wagner-compare] wrote {out_json}", flush=True)

    # Stash per-patient .npz so downstream re-rendering doesn't re-run FBP.
    npz_dir = out_dir / f"wagner_gt_hd_ld_fbp_{args.tag}_arrays"
    npz_dir.mkdir(exist_ok=True, parents=True)
    for r in valid:
        np.savez(
            npz_dir / f"{r['patient']}.npz",
            truth=r["truth"],
            hd_raw=r["hd_raw"], hd_cal=r["hd_cal"],
            ld_raw=r["ld_raw"], ld_cal=r["ld_cal"],
        )
    print(f"[wagner-compare] wrote arrays to {npz_dir}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
