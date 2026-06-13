"""Wagner GT/HD/LD comparison WITH truncation correction + difference images.

Follow-up to compare_gt_hd_ld_fbp_wagner.py (2026-06-13). The 400 mm-FOV
patients (L145, L186) show classic FBP truncation artifacts (bright rim
+ cupping) because the body extends past the scan field-of-measurement,
so the rebinned fan projections are cut off at the detector edges and the
ramp filter amplifies the discontinuity.

This variant adds a **water-cylinder sinogram extrapolation** (Hsieh et
al. 2004 style): each rebinned fan view is extended on both u-edges by a
water-cylinder profile matched in value + slope to the truncation
boundary, decaying smoothly to zero. The FBP then runs on a widened
virtual detector (n_det + 2*pad) with the same image grid, removing the
truncation discontinuity within the 512^2 FOV.

For every patient we reconstruct BOTH arms (raw + truncation-corrected),
each dose (HD/LD), and render difference images vs truth so the residual
problems are visible. Geometry is v3-final + per-patient display-FOV
pixel spacing (see findings.md 2026-06-13).

Outputs (tag default 'v3trunc'):
  results/mayo_debug/wagner_gt_hd_ld_fbp_<tag>.png          (panel: GT|HD_tc|diff)
  results/mayo_debug/wagner_gt_hd_ld_fbp_<tag>.json         (raw+tc metrics)
  results/mayo_debug/wagner_per_patient_<tag>/<pat>.png     (GT|HD_raw|HD_tc|dHD|LD_tc|dLD)
  results/mayo_debug/wagner_gt_hd_ld_fbp_<tag>_arrays/<pat>.npz
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
from scripts.validate_mayo_helix2fan import _load_truth_slice_for_z

WAGNER_ALL = ["L145", "L186", "L209", "L219", "L277",
              "L014", "L056", "L058", "L075", "L123"]
WAGNER_SPLIT_OF = {
    **{p: "train" for p in ["L145", "L186", "L209", "L219"]},
    "L277": "val",
    **{p: "test" for p in ["L014", "L056", "L058", "L075", "L123"]},
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", default=None)
    p.add_argument("--tag", default="v3trunc")
    p.add_argument("--out-dir", default=None)
    p.add_argument("--z-offset-mm", type=float, default=3.5)
    p.add_argument("--slab-half", type=int, default=2)
    p.add_argument("--device", default="cuda")
    p.add_argument("--display-max", type=float, default=0.05)
    p.add_argument("--pad", type=int, default=384,
                   help="Detector channels added per side for the water-"
                        "cylinder extrapolation (widened FBP detector).")
    p.add_argument("--mu-water", type=float, default=0.02,
                   help="Water linear attenuation (mm^-1) for the cylinder "
                        "model; sets the extrapolation decay width.")
    return p.parse_args()


def _edge_slope_weights(k: int):
    """Least-squares slope weights for k equally-spaced points (x=0..k-1):
    slope = sum_i w_i y_i  (per index step)."""
    x = np.arange(k, dtype=np.float64)
    xb = x.mean()
    sxx = ((x - xb) ** 2).sum()
    return ((x - xb) / sxx).astype(np.float32)


def water_cylinder_extrapolate(sino2d: np.ndarray, du_iso: float, pad: int,
                                mu_water: float = 0.02, edge_k: int = 7
                                ) -> np.ndarray:
    """Vectorised water-cylinder edge extrapolation on a fan sino (V, nu).

    For each view + each u-edge: match the measured boundary value p_b and
    the OUTWARD gradient g (mm^-1) to a water cylinder
        f(t) = 2*mu*sqrt(R^2 - (t - t_c)^2),
    then fill the pad channels with the cylinder decaying to 0. Returns a
    (V, nu + 2*pad) array. Self-gating: a near-zero edge -> tiny R -> ~0 pad.
    """
    V, nu = sino2d.shape
    out = np.zeros((V, nu + 2 * pad), dtype=np.float32)
    out[:, pad:pad + nu] = sino2d
    w = _edge_slope_weights(edge_k)
    ch = np.arange(pad, dtype=np.float64)
    Rmax = pad * du_iso

    # ---- left edge (low u) ----
    p_b = np.maximum(sino2d[:, :edge_k].mean(axis=1), 1e-4)          # (V,)
    s_in = sino2d[:, :edge_k] @ w            # d p / d channel, + if rises inward
    g = -s_in / du_iso                       # outward (decreasing u) gradient, mm^-1
    g = np.minimum(g, 0.0)                   # enforce non-increasing outward
    tau = -g * p_b / (4.0 * mu_water ** 2)   # mm, >= 0
    R = np.minimum(np.sqrt((p_b / (2.0 * mu_water)) ** 2 + tau ** 2), Rmax)  # (V,)
    # channel j in [0,pad): outward distance from boundary = (pad - j)*du_iso
    d = (pad - ch)[None, :] * du_iso          # (1, pad)
    t = tau[:, None] + d                      # (V, pad)
    val = np.where(t < R[:, None],
                   2.0 * mu_water * np.sqrt(np.maximum(R[:, None] ** 2 - t ** 2, 0.0)),
                   0.0)
    out[:, :pad] = val.astype(np.float32)

    # ---- right edge (high u) ----
    p_b = np.maximum(sino2d[:, -edge_k:].mean(axis=1), 1e-4)
    s_in = sino2d[:, -edge_k:][:, ::-1] @ w   # inward = decreasing channel
    g = -s_in / du_iso
    g = np.minimum(g, 0.0)
    tau = -g * p_b / (4.0 * mu_water ** 2)
    R = np.minimum(np.sqrt((p_b / (2.0 * mu_water)) ** 2 + tau ** 2), Rmax)
    d = (ch + 1.0)[None, :] * du_iso          # outward distance, channels nu+pad..
    t = tau[:, None] + d
    val = np.where(t < R[:, None],
                   2.0 * mu_water * np.sqrt(np.maximum(R[:, None] ** 2 - t ** 2, 0.0)),
                   0.0)
    out[:, pad + nu:] = val.astype(np.float32)
    return out


def _load_slab(sino_h5, geom_json, zgrid, z_offset_mm, slab_half):
    nz = int(geom_json["nz_rebinned"])
    nz_center = max(0, min(nz - 1, nz // 2 + int(round(z_offset_mm))))
    z_center = float(zgrid[nz_center])
    lo = max(0, nz_center - slab_half)
    hi = min(nz, nz_center + slab_half + 1)
    with h5py.File(sino_h5, "r") as f:
        slab = [np.asarray(f["sino"][:, :, j], dtype=np.float32)
                for j in range(lo, hi)]
    slab = [np.ascontiguousarray(np.flip(s, axis=-1)) for s in slab]
    return slab, z_center


def _fbp_slab(slab, geom_json, pixel_spacing, device, pad, mu_water, edge_k=7):
    """FBP-average a slab via the PRODUCTION projector.

    pad>0 -> PyronnFanBeamProjector(truncation=...) does the water-cylinder
    extrapolation + widened-detector back-projection internally, so this
    run exercises the same code path solvers will use. The reference numpy
    ``water_cylinder_extrapolate`` above is kept for documentation/parity.
    """
    rotview = int(geom_json["rotview"])
    nu = int(geom_json["nu"])
    angle_start = float(geom_json.get("angle_start_corrected", 0.0))
    angle_end = angle_start + 2.0 * math.pi
    fitted = FanBeamGeometry.mayo_ldct_fitted(
        n_angles=rotview, n_det=nu, angle_start=angle_start, angle_end=angle_end)
    geom = FanBeamGeometry(
        image_size=512, pixel_spacing=pixel_spacing, n_angles=rotview,
        n_det=nu, det_spacing=fitted.det_spacing,
        sod=fitted.sod, sdd=fitted.sdd,
        angle_start=angle_start, angle_end=angle_end)
    trunc = {"pad": pad, "mu_water": mu_water, "edge_k": edge_k} if pad > 0 else None
    proj = PyronnFanBeamProjector(
        geom, det_offset_mm=MAYO_LDCT_DET_OFFSET, truncation=trunc).to(device)
    fbps = []
    for s in slab:
        s_t = torch.from_numpy(np.ascontiguousarray(s)).to(device).float()[None, None]
        fbp_one = proj.fbp(s_t).detach()[0, 0].cpu().numpy()
        fbps.append(np.fliplr(np.flipud(fbp_one)))
    return np.ascontiguousarray(np.mean(np.stack(fbps, 0), 0))


def _cal(fbp_np, truth_t, dr, device):
    fbp_t = torch.from_numpy(fbp_np).to(device).float()[None, None].clamp(min=0.0)
    cal = evaluate_calibrated(fbp_t, truth_t, baseline=fbp_t,
                               display_min=0.0, display_max=dr, fov=False)
    return (cal["pred_cal"][0, 0].cpu().numpy().astype(np.float32),
            float(cal["val_ssim"]), float(cal["val_psnr"]), float(cal["val_rmse"]))


def _run_patient(patient, sino_dir, truth_root, args):
    out = {"patient": patient, "split": WAGNER_SPLIT_OF[patient]}
    t0 = time.time()
    sino_hd = sino_dir / f"{patient}_sino_fulldose.h5"
    sino_ld = sino_dir / f"{patient}_sino_lowdose.h5"
    g_hd = sino_dir / f"{patient}_sino_fulldose_geometry.json"
    g_ld = sino_dir / f"{patient}_sino_lowdose_geometry.json"
    z_hd = sino_dir / f"{patient}_sino_fulldose_z_grid.npy"
    z_ld = sino_dir / f"{patient}_sino_lowdose_z_grid.npy"
    for p in [sino_hd, sino_ld, g_hd, g_ld, z_hd, z_ld]:
        if not p.exists():
            return {"patient": patient, "split": out["split"],
                    "error": f"missing {p.name}"}
    geom_hd = json.loads(g_hd.read_text())
    geom_ld = json.loads(g_ld.read_text())
    zgrid_hd = np.load(z_hd)
    zgrid_ld = np.load(z_ld)

    slab_hd, zc_hd = _load_slab(sino_hd, geom_hd, zgrid_hd, args.z_offset_mm, args.slab_half)
    slab_ld, zc_ld = _load_slab(sino_ld, geom_ld, zgrid_ld, args.z_offset_mm, args.slab_half)

    truth_info = _load_truth_slice_for_z(truth_root / patient, zc_hd)
    if truth_info is None:
        return {"patient": patient, "split": out["split"], "error": "no truth"}
    truth, target_pz, bracket, tmeta = truth_info
    truth_ps = float(tmeta["pixel_spacing"])
    ps_eff = 0.700857 * (truth_ps / 0.703125)

    # isocenter channel sampling for the cylinder length unit
    fitted = FanBeamGeometry.mayo_ldct_fitted(n_angles=2, n_det=2)
    du_iso = fitted.det_spacing * fitted.sod / fitted.sdd

    dr = float(args.display_max)
    truth_t = torch.from_numpy(truth).to(args.device).float()[None, None]

    res = {}
    for dose, slab in [("hd", slab_hd), ("ld", slab_ld)]:
        for arm, pad in [("raw", 0), ("tc", args.pad)]:
            fbp = _fbp_slab(slab, geom_hd if dose == "hd" else geom_ld,
                            ps_eff, args.device, pad, du_iso, args.mu_water)
            cal_np, ssim, psnr, rmse = _cal(fbp, truth_t, dr, args.device)
            res[f"{dose}_{arm}"] = {"img": cal_np, "ssim_cal": ssim,
                                     "psnr_cal": psnr, "rmse_cal": rmse}

    out.update({
        "truth": truth.astype(np.float32),
        "truth_z_patient": float(target_pz),
        "truth_pixel_spacing": truth_ps,
        "truth_kernel": str(tmeta["recon_kernel"]),
        "metrics": {k: {kk: vv for kk, vv in v.items() if kk != "img"}
                    for k, v in res.items()},
        "imgs": {k: v["img"] for k, v in res.items()},
        "elapsed_s": time.time() - t0,
    })
    m = out["metrics"]
    print(f"[trunc] {patient} ({out['split']:5}) ps={truth_ps:.4f}  "
          f"HD raw {m['hd_raw']['ssim_cal']:.3f}/{m['hd_raw']['psnr_cal']:.1f}"
          f" -> tc {m['hd_tc']['ssim_cal']:.3f}/{m['hd_tc']['psnr_cal']:.1f}  "
          f"LD raw {m['ld_raw']['ssim_cal']:.3f} -> tc {m['ld_tc']['ssim_cal']:.3f}  "
          f"[{out['elapsed_s']:.0f}s]", flush=True)
    return out


def main() -> int:
    args = parse_args()
    root = Path(args.data_root or os.environ.get(
        "AGENT4CT_DATA", "/cluster/maier/Agent4CT/data")) / "mayo_ldct"
    sino_dir = root / os.environ.get("STAGED_HELIX2FAN_SUBDIR", "staged_helix2fan_v3")
    truth_root = root / "raw"
    out_dir = Path(args.out_dir) if args.out_dir else REPO / "results" / "mayo_debug"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[trunc] sino_dir={sino_dir} pad={args.pad} mu_water={args.mu_water}",
          flush=True)

    rows = []
    for pat in WAGNER_ALL:
        try:
            rows.append(_run_patient(pat, sino_dir, truth_root, args))
        except Exception as e:
            print(f"[trunc] {pat} FAILED: {e!r}", flush=True)
            rows.append({"patient": pat, "split": WAGNER_SPLIT_OF[pat],
                         "error": repr(e)})
    valid = [r for r in rows if "error" not in r]
    if not valid:
        print("[trunc] no valid rows", flush=True)
        return 1
    dr = float(args.display_max)
    ddr = 0.015  # diff window

    # ---- combined panel: GT | HD_tc | diff(HD_tc - GT) ----
    n = len(valid)
    fig, ax = plt.subplots(n, 3, figsize=(12, 3.0 * n))
    if n == 1:
        ax = ax[None, :]
    for i, r in enumerate(valid):
        m = r["metrics"]
        ax[i, 0].imshow(r["truth"], cmap="gray", vmin=0, vmax=dr)
        ax[i, 0].set_title(f"{r['patient']} ({r['split']}) GT  ps={r['truth_pixel_spacing']:.4f}", fontsize=9)
        ax[i, 1].imshow(r["imgs"]["hd_tc"], cmap="gray", vmin=0, vmax=dr)
        ax[i, 1].set_title(f"HD-FBP trunc-corr  SSIM={m['hd_tc']['ssim_cal']:.3f}  PSNR={m['hd_tc']['psnr_cal']:.1f}", fontsize=9)
        d = r["imgs"]["hd_tc"] - r["truth"]
        ax[i, 2].imshow(d, cmap="seismic", vmin=-ddr, vmax=ddr)
        ax[i, 2].set_title(f"HD_tc - GT  RMSE={m['hd_tc']['rmse_cal']:.4f}", fontsize=9)
        for j in range(3):
            ax[i, j].set_xticks([]); ax[i, j].set_yticks([])
    fig.suptitle(f"Wagner GT vs HD-FBP (truncation-corrected) + difference  "
                 f"[tag={args.tag}, pad={args.pad}]", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.99])
    panel = out_dir / f"wagner_gt_hd_ld_fbp_{args.tag}.png"
    fig.savefig(panel, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"[trunc] wrote {panel}", flush=True)

    # ---- per-patient: GT | HD_raw | HD_tc | dHD_tc | LD_tc | dLD_tc ----
    pp_dir = out_dir / f"wagner_per_patient_{args.tag}"
    pp_dir.mkdir(exist_ok=True)
    for r in valid:
        m = r["metrics"]; I = r["imgs"]
        fig, ax = plt.subplots(1, 6, figsize=(30, 5.2))
        panels = [
            (r["truth"], f"{r['patient']} GT", "gray", 0, dr),
            (I["hd_raw"], f"HD raw  SSIM={m['hd_raw']['ssim_cal']:.3f}\nPSNR={m['hd_raw']['psnr_cal']:.1f}", "gray", 0, dr),
            (I["hd_tc"], f"HD trunc-corr  SSIM={m['hd_tc']['ssim_cal']:.3f}\nPSNR={m['hd_tc']['psnr_cal']:.1f}", "gray", 0, dr),
            (I["hd_tc"] - r["truth"], f"HD_tc - GT\nRMSE={m['hd_tc']['rmse_cal']:.4f}", "seismic", -ddr, ddr),
            (I["ld_tc"], f"LD trunc-corr  SSIM={m['ld_tc']['ssim_cal']:.3f}\nPSNR={m['ld_tc']['psnr_cal']:.1f}", "gray", 0, dr),
            (I["ld_tc"] - r["truth"], f"LD_tc - GT\nRMSE={m['ld_tc']['rmse_cal']:.4f}", "seismic", -ddr, ddr),
        ]
        for a, (img, ttl, cm, vmn, vmx) in zip(ax, panels):
            a.imshow(img, cmap=cm, vmin=vmn, vmax=vmx); a.set_title(ttl, fontsize=10)
            a.set_xticks([]); a.set_yticks([])
        fig.suptitle(f"{r['patient']} ({r['split']})  truth_ps={r['truth_pixel_spacing']:.4f} mm  "
                     f"[v3 geometry, per-patient FOV, water-cylinder trunc-corr pad={args.pad}]", fontsize=11)
        fig.tight_layout(rect=[0, 0, 1, 0.97])
        fig.savefig(pp_dir / f"{r['patient']}.png", dpi=110, bbox_inches="tight")
        plt.close(fig)
    print(f"[trunc] wrote per-patient figures to {pp_dir}", flush=True)

    # ---- JSON + arrays ----
    def agg(key):
        s = np.array([r["metrics"][key]["ssim_cal"] for r in valid])
        p = np.array([r["metrics"][key]["psnr_cal"] for r in valid])
        return {"ssim_cal_mean": float(s.mean()), "psnr_cal_mean": float(p.mean()),
                "n": len(valid)}
    summary = {
        "tag": args.tag, "pad": args.pad, "mu_water": args.mu_water,
        "aggregates": {k: agg(k) for k in ["hd_raw", "hd_tc", "ld_raw", "ld_tc"]},
        "patients": [
            {"patient": r["patient"], "split": r["split"],
             **({"error": r["error"]} if "error" in r else {
                 "truth_pixel_spacing": r["truth_pixel_spacing"],
                 "truth_z_patient": r["truth_z_patient"],
                 "metrics": r["metrics"]})}
            for r in rows],
    }
    oj = out_dir / f"wagner_gt_hd_ld_fbp_{args.tag}.json"
    oj.write_text(json.dumps(summary, indent=2))
    print(f"[trunc] wrote {oj}", flush=True)
    arr_dir = out_dir / f"wagner_gt_hd_ld_fbp_{args.tag}_arrays"
    arr_dir.mkdir(exist_ok=True)
    for r in valid:
        np.savez(arr_dir / f"{r['patient']}.npz", truth=r["truth"],
                 **{k: v for k, v in r["imgs"].items()})
    print(f"[trunc] wrote arrays to {arr_dir}", flush=True)

    print("\n[trunc] AGGREGATE SSIM_cal / PSNR_cal:", flush=True)
    for k in ["hd_raw", "hd_tc", "ld_raw", "ld_tc"]:
        a = summary["aggregates"][k]
        print(f"  {k:7}: SSIM {a['ssim_cal_mean']:.4f}  PSNR {a['psnr_cal_mean']:.2f}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
