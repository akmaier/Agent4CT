#!/usr/bin/env python -u
"""v5 — transfer test: does the v4b projection-domain geometry beat v3 in recon?

Locks the ENTIRE geometry to the v4b forward-fit values (SLURM 763582,
results/breast_debug/L014_forward_geom_fit_v4.json) through the full
rebin + FBP stack and fits ONLY the spectral nuisances (slab profile,
H(rho) radial filter, post-FBP a/bg/hi). Same 10-slice / per-slice-L2
protocol as v3. Decision gate (findings.md 2026-06-12 v4 entry):
adopt v4 only if SSIM_mean >= v3's 0.9571 (PSNR 40.79).

Geometry application:
  * curved->flat rebin with du_eff = theta_p x sdd_v4 (hardware angular
    pitch), dv_v4 = 1.084862, sod/sdd = 591.851/1080.558. u0/v0 left at
    tags — the fitted u0_off = -3.378 ch decomposes into -3.25 ch of
    channel-flip bookkeeping (nu - 2*u0_tag, u0_tag = 369.625) that the
    pipeline already handles via its (nu - u0) convention, plus a
    genuine -0.128 ch (-0.164 mm) residual carried into the FBP as a
    detector offset (sign bracketed by arms A-/A+).
  * z: z_positions <- z_c + s_z*(z - z_c) + z0 (s_z = 1.001098,
    z0 = -0.117) folded in BEFORE the SSR; pitch_mm scaled by s_z.
  * SSR sod/sdd/du/dv frozen at the same v4 values (ONE geometry).

FBP arms (each gets its own nuisance fit):
  A-: v4-consistent FBP (sod/sdd = v4, det_spacing = du_eff,
      pixel_spacing = 0.703125 x s_xy = 0.699490), det_offset = -0.164 mm
  A+: same, det_offset = +0.164 mm (sign bracket)
  B : Powell FBP (mayo_ldct_fitted + MAYO_LDCT_DET_OFFSET) — tests
      "v4 rebin + legacy FBP"

Output:
    results/breast_debug/L014_locked_v5.json
    results/mayo_debug/L014_locked_v5.png
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
if str(REPO / "data") not in sys.path:
    sys.path.insert(0, str(REPO / "data"))

from ddssl_ldct.geometry import FanBeamGeometry, MAYO_LDCT_DET_OFFSET
from ddssl_ldct.pyronn_projector import PyronnFanBeamProjector
from ddssl_ldct.helix2fan import read_dicom_ctpd, rebin_curved_to_flat
from data.fetch_mayo_ldct import _find_projection_series

from scripts.fit_rebin_end2end_L014 import (
    _list_truth, _mu, precompute_picks, helical_ssr_torch,
    radial_filter_2d, calc_metrics,
)
from scripts.fit_rebin_end2end_L014_v2 import SLICE_INDICES

V3_BASELINE = {"ssim": 0.9571, "psnr": 40.79}


def load_v4() -> dict:
    p = REPO / "results" / "breast_debug" / "L014_forward_geom_fit_v4.json"
    blob = json.loads(p.read_text())
    return blob["fitted"]


def fit_arm(arm_name, fbp_geom, det_offset_mm, proj_flat, z_pos_eff,
            ffs_dz, indices, rotview, nu, nv, sod, sdd, du, dv,
            truth_stack, truth_list_np, truth_pZ, n_iters=1200) -> dict:
    dev = "cuda"
    proj_fbp = PyronnFanBeamProjector(fbp_geom).to(dev)
    proj_fbp._tensor_geom["detector_origin"] = (
        proj_fbp._tensor_geom["detector_origin"] + det_offset_mm
    )
    print(f"[v5:{arm_name}] FBP sod={fbp_geom.sod:.3f} sdd={fbp_geom.sdd:.3f} "
          f"ps={fbp_geom.pixel_spacing:.6f} ds={fbp_geom.det_spacing:.6f} "
          f"det_off={det_offset_mm:+.4f}", flush=True)

    N_GT = truth_stack.shape[0]
    slab_offsets_mm = [-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0]
    target_z = [-pz for pz in truth_pZ]

    alpha_dz = torch.tensor(1.0, device=dev)
    z_eff_full = z_pos_eff + alpha_dz * ffs_dz
    picks = [
        [precompute_picks(z_eff_full, indices, rotview, zt + off)
         for off in slab_offsets_mm]
        for zt in target_z
    ]

    sod_t = torch.tensor(sod, device=dev)
    sdd_t = torch.tensor(sdd, device=dev)
    du_t = torch.tensor(du, device=dev)
    dv_t = torch.tensor(dv, device=dev)
    u_c = (nu - 1) / 2.0
    v_c = (nv - 1) / 2.0

    delta_z = torch.nn.Parameter(torch.tensor(0.0, device=dev))
    w_logits = torch.nn.Parameter(torch.tensor(
        [-6.0, 0.0, 0.0, 0.0, 0.0, 0.0, -6.0], device=dev))
    h_radial = torch.nn.Parameter(torch.ones(64, device=dev))
    a = torch.nn.Parameter(torch.tensor(1.0, device=dev))
    bg = torch.nn.Parameter(torch.tensor(0.0, device=dev))
    hi = torch.nn.Parameter(torch.tensor(0.05, device=dev))

    fy = torch.fft.fftfreq(512, device=dev).float()
    fyy, fxx = torch.meshgrid(fy, fy, indexing="ij")
    rho = torch.sqrt(fyy ** 2 + fxx ** 2)

    def forward_gt(i):
        w = F.softmax(w_logits, dim=0)
        sino = None
        for kk, off in enumerate(slab_offsets_mm):
            z_eff = target_z[i] + off + delta_z
            s_k = helical_ssr_torch(
                proj_flat, z_eff_full, picks[i][kk], z_eff,
                sod_t, sdd_t, du_t, dv_t, u_c, v_c,
            )
            sino = (w[kk] * s_k) if sino is None else sino + w[kk] * s_k
        sino_in = torch.flip(sino, dims=[-1])[None, None]
        fbp = proj_fbp.fbp(sino_in, filter_name="ramlak")[0, 0]
        img = torch.flip(torch.flip(fbp, dims=[0]), dims=[1])
        fft = torch.fft.fft2(img)
        h2d = radial_filter_2d(h_radial, rho, 64)
        filt = torch.fft.ifft2(torch.complex(h2d * fft.real,
                                              h2d * fft.imag)).real
        out = F.relu(a * (filt - bg))
        return torch.minimum(out, hi)

    opt = torch.optim.Adam([delta_z, w_logits, h_radial, a, bg, hi], lr=2e-3)
    for it in range(n_iters):
        opt.zero_grad()
        losses = [F.mse_loss(forward_gt(i), truth_stack[i]) for i in range(N_GT)]
        data = torch.stack(losses).mean()
        smooth = ((h_radial[2:] - 2 * h_radial[1:-1] + h_radial[:-2]) ** 2).mean()
        (data + 1e-4 * smooth).backward()
        opt.step()
        if it % 200 == 0 or it == n_iters - 1:
            print(f"[v5:{arm_name}] it {it:4d} L2={float(data):.3e} "
                  f"dz={float(delta_z):+.4f} a={float(a):.3f} "
                  f"hi={float(hi):.4f}", flush=True)

    with torch.no_grad():
        per = []
        preds = []
        for i in range(N_GT):
            pred = forward_gt(i).cpu().numpy()
            preds.append(pred)
            per.append(calc_metrics(pred, truth_list_np[i], dr=0.05))
        ssim = np.array([m["ssim"] for m in per])
        psnr = np.array([m["psnr"] for m in per])
    print(f"[v5:{arm_name}] MEAN SSIM={ssim.mean():.4f} PSNR={psnr.mean():.2f} "
          f"(v3 baseline {V3_BASELINE['ssim']:.4f}/{V3_BASELINE['psnr']:.2f})",
          flush=True)
    return {
        "arm": arm_name,
        "det_offset_mm": det_offset_mm,
        "fbp": {"sod": float(fbp_geom.sod), "sdd": float(fbp_geom.sdd),
                 "ps": float(fbp_geom.pixel_spacing),
                 "ds": float(fbp_geom.det_spacing)},
        "per_gt_ssim": [float(x) for x in ssim],
        "per_gt_psnr": [float(x) for x in psnr],
        "ssim_mean": float(ssim.mean()),
        "psnr_mean": float(psnr.mean()),
        "nuisances": {
            "delta_z": float(delta_z), "a": float(a), "bg": float(bg),
            "hi": float(hi),
            "w_slab": [float(x) for x in F.softmax(w_logits, 0).cpu()],
            "h_radial": [float(x) for x in h_radial.detach().cpu()],
        },
        "preds_central": preds[len(preds) // 2],
    }


def main() -> int:
    import os
    v4 = load_v4()
    print(f"[v5] v4b geometry: {json.dumps({k: round(v, 6) for k, v in v4.items()})}",
          flush=True)
    root = Path(os.environ.get("AGENT4CT_DATA",
                                "/cluster/maier/Agent4CT/data")) / "mayo_ldct"
    raw = root / "raw" / "L014"

    series = _find_projection_series(raw, "full dose projections")
    print(f"[v5] reading {series} ...", flush=True)
    proj_curved, geom = read_dicom_ctpd(series)
    theta_p = float(geom["du"]) / float(geom["sdd"])
    du_eff = theta_p * v4["sdd"]
    print(f"[v5] tags: sod={geom['sod']} sdd={geom['sdd']} du={geom['du']:.6f} "
          f"dv={geom['dv']:.6f} u0={geom['u0']} v0={geom['v0']} "
          f"pitch={geom['pitch_mm']:.4f}", flush=True)
    print(f"[v5] v4 overrides: sod={v4['sod']:.3f} sdd={v4['sdd']:.3f} "
          f"du_eff={du_eff:.6f} dv={v4['dv']:.6f} s_z={v4['s_z']:.6f} "
          f"z0={v4['z0']:+.3f}", flush=True)

    # ---- v4-consistent curved->flat ---------------------------------------
    geom["sod"] = float(v4["sod"])
    geom["sdd"] = float(v4["sdd"])
    geom["du"] = float(du_eff)
    geom["dv"] = float(v4["dv"])
    proj_flat_np = rebin_curved_to_flat(proj_curved, geom, n_jobs=-1)
    del proj_curved
    print(f"[v5] proj_flat {proj_flat_np.shape}", flush=True)

    # ---- v4 z-axis ---------------------------------------------------------
    z_pos = np.asarray(geom["z_positions"], np.float64)
    z_c = float(z_pos.mean())
    z_pos_v4 = z_c + float(v4["s_z"]) * (z_pos - z_c) + float(v4["z0"])
    pitch_v4 = float(geom["pitch_mm"]) * float(v4["s_z"])
    rotview = int(round(proj_flat_np.shape[0] / geom["total_rotations"]))
    nu, nv = proj_flat_np.shape[2], proj_flat_np.shape[1]
    n_proj = proj_flat_np.shape[0]
    print(f"[v5] rotview={rotview} nu={nu} nv={nv} pitch_v4={pitch_v4:.4f}",
          flush=True)

    dev = "cuda"
    proj_flat = torch.from_numpy(proj_flat_np.astype(np.float32)).to(dev)
    del proj_flat_np
    z_pos_t = torch.from_numpy(z_pos_v4).to(dev)
    ffs_dz = torch.from_numpy(np.asarray(
        geom.get("ffs_dz", np.zeros(n_proj)), np.float64)).to(dev)
    indices = torch.arange(n_proj, dtype=torch.int64).to(dev)

    # ---- truth -------------------------------------------------------------
    truth_files = _list_truth(raw)
    truth_files.sort(key=lambda t: t[0])
    gt_idx = [i for i in SLICE_INDICES if i < len(truth_files)]
    truth_list_np, truth_pZ = [], []
    for ti in gt_idx:
        pz, fp = truth_files[ti]
        mu, _ = _mu(fp)
        truth_list_np.append(mu)
        truth_pZ.append(pz)
    truth_stack = torch.stack(
        [torch.from_numpy(x).float() for x in truth_list_np]).to(dev)
    print(f"[v5] N_GT={len(gt_idx)} indices={gt_idx}", flush=True)

    angle_start = float(geom.get("angle_start_corrected", 0.0))
    angle_end = angle_start + 2 * math.pi

    # residual detector offset (post flip-decomposition; see docstring)
    resid_off = abs(0.128 * du_eff)

    arms_cfg = [
        ("A_minus", FanBeamGeometry(
            image_size=512, pixel_spacing=0.703125 * float(v4["s_xy"]),
            n_angles=rotview, n_det=nu, det_spacing=du_eff,
            sod=float(v4["sod"]), sdd=float(v4["sdd"]),
            angle_start=angle_start, angle_end=angle_end), -resid_off),
        ("A_plus", FanBeamGeometry(
            image_size=512, pixel_spacing=0.703125 * float(v4["s_xy"]),
            n_angles=rotview, n_det=nu, det_spacing=du_eff,
            sod=float(v4["sod"]), sdd=float(v4["sdd"]),
            angle_start=angle_start, angle_end=angle_end), +resid_off),
        ("B_powell", FanBeamGeometry.mayo_ldct_fitted(
            n_angles=rotview, n_det=nu,
            angle_start=angle_start, angle_end=angle_end),
         MAYO_LDCT_DET_OFFSET),
    ]

    results = []
    for name, fbp_geom, off in arms_cfg:
        res = fit_arm(name, fbp_geom, off, proj_flat, z_pos_t, ffs_dz,
                      indices, rotview, nu, nv,
                      float(v4["sod"]), float(v4["sdd"]), du_eff,
                      float(v4["dv"]),
                      truth_stack, truth_list_np, truth_pZ)
        results.append(res)

    best = max(results, key=lambda r: r["ssim_mean"])
    verdict = ("ADOPT v4 candidate" if best["ssim_mean"] >= V3_BASELINE["ssim"]
               else "KEEP v3 (v4 transfer below baseline)")
    print(f"[v5] BEST arm {best['arm']}: SSIM {best['ssim_mean']:.4f} "
          f"PSNR {best['psnr_mean']:.2f} → {verdict}", flush=True)

    out = {
        "v4_geometry": v4,
        "du_eff": float(du_eff),
        "pitch_v4": pitch_v4,
        "v3_baseline": V3_BASELINE,
        "arms": [{k: v for k, v in r.items() if k != "preds_central"}
                  for r in results],
        "best_arm": best["arm"],
        "verdict": verdict,
        "slice_indices": gt_idx,
        "truth_pZ": [float(z) for z in truth_pZ],
    }
    oj = REPO / "results" / "breast_debug" / "L014_locked_v5.json"
    oj.parent.mkdir(parents=True, exist_ok=True)
    oj.write_text(json.dumps(out, indent=2))
    print(f"[v5] wrote {oj}", flush=True)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    pZ = truth_pZ
    for r in results:
        axes[0].plot(pZ, r["per_gt_ssim"], marker="o", label=r["arm"])
        axes[1].plot(pZ, r["per_gt_psnr"], marker="o", label=r["arm"])
    axes[0].axhline(V3_BASELINE["ssim"], ls="--", c="k", label="v3 mean")
    axes[1].axhline(V3_BASELINE["psnr"], ls="--", c="k", label="v3 mean")
    axes[0].set_title("per-GT SSIM"); axes[0].set_xlabel("patient z (mm)")
    axes[1].set_title("per-GT PSNR (dB)"); axes[1].set_xlabel("patient z (mm)")
    axes[0].legend(); axes[1].legend()
    mid = len(gt_idx) // 2
    axes[2].imshow(best["preds_central"], cmap="gray", vmin=0, vmax=0.05)
    axes[2].set_title(f"best arm ({best['arm']}) central GT #{gt_idx[mid]}\n"
                      f"SSIM={best['per_gt_ssim'][mid]:.4f}")
    axes[2].set_xticks([]); axes[2].set_yticks([])
    fig.suptitle(f"v5 locked-geometry transfer test — best {best['arm']}: "
                 f"SSIM {best['ssim_mean']:.4f} / PSNR {best['psnr_mean']:.2f} "
                 f"vs v3 {V3_BASELINE['ssim']:.4f}/{V3_BASELINE['psnr']:.2f} "
                 f"→ {verdict}")
    fig.tight_layout()
    op = REPO / "results" / "mayo_debug" / "L014_locked_v5.png"
    op.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(op, dpi=110, bbox_inches="tight")
    print(f"[v5] wrote {op}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
