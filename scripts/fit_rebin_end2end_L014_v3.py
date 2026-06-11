#!/usr/bin/env python -u
"""v3 fit: v2 + learnable z-scaling s_z folded into Adam.

Adds a single scalar s_z (init 1.0) that multiplies the per-readout
source-z BEFORE the α_dz FFS shift:

    z_pos_eff = s_z · z_pos_sub + α_dz · ffs_dz

s_z is equivalent at leading order to a global pitch_mm correction
(mechanism A) or a global dv correction (mechanism B); see the
2026-06-11 entry in docs/findings.md for the physics.

Helix-index picks are non-differentiable nearest-integer lookups; we
re-precompute them every PICKS_REFRESH iters so they don't drift more
than a couple of rows from the current s_z. SSR sampling on the picked
readouts IS differentiable in s_z through v_precise.

Everything else mirrors v2: same 10-uniform GT sampling, same per-slice
L2 mean (no FoV mask), same FBP-fixed / SSR-fit split.

Output:
    results/breast_debug/L014_rebin_end2end_fit_v3.json
    results/mayo_debug/L014_rebin_end2end_fit_v3.png
"""
from __future__ import annotations
import math
import sys
import json
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

from ddssl_ldct.geometry import FanBeamGeometry, MAYO_LDCT_DET_OFFSET
from ddssl_ldct.pyronn_projector import PyronnFanBeamProjector
from ddssl_ldct.metrics import intensity_calibrate

from scripts.fit_rebin_end2end_L014 import (
    _list_truth, _mu, precompute_picks, helical_ssr_torch,
    radial_filter_2d, calc_metrics,
)
from scripts.fit_rebin_end2end_L014_v2 import SLICE_INDICES


PICKS_REFRESH = 100   # every N Adam iters, recompute picks at current s_z


def main() -> int:
    import os
    root = Path(os.environ.get("AGENT4CT_DATA",
                                "/cluster/maier/Agent4CT/data")) / "mayo_ldct"
    sino_dir = root / "staged_helix2fan"
    raw_dir = root / "raw" / "L014"

    blob_path = sino_dir / "L014_proj_flat_full.pt"
    if not blob_path.exists():
        print(f"[fit-v3] missing {blob_path}", file=sys.stderr); return 2
    print(f"[fit-v3] loading {blob_path} …", flush=True)
    blob = torch.load(blob_path, weights_only=False, map_location="cpu")
    proj_flat = blob["proj_flat"].to("cuda")
    z_pos_sub = blob["z_positions"].to("cuda")
    orig_idx = blob["original_indices"].to("cuda")
    ffs_dz_sub = blob.get("ffs_dz",
                          torch.zeros_like(z_pos_sub)).to("cuda")
    rotview = int(blob["rotview"])
    nu, nv = int(blob["nu"]), int(blob["nv"])
    angle_start = float(blob["angle_start_corrected"])

    # Truth slices (same 10 indices as v2)
    truth_files = _list_truth(raw_dir)
    truth_files.sort(key=lambda t: t[0])
    gt_indices = [i for i in SLICE_INDICES if i < len(truth_files)]
    truth_list_np = []; truth_pZ_list = []
    pixel_sp_dicom = None
    for ti in gt_indices:
        pZ_i, fp_i = truth_files[ti]
        mu_i, ds_i = _mu(fp_i)
        truth_list_np.append(mu_i); truth_pZ_list.append(pZ_i)
        if pixel_sp_dicom is None:
            pixel_sp_dicom = float(ds_i.PixelSpacing[0])
    truth_stack = torch.stack(
        [torch.from_numpy(x).to("cuda").float() for x in truth_list_np],
        dim=0,
    )
    N_GT = len(gt_indices)
    print(f"[fit-v3] N_GT={N_GT}  slice indices={gt_indices}", flush=True)

    # FBP geometry (HELD FIXED at Powell)
    fbp_geom = FanBeamGeometry.mayo_ldct_fitted(
        n_angles=rotview, n_det=nu,
        angle_start=angle_start, angle_end=angle_start + 2 * math.pi,
    )
    proj_fbp = PyronnFanBeamProjector(fbp_geom).to("cuda")
    proj_fbp._tensor_geom["detector_origin"] = (
        proj_fbp._tensor_geom["detector_origin"] + MAYO_LDCT_DET_OFFSET
    )
    print(f"[fit-v3] FBP geom (FIXED): sod={fbp_geom.sod:.3f}  "
          f"sdd={fbp_geom.sdd:.3f}  pixel_spacing={fbp_geom.pixel_spacing}  "
          f"det_spacing={fbp_geom.det_spacing}  "
          f"det_offset={MAYO_LDCT_DET_OFFSET:+.4f}", flush=True)

    # Learnable parameters — v2 set + s_z
    sod = torch.nn.Parameter(torch.tensor(float(blob["sod"]),
                                            device="cuda"))
    sdd = torch.nn.Parameter(torch.tensor(float(blob["sdd"]),
                                            device="cuda"))
    du = torch.tensor(float(blob["du"]), device="cuda")
    dv = torch.tensor(float(blob["dv"]), device="cuda")
    s_z = torch.nn.Parameter(torch.tensor(1.0, device="cuda"))    # NEW
    delta_z = torch.nn.Parameter(torch.tensor(0.0, device="cuda"))
    n_bins = 64
    h_radial = torch.nn.Parameter(torch.ones(n_bins, device="cuda"))
    a = torch.nn.Parameter(torch.tensor(1.0, device="cuda"))
    bg = torch.nn.Parameter(torch.tensor(0.0, device="cuda"))
    hi = torch.nn.Parameter(torch.tensor(0.05, device="cuda"))
    alpha_dz = torch.tensor(1.0, device="cuda")

    u_centre_nom = (nu - 1) / 2.0
    v_centre_nom = (nv - 1) / 2.0
    slab_offsets_mm = [-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0]
    n_slab = len(slab_offsets_mm)
    init_logits = torch.tensor([-6.0, 0.0, 0.0, 0.0, 0.0, 0.0, -6.0],
                                device="cuda", dtype=torch.float32)
    w_slab_logits = torch.nn.Parameter(init_logits.clone())
    target_source_z_per_gt = [-z for z in truth_pZ_list]

    def refresh_picks(s_z_val: float):
        """Recompute helix-index picks at current s_z value."""
        z_for_picks = s_z_val * z_pos_sub + alpha_dz * ffs_dz_sub
        out = []
        for z_tgt in target_source_z_per_gt:
            per_slab = []
            for off in slab_offsets_mm:
                per_slab.append(precompute_picks(
                    z_for_picks, orig_idx, rotview, z_tgt + off))
            out.append(per_slab)
        return out

    print(f"[fit-v3] initial picks at s_z=1.0 …", flush=True)
    picks = refresh_picks(1.0)

    fy = torch.fft.fftfreq(512, device="cuda").float()
    fx = torch.fft.fftfreq(512, device="cuda").float()
    fyy, fxx = torch.meshgrid(fy, fx, indexing="ij")
    rho = torch.sqrt(fyy ** 2 + fxx ** 2)
    dr = 0.05

    def _forward_one_gt(i_gt: int):
        w_slab = F.softmax(w_slab_logits, dim=0)
        sino_slab = None
        target_z_i = target_source_z_per_gt[i_gt]
        picks_i = picks[i_gt]
        # IMPORTANT: gradient through s_z flows via z_pos_eff[picks].
        z_pos_eff = s_z * z_pos_sub + alpha_dz * ffs_dz_sub
        for k, off in enumerate(slab_offsets_mm):
            z_eff = target_z_i + off + delta_z
            sino_k = helical_ssr_torch(
                proj_flat, z_pos_eff, picks_i[k], z_eff,
                sod, sdd, du, dv, u_centre_nom, v_centre_nom,
            )
            sino_slab = (w_slab[k] * sino_k) if sino_slab is None else (
                sino_slab + w_slab[k] * sino_k
            )
        sino_input = torch.flip(sino_slab, dims=[-1])[None, None]
        fbp_out = proj_fbp.fbp(sino_input, filter_name="ramlak")[0, 0]
        fbp_2d = torch.flip(torch.flip(fbp_out, dims=[0]), dims=[1])
        fft_fbp = torch.fft.fft2(fbp_2d)
        h_2d = radial_filter_2d(h_radial, rho, n_bins)
        filt_fft = torch.complex(h_2d * fft_fbp.real, h_2d * fft_fbp.imag)
        filt = torch.fft.ifft2(filt_fft).real
        scaled = a * (filt - bg)
        clipped = F.relu(scaled)
        clipped = torch.minimum(clipped, hi)
        return clipped

    opt = torch.optim.Adam(
        [sod, sdd, s_z, delta_z, w_slab_logits, h_radial, a, bg, hi],
        lr=2e-3,
    )
    n_iters = 1500
    log_every = max(1, n_iters // 30)
    lam_h = 1e-4

    print(f"[fit-v3] starting Adam, {n_iters} iters, lr=2e-3, "
          f"picks_refresh every {PICKS_REFRESH} iters", flush=True)
    print(f"[fit-v3] init  sod={sod.item():.3f}  sdd={sdd.item():.3f}  "
          f"s_z={s_z.item():.6f}", flush=True)
    hist = []
    for it in range(n_iters):
        if it > 0 and it % PICKS_REFRESH == 0:
            picks = refresh_picks(float(s_z.item()))
        opt.zero_grad()
        per_slice_l2 = []
        for i_gt in range(N_GT):
            pred_i = _forward_one_gt(i_gt)
            l2_i = F.mse_loss(pred_i, truth_stack[i_gt], reduction="mean")
            per_slice_l2.append(l2_i)
        data_loss = torch.stack(per_slice_l2).mean()
        smooth_loss = ((h_radial[2:] - 2 * h_radial[1:-1] +
                        h_radial[:-2]) ** 2).mean()
        total = data_loss + lam_h * smooth_loss
        total.backward()
        opt.step()
        if it % log_every == 0 or it == n_iters - 1:
            with torch.no_grad():
                per_slice_arr = torch.stack(per_slice_l2).cpu().numpy()
                w_show = F.softmax(w_slab_logits, dim=0).cpu().numpy()
            print(f"[fit-v3] iter {it:4d}/{n_iters}  "
                  f"L2_mean={data_loss.item():.3e}  "
                  f"sod={sod.item():.3f}  sdd={sdd.item():.3f}  "
                  f"s_z={s_z.item():.6f}  Δz={delta_z.item():+.4f}  "
                  f"a={a.item():.3f} bg={bg.item():+.5f} hi={hi.item():.4f}  "
                  f"|h|=[{h_radial.min().item():.3f},{h_radial.max().item():.3f}]  "
                  f"w_slab=[{','.join(f'{w:.2f}' for w in w_show)}]",
                  flush=True)
            hist.append({"iter": it, "loss": float(data_loss.item()),
                         "s_z": float(s_z.item()),
                         "sod": float(sod.item()),
                         "sdd": float(sdd.item()),
                         "delta_z": float(delta_z.item())})

    # Final picks at final s_z
    picks = refresh_picks(float(s_z.item()))

    # ---- Final per-slice metrics ----
    with torch.no_grad():
        preds_np = []
        per_gt_metrics = []
        for i_gt in range(N_GT):
            pred_i = _forward_one_gt(i_gt)
            pred_np = pred_i.cpu().numpy()
            preds_np.append(pred_np)
            m = calc_metrics(pred_np, truth_list_np[i_gt], dr=dr)
            per_gt_metrics.append(m)
        ssim_arr = np.array([m["ssim"] for m in per_gt_metrics])
        psnr_arr = np.array([m["psnr"] for m in per_gt_metrics])
        rmse_arr = np.array([m["rmse"] for m in per_gt_metrics])
        print(f"[fit-v3] FINAL per-GT (raw):", flush=True)
        for k, (idx, m_k) in enumerate(zip(gt_indices, per_gt_metrics)):
            print(f"[fit-v3]   GT #{idx:3d} (pZ={truth_pZ_list[k]:+.2f}): "
                  f"SSIM={m_k['ssim']:.4f}  PSNR={m_k['psnr']:6.2f}  "
                  f"RMSE={m_k['rmse']:.5f}", flush=True)
        print(f"[fit-v3] MEAN: SSIM={ssim_arr.mean():.4f}  "
              f"PSNR={psnr_arr.mean():.2f}  RMSE={rmse_arr.mean():.5f}",
              flush=True)

        # Post-hoc intensity_calibrate
        per_gt_cal_metrics = []
        for i_gt in range(N_GT):
            pred_t = torch.from_numpy(preds_np[i_gt]).to("cuda").float()
            truth_t = truth_stack[i_gt]
            pred_cal = intensity_calibrate(pred_t, truth_t, display_max=dr)
            m_cal = calc_metrics(pred_cal.cpu().numpy(),
                                  truth_list_np[i_gt], dr=dr)
            per_gt_cal_metrics.append(m_cal)
        ssim_cal_arr = np.array([m["ssim"] for m in per_gt_cal_metrics])
        psnr_cal_arr = np.array([m["psnr"] for m in per_gt_cal_metrics])
        rmse_cal_arr = np.array([m["rmse"] for m in per_gt_cal_metrics])
        print(f"[fit-v3] FINAL per-GT (post-hoc cal):", flush=True)
        for k, (idx, m_k) in enumerate(zip(gt_indices, per_gt_cal_metrics)):
            print(f"[fit-v3]   GT #{idx:3d}: SSIM={m_k['ssim']:.4f}  "
                  f"PSNR={m_k['psnr']:6.2f}  RMSE={m_k['rmse']:.5f}",
                  flush=True)
        print(f"[fit-v3] MEAN_cal: SSIM={ssim_cal_arr.mean():.4f}  "
              f"PSNR={psnr_cal_arr.mean():.2f}  "
              f"RMSE={rmse_cal_arr.mean():.5f}", flush=True)

    # ---- Save ----
    out_json = REPO / "results" / "breast_debug" / "L014_rebin_end2end_fit_v3.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_blob = {
        "rebin_fitted": {
            "sod": float(sod.item()),
            "sdd": float(sdd.item()),
            "du": float(du.item()),
            "dv": float(dv.item()),
            "s_z": float(s_z.item()),
        },
        "slab_fitted": {
            "delta_z_mm": float(delta_z.item()),
            "w_slab_logits": [float(x) for x in w_slab_logits.detach().cpu().numpy()],
            "w_slab": [float(x) for x in F.softmax(w_slab_logits, dim=0).detach().cpu().numpy()],
            "slab_offsets_mm": slab_offsets_mm,
            "alpha_dz_fixed": float(alpha_dz.item()),
        },
        "post_fbp_fitted": {
            "a": float(a.item()),
            "bg": float(bg.item()),
            "hi": float(hi.item()),
            "h_radial": [float(x) for x in h_radial.detach().cpu().numpy()],
        },
        "fbp_held_fixed": {
            "sod": float(fbp_geom.sod),
            "sdd": float(fbp_geom.sdd),
            "pixel_spacing": float(fbp_geom.pixel_spacing),
            "det_spacing": float(fbp_geom.det_spacing),
            "det_offset_mm": float(MAYO_LDCT_DET_OFFSET),
        },
        "slice_sampling": {
            "indices": gt_indices,
            "patient_z": [float(z) for z in truth_pZ_list],
            "n_total_slices": len(truth_files),
        },
        "loss": {
            "form": "mean of per-slice L2 over the full 512², no FoV mask",
            "n_iters": n_iters,
            "lr": 2e-3,
            "lam_h_smooth": lam_h,
            "picks_refresh_every": PICKS_REFRESH,
            "history": hist,
        },
        "final_metrics_raw": {
            "per_gt_ssim": [float(x) for x in ssim_arr],
            "per_gt_psnr": [float(x) for x in psnr_arr],
            "per_gt_rmse": [float(x) for x in rmse_arr],
            "ssim_mean": float(ssim_arr.mean()),
            "psnr_mean": float(psnr_arr.mean()),
            "rmse_mean": float(rmse_arr.mean()),
        },
        "final_metrics_intensity_calibrated_posthoc": {
            "per_gt_ssim": [float(x) for x in ssim_cal_arr],
            "per_gt_psnr": [float(x) for x in psnr_cal_arr],
            "per_gt_rmse": [float(x) for x in rmse_cal_arr],
            "ssim_mean": float(ssim_cal_arr.mean()),
            "psnr_mean": float(psnr_cal_arr.mean()),
            "rmse_mean": float(rmse_cal_arr.mean()),
        },
    }
    out_json.write_text(json.dumps(out_blob, indent=2))
    print(f"[fit-v3] wrote {out_json}", flush=True)

    # Diagnostic PNG
    out_png = REPO / "results" / "mayo_debug" / "L014_rebin_end2end_fit_v3.png"
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(N_GT, 3, figsize=(12, 3 * N_GT))
    for k in range(N_GT):
        t = truth_list_np[k]; p = preds_np[k]; d = p - t
        axes[k, 0].imshow(t, cmap="gray", vmin=0, vmax=dr)
        axes[k, 0].set_title(
            f"GT #{gt_indices[k]} (pZ={truth_pZ_list[k]:+.1f})", fontsize=9)
        axes[k, 1].imshow(p, cmap="gray", vmin=0, vmax=dr)
        axes[k, 1].set_title(
            f"pred  SSIM={per_gt_metrics[k]['ssim']:.4f}  "
            f"PSNR={per_gt_metrics[k]['psnr']:.2f}", fontsize=9)
        axes[k, 2].imshow(d, cmap="seismic", vmin=-0.01, vmax=0.01)
        axes[k, 2].set_title(
            f"diff RMSE={per_gt_metrics[k]['rmse']:.5f}", fontsize=9)
        for j in range(3):
            axes[k, j].set_xticks([]); axes[k, j].set_yticks([])
    fig.suptitle(
        f"L014 fit v3 (v2 + learnable s_z = {s_z.item():.6f}) — "
        f"SSIM_mean={ssim_arr.mean():.4f}  PSNR_mean={psnr_arr.mean():.2f}",
        fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.99])
    fig.savefig(out_png, dpi=110, bbox_inches="tight")
    print(f"[fit-v3] wrote {out_png}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
