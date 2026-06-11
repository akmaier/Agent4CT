#!/usr/bin/env python -u
"""End-to-end Adam fit of the helical→fan SSR + image-domain filter +
post-FBP scaling against 10 GT slices sampled UNIFORMLY across L014's
full 154-slice 'Full Dose Images' series (vs the central 10 used by
fit_rebin_end2end_L014.py, SLURM 762369).

Loss form: MEAN of per-slice L2 (full 512², NO FoV mask).
Compare to the v1 script which uses FoV-masked sum-over-stack.

Coupling-respecting fix-list (per docs/findings.md):

  HELD FIXED (no gradient):
    FBP geometry: sod_FBP = 595.362, sdd_FBP = 1086.803, pixel_spacing
        = 0.700857, det_spacing = 1.285044, det_offset = -0.0397 mm
        — Powell-fitted (job 762284); PYRO-NN back-projection is not
        differentiable in geometry so these stay fixed.
    Hardware: du = 1.28584, dv = 1.09472 (DICOM tags).
    FFS: alpha_dz = +1, alpha_drho = 0, alpha_dphi = 0.
    Rigid 2D align: none.
    Rebin: angle_start_corrected from blob.

  FIT, all shared across the 10 slices:
    SSR geometry: sod_SSR, sdd_SSR (init from DICOM 595.0, 1085.6).
    Delta z: delta_z (sub-mm slab anchor).
    Slab profile: w_slab_logits (7 bins → softmax).
    Filter: h_radial (64 bins).
    Post-FBP: a, bg, hi.

Slice sampling: indices [7, 23, 39, 55, 71, 87, 103, 119, 135, 151]
of the 154 truth files. These span patient-z ≈ [-462, -36] mm.

Output:
    results/breast_debug/L014_rebin_end2end_fit_v2.json
    results/mayo_debug/L014_rebin_end2end_fit_v2.png
"""
from __future__ import annotations
import math
import sys
import json
from pathlib import Path

import numpy as np
import pydicom
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

# Re-use the v1 helpers — same SSR, same picks, same filter.
from scripts.fit_rebin_end2end_L014 import (
    _list_truth, _mu, precompute_picks, helical_ssr_torch,
    radial_filter_2d, calc_metrics,
)


SLICE_INDICES = [7, 23, 39, 55, 71, 87, 103, 119, 135, 151]  # of 154


def main() -> int:
    import os
    root = Path(os.environ.get("AGENT4CT_DATA",
                                "/cluster/maier/Agent4CT/data")) / "mayo_ldct"
    sino_dir = root / "staged_helix2fan"
    raw_dir = root / "raw" / "L014"

    blob_path = sino_dir / "L014_proj_flat_full.pt"
    if not blob_path.exists():
        print(f"[fit-v2] missing {blob_path} — run "
              f"cache_proj_flat_L014_full.py first", file=sys.stderr)
        return 2
    print(f"[fit-v2] loading {blob_path} …", flush=True)
    blob = torch.load(blob_path, weights_only=False, map_location="cpu")

    proj_flat = blob["proj_flat"].to("cuda")
    z_pos_sub = blob["z_positions"].to("cuda")
    orig_idx = blob["original_indices"].to("cuda")
    ffs_dz_sub = blob.get("ffs_dz",
                          torch.zeros_like(z_pos_sub)).to("cuda")
    rotview = int(blob["rotview"])
    nu, nv = int(blob["nu"]), int(blob["nv"])
    pitch_mm = float(blob["pitch_mm"])
    angle_start = float(blob["angle_start_corrected"])
    print(f"[fit-v2] proj_flat shape={tuple(proj_flat.shape)}  "
          f"z range=[{float(z_pos_sub.min()):.2f}, "
          f"{float(z_pos_sub.max()):.2f}] mm", flush=True)

    # Truth slices
    truth_files = _list_truth(raw_dir)
    if len(truth_files) != 154:
        print(f"[fit-v2] WARN: expected 154 truth files, got {len(truth_files)}",
              flush=True)
    truth_files.sort(key=lambda t: t[0])  # by patient-z ascending
    gt_indices = [i for i in SLICE_INDICES if i < len(truth_files)]
    if len(gt_indices) != 10:
        print(f"[fit-v2] picked {len(gt_indices)} slices "
              f"(some indices out of bounds)", flush=True)
    N_GT = len(gt_indices)

    truth_list_np = []
    truth_pZ_list = []
    pixel_sp_dicom = None
    for ti in gt_indices:
        pZ_i, fp_i = truth_files[ti]
        mu_i, ds_i = _mu(fp_i)
        truth_list_np.append(mu_i)
        truth_pZ_list.append(pZ_i)
        if pixel_sp_dicom is None:
            pixel_sp_dicom = float(ds_i.PixelSpacing[0])
    # Verify each picked target z lies inside the cached helical sweep.
    # In source frame that means -pZ ∈ [z_pos_sub.min, z_pos_sub.max]
    for k, pZ_k in enumerate(truth_pZ_list):
        src_z_k = -pZ_k
        if not (float(z_pos_sub.min()) <= src_z_k <= float(z_pos_sub.max())):
            print(f"[fit-v2] WARN: GT #{gt_indices[k]} at pZ={pZ_k:.2f} "
                  f"(src_z={src_z_k:.2f}) is OUTSIDE the cached helix range",
                  flush=True)

    pixel_sp = 0.700857   # FBP grid (mayo_ldct_fitted; held fixed)
    truth_stack = torch.stack(
        [torch.from_numpy(x).to("cuda").float() for x in truth_list_np],
        dim=0,
    )
    print(f"[fit-v2] N_GT={N_GT}  slice indices={gt_indices}", flush=True)
    for k, (idx, z) in enumerate(zip(gt_indices, truth_pZ_list)):
        print(f"[fit-v2]   GT #{idx:3d}: pZ={z:+8.2f} mm", flush=True)
    print(f"[fit-v2] pixel_sp_FBP={pixel_sp:.6f} (held fixed)  "
          f"pixel_sp_DICOM_truth={pixel_sp_dicom:.6f}", flush=True)

    # ---- FBP geometry: HELD FIXED at Powell (job 762284 == mayo_ldct_fitted defaults) ----
    fbp_geom = FanBeamGeometry.mayo_ldct_fitted(
        n_angles=rotview, n_det=nu,
        angle_start=angle_start, angle_end=angle_start + 2 * math.pi,
    )
    proj_fbp = PyronnFanBeamProjector(fbp_geom).to("cuda")
    proj_fbp._tensor_geom["detector_origin"] = (
        proj_fbp._tensor_geom["detector_origin"] + MAYO_LDCT_DET_OFFSET
    )
    print(f"[fit-v2] FBP geom (FIXED): sod={fbp_geom.sod:.3f}  "
          f"sdd={fbp_geom.sdd:.3f}  pixel_spacing={fbp_geom.pixel_spacing}  "
          f"det_spacing={fbp_geom.det_spacing}  "
          f"det_offset={MAYO_LDCT_DET_OFFSET:+.4f} mm", flush=True)

    # ---- Learnable SSR + post-FBP parameters ----
    sod = torch.nn.Parameter(torch.tensor(float(blob["sod"]),
                                            device="cuda"))
    sdd = torch.nn.Parameter(torch.tensor(float(blob["sdd"]),
                                            device="cuda"))
    du = torch.tensor(float(blob["du"]), device="cuda")   # FIXED (hardware)
    dv = torch.tensor(float(blob["dv"]), device="cuda")   # FIXED (hardware)
    print(f"[fit-v2] SSR init: sod={sod.item():.3f}  sdd={sdd.item():.3f}  "
          f"du={float(du):.5f}(FIXED)  dv={float(dv):.5f}(FIXED)", flush=True)

    n_bins = 64
    h_radial = torch.nn.Parameter(torch.ones(n_bins, device="cuda"))
    a = torch.nn.Parameter(torch.tensor(1.0, device="cuda"))
    bg = torch.nn.Parameter(torch.tensor(0.0, device="cuda"))
    hi = torch.nn.Parameter(torch.tensor(0.05, device="cuda"))
    alpha_dz = torch.tensor(1.0, device="cuda")  # FIXED at ablation winner

    # Geometric centres (PYRO-NN convention)
    u_centre_nom = (nu - 1) / 2.0
    v_centre_nom = (nv - 1) / 2.0

    # Slab profile
    slab_offsets_mm = [-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0]
    n_slab = len(slab_offsets_mm)
    init_logits = torch.tensor([-6.0, 0.0, 0.0, 0.0, 0.0, 0.0, -6.0],
                                device="cuda", dtype=torch.float32)
    w_slab_logits = torch.nn.Parameter(init_logits.clone())
    delta_z = torch.nn.Parameter(torch.tensor(0.0, device="cuda"))

    # Per-(GT, slab) helix-index picks, precomputed with alpha_dz=+1
    z_pos_for_picks = z_pos_sub + 1.0 * ffs_dz_sub
    target_source_z_per_gt = [-z for z in truth_pZ_list]   # source frame
    print(f"[fit-v2] precomputing picks for {N_GT}×{n_slab} = "
          f"{N_GT * n_slab} (GT,slab) combos …", flush=True)
    picked_per_gt_per_slab = []
    for i_gt, z_tgt_i in enumerate(target_source_z_per_gt):
        per_slab = []
        for off in slab_offsets_mm:
            picks = precompute_picks(z_pos_for_picks, orig_idx, rotview,
                                      z_tgt_i + off)
            per_slab.append(picks)
        picked_per_gt_per_slab.append(per_slab)

    # FFT2 radial frequency grid for the 64-bin filter
    fy = torch.fft.fftfreq(512, device="cuda").float()
    fx = torch.fft.fftfreq(512, device="cuda").float()
    fyy, fxx = torch.meshgrid(fy, fx, indexing="ij")
    rho = torch.sqrt(fyy ** 2 + fxx ** 2)

    dr = 0.05

    # ---- Forward pipeline for one GT slice ----
    def _forward_one_gt(i_gt: int):
        w_slab = F.softmax(w_slab_logits, dim=0)
        sino_slab = None
        target_z_i = target_source_z_per_gt[i_gt]
        picks_per_slab_i = picked_per_gt_per_slab[i_gt]
        z_pos_eff = z_pos_sub + alpha_dz * ffs_dz_sub
        for k, off in enumerate(slab_offsets_mm):
            z_eff = target_z_i + off + delta_z
            sino_k = helical_ssr_torch(
                proj_flat, z_pos_eff, picks_per_slab_i[k], z_eff,
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
        return clipped, sino_slab, fbp_2d

    # ---- Adam loop ----
    opt = torch.optim.Adam(
        [sod, sdd, delta_z, w_slab_logits, h_radial, a, bg, hi],
        lr=2e-3,
    )
    n_iters = 1500
    log_every = max(1, n_iters // 30)
    lam_h = 1e-4

    sod0, sdd0 = sod.item(), sdd.item()
    print(f"[fit-v2] starting Adam, {n_iters} iters, lr=2e-3, "
          f"per-slice L2 mean (no FoV mask)", flush=True)
    print(f"[fit-v2] init  sod={sod0:.3f}  sdd={sdd0:.3f}", flush=True)
    hist = []
    for it in range(n_iters):
        opt.zero_grad()
        # Per-slice L2 mean — NO FoV mask, full 512²
        per_slice_l2 = []
        sino_central = None; fbp_central = None
        central_k = N_GT // 2
        for i_gt in range(N_GT):
            pred_i, sino_i, fbp_i = _forward_one_gt(i_gt)
            l2_i = F.mse_loss(pred_i, truth_stack[i_gt], reduction="mean")
            per_slice_l2.append(l2_i)
            if i_gt == central_k:
                sino_central = sino_i; fbp_central = fbp_i
        data_loss = torch.stack(per_slice_l2).mean()
        smooth_loss = ((h_radial[2:] - 2 * h_radial[1:-1] +
                        h_radial[:-2]) ** 2).mean()
        total = data_loss + lam_h * smooth_loss
        total.backward()
        opt.step()
        if it % log_every == 0 or it == n_iters - 1:
            with torch.no_grad():
                w_show = F.softmax(w_slab_logits, dim=0).cpu().numpy()
                per_slice_arr = torch.stack(per_slice_l2).cpu().numpy()
            print(f"[fit-v2] iter {it:4d}/{n_iters}  "
                  f"L2_mean={data_loss.item():.3e}  "
                  f"per_slice L2 [min={per_slice_arr.min():.3e}, "
                  f"max={per_slice_arr.max():.3e}]  "
                  f"sod={sod.item():.3f}  sdd={sdd.item():.3f}  "
                  f"Δz={delta_z.item():+.4f}  "
                  f"a={a.item():.3f} bg={bg.item():+.5f} hi={hi.item():.4f}  "
                  f"|h|=[{h_radial.min().item():.3f}, "
                  f"{h_radial.max().item():.3f}]  "
                  f"w_slab=[{','.join(f'{w:.2f}' for w in w_show)}]",
                  flush=True)
            hist.append({"iter": it, "loss": float(data_loss.item())})

    # ---- Final per-slice metrics ----
    with torch.no_grad():
        preds_np = []
        per_gt_metrics = []
        for i_gt in range(N_GT):
            pred_i, _, _ = _forward_one_gt(i_gt)
            pred_np = pred_i.cpu().numpy()
            preds_np.append(pred_np)
            m = calc_metrics(pred_np, truth_list_np[i_gt], dr=dr)
            per_gt_metrics.append(m)
        ssim_arr = np.array([m["ssim"] for m in per_gt_metrics])
        psnr_arr = np.array([m["psnr"] for m in per_gt_metrics])
        rmse_arr = np.array([m["rmse"] for m in per_gt_metrics])
        print(f"[fit-v2] FINAL per-GT (no intensity_calibrate):", flush=True)
        for k, (idx, m_k) in enumerate(zip(gt_indices, per_gt_metrics)):
            print(f"[fit-v2]   GT #{idx:3d} (pZ={truth_pZ_list[k]:+.2f}): "
                  f"SSIM={m_k['ssim']:.4f}  PSNR={m_k['psnr']:6.2f}  "
                  f"RMSE={m_k['rmse']:.5f}  diff_max={m_k['diff_max']:.4f}",
                  flush=True)
        print(f"[fit-v2] MEAN: SSIM={ssim_arr.mean():.4f}  "
              f"PSNR={psnr_arr.mean():.2f}  RMSE={rmse_arr.mean():.5f}",
              flush=True)
        # Also post-hoc with intensity_calibrate for comparison
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
        print(f"[fit-v2] FINAL per-GT (post-hoc intensity_calibrate):",
              flush=True)
        for k, (idx, m_k) in enumerate(zip(gt_indices, per_gt_cal_metrics)):
            print(f"[fit-v2]   GT #{idx:3d}: SSIM={m_k['ssim']:.4f}  "
                  f"PSNR={m_k['psnr']:6.2f}  RMSE={m_k['rmse']:.5f}",
                  flush=True)
        print(f"[fit-v2] MEAN_cal: SSIM={ssim_cal_arr.mean():.4f}  "
              f"PSNR={psnr_cal_arr.mean():.2f}  "
              f"RMSE={rmse_cal_arr.mean():.5f}", flush=True)

    # ---- Save fit blob ----
    out_json = REPO / "results" / "breast_debug" / "L014_rebin_end2end_fit_v2.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_blob = {
        "rebin_fitted": {
            "sod": float(sod.item()),
            "sdd": float(sdd.item()),
            "du": float(du.item()),
            "dv": float(dv.item()),
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
    print(f"[fit-v2] wrote {out_json}", flush=True)

    # ---- Diagnostic PNG: 10 rows (truth | pred | diff), full range ----
    out_png = REPO / "results" / "mayo_debug" / "L014_rebin_end2end_fit_v2.png"
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(N_GT, 3, figsize=(12, 3 * N_GT))
    for k in range(N_GT):
        truth_k = truth_list_np[k]
        pred_k = preds_np[k]
        diff_k = pred_k - truth_k
        axes[k, 0].imshow(truth_k, cmap="gray", vmin=0, vmax=dr)
        axes[k, 0].set_title(
            f"GT #{gt_indices[k]} (pZ={truth_pZ_list[k]:+.1f})", fontsize=9)
        axes[k, 1].imshow(pred_k, cmap="gray", vmin=0, vmax=dr)
        axes[k, 1].set_title(
            f"pred  SSIM={per_gt_metrics[k]['ssim']:.4f}  "
            f"PSNR={per_gt_metrics[k]['psnr']:.2f}", fontsize=9)
        axes[k, 2].imshow(diff_k, cmap="seismic", vmin=-0.01, vmax=0.01)
        axes[k, 2].set_title(
            f"diff RMSE={per_gt_metrics[k]['rmse']:.5f}", fontsize=9)
        for j in range(3):
            axes[k, j].set_xticks([]); axes[k, j].set_yticks([])
    fig.suptitle(
        f"L014 end-to-end fit v2 — 10 GTs sampled uniformly across 154 "
        f"(per-slice-L2 mean, no FoV mask)\n"
        f"SSIM_mean={ssim_arr.mean():.4f}  PSNR_mean={psnr_arr.mean():.2f} "
        f"(post-hoc cal: SSIM={ssim_cal_arr.mean():.4f}  "
        f"PSNR={psnr_cal_arr.mean():.2f})",
        fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.99])
    fig.savefig(out_png, dpi=110, bbox_inches="tight")
    print(f"[fit-v2] wrote {out_png}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
