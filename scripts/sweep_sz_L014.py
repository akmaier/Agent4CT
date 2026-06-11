#!/usr/bin/env python -u
"""1-parameter z-scaling sweep at the v2 fit's optimum.

Tests whether a single multiplier `s_z` on the per-readout source-z
(equivalent to a global `pitch_mm` correction) lifts edge-slice metrics
toward the centre — diagnostic for an un-modelled z-scaling in the
rebin step (mechanisms A = wrong pitch_mm, B = wrong dv; both look
identical at the fit output).

For each `s_z` in a small grid around 1.0:
  1. Recompute z_pos_eff = s_z · z_pos_sub + α_dz · ffs_dz
  2. Recompute per-(GT, slab) helix-index picks against z_pos_eff
  3. Run the v2 forward pipeline with all v2-fit params frozen
  4. Score per-GT SSIM / PSNR / RMSE / L2 against the same 10 truth
     slices used by v2.

Output:
  results/breast_debug/L014_sz_sweep.json
  results/mayo_debug/L014_sz_sweep.png
"""
from __future__ import annotations
import sys
import json
import math
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

from scripts.fit_rebin_end2end_L014 import (
    _list_truth, _mu, precompute_picks, helical_ssr_torch,
    radial_filter_2d, calc_metrics,
)
from scripts.fit_rebin_end2end_L014_v2 import SLICE_INDICES


def main() -> int:
    import os
    root = Path(os.environ.get("AGENT4CT_DATA",
                                "/cluster/maier/Agent4CT/data")) / "mayo_ldct"
    sino_dir = root / "staged_helix2fan"
    raw_dir = root / "raw" / "L014"

    blob_path = sino_dir / "L014_proj_flat_full.pt"
    fit_path = REPO / "results" / "breast_debug" / "L014_rebin_end2end_fit_v2.json"
    if not blob_path.exists():
        print(f"[sweep] missing {blob_path}", file=sys.stderr); return 2
    if not fit_path.exists():
        print(f"[sweep] missing {fit_path}", file=sys.stderr); return 2

    print(f"[sweep] loading {blob_path} …", flush=True)
    blob = torch.load(blob_path, weights_only=False, map_location="cpu")
    proj_flat = blob["proj_flat"].to("cuda")
    z_pos_sub = blob["z_positions"].to("cuda")
    orig_idx = blob["original_indices"].to("cuda")
    ffs_dz_sub = blob.get("ffs_dz",
                          torch.zeros_like(z_pos_sub)).to("cuda")
    rotview = int(blob["rotview"])
    nu, nv = int(blob["nu"]), int(blob["nv"])
    angle_start = float(blob["angle_start_corrected"])

    print(f"[sweep] loading {fit_path} …", flush=True)
    fit_blob = json.loads(fit_path.read_text())
    # Locked params (from v2 fit)
    sod_val = fit_blob["rebin_fitted"]["sod"]
    sdd_val = fit_blob["rebin_fitted"]["sdd"]
    du_val  = fit_blob["rebin_fitted"]["du"]
    dv_val  = fit_blob["rebin_fitted"]["dv"]
    delta_z_val = fit_blob["slab_fitted"]["delta_z_mm"]
    slab_offsets_mm = fit_blob["slab_fitted"]["slab_offsets_mm"]
    w_slab_np = np.asarray(fit_blob["slab_fitted"]["w_slab"], dtype=np.float32)
    alpha_dz_val = fit_blob["slab_fitted"]["alpha_dz_fixed"]
    a_val  = fit_blob["post_fbp_fitted"]["a"]
    bg_val = fit_blob["post_fbp_fitted"]["bg"]
    hi_val = fit_blob["post_fbp_fitted"]["hi"]
    h_radial_np = np.asarray(fit_blob["post_fbp_fitted"]["h_radial"],
                              dtype=np.float32)
    print(f"[sweep] v2 fit at: sod={sod_val:.3f}  sdd={sdd_val:.3f}  "
          f"Δz={delta_z_val:+.4f}  a={a_val:.3f}  bg={bg_val:+.5f}  hi={hi_val:.4f}",
          flush=True)

    # Truth slices (same 10 indices as v2)
    truth_files = _list_truth(raw_dir)
    truth_files.sort(key=lambda t: t[0])
    gt_indices = [i for i in SLICE_INDICES if i < len(truth_files)]
    truth_list_np = []; truth_pZ_list = []
    for ti in gt_indices:
        pZ_i, fp_i = truth_files[ti]
        mu_i, _ = _mu(fp_i)
        truth_list_np.append(mu_i); truth_pZ_list.append(pZ_i)
    truth_stack = torch.stack(
        [torch.from_numpy(x).to("cuda").float() for x in truth_list_np],
        dim=0,
    )
    N_GT = len(gt_indices)
    print(f"[sweep] N_GT={N_GT}", flush=True)

    # Frozen tensors
    sod = torch.tensor(sod_val, device="cuda")
    sdd = torch.tensor(sdd_val, device="cuda")
    du  = torch.tensor(du_val,  device="cuda")
    dv  = torch.tensor(dv_val,  device="cuda")
    delta_z = torch.tensor(delta_z_val, device="cuda")
    w_slab = torch.from_numpy(w_slab_np).to("cuda")
    a  = torch.tensor(a_val,  device="cuda")
    bg = torch.tensor(bg_val, device="cuda")
    hi = torch.tensor(hi_val, device="cuda")
    h_radial = torch.from_numpy(h_radial_np).to("cuda")
    alpha_dz = torch.tensor(alpha_dz_val, device="cuda")
    n_bins = len(h_radial)

    u_centre_nom = (nu - 1) / 2.0
    v_centre_nom = (nv - 1) / 2.0
    target_source_z_per_gt = [-z for z in truth_pZ_list]

    # FBP projector (fixed at Powell)
    fbp_geom = FanBeamGeometry.mayo_ldct_fitted(
        n_angles=rotview, n_det=nu,
        angle_start=angle_start, angle_end=angle_start + 2 * math.pi,
    )
    proj_fbp = PyronnFanBeamProjector(fbp_geom).to("cuda")
    proj_fbp._tensor_geom["detector_origin"] = (
        proj_fbp._tensor_geom["detector_origin"] + MAYO_LDCT_DET_OFFSET
    )

    fy = torch.fft.fftfreq(512, device="cuda").float()
    fx = torch.fft.fftfreq(512, device="cuda").float()
    fyy, fxx = torch.meshgrid(fy, fx, indexing="ij")
    rho = torch.sqrt(fyy ** 2 + fxx ** 2)

    dr = 0.05

    def forward_one_gt(i_gt: int, picks_per_slab_i, s_z_val: float):
        z_pos_eff = s_z_val * z_pos_sub + alpha_dz * ffs_dz_sub
        sino_slab = None
        target_z_i = target_source_z_per_gt[i_gt]
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
        return clipped

    # ---- Sweep ----
    s_z_grid = np.round(np.arange(0.9950, 1.00501, 0.0005), 6).tolist()
    print(f"[sweep] s_z grid: {len(s_z_grid)} points "
          f"[{s_z_grid[0]}, {s_z_grid[-1]}]", flush=True)
    results = []
    with torch.no_grad():
        for s_z in s_z_grid:
            # Recompute picks for this s_z (cheap; nearest-integer indexing)
            z_pos_for_picks = s_z * z_pos_sub + alpha_dz * ffs_dz_sub
            picks_per_gt_per_slab = []
            for i_gt, z_tgt_i in enumerate(target_source_z_per_gt):
                per_slab = []
                for off in slab_offsets_mm:
                    picks = precompute_picks(z_pos_for_picks, orig_idx,
                                              rotview, z_tgt_i + off)
                    per_slab.append(picks)
                picks_per_gt_per_slab.append(per_slab)

            l2_list = []; ssim_list = []; psnr_list = []; rmse_list = []
            preds_np = []
            for i_gt in range(N_GT):
                pred_i = forward_one_gt(i_gt, picks_per_gt_per_slab[i_gt],
                                          s_z)
                pred_np = pred_i.cpu().numpy()
                preds_np.append(pred_np)
                l2_i = float(((pred_i - truth_stack[i_gt]) ** 2).mean().item())
                m = calc_metrics(pred_np, truth_list_np[i_gt], dr=dr)
                l2_list.append(l2_i)
                ssim_list.append(m["ssim"])
                psnr_list.append(m["psnr"])
                rmse_list.append(m["rmse"])
            entry = {
                "s_z": float(s_z),
                "per_gt": {"ssim": ssim_list, "psnr": psnr_list,
                           "rmse": rmse_list, "l2": l2_list},
                "ssim_mean": float(np.mean(ssim_list)),
                "psnr_mean": float(np.mean(psnr_list)),
                "rmse_mean": float(np.mean(rmse_list)),
                "l2_mean":   float(np.mean(l2_list)),
            }
            results.append(entry)
            print(f"[sweep] s_z={s_z:.5f}  L2_mean={entry['l2_mean']:.3e}  "
                  f"SSIM_mean={entry['ssim_mean']:.4f}  "
                  f"PSNR_mean={entry['psnr_mean']:.2f}  "
                  f"RMSE_mean={entry['rmse_mean']:.5f}", flush=True)

    # ---- Report best ----
    best = min(results, key=lambda r: r["l2_mean"])
    print(f"\n[sweep] BEST by L2_mean: s_z={best['s_z']:.5f}  "
          f"L2_mean={best['l2_mean']:.3e}  "
          f"SSIM_mean={best['ssim_mean']:.4f}  "
          f"PSNR_mean={best['psnr_mean']:.2f}", flush=True)
    baseline = next(r for r in results if abs(r["s_z"] - 1.000) < 1e-6)
    print(f"[sweep] baseline s_z=1.0:  L2_mean={baseline['l2_mean']:.3e}  "
          f"SSIM_mean={baseline['ssim_mean']:.4f}  "
          f"PSNR_mean={baseline['psnr_mean']:.2f}", flush=True)
    print(f"[sweep] Δ vs baseline: ΔSSIM={best['ssim_mean']-baseline['ssim_mean']:+.4f}  "
          f"ΔPSNR={best['psnr_mean']-baseline['psnr_mean']:+.2f} dB  "
          f"ΔL2={(best['l2_mean']-baseline['l2_mean'])/baseline['l2_mean']*100:+.2f}%",
          flush=True)

    # ---- Save JSON ----
    out_json = REPO / "results" / "breast_debug" / "L014_sz_sweep.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps({
        "frozen_v2_params": {
            "sod": sod_val, "sdd": sdd_val, "du": du_val, "dv": dv_val,
            "delta_z_mm": delta_z_val, "a": a_val, "bg": bg_val, "hi": hi_val,
            "alpha_dz_fixed": alpha_dz_val,
            "w_slab": w_slab_np.tolist(),
            "h_radial": h_radial_np.tolist(),
        },
        "slice_indices": gt_indices,
        "slice_patient_z": truth_pZ_list,
        "sweep": results,
        "best": best,
        "baseline_s_z_1.0": baseline,
    }, indent=2))
    print(f"[sweep] wrote {out_json}", flush=True)

    # ---- Plot: per-GT SSIM vs s_z, plus L2_mean curve ----
    out_png = REPO / "results" / "mayo_debug" / "L014_sz_sweep.png"
    out_png.parent.mkdir(parents=True, exist_ok=True)
    s_arr = np.array([r["s_z"] for r in results])
    l2_arr = np.array([r["l2_mean"] for r in results])
    ssim_arr = np.array([r["ssim_mean"] for r in results])
    psnr_arr = np.array([r["psnr_mean"] for r in results])
    per_gt_ssim = np.array([r["per_gt"]["ssim"] for r in results])  # (n_s, N_GT)

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))

    axes[0, 0].plot((s_arr - 1) * 100, l2_arr, "-o", color="C0")
    axes[0, 0].axvline(0.0, color="0.5", lw=0.5)
    axes[0, 0].axvline((best["s_z"] - 1) * 100, color="C3", ls="--",
                        label=f"best s_z={best['s_z']:.5f}")
    axes[0, 0].set_xlabel("s_z − 1  (×100, %)")
    axes[0, 0].set_ylabel("mean per-slice L2")
    axes[0, 0].set_title(f"L2_mean vs s_z (best ε = {(best['s_z']-1)*100:+.3f} %)")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot((s_arr - 1) * 100, ssim_arr, "-o", color="C2",
                     label="mean SSIM")
    ax_psnr = axes[0, 1].twinx()
    ax_psnr.plot((s_arr - 1) * 100, psnr_arr, "-s", color="C1",
                  label="mean PSNR")
    axes[0, 1].axvline((best["s_z"] - 1) * 100, color="C3", ls="--")
    axes[0, 1].set_xlabel("s_z − 1  (×100, %)")
    axes[0, 1].set_ylabel("mean SSIM", color="C2")
    ax_psnr.set_ylabel("mean PSNR (dB)", color="C1")
    axes[0, 1].set_title("Mean metric vs s_z")
    axes[0, 1].grid(True, alpha=0.3)

    # Per-GT SSIM trajectory across s_z
    for k, (idx, pz) in enumerate(zip(gt_indices, truth_pZ_list)):
        axes[1, 0].plot((s_arr - 1) * 100, per_gt_ssim[:, k], "-",
                         alpha=0.7, label=f"#{idx} (pZ={pz:+.0f})")
    axes[1, 0].axvline((best["s_z"] - 1) * 100, color="0.3", ls="--")
    axes[1, 0].set_xlabel("s_z − 1  (×100, %)")
    axes[1, 0].set_ylabel("per-GT SSIM")
    axes[1, 0].set_title("Per-GT SSIM vs s_z (does edge lift toward centre?)")
    axes[1, 0].legend(loc="lower center", ncol=2, fontsize=8)
    axes[1, 0].grid(True, alpha=0.3)

    # Per-GT SSIM at baseline vs best, by patient-z
    pZ_arr = np.array(truth_pZ_list)
    base_ssim = np.array(baseline["per_gt"]["ssim"])
    best_ssim = np.array(best["per_gt"]["ssim"])
    axes[1, 1].plot(pZ_arr, base_ssim, "-o", color="C0",
                     label=f"baseline s_z=1.000  mean={baseline['ssim_mean']:.4f}")
    axes[1, 1].plot(pZ_arr, best_ssim, "-s", color="C3",
                     label=f"best s_z={best['s_z']:.5f}  mean={best['ssim_mean']:.4f}")
    axes[1, 1].set_xlabel("patient z (mm)")
    axes[1, 1].set_ylabel("per-GT SSIM")
    axes[1, 1].set_title("SSIM vs patient z: baseline vs best s_z")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    fig.suptitle(
        f"L014 z-scaling sweep at v2 optimum  (sod={sod_val:.3f}, sdd={sdd_val:.3f}, "
        f"Δz={delta_z_val:+.4f})", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_png, dpi=120, bbox_inches="tight")
    print(f"[sweep] wrote {out_png}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
