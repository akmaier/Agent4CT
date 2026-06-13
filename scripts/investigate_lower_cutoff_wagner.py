"""Numerical SSIM investigation of the lower intensity cutoff (Wagner Mayo).

Motivation (user, 2026-06-13): L277's HD_tc−GT difference image shows a faint
blue (recon < truth) shadow in the posterior, suggesting the lower-end cutoff
in the intensity scaling is set too high (clipping low-density values).

Two lower clips exist in the calibrated-evaluation path:
  * PRE-calibration  : the FBP is clamped `fbp.clamp(min=pre_floor)` before
                       the affine fit (compare scripts use pre_floor=0).
  * POST-calibration : `intensity_calibrate` does `pred_cal.clamp_min(post_floor)`
                       on the affine output (metrics.py uses post_floor=0).

This script reconstructs each patient's production truncation-corrected HD FBP
once, then sweeps each floor independently (the other held at 0) and reports
calibrated SSIM / PSNR / RMSE vs truth. If SSIM rises as a floor goes negative,
that cutoff was indeed too high. Emphasis on L277; all 10 patients run so we can
see whether a change helps L277 without regressing the others.

Output:
  results/mayo_debug/lower_cutoff_sweep.json
  results/mayo_debug/lower_cutoff_sweep.png            (SSIM vs floor curves)
  results/mayo_debug/lower_cutoff_L277_diffs.png       (L277 diff @ floors)
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from ddssl_ldct.metrics import ssim as ssim_metric, psnr as psnr_metric
from scripts.validate_mayo_helix2fan import _load_truth_slice_for_z
from scripts.compare_gt_hd_ld_fbp_wagner_trunc import _load_slab, _fbp_slab

WAGNER_ALL = ["L145", "L186", "L209", "L219", "L277",
              "L014", "L056", "L058", "L075", "L123"]
SPLIT = {**{p: "train" for p in ["L145", "L186", "L209", "L219"]},
         "L277": "val",
         **{p: "test" for p in ["L014", "L056", "L058", "L075", "L123"]}}

DR = 0.05
# floor grid (mu units). 0.0 = current production behaviour.
FLOORS = [-0.020, -0.015, -0.010, -0.0075, -0.005, -0.0025, 0.0, 0.0025]


def calibrated_metrics(raw_fbp: torch.Tensor, truth: torch.Tensor,
                       pre_floor: float, post_floor: float, dr: float = DR):
    """intensity_calibrate with CONFIGURABLE pre/post lower floors.

    Mirrors ddssl_ldct.metrics.intensity_calibrate (two-point bg->0,
    fg_mean->truth_fg_mean, clip [floor, dr]) but exposes both floors.
    Returns (ssim, psnr, rmse, pred_cal).
    """
    pred = raw_fbp.clamp(min=pre_floor)
    fg_thr = 0.05 * dr
    fg = truth > fg_thr
    bg = ~fg
    bg_pred = pred[bg].mean()
    fg_pred = pred[fg].mean()
    fg_truth = truth[fg].mean()
    span = (fg_pred - bg_pred).clamp_min(1e-9)
    a = fg_truth / span
    cal = a * (pred - bg_pred)
    cal = cal.clamp(min=post_floor, max=dr)
    s = float(ssim_metric(cal[None, None], truth[None, None], dr).cpu())
    p = float(psnr_metric(cal[None, None], truth[None, None], dr).cpu())
    r = float(((cal - truth) ** 2).mean().sqrt().cpu())
    return s, p, r, cal


def main() -> int:
    root = Path(os.environ.get("AGENT4CT_DATA",
                                "/cluster/maier/Agent4CT/data")) / "mayo_ldct"
    sino_dir = root / os.environ.get("STAGED_HELIX2FAN_SUBDIR", "staged_helix2fan_v3")
    truth_root = root / "raw"
    out_dir = REPO / "results" / "mayo_debug"
    out_dir.mkdir(parents=True, exist_ok=True)
    dev = "cuda"
    z_off, slab_half, pad, mu_water = 3.5, 2, 384, 0.02

    results = {}
    l277_imgs = {}
    for pat in WAGNER_ALL:
        sino_hd = sino_dir / f"{pat}_sino_fulldose.h5"
        gj = sino_dir / f"{pat}_sino_fulldose_geometry.json"
        zg = sino_dir / f"{pat}_sino_fulldose_z_grid.npy"
        if not (sino_hd.exists() and gj.exists() and zg.exists()):
            print(f"[cutoff] {pat}: missing inputs, skip", flush=True)
            continue
        geom = json.loads(gj.read_text())
        zgrid = np.load(zg)
        slab, zc = _load_slab(sino_hd, geom, zgrid, z_off, slab_half)
        truth_info = _load_truth_slice_for_z(truth_root / pat, zc)
        if truth_info is None:
            print(f"[cutoff] {pat}: no truth, skip", flush=True)
            continue
        truth_np, _, _, tmeta = truth_info
        ps_eff = 0.700857 * (float(tmeta["pixel_spacing"]) / 0.703125)
        # production truncation-corrected HD FBP (raw, pre-clamp)
        raw = _fbp_slab(slab, geom, ps_eff, dev, pad, mu_water)
        raw_t = torch.from_numpy(raw).to(dev).float()
        truth_t = torch.from_numpy(truth_np).to(dev).float()

        pre_sweep = []   # vary pre_floor, post_floor=0
        for f in FLOORS:
            s, p, r, _ = calibrated_metrics(raw_t, truth_t, pre_floor=f, post_floor=0.0)
            pre_sweep.append({"floor": f, "ssim": s, "psnr": p, "rmse": r})
        post_sweep = []  # vary post_floor, pre_floor = raw min (no pre clamp)
        raw_min = float(raw_t.min())
        for f in FLOORS:
            s, p, r, cal = calibrated_metrics(raw_t, truth_t, pre_floor=raw_min, post_floor=f)
            post_sweep.append({"floor": f, "ssim": s, "psnr": p, "rmse": r})
            if pat == "L277" and f in (0.0, -0.010, -0.020):
                l277_imgs[f"post{f}"] = (cal.cpu().numpy(), truth_np)

        base = [x for x in pre_sweep if x["floor"] == 0.0][0]
        best_pre = max(pre_sweep, key=lambda x: x["ssim"])
        best_post = max(post_sweep, key=lambda x: x["ssim"])
        results[pat] = {"split": SPLIT[pat], "truth_ps": float(tmeta["pixel_spacing"]),
                        "raw_min": raw_min, "baseline_ssim": base["ssim"],
                        "pre_sweep": pre_sweep, "post_sweep": post_sweep,
                        "best_pre": best_pre, "best_post": best_post}
        print(f"[cutoff] {pat} ({SPLIT[pat]:5}) base SSIM={base['ssim']:.4f}  "
              f"best_pre {best_pre['ssim']:.4f}@{best_pre['floor']:+.4f}  "
              f"best_post {best_post['ssim']:.4f}@{best_post['floor']:+.4f}  "
              f"raw_min={raw_min:+.4f}", flush=True)

    # ---- curves ----
    fig, ax = plt.subplots(1, 2, figsize=(15, 6))
    for pat, r in results.items():
        lw = 2.5 if pat == "L277" else 1.0
        ax[0].plot([x["floor"] for x in r["pre_sweep"]], [x["ssim"] for x in r["pre_sweep"]],
                   marker="o", lw=lw, label=pat)
        ax[1].plot([x["floor"] for x in r["post_sweep"]], [x["ssim"] for x in r["post_sweep"]],
                   marker="o", lw=lw, label=pat)
    for a, ttl in zip(ax, ["pre-calibration FBP floor (post=0)",
                            "post-calibration output floor (pre=raw min)"]):
        a.axvline(0.0, ls="--", c="k", alpha=0.4, label="production (0.0)")
        a.set_xlabel("lower cutoff (mu, mm^-1)"); a.set_ylabel("calibrated SSIM")
        a.set_title(ttl); a.grid(alpha=0.3)
    ax[0].legend(fontsize=7, ncol=2)
    fig.suptitle("Lower-cutoff SSIM sweep (Wagner HD, truncation-corrected). L277 bold.")
    fig.tight_layout()
    fig.savefig(out_dir / "lower_cutoff_sweep.png", dpi=110, bbox_inches="tight")
    plt.close(fig)

    # ---- L277 diff images at a few post-floors ----
    if l277_imgs:
        keys = sorted(l277_imgs.keys())
        fig, ax = plt.subplots(2, len(keys), figsize=(5 * len(keys), 9))
        for j, k in enumerate(keys):
            cal, truth = l277_imgs[k]
            ax[0, j].imshow(cal, cmap="gray", vmin=0, vmax=DR)
            ax[0, j].set_title(f"L277 cal ({k})", fontsize=10)
            d = cal - truth
            ax[1, j].imshow(d, cmap="seismic", vmin=-0.015, vmax=0.015)
            ax[1, j].set_title(f"cal - GT  RMSE={np.sqrt((d**2).mean()):.4f}", fontsize=10)
            for i in range(2):
                ax[i, j].set_xticks([]); ax[i, j].set_yticks([])
        fig.suptitle("L277: calibrated recon + difference at varying post-cal lower floor")
        fig.tight_layout()
        fig.savefig(out_dir / "lower_cutoff_L277_diffs.png", dpi=110, bbox_inches="tight")
        plt.close(fig)

    (out_dir / "lower_cutoff_sweep.json").write_text(json.dumps(results, indent=2))
    print(f"[cutoff] wrote {out_dir}/lower_cutoff_sweep.json + .png + L277 diffs", flush=True)

    # aggregate: mean SSIM per floor (pre + post)
    print("\n[cutoff] AGGREGATE mean calibrated SSIM vs floor:", flush=True)
    print(f"  {'floor':>8} {'pre(post=0)':>12} {'post(pre=min)':>14}", flush=True)
    for i, f in enumerate(FLOORS):
        pre_m = np.mean([results[p]["pre_sweep"][i]["ssim"] for p in results])
        post_m = np.mean([results[p]["post_sweep"][i]["ssim"] for p in results])
        mark = "  <- production" if f == 0.0 else ""
        print(f"  {f:+8.4f} {pre_m:12.4f} {post_m:14.4f}{mark}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
