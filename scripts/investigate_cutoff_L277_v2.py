"""L277 single lower-cutoff sweep with the scaling calibration HELD FIXED.

Fixes the confound in investigate_lower_cutoff_wagner.py: there, the pre-cal
FBP floor and the post-cal clamp_min(0) cut each other off, so neither showed
an effect. Here we:

  1. reconstruct L277 HD (truncation-corrected) ONCE -> raw FBP (unclamped),
  2. fit the affine calibration (a, bg) ONCE on that raw vs truth and FREEZE it
     (cal = a * (raw - bg)),  -- no per-threshold re-fit, no double clamp,
  3. sweep a SINGLE lower-clip threshold T over BOTH directions
     (T < 0 = lower cutoff, T > 0 = higher cutoff), upper clip fixed at dr:
        img(T) = clamp(cal, min=T, max=dr)
     and report SSIM/PSNR/RMSE(T).

No several reconstructions; the sweep is a cheap clamp on the single recon.
Outputs (L277 only):
  results/mayo_debug/cutoff_L277_v2.json
  results/mayo_debug/cutoff_L277_v2_curve.png
  results/mayo_debug/cutoff_L277_v2_images.png   (recon + diff at several T)
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

DR = 0.05
PAT = "L277"


def main() -> int:
    root = Path(os.environ.get("AGENT4CT_DATA",
                                "/cluster/maier/Agent4CT/data")) / "mayo_ldct"
    sino_dir = root / os.environ.get("STAGED_HELIX2FAN_SUBDIR", "staged_helix2fan_v3")
    truth_root = root / "raw"
    out_dir = REPO / "results" / "mayo_debug"
    out_dir.mkdir(parents=True, exist_ok=True)
    dev = "cuda"

    geom = json.loads((sino_dir / f"{PAT}_sino_fulldose_geometry.json").read_text())
    zgrid = np.load(sino_dir / f"{PAT}_sino_fulldose_z_grid.npy")
    slab, zc = _load_slab(sino_dir / f"{PAT}_sino_fulldose.h5", geom, zgrid, 3.5, 2)
    truth_info = _load_truth_slice_for_z(truth_root / PAT, zc)
    truth_np, _, _, tmeta = truth_info
    ps_eff = 0.700857 * (float(tmeta["pixel_spacing"]) / 0.703125)

    # ONE reconstruction: production truncation-corrected HD FBP (unclamped raw)
    raw = _fbp_slab(slab, geom, ps_eff, dev, 384, 0.02)
    raw_t = torch.from_numpy(raw).to(dev).float()
    truth_t = torch.from_numpy(truth_np).to(dev).float()

    # FIXED scaling calibration (a, bg), fit ONCE on the raw recon vs truth.
    fg_thr = 0.05 * DR
    fg = truth_t > fg_thr
    bg = ~fg
    bg_pred = raw_t[bg].mean()
    a = (truth_t[fg].mean() / (raw_t[fg].mean() - bg_pred).clamp_min(1e-9))
    cal = a * (raw_t - bg_pred)          # frozen affine; NO clamp yet
    print(f"[L277-v2] fixed calibration: a={float(a):.4f}  bg={float(bg_pred):+.5f}  "
          f"cal min/max=[{float(cal.min()):+.4f},{float(cal.max()):.4f}]  "
          f"raw_min={float(raw_t.min()):+.5f}", flush=True)

    # sweep a single lower-clip threshold, both directions
    Ts = [round(t, 5) for t in np.linspace(-0.020, 0.015, 36)]
    sweep = []
    for T in Ts:
        img = cal.clamp(min=float(T), max=DR)
        s = float(ssim_metric(img[None, None], truth_t[None, None], DR).cpu())
        p = float(psnr_metric(img[None, None], truth_t[None, None], DR).cpu())
        r = float(((img - truth_t) ** 2).mean().sqrt().cpu())
        sweep.append({"T": T, "ssim": s, "psnr": p, "rmse": r})

    base = min(sweep, key=lambda x: abs(x["T"] - 0.0))   # nearest to production 0.0
    best = max(sweep, key=lambda x: x["ssim"])
    print(f"[L277-v2] production T~0: SSIM={base['ssim']:.4f} PSNR={base['psnr']:.2f}",
          flush=True)
    print(f"[L277-v2] best  T={best['T']:+.4f}: SSIM={best['ssim']:.4f} "
          f"PSNR={best['psnr']:.2f}  (ΔSSIM {best['ssim']-base['ssim']:+.4f})", flush=True)
    for x in sweep:
        print(f"[L277-v2]   T={x['T']:+.4f}  SSIM={x['ssim']:.4f}  PSNR={x['psnr']:.2f}  "
              f"RMSE={x['rmse']:.5f}", flush=True)

    # ---- curve ----
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot([x["T"] for x in sweep], [x["ssim"] for x in sweep], marker="o")
    ax.axvline(0.0, ls="--", c="k", alpha=0.5, label="production (T=0)")
    ax.axvline(best["T"], ls=":", c="r", alpha=0.7, label=f"best T={best['T']:+.4f}")
    ax.set_xlabel("single lower-clip threshold T (mu, mm^-1)  [calibration FIXED]")
    ax.set_ylabel("calibrated SSIM"); ax.grid(alpha=0.3); ax.legend()
    ax.set_title(f"L277 lower-cutoff sweep, fixed scaling (a={float(a):.3f}, bg={float(bg_pred):+.4f})")
    fig.tight_layout()
    fig.savefig(out_dir / "cutoff_L277_v2_curve.png", dpi=120, bbox_inches="tight")
    plt.close(fig)

    # ---- images at representative T (recon + diff) ----
    show_T = [-0.010, 0.0, 0.005, 0.010]
    fig, ax = plt.subplots(2, len(show_T) + 1, figsize=(5 * (len(show_T) + 1), 9.5))
    ax[0, 0].imshow(truth_np, cmap="gray", vmin=0, vmax=DR)
    ax[0, 0].set_title("L277 GT (truth)", fontsize=11)
    ax[1, 0].axis("off")
    for j, T in enumerate(show_T):
        img = cal.clamp(min=T, max=DR).cpu().numpy()
        d = img - truth_np
        s = [x for x in sweep if abs(x["T"] - T) < 1e-6]
        s = s[0]["ssim"] if s else float(ssim_metric(
            torch.from_numpy(img)[None, None], truth_t.cpu()[None, None], DR))
        c = j + 1
        ax[0, c].imshow(img, cmap="gray", vmin=0, vmax=DR)
        tag = " (production)" if abs(T) < 1e-6 else ""
        ax[0, c].set_title(f"T={T:+.3f}{tag}\nSSIM={s:.4f}", fontsize=11)
        ax[1, c].imshow(d, cmap="seismic", vmin=-0.015, vmax=0.015)
        ax[1, c].set_title(f"recon - GT  RMSE={np.sqrt((d**2).mean()):.4f}", fontsize=10)
    for a_ in ax.ravel():
        a_.set_xticks([]); a_.set_yticks([])
    fig.suptitle("L277: recon (top) + difference (bottom) vs single lower-clip T, calibration fixed",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_dir / "cutoff_L277_v2_images.png", dpi=110, bbox_inches="tight")
    plt.close(fig)

    (out_dir / "cutoff_L277_v2.json").write_text(json.dumps({
        "patient": PAT, "a": float(a), "bg": float(bg_pred),
        "raw_min": float(raw_t.min()), "production_T0": base, "best": best,
        "sweep": sweep,
    }, indent=2))
    print(f"[L277-v2] wrote cutoff_L277_v2.json + _curve.png + _images.png", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
