"""Verify the intensity-calibration OFFSET error across all 10 Wagner Mayo patients.

L277 showed the production calibration maps recon background -> 0 while truth's
background is ~+0.0013 mu, so an offset correction gains +0.042 full-image SSIM
and the (0,0) threshold max was an artifact. Per user: confirm this is
SYSTEMATIC across all 10 Wagner patients (Mayo only; do NOT touch the shared
metric -- other datasets have different/unknown background scaling).

For each patient: truncation-corrected HD FBP, replicate the production
calibration (intensity_calibrate logic: fg_thr=0.05*dr, bg_mask=truth<=fg_thr,
a=fg_truth/(fg_pred-bg_pred), cal=a*(raw-bg_pred)  [maps bg->0]), then:
  - air-corner means: truth vs cal (the offset gap),
  - coarse full-image SSIM surface over (offset d, threshold T):
        img = clamp(cal + d, min=T, max=dr),
    production = (d=0, T=0); report the max and its offset.

Output:
  results/mayo_debug/offset_allwagner.json
  results/mayo_debug/offset_allwagner.png   (per-patient: prod vs best SSIM, air gap)
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

from ddssl_ldct.metrics import ssim as ssim_metric
from scripts.validate_mayo_helix2fan import _load_truth_slice_for_z
from scripts.compare_gt_hd_ld_fbp_wagner_trunc import _load_slab, _fbp_slab

WAGNER = ["L145", "L186", "L209", "L219", "L277", "L014", "L056", "L058", "L075", "L123"]
SPLIT = {**{p: "train" for p in ["L145", "L186", "L209", "L219"]}, "L277": "val",
         **{p: "test" for p in ["L014", "L056", "L058", "L075", "L123"]}}
DR = 0.05


def main() -> int:
    root = Path(os.environ.get("AGENT4CT_DATA", "/cluster/maier/Agent4CT/data")) / "mayo_ldct"
    sino_dir = root / os.environ.get("STAGED_HELIX2FAN_SUBDIR", "staged_helix2fan_v3")
    truth_root = root / "raw"
    out_dir = REPO / "results" / "mayo_debug"; out_dir.mkdir(parents=True, exist_ok=True)
    dev = "cuda"
    offs = np.round(np.linspace(-0.004, 0.004, 17), 5)
    Ts = np.round(np.linspace(-0.003, 0.003, 13), 5)
    cs = 110

    rows = []
    for pat in WAGNER:
        gj = sino_dir / f"{pat}_sino_fulldose_geometry.json"
        zg = sino_dir / f"{pat}_sino_fulldose_z_grid.npy"
        sh = sino_dir / f"{pat}_sino_fulldose.h5"
        if not (gj.exists() and zg.exists() and sh.exists()):
            print(f"[off] {pat}: missing inputs, skip", flush=True); continue
        geom = json.loads(gj.read_text()); zgrid = np.load(zg)
        slab, zc = _load_slab(sh, geom, zgrid, 3.5, 2)
        ti = _load_truth_slice_for_z(truth_root / pat, zc)
        if ti is None:
            print(f"[off] {pat}: no truth, skip", flush=True); continue
        truth, _, _, tmeta = ti
        ps_eff = 0.700857 * (float(tmeta["pixel_spacing"]) / 0.703125)
        raw = _fbp_slab(slab, geom, ps_eff, dev, 384, 0.02)
        raw_t = torch.from_numpy(raw).to(dev).float()
        truth_t = torch.from_numpy(truth).to(dev).float()

        # production calibration (maps bg_pred -> 0)
        fg_thr = 0.05 * DR
        fg = truth_t > fg_thr; bg = ~fg
        bg_pred = raw_t[bg].mean(); fg_pred = raw_t[fg].mean(); fg_truth = truth_t[fg].mean()
        a = (fg_truth / (fg_pred - bg_pred).clamp_min(1e-9))
        cal = a * (raw_t - bg_pred)
        truth_b = truth_t[None, None]

        # air corners
        H, W = truth.shape
        corn = np.zeros((H, W), bool)
        corn[:cs, :cs] = corn[:cs, -cs:] = corn[-cs:, :cs] = corn[-cs:, -cs:] = True
        cm = torch.from_numpy(corn).to(dev)
        t_air = float(truth_t[cm].mean()); c_air = float(cal[cm].mean())

        # surface
        S = np.zeros((len(Ts), len(offs)), np.float32)
        for i, T in enumerate(Ts):
            for j, d in enumerate(offs):
                img = (cal + float(d)).clamp(min=float(T), max=DR)
                S[i, j] = float(ssim_metric(img[None, None], truth_b, DR).cpu())
        i0 = int(np.argmin(np.abs(Ts))); j0 = int(np.argmin(np.abs(offs)))
        prod = float(S[i0, j0]); bi, bj = np.unravel_index(int(np.argmax(S)), S.shape)
        rows.append({"patient": pat, "split": SPLIT[pat], "ps": float(tmeta["pixel_spacing"]),
                     "truth_air": t_air, "cal_air": c_air, "air_gap": t_air - c_air,
                     "prod_ssim": prod, "max_ssim": float(S[bi, bj]),
                     "best_offset": float(offs[bj]), "best_T": float(Ts[bi]),
                     "dssim": float(S[bi, bj]) - prod})
        r = rows[-1]
        print(f"[off] {pat} ({r['split']:5}) air gap(t-c)={r['air_gap']:+.5f}  "
              f"prod={prod:.4f} -> max={r['max_ssim']:.4f} (+{r['dssim']:.4f}) "
              f"@ off={r['best_offset']:+.4f} T={r['best_T']:+.4f}", flush=True)

    # aggregate
    gaps = np.array([r["air_gap"] for r in rows])
    dss = np.array([r["dssim"] for r in rows])
    boff = np.array([r["best_offset"] for r in rows])
    print(f"\n[off] AGGREGATE n={len(rows)}: air_gap mean={gaps.mean():+.5f} "
          f"(all>0: {bool((gaps>0).all())})  dSSIM mean=+{dss.mean():.4f}  "
          f"best_offset mean={boff.mean():+.5f} (all>0: {bool((boff>0).all())})", flush=True)

    # figure
    fig, ax = plt.subplots(1, 2, figsize=(15, 5.5))
    x = np.arange(len(rows)); labs = [r["patient"] for r in rows]
    ax[0].bar(x - 0.2, [r["prod_ssim"] for r in rows], 0.4, label="production (offset=0)")
    ax[0].bar(x + 0.2, [r["max_ssim"] for r in rows], 0.4, label="offset-corrected max")
    ax[0].set_xticks(x); ax[0].set_xticklabels(labs, rotation=45, fontsize=8)
    ax[0].set_ylabel("full-image SSIM"); ax[0].legend(); ax[0].set_ylim(0.5, 1.0)
    ax[0].set_title(f"production vs offset-corrected (mean ΔSSIM +{dss.mean():.4f})")
    ax[1].bar(x, gaps * 1e3, color="C2")
    ax[1].axhline(0, c="k", lw=0.8)
    ax[1].set_xticks(x); ax[1].set_xticklabels(labs, rotation=45, fontsize=8)
    ax[1].set_ylabel("air-level gap (truth − recon) ×10⁻³ μ")
    ax[1].set_title("recon background sits BELOW truth's (gap>0 everywhere ⇒ systematic)")
    fig.suptitle("Wagner 10-patient: intensity-calibration offset error (Mayo only)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_dir / "offset_allwagner.png", dpi=120, bbox_inches="tight")
    plt.close(fig)

    (out_dir / "offset_allwagner.json").write_text(json.dumps({
        "offsets": offs.tolist(), "thresholds": Ts.tolist(),
        "aggregate": {"air_gap_mean": float(gaps.mean()), "air_gap_all_pos": bool((gaps > 0).all()),
                       "dssim_mean": float(dss.mean()), "best_offset_mean": float(boff.mean()),
                       "best_offset_all_pos": bool((boff > 0).all()), "n": len(rows)},
        "patients": rows,
    }, indent=2))
    print("[off] wrote offset_allwagner.png + .json", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
