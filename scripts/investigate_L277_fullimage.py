"""L277 FULL-IMAGE interrogation: no-threshold proof + air texture + offset×threshold surface.

User direction (2026-06-13): only L277; full image (NOT body-masked); the air
has clear texture (a processing artifact, not scatter); thresholding +
intensity calibration are suspect; the SSIM max at (0,0) is fishy. Wants a
full-image surface and proof the swept data isn't ReLU'd.

Operates on the cached calibrated array (results/mayo_debug/cutoff_L277_v2_arrays.npz:
cal = a*(raw-bg) UNCLAMPED, truth, a, bg, dr). No reconstruction.

Produces:
  1. L277_air_verify.png:
       - histogram of cal vs truth (full + zoom on the air/low region): a ReLU
         shows a hard spike at 0 with nothing below; we show the actual tails.
       - air-corner crops (truth vs cal, tight window) + mean/std, so the air
         texture and any recon-vs-truth air-level offset are visible.
  2. L277_fullimage_surface.png + .json:
       full-image SSIM over (intensity OFFSET d, lower THRESHOLD T), gain fixed:
         img = clamp(cal + d, min=T, max=dr)
       Production = (d=0, T=0). If the optimum sits at d!=0, the calibration
       OFFSET is off (i.e. the recon air is not aligned to truth's air) and the
       (0,0) peak was an artifact of that mis-set offset.
"""
from __future__ import annotations

import json
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


def main() -> int:
    out_dir = REPO / "results" / "mayo_debug"
    z = np.load(out_dir / "cutoff_L277_v2_arrays.npz")
    cal = z["cal"].astype(np.float64)
    truth = z["truth"].astype(np.float64)
    a = float(z["a"]); bg = float(z["bg"]); dr = float(z["dr"])
    H, W = truth.shape
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    # ---- air-region (image corners, far from the centred patient) ----
    cs = 110
    corners = np.zeros((H, W), bool)
    corners[:cs, :cs] = corners[:cs, -cs:] = corners[-cs:, :cs] = corners[-cs:, -cs:] = True
    air_t = truth[corners]; air_c = cal[corners]
    print(f"[full] AIR corners: truth mean={air_t.mean():+.5f} std={air_t.std():.5f} "
          f"min={air_t.min():+.5f}  |  cal mean={air_c.mean():+.5f} std={air_c.std():.5f} "
          f"min={air_c.min():+.5f}", flush=True)
    print(f"[full] cal: min={cal.min():+.5f} frac<0={ (cal<0).mean()*100:.1f}%  "
          f"frac==0 (relu spike?)={ (cal==0).mean()*100:.2f}%", flush=True)
    print(f"[full] truth: min={truth.min():+.6f} frac<0={(truth<0).mean()*100:.1f}%  "
          f"frac==0={(truth==0).mean()*100:.2f}%", flush=True)

    # ===== Figure 1: histogram + air crops =====
    fig, ax = plt.subplots(2, 3, figsize=(17, 9.5))
    bins = np.linspace(-0.006, 0.05, 400)
    ax[0, 0].hist(cal.ravel(), bins=bins, histtype="step", log=True, label="cal (recon)")
    ax[0, 0].hist(truth.ravel(), bins=bins, histtype="step", log=True, label="truth")
    ax[0, 0].axvline(0, ls="--", c="k", alpha=0.5)
    ax[0, 0].set_title("full histogram (log y)"); ax[0, 0].legend(fontsize=8)
    ax[0, 0].set_xlabel("mu")
    bz = np.linspace(-0.005, 0.008, 300)
    ax[0, 1].hist(cal.ravel(), bins=bz, histtype="step", log=True, label="cal")
    ax[0, 1].hist(truth.ravel(), bins=bz, histtype="step", log=True, label="truth")
    ax[0, 1].axvline(0, ls="--", c="k", alpha=0.5)
    ax[0, 1].set_title("zoom on air/low region\n(ReLU would be a spike AT 0, nothing below)")
    ax[0, 1].legend(fontsize=8); ax[0, 1].set_xlabel("mu")
    ax[0, 2].hist(air_c, bins=bz, histtype="step", log=True, label="cal air-corners")
    ax[0, 2].hist(air_t, bins=bz, histtype="step", log=True, label="truth air-corners")
    ax[0, 2].axvline(0, ls="--", c="k", alpha=0.5)
    ax[0, 2].axvline(air_c.mean(), ls=":", c="C0"); ax[0, 2].axvline(air_t.mean(), ls=":", c="C1")
    ax[0, 2].set_title(f"AIR-corner histogram\ncal mean={air_c.mean():+.4f} vs truth mean={air_t.mean():+.4f}")
    ax[0, 2].legend(fontsize=8); ax[0, 2].set_xlabel("mu")
    # air crop images (top-left corner), tight window to reveal texture
    win = 0.003
    tl_t = truth[:cs, :cs]; tl_c = cal[:cs, :cs]
    im10 = ax[1, 0].imshow(tl_t, cmap="gray", vmin=-win, vmax=win)
    ax[1, 0].set_title(f"truth air corner [±{win}]\nstd={tl_t.std():.5f}"); fig.colorbar(im10, ax=ax[1,0], fraction=.046)
    im11 = ax[1, 1].imshow(tl_c, cmap="gray", vmin=-win, vmax=win)
    ax[1, 1].set_title(f"recon (cal) air corner [±{win}]\nstd={tl_c.std():.5f}"); fig.colorbar(im11, ax=ax[1,1], fraction=.046)
    im12 = ax[1, 2].imshow(tl_c - tl_t, cmap="seismic", vmin=-win, vmax=win)
    ax[1, 2].set_title("recon - truth, air corner"); fig.colorbar(im12, ax=ax[1,2], fraction=.046)
    for a_ in (ax[1,0], ax[1,1], ax[1,2]):
        a_.set_xticks([]); a_.set_yticks([])
    fig.suptitle(f"L277 full-image air/threshold verification (a={a:.3f}, bg={bg:+.4f})", fontsize=12)
    fig.tight_layout(rect=[0,0,1,0.97])
    fig.savefig(out_dir / "L277_air_verify.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    print("[full] wrote L277_air_verify.png", flush=True)

    # ===== Figure 2: full-image SSIM surface over (offset, threshold) =====
    cal_t = torch.from_numpy(cal).float().to(dev)
    truth_t = torch.from_numpy(truth).float().to(dev)[None, None]
    offs = np.round(np.linspace(-0.004, 0.004, 33), 5)
    Ts = np.round(np.linspace(-0.004, 0.004, 33), 5)
    S = np.zeros((len(Ts), len(offs)), np.float32)
    for i, T in enumerate(Ts):
        for j, d in enumerate(offs):
            img = (cal_t + float(d)).clamp(min=float(T), max=dr)
            S[i, j] = float(ssim_metric(img[None, None], truth_t, dr).cpu())
    i0 = int(np.argmin(np.abs(Ts - 0))); j0 = int(np.argmin(np.abs(offs - 0)))
    prod = float(S[i0, j0]); bi, bj = np.unravel_index(int(np.argmax(S)), S.shape)
    print(f"[full] production (offset=0,T=0) SSIM={prod:.4f}", flush=True)
    print(f"[full] grid-max SSIM={S[bi,bj]:.4f} @ offset={offs[bj]:+.5f} T={Ts[bi]:+.5f}  "
          f"(dSSIM {S[bi,bj]-prod:+.4f})", flush=True)

    OFF, TT = np.meshgrid(offs, Ts)
    fig = plt.figure(figsize=(17, 6.5))
    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    ax1.plot_surface(OFF*1e3, TT*1e3, S, cmap="viridis", linewidth=0, antialiased=True)
    ax1.set_xlabel("intensity offset d (×10⁻³ μ)"); ax1.set_ylabel("threshold T (×10⁻³ μ)")
    ax1.set_zlabel("full-image SSIM"); ax1.view_init(elev=28, azim=-120)
    ax1.set_title("full-image SSIM(offset, threshold)")
    ax2 = fig.add_subplot(1, 2, 2)
    pm = ax2.pcolormesh(offs*1e3, Ts*1e3, S, cmap="viridis", shading="auto")
    fig.colorbar(pm, ax=ax2, label="full-image SSIM")
    cs2 = ax2.contour(offs*1e3, Ts*1e3, S, levels=12, colors="w", linewidths=0.5, alpha=0.6)
    ax2.clabel(cs2, inline=True, fontsize=6, fmt="%.2f")
    ax2.plot(0, 0, "m*", ms=16, label=f"production (0,0)={prod:.3f}")
    ax2.plot(offs[bj]*1e3, Ts[bi]*1e3, "ro", ms=9, mfc="none", label=f"max={S[bi,bj]:.3f}@d={offs[bj]:+.4f}")
    ax2.set_xlabel("intensity offset d (×10⁻³ μ)"); ax2.set_ylabel("threshold T (×10⁻³ μ)")
    ax2.set_title("heatmap — is (0,0) the true max, or is offset≠0 better?")
    ax2.legend(fontsize=7, loc="lower left")
    fig.suptitle("L277 FULL-IMAGE surface: intensity offset × lower threshold (gain fixed)", fontsize=12)
    fig.tight_layout(rect=[0,0,1,0.96])
    fig.savefig(out_dir / "L277_fullimage_surface.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

    (out_dir / "L277_fullimage_surface.json").write_text(json.dumps({
        "a": a, "bg": bg, "dr": dr,
        "air_corner": {"truth_mean": float(air_t.mean()), "truth_std": float(air_t.std()),
                        "cal_mean": float(air_c.mean()), "cal_std": float(air_c.std())},
        "cal_min": float(cal.min()), "cal_frac_neg": float((cal<0).mean()),
        "cal_frac_zero": float((cal==0).mean()),
        "truth_min": float(truth.min()), "truth_frac_neg": float((truth<0).mean()),
        "offsets": offs.tolist(), "thresholds": Ts.tolist(), "ssim": S.tolist(),
        "production_ssim": prod,
        "grid_max": {"ssim": float(S[bi,bj]), "offset": float(offs[bj]), "threshold": float(Ts[bi])},
    }, indent=2))
    print("[full] wrote L277_fullimage_surface.png + .json", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
