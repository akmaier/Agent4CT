"""L277 SCALING-CALIBRATION investigation (resumes the original shadow question).

The cutoff sweeps were a negative result: the lower clip can't touch the body
tissue where the posterior shadow lives. Throughout those we FROZE the
calibration. This script finally probes the calibration itself.

The production calibration is a GLOBAL two-point affine
    cal = a * (raw - bg)
(one gain + one offset for the whole slice). A global affine cannot correct a
spatially-varying intensity error. If the posterior shadow is a smooth
low-frequency bias, it shows up as structure in the BODY residual cal - truth
that a global affine left behind.

We:
  1. body-mask (truth > 0.005 mu, excludes air/table-ish low values),
  2. residual r = cal - truth over the body: RMSE / mean / std + the
     anterior-posterior (per-row) mean profile (the shadow as a number),
  3. fit a smooth spatially-varying correction (2D polynomial, degree 3) to r
     over the body, subtract it -> cal_corr, and report body RMSE/SSIM
     before/after. If a smooth correction removes most of the residual, the
     shadow is a CALIBRATION deficiency (global affine too rigid); if the
     residual is high-frequency / structural, it's a real recon bias.

Loads results/mayo_debug/cutoff_L277_v2_arrays.npz (cal, truth, a, bg, dr).
Output:
  results/mayo_debug/scaling_calib_L277.png
  results/mayo_debug/scaling_calib_L277.json
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

from ddssl_ldct.metrics import ssim as ssim_metric, psnr as psnr_metric

BODY_THR = 0.005    # mu; body = truth above this (excludes air/very-low)
POLY_DEG = 3


def _poly_design(yy, xx, deg):
    """2D polynomial design matrix columns up to total degree `deg`."""
    cols = []
    for dy in range(deg + 1):
        for dx in range(deg + 1 - dy):
            cols.append((yy ** dy) * (xx ** dx))
    return np.stack(cols, axis=-1)   # (..., n_terms)


def _masked_metrics(img, truth, mask, dr):
    m = mask
    rmse = float(np.sqrt(((img[m] - truth[m]) ** 2).mean()))
    mean = float((img[m] - truth[m]).mean())
    # masked SSIM: zero outside body for both (so windows off-body match);
    # report on the full frame but it's dominated by the body now.
    it = torch.from_numpy((img * m).astype(np.float32))[None, None]
    tt = torch.from_numpy((truth * m).astype(np.float32))[None, None]
    ss = float(ssim_metric(it, tt, dr).cpu())
    return rmse, mean, ss


def main() -> int:
    out_dir = REPO / "results" / "mayo_debug"
    z = np.load(out_dir / "cutoff_L277_v2_arrays.npz")
    cal = z["cal"].astype(np.float64)
    truth = z["truth"].astype(np.float64)
    a = float(z["a"]); bg = float(z["bg"]); dr = float(z["dr"])
    H, W = truth.shape
    body = truth > BODY_THR
    print(f"[calib] a={a:.4f} bg={bg:+.5f} dr={dr}  body frac={body.mean()*100:.1f}%", flush=True)

    # production global-affine residual over the body
    r = cal - truth
    rmse0, mean0, ssim0 = _masked_metrics(cal.astype(np.float32),
                                           truth.astype(np.float32), body, dr)
    print(f"[calib] GLOBAL affine  body RMSE={rmse0:.5f}  mean_resid={mean0:+.5f}  "
          f"masked SSIM={ssim0:.4f}", flush=True)

    # anterior-posterior profile: per-row mean residual over the body
    ap = np.full(H, np.nan)
    for y in range(H):
        row = body[y]
        if row.any():
            ap[y] = r[y][row].mean()
    yvalid = np.where(~np.isnan(ap))[0]
    ap_span = (np.nanmax(ap), np.nanmin(ap))
    print(f"[calib] AP residual profile: max={ap_span[0]:+.5f} min={ap_span[1]:+.5f} "
          f"(span {ap_span[0]-ap_span[1]:.5f})", flush=True)

    # spatially-varying correction: fit smooth 2D poly to residual over body
    ys, xs = np.mgrid[0:H, 0:W].astype(np.float64)
    yc = (ys - (H - 1) / 2) / (H / 2)        # normalised coords [-1,1]
    xc = (xs - (W - 1) / 2) / (W / 2)
    A = _poly_design(yc[body], xc[body], POLY_DEG)   # (Nbody, n_terms)
    coef, *_ = np.linalg.lstsq(A, r[body], rcond=None)
    Afull = _poly_design(yc, xc, POLY_DEG)            # (H,W,n_terms)
    smooth_bias = (Afull @ coef)                      # (H,W)
    cal_corr = cal - smooth_bias
    rmse1, mean1, ssim1 = _masked_metrics(cal_corr.astype(np.float32),
                                          truth.astype(np.float32), body, dr)
    print(f"[calib] +poly(deg{POLY_DEG}) body RMSE={rmse1:.5f} (was {rmse0:.5f}, "
          f"-{(1-rmse1/rmse0)*100:.0f}%)  masked SSIM={ssim1:.4f} (was {ssim0:.4f})  "
          f"bias span={smooth_bias[body].max()-smooth_bias[body].min():.5f}", flush=True)

    # ---- figure ----
    rlim = 0.012
    fig, ax = plt.subplots(2, 3, figsize=(16, 10))
    ax[0, 0].imshow(truth, cmap="gray", vmin=0, vmax=dr); ax[0, 0].set_title("L277 GT", fontsize=11)
    ax[0, 1].imshow(cal, cmap="gray", vmin=0, vmax=dr)
    ax[0, 1].set_title(f"global-affine recon\nbody RMSE={rmse0:.4f} SSIM={ssim0:.3f}", fontsize=11)
    rm = np.where(body, r, np.nan)
    im2 = ax[0, 2].imshow(rm, cmap="seismic", vmin=-rlim, vmax=rlim)
    ax[0, 2].set_title("residual (recon - GT), body only\nblue = recon BELOW truth", fontsize=11)
    fig.colorbar(im2, ax=ax[0, 2], fraction=0.046)
    sb = np.where(body, smooth_bias, np.nan)
    im3 = ax[1, 0].imshow(sb, cmap="seismic", vmin=-rlim, vmax=rlim)
    ax[1, 0].set_title(f"fitted smooth bias (poly deg{POLY_DEG})", fontsize=11)
    fig.colorbar(im3, ax=ax[1, 0], fraction=0.046)
    ax[1, 1].imshow(cal_corr, cmap="gray", vmin=0, vmax=dr)
    ax[1, 1].set_title(f"spatially-corrected recon\nbody RMSE={rmse1:.4f} SSIM={ssim1:.3f}", fontsize=11)
    r2m = np.where(body, cal_corr - truth, np.nan)
    im5 = ax[1, 2].imshow(r2m, cmap="seismic", vmin=-rlim, vmax=rlim)
    ax[1, 2].set_title("residual after correction, body only", fontsize=11)
    fig.colorbar(im5, ax=ax[1, 2], fraction=0.046)
    for a_ in ax.ravel():
        a_.set_xticks([]); a_.set_yticks([])
    fig.suptitle(f"L277 scaling-calibration residual study  "
                 f"(global affine vs +smooth poly; body=truth>{BODY_THR})", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_dir / "scaling_calib_L277.png", dpi=120, bbox_inches="tight")
    plt.close(fig)

    # AP profile figure
    fig2, axp = plt.subplots(figsize=(8, 4.5))
    axp.plot(yvalid, ap[yvalid] * 1e3, lw=1.5)
    axp.axhline(0, ls="--", c="k", alpha=0.4)
    axp.set_xlabel("row (anterior → posterior, image y)")
    axp.set_ylabel("mean body residual (recon-GT) ×10⁻³ μ")
    axp.set_title("L277 anterior–posterior residual profile (the shadow as a number)")
    axp.grid(alpha=0.3)
    fig2.tight_layout()
    fig2.savefig(out_dir / "scaling_calib_L277_approfile.png", dpi=120, bbox_inches="tight")
    plt.close(fig2)

    (out_dir / "scaling_calib_L277.json").write_text(json.dumps({
        "a": a, "bg": bg, "body_thr": BODY_THR, "poly_deg": POLY_DEG,
        "global_affine": {"body_rmse": rmse0, "mean_resid": mean0, "masked_ssim": ssim0},
        "plus_poly": {"body_rmse": rmse1, "mean_resid": mean1, "masked_ssim": ssim1,
                       "rmse_reduction_pct": (1 - rmse1 / rmse0) * 100,
                       "bias_span": float(smooth_bias[body].max() - smooth_bias[body].min())},
        "ap_profile_span": float(ap_span[0] - ap_span[1]),
        "verdict_hint": "large RMSE reduction + smooth bias => spatially-varying "
                        "calibration deficiency; small reduction => real recon bias",
    }, indent=2))
    print("[calib] wrote scaling_calib_L277.png + _approfile.png + .json", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
