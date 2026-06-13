"""2D SSIM surface over the TWO lower thresholds (L277), calibration fixed.

The pipeline has two lower clips:
  * pre-calibration FBP floor   : clamp(raw, min=pre)   (raw = cal/a + bg)
  * post-calibration output floor: clamp(cal2, min=post, max=dr)
with the affine (a, bg) FROZEN. This sweeps BOTH on a grid and plots
SSIM(pre, post) as a 3D surface + 2D heatmap, so the interaction
("one threshold cuts off the other") is explicit.

Algebra (why they interact): with a>0,
    a*(max(raw,pre) - bg) = max(a*(raw-bg), a*(pre-bg)) = max(cal, a*(pre-bg))
so the output lower bound is max(a*(pre-bg), post): the two clips are BOTH
lower clamps in calibrated space and only the HIGHER one acts. The surface
should show an L-shaped (max) ridge; the SSIM-optimum lies on the
effective-floor = 0 contour.

Loads results/mayo_debug/cutoff_L277_v2_arrays.npz (cal, truth, a, bg, dr),
written by scripts/investigate_cutoff_L277_v2.py. No reconstruction needed.

Output:
  results/mayo_debug/cutoff_L277_surface.png
  results/mayo_debug/cutoff_L277_surface.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (registers 3d proj)
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from ddssl_ldct.metrics import ssim as ssim_metric


def main() -> int:
    out_dir = REPO / "results" / "mayo_debug"
    npz = np.load(out_dir / "cutoff_L277_v2_arrays.npz")
    cal = torch.from_numpy(npz["cal"]).float()
    truth = torch.from_numpy(npz["truth"]).float()
    a = float(npz["a"]); bg = float(npz["bg"]); dr = float(npz["dr"])
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    cal = cal.to(dev); truth = truth.to(dev)
    truth_b = truth[None, None]
    print(f"[surf] a={a:.4f} bg={bg:+.5f} dr={dr}  cal=[{float(cal.min()):+.4f},"
          f"{float(cal.max()):.4f}]  dev={dev}", flush=True)

    # grids: pre in RAW-FBP units, post in CALIBRATED units (both lower floors)
    pre_grid = np.round(np.linspace(-0.003, 0.008, 23), 5)
    post_grid = np.round(np.linspace(-0.005, 0.008, 27), 5)
    S = np.zeros((len(post_grid), len(pre_grid)), dtype=np.float32)
    for i, post in enumerate(post_grid):
        for j, pre in enumerate(pre_grid):
            eff_lower = max(a * (pre - bg), float(post))
            out = cal.clamp(min=eff_lower, max=dr)
            S[i, j] = float(ssim_metric(out[None, None], truth_b, dr).cpu())

    i0 = int(np.argmin(np.abs(post_grid - 0.0)))
    j0 = int(np.argmin(np.abs(pre_grid - 0.0)))
    prod_ssim = float(S[i0, j0])
    bi, bj = np.unravel_index(int(np.argmax(S)), S.shape)
    print(f"[surf] production (pre=0,post=0) SSIM={prod_ssim:.4f}", flush=True)
    print(f"[surf] grid-max SSIM={S[bi,bj]:.4f} @ pre={pre_grid[bj]:+.4f} "
          f"post={post_grid[bi]:+.4f}", flush=True)

    PRE, POST = np.meshgrid(pre_grid, post_grid)
    fig = plt.figure(figsize=(17, 6.5))

    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    ax1.plot_surface(PRE * 1e3, POST * 1e3, S, cmap="viridis",
                     linewidth=0, antialiased=True, alpha=0.95)
    ax1.set_xlabel("pre-cal FBP floor (×10⁻³ μ)")
    ax1.set_ylabel("post-cal floor (×10⁻³ μ)")
    ax1.set_zlabel("SSIM")
    ax1.set_title("SSIM surface over the two lower thresholds")
    ax1.view_init(elev=28, azim=-130)

    ax2 = fig.add_subplot(1, 2, 2)
    pm = ax2.pcolormesh(pre_grid * 1e3, post_grid * 1e3, S, cmap="viridis",
                        shading="auto")
    fig.colorbar(pm, ax=ax2, label="SSIM")
    cs = ax2.contour(pre_grid * 1e3, post_grid * 1e3, S, levels=10,
                     colors="w", linewidths=0.5, alpha=0.6)
    ax2.clabel(cs, inline=True, fontsize=6, fmt="%.2f")
    # effective-equal line: a*(pre-bg) == post  -> post = a*(pre-bg)
    pe = pre_grid
    ax2.plot(pe * 1e3, (a * (pe - bg)) * 1e3, "r--", lw=1.5,
             label="a·(pre−bg)=post  (clamps equal)")
    ax2.plot(0.0, 0.0, "m*", ms=16, label=f"production (0,0) SSIM={prod_ssim:.3f}")
    ax2.plot(pre_grid[bj] * 1e3, post_grid[bi] * 1e3, "ko", ms=8, mfc="none",
             label=f"grid max SSIM={S[bi,bj]:.3f}")
    ax2.set_xlabel("pre-cal FBP floor (×10⁻³ μ)")
    ax2.set_ylabel("post-cal floor (×10⁻³ μ)")
    ax2.set_title("SSIM heatmap (L-shaped max ridge ⇒ only the higher clip acts)")
    ax2.legend(fontsize=7, loc="lower left")
    fig.suptitle(f"L277 two-threshold SSIM surface, calibration fixed "
                 f"(a={a:.3f}, bg={bg:+.4f})", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_dir / "cutoff_L277_surface.png", dpi=120, bbox_inches="tight")
    plt.close(fig)

    (out_dir / "cutoff_L277_surface.json").write_text(json.dumps({
        "a": a, "bg": bg, "dr": dr,
        "pre_grid": pre_grid.tolist(), "post_grid": post_grid.tolist(),
        "ssim": S.tolist(),
        "production_ssim": prod_ssim,
        "grid_max": {"ssim": float(S[bi, bj]),
                      "pre": float(pre_grid[bj]), "post": float(post_grid[bi])},
        "note": "output lower bound = max(a*(pre-bg), post); surface depends "
                "only on that combined effective floor.",
    }, indent=2))
    print(f"[surf] wrote cutoff_L277_surface.png + .json", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
