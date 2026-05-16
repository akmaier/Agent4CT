"""Reference: NAF — Neural Attenuation Fields for Sparse-View CT.

Per-scene neural-implicit-representation reconstruction. A small MLP with
positional encoding maps 2-D pixel coordinates → linear attenuation μ.
The MLP weights are optimised against the measured sinogram via
‖R · μ(x) − g‖², a la NeRF (but on rays through 2-D pixels for fan-beam
CT). No training set; one optimisation per scan.

Adapted from:
  Zha R., Zhang Y., Li H. "NAF: Neural Attenuation Fields for Sparse-View
  CBCT Reconstruction." MICCAI 2022. doi:10.1007/978-3-031-16446-0_42
  (arXiv:2209.14540, code: github.com/Ruyi-Zha/naf_cbct)

For tractability with 100 val samples in a 20-iter random search we
default val_n=20 and 600 inner-optimisation iterations.
"""
from __future__ import annotations
import argparse, json, math, os, sys, time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import torch
import torch.nn as nn
import torch.nn.functional as F
from ddssl_ldct.geometry import FanBeamGeometry
from ddssl_ldct.pyronn_projector import PyronnFanBeamProjector
from ddssl_ldct.phantoms import random_ellipses_phantom
from ddssl_ldct.simulate import simulate_low_dose
from ddssl_ldct.metrics import psnr, ssim


CONFIG = {
    "image_size": 512, "pixel_spacing": 0.7,
    "n_angles": 128, "n_det": 736, "det_spacing": 1.2858,
    "sod": 595.0, "sdd": 1085.6,
    "val_n": 20, "noise_i0": 1e5, "noise_sigma_e": 10.0, "seed": 42,
    "display_min": 0.0, "display_max": 0.05,
    "naf_n_freqs": 10,         # positional-encoding frequency bands
    "naf_hidden": 128,
    "naf_layers": 4,
    "naf_n_iter": 600,
    "naf_lr": 5e-3,
    "naf_tv_weight": 1e-4,
    "naf_n_clip": 0.05,
}


class PosEnc(nn.Module):
    def __init__(self, n_freqs):
        super().__init__()
        self.freqs = 2 ** torch.arange(n_freqs).float() * math.pi

    def forward(self, x):                       # x: (..., 2) in [-1, 1]
        f = self.freqs.to(x.device).view(*([1] * x.dim()), -1)
        xs = x.unsqueeze(-1) * f                # (..., 2, n_freqs)
        return torch.cat([x, xs.sin().flatten(-2), xs.cos().flatten(-2)], dim=-1)


class NAF(nn.Module):
    def __init__(self, n_freqs=10, hidden=128, layers=4):
        super().__init__()
        self.pe = PosEnc(n_freqs)
        in_dim = 2 + 2 * 2 * n_freqs
        m = [nn.Linear(in_dim, hidden), nn.ReLU(inplace=True)]
        for _ in range(layers - 2):
            m += [nn.Linear(hidden, hidden), nn.ReLU(inplace=True)]
        m += [nn.Linear(hidden, 1), nn.Softplus()]
        self.mlp = nn.Sequential(*m)

    def forward(self, coords):                  # coords: (N, 2)
        return self.mlp(self.pe(coords)).squeeze(-1)


def _coord_grid(H, W, device):
    yy, xx = torch.meshgrid(
        torch.linspace(-1.0, 1.0, H, device=device),
        torch.linspace(-1.0, 1.0, W, device=device),
        indexing="ij",
    )
    return torch.stack([xx, yy], dim=-1).reshape(-1, 2)


def _tv(img):
    dx = (img[..., 1:, :] - img[..., :-1, :]).abs().mean()
    dy = (img[..., :, 1:] - img[..., :, :-1]).abs().mean()
    return dx + dy


def fit_one_scene(noisy_sino, geom, cfg, device):
    """One NAF optimisation against one sinogram. Returns (H, W) μ image."""
    H = W = cfg["image_size"]
    proj = PyronnFanBeamProjector(geom).to(device)
    model = NAF(cfg["naf_n_freqs"], cfg["naf_hidden"], cfg["naf_layers"]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["naf_lr"])
    coords = _coord_grid(H, W, device)
    for it in range(cfg["naf_n_iter"]):
        mu = model(coords).reshape(1, 1, H, W).clamp(0.0, cfg["naf_n_clip"])
        sino_pred = proj.forward_project(mu)
        loss = F.mse_loss(sino_pred, noisy_sino) + cfg["naf_tv_weight"] * _tv(mu)
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        mu = model(coords).reshape(1, 1, H, W).clamp(0.0, cfg["naf_n_clip"])
    return mu


def main(out_dir: Path, cfg: dict | None = None) -> dict:
    env_path = os.environ.get("NAF_CONFIG_PATH")
    if env_path and Path(env_path).exists():
        cfg = {**CONFIG, **json.loads(Path(env_path).read_text()), **(cfg or {})}
    else:
        cfg = {**CONFIG, **(cfg or {})}
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(cfg["seed"])
    print(f"[solver] device={device}  cfg={json.dumps({k:v for k,v in cfg.items() if k.startswith('naf')}, default=str)}", flush=True)

    geom = FanBeamGeometry(
        image_size=cfg["image_size"], pixel_spacing=cfg["pixel_spacing"],
        n_angles=cfg["n_angles"], n_det=cfg["n_det"],
        det_spacing=cfg["det_spacing"], sod=cfg["sod"], sdd=cfg["sdd"])
    proj_full = PyronnFanBeamProjector(geom).to(device)

    # Build val set
    phs = torch.stack([
        random_ellipses_phantom(size=geom.image_size, n_ellipses=10,
                                seed=cfg["seed"] + 1000 + i)[0]
        for i in range(cfg["val_n"])
    ]).to(device)
    with torch.no_grad():
        cleans = proj_full.forward_project(phs)
        noisys = simulate_low_dose(cleans, i0=cfg["noise_i0"],
                                   sigma_e=cfg["noise_sigma_e"],
                                   seed=cfg["seed"] + 10_000)
        fbps = torch.clamp(proj_full.fbp(noisys), min=0.0)

    t0 = time.time()
    preds = []
    for i in range(cfg["val_n"]):
        s = noisys[i:i+1]
        pred_i = fit_one_scene(s, geom, cfg, device)
        preds.append(pred_i.detach())
        if (i + 1) % 5 == 0:
            elapsed = time.time() - t0
            print(f"[fit] {i+1}/{cfg['val_n']}  elapsed={elapsed:.1f}s", flush=True)
        if time.time() - t0 > 600:
            print(f"[fit] 10-min wall at sample {i+1}", flush=True)
            break
    train_time = time.time() - t0
    pred = torch.cat(preds, 0)
    val_ph = phs[:pred.shape[0]]; val_fbp = fbps[:pred.shape[0]]

    dr = cfg["display_max"] - cfg["display_min"]
    val_psnr = float(psnr(pred, val_ph, data_range=dr).cpu())
    val_ssim = float(ssim(pred, val_ph, data_range=dr).cpu())
    val_rmse = float(((pred - val_ph) ** 2).mean().sqrt().cpu())
    baseline_psnr = float(psnr(val_fbp, val_ph, data_range=dr).cpu())
    baseline_rmse = float(((val_fbp - val_ph) ** 2).mean().sqrt().cpu())
    headroom = max(0.0, 1.0 - val_rmse / max(baseline_rmse, 1e-12))
    print(f"[solver] NAF: hr={headroom:.4f} SSIM={val_ssim:.4f} PSNR={val_psnr:.2f}",
          flush=True)

    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        n_show = min(3, pred.shape[0])
        fig, ax = plt.subplots(n_show, 4, figsize=(12, 3 * n_show))
        if n_show == 1: ax = ax[None]
        vmin, vmax = cfg["display_min"], cfg["display_max"]
        for i in range(n_show):
            ax[i, 0].imshow(val_ph[i, 0].cpu(), cmap="gray", vmin=vmin, vmax=vmax)
            ax[i, 0].set_title("truth" if i == 0 else "")
            ax[i, 1].imshow(val_fbp[i, 0].cpu(), cmap="gray", vmin=vmin, vmax=vmax)
            ax[i, 1].set_title(f"FBP (PSNR={baseline_psnr:.1f})" if i == 0 else "")
            ax[i, 2].imshow(pred[i, 0].cpu(), cmap="gray", vmin=vmin, vmax=vmax)
            ax[i, 2].set_title(f"NAF (PSNR={val_psnr:.1f} SSIM={val_ssim:.3f})"
                               if i == 0 else "")
            ax[i, 3].imshow((pred[i, 0] - val_ph[i, 0]).cpu(),
                            cmap="RdBu_r", vmin=-0.01, vmax=0.01)
            ax[i, 3].set_title("residual" if i == 0 else "")
            for a in ax[i]: a.set_axis_off()
        plt.tight_layout(); plt.savefig(out_dir / "comparison.png", dpi=120)
    except Exception as e:
        print(f"[solver] figure failed: {e}", flush=True)

    result = {
        "val_score": val_ssim, "val_psnr": val_psnr, "val_ssim": val_ssim,
        "val_rmse": val_rmse, "baseline_psnr": baseline_psnr,
        "baseline_rmse": baseline_rmse, "headroom": headroom,
        "params_M": sum(p.numel() for p in NAF(cfg["naf_n_freqs"],
                                                cfg["naf_hidden"],
                                                cfg["naf_layers"])
                        .parameters()) / 1e6,
        "train_n": 0, "val_n": pred.shape[0],
        "train_time_s": train_time, "config": cfg,
    }
    (out_dir / "result.json").write_text(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("out_dir")
    args = p.parse_args(); main(Path(args.out_dir))
