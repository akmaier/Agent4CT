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
from ddssl_ldct.metrics import psnr, ssim, evaluate_calibrated, make_4panel_comparison, supervised_recon_loss, negativity_penalty
from challenges.demo_dl.geometry import DEFAULTS as DEMO_DL_DEFAULTS


CONFIG = {
    **DEMO_DL_DEFAULTS,
    "val_n": 20,
    "naf_n_freqs": 10,         # positional-encoding frequency bands
    "naf_hidden": 192,
    "naf_layers": 5,
    "naf_n_iter": 2000,        # was 600; per-scene Adam needs more for NeRF-style fit
    "naf_lr": 5e-3,
    "naf_tv_weight": 1e-4,
    "naf_n_clip": 0.05,
    # Outer wall (s) for the whole val_n loop; per_scene = outer / val_n.
    "naf_outer_wall_s": 2400,  # 40 min (was 600 s)
    "naf_per_scene_s": None,   # optional override
}


def build_dataset(geom, n, seed, i0, sigma_e, device):
    # Dispatches on AGENT4CT_DATASET / cfg["dataset_kind"]. Phantom path
    # is backwards-compatible; staged paths load from disk.
    from ddssl_ldct.staged_dataset import load_val_split
    import os
    kind = os.environ.get("AGENT4CT_DATASET", "phantoms")
    split = "val" if (seed % 100_000) >= 1000 else "train"
    return load_val_split(kind, split, n, device=device,
                          seed=seed, noise_i0=i0, noise_sigma_e=sigma_e,
                          geom=geom)


class PosEnc(nn.Module):
    def __init__(self, n_freqs):
        super().__init__()
        self.freqs = 2 ** torch.arange(n_freqs).float() * math.pi

    def forward(self, x):                       # x: (..., 2) in [-1, 1]
        f = self.freqs.to(x.device).view(*([1] * x.dim()), -1)
        xs = x.unsqueeze(-1) * f                # (..., 2, n_freqs)
        return torch.cat([x, xs.sin().flatten(-2), xs.cos().flatten(-2)], dim=-1)


class NAF(nn.Module):
    """Coordinate MLP → linear attenuation. Two debug fixes vs v1:

    1.  Output activation = ``sigmoid * out_scale``. Bias-init the
        final-layer linear so the network starts producing ~water-μ
        (0.005 mm⁻¹) everywhere. The old softplus + post-clamp(0, 0.05)
        saturated at the top of the clip on init, killing the gradient
        and locking every scene to a flat 0.05 image — that was the
        hr=0 / SSIM≈0.25 collapse in the 20260516 search.
    2.  Drop the post-clamp during training; the sigmoid head bounds μ
        naturally and gradients flow through it everywhere.
    """

    def __init__(self, n_freqs=10, hidden=128, layers=4, out_scale=0.05,
                 init_mu=0.005):
        super().__init__()
        self.pe = PosEnc(n_freqs)
        self.out_scale = out_scale
        in_dim = 2 + 2 * 2 * n_freqs
        m = [nn.Linear(in_dim, hidden), nn.ReLU(inplace=True)]
        for _ in range(layers - 2):
            m += [nn.Linear(hidden, hidden), nn.ReLU(inplace=True)]
        m.append(nn.Linear(hidden, 1))
        # Bias-init: target sigmoid(b) * out_scale = init_mu  →  b = logit(init_mu/out_scale)
        with torch.no_grad():
            target = max(min(init_mu / out_scale, 0.999), 1e-3)
            b0 = math.log(target / (1.0 - target))
            nn.init.zeros_(m[-1].weight)
            m[-1].bias.fill_(b0)
        self.mlp = nn.Sequential(*m)

    def forward(self, coords):                  # coords: (N, 2)
        h = self.mlp(self.pe(coords)).squeeze(-1)
        return torch.sigmoid(h) * self.out_scale


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


def fit_one_scene(noisy_sino, geom, cfg, device, t_limit=None):
    """One NAF optimisation against one sinogram. Returns (H, W) μ image.

    `t_limit` (seconds) is the per-scene wall clamp — set externally so the
    outer loop can give each scene a fair budget regardless of n_iter.
    """
    H = W = cfg["image_size"]
    proj = PyronnFanBeamProjector(geom).to(device)
    model = NAF(cfg["naf_n_freqs"], cfg["naf_hidden"], cfg["naf_layers"],
                out_scale=cfg["naf_n_clip"], init_mu=0.005).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["naf_lr"])
    coords = _coord_grid(H, W, device)
    t0 = time.time()
    last_loss = float("inf")
    for it in range(cfg["naf_n_iter"]):
        mu = model(coords).reshape(1, 1, H, W)
        sino_pred = proj.forward_project(mu)
        loss = F.mse_loss(sino_pred, noisy_sino) + cfg["naf_tv_weight"] * _tv(mu) + 1.0 * negativity_penalty(mu)
        opt.zero_grad(); loss.backward(); opt.step()
        last_loss = float(loss.detach().cpu())
        if t_limit is not None and (time.time() - t0) > t_limit:
            break
    with torch.no_grad():
        mu = model(coords).reshape(1, 1, H, W)
    return mu, last_loss, it + 1


def main(out_dir: Path, cfg: dict | None = None) -> dict:
    env_path = os.environ.get("NAF_CONFIG_PATH")
    if env_path and Path(env_path).exists():
        cfg = {**CONFIG, **json.loads(Path(env_path).read_text()), **(cfg or {})}
    else:
        cfg = {**CONFIG, **(cfg or {})}
    # Dataset dispatch (Track B/C of workplan). When dataset_kind != "phantoms"
    # we override the geometry to match the staged data.
    from ddssl_ldct.staged_dataset import get_dataset_kind, geometry_overrides
    cfg["dataset_kind"] = get_dataset_kind(cfg)
    if cfg["dataset_kind"] != "phantoms":
        cfg.update(geometry_overrides(cfg["dataset_kind"]))
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(cfg["seed"])
    print(f"[solver] device={device}  cfg={json.dumps({k:v for k,v in cfg.items() if k.startswith('naf')}, default=str)}", flush=True)

    geom = FanBeamGeometry(
        image_size=cfg["image_size"], pixel_spacing=cfg["pixel_spacing"],
        n_angles=cfg["n_angles"], n_det=cfg["n_det"],
        det_spacing=cfg["det_spacing"], sod=cfg["sod"], sdd=cfg["sdd"])
    proj_full = PyronnFanBeamProjector(geom).to(device)

    # Build val set (dispatches to phantom path or staged path based on env).
    phs, _, noisys = build_dataset(
        geom, cfg["val_n"], cfg["seed"] + 1000,
        cfg["noise_i0"], cfg["noise_sigma_e"], device)
    with torch.no_grad():
        fbps = torch.clamp(proj_full.fbp(noisys), min=0.0)

    outer_wall = cfg.get("naf_outer_wall_s", 600)            # 10 min default; 4x = 2400
    per_scene_wall = cfg.get("naf_per_scene_s", outer_wall / max(1, cfg["val_n"]))
    t0 = time.time()
    preds = []
    for i in range(cfg["val_n"]):
        s = noisys[i:i+1]
        pred_i, last_loss, n_done = fit_one_scene(s, geom, cfg, device,
                                                   t_limit=per_scene_wall)
        preds.append(pred_i.detach())
        if (i + 1) % 2 == 0:
            print(f"[fit] {i+1}/{cfg['val_n']}  inner_iters={n_done}  "
                  f"last_loss={last_loss:.4g}  elapsed={time.time()-t0:.1f}s",
                  flush=True)
        if time.time() - t0 > outer_wall:
            print(f"[fit] outer wall {outer_wall}s hit at sample {i+1}", flush=True)
            break
    train_time = time.time() - t0
    pred = torch.cat(preds, 0)
    val_ph = phs[:pred.shape[0]]; val_fbp = fbps[:pred.shape[0]]

    # Restore pre-calibration ReLU clamp (CONVENTIONS.md rule 2):
    # negative outliers in the raw pred would otherwise pull the bg mean
    # negative inside evaluate_calibrated and bias the linear calibration.
    pred = pred.clamp_min(0.0)
    val_fbp = val_fbp.clamp_min(0.0)
    metrics = evaluate_calibrated(
        pred, val_ph, baseline=val_fbp,
        display_min=cfg["display_min"], display_max=cfg["display_max"])
    pred_cal = metrics["pred_cal"]
    baseline_cal = metrics["baseline_cal"]
    val_psnr, val_ssim, val_rmse = metrics["val_psnr"], metrics["val_ssim"], metrics["val_rmse"]
    baseline_psnr, baseline_rmse = metrics["baseline_psnr"], metrics["baseline_rmse"]
    headroom = metrics["headroom"]
    params_M = sum(p.numel() for p in NAF(cfg["naf_n_freqs"],
                                            cfg["naf_hidden"],
                                            cfg["naf_layers"])
                    .parameters()) / 1e6

    result = {
        "val_score": val_ssim, "val_psnr": val_psnr, "val_ssim": val_ssim,
        "val_rmse": val_rmse, "baseline_psnr": baseline_psnr,
        "baseline_ssim": metrics.get("baseline_ssim"),
        "baseline_rmse": baseline_rmse, "headroom": headroom,
        "calibration": metrics["calibration"],
        "fg_threshold": metrics["fg_threshold"],
        "params_M": params_M,
        "train_n": 0, "val_n": int(pred.shape[0]),
        "train_time_s": train_time, "config": cfg,
    }
    (out_dir / "result.json").write_text(json.dumps(result, indent=2))
    print(f"[solver] NAF: hr={headroom:.4f}  SSIM={val_ssim:.4f}  PSNR={val_psnr:.2f}  "
          f"RMSE={val_rmse:.5f}  baseline_PSNR={baseline_psnr:.2f}  "
          f"(intensity-calibrated)", flush=True)

    try:
        make_4panel_comparison(
            truth=val_ph, fbp=baseline_cal, recon=pred_cal,
            out_path=out_dir / "comparison.png",
            display_min=cfg["display_min"], display_max=cfg["display_max"],
            n_show=4, solver_label="NAF", headroom=headroom)
    except Exception as e:
        print(f"[solver] comparison.png failed: {e}", flush=True)

    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("out_dir")
    args = p.parse_args(); main(Path(args.out_dir))
