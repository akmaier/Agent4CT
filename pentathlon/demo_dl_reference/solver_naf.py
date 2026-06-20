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
    # --- Coordinate encoder selector (audit fix A, 2026-06-20) -----------
    # "fourier" (DEFAULT) = the original Fourier PosEnc -> vanilla MLP path.
    #   Reproduces the OLD behavior bit-for-bit so concurrent jobs / old cfgs
    #   are unaffected.
    # "hash"    = the paper's Instant-NGP multi-resolution HASH grid encoder
    #   feeding a SMALL MLP, per the NAF reference chest_50.yaml. The hash
    #   levels are config-gated by the naf_hash_* keys below.
    "naf_encoding": "fourier",
    # Instant-NGP hash-grid hyper-params (only used when naf_encoding="hash").
    "naf_hash_levels": 16,        # n resolution levels
    "naf_hash_feat_dim": 2,       # features per level (level_dim)
    "naf_hash_log2_size": 19,     # log2 hashmap size per level
    "naf_hash_base_res": 16,      # coarsest grid resolution
    "naf_hash_max_res": 512,      # finest grid resolution (geometric growth)
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


class HashEncoding(nn.Module):
    """Pure-torch 2-D multi-resolution hash encoding (Instant-NGP / NAF).

    The headline NAF contribution: instead of a Fourier feature map into a big
    MLP, hash a coordinate into ``n_levels`` learnable feature tables at
    geometrically-growing grid resolutions, bilinearly interpolate the per-level
    features, and concat. A SMALL MLP then maps the (n_levels * feat_dim)-vector
    to attenuation. Reference: Müller et al. 2022 (Instant-NGP); NAF
    chest_50.yaml uses n_levels=16, level_dim=2, log2_hashmap_size=19,
    base_resolution=16. No tiny-cuda-nn dependency — all torch.

    Trainable params: n_levels tables, each (2**log2_size, feat_dim). Hashing is
    the standard spatial hash  h = (x*pi_1) XOR (y*pi_2)  mod hashmap_size, with
    primes (1, 2654435761). Input coords are 2-D in [0, 1]^2.
    """

    PRIMES = (1, 2654435761)

    def __init__(self, n_levels=16, feat_dim=2, log2_hashmap_size=19,
                 base_res=16, max_res=512):
        super().__init__()
        self.n_levels = int(n_levels)
        self.feat_dim = int(feat_dim)
        self.hashmap_size = 1 << int(log2_hashmap_size)
        # Per-level resolution: geometric growth base_res -> max_res.
        if self.n_levels > 1:
            b = math.exp((math.log(max_res) - math.log(base_res)) /
                         (self.n_levels - 1))
        else:
            b = 1.0
        res = [int(round(base_res * (b ** L))) for L in range(self.n_levels)]
        self.register_buffer("resolutions",
                             torch.tensor(res, dtype=torch.long), persistent=False)
        # One learnable feature table per level. Standard Instant-NGP init:
        # small uniform so the encoded field starts ~flat.
        self.tables = nn.ParameterList([
            nn.Parameter(torch.empty(self.hashmap_size, self.feat_dim)
                         .uniform_(-1e-4, 1e-4))
            for _ in range(self.n_levels)
        ])
        self.out_dim = self.n_levels * self.feat_dim

    def _hash(self, ix, iy):
        # ix, iy: integer grid coords (long). Standard spatial hash.
        h = (ix * self.PRIMES[0]) ^ (iy * self.PRIMES[1])
        return (h % self.hashmap_size).long()

    def forward(self, x):                       # x: (N, 2) in [0, 1]
        # Bilinear-interpolate hashed features at each resolution level.
        feats = []
        for L in range(self.n_levels):
            res = int(self.resolutions[L].item())
            # Scale to [0, res-1]; corner indices + interpolation weights.
            xs = x.clamp(0.0, 1.0) * (res - 1)
            x0 = torch.floor(xs).long()
            x1 = x0 + 1
            w = xs - x0.float()                 # (N, 2) in [0,1]
            x0 = x0.clamp(0, res - 1); x1 = x1.clamp(0, res - 1)
            ix0, iy0 = x0[:, 0], x0[:, 1]
            ix1, iy1 = x1[:, 0], x1[:, 1]
            wx, wy = w[:, 0:1], w[:, 1:2]
            t = self.tables[L]
            c00 = t[self._hash(ix0, iy0)]
            c10 = t[self._hash(ix1, iy0)]
            c01 = t[self._hash(ix0, iy1)]
            c11 = t[self._hash(ix1, iy1)]
            c0 = c00 * (1 - wx) + c10 * wx
            c1 = c01 * (1 - wx) + c11 * wx
            feats.append(c0 * (1 - wy) + c1 * wy)
        return torch.cat(feats, dim=-1)         # (N, n_levels*feat_dim)


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
                 init_mu=0.005, encoding="fourier", hash_cfg=None):
        super().__init__()
        self.out_scale = out_scale
        self.encoding = encoding
        if encoding == "hash":
            # Instant-NGP multi-res hash grid -> SMALL MLP (paper's headline
            # encoder). Coords arrive in [-1,1]; HashEncoding maps to [0,1].
            hc = hash_cfg or {}
            self.pe = HashEncoding(
                n_levels=hc.get("n_levels", 16),
                feat_dim=hc.get("feat_dim", 2),
                log2_hashmap_size=hc.get("log2_hashmap_size", 19),
                base_res=hc.get("base_res", 16),
                max_res=hc.get("max_res", 512))
            in_dim = self.pe.out_dim
        else:
            self.pe = PosEnc(n_freqs)
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

    def forward(self, coords):                  # coords: (N, 2) in [-1, 1]
        # Hash encoder expects coords in [0,1]; Fourier expects [-1,1].
        enc = self.pe((coords + 1.0) * 0.5 if self.encoding == "hash"
                      else coords)
        h = self.mlp(enc).squeeze(-1)
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


def _build_naf(cfg):
    """Construct a NAF model honoring the cfg encoder selector.

    DEFAULT (naf_encoding absent / "fourier") reproduces the OLD model exactly.
    When "hash", swap in the Instant-NGP hash-grid encoder + small MLP.
    """
    encoding = cfg.get("naf_encoding", "fourier")
    hash_cfg = {
        "n_levels": cfg.get("naf_hash_levels", 16),
        "feat_dim": cfg.get("naf_hash_feat_dim", 2),
        "log2_hashmap_size": cfg.get("naf_hash_log2_size", 19),
        "base_res": cfg.get("naf_hash_base_res", 16),
        "max_res": cfg.get("naf_hash_max_res", 512),
    }
    return NAF(cfg["naf_n_freqs"], cfg["naf_hidden"], cfg["naf_layers"],
               out_scale=cfg["naf_n_clip"], init_mu=0.005,
               encoding=encoding, hash_cfg=hash_cfg)


def fit_one_scene(noisy_sino, geom, cfg, device, t_limit=None):
    """One NAF optimisation against one sinogram. Returns (H, W) μ image.

    `t_limit` (seconds) is the per-scene wall clamp — set externally so the
    outer loop can give each scene a fair budget regardless of n_iter.
    """
    H = W = cfg["image_size"]
    proj = PyronnFanBeamProjector(geom).to(device)
    model = _build_naf(cfg).to(device)
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

    # Mayo: val split is a single patient (L277) -> reconstruct at its native
    # pixel-spacing (default ps mis-scales L277 ~5% and reads as broken).
    if cfg.get("dataset_kind") == "mayo_ldct_2d":
        from ddssl_ldct.staged_dataset import load_val_split as _lvs
        _g0 = FanBeamGeometry(
            image_size=cfg["image_size"], pixel_spacing=cfg["pixel_spacing"],
            n_angles=cfg["n_angles"], n_det=cfg["n_det"],
            det_spacing=cfg["det_spacing"], sod=cfg["sod"], sdd=cfg["sdd"])
        try:
            _vps = _lvs("mayo_ldct_2d", "val", cfg["val_n"], device=device,
                        seed=cfg["seed"] + 1000, noise_i0=cfg["noise_i0"],
                        noise_sigma_e=cfg["noise_sigma_e"], geom=_g0,
                        return_ps=True)[-1]
            if _vps is not None:
                import numpy as _np
                cfg["pixel_spacing"] = round(float(_np.median(_np.asarray(_vps, float))), 5)
                print(f"[solver] Mayo val ps -> pixel_spacing={cfg['pixel_spacing']}", flush=True)
        except Exception as _e:
            print(f"[solver] val-ps probe failed ({_e}); using default ps", flush=True)

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
    params_M = sum(p.numel() for p in _build_naf(cfg).parameters()) / 1e6

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
