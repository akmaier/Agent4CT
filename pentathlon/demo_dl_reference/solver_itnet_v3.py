"""ItNet v3: End-to-end trained with 5-level U-Net, tied weights, TV lambda DC weight.

Key changes from v2:
  1. Five-level U-Net (deeper, 2.5M params)
  2. Tied weights: same denoiser at all k=3 iterations
  3. End-to-end training: gradients flow through full unrolled loop
  4. Alpha initialized from TV lambda (0.0037), learnable
  5. k=3 iterations (fixed)
"""
from __future__ import annotations
import argparse
import json
import math
import sys
import time
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


def _pick_groups(c: int, target: int = 8) -> int:
    for g in range(min(c, target), 0, -1):
        if c % g == 0:
            return g
    return 1


class _DoubleConv(nn.Module):
    def __init__(self, c_in, c_out):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(c_in, c_out, 3, padding=1),
            nn.GroupNorm(_pick_groups(c_out), c_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(c_out, c_out, 3, padding=1),
            nn.GroupNorm(_pick_groups(c_out), c_out),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class UNet5(nn.Module):
    """5-level U-Net for ItNet v3. Deep enough to capture non-local streaks."""

    def __init__(self, c: int = 16, residual: bool = True):
        super().__init__()
        self.residual = residual
        # Encoder
        self.enc1 = _DoubleConv(1, c)
        self.enc2 = _DoubleConv(c, c * 2)
        self.enc3 = _DoubleConv(c * 2, c * 4)
        self.enc4 = _DoubleConv(c * 4, c * 8)
        self.enc5 = _DoubleConv(c * 8, c * 16)
        # Bottleneck
        self.bot = _DoubleConv(c * 16, c * 16)
        # Decoder
        self.up5 = nn.ConvTranspose2d(c * 16, c * 16, 2, stride=2)
        self.dec5 = _DoubleConv(c * 32, c * 8)
        self.up4 = nn.ConvTranspose2d(c * 8, c * 8, 2, stride=2)
        self.dec4 = _DoubleConv(c * 16, c * 4)
        self.up3 = nn.ConvTranspose2d(c * 4, c * 4, 2, stride=2)
        self.dec3 = _DoubleConv(c * 8, c * 2)
        self.up2 = nn.ConvTranspose2d(c * 2, c * 2, 2, stride=2)
        self.dec2 = _DoubleConv(c * 4, c)
        self.up1 = nn.ConvTranspose2d(c, c, 2, stride=2)
        self.dec1 = _DoubleConv(c * 2, c)
        self.head = nn.Conv2d(c, 1, 1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, x):
        h, w = x.shape[-2:]
        ph = (32 - h % 32) % 32  # pad to multiple of 32 for 5 pools
        pw = (32 - w % 32) % 32
        x_in = F.pad(x, (0, pw, 0, ph), mode='reflect') if (ph or pw) else x

        e1 = self.enc1(x_in)
        e2 = self.enc2(F.avg_pool2d(e1, 2))
        e3 = self.enc3(F.avg_pool2d(e2, 2))
        e4 = self.enc4(F.avg_pool2d(e3, 2))
        e5 = self.enc5(F.avg_pool2d(e4, 2))
        b = self.bot(F.avg_pool2d(e5, 2))
        d5 = self.dec5(torch.cat([self.up5(b), e5], dim=1))
        d4 = self.dec4(torch.cat([self.up4(d5), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        y = self.head(d1)
        if ph or pw:
            y = y[..., :h, :w]
        return x - y if self.residual else y


class ItNetV3(nn.Module):
    """ItNet v3: end-to-end trained with tied-weight denoiser."""

    def __init__(self, geometry, denoiser, k: int = 3, alpha_init: float = 0.0037):
        super().__init__()
        self.denoiser = denoiser
        self.k = k
        # Learnable step size, initialized from TV lambda
        self.log_alpha = nn.Parameter(torch.tensor(math.log(alpha_init)))
        self.projector = PyronnFanBeamProjector(geometry)

    @property
    def alpha(self):
        return torch.nn.functional.softplus(self.log_alpha)

    def dc_step(self, x, sinogram):
        """Differentiable data consistency."""
        R_x = self.projector.forward_project(x)
        residual = R_x - sinogram
        # Approximate R^T via FBP (differentiable through PyRoNN)
        R_T_residual = self.projector.fbp(residual)
        return x - self.alpha * R_T_residual

    def forward(self, fbp_init, sinogram):
        x = fbp_init
        for _ in range(self.k):
            x_denoised = self.denoiser(x)
            x = self.dc_step(x_denoised, sinogram)
        return x


def build_dataset(geom, n, seed, i0, sigma_e, device):
    proj = PyronnFanBeamProjector(geom).to(device)
    phantoms = torch.stack([
        random_ellipses_phantom(size=geom.image_size, n_ellipses=10, seed=seed + i)[0]
        for i in range(n)
    ]).to(device)
    with torch.no_grad():
        clean = proj.forward_project(phantoms)
        noisy = simulate_low_dose(clean, i0=i0, sigma_e=sigma_e, seed=seed + 10_000)
    return phantoms, clean, noisy


CONFIG = {
    "image_size": 512, "pixel_spacing": 0.7,
    "n_angles": 128, "n_det": 736, "det_spacing": 1.2858,
    "sod": 595.0, "sdd": 1085.6,
    "train_n": 200, "val_n": 100,
    "noise_i0": 1e5, "noise_sigma_e": 10.0, "seed": 42,
    "display_min": 0.0, "display_max": 0.05,
    # U-Net
    "unet_c": 12,
    # ItNet
    "itnet_k": 3,
    "alpha_init": 0.0037,  # TV lambda
    # Training
    "epochs": 10,
    "batch_size": 20,
    "lr": 5e-4,
}


def main(out_dir: Path, cfg: dict | None = None) -> dict:
    import os
    env_path = os.environ.get("ITNET_CONFIG_PATH")
    if env_path and Path(env_path).exists():
        cfg = {**CONFIG, **json.loads(Path(env_path).read_text())}
    elif cfg is not None:
        cfg = {**CONFIG, **cfg}
    else:
        cfg = CONFIG.copy()

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(cfg["seed"])

    geom = FanBeamGeometry(
        image_size=cfg["image_size"], pixel_spacing=cfg["pixel_spacing"],
        n_angles=cfg["n_angles"], n_det=cfg["n_det"],
        det_spacing=cfg["det_spacing"], sod=cfg["sod"], sdd=cfg["sdd"],
    )

    # Datasets
    train_ph, train_clean, train_noisy = build_dataset(
        geom, cfg["train_n"], cfg["seed"], cfg["noise_i0"], cfg["noise_sigma_e"], device)
    val_ph, val_clean, val_noisy = build_dataset(
        geom, cfg["val_n"], cfg["seed"] + 1000, cfg["noise_i0"], cfg["noise_sigma_e"], device)

    # Compute FBPs
    proj = PyronnFanBeamProjector(geom).to(device)
    with torch.no_grad():
        train_fbp = torch.clamp(proj.fbp(train_noisy), min=0.0)
        val_fbp = torch.clamp(proj.fbp(val_noisy), min=0.0)

    # Model
    denoiser = UNet5(c=cfg["unet_c"], residual=True)
    itnet = ItNetV3(geom, denoiser, k=cfg["itnet_k"], alpha_init=cfg["alpha_init"])
    itnet.to(device)

    params_total = sum(p.numel() for p in itnet.parameters() if p.requires_grad)
    print(f"[solver] ItNet-v3 params: {params_total/1e6:.3f} M", flush=True)

    # End-to-end training
    opt = torch.optim.Adam(itnet.parameters(), lr=cfg["lr"])
    train_start = time.time()
    best_loss = float('inf')

    for ep in range(cfg["epochs"]):
        itnet.train()
        perm = torch.randperm(cfg["train_n"])
        running = 0.0
        n_batches = 0

        for i in range(0, cfg["train_n"], cfg["batch_size"]):
            idx = perm[i:i + cfg["batch_size"]]
            x0 = train_fbp[idx]
            sino = train_noisy[idx]
            truth = train_ph[idx]

            pred = itnet(x0, sino)
            loss = F.mse_loss(pred, truth)

            opt.zero_grad()
            loss.backward()
            opt.step()

            running += float(loss.detach().cpu())
            n_batches += 1

        avg_loss = running / max(1, n_batches)
        print(f"[train] epoch {ep+1}/{cfg['epochs']}  loss={avg_loss:.6f}  alpha={float(itnet.alpha.cpu()):.6f}", flush=True)

        if avg_loss < best_loss * 0.995:  # 0.5% improvement threshold
            best_loss = avg_loss
        # Hard time limit: stop if >4min
        if time.time() - train_start > 240:
            print(f"[train] Time limit (4min) reached at epoch {ep+1}", flush=True)
            break

    train_time = time.time() - train_start

    # Validation
    itnet.eval()
    with torch.no_grad():
        chunk = 10
        preds = []
        for i in range(0, val_noisy.shape[0], chunk):
            preds.append(itnet(val_fbp[i:i+chunk], val_noisy[i:i+chunk]))
        pred = torch.cat(preds, dim=0)

    # Metrics vs truth phantom
    data_range = cfg["display_max"] - cfg["display_min"]
    val_psnr = float(psnr(pred, val_ph, data_range=data_range).cpu())
    val_ssim = float(ssim(pred, val_ph, data_range=data_range).cpu())
    val_rmse = float(((pred - val_ph) ** 2).mean().sqrt().cpu())
    baseline_psnr = float(psnr(val_fbp, val_ph, data_range=data_range).cpu())
    baseline_rmse = float(((val_fbp - val_ph) ** 2).mean().sqrt().cpu())
    headroom = max(0.0, 1.0 - val_rmse / max(baseline_rmse, 1e-12))
    val_score = val_ssim

    print(f"[solver] alpha (learned) = {float(itnet.alpha.cpu()):.6f}", flush=True)
    print(f"[solver] ItNet-v3: val_score={val_score:.4f} headroom={headroom:.4f} PSNR={val_psnr:.2f} SSIM={val_ssim:.4f}", flush=True)

    # Comparison figure
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        n_show = min(3, cfg["val_n"])
        fig, ax = plt.subplots(n_show, 4, figsize=(12, 3 * n_show))
        if n_show == 1:
            ax = ax[None]
        vmin, vmax = cfg["display_min"], cfg["display_max"]
        for i in range(n_show):
            ax[i, 0].imshow(val_ph[i, 0].cpu(), cmap="gray", vmin=vmin, vmax=vmax)
            ax[i, 0].set_title("truth" if i == 0 else "")
            ax[i, 1].imshow(val_fbp[i, 0].cpu(), cmap="gray", vmin=vmin, vmax=vmax)
            ax[i, 1].set_title(f"FBP  (PSNR={baseline_psnr:.1f})" if i == 0 else "")
            ax[i, 2].imshow(pred[i, 0].cpu(), cmap="gray", vmin=vmin, vmax=vmax)
            ax[i, 2].set_title(f"ItNet-v3  (PSNR={val_psnr:.1f} SSIM={val_ssim:.3f})" if i == 0 else "")
            ax[i, 3].imshow((pred[i, 0] - val_ph[i, 0]).cpu(), cmap="RdBu_r", vmin=-0.01, vmax=0.01)
            ax[i, 3].set_title("residual" if i == 0 else "")
            for a in ax[i]:
                a.set_axis_off()
        plt.tight_layout()
        figpath = out_dir / "comparison.png"
        plt.savefig(figpath, dpi=120)
        print(f"[solver] saved {figpath}", flush=True)
    except Exception as e:
        print(f"[solver] figure failed: {e}", flush=True)

    result = {
        "val_score": val_score, "val_psnr": val_psnr, "val_ssim": val_ssim,
        "val_rmse": val_rmse, "baseline_psnr": baseline_psnr,
        "baseline_rmse": baseline_rmse, "headroom": headroom,
        "params_M": params_total / 1e6, "train_n": cfg["train_n"],
        "val_n": cfg["val_n"], "train_time_s": train_time,
        "alpha_learned": float(itnet.alpha.cpu()),
        "config": cfg,
    }
    (out_dir / "result.json").write_text(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("out_dir")
    args = p.parse_args()
    main(Path(args.out_dir))
