"""Reference: U-Swin — U-Net + Swin-transformer hybrid for sparse-view CT.

Adapted from Xu et al. 2024, "Hybrid U-Net and Swin-transformer network for
limited-angle cardiac CT" (Phys. Med. Biol. 69:105012; DOI 10.1088/1361-6560/ad3db9).

Hybrid pipeline: a 4-level U-Net (local CNN prior for structure) with a
Swin-transformer block inserted at every encoder level (windowed
self-attention for non-local streak/aliasing artifacts). Trained MSE on
the truth phantom from the noisy FBP starting iterate.

See literature/transformer_ct_comparison.md for the side-by-side with
TransCT and the rationale for picking U-Swin for sparse-view.
"""
from __future__ import annotations
import argparse
import json
import math
import os
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
from ddssl_ldct.metrics import psnr, ssim, evaluate_calibrated, make_4panel_comparison, supervised_recon_loss, negativity_penalty
from challenges.demo_dl.geometry import DEFAULTS as DEMO_DL_DEFAULTS


CONFIG = {
    **DEMO_DL_DEFAULTS,
    "train_n": 200,
    # U-Swin architecture
    "uswin_c": 24,             # base channel count
    "swin_window": 8,          # swin attention window size (pixels)
    "swin_heads": 4,
    # Training
    "epochs": 10,
    "batch_size": 4,
    "lr": 5e-4,
}


def _g(c, target=8):
    for g in range(min(c, target), 0, -1):
        if c % g == 0:
            return g
    return 1


class WindowMSA(nn.Module):
    """Simple non-shifted windowed multi-head self-attention.
    Operates on (B, C, H, W) — partitions into WxW windows, runs
    per-window MHA on flattened tokens, then reassembles.
    """
    def __init__(self, c, window, heads):
        super().__init__()
        self.c = c
        self.window = window
        self.heads = heads
        self.norm = nn.LayerNorm(c)
        self.qkv = nn.Linear(c, 3 * c)
        self.proj = nn.Linear(c, c)
        self.mlp = nn.Sequential(
            nn.LayerNorm(c), nn.Linear(c, 2 * c),
            nn.GELU(), nn.Linear(2 * c, c),
        )

    def forward(self, x):
        # x: (B, C, H, W). Right/bottom-pad so H, W are multiples of window.
        B, C, H, W = x.shape
        ws = self.window
        ph = (ws - H % ws) % ws
        pw = (ws - W % ws) % ws
        if ph or pw:
            x = F.pad(x, (0, pw, 0, ph))
        Hn, Wn = H + ph, W + pw
        # → (B, H/ws, ws, W/ws, ws, C) → windows
        xp = x.permute(0, 2, 3, 1).contiguous()                      # (B,H,W,C)
        xw = xp.view(B, Hn // ws, ws, Wn // ws, ws, C)
        xw = xw.permute(0, 1, 3, 2, 4, 5).contiguous()
        xw = xw.view(-1, ws * ws, C)                                 # (Bn, ws², C)
        # Attention block
        z = self.norm(xw)
        qkv = self.qkv(z).reshape(z.size(0), z.size(1), 3, self.heads,
                                  C // self.heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) / math.sqrt(C // self.heads)
        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(z.size(0), z.size(1), C)
        xw = xw + self.proj(out)
        xw = xw + self.mlp(xw)
        # Un-window → (B, C, Hn, Wn)
        xw = xw.view(B, Hn // ws, Wn // ws, ws, ws, C)
        xw = xw.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, Hn, Wn, C)
        x = xw.permute(0, 3, 1, 2).contiguous()
        if ph or pw:
            x = x[..., :H, :W]
        return x


class DoubleConv(nn.Module):
    def __init__(self, ci, co):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(ci, co, 3, padding=1),
            nn.GroupNorm(_g(co), co), nn.ReLU(inplace=True),
            nn.Conv2d(co, co, 3, padding=1),
            nn.GroupNorm(_g(co), co), nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class USwin(nn.Module):
    """4-level U-Net with a Swin-MSA block at each encoder level."""
    def __init__(self, c=24, window=8, heads=4):
        super().__init__()
        self.enc1 = DoubleConv(1, c)
        self.sw1 = WindowMSA(c, window, heads)
        self.enc2 = DoubleConv(c, c * 2)
        self.sw2 = WindowMSA(c * 2, window, heads)
        self.enc3 = DoubleConv(c * 2, c * 4)
        self.sw3 = WindowMSA(c * 4, window, heads)
        self.enc4 = DoubleConv(c * 4, c * 8)
        self.sw4 = WindowMSA(c * 8, window, heads)
        self.bot = DoubleConv(c * 8, c * 16)
        self.up4 = nn.ConvTranspose2d(c * 16, c * 8, 2, stride=2)
        self.dec4 = DoubleConv(c * 16, c * 4)
        self.up3 = nn.ConvTranspose2d(c * 4, c * 4, 2, stride=2)
        self.dec3 = DoubleConv(c * 8, c * 2)
        self.up2 = nn.ConvTranspose2d(c * 2, c * 2, 2, stride=2)
        self.dec2 = DoubleConv(c * 4, c)
        self.up1 = nn.ConvTranspose2d(c, c, 2, stride=2)
        self.dec1 = DoubleConv(c * 2, c)
        self.head = nn.Conv2d(c, 1, 1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, x):
        h, w = x.shape[-2:]
        ph = (16 - h % 16) % 16
        pw = (16 - w % 16) % 16
        xin = F.pad(x, (0, pw, 0, ph), mode="reflect") if (ph or pw) else x
        e1 = self.sw1(self.enc1(xin))
        e2 = self.sw2(self.enc2(F.avg_pool2d(e1, 2)))
        e3 = self.sw3(self.enc3(F.avg_pool2d(e2, 2)))
        e4 = self.sw4(self.enc4(F.avg_pool2d(e3, 2)))
        b = self.bot(F.avg_pool2d(e4, 2))
        d4 = self.dec4(torch.cat([self.up4(b), e4], 1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], 1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], 1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], 1))
        y = self.head(d1)
        if ph or pw:
            y = y[..., :h, :w]
        # Residual: predict the correction to the FBP input.
        return x - y


def build_dataset(geom, n, seed, i0, sigma_e, device):
    # Dispatches on AGENT4CT_DATASET / cfg["dataset_kind"]. Phantom path
    # is backwards-compatible; staged paths load from disk. Split is picked
    # from the existing seed convention (train: seed=cfg["seed"]; val:
    # seed=cfg["seed"]+1000).
    from ddssl_ldct.staged_dataset import load_val_split
    import os
    kind = os.environ.get("AGENT4CT_DATASET", "phantoms")
    split = "val" if (seed % 100_000) >= 1000 else "train"
    return load_val_split(kind, split, n, device=device,
                          seed=seed, noise_i0=i0, noise_sigma_e=sigma_e,
                          geom=geom)


def main(out_dir: Path, cfg: dict | None = None) -> dict:
    env_path = os.environ.get("USWIN_CONFIG_PATH")
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
    print(f"[solver] device={device}  config={json.dumps({k:v for k,v in cfg.items() if k.startswith('uswin') or k in ('epochs','batch_size','lr','swin_window','swin_heads')}, default=str)}",
          flush=True)
    torch.manual_seed(cfg["seed"])

    geom = FanBeamGeometry(
        image_size=cfg["image_size"], pixel_spacing=cfg["pixel_spacing"],
        n_angles=cfg["n_angles"], n_det=cfg["n_det"],
        det_spacing=cfg["det_spacing"], sod=cfg["sod"], sdd=cfg["sdd"],
    )

    train_ph, _, train_noisy = build_dataset(geom, cfg["train_n"], cfg["seed"],
        cfg["noise_i0"], cfg["noise_sigma_e"], device)
    val_ph, _, val_noisy = build_dataset(geom, cfg["val_n"], cfg["seed"] + 1000,
        cfg["noise_i0"], cfg["noise_sigma_e"], device)
    proj = PyronnFanBeamProjector(geom).to(device)
    with torch.no_grad():
        train_fbp = torch.clamp(proj.fbp(train_noisy), min=0.0)
        val_fbp = torch.clamp(proj.fbp(val_noisy), min=0.0)

    model = USwin(c=cfg["uswin_c"], window=cfg["swin_window"],
                  heads=cfg["swin_heads"]).to(device)
    params_total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[solver] U-Swin params: {params_total/1e6:.3f} M", flush=True)

    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    t0 = time.time()
    for ep in range(cfg["epochs"]):
        model.train()
        perm = torch.randperm(cfg["train_n"])
        running = 0.0; nb = 0
        for i in range(0, cfg["train_n"], cfg["batch_size"]):
            idx = perm[i:i + cfg["batch_size"]]
            pred = model(train_fbp[idx])
            loss = supervised_recon_loss(pred, train_ph[idx], lambda_neg=1.0)
            opt.zero_grad(); loss.backward(); opt.step()
            running += float(loss.detach().cpu()); nb += 1
        print(f"[train] epoch {ep+1}/{cfg['epochs']}  loss={running/max(1,nb):.6g}",
              flush=True)
        if time.time() - t0 > 480:
            print(f"[train] 8-min wall at epoch {ep+1}", flush=True); break
    train_time = time.time() - t0

    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, val_fbp.shape[0], cfg["batch_size"]):
            preds.append(model(val_fbp[i:i + cfg["batch_size"]]))
    pred = torch.cat(preds, 0)

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

    result = {
        "val_score": val_ssim, "val_psnr": val_psnr, "val_ssim": val_ssim,
        "val_rmse": val_rmse, "baseline_psnr": baseline_psnr,
        "baseline_ssim": metrics.get("baseline_ssim"),
        "baseline_rmse": baseline_rmse, "headroom": headroom,
        "calibration": metrics["calibration"],
        "fg_threshold": metrics["fg_threshold"],
        "params_M": params_total / 1e6, "train_n": cfg["train_n"],
        "val_n": cfg["val_n"], "train_time_s": train_time, "config": cfg,
    }
    (out_dir / "result.json").write_text(json.dumps(result, indent=2))
    print(f"[solver] U-Swin: hr={headroom:.4f}  SSIM={val_ssim:.4f}  PSNR={val_psnr:.2f}  "
          f"RMSE={val_rmse:.5f}  baseline_PSNR={baseline_psnr:.2f}  "
          f"(intensity-calibrated)", flush=True)

    try:
        make_4panel_comparison(
            truth=val_ph, fbp=baseline_cal, recon=pred_cal,
            out_path=out_dir / "comparison.png",
            display_min=cfg["display_min"], display_max=cfg["display_max"],
            n_show=4, solver_label="U-Swin", headroom=headroom)
    except Exception as e:
        print(f"[solver] comparison.png failed: {e}", flush=True)

    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("out_dir")
    args = p.parse_args(); main(Path(args.out_dir))
