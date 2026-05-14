"""DL-Sparse-View CT solver — RESIDUAL-STACK denoiser variant.

This is the spawn agent's solver, exploring an alternative denoiser
architecture family: a stack of plain residual blocks (no down/up-sampling,
no skip-encoder-decoder), inspired by DnCNN (Zhang 2017) and the residual
networks in the Sidky 2022 DL-Sparse-View report (§ 5 mentions multiple
top teams using residual / variational networks rather than U-Nets).

Inductive bias contrast vs. the main agent's `pentathlon/dl_sparse_view`
solver:
    U-Net  : multi-scale, large receptive field via pooling, lots of skip
             routing. Good at large streak artefacts.
    ResNet : no down/up-sampling, smaller receptive field but very dense
             local processing at full resolution. Cheap per-pixel
             refinement, hopefully sharper edges and less smoothing.

The denoiser keeps the SmallUNet contract from `ddssl_ldct.models` (single
1-channel in / out, predicts the residual if `residual=True`) and plugs
into the same `DualDomainPipeline`.
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
from ddssl_ldct.simulate import simulate_low_dose, split_projections
from ddssl_ldct.training import DualDomainPipeline
from ddssl_ldct.models import TrainableBilateralFilter2d
from ddssl_ldct.metrics import psnr, ssim


# ----------------------------------------------------------------------- #
#  CONFIG  —  the spawn agent edits this block + the model below.
# ----------------------------------------------------------------------- #

CONFIG = {
    # Geometry (FIXED — Wagner / Siemens AS @ 128 sparse views).
    "image_size":    512,
    "pixel_spacing": 0.7,
    "n_angles":      128,
    "n_det":         736,
    "det_spacing":   1.2858,
    "sod":           595.0,
    "sdd":           1085.6,

    # Training subset for the 5-minute iteration budget.
    "train_n":       400,         # iter-76 closed 450 -4.67; data axis final-closed
    "val_n":         100,

    # Editable: training schedule.
    "epochs":        8,          # iter-75 closed epochs=9, 10 DISCARD; 8 optimum
    "batch_size":    1,          # iter-47 closed batch=2 at -2.75pp (cross-substrate)
    "input_dropout": 0.0,        # iter-68 closed
    "lr_warmup_epochs": 1,        # iter-80 KEEP
    "res_n_bf":      0,            # iter-63 closed BF on batch substrate too
    "bf_kernel":     7,
    "bf_sigma_x":    1.5,
    "bf_sigma_y":    1.5,
    "bf_sigma_r":    0.01,
    "lr":            1e-4,       # revert
    "adamw_eps":     1e-10,      # iter-53 KEEP (iter-55/56 both DISCARD around it; basin shallow)
    "optimizer":     "adamw",   # iter-54 closed adam at -1.01pp; AdamW + wd_split is right for this slug
    "weight_decay":  5e-5,       # iter-82: 1e-4 -> 5e-5 retest on batch+warmup substrate
    "lr_schedule":   "constant", # iter-42: revert schedule cruft; LR axis closed

    # Editable: residual-stack model architecture.
    "res_blocks":    6,          # iter-65 closed 7 on batch -8.89pp; capacity firmly at 6
    "res_bias":      True,       # iter-61 closed bias=False at -1.89pp
    "res_channels":  32,         # iter-2: 48 -> 32 (saves ~2.2x flops)
    "res_norm":      "batch",    # iter-62: group -> batch (untested; group iter-2 KEEP, none iter-9 DISCARD)
    "res_act":       "relu",     # iter-72 closed gelu+combined
    "res_kernel":    3,
    "res_dropout":   0.0,         # iter-73 closed res_dropout on batch -3.17pp
    "residual":      True,       # global residual (predict noise)
    "res_scale":     0.1,        # iter-64 closed alpha=0.2 on batch -0.79pp
    "swa_last_n":    0,           # SWA family closed
    "adamw_beta2":   0.999,      # iter-46 closed beta2=0.99 axis (-1.28pp)

    # iter-42 (DISCARD, -1.82pp): BF tail does NOT cross-port from NAFNet
    # to resnet substrate. Same finding as Agent C iter-26. Disable.
    "res_n_bf":      0,
    "bf_kernel":     7,
    "bf_sigma_x":    1.5,
    "bf_sigma_y":    1.5,
    "bf_sigma_r":    0.01,


    # Noise simulation — FIXED so headroom comparable across iter / runs.
    "noise_i0":      5e4,
    "noise_sigma_e": 5.0,
    "seed":          42,
}


# ----------------------------------------------------------------------- #
#  Residual-stack denoiser — agent edits HERE for architecture changes.
# ----------------------------------------------------------------------- #

def _pick_groups(c: int, target: int = 8) -> int:
    for g in range(min(c, target), 0, -1):
        if c % g == 0:
            return g
    return 1


def _make_norm(name: str, c: int) -> nn.Module:
    if name == "group":
        return nn.GroupNorm(_pick_groups(c), c)
    if name == "batch":
        return nn.BatchNorm2d(c)
    return nn.Identity()


def _make_act(name: str) -> nn.Module:
    if name == "gelu":
        return nn.GELU()
    if name == "swish":
        return nn.SiLU(inplace=False)
    return nn.ReLU(inplace=True)


class ResBlock(nn.Module):
    """Standard residual block: conv -> norm -> act -> conv -> norm -> + x.

    The activation after the addition is omitted (pre-activation style with
    addition at the end works well in practice for low-level vision; cf.
    DnCNN / EDSR conventions). A dropout can sit between the two convs.

    iter-14: optional learnable residual-scaling scalar alpha (per block),
    so the block becomes x + alpha * h. Init alpha small (0.1) so the stack
    starts as a near-identity at random init (cf. EDSR Lim et al. 2017
    residual-scaling and ReZero Bachlechner et al. 2020 alpha=0 init).
    Combined with the zero-init tail conv this gives a very gentle info
    path early in training — the network must actively learn to use the
    residual blocks rather than rely on them by default.
    """
    def __init__(self, c: int, kernel: int = 3, norm: str = "group",
                 act: str = "relu", dropout: float = 0.0,
                 res_scale: float | None = None,
                 bias: bool = True):
        super().__init__()
        pad = kernel // 2
        self.conv1 = nn.Conv2d(c, c, kernel, padding=pad, bias=bias)
        self.n1 = _make_norm(norm, c)
        self.act1 = _make_act(act)
        self.drop = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.conv2 = nn.Conv2d(c, c, kernel, padding=pad, bias=bias)
        self.n2 = _make_norm(norm, c)
        if res_scale is None:
            self.alpha = None
        else:
            self.alpha = nn.Parameter(torch.tensor(float(res_scale)))

    def forward(self, x):
        h = self.conv1(x)
        h = self.n1(h)
        h = self.act1(h)
        h = self.drop(h)
        h = self.conv2(h)
        h = self.n2(h)
        if self.alpha is None:
            return x + h
        return x + self.alpha * h


class ResidualStack(nn.Module):
    """Stack of residual blocks at a single spatial resolution.

    Architecture:
        head conv (1 -> c)
        N x ResBlock(c, c)
        tail conv (c -> 1, zero-init so the network starts as identity)
        + (optional) global residual: y = x - tail(features)

    The zero-init of the tail matches `SmallUNet`'s behaviour: the network
    starts as the identity (or as `x` when residual=True), so optimisation
    starts at a sensible point even at random init.
    """
    def __init__(self, n_blocks: int = 8, c: int = 48,
                 kernel: int = 3, norm: str = "group", act: str = "relu",
                 dropout: float = 0.0, residual: bool = True,
                 res_scale: float | None = None,
                 input_dropout: float = 0.0,
                 bias: bool = True):
        super().__init__()
        self.residual = residual
        pad = kernel // 2
        self.input_dropout = nn.Dropout2d(input_dropout) if input_dropout > 0 else nn.Identity()
        self.head = nn.Conv2d(1, c, kernel, padding=pad, bias=bias)
        self.head_act = _make_act(act)
        self.blocks = nn.Sequential(*[
            ResBlock(c, kernel=kernel, norm=norm, act=act, dropout=dropout,
                     res_scale=res_scale, bias=bias)
            for _ in range(n_blocks)
        ])
        self.tail = nn.Conv2d(c, 1, kernel, padding=pad)
        nn.init.zeros_(self.tail.weight)
        nn.init.zeros_(self.tail.bias)

    def forward(self, x):
        x_inp = self.input_dropout(x)
        h = self.head(x_inp)
        h = self.head_act(h)
        h = self.blocks(h)
        y = self.tail(h)
        return x - y if self.residual else y


class ResidualStackPlusBFs(nn.Module):
    """ResidualStack followed by N stacked TrainableBilateralFilter2d tails.
    Cross-port from main/A NAFNet+BF recipe (+0.04..+0.28pp per BF, compounding)."""
    def __init__(self, stack: nn.Module, n_bf: int, kernel: int,
                 sigma_x: float, sigma_y: float, sigma_r: float):
        super().__init__()
        self.stack = stack
        self.bfs = nn.ModuleList([
            TrainableBilateralFilter2d(
                kernel_size=kernel, sigma_x=sigma_x,
                sigma_y=sigma_y, sigma_r=sigma_r)
            for _ in range(n_bf)
        ])
    def forward(self, x):
        h = self.stack(x)
        for bf in self.bfs:
            h = bf(h)
        return h


def build_denoisers(cfg: dict) -> tuple[nn.Module, nn.Module]:
    """Return (proj_denoiser, image_denoiser).

    Symmetric residual stacks — the main agent's runs showed asymmetric
    capacity overfits, so we start symmetric here too.
    """
    def make():
        return ResidualStack(
            n_blocks=cfg["res_blocks"],
            c=cfg["res_channels"],
            kernel=cfg["res_kernel"],
            norm=cfg["res_norm"],
            act=cfg["res_act"],
            dropout=cfg["res_dropout"],
            residual=cfg["residual"],
            res_scale=cfg.get("res_scale", None),
            input_dropout=float(cfg.get("input_dropout", 0.0)),
            bias=bool(cfg.get("res_bias", True)),
        )
    proj_dn = make()
    img_dn = make()
    n_bf = int(cfg.get("res_n_bf", 0))
    if n_bf > 0:
        img_dn = ResidualStackPlusBFs(
            img_dn, n_bf=n_bf,
            kernel=int(cfg.get("bf_kernel", 7)),
            sigma_x=float(cfg.get("bf_sigma_x", 1.5)),
            sigma_y=float(cfg.get("bf_sigma_y", 1.5)),
            sigma_r=float(cfg.get("bf_sigma_r", 0.01)),
        )
    return proj_dn, img_dn


# ----------------------------------------------------------------------- #
#  Data + training harness — leave alone (changes here count as solver
#  changes; prefer model / schedule edits).
# ----------------------------------------------------------------------- #

def build_geometry(cfg: dict) -> FanBeamGeometry:
    return FanBeamGeometry(
        image_size=cfg["image_size"], pixel_spacing=cfg["pixel_spacing"],
        n_angles=cfg["n_angles"], n_det=cfg["n_det"],
        det_spacing=cfg["det_spacing"],
        sod=cfg["sod"], sdd=cfg["sdd"],
    )


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


def make_optimizer(model: nn.Module, cfg):
    beta2 = float(cfg.get("adamw_beta2", 0.999))
    betas = (0.9, beta2)
    eps = float(cfg.get("adamw_eps", 1e-8))
    if cfg["optimizer"] == "adam":
        return torch.optim.Adam(model.parameters(), lr=cfg["lr"], betas=betas, eps=eps)
    # iter-34: split params into wd-regulated and wd-excluded groups.
    # alpha (EDSR residual-scaling scalar, init 0.1) lives at module.alpha
    # in ResBlock. AdamW WD pulls alpha toward 0 throughout training, which
    # iter-21 (global wd=0) showed could free alpha but came at cost of
    # under-regularised conv weights (-0.51pp). Targeted: keep WD on conv
    # weights, exclude alpha (and biases, and norm scales — standard practice
    # for the 1-D / per-channel params that don't benefit from L2 shrinkage).
    no_wd_params, wd_params = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if name.endswith(".alpha") or name.endswith("bias") or "n1" in name or "n2" in name:
            no_wd_params.append(p)
        else:
            wd_params.append(p)
    print(f"[solver-res] AdamW param-groups: wd={len(wd_params)} (wd={cfg['weight_decay']})  no_wd={len(no_wd_params)} (wd=0)",
          flush=True)
    return torch.optim.AdamW(
        [
            {"params": wd_params, "weight_decay": cfg["weight_decay"]},
            {"params": no_wd_params, "weight_decay": 0.0},
        ],
        lr=cfg["lr"],
        betas=betas,
        eps=eps,
    )


def main(out_dir: Path, cfg: dict | None = None) -> dict:
    cfg = {**CONFIG, **(cfg or {})}
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[solver-res] device={device}  config={json.dumps(cfg)}", flush=True)
    torch.manual_seed(cfg["seed"])

    geom = build_geometry(cfg)
    train_ph, train_clean, train_noisy = build_dataset(
        geom, cfg["train_n"], cfg["seed"], cfg["noise_i0"], cfg["noise_sigma_e"], device)
    val_ph, val_clean, val_noisy = build_dataset(
        geom, cfg["val_n"], cfg["seed"] + 1000, cfg["noise_i0"], cfg["noise_sigma_e"], device)

    with torch.no_grad():
        R_full = PyronnFanBeamProjector(geom).to(device)
        val_ref = R_full.fbp(val_clean)
        ld_fbp = R_full.fbp(val_noisy)

    proj_dn, img_dn = build_denoisers(cfg)
    pipe = DualDomainPipeline(
        geometry=geom, proj_denoiser=proj_dn, image_denoiser=img_dn,
    ).to(device)
    params_total = sum(p.numel() for p in pipe.parameters() if p.requires_grad)
    print(f"[solver-res] params = {params_total/1e6:.3f} M", flush=True)

    opt = make_optimizer(pipe, cfg)
    lr_schedule = cfg.get("lr_schedule", "constant")
    lr_max = float(cfg["lr"])
    lr_min = float(cfg.get("lr_min", 0.0))
    lr_step_factor = float(cfg.get("lr_step_factor", 0.3333))
    n_epochs = int(cfg["epochs"])
    if lr_schedule == "cosine":
        print(f"[solver-res] LR schedule: cosine {lr_max:.2e} -> {lr_min:.2e} over {n_epochs} epochs",
              flush=True)
    elif lr_schedule == "step7":
        print(f"[solver-res] LR schedule: step7 (lr={lr_max:.2e} for ep 1-7, then x{lr_step_factor} for ep 8)",
              flush=True)
    # iter-45: per-step SWA cross-port from main iter-60 (+0.32pp on NAFNet).
    # Average model weights over EVERY optimizer step in the last swa_last_n epochs.
    swa_last_n = int(cfg.get("swa_last_n", 0))
    swa_start_ep = n_epochs - swa_last_n
    swa_state = None
    swa_count = 0
    t0 = time.time()
    for ep in range(n_epochs):
        if lr_schedule == "cosine" and n_epochs > 1:
            progress = ep / (n_epochs - 1)
            cur_lr = lr_min + 0.5 * (lr_max - lr_min) * (1.0 + math.cos(math.pi * progress))
            for pg in opt.param_groups:
                pg["lr"] = cur_lr
        elif lr_schedule == "step7":
            cur_lr = lr_max * lr_step_factor if ep == n_epochs - 1 else lr_max
            for pg in opt.param_groups:
                pg["lr"] = cur_lr
        else:
            cur_lr = lr_max
        # iter-49: optional linear warmup over first lr_warmup_epochs epochs.
        warmup_epochs = int(cfg.get("lr_warmup_epochs", 0))
        if warmup_epochs > 0 and ep < warmup_epochs:
            cur_lr = lr_max * (ep + 1) / max(1, warmup_epochs)
            for pg in opt.param_groups:
                pg["lr"] = cur_lr
        pipe.train()
        perm = torch.randperm(train_noisy.shape[0])
        running = 0.0
        for i in range(0, train_noisy.shape[0], cfg["batch_size"]):
            idx = perm[i:i + cfg["batch_size"]]
            batch = train_noisy[idx].to(device)
            losses = pipe.training_step(batch)
            opt.zero_grad(set_to_none=True)
            losses["loss"].backward()
            opt.step()
            running += float(losses["loss"].detach().cpu())
            if swa_last_n > 0 and ep >= swa_start_ep:
                if swa_state is None:
                    swa_state = {k: v.detach().clone() for k, v in pipe.state_dict().items()
                                 if v.dtype.is_floating_point}
                    swa_count = 1
                else:
                    swa_count += 1
                    w = 1.0 / swa_count
                    for k, v in pipe.state_dict().items():
                        if v.dtype.is_floating_point:
                            swa_state[k].mul_(1.0 - w).add_(v.detach(), alpha=w)
        mean_loss = running / max(1, train_noisy.shape[0])
        print(f"[solver-res] epoch {ep+1:3d}/{n_epochs}  loss={mean_loss:.5f}  lr={cur_lr:.2e}",
              flush=True)
    train_time = time.time() - t0
    print(f"[solver-res] training took {train_time:.1f}s", flush=True)
    if swa_state is not None:
        live = {k: v.detach().clone() for k, v in pipe.state_dict().items()}
        merged = dict(live); merged.update(swa_state)
        pipe.load_state_dict(merged, strict=False)
        print(f"[solver-res] loaded SWA-averaged weights ({swa_count} per-step snapshots)", flush=True)

    pipe.eval()
    # Chunked validation so we don't OOM on smaller VRAM (Q5000 16GB) when
    # the full-res residual stack is run on val_n samples at once.
    val_chunk = int(cfg.get("val_chunk", 8))
    pred_list = []
    with torch.no_grad():
        for s in range(0, val_noisy.shape[0], val_chunk):
            pred_list.append(pipe.predict(val_noisy[s:s + val_chunk]))
    pred = torch.cat(pred_list, dim=0)
    val_psnr = float(psnr(pred, val_ref).cpu())
    val_ssim = float(ssim(pred, val_ref).cpu())
    val_rmse = float(((pred - val_ref) ** 2).mean().sqrt().cpu())
    baseline_psnr = float(psnr(ld_fbp, val_ref).cpu())
    baseline_rmse = float(((ld_fbp - val_ref) ** 2).mean().sqrt().cpu())
    headroom = max(0.0, 1.0 - val_rmse / max(baseline_rmse, 1e-12))
    val_score = val_ssim

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        n_show = min(3, cfg["val_n"])
        fig, ax = plt.subplots(n_show, 4, figsize=(12, 3 * n_show))
        if n_show == 1:
            ax = ax[None]
        vmax = float(val_ref.max())
        for i in range(n_show):
            ax[i, 0].imshow(val_ref[i, 0].cpu(), cmap="gray", vmin=0, vmax=vmax)
            ax[i, 0].set_title("reference" if i == 0 else "")
            ax[i, 1].imshow(ld_fbp[i, 0].cpu(), cmap="gray", vmin=0, vmax=vmax)
            ax[i, 1].set_title(f"sparse-view FBP  (PSNR={baseline_psnr:.1f})" if i == 0 else "")
            ax[i, 2].imshow(pred[i, 0].cpu(), cmap="gray", vmin=0, vmax=vmax)
            ax[i, 2].set_title(f"dual-domain ResNet  (PSNR={val_psnr:.1f} SSIM={val_ssim:.3f})" if i == 0 else "")
            ax[i, 3].imshow(val_ph[i, 0].cpu(), cmap="gray")
            ax[i, 3].set_title("phantom" if i == 0 else "")
            for a in ax[i]:
                a.set_axis_off()
        plt.tight_layout()
        figpath = out_dir / "comparison.png"
        plt.savefig(figpath, dpi=120)
        print(f"[solver-res] saved {figpath}", flush=True)
    except Exception as e:
        print(f"[solver-res] figure failed: {e}", flush=True)

    result = {
        "val_score":      val_score,
        "val_psnr":       val_psnr,
        "val_ssim":       val_ssim,
        "val_rmse":       val_rmse,
        "baseline_psnr":  baseline_psnr,
        "baseline_rmse":  baseline_rmse,
        "headroom":       headroom,
        "params_M":       params_total / 1e6,
        "train_n":        cfg["train_n"],
        "val_n":          cfg["val_n"],
        "train_time_s":   train_time,
        "config":         cfg,
    }
    (out_dir / "result.json").write_text(json.dumps(result, indent=2))
    print(f"[solver-res] result: val_score={val_score:.4f}  headroom={headroom:.4f} "
          f"PSNR={val_psnr:.2f} SSIM={val_ssim:.4f}",
          flush=True)
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("out_dir", help="directory to write result.json + comparison.png")
    args = p.parse_args()
    main(Path(args.out_dir))
