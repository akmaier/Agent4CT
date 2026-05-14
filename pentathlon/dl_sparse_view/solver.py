"""DL-Sparse-View CT solver — the file the autoresearch agent edits.

Per the autoresearch contract, **this is the only file the agent should
modify per iteration**. The harness, geometry, projector, simulation, and
metric definitions are *fixed* (see ``ddssl_ldct/`` and
``challenges/dl_sparse_view/README.md``).

Inputs:
    - reads the geometry from the constants below
    - synthesises random-ellipse phantoms as a stand-in for the real
      AAPM DL-Sparse-View breast phantom (real data download is gated;
      see challenges/dl_sparse_view/README.md). The forward problem is
      the same: 2D fan-beam, sparse views, perfectly-known ground truth.

Outputs (written into the directory passed as argv[1]):
    - result.json with keys: val_score, val_psnr, val_ssim, val_rmse,
      headroom, baseline_score, oracle_score, params_M, train_n,
      change_class, rationale, advice_for_others
    - comparison.png — reference / low-view FBP / dual-domain / phantom

Scoring (Choice A, headroom-recovered):
    baseline = sparse-view FBP, no learning
    oracle   = RMSE = 0 against the truth (score = 1.0 by construction)
    score    = 1 - val_rmse / baseline_rmse  ∈ [0, 1]

Anti-overfit rules (see docs/agents.md):
    - total trainable parameters under ~10 × train_n × pixels_per_sample
    - stage_gap > 0 surfaces in the journal; agent must regularise next
    - one change per iteration
    - cite the iteration that inspired the change in --rationale
"""
from __future__ import annotations
import argparse
import json
import math
import sys
import time
from pathlib import Path

# Ensure the project package is importable when this script is run from a
# Slurm-allocated working dir.
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
from ddssl_ldct.models import SmallUNet, TrainableBilateralFilter2d
from ddssl_ldct.training import DualDomainPipeline, train
from ddssl_ldct.metrics import psnr, ssim


# ----------------------------------------------------------------------- #
#  CONFIG  —  the agent edits this block + the model-builder below.
# ----------------------------------------------------------------------- #

CONFIG = {
    # Geometry (fixed for now — Wagner / Siemens AS @ 128 sparse views).
    "image_size":    512,
    "pixel_spacing": 0.7,
    "n_angles":      128,       # sparse view!
    "n_det":         736,
    "det_spacing":   1.2858,
    "sod":           595.0,
    "sdd":           1085.6,

    # Training subset for the 5-minute iteration budget.
    "train_n":       400,
    "val_n":         100,

    # Editable: training schedule.
    "epochs":        6,    # iter-62: match Agent A
    "batch_size":    1,
    "lr":            8e-5,    # iter-79: 1e-4 -> 8e-5 (fine bisection lower side; never directly tested on main)
    "optimizer":     "adam",    # iter-77 KEEP +0.01pp marginal
    # iter-32: weight_decay 1e-4 -> 0. Hypothesis: WD also decays the
    # learnable per-block alpha scalars (init 0.1), pulling them
    # toward 0 and preventing them from growing. WD=0 lets alpha
    # discover its useful magnitude. Spawn B iter-22 advice: WD in
    # [1e-5, 1e-3] is essentially equivalent on this arch — so the
    # remaining-params regularisation cost is near-zero.
    "weight_decay":  1e-4, # iter-59: split mode below; this is the conv-weights value

    # Editable: model architecture.
    "unet_c":        16,
    # "unet" | "unet_plus_bf" (Wagner 2022 BF tail) | "resnet"
    #   resnet: plain residual stack at full res (DnCNN / Sidky-2022 top-team
    #           family, cross-ported from spawn agent dl-sparse-view-res
    #           iter-2 which beat the U-Net+BF baseline on headroom).
    # iter-56: cross-port Agent A NAFNet (c=32, 6 blocks, ReLU gate, dw=True)
    # onto main's wd=0 substrate. "nafnet" routes via build_denoisers.
    "img_denoiser":  "nafnet",
    "naf_blocks":    6,   # iter-67 KEEP (iter-70 7-blocks timed out)
    "naf_alpha_init": 0.1,    # iter-78 closed 0.15 at -0.14pp
    # iter-61: gate ReLU -> GELU (Agent A iter-38 change, +0.04pp on their substrate).
    "naf_gate": "gelu",
    # iter-57: stack 3 trainable Wagner BF tails on the image NAFNet
    # output. Agent A iter-29/31/38 confirmed +0.28/+0.63/+0.31pp from
    # 1/2/3 BFs (on their wd=1e-4 substrate). Tests if the BF-stacking
    # lift transfers to mains wd=0 substrate.
    # iter-63: 3 -> 4 BF tails. Agent A had +0.28/+0.63/+0.31pp from
    # 1/2/3 BFs (compounding) and timed out testing iter-33 bf_kernel=9.
    # Push naf_n_bf to 4 (smaller compute cost than wider kernel).
    "naf_n_bf":      8,    # iter-67 KEEP (iter-68 9 BFs timed out — BF saturation at 8)
    # iter-58: add SWA over the last 6-of-8 epoch-end snapshots
    # (full-window mode that won Agent A iter-36, +0.42pp). Tests
    # whether the SWA-on-NAFNet-BF composition transfers to mains wd=0.
    "swa_last_n":    4,   # iter-71 KEEP (iter-72 narrower DISCARD, iter-73 wider near-flat)
    "bf_kernel":     7,   # iter-74 closed kernel=5 at -0.33pp
    "bf_sigma_r":    0.01,   # iter-76 closed 0.02 at -0.94pp (BF basin tight at 0.01)
    # Editable: residual-stack architecture (only used when img_denoiser="resnet").
    # Default to spawn agent B iter-2 winner (6 blocks, c=32, GroupNorm, ReLU).
    "res_blocks":    6,
    "res_channels":  32,
    "res_norm":      "group",   # group | batch | none
    "res_act":       "relu",    # relu | gelu | swish
    "res_kernel":    3,
    "res_dropout":   0.0,
    "res_residual":  True,      # global residual head (predicts noise)
    # iter-27: per-block residual scaling scalar (EDSR/ReZero). Init at
    # 0.1 so the network starts near-identity. Cross-port of spawn agent
    # B iter-14 winner.
    "res_scale_init": 0.1,

    # Noise simulation — kept fixed so headroom is comparable across iter.
    "noise_i0":      5e4,
    "noise_sigma_e": 5.0,
    "seed":          42,

    # Intensity calibration. CT images live on a standard scale (HU for
    # real data; for the synthetic ellipse phantoms here the canonical
    # max is 0.05 in attenuation-coefficient units). PSNR and SSIM are
    # computed with a FIXED data_range = display_max - display_min so
    # numbers are comparable across iterations and across agents — auto
    # data-range drifts with FBP overshoot and breaks comparability.
    # Display vmin/vmax are the same fixed range for every column so a
    # comparison image's grey value means the same thing across columns.
    "display_min":   0.0,
    "display_max":   0.05,
}


# ----------------------------------------------------------------------- #
#  Model builder — agent edits HERE for architecture changes.
# ----------------------------------------------------------------------- #

class UNetPlusBF(nn.Module):
    """SmallUNet followed by a Wagner-2022 4-param trainable bilateral
    filter. The BF is interpretable (3 spatial σ + 1 range σ) and acts
    as an edge-preserving sharpening / denoising layer on top of the
    U-Net's coarse denoising."""
    def __init__(self, c: int = 16, bf_kernel: int = 7,
                 bf_sigma_x: float = 1.5, bf_sigma_y: float = 1.5,
                 bf_sigma_r: float = 0.01):
        super().__init__()
        self.unet = SmallUNet(c=c)
        self.bf = TrainableBilateralFilter2d(
            kernel_size=bf_kernel,
            sigma_x=bf_sigma_x, sigma_y=bf_sigma_y, sigma_r=bf_sigma_r)
    def forward(self, x):
        return self.bf(self.unet(x))


# ----------------------------------------------------------------------- #
#  Residual-stack denoiser  (cross-ported from spawn agent
#  dl-sparse-view-res iter-2 winner: 6 blocks @ c=32, GroupNorm + ReLU,
#  global residual head; 0.225 M params total, beats U-Net+BF on headroom).
# ----------------------------------------------------------------------- #

# iter-56: NAFNet cross-port from Agent A iter-38 (team leader hr=0.5988).
# Test whether Agent A's NAFNet block + main's wd=0 substrate beats the
# saturated residual-α basin. Composition test (multi-knob).

class _LayerNorm2d(nn.Module):
    """Per-(B, H, W) LayerNorm over channels (NAFNet's choice)."""
    def __init__(self, c: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(c))
        self.bias = nn.Parameter(torch.zeros(c))
        self.eps = eps
    def forward(self, x):
        mu = x.mean(dim=1, keepdim=True)
        var = x.var(dim=1, keepdim=True, unbiased=False)
        x = (x - mu) / torch.sqrt(var + self.eps)
        return x * self.weight.view(1, -1, 1, 1) + self.bias.view(1, -1, 1, 1)


class NafBlock(nn.Module):
    def __init__(self, c: int, expand: int = 2, alpha_init: float = 0.1,
                 gate: str = "gelu"):
        super().__init__()
        c_mid = c * expand
        self.norm = _LayerNorm2d(c)
        self.pw_in = nn.Conv2d(c, c_mid, 1)
        self.dw = nn.Conv2d(c_mid, c_mid, 3, padding=1, groups=c_mid)
        # iter-61: matching Agent A iter-38 GELU gate (+0.04pp on their substrate).
        self.gate = nn.GELU() if gate == "gelu" else nn.ReLU(inplace=True)
        self.pw_out = nn.Conv2d(c_mid, c, 1)
        self.alpha = nn.Parameter(torch.tensor(float(alpha_init)))
        nn.init.zeros_(self.pw_out.weight)
        nn.init.zeros_(self.pw_out.bias)
    def forward(self, x):
        h = self.norm(x); h = self.pw_in(h); h = self.dw(h)
        h = self.gate(h); h = self.pw_out(h)
        return x + self.alpha * h


class NafNetStack(nn.Module):
    def __init__(self, c: int = 32, n_blocks: int = 6, alpha_init: float = 0.1,
                 residual: bool = True, n_bf: int = 0, bf_kernel: int = 7,
                 bf_sigma_x: float = 1.5, bf_sigma_y: float = 1.5,
                 bf_sigma_r: float = 0.01, gate: str = "gelu"):
        super().__init__()
        self.residual = residual
        self.stem = nn.Conv2d(1, c, 3, padding=1)
        self.blocks = nn.ModuleList([NafBlock(c, alpha_init=alpha_init, gate=gate) for _ in range(n_blocks)])
        self.head = nn.Conv2d(c, 1, 3, padding=1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)
        # iter-57: stack `n_bf` trainable Wagner BF tails on the output.
        # Cross-port of Agent A iter-29/31/38: BF tails compose well with
        # NAFNet (+0.28pp / +0.63pp / +0.31pp from 1/2/3 BFs).
        self.bfs = nn.ModuleList([
            TrainableBilateralFilter2d(
                kernel_size=bf_kernel, sigma_x=bf_sigma_x,
                sigma_y=bf_sigma_y, sigma_r=bf_sigma_r)
            for _ in range(n_bf)
        ])
    def forward(self, x):
        h = self.stem(x)
        for blk in self.blocks:
            h = blk(h)
        y = self.head(h)
        out = x - y if self.residual else y
        for bf in self.bfs:
            out = bf(out)
        return out


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
    """Residual block: conv -> norm -> act -> (drop) -> conv -> norm -> + alpha*h.

    iter-27 (cross-port of spawn agent B iter-14): adds a learnable
    per-block scaling scalar alpha (init 0.1), so each block returns
    x + alpha * h instead of x + h. EDSR-style residual scaling
    (Lim et al. 2017) + ReZero spirit (Bachlechner et al. 2020) —
    the network starts as a near-identity and the optimiser discovers
    useful residual magnitudes. 12 extra scalar params total (6 blocks
    x 2 nets), no architectural cost.
    """
    def __init__(self, c: int, kernel: int = 3, norm: str = "group",
                 act: str = "relu", dropout: float = 0.0,
                 res_scale_init: float = 0.1):
        super().__init__()
        pad = kernel // 2
        self.conv1 = nn.Conv2d(c, c, kernel, padding=pad)
        self.n1 = _make_norm(norm, c)
        self.act1 = _make_act(act)
        self.drop = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.conv2 = nn.Conv2d(c, c, kernel, padding=pad)
        self.n2 = _make_norm(norm, c)
        self.alpha = nn.Parameter(torch.tensor(float(res_scale_init)))
    def forward(self, x):
        h = self.conv1(x)
        h = self.n1(h)
        h = self.act1(h)
        h = self.drop(h)
        h = self.conv2(h)
        h = self.n2(h)
        return x + self.alpha * h


class ResidualStack(nn.Module):
    """Single-resolution residual stack with zero-init tail (identity start).

    Architecture:
        head conv (1 -> c) -> act
        N x ResBlock(c)
        tail conv (c -> 1, zero-init)
        + (optional) global residual: y = x - tail(features)   (predicts noise)
    """
    def __init__(self, n_blocks: int = 6, c: int = 32, kernel: int = 3,
                 norm: str = "group", act: str = "relu",
                 dropout: float = 0.0, residual: bool = True,
                 res_scale_init: float = 0.1):
        super().__init__()
        self.residual = residual
        pad = kernel // 2
        self.head = nn.Conv2d(1, c, kernel, padding=pad)
        self.head_act = _make_act(act)
        self.blocks = nn.Sequential(*[
            ResBlock(c, kernel=kernel, norm=norm, act=act, dropout=dropout,
                     res_scale_init=res_scale_init)
            for _ in range(n_blocks)
        ])
        self.tail = nn.Conv2d(c, 1, kernel, padding=pad)
        nn.init.zeros_(self.tail.weight)
        nn.init.zeros_(self.tail.bias)
    def forward(self, x):
        h = self.head(x)
        h = self.head_act(h)
        h = self.blocks(h)
        y = self.tail(h)
        return x - y if self.residual else y


def build_denoisers(cfg: dict) -> tuple[nn.Module, nn.Module]:
    """Return (proj_denoiser, image_denoiser)."""
    family = cfg.get("img_denoiser", "unet")
    if family == "nafnet":
        # iter-56: cross-port Agent A iter-38 NAFNet block. Symmetric on
        # both domains, wd=0 substrate (main's iter-32 keep).
        def make_naf(with_bf: bool):
            return NafNetStack(
                c=cfg["res_channels"],
                n_blocks=cfg.get("naf_blocks", 6),
                alpha_init=cfg.get("naf_alpha_init", 0.1),
                residual=cfg["res_residual"],
                n_bf=cfg.get("naf_n_bf", 0) if with_bf else 0,
                bf_kernel=cfg.get("bf_kernel", 7),
                bf_sigma_r=cfg.get("bf_sigma_r", 0.01),
                gate=cfg.get("naf_gate", "gelu"),
            )
        # BF tail on image domain only (Agent B iter-17 finding: sino
        # domain has no anatomical edges for BF to exploit).
        return make_naf(with_bf=False), make_naf(with_bf=True)
    if family == "resnet":
        def make_res():
            return ResidualStack(
                n_blocks=cfg["res_blocks"],
                c=cfg["res_channels"],
                kernel=cfg["res_kernel"],
                norm=cfg["res_norm"],
                act=cfg["res_act"],
                dropout=cfg["res_dropout"],
                residual=cfg["res_residual"],
                res_scale_init=cfg.get("res_scale_init", 0.1),
            )
        # Symmetric residual denoisers on both domains (matches spawn agent B).
        return make_res(), make_res()
    c = cfg["unet_c"]
    proj = SmallUNet(c=c)
    if family == "unet_plus_bf":
        img = UNetPlusBF(c=c)
    else:
        img = SmallUNet(c=c)
    return proj, img


# ----------------------------------------------------------------------- #
#  Data + training harness — leave alone (changes here count as solver
#  changes, but are usually unproductive — prefer model / schedule edits).
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


def make_optimizer(params, cfg):
    if cfg["optimizer"] == "adam":
        return torch.optim.Adam(params, lr=cfg["lr"])
    return torch.optim.AdamW(params, lr=cfg["lr"], weight_decay=cfg["weight_decay"])


def main(out_dir: Path, cfg: dict | None = None) -> dict:
    cfg = {**CONFIG, **(cfg or {})}
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[solver] device={device}  config={json.dumps(cfg)}", flush=True)
    torch.manual_seed(cfg["seed"])

    geom = build_geometry(cfg)
    train_ph, train_clean, train_noisy = build_dataset(
        geom, cfg["train_n"], cfg["seed"], cfg["noise_i0"], cfg["noise_sigma_e"], device)
    val_ph, val_clean, val_noisy = build_dataset(
        geom, cfg["val_n"], cfg["seed"] + 1000, cfg["noise_i0"], cfg["noise_sigma_e"], device)

    # Reference (oracle-ish) recon used for PSNR/SSIM logging.
    with torch.no_grad():
        R_full = PyronnFanBeamProjector(geom).to(device)
        val_ref = R_full.fbp(val_clean)
        ld_fbp = R_full.fbp(val_noisy)

    # --- Build the dual-domain pipeline -------------------------------- #
    proj_dn, img_dn = build_denoisers(cfg)
    pipe = DualDomainPipeline(
        geometry=geom, proj_denoiser=proj_dn, image_denoiser=img_dn,
    ).to(device)
    # Replace train's default optimiser with our config.
    params_total = sum(p.numel() for p in pipe.parameters() if p.requires_grad)
    print(f"[solver] params = {params_total/1e6:.3f} M", flush=True)

    # Manual training loop (the upstream `train` helper hard-codes Adam;
    # we use our chosen optimiser instead).
    opt = make_optimizer(pipe.parameters(), cfg)
    # iter-60: per-STEP SWA averaging (Agent A vectorised pattern).
    # `swa_last_n` is the # of EPOCHS at the END to include in SWA;
    # we average the model params after EVERY optimiser step inside
    # those epochs, giving ~400× more samples than per-epoch snapshots.
    swa_last_n = int(cfg.get("swa_last_n", 0))
    swa_start_ep = cfg["epochs"] - swa_last_n  # SWA active for ep >= swa_start_ep
    swa_state = None
    swa_count = 0
    t0 = time.time()
    for ep in range(cfg["epochs"]):
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
                # Per-step SWA accumulation: shadow = ((n-1)*shadow + p) / n.
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
        print(f"[solver] epoch {ep+1:3d}/{cfg['epochs']}  loss={mean_loss:.5f}",
              flush=True)
    train_time = time.time() - t0
    print(f"[solver] training took {train_time:.1f}s", flush=True)
    if swa_state is not None:
        live = {k: v.detach().clone() for k, v in pipe.state_dict().items()}
        merged = dict(live); merged.update(swa_state)
        pipe.load_state_dict(merged, strict=False)
        print(f"[solver] loaded SWA-averaged weights ({swa_count} per-step snapshots)", flush=True)

    # --- Validation --------------------------------------------------- #
    pipe.eval()
    with torch.no_grad():
        # Chunked validation so a smaller GPU (or a wider BF kernel) does
        # not OOM on val_n images in one batch. Concatenate the per-chunk
        # predictions.
        chunk = cfg.get("val_chunk", 10)
        preds = []
        for i in range(0, val_noisy.shape[0], chunk):
            preds.append(pipe.predict(val_noisy[i:i + chunk]))
        pred = torch.cat(preds, dim=0)
    # Log magnitudes so we can verify projection-count scaling in the journal.
    print(f"[solver] phantom range = [{float(val_ph.min()):.4f}, "
          f"{float(val_ph.max()):.4f}]", flush=True)
    print(f"[solver] val_ref range = [{float(val_ref.min()):.4f}, "
          f"{float(val_ref.max()):.4f}]", flush=True)
    print(f"[solver] ld_fbp  range = [{float(ld_fbp.min()):.4f}, "
          f"{float(ld_fbp.max()):.4f}]", flush=True)
    print(f"[solver] pred    range = [{float(pred.min()):.4f}, "
          f"{float(pred.max()):.4f}]", flush=True)

    # Fixed data_range for PSNR/SSIM (calibration to the canonical phantom
    # max). Without this, auto-data-range = val_ref.max() - val_ref.min()
    # drifts iteration-to-iteration with FBP overshoot and silently
    # changes the PSNR denominator, breaking cross-iter comparability.
    data_range = cfg["display_max"] - cfg["display_min"]
    val_psnr = float(psnr(pred, val_ref, data_range=data_range).cpu())
    val_ssim = float(ssim(pred, val_ref, data_range=data_range).cpu())
    val_rmse = float(((pred - val_ref) ** 2).mean().sqrt().cpu())
    baseline_psnr = float(psnr(ld_fbp, val_ref, data_range=data_range).cpu())
    baseline_rmse = float(((ld_fbp - val_ref) ** 2).mean().sqrt().cpu())
    # Headroom = fraction of the gap between noisy baseline and perfect
    # reconstruction (oracle = 0 RMSE).
    headroom = max(0.0, 1.0 - val_rmse / max(baseline_rmse, 1e-12))
    val_score = val_ssim   # primary single-number score for the run

    # --- Comparison figure ------------------------------------------- #
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        n_show = min(3, cfg["val_n"])
        fig, ax = plt.subplots(n_show, 4, figsize=(12, 3 * n_show))
        if n_show == 1:
            ax = ax[None]
        # Common display vmin/vmax = the calibration scale (display_min,
        # display_max). All four columns then share the *same* grey
        # mapping — a grey value of, say, 50% means the same physical
        # intensity in every column. Anything outside [display_min,
        # display_max] (FBP overshoot etc.) is clipped at the display, not
        # in the data.
        vmin, vmax = cfg["display_min"], cfg["display_max"]
        for i in range(n_show):
            ax[i, 0].imshow(val_ref[i, 0].cpu(), cmap="gray", vmin=vmin, vmax=vmax)
            ax[i, 0].set_title("reference" if i == 0 else "")
            ax[i, 1].imshow(ld_fbp[i, 0].cpu(), cmap="gray", vmin=vmin, vmax=vmax)
            ax[i, 1].set_title(f"sparse-view FBP  (PSNR={baseline_psnr:.1f})" if i == 0 else "")
            ax[i, 2].imshow(pred[i, 0].cpu(), cmap="gray", vmin=vmin, vmax=vmax)
            ax[i, 2].set_title(f"dual-domain  (PSNR={val_psnr:.1f} SSIM={val_ssim:.3f})" if i == 0 else "")
            ax[i, 3].imshow(val_ph[i, 0].cpu(), cmap="gray", vmin=vmin, vmax=vmax)
            ax[i, 3].set_title("phantom" if i == 0 else "")
            for a in ax[i]:
                a.set_axis_off()
        plt.tight_layout()
        figpath = out_dir / "comparison.png"
        plt.savefig(figpath, dpi=120)
        print(f"[solver] saved {figpath}", flush=True)
    except Exception as e:
        print(f"[solver] figure failed: {e}", flush=True)

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
    print(f"[solver] result: val_score={val_score:.4f}  headroom={headroom:.4f} "
          f"PSNR={val_psnr:.2f} SSIM={val_ssim:.4f}",
          flush=True)
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("out_dir", help="directory to write result.json + comparison.png")
    args = p.parse_args()
    main(Path(args.out_dir))
