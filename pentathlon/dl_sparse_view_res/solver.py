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

iter-35: NAFNet cross-port (Chen et al. 2022, "Simple Baselines for Image
Restoration"). The image_denoiser slot is replaced with a NafNetStack
(LayerNorm -> 1x1 expand -> 3x3 dwconv -> SimpleGate -> 1x1 squeeze -> +x);
proj_denoiser stays as ResidualStack to isolate the image-side change.
The iter-34 AdamW param-group split is preserved on a substrate-agnostic
filter (1-D params and ".alpha" exempt from weight-decay).
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
    "train_n":       400,
    "val_n":         100,

    # Editable: training schedule.
    "epochs":        8,
    "batch_size":    1,
    "lr":            1e-4,
    "optimizer":     "adamw",
    "weight_decay":  1e-4,

    # Editable: residual-stack model architecture.
    "res_blocks":    6,          # iter-2: 8 -> 6 to fit 5-min budget
    "res_channels":  32,         # iter-2: 48 -> 32 (saves ~2.2x flops)
    "res_norm":      "group",    # "group" | "none" | "batch"
    "res_act":       "relu",     # "relu" | "gelu" | "swish"
    "res_kernel":    3,
    "res_dropout":   0.0,
    "residual":      True,       # global residual (predict noise)
    "res_scale":     0.1,        # iter-14: EDSR-style learnable per-block residual scaling, init 0.1

    # iter-35: cross-port NAFNet (Chen et al. 2022) into the IMAGE slot only.
    # proj_denoiser stays as ResidualStack so we isolate the image-side
    # architecture swap. Image slot params:
    #   c=32, 6 NAF blocks, expand=2 -> c_mid=64, depthwise 3x3, SimpleGate
    #   (halves channels: 64 -> 32), 1x1 squeeze (32 -> 32 -> 32). Residual,
    #   zero-init head. Per-block learnable alpha=0.1 (ReZero-style start).
    # Approx params/NAF block (c=32, expand=2, dw):
    #   LayerNorm: 2*32 = 64
    #   pw_in (1x1): 32*64 + 64 = 2112
    #   dw (3x3, groups=64): 64*1*3*3 + 64 = 640
    #   pw_out (1x1, after SimpleGate halve): 32*32 + 32 = 1056
    #   alpha: 1
    #   -> ~3.87k per block, 6 blocks ~23.2k
    # Plus stem (1->32, 3x3, +bias) = 320 and head (32->1, 3x3, +bias) = 289.
    # Image side total: ~23.9k. ResidualStack proj side (iter-34): ~115k.
    # Combined: ~139k -- well under iter-34 baseline (~225k), so we have
    # capacity to widen later if iter-35 lands well.
    "img_denoiser":   "nafnet",   # iter-35: NAFNet for image_denoiser
    "proj_denoiser":  "resstack", # keep ResidualStack on projection slot
    "naf_blocks":     6,          # 6 NAF blocks (matches Agent A's recipe)
    "naf_channels":   32,         # NAF stack width (matches Agent A)
    "naf_expand":     2,          # 1x1 expansion factor
    "naf_dw":         True,       # depthwise 3x3 mid-conv
    "naf_gate":       "simple",   # iter-36 of Agent A's branch confirmed SimpleGate
    "naf_alpha":      0.1,        # EDSR-style learnable residual scale, init 0.1


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
                 res_scale: float | None = None):
        super().__init__()
        pad = kernel // 2
        self.conv1 = nn.Conv2d(c, c, kernel, padding=pad)
        self.n1 = _make_norm(norm, c)
        self.act1 = _make_act(act)
        self.drop = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.conv2 = nn.Conv2d(c, c, kernel, padding=pad)
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
                 res_scale: float | None = None):
        super().__init__()
        self.residual = residual
        pad = kernel // 2
        self.head = nn.Conv2d(1, c, kernel, padding=pad)
        self.head_act = _make_act(act)
        self.blocks = nn.Sequential(*[
            ResBlock(c, kernel=kernel, norm=norm, act=act, dropout=dropout,
                     res_scale=res_scale)
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


# ----------------------------------------------------------------------- #
#  NAFNet — Chen et al. 2022, "Simple Baselines for Image Restoration".
#  Ported from Agent A's pentathlon/dl_sparse_view_iter/solver.py
#  (iter-36 KEEP recipe: SimpleGate gate, 6 blocks at c=32, depthwise 3x3).
# ----------------------------------------------------------------------- #


class _LayerNorm2d(nn.Module):
    """Channel-wise LayerNorm over (B, C, H, W) tensors.

    NAFNet's choice over BatchNorm and GroupNorm: invariant to batch size
    and provides full-channel mixing through a learnable per-channel
    affine. Channel-axis mean/var only — spatial dims are NOT averaged
    over (unlike vanilla 2D LayerNorm on (H, W)).
    """

    def __init__(self, c: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(c))
        self.bias = nn.Parameter(torch.zeros(c))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mu = x.mean(dim=1, keepdim=True)
        var = x.var(dim=1, keepdim=True, unbiased=False)
        x = (x - mu) / torch.sqrt(var + self.eps)
        return x * self.weight.view(1, -1, 1, 1) + self.bias.view(1, -1, 1, 1)


class _SimpleGate(nn.Module):
    """NAFNet's SimpleGate: split tensor along channel dim, multiply halves.

    Replaces both Swish/GELU activation and the multiplicative channel
    attention in a single op. Effective receptive-field-free non-linearity
    (Chen et al. 2022 §4). Halves the channel count.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a, b = x.chunk(2, dim=1)
        return a * b


class NafBlock(nn.Module):
    """A single NAFNet block (image-restoration baseline, no SE / attention).

        x ---> LayerNorm
              -> 1x1 conv  (c -> expand*c)
              -> 3x3 dwconv (groups=expand*c if dw, else dense)
              -> SimpleGate (halves channels) | ReLU | GELU
              -> 1x1 conv  (c_after_gate -> c)
              -> alpha * h
              -> + x

    iter-35 (this branch): gate="simple", dw=True, expand=2, alpha=0.1.
    EDSR/ReZero-style learnable per-block alpha (init 0.1, no weight-decay).
    Zero-init pw_out so the block starts as identity at random init.
    """

    def __init__(self, c: int, expand: int = 2,
                 dw: bool = True, gate: str = "simple",
                 alpha_init: float = 0.1):
        super().__init__()
        assert gate in ("simple", "relu", "gelu"), gate
        self.gate_kind = gate
        c_mid = c * expand
        self.norm = _LayerNorm2d(c)
        self.pw_in = nn.Conv2d(c, c_mid, kernel_size=1)
        if dw:
            self.dw = nn.Conv2d(c_mid, c_mid, kernel_size=3, padding=1,
                                groups=c_mid)
        else:
            self.dw = nn.Conv2d(c_mid, c_mid, kernel_size=3, padding=1)
        if gate == "simple":
            assert c_mid % 2 == 0, "SimpleGate needs even c_mid"
            self.gate = _SimpleGate()
            c_after = c_mid // 2
        elif gate == "gelu":
            self.gate = nn.GELU()
            c_after = c_mid
        else:
            self.gate = nn.ReLU(inplace=True)
            c_after = c_mid
        self.pw_out = nn.Conv2d(c_after, c, kernel_size=1)
        self.alpha = nn.Parameter(torch.tensor(float(alpha_init)))
        nn.init.zeros_(self.pw_out.weight)
        nn.init.zeros_(self.pw_out.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        h = self.pw_in(h)
        h = self.dw(h)
        h = self.gate(h)
        h = self.pw_out(h)
        return x + self.alpha * h


class NafNetStack(nn.Module):
    """Plain stack of NAFNet blocks at full resolution.

    Single-pass denoiser — no iteration, no down/up-sampling, no skip
    connections beyond per-block residual. Stem 3x3 (1 -> c). Body: N
    blocks. Head: 3x3 (c -> 1, zero-init). Predicts noise residual if
    residual=True (output = x - head(blocks(stem(x)))). Matches the
    SmallUNet contract / ResidualStack contract so it plugs into the same
    DualDomainPipeline slot.
    """

    def __init__(self, c: int = 32, n_blocks: int = 6,
                 expand: int = 2, dw: bool = True,
                 gate: str = "simple", alpha_init: float = 0.1,
                 residual: bool = True):
        super().__init__()
        self.residual = residual
        self.stem = nn.Conv2d(1, c, kernel_size=3, padding=1)
        self.blocks = nn.ModuleList([
            NafBlock(c=c, expand=expand, dw=dw, gate=gate,
                     alpha_init=alpha_init)
            for _ in range(n_blocks)
        ])
        self.head = nn.Conv2d(c, 1, kernel_size=3, padding=1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.stem(x)
        for blk in self.blocks:
            h = blk(h)
        y = self.head(h)
        return x - y if self.residual else y


def build_denoisers(cfg: dict) -> tuple[nn.Module, nn.Module]:
    """Return (proj_denoiser, image_denoiser).

    iter-35: support per-slot kind selection via cfg["proj_denoiser"]
    and cfg["img_denoiser"] ("resstack" | "nafnet"). Default both to
    resstack (iter-34 baseline). Operator's iter-35 plan keeps proj as
    resstack and swaps image to nafnet so the architecture difference
    is isolated to the image-domain refinement stage.
    """
    def make_resstack():
        return ResidualStack(
            n_blocks=cfg["res_blocks"],
            c=cfg["res_channels"],
            kernel=cfg["res_kernel"],
            norm=cfg["res_norm"],
            act=cfg["res_act"],
            dropout=cfg["res_dropout"],
            residual=cfg["residual"],
            res_scale=cfg.get("res_scale", None),
        )

    def make_nafnet():
        return NafNetStack(
            c=int(cfg.get("naf_channels", 32)),
            n_blocks=int(cfg.get("naf_blocks", 6)),
            expand=int(cfg.get("naf_expand", 2)),
            dw=bool(cfg.get("naf_dw", True)),
            gate=str(cfg.get("naf_gate", "simple")),
            alpha_init=float(cfg.get("naf_alpha", 0.1)),
            residual=True,
        )

    def make(kind: str):
        if kind == "nafnet":
            return make_nafnet()
        return make_resstack()

    proj_kind = str(cfg.get("proj_denoiser", "resstack"))
    img_kind = str(cfg.get("img_denoiser", "resstack"))
    return make(proj_kind), make(img_kind)


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
    if cfg["optimizer"] == "adam":
        return torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    # iter-34: split params into wd-regulated and wd-excluded groups.
    # iter-35 generalisation: filter on param dim + name suffix rather than
    # ResBlock-specific "n1"/"n2" substrings. This way the same filter works
    # for ResidualStack (GroupNorm) and NafNetStack (LayerNorm + alpha), and
    # any future substrate. The rule:
    #   - p.dim() <= 1            => 1-D/scalar (norm scales, norm biases,
    #                                conv biases, alpha) => no_wd
    #   - name.endswith(".alpha") => EDSR residual scalar => no_wd
    #   - else                   => conv weight (2-D/4-D) => wd
    # alpha (EDSR residual-scaling scalar, init 0.1) lives at module.alpha
    # in ResBlock / NafBlock. AdamW WD pulls alpha toward 0 throughout
    # training, which iter-21 (global wd=0) showed could free alpha but came
    # at cost of under-regularised conv weights (-0.51pp). Targeted: keep WD
    # on conv weights, exclude alpha / biases / norm scales (standard
    # practice for the 1-D / per-channel params that don't benefit from L2
    # shrinkage).
    no_wd_params, wd_params = [], []
    no_wd_names, wd_names = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.dim() <= 1 or name.endswith(".alpha"):
            no_wd_params.append(p)
            no_wd_names.append(name)
        else:
            wd_params.append(p)
            wd_names.append(name)
    wd_param_count = sum(p.numel() for p in wd_params)
    no_wd_param_count = sum(p.numel() for p in no_wd_params)
    print(f"[solver-res] AdamW param-groups: "
          f"wd={len(wd_params)} tensors / {wd_param_count} params (wd={cfg['weight_decay']})  "
          f"no_wd={len(no_wd_params)} tensors / {no_wd_param_count} params (wd=0)",
          flush=True)
    return torch.optim.AdamW(
        [
            {"params": wd_params, "weight_decay": cfg["weight_decay"]},
            {"params": no_wd_params, "weight_decay": 0.0},
        ],
        lr=cfg["lr"],
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
        mean_loss = running / max(1, train_noisy.shape[0])
        print(f"[solver-res] epoch {ep+1:3d}/{cfg['epochs']}  loss={mean_loss:.5f}",
              flush=True)
    train_time = time.time() - t0
    print(f"[solver-res] training took {train_time:.1f}s", flush=True)

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
