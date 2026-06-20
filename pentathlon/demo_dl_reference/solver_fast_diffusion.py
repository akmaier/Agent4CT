"""Reference: fast few-step diffusion priors trained on truth images.

This is a NEW solver family that mirrors ``solver_ddpm.py`` but trains a
**rectified-flow / flow-matching** prior instead of an ε-prediction DDPM.
Flow-matching priors generate (and reconstruct) in a handful of Euler
steps rather than the 50-1000 reverse steps a DDPM needs, which is the
whole point: "fast diffusion".

Two prior *domains*, selected via ``fd_domain``:

  * ``"pixel"``   — flow matching directly in image (pixel) space. The
    velocity field U-Net has 1 input / 1 output channel.
  * ``"wavelet"`` — flow matching in the single-level 2D Haar DWT domain
    (the WDM idea, Friedrich et al. 2024). The velocity field U-Net has
    4 input / 4 output channels (LL,LH,HL,HH), each H/2 x W/2.

Two data *regimes*, selected via ``fd_mode`` (mirrors ``ddpm_mode``):

  * ``"unconstrained"`` — ``fd_n_train`` train slices ("best the prior can
    be" upper bound).
  * ``"constrained"`` — only the ``fd_n_train_constrained`` train slices the
    other dl_reference solvers see (apples-to-apples "what can a diffusion
    prior add with no extra data?").

**BOTH modes load from the "train" split only** — never val / test — so
there is no leakage into the inference-time evaluation, exactly as
``solver_ddpm.py`` does.

The four advertised solvers (pixel/wavelet x constrained/unconstrained)
are just four checkpoints + four registry entries; this single file
trains all four via config. ``solver_fast_recon.py`` then loads any one
checkpoint and reconstructs from a measured sinogram with a corrected
few-step DC-guided sampler.

Checkpoint contract (saved at ``fd_ckpt``):
    model_state       EMA weights (NOT the raw training weights)
    fd_domain         "pixel" | "wavelet"
    fd_ch             base channels of the FlowUNet
    fd_out_scale      μ -> [0,1] normaliser
    fd_mode           "constrained" | "unconstrained"
    n_train           number of train slices actually used
    train_seed        seed offset used for the train split (phantoms only)
    final_train_loss  last epoch's mean flow-MSE
    history           list of {epoch, train_loss, val_loss, elapsed_s}
    config            the full resolved cfg dict

Training surrogate (for the autoresearch dashboard), mirroring DDPM's
ε-loss surrogate but using the flow-matching velocity MSE on a held-out
val split:
    val_score = -val_flow_loss
    headroom  = max(0, 1 - val_flow_loss / baseline)      baseline = 1.0

Citations:
  Liu X. et al. "Flow Straight and Fast: Learning to Generate and
    Transfer Data with Rectified Flow." 2022. arXiv:2209.03003
  Lipman Y. et al. "Flow Matching for Generative Modeling." 2022.
    arXiv:2210.02747
  Friedrich P. et al. "WDM: 3D Wavelet Diffusion Models for High-
    Resolution Medical Image Synthesis." 2024. arXiv:2402.19043
"""
from __future__ import annotations
import argparse, copy, json, math, os, sys, time
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
from challenges.demo_dl.geometry import DEFAULTS as DEMO_DL_DEFAULTS
# Reuse the time-embedding + ResBlk building blocks from the DDPM trainer so
# a single architecture definition is the source of truth.
from pentathlon.demo_dl_reference.solver_ddpm import (
    TimeEmb, ResBlk, _g, build_phantoms,
)


CONFIG = {
    **DEMO_DL_DEFAULTS,
    # Prior domain + data regime
    "fd_domain":              "pixel",          # "pixel" | "wavelet"
    "fd_mode":                "unconstrained",  # "constrained" | "unconstrained"
    "fd_n_train":             3000,             # used only when unconstrained
    "fd_n_train_constrained": 200,              # matches other dl_ref solvers' train_n
    "fd_n_val":               100,              # held-out flow-loss val set
    # Normalisation: μ -> [0,1]. auto=display_max (set below for staged data).
    "fd_out_scale":           0.05,
    "fd_out_scale_auto":      True,
    # Architecture
    "fd_ch":                  32,               # base channels of the FlowUNet
    # Optimiser
    "fd_epochs":              30,
    "fd_batch":               8,
    "fd_lr":                  2e-4,
    "fd_weight_decay":        0.0,
    "fd_ema_decay":           0.999,
    # Few-step generation for the training-time comparison figure.
    "fd_gen_steps":           6,
    # Wall clamp (s)
    "fd_train_wall_s":        3600,
    # Checkpoint output path. Override via env / cfg for per-variant files.
    "fd_ckpt":                "/cluster/maier/Agent4CT/checkpoints/fast_diffusion_search.pt",
}


# ===========================================================================
# Single-level 2D Haar DWT / IDWT (orthonormal)
# ===========================================================================
#
# One level of the orthonormal 2D Haar transform. Given x of shape
# (B, 1, H, W) with H, W even, we form the 2x2-block averages / differences:
#
#     a = x[..., 0::2, 0::2]   b = x[..., 0::2, 1::2]
#     c = x[..., 1::2, 0::2]   d = x[..., 1::2, 1::2]
#
#     LL = (a + b + c + d) / 2     (approximation)
#     LH = (a + b - c - d) / 2     (horizontal-edge detail)
#     HL = (a - b + c - d) / 2     (vertical-edge detail)
#     HH = (a - b - c + d) / 2     (diagonal detail)
#
# The /2 factor (= 1/sqrt(2) per 1-D axis, applied twice) makes the
# transform orthonormal, so IDWT is the exact transpose / inverse:
#
#     a = (LL + LH + HL + HH) / 2 ,  b = (LL + LH - HL - HH) / 2 ,
#     c = (LL - LH + HL - HH) / 2 ,  d = (LL - LH - HL + HH) / 2
#
# Output of DWT: (B, 4, H/2, W/2) stacked as [LL, LH, HL, HH].
# IDWT is the exact left/right inverse of DWT for any even H, W.

def haar_dwt(x: torch.Tensor) -> torch.Tensor:
    """Single-level orthonormal 2D Haar DWT.

    ``x``: (B, 1, H, W) with H, W even -> (B, 4, H/2, W/2) = [LL, LH, HL, HH].
    """
    if x.shape[1] != 1:
        raise ValueError(f"haar_dwt expects 1 input channel, got {x.shape[1]}")
    a = x[..., 0::2, 0::2]
    b = x[..., 0::2, 1::2]
    c = x[..., 1::2, 0::2]
    d = x[..., 1::2, 1::2]
    ll = (a + b + c + d) * 0.5
    lh = (a + b - c - d) * 0.5
    hl = (a - b + c - d) * 0.5
    hh = (a - b - c + d) * 0.5
    return torch.cat([ll, lh, hl, hh], dim=1)   # (B, 4, H/2, W/2)


def haar_idwt(y: torch.Tensor) -> torch.Tensor:
    """Inverse of :func:`haar_dwt`.

    ``y``: (B, 4, H/2, W/2) = [LL, LH, HL, HH] -> (B, 1, H, W). Exact inverse
    (and differentiable), so DPS gradients flow through it in the recon.
    """
    if y.shape[1] != 4:
        raise ValueError(f"haar_idwt expects 4 input channels, got {y.shape[1]}")
    ll, lh, hl, hh = y[:, 0:1], y[:, 1:2], y[:, 2:3], y[:, 3:4]
    a = (ll + lh + hl + hh) * 0.5
    b = (ll + lh - hl - hh) * 0.5
    c = (ll - lh + hl - hh) * 0.5
    d = (ll - lh - hl + hh) * 0.5
    B, _, h2, w2 = ll.shape
    out = torch.zeros(B, 1, h2 * 2, w2 * 2, device=y.device, dtype=y.dtype)
    out[..., 0::2, 0::2] = a
    out[..., 0::2, 1::2] = b
    out[..., 1::2, 0::2] = c
    out[..., 1::2, 1::2] = d
    return out


def to_model_domain(x_img_norm: torch.Tensor, domain: str) -> torch.Tensor:
    """Map a normalised image (B,1,H,W) into the model's training domain."""
    return haar_dwt(x_img_norm) if domain == "wavelet" else x_img_norm


def from_model_domain(x_dom: torch.Tensor, domain: str) -> torch.Tensor:
    """Map a model-domain tensor back to a normalised image (B,1,H,W)."""
    return haar_idwt(x_dom) if domain == "wavelet" else x_dom


# ===========================================================================
# Velocity-field U-Net (flow matching)
# ===========================================================================
#
# COPY of solver_ddpm.SmallDDPM but with parameterised in/out channels so the
# same network serves the pixel prior (in=out=1) and the wavelet prior
# (in=out=4). Keeps the zero-init head (so the network starts as the identity
# velocity ~ 0, which is a stable init for flow matching). The continuous
# flow time t in [0,1] is fed to the DDPM-style TimeEmb after scaling by 1000
# (so it occupies the same numeric band the sinusoidal embedding expects).

class FlowUNet(nn.Module):
    """Tiny 3-level time-embedded U-Net predicting the flow velocity v(x_t, t).

    Architecture is identical to ``solver_ddpm.SmallDDPM`` except in/out
    channels are configurable (1 for pixel, 4 for wavelet).
    """
    def __init__(self, in_ch: int = 1, out_ch: int = 1, ch: int = 32):
        super().__init__()
        te_dim = ch * 4
        self.ch = ch
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.t_emb = nn.Sequential(TimeEmb(ch), nn.Linear(ch, te_dim),
                                   nn.SiLU(), nn.Linear(te_dim, te_dim))
        self.enc1 = ResBlk(in_ch, ch, te_dim)
        self.enc2 = ResBlk(ch, ch * 2, te_dim)
        self.enc3 = ResBlk(ch * 2, ch * 4, te_dim)
        self.bot = ResBlk(ch * 4, ch * 4, te_dim)
        self.dec3 = ResBlk(ch * 8, ch * 2, te_dim)
        self.dec2 = ResBlk(ch * 4, ch, te_dim)
        self.dec1 = ResBlk(ch * 2, ch, te_dim)
        self.head = nn.Conv2d(ch, out_ch, 1)
        nn.init.zeros_(self.head.weight); nn.init.zeros_(self.head.bias)

    def forward(self, x, t):
        h, w = x.shape[-2:]
        ph = (8 - h % 8) % 8; pw = (8 - w % 8) % 8
        if ph or pw: x = F.pad(x, (0, pw, 0, ph), mode="reflect")
        te = self.t_emb(t.float())
        e1 = self.enc1(x, te)
        e2 = self.enc2(F.avg_pool2d(e1, 2), te)
        e3 = self.enc3(F.avg_pool2d(e2, 2), te)
        b = self.bot(F.avg_pool2d(e3, 2), te)
        b = F.interpolate(b, scale_factor=2, mode="nearest")
        d3 = self.dec3(torch.cat([b, e3], 1), te)
        d3 = F.interpolate(d3, scale_factor=2, mode="nearest")
        d2 = self.dec2(torch.cat([d3, e2], 1), te)
        d2 = F.interpolate(d2, scale_factor=2, mode="nearest")
        d1 = self.dec1(torch.cat([d2, e1], 1), te)
        y = self.head(d1)
        if ph or pw: y = y[..., :h, :w]
        return y


# ---------------------------------------------------------------------------
# EMA helper (the audit's missing piece — checkpoint stores EMA weights).
# ---------------------------------------------------------------------------
class EMA:
    """Exponential moving average of model parameters (and buffers).

    Maintains a shadow copy that trails the training weights with decay
    ``d``: ``shadow <- d*shadow + (1-d)*param``. ``copy_to(model)`` swaps the
    shadow into a model for evaluation / checkpointing.
    """
    def __init__(self, model: nn.Module, decay: float):
        self.decay = float(decay)
        self.shadow = copy.deepcopy(model).eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        d = self.decay
        for s, p in zip(self.shadow.parameters(), model.parameters()):
            s.mul_(d).add_(p.detach(), alpha=1.0 - d)
        # Buffers (e.g. GroupNorm has none, but be safe) tracked exactly.
        for s, p in zip(self.shadow.buffers(), model.buffers()):
            s.copy_(p)

    def state_dict(self) -> dict:
        return self.shadow.state_dict()


# ===========================================================================
# Flow-matching loss + few-step generation
# ===========================================================================

def flow_loss_batch(model, x0_dom, device):
    """Rectified-flow / flow-matching MSE for one batch of *model-domain*
    clean samples ``x0_dom`` (already DWT'd in wavelet mode).

    x1 ~ N(0, I);  t ~ U(0,1);  x_t = (1-t) x0 + t x1;  target v = x1 - x0.
    Loss = MSE(v_theta(x_t, t*1000), v).
    """
    x1 = torch.randn_like(x0_dom)
    t = torch.rand(x0_dom.shape[0], device=device)        # U(0,1)
    tv = t.view(-1, 1, 1, 1)
    x_t = (1.0 - tv) * x0_dom + tv * x1
    target = x1 - x0_dom
    pred = model(x_t, t * 1000.0)                          # scale into TimeEmb band
    return F.mse_loss(pred, target)


def train_one_epoch(model, ema, train_dom, batch, opt, device):
    """One epoch of flow-matching training over model-domain clean samples."""
    N = train_dom.shape[0]
    perm = torch.randperm(N)
    running = 0.0; nb = 0
    for i in range(0, N, batch):
        idx = perm[i:i + batch]
        x0 = train_dom[idx]
        loss = flow_loss_batch(model, x0, device)
        opt.zero_grad(); loss.backward(); opt.step()
        ema.update(model)
        running += float(loss.detach().cpu()); nb += 1
    return running / max(1, nb)


@torch.no_grad()
def eval_flow_loss(model, val_dom, batch, device, n_t_per_image=4):
    """Mean flow-matching velocity MSE on held-out samples (model domain),
    averaged over several random (t, noise) draws per image to cut variance.
    Mirrors ``solver_ddpm.eval_eps_loss``."""
    was_training = model.training
    model.eval()
    N = val_dom.shape[0]
    tot = 0.0; nb = 0
    for i in range(0, N, batch):
        x0 = val_dom[i:i + batch]
        for _ in range(n_t_per_image):
            tot += float(flow_loss_batch(model, x0, device).cpu()); nb += 1
    if was_training:
        model.train()
    return tot / max(1, nb)


@torch.no_grad()
def flow_generate(model, *, domain, n, image_size, ch_dom, n_steps, device):
    """Unconditional few-step generation (Euler, t: 1 -> 0).

    Start x ~ N(0,I) at t=1, integrate dx/dt = v over ``n_steps`` uniform
    Euler steps down to t=0:  ``v = model(x, t); x = x - dt*v; t -= dt``.
    Returns a NORMALISED image (B,1,H,W) in ~[0,1] (IDWT applied in wavelet
    mode). Used only for the training-time comparison figure.
    """
    was_training = model.training
    model.eval()
    if domain == "wavelet":
        x = torch.randn(n, 4, image_size // 2, image_size // 2, device=device)
    else:
        x = torch.randn(n, 1, image_size, image_size, device=device)
    dt = 1.0 / n_steps
    for k in range(n_steps):
        t = 1.0 - k * dt
        t_b = torch.full((n,), t, device=device)
        v = model(x, t_b * 1000.0)
        x = x - dt * v
    img = from_model_domain(x, domain)
    if was_training:
        model.train()
    return img


# ===========================================================================
def main(out_dir: Path, cfg: dict | None = None) -> dict:
    env_path = os.environ.get("FAST_DIFFUSION_CONFIG_PATH")
    if env_path and Path(env_path).exists():
        cfg = {**CONFIG, **json.loads(Path(env_path).read_text()), **(cfg or {})}
    else:
        cfg = {**CONFIG, **(cfg or {})}
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(cfg["seed"])

    # Dataset dispatch (mirrors solver_ddpm). When AGENT4CT_DATASET is set we
    # pull truth from the staged HDF5 and override the geometry + normaliser.
    from ddssl_ldct.staged_dataset import get_dataset_kind, geometry_overrides
    cfg["dataset_kind"] = get_dataset_kind(cfg)
    if cfg["dataset_kind"] != "phantoms":
        overrides = geometry_overrides(cfg["dataset_kind"])
        cfg.update(overrides)
        if cfg.get("fd_out_scale_auto", True):
            cfg["fd_out_scale"] = float(overrides["display_max"])

    domain = cfg["fd_domain"]
    mode = cfg["fd_mode"]
    out_scale = cfg["fd_out_scale"]
    if domain not in ("pixel", "wavelet"):
        raise ValueError(f"unknown fd_domain={domain!r}")
    print(f"[solver] device={device}  domain={domain}  mode={mode}  "
          f"dataset_kind={cfg['dataset_kind']}  out_scale={out_scale}", flush=True)
    print(f"[solver] cfg={json.dumps({k:v for k,v in cfg.items() if k.startswith('fd_')}, default=str)}",
          flush=True)

    # ---- training & val data (BOTH from the train/val split only) ------------
    if mode == "unconstrained":
        train_seed = cfg["seed"] + 100_000
        n_train = cfg["fd_n_train"]
    elif mode == "constrained":
        train_seed = cfg["seed"]
        n_train = cfg["fd_n_train_constrained"]
    else:
        raise ValueError(f"unknown fd_mode={mode!r}")
    print(f"[solver] data regime: {mode}  n_train={n_train}  train_seed={train_seed}",
          flush=True)

    if cfg["dataset_kind"] == "phantoms":
        train_imgs = build_phantoms(cfg["image_size"], n_train, train_seed, device)
        val_imgs   = build_phantoms(cfg["image_size"], cfg["fd_n_val"],
                                     cfg["seed"] + 500_000, device)
    else:
        from ddssl_ldct.staged_dataset import load_val_split
        # BOTH modes pull from the *train* split (no val/test leakage);
        # constrained just uses fewer of those train slices.
        train_imgs, _, _ = load_val_split(cfg["dataset_kind"], "train",
                                           n_train, device=device)
        val_imgs, _, _ = load_val_split(cfg["dataset_kind"], "val",
                                         cfg["fd_n_val"], device=device)

    train_imgs = (train_imgs / out_scale).clamp(0.0, 1.0)
    val_imgs   = (val_imgs   / out_scale).clamp(0.0, 1.0)

    # Precompute the model-domain clean targets once (DWT for wavelet).
    train_dom = to_model_domain(train_imgs, domain)
    val_dom   = to_model_domain(val_imgs,   domain)
    in_ch = out_ch = (4 if domain == "wavelet" else 1)

    # ---- model + EMA + optim -------------------------------------------------
    model = FlowUNet(in_ch=in_ch, out_ch=out_ch, ch=cfg["fd_ch"]).to(device)
    ema = EMA(model, decay=cfg["fd_ema_decay"])
    params_total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[solver] FlowUNet[{domain}] params: {params_total/1e6:.3f} M", flush=True)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["fd_lr"],
                           weight_decay=cfg["fd_weight_decay"])

    # ---- train ---------------------------------------------------------------
    t0 = time.time()
    last_train_loss = None
    best_val = float("inf")
    history = []
    for ep in range(cfg["fd_epochs"]):
        last_train_loss = train_one_epoch(model, ema, train_dom,
                                           cfg["fd_batch"], opt, device)
        if (ep + 1) % max(1, cfg["fd_epochs"] // 10) == 0 or ep == 0:
            # Validate on the EMA weights (what we ship), not the raw weights.
            vloss = eval_flow_loss(ema.shadow, val_dom, cfg["fd_batch"], device)
            history.append({"epoch": ep + 1, "train_loss": last_train_loss,
                            "val_loss": vloss, "elapsed_s": time.time() - t0})
            if vloss < best_val:
                best_val = vloss
            print(f"[train] ep {ep+1}/{cfg['fd_epochs']}  "
                  f"train={last_train_loss:.4f}  val(ema)={vloss:.4f}  "
                  f"best_val={best_val:.4f}  elapsed={time.time()-t0:.1f}s",
                  flush=True)
        if time.time() - t0 > cfg["fd_train_wall_s"]:
            print(f"[train] wall {cfg['fd_train_wall_s']}s hit at ep={ep+1}",
                  flush=True)
            break
    train_time = time.time() - t0

    final_val = eval_flow_loss(ema.shadow, val_dom, cfg["fd_batch"], device,
                               n_t_per_image=16)   # tighter final estimate

    # ---- save ckpt (EMA weights are the shipped model_state) -----------------
    ckpt_path = Path(cfg["fd_ckpt"])
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state": ema.state_dict(),     # EMA weights
        "fd_domain": domain,
        "fd_ch": cfg["fd_ch"],
        "fd_out_scale": out_scale,
        "fd_mode": mode,
        "n_train": n_train,
        "train_seed": train_seed,
        "final_train_loss": last_train_loss,
        "final_val_loss": final_val,
        "best_val_loss": best_val,
        "history": history,
        "config": cfg,
    }, ckpt_path)
    print(f"[solver] saved FastDiff ckpt to {ckpt_path}  "
          f"(domain={domain}  final_val_flow_loss={final_val:.4f})", flush=True)

    # ---- comparison.png: few-step EMA samples vs val truth -------------------
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        n_show = 3
        samples = flow_generate(ema.shadow, domain=domain, n=n_show,
                                image_size=cfg["image_size"], ch_dom=in_ch,
                                n_steps=cfg["fd_gen_steps"], device=device)
        samples = (samples * out_scale).clamp(0.0, out_scale).cpu()
        fig, ax = plt.subplots(2, n_show, figsize=(3 * n_show, 6))
        vmin, vmax = cfg["display_min"], cfg["display_max"]
        for i in range(n_show):
            ax[0, i].imshow(samples[i, 0], cmap="gray", vmin=vmin, vmax=vmax)
            ax[0, i].set_title(f"FastDiff[{domain}] sample" if i == 0 else "")
            ax[0, i].set_axis_off()
            ax[1, i].imshow((val_imgs[i, 0] * out_scale).cpu(), cmap="gray",
                            vmin=vmin, vmax=vmax)
            ax[1, i].set_title("val truth" if i == 0 else ""); ax[1, i].set_axis_off()
        plt.tight_layout(); plt.savefig(out_dir / "comparison.png", dpi=120)
    except Exception as e:
        print(f"[solver] figure failed: {e}", flush=True)

    # ---- result.json (training surrogate) ------------------------------------
    baseline_val = 1.0   # zero-velocity predictor on unit-variance target ~ 1.0
    headroom = max(0.0, 1.0 - final_val / max(baseline_val, 1e-12))
    val_score = -final_val
    result = {
        "val_score": val_score,
        "val_psnr": 0.0,
        "val_ssim": -final_val,        # mirror val_score for dashboard consistency
        "val_rmse": final_val,
        "baseline_psnr": 0.0,
        "baseline_rmse": baseline_val,
        "headroom": headroom,
        "params_M": params_total / 1e6,
        "train_n": n_train, "val_n": cfg["fd_n_val"],
        "train_time_s": train_time,
        "fd_ckpt_path": str(ckpt_path),
        "fd_domain": domain,
        "fd_mode": mode,
        "final_val_flow_loss": final_val,
        "best_val_flow_loss": best_val,
        "history": history,
        "config": cfg,
    }
    (out_dir / "result.json").write_text(json.dumps(result, indent=2))
    print(f"[solver] FastDiff[{domain}/{mode}]: val_flow_loss={final_val:.4f}  "
          f"hr={headroom:.4f}", flush=True)
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("out_dir")
    args = p.parse_args(); main(Path(args.out_dir))
