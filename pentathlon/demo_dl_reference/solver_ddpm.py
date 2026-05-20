"""Reference: Unconditional pixel-space DDPM trained on random-ellipse phantoms.

Two data regimes, selected via ``ddpm_mode``:

  * ``"unconstrained"`` — reproducible random-ellipse generator gives the
    DDPM unlimited training samples (different seed per training image,
    ``ddpm_n_train`` total). This is the "best the prior can be" upper bound.
  * ``"constrained"`` — DDPM trains only on the SAME ``train_n`` phantoms
    that the other dl_reference solvers (ItNet, Dual-Domain, U-Swin, …)
    see at training time. Apples-to-apples comparison of "what can a
    diffusion prior add when it has no extra data?"

Both modes write a checkpoint at ``ddpm_ckpt`` containing model weights
+ noise schedule + the full config. ``solver_diffusion_recon.py`` then
loads one of these and does posterior sampling under various data
consistency strategies.

Optimised by the autoresearch agent against the held-out epsilon
prediction loss on a fixed 100-phantom validation set:
  val_score = -val_eps_loss
  headroom  = max(0, 1 - val_eps_loss / baseline_eps_loss)
where baseline_eps_loss is the loss of a zero predictor (≈ 1.0 for
unit-variance noise).

Citation backbone: Ho et al. 2020 (DDPM, arXiv:2006.11239) cosine
schedule from Nichol & Dhariwal 2021 (arXiv:2102.09672), tiny 3-level
U-Net with time-embedding similar to OpenAI guided-diffusion.
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
from ddssl_ldct.metrics import psnr, ssim


CONFIG = {
    "image_size": 512, "pixel_spacing": 0.7,
    "n_angles": 128, "n_det": 736, "det_spacing": 1.2858,
    "sod": 595.0, "sdd": 1085.6,
    "noise_i0": 1e5, "noise_sigma_e": 10.0, "seed": 42,
    "display_min": 0.0, "display_max": 0.05,
    # Data regime
    "ddpm_mode":              "unconstrained",     # or "constrained"
    "ddpm_n_train":           3000,                # used only when unconstrained
    "ddpm_n_train_constrained": 200,               # matches other dl_ref solvers' train_n
    "ddpm_n_val":             100,                 # held-out eps-loss val set
    "ddpm_out_scale":         0.05,                # normalise μ → [0,1] for DDPM
    # Architecture (kept fixed during hyperparam search; varied if you like)
    "ddpm_ch":                32,                  # base channels of the 3-level UNet
    "ddpm_n_steps":           1000,                # noise schedule length T
    # Optimiser
    "ddpm_epochs":            30,
    "ddpm_batch":             8,
    "ddpm_lr":                2e-4,
    "ddpm_weight_decay":      0.0,
    # Wall clamps (s)
    "ddpm_train_wall_s":      3600,                # 1 hour cap for one training
    # Where to save the checkpoint. Overrideable via env to write per-mode
    # files (ddpm_unconstrained_final.pt vs ddpm_constrained_final.pt).
    "ddpm_ckpt":              "/cluster/maier/Agent4CT/checkpoints/ddpm_search.pt",
    # Optional: hold a small per-search ckpt path so different search iters
    # don't clobber each other's caches (set by the search agent).
    "ddpm_keep_search_ckpts": False,
}


# ---------------------------------------------------------------------------
def _g(c, target=8):
    for g in range(min(c, target), 0, -1):
        if c % g == 0:
            return g
    return 1


class TimeEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half = self.dim // 2
        f = math.log(10000) / (half - 1)
        f = torch.exp(torch.arange(half, device=t.device) * -f)
        e = t[:, None] * f[None, :]
        return torch.cat([e.sin(), e.cos()], -1)


class ResBlk(nn.Module):
    def __init__(self, ci, co, te_dim):
        super().__init__()
        self.norm1 = nn.GroupNorm(_g(ci), ci)
        self.conv1 = nn.Conv2d(ci, co, 3, padding=1)
        self.te = nn.Linear(te_dim, co)
        self.norm2 = nn.GroupNorm(_g(co), co)
        self.conv2 = nn.Conv2d(co, co, 3, padding=1)
        self.skip = nn.Conv2d(ci, co, 1) if ci != co else nn.Identity()

    def forward(self, x, te):
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.te(F.silu(te))[..., None, None]
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)


class SmallDDPM(nn.Module):
    """Tiny 3-level U-Net with time-embedding, for unconditional ε-pred."""
    def __init__(self, ch=32):
        super().__init__()
        te_dim = ch * 4
        self.ch = ch
        self.t_emb = nn.Sequential(TimeEmb(ch), nn.Linear(ch, te_dim),
                                   nn.SiLU(), nn.Linear(te_dim, te_dim))
        self.enc1 = ResBlk(1, ch, te_dim)
        self.enc2 = ResBlk(ch, ch * 2, te_dim)
        self.enc3 = ResBlk(ch * 2, ch * 4, te_dim)
        self.bot = ResBlk(ch * 4, ch * 4, te_dim)
        self.dec3 = ResBlk(ch * 8, ch * 2, te_dim)
        self.dec2 = ResBlk(ch * 4, ch, te_dim)
        self.dec1 = ResBlk(ch * 2, ch, te_dim)
        self.head = nn.Conv2d(ch, 1, 1)
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


class NoiseSchedule:
    """Cosine α-bar schedule (Nichol & Dhariwal 2021)."""
    def __init__(self, T=1000, device="cpu"):
        self.T = T
        s = 0.008
        t = torch.arange(T + 1, device=device, dtype=torch.float32) / T
        ac = torch.cos(((t + s) / (1 + s)) * math.pi / 2) ** 2
        ac = ac / ac[0]
        self.alpha_bar = ac.clamp(min=1e-6, max=0.9999)
        self.alpha = self.alpha_bar[1:] / self.alpha_bar[:-1]
        self.beta = 1 - self.alpha


# ---------------------------------------------------------------------------
def build_phantoms(image_size, n, seed, device):
    """Reproducible per-image-seeded phantom batch in μ-units."""
    return torch.stack([
        random_ellipses_phantom(size=image_size, n_ellipses=10, seed=seed + i)[0]
        for i in range(n)
    ]).to(device)


def train_one_epoch(model, sched, train_imgs, batch, opt, device):
    """One epoch of ε-prediction training. Returns mean loss."""
    N = train_imgs.shape[0]; T = sched.T
    perm = torch.randperm(N)
    running = 0.0; nb = 0
    for i in range(0, N, batch):
        idx = perm[i:i + batch]
        x0 = train_imgs[idx]
        t = torch.randint(1, T + 1, (x0.shape[0],), device=device)
        noise = torch.randn_like(x0)
        ab = sched.alpha_bar[t].view(-1, 1, 1, 1)
        xt = ab.sqrt() * x0 + (1 - ab).sqrt() * noise
        pred = model(xt, t)
        loss = F.mse_loss(pred, noise)
        opt.zero_grad(); loss.backward(); opt.step()
        running += float(loss.detach().cpu()); nb += 1
    return running / max(1, nb)


def eval_eps_loss(model, sched, val_imgs, batch, device, n_t_per_image=4):
    """Mean ε-prediction loss on held-out phantoms, averaged over several
    random t per image to reduce variance."""
    model.eval()
    N = val_imgs.shape[0]; T = sched.T
    tot = 0.0; nb = 0
    with torch.no_grad():
        for i in range(0, N, batch):
            x0 = val_imgs[i:i + batch]
            for _ in range(n_t_per_image):
                t = torch.randint(1, T + 1, (x0.shape[0],), device=device)
                noise = torch.randn_like(x0)
                ab = sched.alpha_bar[t].view(-1, 1, 1, 1)
                xt = ab.sqrt() * x0 + (1 - ab).sqrt() * noise
                pred = model(xt, t)
                tot += float(F.mse_loss(pred, noise).cpu())
                nb += 1
    model.train()
    return tot / max(1, nb)


def main(out_dir: Path, cfg: dict | None = None) -> dict:
    env_path = os.environ.get("DDPM_CONFIG_PATH")
    if env_path and Path(env_path).exists():
        cfg = {**CONFIG, **json.loads(Path(env_path).read_text()), **(cfg or {})}
    else:
        cfg = {**CONFIG, **(cfg or {})}
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(cfg["seed"])

    # Dataset dispatch (Track B/C of workplan): when AGENT4CT_DATASET is set
    # (or cfg["dataset_kind"] != "phantoms"), pull training/val truth images
    # from the staged HDF5 instead of synthesizing ellipse phantoms. The
    # image geometry (size, μ range / out_scale) gets overridden so the
    # checkpoint we save matches the real dataset's distribution.
    from ddssl_ldct.staged_dataset import (
        get_dataset_kind, geometry_overrides, GEOMETRIES,
    )
    cfg["dataset_kind"] = get_dataset_kind(cfg)
    if cfg["dataset_kind"] != "phantoms":
        overrides = geometry_overrides(cfg["dataset_kind"])
        cfg.update(overrides)
        # Use the dataset's display_max as the [0,1] normaliser if not
        # explicitly overridden; this matches what the matching
        # diffusion-recon solver does at inference time.
        if cfg.get("ddpm_out_scale_auto", True):
            cfg["ddpm_out_scale"] = float(overrides["display_max"])

    mode = cfg["ddpm_mode"]
    out_scale = cfg["ddpm_out_scale"]
    print(f"[solver] device={device}  mode={mode}  "
          f"dataset_kind={cfg['dataset_kind']}  out_scale={out_scale}",
          flush=True)
    print(f"[solver] cfg={json.dumps({k:v for k,v in cfg.items() if k.startswith('ddpm_')}, default=str)}",
          flush=True)

    # ---- training & val data --------------------------------------------------
    if mode == "unconstrained":
        # Different seed range, lots of samples.
        train_seed = cfg["seed"] + 100_000
        n_train = cfg["ddpm_n_train"]
    elif mode == "constrained":
        # SAME seeds as the other dl_reference solvers (which start at cfg["seed"]).
        train_seed = cfg["seed"]
        n_train = cfg["ddpm_n_train_constrained"]
    else:
        raise ValueError(f"unknown ddpm_mode={mode!r}")
    print(f"[solver] data regime: {mode}  n_train={n_train}  train_seed={train_seed}",
          flush=True)

    if cfg["dataset_kind"] == "phantoms":
        train_imgs = build_phantoms(cfg["image_size"], n_train, train_seed, device)
        val_imgs   = build_phantoms(cfg["image_size"], cfg["ddpm_n_val"],
                                     cfg["seed"] + 500_000, device)
    else:
        # Load real truth images from staged HDF5. Train/val splits come
        # from disk; the seed-offset convention used by the phantoms path
        # doesn't apply (the dataset's split is what defines train vs val).
        from ddssl_ldct.staged_dataset import load_val_split
        # load_val_split returns (truth, clean, noisy) as (N, 1, H, W);
        # build_phantoms also produces (N, 1, H, W), so just pass through.
        train_imgs, _, _ = load_val_split(cfg["dataset_kind"], "train",
                                            n_train, device=device)
        val_imgs, _, _ = load_val_split(cfg["dataset_kind"], "val",
                                          cfg["ddpm_n_val"], device=device)

    train_imgs = (train_imgs / out_scale).clamp(0.0, 1.0)
    val_imgs   = (val_imgs   / out_scale).clamp(0.0, 1.0)

    # ---- model + optim --------------------------------------------------------
    sched = NoiseSchedule(T=cfg["ddpm_n_steps"], device=device)
    model = SmallDDPM(ch=cfg["ddpm_ch"]).to(device)
    params_total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[solver] DDPM params: {params_total/1e6:.3f} M", flush=True)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["ddpm_lr"],
                            weight_decay=cfg["ddpm_weight_decay"])

    # ---- train ---------------------------------------------------------------
    t0 = time.time()
    last_train_loss = None
    best_val = float("inf")
    history = []
    for ep in range(cfg["ddpm_epochs"]):
        last_train_loss = train_one_epoch(model, sched, train_imgs,
                                           cfg["ddpm_batch"], opt, device)
        if (ep + 1) % max(1, cfg["ddpm_epochs"] // 10) == 0 or ep == 0:
            vloss = eval_eps_loss(model, sched, val_imgs, cfg["ddpm_batch"], device)
            history.append({"epoch": ep + 1, "train_loss": last_train_loss,
                            "val_eps_loss": vloss,
                            "elapsed_s": time.time() - t0})
            if vloss < best_val:
                best_val = vloss
            print(f"[train] ep {ep+1}/{cfg['ddpm_epochs']}  "
                  f"train={last_train_loss:.4f}  val={vloss:.4f}  "
                  f"best_val={best_val:.4f}  elapsed={time.time()-t0:.1f}s",
                  flush=True)
        if time.time() - t0 > cfg["ddpm_train_wall_s"]:
            print(f"[train] wall {cfg['ddpm_train_wall_s']}s hit at ep={ep+1}",
                  flush=True)
            break
    train_time = time.time() - t0

    final_val = eval_eps_loss(model, sched, val_imgs, cfg["ddpm_batch"], device,
                               n_t_per_image=16)   # tighter final estimate

    # Save weights to the configured path (so the recon solver can find them).
    ckpt_path = Path(cfg["ddpm_ckpt"])
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state": model.state_dict(),
        "ddpm_ch": cfg["ddpm_ch"],
        "ddpm_n_steps": cfg["ddpm_n_steps"],
        "ddpm_out_scale": out_scale,
        "ddpm_mode": mode,
        "train_seed": train_seed,
        "n_train": n_train,
        "final_train_loss": last_train_loss,
        "final_val_loss": final_val,
        "best_val_loss": best_val,
        "history": history,
        "config": cfg,
    }, ckpt_path)
    print(f"[solver] saved DDPM ckpt to {ckpt_path}  "
          f"(final_val_eps_loss={final_val:.4f})", flush=True)

    # ---- sample a few images for the comparison.png --------------------------
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        n_show = 3
        model.eval()
        with torch.no_grad():
            x = torch.randn(n_show, 1, cfg["image_size"], cfg["image_size"], device=device)
            T = sched.T
            # Crude DDIM η=0 ancestral sampling, 50 steps; just for visualisation.
            times = torch.linspace(T, 1, 51).long().tolist()
            for k in range(len(times) - 1):
                t_now = times[k]; t_next = times[k + 1]
                t_tensor = torch.tensor([t_now] * n_show, device=device)
                eps = model(x, t_tensor)
                ab_now = sched.alpha_bar[t_now]
                ab_next = sched.alpha_bar[t_next] if t_next > 0 else torch.tensor(1.0,
                                                                                   device=device)
                x0_hat = (x - (1 - ab_now).sqrt() * eps) / ab_now.sqrt().clamp(min=1e-6)
                x = ab_next.sqrt() * x0_hat + (1 - ab_next).sqrt() * eps
        samples = (x * out_scale).clamp(0.0, out_scale).cpu()
        fig, ax = plt.subplots(2, n_show, figsize=(3 * n_show, 6))
        vmin, vmax = cfg["display_min"], cfg["display_max"]
        for i in range(n_show):
            ax[0, i].imshow(samples[i, 0], cmap="gray", vmin=vmin, vmax=vmax)
            ax[0, i].set_title("DDPM sample" if i == 0 else ""); ax[0, i].set_axis_off()
            ax[1, i].imshow((val_imgs[i, 0] * out_scale).cpu(), cmap="gray", vmin=vmin, vmax=vmax)
            ax[1, i].set_title("val phantom" if i == 0 else ""); ax[1, i].set_axis_off()
        plt.tight_layout(); plt.savefig(out_dir / "comparison.png", dpi=120)
    except Exception as e:
        print(f"[solver] figure failed: {e}", flush=True)

    # ---- write result.json ---------------------------------------------------
    # autoresearch surrogate: lower val_eps_loss is better.
    # Baseline = loss of a zero predictor ≈ 1.0 (for standard normal noise).
    baseline_val = 1.0
    headroom = max(0.0, 1.0 - final_val / max(baseline_val, 1e-12))
    val_score = -final_val   # search agent maximises val_score
    result = {
        "val_score": val_score,
        "val_psnr": 0.0,            # unused for DDPM training
        "val_ssim": -final_val,     # mirror val_score for dashboard consistency
        "val_rmse": final_val,
        "baseline_psnr": 0.0,
        "baseline_rmse": baseline_val,
        "headroom": headroom,
        "params_M": params_total / 1e6,
        "train_n": n_train, "val_n": cfg["ddpm_n_val"],
        "train_time_s": train_time,
        "ddpm_ckpt_path": str(ckpt_path),
        "ddpm_mode": mode,
        "final_val_eps_loss": final_val,
        "best_val_eps_loss": best_val,
        "history": history,
        "config": cfg,
    }
    (out_dir / "result.json").write_text(json.dumps(result, indent=2))
    print(f"[solver] DDPM-{mode}: val_eps_loss={final_val:.4f}  hr={headroom:.4f}",
          flush=True)
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("out_dir")
    args = p.parse_args(); main(Path(args.out_dir))
