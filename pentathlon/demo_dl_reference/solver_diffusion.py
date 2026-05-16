"""Reference: Two diffusion-prior sparse-view CT solvers (pixel-space DDPM).

Both methods share a small unconditional pixel-space DDPM trained on the
synthetic random-ellipse phantoms. The DDPM is trained ONCE per cluster
job and cached to disk so the 20-iter search reuses it. Sampling-time
hyperparameters (step count, guidance weight, etc.) are what the search
varies.

  * mode='dps'   — Diffusion Posterior Sampling (Chung et al. 2023, ICLR)
                   Reverse-process DC-grad: x_t ← x_t − η · ∇_x ‖A x̂₀(x_t) − y‖²
  * mode='mcg'   — Manifold-Constrained Gradient (Chung et al. 2022, NeurIPS)
                   Reverse-process pseudo-inverse: g_t := ∇_x ‖A† A x̂₀ − A† y‖²
                   then x_t ← x_t − η g_t (a single-step pseudoinverse update;
                   here A† is approximated by FBP).

Both are pixel-space DDIM-sampled. The DM4CT benchmark (Shi 2026)
Table 1 lists these as the canonical DC-grad and pseudoinverse-guided
baselines.

Citations:
  Chung H. et al. "Diffusion Posterior Sampling for General Noisy Inverse
    Problems." ICLR 2023. arXiv:2209.14687
  Chung H. et al. "Improving Diffusion Models for Inverse Problems
    Using Manifold Constraints." NeurIPS 2022. arXiv:2206.00941
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
    "diff_mode": "dps",                # "dps" or "mcg"
    "diff_n_train_phantoms": 3000,     # was 1000; need more to learn ellipse stats
    "diff_train_epochs": 25,           # was 8 — under-trained → hr=0 in v1
    "diff_train_batch": 8,
    "diff_train_lr": 2e-4,
    "diff_n_steps": 1000,              # DDPM training noise schedule length
    "diff_sample_steps": 50,           # DDIM sampling steps
    "diff_eta": 1.0,                   # data-fidelity guidance weight
    "diff_init": "fbp",                # init x_T from FBP or pure noise
    "diff_ch": 32,                     # tiny UNet base channels
    "diff_out_scale": 0.05,            # μ-units scale; DDPM trains in x/scale ∈ [0,1]
    "diff_train_wall_s": 1800,         # 30-min cap on the one-time DDPM pretrain
    # v2 checkpoint (v1 was trained un-normalised and useless).
    "diff_ckpt": "/cluster/maier/Agent4CT/checkpoints/ddpm_ellipses_v2.pt",
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


# ---------------------------------------------------------------------------
class NoiseSchedule:
    def __init__(self, T=1000, device="cpu"):
        self.T = T
        # Cosine schedule (Nichol & Dhariwal 2021).
        s = 0.008
        t = torch.arange(T + 1, device=device, dtype=torch.float32) / T
        ac = torch.cos(((t + s) / (1 + s)) * math.pi / 2) ** 2
        ac = ac / ac[0]
        self.alpha_bar = ac.clamp(min=1e-6, max=0.9999)
        self.alpha = self.alpha_bar[1:] / self.alpha_bar[:-1]
        self.beta = 1 - self.alpha


def train_ddpm(model, sched, train_imgs, epochs, batch, lr, device,
                wall_s=1800):
    """Train ε-prediction DDPM. `train_imgs` must already be normalised to ~[0, 1]."""
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    N = train_imgs.shape[0]; T = sched.T
    t0 = time.time()
    for ep in range(epochs):
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
            if time.time() - t0 > wall_s:
                print(f"[train] {wall_s}s wall hit at ep={ep+1} step={i+batch}",
                      flush=True)
                return
        print(f"[train] ep {ep+1}/{epochs}  loss={running/max(1,nb):.4f}  "
              f"elapsed={time.time()-t0:.1f}s", flush=True)


# ---------------------------------------------------------------------------
def x0_from_eps(xt, eps, ab):
    return (xt - (1 - ab).sqrt() * eps) / ab.sqrt().clamp(min=1e-6)


def ddim_step(xt, eps_pred, sched, t_now, t_next):
    """One DDIM (η=0) step from time t_now to t_next."""
    ab_now = sched.alpha_bar[t_now]
    ab_next = sched.alpha_bar[t_next] if t_next > 0 else torch.tensor(1.0,
                                                                       device=xt.device)
    x0_hat = x0_from_eps(xt, eps_pred, ab_now)
    return ab_next.sqrt() * x0_hat + (1 - ab_next).sqrt() * eps_pred


def sample_guided(model, sched, proj, sino, fbp_init, mode, n_steps,
                  eta, init_kind, device, out_scale=0.05):
    """One DDIM sample with DC guidance — operates in the normalised x/out_scale
    space the DDPM was trained on, scaling x̂₀ back to μ-units before each
    projector call so the sinogram residual is in the right units.

    sino, fbp_init: in μ-units (mm⁻¹).  Returns μ-units image.
    """
    T = sched.T
    times = torch.linspace(T, 1, n_steps + 1).long().tolist()
    # Work in normalised space.
    fbp_norm = (fbp_init / out_scale).clamp(0.0, 1.0)
    if init_kind == "fbp":
        ab_T = sched.alpha_bar[T]
        x = ab_T.sqrt() * fbp_norm + (1 - ab_T).sqrt() * torch.randn_like(fbp_norm)
    else:
        x = torch.randn_like(fbp_norm)

    for k in range(n_steps):
        t_now = times[k]; t_next = times[k + 1] if k + 1 < len(times) else 0
        t_tensor = torch.tensor([t_now], device=device)
        x_req = x.detach().requires_grad_(True)
        with torch.enable_grad():
            eps = model(x_req, t_tensor)
            ab_now = sched.alpha_bar[t_now]
            x0_hat = x0_from_eps(x_req, eps, ab_now)           # normalised x̂₀
            # Scale back to μ-units for the projector; no hard clamp (kills grad).
            x0_mu = x0_hat * out_scale
            sino_pred = proj.forward_project(x0_mu)
            if mode == "dps":
                loss = ((sino_pred - sino) ** 2).mean()
                grad = torch.autograd.grad(loss, x_req)[0]
            elif mode == "mcg":
                # ‖A†(A x̂₀ − y)‖² — pseudo-inverse approx A† via FBP.
                resid = sino_pred - sino
                back = proj.fbp(resid)
                loss = (back ** 2).mean()
                grad = torch.autograd.grad(loss, x_req)[0]
            else:
                raise ValueError(mode)
        with torch.no_grad():
            x_clean = ddim_step(x.detach(), eps.detach(), sched, t_now, t_next)
            x = x_clean - eta * grad.detach()
    # Final denorm + clip (only at the very end, after grad work is done).
    return (x.detach() * out_scale).clamp(0.0, out_scale)


# ---------------------------------------------------------------------------
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


def main(out_dir: Path, cfg: dict | None = None) -> dict:
    env_path = os.environ.get("DIFFUSION_CONFIG_PATH")
    if env_path and Path(env_path).exists():
        cfg = {**CONFIG, **json.loads(Path(env_path).read_text()), **(cfg or {})}
    else:
        cfg = {**CONFIG, **(cfg or {})}
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(cfg["seed"])
    print(f"[solver] device={device}  mode={cfg['diff_mode']}", flush=True)
    print(f"[solver] cfg={json.dumps({k:v for k,v in cfg.items() if k.startswith('diff_')}, default=str)}",
          flush=True)

    geom = FanBeamGeometry(
        image_size=cfg["image_size"], pixel_spacing=cfg["pixel_spacing"],
        n_angles=cfg["n_angles"], n_det=cfg["n_det"],
        det_spacing=cfg["det_spacing"], sod=cfg["sod"], sdd=cfg["sdd"])
    proj = PyronnFanBeamProjector(geom).to(device)

    # 1. Train DDPM once (or load cached).
    sched = NoiseSchedule(T=cfg["diff_n_steps"], device=device)
    model = SmallDDPM(ch=cfg["diff_ch"]).to(device)
    ckpt = Path(cfg["diff_ckpt"])
    if ckpt.exists():
        state = torch.load(ckpt, map_location=device)
        if state.get("ch") == cfg["diff_ch"]:
            model.load_state_dict(state["model"])
            print(f"[solver] Loaded cached DDPM from {ckpt}", flush=True)
        else:
            print(f"[solver] Cached ch mismatch; retraining", flush=True)
            ckpt.unlink()
    if not ckpt.exists():
        print(f"[solver] Training DDPM ({cfg['diff_n_train_phantoms']} phantoms, "
              f"{cfg['diff_train_epochs']} ep, wall {cfg['diff_train_wall_s']}s)…",
              flush=True)
        train_phs, _, _ = build_dataset(geom, cfg["diff_n_train_phantoms"],
                                        cfg["seed"] + 100_000,
                                        cfg["noise_i0"], cfg["noise_sigma_e"],
                                        device)
        # Normalise into [0, 1] so DDPM noise injection has the right SNR.
        train_phs = (train_phs / cfg["diff_out_scale"]).clamp(0.0, 1.0)
        t0 = time.time()
        train_ddpm(model, sched, train_phs, cfg["diff_train_epochs"],
                   cfg["diff_train_batch"], cfg["diff_train_lr"], device,
                   wall_s=cfg["diff_train_wall_s"])
        print(f"[solver] DDPM train time {time.time()-t0:.1f}s", flush=True)
        ckpt.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model": model.state_dict(), "ch": cfg["diff_ch"]}, ckpt)
    model.eval()

    # 2. Build val set.
    val_phs, _, val_noisy = build_dataset(geom, cfg["val_n"],
                                          cfg["seed"] + 1000,
                                          cfg["noise_i0"],
                                          cfg["noise_sigma_e"], device)
    with torch.no_grad():
        val_fbp = torch.clamp(proj.fbp(val_noisy), min=0.0)

    # 3. Guided sample each val scene.
    t0 = time.time()
    preds = []
    for i in range(cfg["val_n"]):
        pred_i = sample_guided(model, sched, proj,
                                val_noisy[i:i + 1], val_fbp[i:i + 1],
                                cfg["diff_mode"], cfg["diff_sample_steps"],
                                cfg["diff_eta"], cfg["diff_init"], device,
                                out_scale=cfg["diff_out_scale"])
        preds.append(pred_i)
        if (i + 1) % 5 == 0:
            print(f"[sample] {i+1}/{cfg['val_n']}  elapsed={time.time()-t0:.1f}s",
                  flush=True)
        if time.time() - t0 > 600:
            print(f"[sample] 10-min wall at {i+1}", flush=True); break
    sample_time = time.time() - t0
    pred = torch.cat(preds, 0)
    val_ph = val_phs[:pred.shape[0]]; val_fbp = val_fbp[:pred.shape[0]]

    dr = cfg["display_max"] - cfg["display_min"]
    val_psnr = float(psnr(pred, val_ph, data_range=dr).cpu())
    val_ssim = float(ssim(pred, val_ph, data_range=dr).cpu())
    val_rmse = float(((pred - val_ph) ** 2).mean().sqrt().cpu())
    baseline_psnr = float(psnr(val_fbp, val_ph, data_range=dr).cpu())
    baseline_rmse = float(((val_fbp - val_ph) ** 2).mean().sqrt().cpu())
    headroom = max(0.0, 1.0 - val_rmse / max(baseline_rmse, 1e-12))
    print(f"[solver] {cfg['diff_mode'].upper()}: hr={headroom:.4f} "
          f"SSIM={val_ssim:.4f} PSNR={val_psnr:.2f}", flush=True)

    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        n_show = min(3, pred.shape[0])
        fig, ax = plt.subplots(n_show, 4, figsize=(12, 3 * n_show))
        if n_show == 1: ax = ax[None]
        vmin, vmax = cfg["display_min"], cfg["display_max"]
        title = f"DPS" if cfg["diff_mode"] == "dps" else "MCG"
        for i in range(n_show):
            ax[i, 0].imshow(val_ph[i, 0].cpu(), cmap="gray", vmin=vmin, vmax=vmax)
            ax[i, 0].set_title("truth" if i == 0 else "")
            ax[i, 1].imshow(val_fbp[i, 0].cpu(), cmap="gray", vmin=vmin, vmax=vmax)
            ax[i, 1].set_title(f"FBP (PSNR={baseline_psnr:.1f})" if i == 0 else "")
            ax[i, 2].imshow(pred[i, 0].cpu(), cmap="gray", vmin=vmin, vmax=vmax)
            ax[i, 2].set_title(f"{title} (PSNR={val_psnr:.1f} SSIM={val_ssim:.3f})"
                               if i == 0 else "")
            ax[i, 3].imshow((pred[i, 0] - val_ph[i, 0]).cpu(),
                            cmap="RdBu_r", vmin=-0.01, vmax=0.01)
            ax[i, 3].set_title("residual" if i == 0 else "")
            for a in ax[i]: a.set_axis_off()
        plt.tight_layout(); plt.savefig(out_dir / "comparison.png", dpi=120)
    except Exception as e:
        print(f"[solver] figure failed: {e}", flush=True)

    params_total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    result = {
        "val_score": val_ssim, "val_psnr": val_psnr, "val_ssim": val_ssim,
        "val_rmse": val_rmse, "baseline_psnr": baseline_psnr,
        "baseline_rmse": baseline_rmse, "headroom": headroom,
        "params_M": params_total / 1e6,
        "train_n": cfg["diff_n_train_phantoms"], "val_n": pred.shape[0],
        "train_time_s": sample_time, "config": cfg,
    }
    (out_dir / "result.json").write_text(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("out_dir")
    args = p.parse_args(); main(Path(args.out_dir))
