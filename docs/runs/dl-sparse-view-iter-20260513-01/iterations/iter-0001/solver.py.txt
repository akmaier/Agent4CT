"""DL-Sparse-View CT solver — UNROLLED ITERATIVE RECON branch (Spawn-agent A).

This solver replaces the image_denoiser of the dual-domain pipeline with an
**unrolled iterative reconstruction** module inspired by the top-five teams in
the AAPM DL-Sparse-View challenge (Sidky & Pan 2022, §III):

    1. Robust-and-stable / ItNet:       U-Net + data-consistency layer (LS step)
    2. YM&RH:                           variational network with proximal U-Net
    3. DEEP UL:                         TV-LS first stage + HF U-Net cleanup
    4. deepx:                           scale-attention image-only
    5. HBB / JSR-Net:                   ADMM, image + sino regularisation

Concretely: the image_denoiser receives the half-set FBP image r and runs K
unrolled steps of denoise -> re-project -> data-fidelity gradient correct.
The data target g = R(r) is the projection of the input image (the only
half-set sinogram structurally available inside image_denoiser without
modifying DualDomainPipeline); the iterate is pulled back toward that
projection with a learnable step-size alpha.

This file MUST stay out of the main agent's way — the main agent edits
pentathlon/dl_sparse_view/solver.py; we are pentathlon/dl_sparse_view_iter/.

The harness (DualDomainPipeline, R_full, R_half, FBP, training loop, metrics)
is identical to the main solver — only build_denoisers() differs.
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
from ddssl_ldct.models import SmallUNet, TrainableBilateralFilter2d
from ddssl_ldct.training import DualDomainPipeline, train
from ddssl_ldct.metrics import psnr, ssim


# ----------------------------------------------------------------------- #
#  CONFIG  —  one change per iteration on this branch.
# ----------------------------------------------------------------------- #

CONFIG = {
    # Geometry (fixed — Wagner / Siemens AS @ 128 sparse views).
    "image_size":    512,
    "pixel_spacing": 0.7,
    "n_angles":      128,
    "n_det":         736,
    "det_spacing":   1.2858,
    "sod":           595.0,
    "sdd":           1085.6,

    # Training subset for the 5-min iteration budget.
    "train_n":       400,
    "val_n":         100,

    # Training schedule (matches the main agent's best, iter-16 of
    # dl-sparse-view-20260513-01: lr=1e-4 / wd=1e-4 / epochs=8).
    "epochs":        8,
    "batch_size":    1,
    "lr":            1e-4,
    "optimizer":     "adamw",
    "weight_decay":  1e-4,

    # Model — the unrolled image_denoiser.
    "unet_c":        16,
    "img_denoiser":  "unrolled_lpd",  # unrolled_lpd | unet | unet_plus_bf
    "n_unroll":      2,                # number of unrolled denoise-then-DC steps
    "alpha_init":    0.1,              # initial DC step-size (learnable, log-parametrised)
    "n_bf":          1,                # BF tail count after the unroll

    # Noise simulation — fixed so headroom is comparable.
    "noise_i0":      5e4,
    "noise_sigma_e": 5.0,
    "seed":          42,
}


# ----------------------------------------------------------------------- #
#  Unrolled-iterative image denoiser.
# ----------------------------------------------------------------------- #

class UnrolledLPDImageDenoiser(nn.Module):
    """Learned Primal-Dual-style unrolled iterative refinement, in the image
    domain only. K unrolled steps of:

        z_k     = U(x_{k-1})                          # learned denoiser
        x_k     = z_k - alpha_k * R_half^T (R_half z_k - g)

    where g = R_half(input) is the projection of the half-set FBP input image
    — the only data tensor structurally available inside an image_denoiser
    without changing DualDomainPipeline. This is an LPD-lite: the regulariser
    is a CNN, the data-fidelity step is a single gradient on
    ||R z - g||^2. After K steps a 4-param trainable bilateral filter
    sharpens edges (cheap, +3 params, well-tested on iter-16 of the main
    run).

    Parameters: K * (U-Net params)  +  K alphas  +  3 BF params per BF.
    Shared-weight K=2 with c=16:  ~0.23 M params (same order as iter-16).
    """

    def __init__(self, geometry: FanBeamGeometry,
                 c: int = 16, K: int = 2, alpha_init: float = 0.1,
                 n_bf: int = 1, share_weights: bool = False):
        super().__init__()
        assert K >= 1
        self.K = K
        self.share_weights = share_weights
        # Per-step or shared U-Net regulariser.
        if share_weights:
            self.unet = SmallUNet(c=c, residual=True)
        else:
            self.unets = nn.ModuleList(
                [SmallUNet(c=c, residual=True) for _ in range(K)]
            )
        # Per-step learnable log-alpha (so it stays positive after exp).
        self.log_alpha = nn.Parameter(
            torch.full((K,), math.log(max(alpha_init, 1e-6)))
        )
        # Half-set projector — re-uses the same PYRO-NN geometry as the
        # outer pipeline's R_half (split_angles()[0]).
        self.R_half = PyronnFanBeamProjector(geometry.split_angles()[0])
        # Optional bilateral tail (Wagner 2022).
        self.bfs = nn.ModuleList([
            TrainableBilateralFilter2d(kernel_size=7,
                                       sigma_x=1.5, sigma_y=1.5, sigma_r=0.01)
            for _ in range(n_bf)
        ])

    def _denoiser(self, k: int, x: torch.Tensor) -> torch.Tensor:
        if self.share_weights:
            return self.unet(x)
        return self.unets[k](x)

    def forward(self, x_in: torch.Tensor) -> torch.Tensor:
        # The half-set FBP image is the *data anchor* for our DC term.
        # g acts like the "measurement we trust" inside the image domain.
        # Computed once and held fixed across the K unrolled steps.
        with torch.no_grad():
            g = self.R_half.forward_project(x_in)
        x = x_in
        for k in range(self.K):
            z = self._denoiser(k, x)                              # learned regulariser
            r_z = self.R_half.forward_project(z)                  # forward project
            grad = self.R_half.back_project(r_z - g)              # adjoint of residual
            # Optional simple normalization so alpha is order-1; use mean-abs
            # of x as a scale proxy. (Constant across step; not a Hyper-param.)
            alpha = torch.exp(self.log_alpha[k])
            x = z - alpha * grad
        for bf in self.bfs:
            x = bf(x)
        return x


def build_denoisers(cfg: dict, geometry: FanBeamGeometry
                    ) -> tuple[nn.Module, nn.Module]:
    """Return (proj_denoiser, image_denoiser).

    proj_denoiser stays a SmallUNet (matches main-agent iter-16 best).
    image_denoiser is the unrolled iterative module above.
    """
    c = cfg["unet_c"]
    proj = SmallUNet(c=c, residual=True)
    img_type = cfg.get("img_denoiser", "unrolled_lpd")
    if img_type == "unrolled_lpd":
        img = UnrolledLPDImageDenoiser(
            geometry=geometry,
            c=c,
            K=cfg.get("n_unroll", 2),
            alpha_init=cfg.get("alpha_init", 0.1),
            n_bf=cfg.get("n_bf", 1),
            share_weights=cfg.get("share_weights", False),
        )
    elif img_type == "unet_plus_bf":
        # Fallback parity with main agent for comparison runs.
        from pentathlon.dl_sparse_view.solver import UNetPlusBF
        img = UNetPlusBF(c=c, n_bf=cfg.get("n_bf", 1))
    else:
        img = SmallUNet(c=c, residual=True)
    return proj, img


# ----------------------------------------------------------------------- #
#  Data + training harness — identical to the main solver.
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
    print(f"[solver-iter] device={device}  config={json.dumps(cfg)}", flush=True)
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

    proj_dn, img_dn = build_denoisers(cfg, geom)
    pipe = DualDomainPipeline(
        geometry=geom, proj_denoiser=proj_dn, image_denoiser=img_dn,
    ).to(device)
    params_total = sum(p.numel() for p in pipe.parameters() if p.requires_grad)
    print(f"[solver-iter] params = {params_total/1e6:.3f} M", flush=True)

    opt = make_optimizer(pipe.parameters(), cfg)
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
        # Log the current alpha(s) so we see what the iterative DC step
        # learned to weight.
        try:
            alphas = torch.exp(img_dn.log_alpha).detach().cpu().tolist()
        except AttributeError:
            alphas = []
        print(f"[solver-iter] epoch {ep+1:3d}/{cfg['epochs']}  loss={mean_loss:.5f}  "
              f"alphas={['%.4f' % a for a in alphas]}",
              flush=True)
    train_time = time.time() - t0
    print(f"[solver-iter] training took {train_time:.1f}s", flush=True)

    pipe.eval()
    with torch.no_grad():
        pred = pipe.predict(val_noisy)
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
            ax[i, 2].set_title(f"unrolled-iter  (PSNR={val_psnr:.1f} SSIM={val_ssim:.3f})" if i == 0 else "")
            ax[i, 3].imshow(val_ph[i, 0].cpu(), cmap="gray")
            ax[i, 3].set_title("phantom" if i == 0 else "")
            for a in ax[i]:
                a.set_axis_off()
        plt.tight_layout()
        figpath = out_dir / "comparison.png"
        plt.savefig(figpath, dpi=120)
        print(f"[solver-iter] saved {figpath}", flush=True)
    except Exception as e:
        print(f"[solver-iter] figure failed: {e}", flush=True)

    # Capture learned alphas in result.json for the audit trail.
    try:
        learned_alphas = torch.exp(img_dn.log_alpha).detach().cpu().tolist()
    except AttributeError:
        learned_alphas = []
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
        "learned_alphas": learned_alphas,
        "config":         cfg,
    }
    (out_dir / "result.json").write_text(json.dumps(result, indent=2))
    print(f"[solver-iter] result: val_score={val_score:.4f}  headroom={headroom:.4f} "
          f"PSNR={val_psnr:.2f} SSIM={val_ssim:.4f}  "
          f"alphas={learned_alphas}",
          flush=True)
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("out_dir", help="directory to write result.json + comparison.png")
    args = p.parse_args()
    main(Path(args.out_dir))
