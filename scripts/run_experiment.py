"""End-to-end reproduction of arXiv:2211.01111 with PYRO-NN as the recon backbone.

Geometry defaults to Wagner et al.'s rebinned Mayo LDCT setup (Siemens AS):
512×512 image at 0.7 mm, 1152 views over 2π, 736 detector channels at 1.2858 mm,
SOD 595 mm, SDD 1085.6 mm — see ddssl_ldct/geometry.py.

Three configurations are evaluated against the high-dose FBP reference:
  A) low-dose FBP (no denoising) — input baseline
  B) image-only post-processing  — Noise2Inverse on the image domain only
  C) dual-domain (paper)         — Noise2Inverse on projection + image domains

Until real Mayo DICOM-CT-PD data is rebinned via faebstn96/helix2fan, the
runner generates a small set of random-ellipse phantoms as synthetic inputs.
Replace `build_dataset()` with a real-data loader once data is in place.

CUDA required (PYRO-NN). Submit on the LME cluster via
`cluster/slurm/ddssl_smoke.sbatch`.
"""
from __future__ import annotations
import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from ddssl_ldct.geometry import FanBeamGeometry
from ddssl_ldct.pyronn_projector import PyronnFanBeamProjector
from ddssl_ldct.phantoms import random_ellipses_phantom
from ddssl_ldct.simulate import simulate_low_dose, split_projections
from ddssl_ldct.models import SmallUNet
from ddssl_ldct.training import DualDomainPipeline, train
from ddssl_ldct.metrics import psnr, ssim


# ----- data ----------------------------------------------------------------- #

def build_dataset(geom: FanBeamGeometry, n: int, seed: int, i0: float,
                  sigma_e: float, device: str
                  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """(phantoms, clean sino, low-dose sino), all (n,1,…). Synthetic for now."""
    proj = PyronnFanBeamProjector(geom).to(device)
    phantoms = torch.stack([
        random_ellipses_phantom(size=geom.image_size, n_ellipses=18, seed=seed + i)[0]
        for i in range(n)
    ]).to(device)
    with torch.no_grad():
        clean = proj.forward_project(phantoms)
        noisy = simulate_low_dose(clean, i0=i0, sigma_e=sigma_e, seed=seed + 10_000)
    return phantoms, clean, noisy


# ----- image-only baseline -------------------------------------------------- #

class ImageOnlyPipeline(nn.Module):
    """Noise2Inverse with the proj-domain operator collapsed to identity."""

    def __init__(self, geom: FanBeamGeometry, denoiser: nn.Module):
        super().__init__()
        self.geom = geom
        self.denoiser = denoiser
        assert geom.n_angles % 2 == 0
        self.R_full = PyronnFanBeamProjector(geom)
        self.R_half = PyronnFanBeamProjector(geom.split_angles()[0])

    def training_step(self, sino_full):
        x_a, x_b = split_projections(sino_full)
        y_hat_a = self.denoiser(self.R_half.fbp(x_a))
        with torch.no_grad():
            y_tgt_b = self.R_half.fbp(x_b)
        loss_ab = F.mse_loss(y_hat_a, y_tgt_b)
        y_hat_b = self.denoiser(self.R_half.fbp(x_b))
        with torch.no_grad():
            y_tgt_a = self.R_half.fbp(x_a)
        loss_ba = F.mse_loss(y_hat_b, y_tgt_a)
        loss = 0.5 * (loss_ab + loss_ba)
        return {"loss": loss, "loss_ab": loss_ab.detach(), "loss_ba": loss_ba.detach()}

    @torch.no_grad()
    def predict(self, sino_full):
        x_a, x_b = split_projections(sino_full)
        return 0.5 * (
            self.denoiser(self.R_half.fbp(x_a)) +
            self.denoiser(self.R_half.fbp(x_b))
        )


# ----- main ----------------------------------------------------------------- #

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--size", type=int, default=512)
    p.add_argument("--pixel_spacing", type=float, default=0.7)
    p.add_argument("--angles", type=int, default=1152)
    p.add_argument("--dets", type=int, default=736)
    p.add_argument("--det_spacing", type=float, default=1.2858)
    p.add_argument("--sod", type=float, default=595.0)
    p.add_argument("--sdd", type=float, default=1085.6)
    p.add_argument("--train_n", type=int, default=32)
    p.add_argument("--val_n", type=int, default=4)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--i0", type=float, default=1e4, help="incident photons (lower = noisier)")
    p.add_argument("--sigma_e", type=float, default=5.0)
    p.add_argument("--unet_c", type=int, default=32)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out_dir", default="runs/exp_pyronn")
    p.add_argument("--skip_image_only", action="store_true")
    args = p.parse_args()
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    geom = FanBeamGeometry(
        image_size=args.size, pixel_spacing=args.pixel_spacing,
        n_angles=args.angles, n_det=args.dets, det_spacing=args.det_spacing,
        sod=args.sod, sdd=args.sdd,
    )
    print(f"[geometry] {geom}", flush=True)

    # ---- data
    t0 = time.time()
    train_ph, train_clean, train_noisy = build_dataset(
        geom, args.train_n, seed=args.seed, i0=args.i0, sigma_e=args.sigma_e,
        device=args.device)
    val_ph, val_clean, val_noisy = build_dataset(
        geom, args.val_n, seed=args.seed + 1000, i0=args.i0, sigma_e=args.sigma_e,
        device=args.device)
    print(f"[data] train={tuple(train_noisy.shape)}, val={tuple(val_noisy.shape)}, "
          f"{time.time() - t0:.1f}s", flush=True)

    # ---- high-dose reference (clean recon) for metric purposes only
    with torch.no_grad():
        R = PyronnFanBeamProjector(geom).to(args.device)
        val_ref = R.fbp(val_clean)

    # ---- baseline: low-dose FBP, no denoising
    with torch.no_grad():
        ld_fbp = R.fbp(val_noisy)
        p_ld = psnr(ld_fbp, val_ref).item()
        s_ld = ssim(ld_fbp, val_ref).item()
    print(f"[A baseline low-dose FBP]   PSNR={p_ld:.2f} dB   SSIM={s_ld:.3f}",
          flush=True)
    history = {"low_dose_fbp": {"psnr": p_ld, "ssim": s_ld}}

    # ---- image-only post-processing
    if not args.skip_image_only:
        print("\n=== [B] image-only post-processing baseline ===", flush=True)
        torch.manual_seed(args.seed + 100)
        img_only = ImageOnlyPipeline(geom, SmallUNet(c=args.unet_c)).to(args.device)
        train(img_only, train_noisy, epochs=args.epochs, lr=args.lr,
              device=args.device, val_sinos=val_noisy, val_ground_truth=val_ref,
              log_every=max(args.epochs // 10, 1))
        with torch.no_grad():
            pred_b = img_only.predict(val_noisy)
            p_b = psnr(pred_b, val_ref).item()
            s_b = ssim(pred_b, val_ref).item()
        print(f"[B image-only result]      PSNR={p_b:.2f} dB   SSIM={s_b:.3f}",
              flush=True)
        history["image_only"] = {"psnr": p_b, "ssim": s_b}

    # ---- dual-domain (paper)
    print("\n=== [C] dual-domain self-supervised (paper) ===", flush=True)
    torch.manual_seed(args.seed + 200)
    pipe = DualDomainPipeline(
        geometry=geom,
        proj_denoiser=SmallUNet(c=args.unet_c),
        image_denoiser=SmallUNet(c=args.unet_c),
    ).to(args.device)
    train(pipe, train_noisy, epochs=args.epochs, lr=args.lr,
          device=args.device, val_sinos=val_noisy, val_ground_truth=val_ref,
          log_every=max(args.epochs // 10, 1))
    with torch.no_grad():
        pred_c = pipe.predict(val_noisy)
        p_c = psnr(pred_c, val_ref).item()
        s_c = ssim(pred_c, val_ref).item()
    print(f"[C dual-domain result]     PSNR={p_c:.2f} dB   SSIM={s_c:.3f}",
          flush=True)
    history["dual_domain"] = {"psnr": p_c, "ssim": s_c}

    # ---- summary + comparison PNG
    print("\n========== summary ==========")
    for k, v in history.items():
        print(f"  {k:22s} PSNR={v['psnr']:.2f}   SSIM={v['ssim']:.3f}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        n_show = min(3, args.val_n)
        cols = 4 if args.skip_image_only else 5
        fig, ax = plt.subplots(n_show, cols, figsize=(2.4 * cols, 2.4 * n_show))
        if n_show == 1:
            ax = ax[None]
        for i in range(n_show):
            ax[i, 0].imshow(val_ref[i, 0].cpu(), cmap="gray"); ax[i, 0].set_title("reference" if i == 0 else "")
            ax[i, 1].imshow(ld_fbp[i, 0].cpu(), cmap="gray"); ax[i, 1].set_title("low-dose FBP" if i == 0 else "")
            col = 2
            if not args.skip_image_only:
                ax[i, col].imshow(pred_b[i, 0].cpu(), cmap="gray"); ax[i, col].set_title("image-only" if i == 0 else "")
                col += 1
            ax[i, col].imshow(pred_c[i, 0].cpu(), cmap="gray"); ax[i, col].set_title("dual-domain" if i == 0 else "")
            col += 1
            ax[i, col].imshow(val_ph[i, 0].cpu(), cmap="gray"); ax[i, col].set_title("phantom" if i == 0 else "")
            for a in ax[i]:
                a.set_axis_off()
        plt.tight_layout()
        figpath = out / "comparison.png"
        plt.savefig(figpath, dpi=120)
        print(f"saved {figpath}")
    except ImportError:
        pass


if __name__ == "__main__":
    main()
