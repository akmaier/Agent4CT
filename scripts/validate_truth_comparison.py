"""Verify that demo solvers should compare against ground truth phantom, not FBP reference."""
from __future__ import annotations
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import torch
from ddssl_ldct.geometry import FanBeamGeometry
from ddssl_ldct.pyronn_projector import PyronnFanBeamProjector
from ddssl_ldct.phantoms import random_ellipses_phantom
from ddssl_ldct.simulate import simulate_low_dose
from ddssl_ldct.metrics import ssim, psnr


def main():
    device = "cuda"
    geom = FanBeamGeometry(512, 0.7, 128, 736, 1.2858, 595.0, 1085.6)
    proj = PyronnFanBeamProjector(geom).to(device)

    # Generate 5 phantoms
    ph = torch.stack([random_ellipses_phantom(512, 10, 42+i)[0] for i in range(5)]).to(device)
    clean = proj.forward_project(ph)
    noisy = simulate_low_dose(clean, 1e5, 10.0, 42)

    ref = proj.fbp(clean)  # noiseless FBP (this is what solvers use as "reference")
    fbp = proj.fbp(noisy)  # noisy FBP (baseline)

    data_range = 0.05

    print("=" * 70)
    print("SSIM COMPUTED AGAINST DIFFERENT TARGETS")
    print("=" * 70)

    # 1. SSIM of noisy FBP vs noiseless FBP (what solvers currently report)
    ssim_fbp_vs_ref = float(ssim(fbp, ref, data_range).cpu())
    psnr_fbp_vs_ref = float(psnr(fbp, ref, data_range).cpu())
    print(f"\n1. Noisy FBP vs Noiseless FBP (solvers' current 'reference'):")
    print(f"   SSIM: {ssim_fbp_vs_ref:.4f}")
    print(f"   PSNR: {psnr_fbp_vs_ref:.2f} dB")
    print(f"   -> This treats noiseless FBP as the gold standard")

    # 2. SSIM of noisy FBP vs phantom truth (what we SHOULD report)
    ssim_fbp_vs_ph = float(ssim(fbp, ph, data_range).cpu())
    psnr_fbp_vs_ph = float(psnr(fbp, ph, data_range).cpu())
    print(f"\n2. Noisy FBP vs Ground Truth Phantom:")
    print(f"   SSIM: {ssim_fbp_vs_ph:.4f}")
    print(f"   PSNR: {psnr_fbp_vs_ph:.2f} dB")
    print(f"   -> This is the TRUE baseline for the challenge")

    # 3. SSIM of noiseless FBP vs phantom truth (FBP reconstruction error)
    ssim_ref_vs_ph = float(ssim(ref, ph, data_range).cpu())
    psnr_ref_vs_ph = float(psnr(ref, ph, data_range).cpu())
    print(f"\n3. Noiseless FBP vs Ground Truth Phantom:")
    print(f"   SSIM: {ssim_ref_vs_ph:.4f}")
    print(f"   PSNR: {psnr_ref_vs_ph:.2f} dB")
    print(f"   -> Even perfect FBP has limited accuracy!")

    print(f"\n" + "=" * 70)
    print("IMPLICATIONS:")
    print("=" * 70)
    print(f"\nCurrent solver behavior (comparing vs noiseless FBP):")
    print(f"  - Headroom measures improvement over FBP artifacts")
    print(f"  - SSIM is inflated because FBP already has structured error")
    print(f"  - Does NOT measure how close we are to TRUE phantom")
    print(f"\nCorrect behavior (comparing vs ground truth phantom):")
    print(f"  - Headroom measures improvement over raw noisy data")
    print(f"  - Matches the AAPM challenge scoring (RMSE vs truth)")
    print(f"  - Realistic assessment of reconstruction quality")

    # Show phantom images have much higher SSIM with themselves
    print(f"\n" + "=" * 70)
    print("DATA RANGE ISSUE:")
    print("=" * 70)
    print(f"\nPhantom range: [{float(ph.min()):.4f}, {float(ph.max()):.4f}]")
    print(f"Noiseless FBP range: [{float(ref.min()):.4f}, {float(ref.max()):.4f}]")
    print(f"Noisy FBP range: [{float(fbp.min()):.4f}, {float(fbp.max()):.4f}]")
    print(f"Configured data_range: {data_range}")
    print(f"\nThe phantom typically maxes around 0.04-0.05")
    print(f"But FBP reconstructions can overshoot to 0.06-0.07")
    print(f"Using data_range=0.05 clips high values in comparisons")


if __name__ == "__main__":
    main()
