"""SSIM validation script - compare our implementation vs scikit-image."""
from __future__ import annotations
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import torch
import numpy as np
from skimage.metrics import structural_similarity as ski_ssim

from ddssl_ldct.phantoms import random_ellipses_phantom
from ddssl_ldct.geometry import FanBeamGeometry
from ddssl_ldct.pyronn_projector import PyronnFanBeamProjector
from ddssl_ldct.simulate import simulate_low_dose
from ddssl_ldct.metrics import ssim as our_ssim, psnr


def main():
    print("SSIM Validation: Our implementation vs scikit-image")
    print("=" * 60)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Generate a simple phantom
    geom = FanBeamGeometry(
        image_size=512, pixel_spacing=0.7,
        n_angles=128, n_det=736, det_spacing=1.2858,
        sod=595.0, sdd=1085.6,
    )
    
    # Generate one phantom
    ph = random_ellipses_phantom(size=512, n_ellipses=10, seed=42)[None, None].to(device)
    
    # Project and add noise
    proj = PyronnFanBeamProjector(geom).to(device)
    clean = proj.forward_project(ph)
    noisy = simulate_low_dose(clean, i0=1e5, sigma_e=10.0, seed=42)
    
    # Reconstruct
    fbp = proj.fbp(noisy)
    ref = proj.fbp(clean)
    
    # Compute metrics
    data_range = 0.05  # As used in the solvers
    
    # Our SSIM
    our_val = float(our_ssim(fbp, ref, data_range=data_range).cpu())
    
    # Scikit-image SSIM
    fbp_np = fbp[0, 0].cpu().numpy()
    ref_np = ref[0, 0].cpu().numpy()
    ski_val = ski_ssim(fbp_np, ref_np, data_range=data_range)
    
    print(f"\nResults:")
    print(f"  Our SSIM:    {our_val:.6f}")
    print(f"  Skimage SSIM: {ski_val:.6f}")
    print(f"  Difference:  {abs(our_val - ski_val):.6f}")
    
    # Also compute against phantom (truth)
    our_vs_ph = float(our_ssim(fbp, ph, data_range=data_range).cpu())
    ski_vs_ph = ski_ssim(fbp_np, ph[0, 0].cpu().numpy(), data_range=data_range)
    
    print(f"\n  vs Phantom (truth):")
    print(f"    Our SSIM:    {our_vs_ph:.6f}")
    print(f"    Skimage SSIM: {ski_vs_ph:.6f}")
    print(f"    Difference:  {abs(our_vs_ph - ski_vs_ph):.6f}")
    
    # PSNR comparison
    our_psnr = float(psnr(fbp, ref, data_range=data_range).cpu())
    mse = np.mean((fbp_np - ref_np)**2)
    ski_psnr = 10 * np.log10((data_range ** 2) / mse)
    
    print(f"\n  PSNR:")
    print(f"    Our PSNR:    {our_psnr:.2f} dB")
    print(f"    Manual PSNR: {ski_psnr:.2f} dB")
    
    # Check data ranges
    print(f"\n  Data ranges:")
    print(f"    FBP range: [{fbp_np.min():.4f}, {fbp_np.max():.4f}]")
    print(f"    Ref range: [{ref_np.min():.4f}, {ref_np.max():.4f}]")
    print(f"    Phantom:   [{ph[0,0].min():.4f}, {ph[0,0].max():.4f}]")
    print(f"    Using data_range={data_range}")
    
    # Investigate SSIM components
    print(f"\n  SSIM components (our implementation):")
    # Manual check with window
    from ddssl_ldct.metrics import _gaussian_window
    w = _gaussian_window(11, 1.5, fbp.device, fbp.dtype)[None, None]
    pad = 5
    
    mu_x = torch.nn.functional.conv2d(fbp, w, padding=pad)
    mu_y = torch.nn.functional.conv2d(ref, w, padding=pad)
    mu_xy = mu_x * mu_y
    mu_xx = mu_x ** 2
    mu_yy = mu_y ** 2
    sigma_xx = torch.nn.functional.conv2d(fbp * fbp, w, padding=pad) - mu_xx
    sigma_yy = torch.nn.functional.conv2d(ref * ref, w, padding=pad) - mu_yy
    sigma_xy = torch.nn.functional.conv2d(fbp * ref, w, padding=pad) - mu_xy
    
    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2
    
    print(f"    C1 = {C1:.2e}, C2 = {C2:.2e}")
    print(f"    mu_x range: [{mu_x.min():.4f}, {mu_x.max():.4f}]")
    print(f"    sigma_xx range: [{sigma_xx.min():.4f}, {sigma_xx.max():.4f}]")
    print(f"    sigma_xy range: [{sigma_xy.min():.4f}, {sigma_xy.max():.4f}]")
    
    # Check if padding is causing issues
    print(f"\n    Image shape: {fbp.shape}")
    print(f"    mu_x shape (after conv): {mu_x.shape}")
    

if __name__ == "__main__":
    main()
