"""Is the staged-pipeline mismatch a per-patient FOV (pixel-spacing) issue?
The validated baseline reconstructed L277 at ps_eff = 0.700857*0.7422/0.703125
= 0.7398 (380 mm FOV); the staged dataset uses a FIXED ps=0.700857 for all
patients. Reconstruct the staged val sino at several ps and score vs truth.
Also dump staged-truth pixel stats to sanity-check it is really L277 mu."""
import os, sys
os.environ.setdefault("AGENT4CT_DATASET", "mayo_ldct_2d")
sys.path.insert(0, "/cluster/maier/Agent4CT")
import numpy as np, torch
from ddssl_ldct.staged_dataset import load_val_split, GEOMETRIES
from ddssl_ldct.geometry import FanBeamGeometry
from ddssl_ldct.pyronn_projector import PyronnFanBeamProjector
from ddssl_ldct.metrics import evaluate_calibrated

dev = "cuda"
truth, _c, noisy = load_val_split("mayo_ldct_2d", "val", 214, device=dev)
info = GEOMETRIES["mayo_ldct_2d"]
print(f"truth stats: min={float(truth.min()):.5f} max={float(truth.max()):.5f} "
      f"mean={float(truth.mean()):.5f}  (L277 mu: air~0.0005, soft tissue~0.02)", flush=True)
dr = 0.05
for ps in [0.700857, 0.7398, 0.6641, 0.7422, 0.760]:
    geom = FanBeamGeometry(image_size=512, pixel_spacing=ps,
                           n_angles=info.n_angles, n_det=info.n_det,
                           det_spacing=info.det_spacing, sod=info.sod, sdd=info.sdd)
    proj = PyronnFanBeamProjector(geom).to(dev)
    ld = proj.fbp(noisy).clamp(min=0.0)
    r = evaluate_calibrated(ld, truth, baseline=ld, display_min=0.0, display_max=dr, fov=False)
    # also the image-flip variant at this ps (orientation x FOV)
    ldf = torch.flip(ld, dims=[-2, -1])
    rf = evaluate_calibrated(ldf, truth, baseline=ldf, display_min=0.0, display_max=dr, fov=False)
    print(f"ps={ps:.4f}  SSIM={r['val_ssim']:.4f}   (image-flip {rf['val_ssim']:.4f})", flush=True)
