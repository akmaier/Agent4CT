"""Is the staged val a PERMUTATION mismatch (sino slot i != truth slot i)?
For a few staged-val recon slices, find the best-matching truth slice among
all 214. If argmax index != i but SSIM is high -> ordering/z-mapping bug in
staging (re-stage with correct order fixes it). If even the best match is low
-> deeper (truth values differ from what the sino reconstructs)."""
import os, sys
os.environ.setdefault("AGENT4CT_DATASET", "mayo_ldct_2d")
sys.path.insert(0, "/cluster/maier/Agent4CT")
import numpy as np, torch
from ddssl_ldct.staged_dataset import load_val_split, GEOMETRIES
from ddssl_ldct.geometry import FanBeamGeometry
from ddssl_ldct.pyronn_projector import PyronnFanBeamProjector
from ddssl_ldct.metrics import ssim

dev = "cuda"
truth, _c, noisy = load_val_split("mayo_ldct_2d", "val", 214, device=dev)
info = GEOMETRIES["mayo_ldct_2d"]
geom = FanBeamGeometry(image_size=512, pixel_spacing=info.pixel_spacing,
                       n_angles=info.n_angles, n_det=info.n_det,
                       det_spacing=info.det_spacing, sod=info.sod, sdd=info.sdd)
proj = PyronnFanBeamProjector(geom).to(dev)
ld = proj.fbp(noisy).clamp(min=0.0)
N = ld.shape[0]
dr = 0.05
# normalize each image to its own max for a coarse structural match (cheap)
def best_match(i):
    r = ld[i:i+1]
    best_j, best_s = -1, -1.0
    for j in range(N):
        s = float(ssim(r, truth[j:j+1], data_range=dr))
        if s > best_s:
            best_s, best_j = s, j
    return best_j, best_s
for i in [0, 50, 107, 160, 213]:
    j, s = best_match(i)
    diag = float(ssim(ld[i:i+1], truth[i:i+1], data_range=dr))
    print(f"recon[{i:3}] best-match truth[{j:3}] SSIM={s:.4f} | diagonal truth[{i:3}] SSIM={diag:.4f}", flush=True)
