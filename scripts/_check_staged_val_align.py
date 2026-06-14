"""Decisive alignment check: is the STAGED val (val_sino_lowdose.h5 +
val_truth.h5) FBP aligned with truth? Replicates exactly what the solvers do
(load_val_split -> proj.fbp). If SSIM ~0.81 the staged data is fine and the low
solver scores are a training issue; if ~0.31 the staged truth/sino are
misaligned (the all-solvers-0.31 symptom)."""
import os, sys
os.environ.setdefault("AGENT4CT_DATASET", "mayo_ldct_2d")
sys.path.insert(0, "/cluster/maier/Agent4CT")
import numpy as np, torch
from ddssl_ldct.staged_dataset import load_val_split, GEOMETRIES
from ddssl_ldct.geometry import FanBeamGeometry
from ddssl_ldct.pyronn_projector import PyronnFanBeamProjector
from ddssl_ldct.metrics import evaluate_calibrated, ssim

dev = "cuda"
truth, _clean, noisy = load_val_split("mayo_ldct_2d", "val", 214, device=dev)
info = GEOMETRIES["mayo_ldct_2d"]
geom = FanBeamGeometry(image_size=512, pixel_spacing=info.pixel_spacing,
                       n_angles=info.n_angles, n_det=info.n_det,
                       det_spacing=info.det_spacing, sod=info.sod, sdd=info.sdd)
proj = PyronnFanBeamProjector(geom).to(dev)  # env auto-applies trunc + det_offset
print("shapes truth", tuple(truth.shape), "noisy", tuple(noisy.shape), flush=True)

ld = proj.fbp(noisy).clamp(min=0.0)
dr = 0.05
# (a) as-is
r = evaluate_calibrated(ld, truth, baseline=ld, display_min=0.0, display_max=dr, fov=False)
print(f"[A] staged LD-FBP vs truth  SSIM={r['val_ssim']:.4f}  (expect ~0.81 if aligned)", flush=True)
# (b) image flips (orientation test)
for name, t in [("flipud", torch.flip(ld, dims=[-2])),
                ("fliplr", torch.flip(ld, dims=[-1])),
                ("flipud+fliplr", torch.flip(ld, dims=[-2, -1]))]:
    rr = evaluate_calibrated(t, truth, baseline=t, display_min=0.0, display_max=dr, fov=False)
    print(f"[B] {name:14} SSIM={rr['val_ssim']:.4f}", flush=True)
# (c) per-slice spread (mid + a few), to see if it's uniform-low (alignment) vs varied
mid = ld.shape[0] // 2
for i in [5, mid, ld.shape[0] - 6]:
    si = float(ssim(ld[i:i+1].clamp(0), truth[i:i+1], data_range=dr))
    print(f"[C] slice {i:3} per-slice SSIM(raw,uncal)={si:.4f}", flush=True)
# (d) z-shift test: does truth[i] match a SHIFTED ld[i+k]? (staging slot offset)
for k in [-2, -1, 1, 2, 5, -5]:
    a = ld[max(0,k):ld.shape[0]+min(0,k)]
    b = truth[max(0,-k):truth.shape[0]+min(0,-k)]
    rr = evaluate_calibrated(a, b, baseline=a, display_min=0.0, display_max=dr, fov=False)
    print(f"[D] z-shift {k:+d}  SSIM={rr['val_ssim']:.4f}", flush=True)
# (e) BASELINE TRANSFORM: the all-slices baseline script did
#     fliplr(flipud(fbp(flip_u(sino)))) and got 0.8078. Test each piece.
ld_uflip = proj.fbp(torch.flip(noisy, dims=[-1])).clamp(min=0.0)   # sino u-flip only
re_ = evaluate_calibrated(ld_uflip, truth, baseline=ld_uflip, display_min=0.0, display_max=dr, fov=False)
print(f"[E1] sino-u-flip only           SSIM={re_['val_ssim']:.4f}", flush=True)
ld_full = torch.flip(ld_uflip, dims=[-2, -1])                       # + image flipud+fliplr
re2 = evaluate_calibrated(ld_full, truth, baseline=ld_full, display_min=0.0, display_max=dr, fov=False)
print(f"[E2] sino-u-flip + image-flip   SSIM={re2['val_ssim']:.4f}  (baseline script got 0.8078)", flush=True)
ld_imgonly = torch.flip(ld, dims=[-2, -1])
re3 = evaluate_calibrated(ld_imgonly, truth, baseline=ld_imgonly, display_min=0.0, display_max=dr, fov=False)
print(f"[E3] image-flip only            SSIM={re3['val_ssim']:.4f}", flush=True)
