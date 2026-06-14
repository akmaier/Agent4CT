"""Confirm the rotation root-cause: reconstruct the staged val (L277) sino with
the per-patient angle_start_corrected (and per-patient ps) and check SSIM vs
truth. If ~0.81, the staged uniform-geom pipeline's angle_start=0 (+ fixed ps)
is THE bug."""
import os, sys, json, math
os.environ.setdefault("AGENT4CT_DATASET", "mayo_ldct_2d")
sys.path.insert(0, "/cluster/maier/Agent4CT")
import torch
from ddssl_ldct.staged_dataset import load_val_split, GEOMETRIES
from ddssl_ldct.geometry import FanBeamGeometry
from ddssl_ldct.pyronn_projector import PyronnFanBeamProjector
from ddssl_ldct.metrics import evaluate_calibrated

dev = "cuda"
truth, _c, noisy = load_val_split("mayo_ldct_2d", "val", 214, device=dev)
info = GEOMETRIES["mayo_ldct_2d"]
gj = json.load(open("/cluster/maier/Agent4CT/data/mayo_ldct/staged_helix2fan_v3/L277_sino_lowdose_geometry.json"))
a0 = float(gj["angle_start_corrected"]); rot = int(gj["rotview"]); nu = int(gj["nu"])
print(f"L277 angle_start_corrected={a0:.4f} rotview={rot} nu={nu}", flush=True)
dr = 0.05
for tag, astart, ps in [("staged-default a0=0 ps=.7009", 0.0, 0.700857),
                        ("a_corr ps=.7009", a0, 0.700857),
                        ("a_corr ps=.7398", a0, 0.7398),
                        ("a_corr ps=.7398 +imgflip", a0, 0.7398)]:
    geom = FanBeamGeometry(image_size=512, pixel_spacing=ps, n_angles=rot, n_det=nu,
                           det_spacing=info.det_spacing, sod=info.sod, sdd=info.sdd,
                           angle_start=astart, angle_end=astart + 2 * math.pi)
    proj = PyronnFanBeamProjector(geom).to(dev)
    ld = proj.fbp(noisy).clamp(min=0.0)
    if "imgflip" in tag:
        ld = torch.flip(ld, dims=[-2, -1])
    r = evaluate_calibrated(ld, truth, baseline=ld, display_min=0.0, display_max=dr, fov=False)
    print(f"[{tag}]  SSIM={r['val_ssim']:.4f}", flush=True)
