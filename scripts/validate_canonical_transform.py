"""Find the canonical-frame transform: roll + flips applied to the raw v3
per-patient sino so that a UNIFORM-geometry FBP (angle_start=0) reproduces the
validated baseline recon (~0.81 LD-FBP SSIM vs truth) on L277.

Replicates the baseline's per-patient z-map + slab + u-flip, then re-indexes the
views by r=round(-angle_start_corrected/Δ) so angle_start=0 is physically
correct. Sweeps a few roll/flip variants to confirm which hits ~0.81; that
transform is what stage_mayo_canonical will bake in.
"""
import os, sys, json, math
os.environ.setdefault("AGENT4CT_DATASET", "mayo_ldct_2d")
sys.path.insert(0, "/cluster/maier/Agent4CT")
import numpy as np, torch
from ddssl_ldct.geometry import FanBeamGeometry
from ddssl_ldct.pyronn_projector import PyronnFanBeamProjector
from ddssl_ldct.metrics import evaluate_calibrated
from scripts.compare_hd_ld_fbp_allslices import enumerate_truth, _load_mu

PAT = "L277"
ROOT = "/cluster/maier/Agent4CT/data/mayo_ldct"
SUB = "staged_helix2fan_v3"
dev = "cuda"
dr = 0.05
slab_half = 2

gj = json.load(open(f"{ROOT}/{SUB}/{PAT}_sino_lowdose_geometry.json"))
a0 = float(gj["angle_start_corrected"]); rot = int(gj["rotview"]); nu = int(gj["nu"])
delta = 2 * math.pi / rot
r_analytic = int(round(-a0 / delta)) % rot
truth_ps = None
import h5py
with h5py.File(f"{ROOT}/{SUB}/{PAT}_sino_lowdose.h5", "r") as f:
    sino = np.asarray(f["sino"][...], dtype=np.float32)         # (rot, nu, nz)
zgrid = np.load(f"{ROOT}/{SUB}/{PAT}_sino_lowdose_z_grid.npy")
tfiles, tmeta = enumerate_truth(__import__("pathlib").Path(f"{ROOT}/raw/{PAT}"))
truth_ps = tmeta["pixel_spacing"]
ps_eff = 0.700857 * (truth_ps / 0.703125)
print(f"{PAT}: a0={a0:.4f} rot={rot} delta={delta:.5f} r_analytic={r_analytic} "
      f"truth_ps={truth_ps} ps_eff={ps_eff:.4f} n_truth={len(tfiles)}", flush=True)

# sample 30 truth slices for speed
idxs = list(range(0, len(tfiles), max(1, len(tfiles) // 30)))
zmin, zmax = float(zgrid.min()), float(zgrid.max())

def build(uflip, roll_r, ps, tflip, angle_start, imgflip_post):
    geom = FanBeamGeometry(image_size=512, pixel_spacing=ps, n_angles=rot, n_det=nu,
                           det_spacing=1.285044, sod=595.362, sdd=1086.803,
                           angle_start=angle_start, angle_end=angle_start + 2 * math.pi)
    proj = PyronnFanBeamProjector(geom).to(dev)
    r = roll_r % rot
    preds, truths = [], []
    for si in idxs:
        pz, fp = tfiles[si]
        sz = -pz
        if not (zmin - 5 <= sz <= zmax + 5):
            continue
        j = int(np.argmin(np.abs(zgrid - sz)))
        lo, hi = max(0, j - slab_half), min(sino.shape[2], j + slab_half + 1)
        slab = sino[:, :, lo:hi].mean(axis=2)                  # (rot, nu)
        if uflip:
            slab = np.ascontiguousarray(np.flip(slab, axis=-1))
        if r:
            slab = np.roll(slab, r, axis=0)
        st = torch.from_numpy(np.ascontiguousarray(slab)).to(dev).float()[None, None]
        fbp = proj.fbp(st).clamp(min=0.0)
        if imgflip_post:
            fbp = torch.flip(fbp, dims=[-2, -1])
        tr = _load_mu(fp)
        if tflip:
            tr = np.ascontiguousarray(np.flip(np.flip(tr, 0), 1))
        preds.append(fbp); truths.append(torch.from_numpy(tr).to(dev).float()[None, None])
    P = torch.cat(preds, 0); T = torch.cat(truths, 0)
    return float(evaluate_calibrated(P, T, baseline=P, display_min=0.0,
                                     display_max=dr, fov=False)["val_ssim"])

print("ANCHOR (baseline: angle_start=a_corr, uflip, image-flip post) — MUST be ~0.81:", flush=True)
print(f"  anchor SSIM={build(True, 0, ps_eff, False, a0, True):.4f}", flush=True)
print("CANONICAL roll sweep (angle_start=0, find the roll that == anchor):", flush=True)
for roll_r in [-r_analytic, r_analytic, -r_analytic + rot // 2, r_analytic + rot // 2]:
    for uflip in [True, False]:
        s = build(uflip, roll_r, ps_eff, False, 0.0, True)
        print(f"  roll={roll_r%rot:5d} uflip={uflip} imgflip_post=T  SSIM={s:.4f}", flush=True)
