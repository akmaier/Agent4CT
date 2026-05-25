#!/bin/bash
#SBATCH --job-name=measure-magnification
#SBATCH --partition=main
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --exclude=lme49,lme53,lme171
#SBATCH --mem=24G
#SBATCH --time=00:15:00
#SBATCH --output=/cluster/maier/Agent4CT/results/slurm/%x-%j.out
#SBATCH --error=/cluster/maier/Agent4CT/results/slurm/%x-%j.err

# Compare body extent in 512² FBP recons at pixel_spacing 0.7 vs 0.703125 mm
# vs the truth (Mayo Siemens 0.703125 mm). All three should agree on body
# physical width (in mm); pixel widths will differ by the pixel-spacing
# ratio. Goal: verify the magnification fix doesn't OVERshoot.

set -euo pipefail
REPO=/cluster/maier/Agent4CT
cd "$REPO"
VENV="$REPO/.venv"
export CUDA_HOME="$VENV/cuda-pip-bundle"
export PATH="$CUDA_HOME/bin:$PATH"
source "$VENV/bin/activate"
export PYTHONPATH="$REPO:${PYTHONPATH:-}"

python <<'PYEOF'
import sys, json, math, numpy as np, h5py, torch, pydicom
from pathlib import Path
sys.path.insert(0, '/cluster/maier/Agent4CT')
from ddssl_ldct.geometry import FanBeamGeometry
from ddssl_ldct.pyronn_projector import PyronnFanBeamProjector

sino_dir = Path('/cluster/maier/Agent4CT/data/mayo_ldct/staged_helix2fan')
raw_dir  = Path('/cluster/maier/Agent4CT/data/mayo_ldct/raw/L014')
geom_json = json.loads((sino_dir/'L014_sino_fulldose_geometry.json').read_text())
nu = int(geom_json['nu']); rotview = int(geom_json['rotview']); nz = int(geom_json['nz_rebinned'])
du = float(geom_json['du'])
nz_center = nz//2 + 4  # +3.5mm slab anchor

with h5py.File(sino_dir/'L014_sino_fulldose.h5', 'r') as f:
    s = np.asarray(f['sino'][:, :, nz_center], dtype=np.float32)
s = np.ascontiguousarray(np.flip(s, axis=-1))

angle_start = float(geom_json['angle_start_corrected'])
def fbp_at_px(px):
    g = FanBeamGeometry(image_size=512, pixel_spacing=px,
                        n_angles=rotview, n_det=nu, det_spacing=du,
                        sod=595.0, sdd=1085.6,
                        angle_start=angle_start, angle_end=angle_start+2*math.pi)
    p = PyronnFanBeamProjector(g).to('cuda')
    t = torch.from_numpy(s).to('cuda').float()[None, None]
    out = p.fbp(t).detach()[0,0].cpu().numpy()
    return np.fliplr(np.flipud(out))

fbp_07  = fbp_at_px(0.7)
fbp_703 = fbp_at_px(0.703125)

# Body extent at row of max width (treat thresh = 0.005 1/mm = soft tissue)
def body_extent(im, thresh=0.005):
    mask = im > thresh
    if not mask.any(): return None, None, None
    widths = mask.sum(axis=1)
    r_best = int(widths.argmax())
    row = mask[r_best]
    cols = np.where(row)[0]
    return r_best, int(cols.min()), int(cols.max())

# Find truth slice nearest patient_z = -256.86
SOP_CT = '1.2.840.10008.5.1.4.1.1.2'
truth_files = []
for series_dir in sorted(raw_dir.iterdir()):
    sample = next(series_dir.iterdir(), None)
    if sample is None: continue
    try:
        head = pydicom.dcmread(str(sample), stop_before_pixels=True)
    except Exception: continue
    if getattr(head,'SOPClassUID','') != SOP_CT: continue
    desc = getattr(head,'SeriesDescription','').lower()
    if 'full' in desc and 'image' in desc:
        for fp in series_dir.iterdir():
            try:
                m = pydicom.dcmread(str(fp), stop_before_pixels=True)
                z = float(m.ImagePositionPatient[2])
                truth_files.append((z, fp))
            except Exception:
                continue
        break
truth_files.sort()
zs = np.array([t[0] for t in truth_files])
idx = int(np.argmin(np.abs(zs + 256.86)))
ds = pydicom.dcmread(str(truth_files[idx][1]))
truth_mu = 0.02*(1+(ds.pixel_array*float(ds.RescaleSlope)+float(ds.RescaleIntercept))/1000.0).astype(np.float32)
print(f'truth slice z={truth_files[idx][0]:.2f} mm, PixelSpacing={float(ds.PixelSpacing[0]):.6f} mm')
print(f'truth FOV span = {512*float(ds.PixelSpacing[0]):.2f} mm  (Mayo ReconDiameter)')
print(f'FBP  FOV at px=0.7:    {512*0.7:.2f} mm')
print(f'FBP  FOV at px=0.703125: {512*0.703125:.2f} mm')

print('\nBody extent at widest row (threshold > 0.005 1/mm):')
for name, im, sp in [
        ('truth        (px=0.703125)', truth_mu, 0.703125),
        ('FBP px=0.7  (legacy)      ', fbp_07.clip(0), 0.7),
        ('FBP px=0.703125 (fix)     ', fbp_703.clip(0), 0.703125)]:
    r, lo, hi = body_extent(im)
    px_width = hi - lo
    phys_width = px_width * sp
    print(f'  {name}: row {r:3d}  cols [{lo:3d}, {hi:3d}]  width={px_width:3d} px = {phys_width:.2f} mm')

# Centroid of body (HU > 0)
def centroid(im, thresh=0.005):
    mask = im > thresh
    rs, cs = np.where(mask)
    return rs.mean(), cs.mean()

print('\nBody centroid (px-coords):')
for name, im, sp in [
        ('truth        (px=0.703125)', truth_mu, 0.703125),
        ('FBP px=0.7  (legacy)      ', fbp_07.clip(0), 0.7),
        ('FBP px=0.703125 (fix)     ', fbp_703.clip(0), 0.703125)]:
    r, c = centroid(im)
    print(f'  {name}: centroid (row, col) = ({r:.1f}, {c:.1f})  in physical (mm rel iso) = ({(r-255.5)*sp:+.2f}, {(c-255.5)*sp:+.2f})')
PYEOF
echo "[measure] done."
