"""Canonical-frame re-stage of the Mayo Wagner data (2026-06-14).

The old staged_*.h5 reconstructed every patient with ONE uniform geometry, but
the rebin emits per-patient angle_start_corrected (≈88° spread) + per-patient
pixel-spacing → uniform-geom FBP is rotated/mis-scaled vs truth (SSIM 0.24 vs
the 0.81 per-patient baseline). See findings.md 2026-06-14.

This bakes each patient's per-patient geometry into a CANONICAL frame so that a
single uniform geometry (angle_start=0, ps=0.700857) reconstructs every patient
correctly. Validated transform (job 763705, L277 SSIM 0.807 == baseline):
  canonical_sino  = roll(flip_u(slab_mean), round(angle_start_corrected/Δ))
  canonical_truth = fliplr(flipud(resample(truth, native_ps -> 0.700857)))
with Δ = 2π/rotview. The image-flip (fliplr+flipud) that the baseline applied
post-FBP is folded into the stored truth (uniform across patients), so the
solver's plain proj.fbp(canonical_sino) lands on canonical_truth.

Truth is resampled to a COMMON pixel-spacing (0.700857) so one uniform image
grid serves all patients (per-patient FOV differs: 0.66–0.78). Slot i of the
sino and truth h5s correspond to the same physical slice.

Writes data/mayo_ldct/staged_canonical/{split}_{sino_lowdose,sino_fulldose,
truth}.h5. With --validate, FBPs the val split and prints SSIM vs canonical
truth (must be ~0.81). Does NOT touch the broken staged/ dir.
"""
from __future__ import annotations
import argparse, math, os, sys, time
from pathlib import Path
import h5py, numpy as np, torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
from scripts.compare_hd_ld_fbp_allslices import enumerate_truth, _load_mu, WAGNER_SPLIT_OF

WAGNER = {"train": ["L145", "L186", "L209", "L219"], "val": ["L277"],
          "test": ["L014", "L056", "L058", "L075", "L123"]}
COMMON_PS = 0.700857
SLAB_HALF = 2


def resample_truth(truth_np, native_ps, common_ps=COMMON_PS, size=512):
    """Resample a 512² truth from native ps to common ps (same physical centre),
    so it shares the recon's image grid. scale>1 zooms in (native_ps>common_ps)."""
    t = torch.from_numpy(np.ascontiguousarray(truth_np))[None, None].float()
    scale = native_ps / common_ps
    new = max(1, int(round(size * scale)))
    t = F.interpolate(t, size=(new, new), mode="bilinear", align_corners=False)
    if new >= size:
        o = (new - size) // 2
        t = t[..., o:o + size, o:o + size]
    else:
        pad = size - new; l = pad // 2; r = pad - l
        t = F.pad(t, (l, r, l, r))
    return t[0, 0].numpy().astype(np.float32)


def patient_roll(geom_json):
    a0 = float(geom_json["angle_start_corrected"]); rot = int(geom_json["rotview"])
    return int(round(a0 / (2 * math.pi / rot))) % rot


def stage_split(split, sino_dir, truth_root, out_dir, doses, force):
    import json, hdf5plugin
    pats = WAGNER[split]
    # enumerate truth slices per patient (sorted by patient_z), build slot list
    triples = []   # (patient, patient_z, fp, native_ps)
    for pid in pats:
        tfiles, tmeta = enumerate_truth(truth_root / pid)
        if not tfiles:
            print(f"[canon] {split}: {pid} no truth, skip", flush=True); continue
        for pz, fp in tfiles:
            triples.append((pid, pz, fp, tmeta["pixel_spacing"]))
    n = len(triples)
    print(f"[canon] {split}: {n} slices across {len(pats)} patients", flush=True)

    # truth h5 (dose-independent) — written once
    truth_h5 = out_dir / f"{split}_truth.h5"
    if force or not truth_h5.exists():
        with h5py.File(truth_h5, "w") as ft:
            dt = ft.create_dataset("truth", shape=(n, 512, 512), dtype="float32",
                                   chunks=(1, 512, 512), **hdf5plugin.LZ4())
            for i, (pid, pz, fp, nps) in enumerate(triples):
                tr = resample_truth(_load_mu(fp), nps)
                dt[i] = np.flip(np.flip(tr, 0), 1)   # folded image-flip
        print(f"[canon] wrote {truth_h5.name}", flush=True)

    # sino h5 per dose
    for dose in doses:
        out_h5 = out_dir / f"{split}_sino_{dose}.h5"
        if out_h5.exists() and not force:
            print(f"[canon] {out_h5.name} exists; --force to overwrite", flush=True); continue
        # group slots by patient to load each raw sino once
        per_pat = {}
        for i, (pid, pz, fp, nps) in enumerate(triples):
            per_pat.setdefault(pid, []).append((i, pz))
        # shape from first patient
        gj0 = json.loads((sino_dir / f"{pats[0]}_sino_{dose}_geometry.json").read_text())
        rot, nu = int(gj0["rotview"]), int(gj0["nu"])
        with h5py.File(out_h5, "w") as fo:
            ds = fo.create_dataset("sino", shape=(n, rot, nu), dtype="float32",
                                   chunks=(1, rot, nu), **hdf5plugin.LZ4())
            for pid, slots in per_pat.items():
                gj = json.loads((sino_dir / f"{pid}_sino_{dose}_geometry.json").read_text())
                r = patient_roll(gj)
                zgrid = np.load(sino_dir / f"{pid}_sino_{dose}_z_grid.npy")
                t0 = time.time()
                with h5py.File(sino_dir / f"{pid}_sino_{dose}.h5", "r") as fs:
                    arr = np.asarray(fs["sino"][...], dtype=np.float32)   # (rot,nu,nz)
                for slot, pz in slots:
                    sz = -pz
                    j = int(np.argmin(np.abs(zgrid - sz)))
                    lo, hi = max(0, j - SLAB_HALF), min(arr.shape[2], j + SLAB_HALF + 1)
                    slab = np.flip(arr[:, :, lo:hi].mean(axis=2), axis=-1)  # u-flip + slab
                    ds[slot] = np.roll(np.ascontiguousarray(slab), r, axis=0)
                print(f"[canon] {split}/{dose}: {pid} roll={r} {len(slots)} slices "
                      f"[{time.time()-t0:.0f}s]", flush=True)
                del arr
        print(f"[canon] wrote {out_h5.name}", flush=True)
    return n


def validate(out_dir, device="cuda"):
    from ddssl_ldct.geometry import FanBeamGeometry, MAYO_LDCT_DET_OFFSET, MAYO_LDCT_TRUNCATION
    from ddssl_ldct.pyronn_projector import PyronnFanBeamProjector
    from ddssl_ldct.metrics import evaluate_calibrated
    os.environ.setdefault("AGENT4CT_DATASET", "mayo_ldct_2d")
    with h5py.File(out_dir / "val_sino_lowdose.h5", "r") as f:
        sino = torch.from_numpy(f["sino"][:]).to(device).float().unsqueeze(1)
    with h5py.File(out_dir / "val_truth.h5", "r") as f:
        truth = torch.from_numpy(f["truth"][:]).to(device).float().unsqueeze(1)
    rot, nu = sino.shape[-2], sino.shape[-1]
    geom = FanBeamGeometry(image_size=512, pixel_spacing=COMMON_PS, n_angles=rot,
                           n_det=nu, det_spacing=1.285044, sod=595.362, sdd=1086.803,
                           angle_start=0.0, angle_end=2 * math.pi)
    proj = PyronnFanBeamProjector(geom).to(device)  # env-applied det_offset + truncation
    ld = proj.fbp(sino).clamp(min=0.0)
    r = evaluate_calibrated(ld, truth, baseline=ld, display_min=0.0, display_max=0.05, fov=False)
    print(f"[canon] VALIDATE val canonical LD-FBP SSIM={r['val_ssim']:.4f} "
          f"(expect ~0.81; per-patient baseline was 0.8078)", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--splits", default="val,train,test")
    p.add_argument("--doses", default="lowdose,fulldose")
    p.add_argument("--force", action="store_true")
    p.add_argument("--validate", action="store_true")
    p.add_argument("--data-root", default="/cluster/maier/Agent4CT/data/mayo_ldct")
    p.add_argument("--subdir", default="staged_helix2fan_v3")
    a = p.parse_args()
    root = Path(a.data_root)
    sino_dir = root / a.subdir
    truth_root = root / "raw"
    out_dir = root / "staged_canonical"
    out_dir.mkdir(parents=True, exist_ok=True)
    doses = a.doses.split(",")
    for split in a.splits.split(","):
        stage_split(split, sino_dir, truth_root, out_dir, doses, a.force)
    if a.validate:
        validate(out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
