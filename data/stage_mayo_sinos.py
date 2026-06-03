"""Aggregate per-patient helix2fan sinos into per-split H5s aligned to the
existing per-split truth H5s.

After the bulk helix2fan rebin (cluster/slurm/rebin_mayo_helix2fan.sbatch)
finishes, this script:

1. Walks the same `_collect_split` + permutation that `stage_h5` used
   to pack `{train,val,test}_truth.h5` — guarantees slot-by-slot
   alignment.
2. For each output slot i = (patient, dicom_z), opens the matching
   `staged_helix2fan/<patient>_sino_<dose>.h5`, finds the sino
   z-slice closest to patient_z (via the Mayo head-first DICOM
   convention `source_z = -patient_z + 0` verified for L014, 2026-05-24),
   and packs the (rotview, nu) sino at that z into the per-split file.
3. Writes both `_sino_lowdose.h5` and `_sino_fulldose.h5` per split.
   Solvers usually want LD as the input (LDCT denoising convention),
   but having both makes baseline FBP comparison easy.

Output layout (in `data/mayo_ldct/staged/`):

    train_truth.h5         (already exists from stage_h5)
    train_sino_fulldose.h5 shape (N_train, rotview, nu)  float32   ← this script
    train_sino_lowdose.h5  shape (N_train, rotview, nu)  float32   ← this script
    val_*  same
    test_* same

Run after the helix2fan bulk rebin:

    python data/stage_mayo_sinos.py [--dose fulldose|lowdose|both]
"""
from __future__ import annotations
import argparse
import json
import os
import sys
from pathlib import Path

import h5py
import numpy as np
import pydicom

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

# Pull the same constants the truth-staging script used. Tolerant of
# either invocation: `python data/stage_mayo_sinos.py` (data/ implicit
# namespace package) or `cd data && python stage_mayo_sinos.py`.
try:
    from data.fetch_mayo_ldct import WAGNER_SPLITS, _collect_split
except ImportError:
    # Sibling-module import when REPO root happens to be `data/`.
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from fetch_mayo_ldct import WAGNER_SPLITS, _collect_split   # type: ignore


DEFAULT_DATA_ROOT = Path(os.environ.get(
    "AGENT4CT_DATA", "/cluster/maier/Agent4CT/data"))


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT),
                   help="Agent4CT data root; defaults to env or "
                        "/cluster/maier/Agent4CT/data")
    p.add_argument("--dose", choices=["fulldose", "lowdose", "both"],
                   default="both",
                   help="Which sino flavor to stage (default both).")
    p.add_argument("--shuffle-seed", type=int, default=20260516,
                   help="MUST match the seed used by `stage_h5` for the "
                        "truth h5s — otherwise sino and truth misalign.")
    p.add_argument("--force", action="store_true",
                   help="Overwrite existing output H5s.")
    return p.parse_args()


def _patient_z_from_dicom(fp: Path) -> float:
    ds = pydicom.dcmread(str(fp), stop_before_pixels=True)
    return float(ds.ImagePositionPatient[2])


def _find_sino_z_slice(sino_h5: Path, z_grid: np.ndarray,
                       target_source_z: float) -> np.ndarray:
    """Z-interpolate the (rotview, nu) sino slice at `target_source_z`.

    Returns a (rotview, nu) float32 array. Uses linear interpolation
    between the two bracketing z-grid slices.
    """
    n_z = z_grid.size
    if target_source_z <= z_grid[0]:
        i_lo = i_hi = 0; w = 1.0
    elif target_source_z >= z_grid[-1]:
        i_lo = i_hi = n_z - 1; w = 1.0
    else:
        # Sorted ascending — searchsorted is fine.
        idx = int(np.searchsorted(z_grid, target_source_z, side="left"))
        i_hi = idx
        i_lo = idx - 1
        z_lo, z_hi = float(z_grid[i_lo]), float(z_grid[i_hi])
        w = (z_hi - target_source_z) / (z_hi - z_lo)
        w = float(max(0.0, min(1.0, w)))
    with h5py.File(sino_h5, "r") as f:
        if i_lo == i_hi:
            arr = f["sino"][:, :, i_lo].astype(np.float32)
        else:
            arr_lo = f["sino"][:, :, i_lo].astype(np.float32)
            arr_hi = f["sino"][:, :, i_hi].astype(np.float32)
            arr = (w * arr_lo + (1.0 - w) * arr_hi).astype(np.float32)
    return arr


def _interp_slice_from_ram(sino_arr: np.ndarray, z_grid: np.ndarray,
                            target_source_z: float) -> np.ndarray:
    """In-memory z-interpolation. `sino_arr` shape (rotview, nu, nz);
    returns (rotview, nu) float32."""
    n_z = z_grid.size
    if target_source_z <= z_grid[0]:
        return sino_arr[:, :, 0].astype(np.float32, copy=False)
    if target_source_z >= z_grid[-1]:
        return sino_arr[:, :, n_z - 1].astype(np.float32, copy=False)
    idx = int(np.searchsorted(z_grid, target_source_z, side="left"))
    i_hi = idx
    i_lo = idx - 1
    z_lo, z_hi = float(z_grid[i_lo]), float(z_grid[i_hi])
    w = (z_hi - target_source_z) / (z_hi - z_lo)
    w = float(max(0.0, min(1.0, w)))
    arr_lo = sino_arr[:, :, i_lo]
    arr_hi = sino_arr[:, :, i_hi]
    return (w * arr_lo + (1.0 - w) * arr_hi).astype(np.float32, copy=False)


def stage_split_sinos(split: str, triples_ordered, staged_dir: Path,
                      sino_dir: Path, dose: str, force: bool):
    """Pack one split's sinos.

    Performance note (2026-06-03): the original per-slice path opened
    each per-patient H5 sino once per slice. Combined with chunks=
    (1, nu, nz) — chunk along the rotview axis — extracting an axis-2
    z-slice required decompressing ALL ~2304 chunks (≈2 GB) per slice,
    giving ~1 slice/min throughput (SLURM 762641 timed out at 10 h with
    only 1.25/8 H5s done).

    Fix: group triples by patient, open each patient's H5 ONCE, load
    the full ~3 GB sino into RAM, and z-interp from memory. Peak RAM
    ~3 GB per patient is comfortable on a 24 GB node. Expected
    speedup: ~50–100× (~ minutes per split instead of hours)."""
    out_h5 = staged_dir / f"{split}_sino_{dose}.h5"
    if out_h5.exists() and not force:
        print(f"[stage-sino] {out_h5.name} exists; pass --force to overwrite.")
        return
    n = len(triples_ordered)
    if n == 0:
        print(f"[stage-sino] split={split} dose={dose}: empty, skip.")
        return

    # Group triples by patient ID. ordered_indices preserves the global
    # output-slot mapping the truth h5 used.
    per_patient: dict[str, list[tuple[int, Path]]] = {}
    for out_idx, (pid, _k, fp) in enumerate(triples_ordered):
        per_patient.setdefault(pid, []).append((out_idx, fp))
    n_patients = len(per_patient)

    # First pass: determine output (rotview, nu) shape from the first patient.
    pid0 = next(iter(per_patient))
    sino_h5_0 = sino_dir / f"{pid0}_sino_{dose}.h5"
    if not sino_h5_0.exists():
        raise FileNotFoundError(f"missing helix2fan sino: {sino_h5_0}")
    with h5py.File(sino_h5_0, "r") as f:
        rotview, nu, _ = f["sino"].shape
    print(f"[stage-sino] {split}/{dose}: {n} slices across {n_patients} "
          f"patients, target shape ({n}, {rotview}, {nu})", flush=True)

    import hdf5plugin
    import time
    t0 = time.time()
    written = 0
    with h5py.File(out_h5, "w") as fout:
        ds = fout.create_dataset(
            "sino", shape=(n, rotview, nu), dtype="float32",
            chunks=(1, rotview, nu),
            **hdf5plugin.LZ4(),
        )
        # Also store per-slice patient_z + source_z for diagnostic.
        z_meta = fout.create_dataset("z_meta", shape=(n, 2), dtype="float64")

        for k_pat, (pid, idx_fp_list) in enumerate(per_patient.items()):
            sino_h5 = sino_dir / f"{pid}_sino_{dose}.h5"
            z_grid_path = sino_dir / f"{pid}_sino_{dose}_z_grid.npy"
            if not sino_h5.exists() or not z_grid_path.exists():
                print(f"[stage-sino] {split}/{dose}: {pid} missing files, "
                      f"skipping its {len(idx_fp_list)} slices", flush=True)
                continue
            z_grid = np.load(z_grid_path)
            t1 = time.time()
            # Full read into RAM — ~3 GB for a 24 GB Mayo node, easy.
            with h5py.File(sino_h5, "r") as f:
                sino_arr = f["sino"][:]
            load_s = time.time() - t1
            print(f"[stage-sino] {split}/{dose}: patient {k_pat+1}/{n_patients} "
                  f"{pid} loaded {sino_arr.nbytes/1e9:.2f} GB in {load_s:.1f}s, "
                  f"interpolating {len(idx_fp_list)} slices …", flush=True)
            for out_idx, fp in idx_fp_list:
                patient_z = _patient_z_from_dicom(fp)
                source_z = -patient_z   # head-first Mayo convention
                slc = _interp_slice_from_ram(sino_arr, z_grid, source_z)
                ds[out_idx] = slc
                z_meta[out_idx] = (patient_z, source_z)
                written += 1
            del sino_arr  # free 3 GB before next patient

    elapsed = time.time() - t0
    print(f"[stage-sino] {split}/{dose}: wrote {out_h5}  ({written} slices, "
          f"{elapsed:.1f}s total = {written/max(elapsed,1.0):.1f} slices/s)",
          flush=True)


def main():
    args = parse_args()
    data_root = Path(args.data_root)
    challenge = data_root / "mayo_ldct"
    raw_dir = challenge / "raw"
    # Source directory for per-patient helix2fan sinos. Mirrors the
    # env-var pattern in fetch_mayo_ldct.py: default "staged_helix2fan/"
    # (legacy DICOM-nominal SSR), or set STAGED_HELIX2FAN_SUBDIR=
    # "staged_helix2fan_ssr_fitted" to read the multi-GT-fitted rebin
    # produced by `rebin_mayo_helix2fan_ssr_fitted.sbatch`.
    sino_subdir = os.environ.get("STAGED_HELIX2FAN_SUBDIR",
                                   "staged_helix2fan")
    sino_dir = challenge / sino_subdir
    staged_dir = challenge / "staged"
    print(f"[stage-sino] sino_dir = {sino_dir}", flush=True)
    if not staged_dir.exists():
        raise FileNotFoundError(
            f"{staged_dir} missing — run data/fetch_mayo_ldct.py first "
            "to produce the truth h5s.")
    if not sino_dir.exists():
        raise FileNotFoundError(
            f"{sino_dir} missing — run "
            "cluster/slurm/rebin_mayo_helix2fan{_ssr_fitted}.sbatch first.")

    doses = ["fulldose", "lowdose"] if args.dose == "both" else [args.dose]
    for split, pids in WAGNER_SPLITS.items():
        triples = _collect_split(pids, raw_dir)
        if not triples:
            print(f"[stage-sino] {split}: no slices, skip.")
            continue
        rng = np.random.default_rng(
            np.uint64(args.shuffle_seed + hash(split) % (1 << 31)))
        perm = rng.permutation(len(triples))
        triples_ordered = [triples[i] for i in perm]

        # Sanity: verify the per-slice ordering matches the truth h5.
        truth_h5 = staged_dir / f"{split}_truth.h5"
        if truth_h5.exists():
            with h5py.File(truth_h5, "r") as f:
                n_truth = f["image"].shape[0]
            if n_truth != len(triples_ordered):
                print(f"[stage-sino] WARNING {split}: truth h5 has {n_truth} "
                      f"slices but collected {len(triples_ordered)} triples — "
                      f"alignment may be off!", file=sys.stderr)

        for dose in doses:
            try:
                stage_split_sinos(split, triples_ordered, staged_dir, sino_dir,
                                  dose, force=args.force)
            except FileNotFoundError as e:
                print(f"[stage-sino] {split}/{dose}: missing helix2fan files: {e}",
                      file=sys.stderr)
                continue


if __name__ == "__main__":
    main()
