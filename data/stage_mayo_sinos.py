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


def stage_split_sinos(split: str, triples_ordered, staged_dir: Path,
                      sino_dir: Path, dose: str, force: bool):
    """Pack one split's sinos."""
    out_h5 = staged_dir / f"{split}_sino_{dose}.h5"
    if out_h5.exists() and not force:
        print(f"[stage-sino] {out_h5.name} exists; pass --force to overwrite.")
        return
    n = len(triples_ordered)
    if n == 0:
        print(f"[stage-sino] split={split} dose={dose}: empty, skip.")
        return

    # Cache per-patient z_grid so we don't reload it per slice.
    z_grid_cache: dict[str, np.ndarray] = {}

    def _load_zgrid(pid: str) -> np.ndarray:
        if pid not in z_grid_cache:
            z_grid_cache[pid] = np.load(
                sino_dir / f"{pid}_sino_{dose}_z_grid.npy")
        return z_grid_cache[pid]

    # First pass: determine output (rotview, nu) shape from the first sino.
    pid0, _k, _fp = triples_ordered[0]
    sino_h5_0 = sino_dir / f"{pid0}_sino_{dose}.h5"
    if not sino_h5_0.exists():
        raise FileNotFoundError(f"missing helix2fan sino: {sino_h5_0}")
    with h5py.File(sino_h5_0, "r") as f:
        rotview, nu, _ = f["sino"].shape
    print(f"[stage-sino] {split}/{dose}: {n} slices, "
          f"target shape ({n}, {rotview}, {nu})")

    import hdf5plugin
    with h5py.File(out_h5, "w") as fout:
        ds = fout.create_dataset(
            "sino", shape=(n, rotview, nu), dtype="float32",
            chunks=(1, rotview, nu),
            **hdf5plugin.LZ4(),
        )
        # Also store per-slice patient_z + source_z for diagnostic.
        z_meta = fout.create_dataset("z_meta", shape=(n, 2), dtype="float64")

        for i, (pid, _k, fp) in enumerate(triples_ordered):
            patient_z = _patient_z_from_dicom(fp)
            source_z = -patient_z  # mapping verified for L014; same for all
                                    # head-first Mayo scans
            sino_h5 = sino_dir / f"{pid}_sino_{dose}.h5"
            z_grid = _load_zgrid(pid)
            slc = _find_sino_z_slice(sino_h5, z_grid, source_z)
            ds[i] = slc
            z_meta[i] = (patient_z, source_z)
            if (i + 1) % 50 == 0 or i == 0:
                print(f"[stage-sino] {split}/{dose}: {i+1}/{n}  "
                      f"pid={pid} patient_z={patient_z:.2f} "
                      f"source_z={source_z:.2f}", flush=True)
    print(f"[stage-sino] {split}/{dose}: wrote {out_h5}  ({n} slices)")


def main():
    args = parse_args()
    data_root = Path(args.data_root)
    challenge = data_root / "mayo_ldct"
    raw_dir = challenge / "raw"
    sino_dir = challenge / "staged_helix2fan"
    staged_dir = challenge / "staged"
    if not staged_dir.exists():
        raise FileNotFoundError(
            f"{staged_dir} missing — run data/fetch_mayo_ldct.py first "
            "to produce the truth h5s.")
    if not sino_dir.exists():
        raise FileNotFoundError(
            f"{sino_dir} missing — run "
            "cluster/slurm/rebin_mayo_helix2fan.sbatch first.")

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
