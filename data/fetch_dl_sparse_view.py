"""Fetch + stage the AAPM DL-Sparse-View CT 2021 challenge data.

Source: Zenodo 14173522 (Sidky 2024 public release of the 2021 challenge
training set). See:
  Sidky E.Y., Pan X. "Report on the AAPM deep-learning sparse-view CT
  grand challenge." Med Phys. 2022; 49: 4935-4943.

Raw layout on Zenodo (13 files, ~5 GB compressed, ~10 GB decompressed):
    Phantom_batch{1..4}.npy.gz    each (1000, 512, 512) float32   = truth
    FBP128_batch{1..4}.npy.gz     each (1000, 512, 512) float32   = 128-view FBP ref
    Sinogram_batch{1..4}.npy.gz   each (1000, 128, 1024) float32  = 128-view sinos
    metrics_script.py             (the original scoring script)

Staged layout (this script's output):
    dl_sparse_view/
        raw/                                 # untouched .npy.gz archives
        staged/
            train_truth.h5      (3600, 512, 512)   float32
            train_sinograms.h5  (3600, 128, 1024)  float32
            train_fbp128.h5     (3600, 512, 512)   float32   # reference recon
            val_truth.h5        (400,  512, 512)   float32
            val_sinograms.h5    (400,  128, 1024)  float32
            val_fbp128.h5       (400,  512, 512)   float32
            manifest.json

Geometry pinned from Sidky et al. 2022 + the challenge `parameters.txt`:
128 projections evenly over 360 degrees onto a 1024-pixel linear (flat)
detector. Pixel spacing 0.7 mm, source-object distance 595 mm,
source-detector distance 1085.6 mm, detector spacing 1.2858 mm. This is
identical to the geometry used by `pentathlon/demo_dl_reference/`
solvers, so the staged data and the harness's forward-projector agree.

Run on the cluster:
    python data/fetch_dl_sparse_view.py
Re-run conversion only (skip download):
    python data/fetch_dl_sparse_view.py --skip-download
Dry plan:
    python data/fetch_dl_sparse_view.py --dry-run
"""

from __future__ import annotations
import argparse
import gzip
import json
import sys
import time
from pathlib import Path

import numpy as np

from _common import (
    assert_budget,
    data_root,
    download,
    pack_h5,
    sha256_file,
    write_manifest,
)

# ---------------------------------------------------------------- constants
CHALLENGE = "dl_sparse_view"
ZENODO_RECORD = "14173522"
N_BATCHES = 4
PER_BATCH = 1000   # the public release packs 1000 cases per batch file

# Files we actually need (we ignore metrics_script.py for staging).
WANT_PATTERNS = ("Phantom_batch", "FBP128_batch", "Sinogram_batch")

# Fan-beam geometry pinned from Sidky 2022 Table 1 + parameters.txt.
GEOMETRY = {
    "image_size": 512,
    "pixel_spacing": 0.7,         # mm
    "n_angles": 128,
    "n_det": 1024,
    "det_spacing": 1.2858,        # mm (linear/flat detector)
    "sod": 595.0,                 # mm (source-object distance)
    "sdd": 1085.6,                # mm (source-detector distance)
    "det_layout": "linear (flat), 128 projections evenly over 360 degrees",
    "angle_range_deg": 360.0,
}

# 4000 total -> 3600 train / 400 val. The challenge's test set is not in
# the public Zenodo release; if we get it later, stage as test_*.h5.
SPLIT_SIZES = {"train": 3600, "val": 400}


# ---------------------------------------------------------------- pipeline
def _filter_files(records: list[dict]) -> list[dict]:
    """Keep only the Phantom/FBP128/Sinogram batch files (drop metrics_script.py)."""
    keep = []
    for f in records:
        name = f["key"]
        if any(p in name for p in WANT_PATTERNS) and name.endswith(".npy.gz"):
            keep.append(f)
    return keep


def fetch_raw(raw_dir: Path) -> list[Path]:
    """Download all .npy.gz files into raw_dir. Returns the local paths."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    import urllib.request
    url = f"https://zenodo.org/api/records/{ZENODO_RECORD}"
    with urllib.request.urlopen(url) as r:
        record = json.load(r)
    files = _filter_files(record.get("files", []))
    if len(files) != 3 * N_BATCHES:
        raise RuntimeError(
            f"Expected {3 * N_BATCHES} batch files in Zenodo {ZENODO_RECORD}, "
            f"got {len(files)}. List: {[f['key'] for f in files]}"
        )
    local: list[Path] = []
    for f in files:
        name = f["key"]
        dst = raw_dir / name
        # Zenodo publishes md5 (not sha256) — pass None so we don't try to
        # match. Verification happens via the staged manifest's sha256.
        download(f["links"]["self"], dst, expected_sha256=None)
        local.append(dst)
    return local


def _load_gz_npy(path: Path) -> np.ndarray:
    """Decompress + load a .npy.gz into a numpy array. Streams; uses ~max-decompressed memory once."""
    print(f"[load]   {path.name}", flush=True)
    t0 = time.time()
    with gzip.open(path, "rb") as f:
        arr = np.load(f)
    print(f"[load]     -> shape={arr.shape}  dtype={arr.dtype}  "
          f"in {time.time() - t0:.1f}s", flush=True)
    return arr


def _stage_one_kind(raw_dir: Path, staged_dir: Path, *,
                    kind: str, per_sample_shape: tuple,
                    split_sizes: dict) -> None:
    """Stage one data kind (Phantom / FBP128 / Sinogram) across all 4 batches.

    Concatenates the 4 batches into one logical (4000, *per_sample_shape)
    array, then writes split-named HDF5 files according to split_sizes.

    File naming convention in the staged dir:
        Phantom  -> {split}_truth.h5         (dataset name "image")
        FBP128   -> {split}_fbp128.h5        (dataset name "image")
        Sinogram -> {split}_sinograms.h5     (dataset name "sino")
    """
    out_name_by_kind = {
        "Phantom":  ("truth",     "image"),
        "FBP128":   ("fbp128",    "image"),
        "Sinogram": ("sinograms", "sino"),
    }
    file_suffix, ds_name = out_name_by_kind[kind]

    # Pre-load all 4 batches (each ~0.5-2 GB decompressed). Keep them in
    # RAM long enough to write the split files. Peak ~ 4 GB for FBP/Phantom,
    # ~2 GB for Sinograms.
    batches = []
    for b in range(1, N_BATCHES + 1):
        p = raw_dir / f"{kind}_batch{b}.npy.gz"
        if not p.exists():
            raise FileNotFoundError(p)
        a = _load_gz_npy(p)
        if a.shape != (PER_BATCH,) + per_sample_shape:
            raise RuntimeError(
                f"Unexpected shape for {p.name}: {a.shape}, "
                f"want ({PER_BATCH},) + {per_sample_shape}"
            )
        batches.append(a)

    # Each split takes a contiguous chunk of the concatenated [0..4000)
    # sequence. With sizes (3600, 400) that's:
    #     train -> [0    .. 3600)
    #     val   -> [3600 .. 4000)
    cursor = 0
    for split, n in split_sizes.items():
        out = staged_dir / f"{split}_{file_suffix}.h5"
        shape = (n,) + per_sample_shape

        # Build a closure-bound generator that yields (idx, arr) pairs from
        # the in-memory concatenated batches at the right offsets.
        def gen(_cursor=cursor, _n=n):
            for i in range(_n):
                global_i = _cursor + i
                batch_i = global_i // PER_BATCH
                inner_i = global_i % PER_BATCH
                yield i, batches[batch_i][inner_i]

        pack_h5(out, name=ds_name, shape=shape, dtype="float32",
                cases=gen(), compression="lz4")
        cursor += n


def stage_h5(raw_dir: Path, staged_dir: Path) -> None:
    """Convert the 12 .npy.gz batches into 6 staged HDF5 files (3 kinds × 2 splits)."""
    staged_dir.mkdir(parents=True, exist_ok=True)

    _stage_one_kind(raw_dir, staged_dir,
                    kind="Phantom",
                    per_sample_shape=(512, 512),
                    split_sizes=SPLIT_SIZES)
    _stage_one_kind(raw_dir, staged_dir,
                    kind="FBP128",
                    per_sample_shape=(512, 512),
                    split_sizes=SPLIT_SIZES)
    _stage_one_kind(raw_dir, staged_dir,
                    kind="Sinogram",
                    per_sample_shape=(128, 1024),
                    split_sizes=SPLIT_SIZES)


# ---------------------------------------------------------------- entrypoint
def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", default=None,
                   help="Override AGENT4CT_DATA (default: /cluster/maier/Agent4CT/data)")
    p.add_argument("--skip-download", action="store_true",
                   help="Reuse existing raw/ files; only run the conversion.")
    p.add_argument("--dry-run", action="store_true",
                   help="Print plan and exit without touching the network.")
    args = p.parse_args(argv)

    root = data_root(args.data_root)
    challenge_dir = root / CHALLENGE
    raw_dir = challenge_dir / "raw"
    staged_dir = challenge_dir / "staged"

    print(f"[plan] AGENT4CT_DATA = {root}")
    print(f"[plan] challenge dir = {challenge_dir}")
    print(f"[plan] expected raw  = ~5 GB at {raw_dir}")
    print(f"[plan] staged splits = {SPLIT_SIZES}")
    print(f"[plan] zenodo record = {ZENODO_RECORD}")
    if args.dry_run:
        return 0

    # Need ~5 GB raw + ~10 GB staged (uncompressed in HDF5 even with lz4).
    assert_budget(root, need_gb=20.0)

    if not args.skip_download:
        raw_paths = fetch_raw(raw_dir)
    else:
        raw_paths = sorted(raw_dir.glob("*.npy.gz"))
        if not raw_paths:
            print("[fetch] --skip-download but raw/ is empty; nothing to do.",
                  file=sys.stderr)
            return 1

    if (staged_dir / "manifest.json").exists():
        print(f"[stage] {staged_dir}/manifest.json present — skip conversion.")
    else:
        stage_h5(raw_dir, staged_dir)
        write_manifest(
            staged_dir,
            source=f"https://zenodo.org/records/{ZENODO_RECORD}",
            geometry=GEOMETRY, splits=SPLIT_SIZES,
            extra={
                "layout": "truth + 128-view sinograms + 128-view FBP128 reference",
                "dataset_files": {
                    "truth":     "Phantom_batch{1..4}.npy.gz",
                    "sino":      "Sinogram_batch{1..4}.npy.gz",
                    "fbp128":    "FBP128_batch{1..4}.npy.gz",
                },
                "citation": (
                    "Sidky EY, Pan X. Report on the AAPM deep-learning "
                    "sparse-view CT grand challenge. Med Phys. 2022;49:4935-4943."
                ),
                "acknowledgement": "Emil Sidky (University of Chicago) — public release Nov 2024",
            },
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
