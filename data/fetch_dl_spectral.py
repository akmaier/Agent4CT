"""Fetch + stage the AAPM DL-Spectral CT 2022 challenge data.

Source: Zenodo 14262737 (training + supporting files for the 2022 challenge).
Size:   ~30-50 GB raw estimated (re-measure on first download); ~25-40 GB
        staged HDF5 with lz4 / gzip-1.

Two-energy sparse fan-beam reconstruction. Each case has:
  - sinograms at two source spectra (kVp settings)
  - truth tissue-class density maps (adipose, fibroglandular, calcification)

Layout produced:
    dl_spectral/
        raw/
        staged/
            train_sinograms.h5  (N_train, 2, A, D) float32   # 2 = energies
            train_truth.h5      (N_train, 3, H, W) float32   # 3 = tissue maps
            val_*.h5
            manifest.json
"""

from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

from _common import (
    assert_budget,
    data_root,
    download,
    write_manifest,
)

CHALLENGE = "dl_spectral"
ZENODO_RECORD = "14262737"

# Populate after first fetch from the Zenodo /api/records/<id> response.
# Until then, the script downloads everything in the record and warns.
FILES: dict[str, str] = {}

# Geometry confirmed from data inspection 2026-05-16: 256 source angles,
# 1024 detector elements per sinogram. Phantom volumes are 512x512.
GEOMETRY = {
    "image_size": 512,
    "n_angles": 256,
    "n_det": 1024,
    "n_spectra": 2,                         # high-kVp + low-kVp
    "n_tissues": 3,                         # adipose, fibroglandular, calc
    "note": "spectral challenge; sinograms in transmission units (post-log NOT applied)",
}

# Splits are chosen on the fly from the 1000 cases in the Zenodo record:
# 80/10/10 cases for train/val/test.
DEFAULT_SPLITS = {"train": 800, "val": 100, "test": 100}


def fetch_raw(raw_dir: Path) -> list[Path]:
    """Pull every file in the Zenodo record."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    import urllib.request
    url = f"https://zenodo.org/api/records/{ZENODO_RECORD}"
    with urllib.request.urlopen(url) as r:
        record = json.load(r)
    local: list[Path] = []
    for f in record.get("files", []):
        name = f["key"]
        dst = raw_dir / name
        # Zenodo publishes md5; we don't pin sha256 yet (warn-only mode).
        download(f["links"]["self"], dst, expected_sha256=None)
        local.append(dst)
    return local


def _decompress_to_tmp(src: Path, tmp_dir: Path) -> Path:
    """Decompress an .npy.gz into a memory-mappable .npy in tmp_dir.

    Decompressing on disk lets us mmap one sample at a time instead of
    loading 1000-case arrays into RAM (which OOMs on the login node).
    """
    import gzip
    import shutil
    tmp_dir.mkdir(parents=True, exist_ok=True)
    dst = tmp_dir / src.name.removesuffix(".gz")
    if dst.exists():
        return dst
    print(f"[stage]   decompressing {src.name} -> {dst.name}", flush=True)
    with gzip.open(src) as g, open(dst, "wb") as out:
        shutil.copyfileobj(g, out)
    return dst


def stage_h5(raw_paths: list[Path], staged_dir: Path, *,
             splits: dict | None = None,
             shuffle_seed: int = 20260516,
             tmp_dir: Path | None = None) -> dict:
    """Pack 3-tissue phantoms + 2-kVp sinograms into per-split multi-channel
    HDF5 files.

    Layout:
        train_truth.h5      'image' shape (N_train, 3, 512, 512) float32
                              channels = [adipose, fibroglandular, calcification]
        train_sinograms.h5  'sino'  shape (N_train, 2, 256, 1024) float32
                              channels = [highkVp, lowkVp] transmission
        val_*.h5 / test_*.h5 likewise

    The .npy.gz inputs are decompressed once into `tmp_dir` (default:
    `staged_dir/../_tmp_decompress`) and mmapped — sustained working set
    stays under ~30 MB regardless of pool size.
    """
    import numpy as np
    from _common import pack_h5

    staged_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = raw_paths[0].parent if raw_paths else None
    if raw_dir is None or not raw_dir.exists():
        raise RuntimeError("dl_spectral raw_dir not found")

    tmp_dir = tmp_dir or (staged_dir.parent / "_tmp_decompress")

    print("[stage] decompressing .npy.gz inputs (one-time, ~7 GB tmp)…",
          flush=True)
    adi_p = _decompress_to_tmp(raw_dir / "Phantom_Adipose.npy.gz", tmp_dir)
    fib_p = _decompress_to_tmp(raw_dir / "Phantom_Fibroglandular.npy.gz", tmp_dir)
    cal_p = _decompress_to_tmp(raw_dir / "Phantom_Calcification.npy.gz", tmp_dir)
    hi_p = _decompress_to_tmp(raw_dir / "highkVpTransmission.npy.gz", tmp_dir)
    lo_p = _decompress_to_tmp(raw_dir / "lowkVpTransmission.npy.gz", tmp_dir)

    adi = np.load(adi_p, mmap_mode="r")
    fib = np.load(fib_p, mmap_mode="r")
    cal = np.load(cal_p, mmap_mode="r")
    hi = np.load(hi_p, mmap_mode="r")
    lo = np.load(lo_p, mmap_mode="r")
    n_total = adi.shape[0]
    for arr, name in [(fib, "fib"), (cal, "cal"), (hi, "hi"), (lo, "lo")]:
        if arr.shape[0] != n_total:
            raise RuntimeError(
                f"{name} N={arr.shape[0]} != adipose N={n_total}; "
                f"Zenodo record changed?"
            )

    splits = splits or DEFAULT_SPLITS
    total_requested = sum(splits.values())
    if total_requested > n_total:
        raise RuntimeError(
            f"requested splits sum {total_requested} > {n_total} cases")

    rng = np.random.default_rng(np.uint64(shuffle_seed))
    perm = rng.permutation(n_total)
    cursor = 0
    splits_out: dict[str, int] = {}
    H, W = adi.shape[1], adi.shape[2]
    A, D = hi.shape[1], hi.shape[2]
    for split, n_split in splits.items():
        idx = perm[cursor:cursor + n_split].tolist()
        cursor += n_split

        def truth_emitter(idx=idx):
            for i, src in enumerate(idx):
                stk = np.stack([adi[src], fib[src], cal[src]], axis=0
                               ).astype(np.float32)
                yield i, stk

        def sino_emitter(idx=idx):
            for i, src in enumerate(idx):
                stk = np.stack([hi[src], lo[src]], axis=0).astype(np.float32)
                yield i, stk

        pack_h5(staged_dir / f"{split}_truth.h5", name="image",
                shape=(n_split, 3, H, W), dtype="float32",
                cases=truth_emitter())
        pack_h5(staged_dir / f"{split}_sinograms.h5", name="sino",
                shape=(n_split, 2, A, D), dtype="float32",
                cases=sino_emitter())
        splits_out[split] = n_split
        print(f"[stage] {split}: {n_split} cases", flush=True)

    print(f"[stage] tmp decompress dir: {tmp_dir} — safe to rm after manifest "
          f"is written.", flush=True)
    return splits_out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", default=None)
    p.add_argument("--skip-download", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args(argv)

    root = data_root(args.data_root)
    challenge_dir = root / CHALLENGE
    raw_dir = challenge_dir / "raw"
    staged_dir = challenge_dir / "staged"

    print(f"[plan] AGENT4CT_DATA = {root}")
    print(f"[plan] challenge dir = {challenge_dir}")
    print(f"[plan] expected raw  = ~30-50 GB at {raw_dir}")
    if args.dry_run:
        return 0

    assert_budget(root, need_gb=60.0)

    if not args.skip_download:
        raw_paths = fetch_raw(raw_dir)
    else:
        raw_paths = sorted(raw_dir.iterdir()) if raw_dir.exists() else []
        if not raw_paths:
            print("--skip-download but raw/ empty.", file=sys.stderr)
            return 1

    if staged_dir.exists() and (staged_dir / "manifest.json").exists():
        print(f"[stage] {staged_dir}/manifest.json present — skip.")
    else:
        splits = stage_h5(raw_paths, staged_dir)
        write_manifest(
            staged_dir,
            source=f"https://zenodo.org/records/{ZENODO_RECORD}",
            geometry=GEOMETRY,
            splits=splits,
            extra={"layout": "multi-channel: truth (N,3,H,W); sino (N,2,A,D)",
                   "phantom_channels": ["adipose", "fibroglandular", "calcification"],
                   "sino_channels": ["highkVp", "lowkVp"]},
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
