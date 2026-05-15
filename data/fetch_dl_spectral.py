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

# Geometry: 2D fan-beam, two kVp settings. Re-verify from challenge
# `parameters.txt` after first download.
GEOMETRY = {
    "image_size": 512,
    "pixel_spacing": 0.7,                   # TODO confirm
    "n_angles": 128,                        # sparse-view; confirm
    "n_det": 1024,                          # confirm
    "det_spacing": 1.2858,                  # confirm
    "sod": 595.0,                           # confirm
    "sdd": 1085.6,                          # confirm
    "n_spectra": 2,                         # two kVp
    "n_tissues": 3,                         # adipose, fibroglandular, calc
}

# Final splits are challenge-defined; we hold out a val from train.
SPLIT_SIZES = {"train": None, "val": None}  # filled in after raw inspection


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


def stage_h5(raw_paths: list[Path], staged_dir: Path) -> None:
    raise NotImplementedError(
        "Per-challenge conversion not implemented yet. Once a sample of the "
        f"Zenodo record {ZENODO_RECORD} is on disk, inspect a file to "
        "determine the array layout (raw float32 vs MAT vs HDF5 already), "
        "then fill in this function. Output must match the docstring layout "
        "exactly so ddssl_ldct.staged_dataset.StagedH5Dataset can read it."
    )


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
        stage_h5(raw_paths, staged_dir)
        write_manifest(
            staged_dir,
            source=f"https://zenodo.org/records/{ZENODO_RECORD}",
            geometry=GEOMETRY,
            splits=SPLIT_SIZES,
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
