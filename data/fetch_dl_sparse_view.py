"""Fetch + stage the AAPM DL-Sparse-View CT 2021 challenge data.

** CURRENTLY BROKEN — data is CodaLab-gated, no public mirror. **

The earlier challenges/dl_sparse_view/README.md claim of "Zenodo 13882980"
was wrong; that record is the DL-Spectral info record (0 files). The
actual DL-Sparse-View dataset is hosted on the CodaLab competition portal
and requires per-user registration:

    https://dl-sparse-view-ct-challenge.eastus.cloudapp.azure.com/competitions/1

Until access is sorted (register manually, accept rules, download via
CodaLab UI), this script's network half cannot run.

Original (aspirational) docstring follows:
Size:   ~10-20 GB raw, ~8-15 GB staged (HDF5 with lz4/gzip).
Splits: 3600 train, 400 val (held-out by us), 0 test (organisers' test set
        is not in the public release; if we get it later, stage as test_*).

Layout produced (relative to AGENT4CT_DATA):

    dl_sparse_view/
        raw/                                    # untouched archive
        staged/
            train_sinograms.h5  (3600, 128, 1024)  float32
            train_truth.h5      (3600, 512, 512)   float32
            val_sinograms.h5    (400,  128, 1024)  float32
            val_truth.h5        (400,  512, 512)   float32
            manifest.json

Run from the cluster:
    python data/fetch_dl_sparse_view.py
Or local dry-run:
    python data/fetch_dl_sparse_view.py --data-root ./tmp/data --dry-run
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
    pack_h5,
    sha256_file,
    write_manifest,
)

# ---------------------------------------------------------------- constants
CHALLENGE = "dl_sparse_view"
ZENODO_RECORD = "13882980"

# NOTE: Zenodo file list + sha256s must be verified once at fetch time.
# Populate this dict after first successful download by reading
# https://zenodo.org/api/records/13882980 ; the script then refuses to
# proceed without checksum match on subsequent runs. Until then, the dict
# stays empty and the script downloads but does NOT verify (warns instead).
FILES = {
    # "Phantoms.tar.gz": "<sha256>",
    # "Sinograms_full.tar.gz": "<sha256>",
    # ... fill in from the actual Zenodo manifest
}

# Fan-beam geometry from Sidky et al. 2022, Table 1. Pixel spacing and SDD
# are taken from the challenge `parameters.txt`. Re-verify against the
# parameters file that ships with the download.
GEOMETRY = {
    "image_size": 512,
    "pixel_spacing": 0.7,
    "n_angles": 128,
    "n_det": 1024,
    "det_spacing": 1.2858,
    "sod": 595.0,
    "sdd": 1085.6,
}

SPLIT_SIZES = {"train": 3600, "val": 400}  # 4000 total; we hold out 400 as val


# ---------------------------------------------------------------- pipeline
def fetch_raw(raw_dir: Path) -> list[Path]:
    """Download the Zenodo files into raw_dir. Returns the local paths."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    local = []
    if not FILES:
        print("[fetch] FILES dict is empty — running in download-only mode "
              "(no checksum verification). Populate FILES after first run.",
              file=sys.stderr)
        # Without a populated manifest, ask Zenodo for the file list.
        url = f"https://zenodo.org/api/records/{ZENODO_RECORD}"
        import urllib.request
        with urllib.request.urlopen(url) as r:
            record = json.load(r)
        for f in record.get("files", []):
            name = f["key"]
            dst = raw_dir / name
            download(f["links"]["self"], dst,
                     expected_sha256=f.get("checksum", "").removeprefix("md5:")
                     if False else None)  # Zenodo uses md5; sha256 separately
            local.append(dst)
        return local
    for name, want_sha in FILES.items():
        dst = raw_dir / name
        url = f"https://zenodo.org/records/{ZENODO_RECORD}/files/{name}"
        download(url, dst, expected_sha256=want_sha)
        local.append(dst)
    return local


def stage_h5(raw_paths: list[Path], staged_dir: Path) -> None:
    """Convert the raw archives into the staged HDF5 layout.

    The conversion logic is challenge-specific and depends on what's in the
    archive (DICOM vs raw float32 vs MAT). We split this into a separate
    helper so the network half (fetch) and CPU half (convert) can be
    re-run independently.
    """
    raise NotImplementedError(
        "Per-challenge conversion not yet implemented. Inspect the raw "
        "files under data/dl_sparse_view/raw/ and fill in this function — "
        "the expected output is described in the module docstring."
    )


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
    print(f"[plan] expected raw  = ~10-20 GB at {raw_dir}")
    print(f"[plan] staged splits = {SPLIT_SIZES}")
    if args.dry_run:
        return 0

    assert_budget(root, need_gb=25.0)

    if not args.skip_download:
        raw_paths = fetch_raw(raw_dir)
    else:
        raw_paths = sorted(raw_dir.iterdir()) if raw_dir.exists() else []
        if not raw_paths:
            print("[fetch] --skip-download but raw/ is empty; nothing to do.",
                  file=sys.stderr)
            return 1

    if staged_dir.exists() and (staged_dir / "manifest.json").exists():
        print(f"[stage] {staged_dir}/manifest.json present — skip conversion.")
    else:
        stage_h5(raw_paths, staged_dir)
        write_manifest(staged_dir,
                       source=f"https://zenodo.org/records/{ZENODO_RECORD}",
                       geometry=GEOMETRY, splits=SPLIT_SIZES)

    return 0


if __name__ == "__main__":
    sys.exit(main())
