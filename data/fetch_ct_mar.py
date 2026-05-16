"""Fetch + stage the AAPM CT-MAR 2024 challenge data.

Sources:
  * RPI Box (full challenge dataset, 110 GB on the cluster):
      https://rpi.app.box.com/s/7p8tkqj5ewhtdad2h8kx975i9qg6b7a4
      The shared link is authenticated-only even though it looks
      public-shareable: every Box public-API endpoint (folder-zip download,
      shared_items, static) returns 401/404/512 without an OAuth bearer
      token. Programmatic download from this script requires:
        Box developer access token (Settings -> Developer Apps on
        box.com, generate token, paste as $BOX_TOKEN env var)
  * github.com/xcist/example/tree/main/AAPM_datachallenge — public mirror
    with simulator code + a small set of example cases. Optional secondary
    source if only the small example set is needed.

Size:   The full Box dataset is 110 GB compressed (14 000 cases x 5 tensors).
        Already on the cluster at /cluster/maier/Agent4CT/data/ct_mar/raw/.

Raw layout (inside each `body{N}.tar.gz` / `head{N}.tar.gz`):
    body{N}/
        Target/   training_body_nometal_img{idx}_512x512x1.raw     # truth image (HU)
                  training_body_nometal_sino{idx}_900x1000.raw     # clean sinogram
        Baseline/ training_body_metalart_img{idx}_512x512x1.raw    # metal-corrupted
                  training_body_metalart_sino{idx}_900x1000.raw    # metal-corrupted sino
        Mask/     training_body_metalonlymask_img{idx}_512x512x1.raw
                  training_body_metalinfo{idx}.json
    All .raw are headerless little-endian float32. Image dims are
    (512, 512), sino dims are (900, 1000) = (angles, detectors).
    Reconstruction FOV: 400 mm body / 220.16 mm head, both into 512x512.

Staged layout (truth-only — harness forward-projects through its own
challenge geometry at train time; see ddssl_ldct.staged_dataset):
    ct_mar/
        staged/
            train_truth.h5  (N_train_pool, 512, 512) float32   # HU
            val_truth.h5    (N_val_pool,  512, 512)  float32
            test_truth.h5   (N_test_pool, 512, 512)  float32
            manifest.json
"""

from __future__ import annotations
import argparse
import json
import os
import re
import subprocess
import sys
import tarfile
from pathlib import Path

import numpy as np

from _common import (
    assert_budget,
    data_root,
    pack_h5,
    write_manifest,
)

CHALLENGE = "ct_mar"
MIRROR_REPO = "https://github.com/xcist/example.git"
MIRROR_SUBDIR = "AAPM_datachallenge"

GEOMETRY = {
    # CT-MAR's native fan-beam geometry from the README. Documented for
    # provenance; the harness will forward-project staged truth images
    # through its OWN sparse-view challenge geometry at train time, so
    # this block is informational rather than the geometry the solver sees.
    "image_size": 512,
    "pixel_spacing_body_mm": 400.0 / 512,    # 400 mm FOV / 512 pixels
    "pixel_spacing_head_mm": 220.16 / 512,
    "n_angles": 900,
    "n_det": 1000,
    "det_layout": "fan-beam, equally-spaced over 360 degrees",
}

# Tar-files to consume per split. With ~1000 cases per tar.gz this gives a
# big enough pool for repeated train_n=400 epoch rotation. Tune by editing
# this dict if a different split is desired.
DEFAULT_SOURCES = {
    "train": ["body1.tar.gz", "body2.tar.gz", "body3.tar.gz", "body4.tar.gz"],
    "val":   ["body5.tar.gz"],
    "test":  ["body6.tar.gz"],
}
# Hard caps so a single staging run doesn't try to pack 14 k cases into one
# HDF5. Each entry is N images @ 1 MB float32 = N MB on disk before lz4.
DEFAULT_LIMITS = {"train": 4000, "val": 1000, "test": 1000}


IMG_NAME_RE = re.compile(
    r"^(body|head)\d+/Target/training_(body|head)_nometal_img(\d+)_512x512x1\.raw$"
)


def _iter_target_images(tar_path: Path, limit: int):
    """Yield (case_id, image_array) for every nometal image in a tar.gz."""
    with tarfile.open(tar_path, "r:gz") as tf:
        yielded = 0
        for member in tf:
            if yielded >= limit:
                return
            if not member.isfile():
                continue
            m = IMG_NAME_RE.match(member.name)
            if not m:
                continue
            buf = tf.extractfile(member)
            if buf is None:
                continue
            raw = buf.read()
            arr = np.frombuffer(raw, dtype="<f4").reshape(512, 512).copy()
            yield int(m.group(3)), arr
            yielded += 1


def _pack_split(staged_dir: Path, raw_dir: Path, split: str,
                tar_names: list[str], limit: int, *, shuffle_seed: int = 0,
                ) -> int:
    """Pack one split. Returns the number of images written."""
    cases: list[tuple[int, np.ndarray]] = []
    for tar_name in tar_names:
        tp = raw_dir / tar_name
        if not tp.exists():
            print(f"[stage] missing {tp}, skipping.", flush=True)
            continue
        per_tar = max(1, limit // max(1, len(tar_names)))
        print(f"[stage] streaming {tar_name} (up to {per_tar} images)…",
              flush=True)
        for idx, img in _iter_target_images(tp, per_tar):
            cases.append((idx, img))
            if len(cases) >= limit:
                break
        if len(cases) >= limit:
            break

    if not cases:
        raise RuntimeError(f"No cases for split={split!r}")

    rng = np.random.default_rng(np.uint64(shuffle_seed))
    perm = rng.permutation(len(cases))
    cases_ordered = [cases[i] for i in perm]

    def emitter():
        for i, (_, arr) in enumerate(cases_ordered):
            yield i, arr

    out = staged_dir / f"{split}_truth.h5"
    pack_h5(out, name="image",
            shape=(len(cases_ordered), 512, 512), dtype="float32",
            cases=emitter())
    return len(cases_ordered)


def stage_h5(raw_dir: Path, staged_dir: Path, *,
             sources: dict | None = None, limits: dict | None = None,
             shuffle_seed: int = 20260516) -> dict:
    """Stream-extract Target/no-metal truth images into per-split HDF5s.

    Returns the splits dict suitable for `write_manifest`.
    """
    staged_dir.mkdir(parents=True, exist_ok=True)
    sources = sources or DEFAULT_SOURCES
    limits = limits or DEFAULT_LIMITS
    splits: dict[str, int] = {}
    for split, tar_names in sources.items():
        n = _pack_split(staged_dir, raw_dir, split, tar_names,
                        limits.get(split, 1000),
                        shuffle_seed=shuffle_seed + hash(split) % (1 << 31))
        splits[split] = n
        print(f"[stage] {split}: {n} cases -> {staged_dir/(split+'_truth.h5')}",
              flush=True)
    return splits


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
    print(f"[plan] raw    expected at {raw_dir} (110 GB tar.gz already on cluster)")
    print(f"[plan] staged expected at {staged_dir} (~6 GB after lz4)")
    if args.dry_run:
        return 0

    # Staging only writes train+val+test truth HDF5s; no extra downloads.
    assert_budget(root, need_gb=20.0, reserve_gb=200.0)

    if not raw_dir.exists() or not any(raw_dir.glob("body*.tar.gz")):
        print(f"[stage] no body*.tar.gz in {raw_dir} — see header docstring "
              f"for how to download.", file=sys.stderr)
        return 1

    if staged_dir.exists() and (staged_dir / "manifest.json").exists():
        print(f"[stage] {staged_dir}/manifest.json present — skip.")
    else:
        splits = stage_h5(raw_dir, staged_dir)
        write_manifest(
            staged_dir,
            source="https://rpi.app.box.com/s/7p8tkqj5ewhtdad2h8kx975i9qg6b7a4",
            geometry=GEOMETRY,
            splits=splits,
            extra={"layout": "truth-only; harness forward-projects",
                   "sources_per_split": DEFAULT_SOURCES,
                   "limits_per_split": DEFAULT_LIMITS},
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
