"""Fetch + stage the AAPM CT-MAR 2024 challenge data.

Sources:
  * RPI Box (full challenge dataset, 110 GB on the cluster):
      https://rpi.app.box.com/s/7p8tkqj5ewhtdad2h8kx975i9qg6b7a4
      Already on the cluster at /cluster/maier/Agent4CT/data/ct_mar/raw/
      (13 body*.tar.gz + 2 head*.tar.gz). Download from outside the
      cluster needs a Box developer access token (this script does not
      try to re-download; it stages from existing raw/).
  * github.com/xcist/example/tree/main/AAPM_datachallenge — public mirror
    with simulator code + example cases.

Raw layout (inside each `body{N}.tar.gz`):
    body{N}/
        Target/   training_body_nometal_img{idx}_512x512x1.raw     # clean truth image (HU)
                  training_body_nometal_sino{idx}_900x1000.raw     # clean sinogram (line integrals)
        Baseline/ training_body_metalart_img{idx}_512x512x1.raw    # metal-corrupted FBP image (HU)
                  training_body_metalart_sino{idx}_900x1000.raw    # metal-corrupted sinogram
        Mask/     training_body_metalonlymask_img{idx}_512x512x1.raw  # binary metal segmentation
                  training_body_metalinfo{idx}.json                 # metal location/material metadata
    All .raw are headerless little-endian float32. Image dims (512,512),
    sino dims (900,1000) = (angles, detectors). Reconstruction FOV is
    400 mm body / 220.16 mm head, both into 512 x 512.

Why we stage all 5 tensors (and not just the truth image):
Metal-artifact reduction approaches need access to both the corrupted
sinogram (projection-domain MAR: NMAR, FSMAR, dual-domain) and the
corrupted image (image-domain MAR: U-Net inpainting baselines), with
the clean sinogram + truth image as supervision targets. The earlier
"truth-only" staging assumed the harness would forward-project the
truth to recreate the corrupted sinogram — but a simple monoenergetic
projector cannot reproduce the beam-hardening / photon-starvation /
scatter physics that *cause* the metal artifact in the first place.
The challenge-provided corrupted sino IS the input MAR methods are
designed to consume.

Staged layout (5 HDF5 files per split + 1 sidecar JSON):
    ct_mar/
        staged/
            train_truth.h5           (N, 512, 512)   float32   # nometal_img, mu (mm^-1)
            train_sino_clean.h5      (N, 900, 1000)  float32   # nometal_sino (raw)
            train_img_corrupted.h5   (N, 512, 512)   float32   # metalart_img, mu (mm^-1)
            train_sino_corrupted.h5  (N, 900, 1000)  float32   # metalart_sino (raw)
            train_metal_mask.h5      (N, 512, 512)   uint8     # 0/1 metal segmentation
            train_metalinfo.json                              # [{case_id, ...}] aggregated
            val_*.h5 / val_metalinfo.json
            test_*.h5 / test_metalinfo.json
            manifest.json

Run from the cluster:
    python data/fetch_ct_mar.py
Dry plan:
    python data/fetch_ct_mar.py --dry-run
"""

from __future__ import annotations
import argparse
import json
import re
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

GEOMETRY = {
    # CT-MAR's native fan-beam geometry from the README. Documented for
    # provenance; the harness re-projects truth through its OWN sparse-view
    # geometry at train time when running the dl_sparse_view challenge, but
    # for MAR work the staged sino_corrupted IS the input — the harness
    # should consume it directly without re-projection.
    "image_size": 512,
    "pixel_spacing_body_mm": 400.0 / 512,    # 400 mm FOV / 512 pixels
    "pixel_spacing_head_mm": 220.16 / 512,
    "n_angles": 900,
    "n_det": 1000,
    "det_layout": "fan-beam, equally-spaced over 360 degrees, linear detector",
    "sino_units": "raw float32 from challenge (line integrals; CT-MAR convention)",
    "img_units": "mu (mm^-1) after HU->mu at staging time (water=0.02)",
}

# 6 body tars available (body1..body6 each ~1000 cases). With all 5 tensors
# per case the per-case storage is ~10 MB raw, ~5-7 MB after lz4. Limits
# below stage 6000 total -> ~35-45 GB on disk.
DEFAULT_SOURCES = {
    "train": ["body1.tar.gz", "body2.tar.gz", "body3.tar.gz", "body4.tar.gz"],
    "val":   ["body5.tar.gz"],
    "test":  ["body6.tar.gz"],
}
DEFAULT_LIMITS = {"train": 4000, "val": 1000, "test": 1000}


# Filename patterns inside each tar.
_RX = {
    "nometal_img":   re.compile(r"^(?:body|head)\d+/Target/training_(?:body|head)_nometal_img(\d+)_512x512x1\.raw$"),
    "nometal_sino":  re.compile(r"^(?:body|head)\d+/Target/training_(?:body|head)_nometal_sino(\d+)_900x1000\.raw$"),
    "metalart_img":  re.compile(r"^(?:body|head)\d+/Baseline/training_(?:body|head)_metalart_img(\d+)_512x512x1\.raw$"),
    "metalart_sino": re.compile(r"^(?:body|head)\d+/Baseline/training_(?:body|head)_metalart_sino(\d+)_900x1000\.raw$"),
    "metal_mask":    re.compile(r"^(?:body|head)\d+/Mask/training_(?:body|head)_metalonlymask_img(\d+)_512x512x1\.raw$"),
    "metalinfo":     re.compile(r"^(?:body|head)\d+/Mask/training_(?:body|head)_metalinfo(\d+)\.json$"),
}

MU_WATER_PER_MM = 0.02   # shared with mayo_ldct + ddssl_ldct.phantoms


def _hu_to_mu(hu: np.ndarray) -> np.ndarray:
    """Hounsfield Units -> linear attenuation (mm^-1), water = 0.02 mm^-1."""
    return MU_WATER_PER_MM * (1.0 + hu.astype(np.float32) / 1000.0)


def _iter_cases(tar_path: Path, limit: int):
    """Yield dicts with all 5 tensors + metal_info for each case in a tar.

    Single-pass through the tar (tar archives are sequential) — for each
    member, we add it to a per-case staging dict keyed on the case id;
    when all 6 file types for a case are present, we yield the case.
    """
    pending: dict[int, dict] = {}
    yielded = 0
    with tarfile.open(tar_path, "r:gz") as tf:
        for member in tf:
            if yielded >= limit:
                return
            if not member.isfile():
                continue
            kind = None
            cid = None
            for k, rx in _RX.items():
                m = rx.match(member.name)
                if m:
                    kind, cid = k, int(m.group(1))
                    break
            if kind is None:
                continue
            buf = tf.extractfile(member)
            if buf is None:
                continue
            raw = buf.read()
            if kind == "metalinfo":
                pending.setdefault(cid, {})[kind] = json.loads(raw)
            elif kind in ("nometal_img", "metalart_img", "metal_mask"):
                arr = np.frombuffer(raw, dtype="<f4").reshape(512, 512).copy()
                pending.setdefault(cid, {})[kind] = arr
            else:  # sinograms
                arr = np.frombuffer(raw, dtype="<f4").reshape(900, 1000).copy()
                pending.setdefault(cid, {})[kind] = arr

            # If we have all 6 parts for this case, emit + free memory.
            if len(pending[cid]) == len(_RX):
                case = pending.pop(cid)
                yield cid, case
                yielded += 1


def _stage_split(staged_dir: Path, raw_dir: Path, split: str,
                 tar_names: list[str], limit: int, *,
                 shuffle_seed: int) -> tuple[int, list[dict]]:
    """Stage one split: stream cases out of the tars, write 5 HDF5 files +
    return the (count, aggregated_metalinfo) so a sidecar JSON can be written.
    """
    cases: list[tuple[int, dict]] = []
    for tar_name in tar_names:
        tp = raw_dir / tar_name
        if not tp.exists():
            print(f"[stage] missing {tp}, skipping.", flush=True)
            continue
        per_tar = max(1, limit // max(1, len(tar_names)))
        print(f"[stage] streaming {tar_name} (up to {per_tar} cases)…",
              flush=True)
        for cid, case in _iter_cases(tp, per_tar):
            cases.append((cid, case))
            if len(cases) % 100 == 0:
                print(f"[stage]   {tar_name}: {len(cases)} cases so far",
                      flush=True)
            if len(cases) >= limit:
                break
        if len(cases) >= limit:
            break

    if not cases:
        raise RuntimeError(f"No cases for split={split!r}")

    rng = np.random.default_rng(np.uint64(shuffle_seed))
    perm = rng.permutation(len(cases))
    cases_ordered = [cases[i] for i in perm]
    n = len(cases_ordered)

    def emit(key, transform=lambda x: x):
        for i, (_cid, c) in enumerate(cases_ordered):
            yield i, transform(c[key])

    # Truth image (clean) - convert HU -> mu
    pack_h5(staged_dir / f"{split}_truth.h5",
            name="image", shape=(n, 512, 512), dtype="float32",
            cases=emit("nometal_img", _hu_to_mu))
    # Corrupted image - convert HU -> mu
    pack_h5(staged_dir / f"{split}_img_corrupted.h5",
            name="image", shape=(n, 512, 512), dtype="float32",
            cases=emit("metalart_img", _hu_to_mu))
    # Clean sinogram - raw float32 (challenge units, line integrals)
    pack_h5(staged_dir / f"{split}_sino_clean.h5",
            name="sino", shape=(n, 900, 1000), dtype="float32",
            cases=emit("nometal_sino"))
    # Corrupted sinogram - raw float32
    pack_h5(staged_dir / f"{split}_sino_corrupted.h5",
            name="sino", shape=(n, 900, 1000), dtype="float32",
            cases=emit("metalart_sino"))
    # Binary metal segmentation - cast float32 mask to uint8 (it's 0/1)
    pack_h5(staged_dir / f"{split}_metal_mask.h5",
            name="mask", shape=(n, 512, 512), dtype="uint8",
            cases=emit("metal_mask",
                       lambda m: (m > 0.5).astype("uint8")))

    # Aggregated metal_info sidecar JSON
    metalinfo = [
        {"case_id": cid, **(c.get("metalinfo") or {})}
        for cid, c in cases_ordered
    ]
    (staged_dir / f"{split}_metalinfo.json").write_text(
        json.dumps(metalinfo, indent=2))

    return n, metalinfo


def stage_h5(raw_dir: Path, staged_dir: Path, *,
             sources: dict | None = None,
             limits: dict | None = None,
             shuffle_seed: int = 20260518) -> dict:
    staged_dir.mkdir(parents=True, exist_ok=True)
    sources = sources or DEFAULT_SOURCES
    limits = limits or DEFAULT_LIMITS
    splits: dict[str, int] = {}
    for split, tar_names in sources.items():
        n, _info = _stage_split(staged_dir, raw_dir, split, tar_names,
                                limits.get(split, 1000),
                                shuffle_seed=shuffle_seed + (hash(split) & 0xFFFF))
        splits[split] = n
        print(f"[stage] {split}: {n} cases  (5 HDF5 + 1 JSON)", flush=True)
    return splits


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", default=None)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--force", action="store_true",
                   help="Re-stage even if staged/manifest.json exists "
                        "(overwrites existing HDF5 files in place).")
    args = p.parse_args(argv)

    root = data_root(args.data_root)
    challenge_dir = root / CHALLENGE
    raw_dir = challenge_dir / "raw"
    staged_dir = challenge_dir / "staged"

    print(f"[plan] AGENT4CT_DATA = {root}")
    print(f"[plan] challenge dir = {challenge_dir}")
    print(f"[plan] raw    = {raw_dir} (body*.tar.gz already on cluster)")
    print(f"[plan] staged = {staged_dir} (~35-45 GB after lz4)")
    print(f"[plan] limits = {DEFAULT_LIMITS}")
    if args.dry_run:
        return 0

    # Conservative: need ~50 GB headroom (peak during writing).
    assert_budget(root, need_gb=50.0, reserve_gb=200.0)

    if not raw_dir.exists() or not any(raw_dir.glob("body*.tar.gz")):
        print(f"[stage] no body*.tar.gz in {raw_dir} — see header docstring.",
              file=sys.stderr)
        return 1

    if (staged_dir / "manifest.json").exists() and not args.force:
        print(f"[stage] {staged_dir}/manifest.json present — pass --force "
              f"to re-stage. (Existing truth-only staging will be overwritten.)")
        return 0

    splits = stage_h5(raw_dir, staged_dir)
    write_manifest(
        staged_dir,
        source="https://rpi.app.box.com/s/7p8tkqj5ewhtdad2h8kx975i9qg6b7a4",
        geometry=GEOMETRY,
        splits=splits,
        extra={
            "layout": (
                "5 HDF5 per split + sidecar metalinfo.json. "
                "Order per file is consistent across files (same shuffle perm)."
            ),
            "file_kinds": {
                "truth":         "nometal_img -> mu (mm^-1) via HU->mu",
                "img_corrupted": "metalart_img -> mu (mm^-1) via HU->mu",
                "sino_clean":    "nometal_sino (raw float32, line integrals)",
                "sino_corrupted":"metalart_sino (raw float32, line integrals)",
                "metal_mask":    "metalonlymask_img cast to uint8 (0/1)",
                "metalinfo":     "per-case metal-implant metadata (JSON list)",
            },
            "sources_per_split": DEFAULT_SOURCES,
            "limits_per_split": DEFAULT_LIMITS,
            "citation": (
                "Yu, Zhang, Pan, et al. AAPM 2024 CT-MAR Grand Challenge "
                "(RPI / Mayo). See README_training_data.txt in raw/."
            ),
        },
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
