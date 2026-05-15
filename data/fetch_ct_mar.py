"""Fetch + stage the AAPM CT-MAR 2024 challenge data.

Sources:
  * RPI Box (full challenge dataset):
      https://rpi.app.box.com/s/7p8tkqj5ewhtdad2h8kx975i9qg6b7a4
      The shared link is authenticated-only even though it looks
      public-shareable: every Box public-API endpoint (folder-zip download,
      shared_items, static) returns 401/404/512 without an OAuth bearer
      token. Programmatic download from this script requires one of:
        (a) Box developer access token (Settings -> Developer Apps on
            box.com, generate token, paste as $BOX_TOKEN env var), OR
        (b) Operator-side browser session: open the link, click
            "Download" to get the folder zip, upload to
            /cluster/maier/Agent4CT/data/ct_mar/raw/ and re-run with
            --skip-download.
  * github.com/xcist/example/tree/main/AAPM_datachallenge — public mirror
    with simulator code + a small set of example cases. Useful for
    bootstrapping the parser before the full Box dataset lands.

Size:   The full Box dataset is ~150-300 GB est. (14 000 cases x 5 tensors).
        The xcist/example mirror is < 1 GB (example cases only).

Layout produced:
    ct_mar/
        raw/
        staged/
            train_sinograms.h5  (N, A, D) float32  -- sino with metal
            train_clean.h5      (N, A, D) float32  -- sino without metal (target domain)
            train_truth.h5      (N, H, W) float32  -- clean image
            train_mask.h5       (N, H, W) uint8    -- metal pixel mask
            val_*.h5
            manifest.json
"""

from __future__ import annotations
import argparse
import subprocess
import sys
from pathlib import Path

from _common import (
    assert_budget,
    data_root,
    write_manifest,
)

CHALLENGE = "ct_mar"
MIRROR_REPO = "https://github.com/xcist/example.git"
MIRROR_SUBDIR = "AAPM_datachallenge"

GEOMETRY = {
    # XCIST CatSim default fan-beam geometry. Re-verify against the example
    # repo's parameter files.
    "image_size": 512,
    "pixel_spacing": 0.7,
    "n_angles": 1024,
    "n_det": 900,
    "det_spacing": 1.0,
    "sod": 541.0,
    "sdd": 949.075,
}

SPLIT_SIZES = {"train": None, "val": None}


def clone_mirror(raw_dir: Path) -> Path:
    """Shallow-clone xcist/example into raw_dir. Returns the subdir path."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    target = raw_dir / "xcist-example"
    if target.exists() and (target / ".git").exists():
        print(f"[fetch] {target} already cloned; pulling updates.")
        subprocess.run(["git", "-C", str(target), "pull", "--ff-only"],
                       check=True)
    else:
        print(f"[fetch] shallow clone {MIRROR_REPO} -> {target}")
        subprocess.run(
            ["git", "clone", "--depth", "1", MIRROR_REPO, str(target)],
            check=True,
        )
    subdir = target / MIRROR_SUBDIR
    if not subdir.exists():
        raise FileNotFoundError(
            f"{subdir} missing — the upstream repo layout changed."
        )
    return subdir


def stage_h5(raw_dir: Path, staged_dir: Path) -> None:
    raise NotImplementedError(
        "Per-challenge conversion not implemented yet. Read the README in "
        f"the cloned {MIRROR_SUBDIR}/ first to see whether the example "
        "ships full sinograms+masks or only pointers. If the full data "
        "needs a Box link, document that here and abort with a clear "
        "message asking the operator to supply credentials."
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
    print(f"[plan] mirror = {MIRROR_REPO} (size depends on what's hosted there)")
    if args.dry_run:
        return 0

    # Budget unknown ahead of time; assume worst case from the repo audit.
    assert_budget(root, need_gb=300.0, reserve_gb=200.0)

    if not args.skip_download:
        subdir = clone_mirror(raw_dir)
    else:
        subdir = raw_dir / "xcist-example" / MIRROR_SUBDIR
        if not subdir.exists():
            print("--skip-download but mirror not present.", file=sys.stderr)
            return 1

    if staged_dir.exists() and (staged_dir / "manifest.json").exists():
        print(f"[stage] {staged_dir}/manifest.json present — skip.")
    else:
        stage_h5(subdir, staged_dir)
        write_manifest(
            staged_dir,
            source=MIRROR_REPO,
            geometry=GEOMETRY,
            splits=SPLIT_SIZES,
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
