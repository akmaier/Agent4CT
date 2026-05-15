"""Verify a staged dataset against its manifest.json.

Usage:
    python data/verify_staged.py <challenge>            # default data root
    python data/verify_staged.py <challenge> --data-root /path
"""

from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

from _common import data_root, sha256_file

try:
    import h5py
except ImportError:
    h5py = None


def verify(challenge: str, root: Path) -> int:
    staged = root / challenge / "staged"
    manifest_path = staged / "manifest.json"
    if not manifest_path.exists():
        print(f"FAIL: {manifest_path} not found.", file=sys.stderr)
        return 1
    m = json.loads(manifest_path.read_text())
    ok = True
    for entry in m["files"]:
        p = staged / entry["name"]
        if not p.exists():
            print(f"FAIL: missing {p}")
            ok = False
            continue
        got = sha256_file(p)
        if got != entry["sha256"]:
            print(f"FAIL: sha256 mismatch on {entry['name']}")
            print(f"      want {entry['sha256']}")
            print(f"      got  {got}")
            ok = False
            continue
        if h5py is not None and p.suffix == ".h5":
            with h5py.File(p, "r") as f:
                shapes = {k: f[k].shape for k in f.keys()}
            print(f"OK:   {entry['name']}  {shapes}")
        else:
            print(f"OK:   {entry['name']}")
    if ok:
        print(f"\nverify_staged({challenge}): all {len(m['files'])} files match.")
    return 0 if ok else 2


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("challenge", help="e.g. dl_sparse_view, htc2022, lodopab")
    p.add_argument("--data-root", default=None)
    args = p.parse_args(argv)
    return verify(args.challenge, data_root(args.data_root))


if __name__ == "__main__":
    sys.exit(main())
