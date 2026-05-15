"""Shared helpers for pentathlon data fetchers."""

from __future__ import annotations
import hashlib
import json
import os
import shutil
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Iterator

try:
    import h5py
    _HAS_H5PY = True
except ImportError:
    _HAS_H5PY = False


def data_root(override: str | None = None) -> Path:
    """Root for all staged datasets. Override > env > default."""
    if override:
        return Path(override).expanduser().resolve()
    env = os.environ.get("AGENT4CT_DATA")
    if env:
        return Path(env).expanduser().resolve()
    return Path("/cluster/maier/Agent4CT/data")


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            buf = f.read(chunk)
            if not buf:
                break
            h.update(buf)
    return h.hexdigest()


def download(url: str, dst: Path, *, expected_sha256: str | None = None,
             resume: bool = True) -> None:
    """Download `url` to `dst`. If `expected_sha256` is given, verify after."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and expected_sha256 and sha256_file(dst) == expected_sha256:
        print(f"[fetch] {dst.name} already present + checksum ok — skip.")
        return
    tmp = dst.with_suffix(dst.suffix + ".part")
    mode = "ab" if (resume and tmp.exists()) else "wb"
    headers = {}
    if mode == "ab":
        headers["Range"] = f"bytes={tmp.stat().st_size}-"
    req = urllib.request.Request(url, headers=headers)
    t0 = time.time()
    bytes_seen = tmp.stat().st_size if mode == "ab" else 0
    with urllib.request.urlopen(req) as resp, tmp.open(mode) as out:
        while True:
            buf = resp.read(1 << 20)
            if not buf:
                break
            out.write(buf)
            bytes_seen += len(buf)
            if bytes_seen % (50 << 20) < (1 << 20):
                mb = bytes_seen / (1 << 20)
                rate = mb / max(time.time() - t0, 1e-3)
                print(f"[fetch]   {mb:.0f} MB  ({rate:.1f} MB/s)", flush=True)
    tmp.rename(dst)
    if expected_sha256:
        got = sha256_file(dst)
        if got != expected_sha256:
            dst.unlink()
            raise RuntimeError(
                f"checksum mismatch for {dst.name}: want {expected_sha256}, got {got}"
            )
    print(f"[fetch] {dst.name} ok ({bytes_seen / 1e9:.2f} GB)")


def pack_h5(out: Path, name: str, shape: tuple, dtype: str,
            cases: Iterator, *, compression: str = "lz4") -> None:
    """Pack an iterator of arrays into a single HDF5 dataset.

    `cases` yields `(idx, array)` pairs where idx is the sample index and
    array has shape == shape[1:] (the per-sample part of `shape`).
    """
    if not _HAS_H5PY:
        raise RuntimeError("h5py is required: pip install h5py")
    out.parent.mkdir(parents=True, exist_ok=True)
    chunks = (1,) + shape[1:]
    # lz4 only available with hdf5plugin; fall back to gzip-1
    try:
        import hdf5plugin                                # noqa: F401
        comp_opts = {"compression": 32004, "compression_opts": (0,)} \
            if compression == "lz4" else {"compression": "gzip",
                                          "compression_opts": 1}
    except ImportError:
        comp_opts = {"compression": "gzip", "compression_opts": 1}
    with h5py.File(out, "w", libver="latest") as f:
        ds = f.create_dataset(name, shape=shape, dtype=dtype, chunks=chunks,
                              **comp_opts)
        n_written = 0
        for idx, arr in cases:
            ds[idx] = arr
            n_written += 1
            if n_written % 100 == 0:
                print(f"[pack]   {n_written}/{shape[0]} -> {out.name}",
                      flush=True)
    print(f"[pack] {out.name} ok ({shape[0]} cases, "
          f"{out.stat().st_size / 1e9:.2f} GB on disk)")


def write_manifest(out_dir: Path, *, source: str, geometry: dict,
                   splits: dict, extra: dict | None = None) -> None:
    """Write staged/manifest.json with checksums of every file in `out_dir`."""
    files = sorted(p for p in out_dir.glob("*.h5"))
    manifest = {
        "source": source,
        "geometry": geometry,
        "splits": splits,
        "files": [
            {
                "name": p.name,
                "bytes": p.stat().st_size,
                "sha256": sha256_file(p),
            }
            for p in files
        ],
        "fetched_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    if extra:
        manifest.update(extra)
    out = out_dir / "manifest.json"
    out.write_text(json.dumps(manifest, indent=2))
    print(f"[manifest] {out} ({len(files)} h5 files)")


def disk_free_gb(path: Path) -> float:
    """Free space in GB on the filesystem containing `path`."""
    stat = shutil.disk_usage(path if path.exists() else path.parent)
    return stat.free / 1e9


def assert_budget(target_root: Path, need_gb: float, *,
                  reserve_gb: float = 200.0) -> None:
    """Refuse to start if pulling `need_gb` would dip below `reserve_gb` free."""
    free = disk_free_gb(target_root)
    if free - need_gb < reserve_gb:
        raise RuntimeError(
            f"Aborting: need {need_gb:.0f} GB but only {free:.0f} GB free at "
            f"{target_root} (must leave >={reserve_gb:.0f} GB headroom)."
        )
    print(f"[budget] {free:.0f} GB free at {target_root}, "
          f"need {need_gb:.0f} GB — ok.")
