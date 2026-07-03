"""Re-partition the staged Breast-CT (Sidky DL-Sparse-View) data into a
Mayo-style train / val / test split.

Context (paper_CTautoresearch.md §5.0): the public Zenodo release only ships
the 4000-case TRAIN set (Sidky & Pan 2022's official 10-case val + 100-case
test have their truth withheld). Our pipeline had staged it as train 3600 /
val 400 with NO held-out test set, so breast could only be val-ranked.

Decision (user, 2026-07-03): give breast a real held-out test set by splitting
the existing 400-case val POOL into val 200 + test 200. Train (3600) is left
completely untouched, so nothing about the training distribution changes — this
only carves a reporting test set out of the former val pool.

The split is DETERMINISTIC (fixed seed) and DISJOINT, recorded case-by-case in
breast_splits.json so it is reproducible forever (mirrors Mayo's WAGNER_SPLITS).

Idempotent + non-destructive: the original 400-case files are preserved as
valpool400_*.h5; re-running rebuilds val_/test_ from that pool.

Run on the cluster:
    cd /cluster/maier/Agent4CT && source .venv/bin/activate
    python scripts/restage_breast_valtest_split.py \
        --staged data/dl_sparse_view/staged
    python scripts/restage_breast_valtest_split.py --staged ... --verify-only
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path

import numpy as np

os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")
try:
    # staged breast h5 datasets are written with a blosc/zstd filter; hdf5plugin
    # registers it so h5py can read them (see lme_cluster_agent_guide.md).
    import hdf5plugin  # noqa: F401
except Exception:
    pass
import h5py  # noqa: E402

SEED = 20260703
N_POOL = 400
N_VAL = 200
N_TEST = 200

# (staged file stem, hdf5 dataset key)
KINDS = [
    ("truth", "image"),
    ("sinograms", "sino"),
    ("fbp128", "image"),
]


def _pool_path(staged: Path, stem: str) -> Path:
    """The preserved 400-case pool file (created on first run from val_*)."""
    return staged / f"valpool400_{stem}.h5"


def _ensure_pool(staged: Path) -> None:
    """On first run, snapshot the original 400-case val_* files as the pool."""
    for stem, _ in KINDS:
        pool = _pool_path(staged, stem)
        orig = staged / f"val_{stem}.h5"
        if pool.exists():
            continue
        if not orig.exists():
            raise FileNotFoundError(
                f"neither pool {pool.name} nor original {orig.name} present")
        # hard-link is cheapest and keeps a true immutable snapshot; fall back
        # to a copy across filesystems.
        try:
            os.link(orig, pool)
        except OSError:
            import shutil
            shutil.copy2(orig, pool)
        print(f"  snapshot {orig.name} -> {pool.name}")


def compute_split() -> tuple[list[int], list[int]]:
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(N_POOL)
    val_idx = sorted(int(i) for i in perm[:N_VAL])
    test_idx = sorted(int(i) for i in perm[N_VAL:N_VAL + N_TEST])
    return val_idx, test_idx


def write_split(staged: Path, val_idx: list[int], test_idx: list[int]) -> None:
    for stem, key in KINDS:
        pool = _pool_path(staged, stem)
        with h5py.File(pool, "r") as f:
            data = f[key]
            assert data.shape[0] == N_POOL, (
                f"{pool.name}: expected {N_POOL} cases, got {data.shape[0]}")
            val_arr = data[val_idx]
            test_arr = data[test_idx]
        for split, arr in (("val", val_arr), ("test", test_arr)):
            out = staged / f"{split}_{stem}.h5"
            tmp = out.with_suffix(".h5.tmp")
            with h5py.File(tmp, "w") as f:
                f.create_dataset(key, data=arr, dtype="float32")
            os.replace(tmp, out)
            print(f"  wrote {out.name}: {arr.shape} [{key}]")


def write_manifest(staged: Path, val_idx: list[int], test_idx: list[int]) -> Path:
    manifest = {
        "dataset": "breast_ct",
        "source": "Sidky DL-Sparse-View 4000-case public train set (Zenodo 14173522)",
        "note": "train 3600 untouched; former 400-case val POOL split into "
                "val 200 + test 200 (indices are into the 0..399 val pool).",
        "seed": SEED,
        "n_pool": N_POOL,
        "n_val": N_VAL,
        "n_test": N_TEST,
        "val_pool_indices": val_idx,
        "test_pool_indices": test_idx,
    }
    out = staged / "breast_splits.json"
    out.write_text(json.dumps(manifest, indent=1))
    return out


def verify(staged: Path, val_idx: list[int], test_idx: list[int]) -> None:
    # disjoint + full cover of the pool
    sv, st = set(val_idx), set(test_idx)
    assert len(sv) == N_VAL and len(st) == N_TEST, "duplicate indices"
    assert sv.isdisjoint(st), "val/test overlap!"
    assert sv | st == set(range(N_POOL)), "val+test do not cover the 400 pool"
    # on-disk shapes
    for stem, key in KINDS:
        for split, n in (("val", N_VAL), ("test", N_TEST)):
            p = staged / f"{split}_{stem}.h5"
            with h5py.File(p, "r") as f:
                assert f[key].shape[0] == n, f"{p.name}: {f[key].shape[0]} != {n}"
    # spot-check: a val row equals its pool source row (no shuffle bug)
    stem, key = KINDS[0]
    with h5py.File(_pool_path(staged, stem), "r") as fp, \
         h5py.File(staged / f"val_{stem}.h5", "r") as fv, \
         h5py.File(staged / f"test_{stem}.h5", "r") as ft:
        assert np.array_equal(fp[key][val_idx[0]], fv[key][0]), "val row mismatch"
        assert np.array_equal(fp[key][test_idx[0]], ft[key][0]), "test row mismatch"
    print("VERIFY OK: disjoint, full-cover, shapes match, rows trace to pool.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--staged", type=Path,
                    default=Path("data/dl_sparse_view/staged"))
    ap.add_argument("--verify-only", action="store_true")
    args = ap.parse_args()
    staged = args.staged
    assert staged.is_dir(), f"no staged dir: {staged}"

    val_idx, test_idx = compute_split()
    print(f"split seed={SEED}: val={N_VAL} test={N_TEST} "
          f"(val[0:3]={val_idx[:3]} test[0:3]={test_idx[:3]})")

    if not args.verify_only:
        _ensure_pool(staged)
        write_split(staged, val_idx, test_idx)
        mp = write_manifest(staged, val_idx, test_idx)
        print(f"  wrote {mp.name}")
    verify(staged, val_idx, test_idx)


if __name__ == "__main__":
    main()
