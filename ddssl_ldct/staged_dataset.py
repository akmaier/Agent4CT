"""Read pentathlon challenge data from the staged HDF5 layout.

Layout (produced by `data/fetch_<challenge>.py`):

    data/<challenge>/staged/
        train_sinograms.h5      # dataset 'sino', shape (N_pool, A, D), float32
                                #   (omitted when the harness forward-projects
                                #   from truth on the fly — see StagedTruthDataset)
        train_truth.h5          # dataset 'image', shape (N_pool, H, W), float32
        val_*.h5
        test_*.h5
        manifest.json           # {geometry, splits, files[], fetched_at_utc, ...}

`N_pool` is the FULL training pool (typically much larger than `train_n=400`,
the per-epoch budget for 5-min iter runs). `RotatingSubsetDataset` rotates
which `train_n` samples each epoch sees so that, across many epochs, the
agent visits the whole pool.

Usage (truth + stored sinograms — e.g. CT-MAR which ships both):

    from ddssl_ldct.staged_dataset import (
        StagedH5Dataset, RotatingSubsetDataset, FanBeamGeometryFromManifest,
    )
    train_pool = StagedH5Dataset(
        root=Path("/cluster/maier/Agent4CT/data/ct_mar/staged"),
        split="train",                              # full pool
    )
    train = RotatingSubsetDataset(train_pool, n_per_epoch=400, seed=cfg["seed"])
    geom = FanBeamGeometryFromManifest(train_pool.manifest_path)
    for ep in range(cfg["epochs"]):
        train.set_epoch(ep)                         # rotates the subset
        for batch in DataLoader(train, batch_size=cfg["batch_size"]):
            ...

Usage (truth-only — Mayo / DL-Spectral / any image source where the harness
forward-projects through the challenge geometry on the fly):

    pool = StagedTruthDataset(root=..., split="train")
    train = RotatingSubsetDataset(pool, n_per_epoch=400, seed=cfg["seed"])

Design notes (cf. docs/performance.md):

- HDF5 files are opened LAZILY in each DataLoader worker, not in `__init__`.
  Opening in `__init__` would share the file handle across forked workers
  and crash on parallel reads.
- Chunks are `(1, A, D)` / `(1, H, W)`, so each `__getitem__` is a single
  chunk read — O(1) random access on the page cache.

Reproducibility contract for agents using this loader
-----------------------------------------------------
1. **Fix `cfg["seed"]`.** The same seed must yield the same per-epoch
   subset on every machine. `RotatingSubsetDataset` derives the per-epoch
   RNG state from `(seed, epoch)` — no global RNG calls.
2. **Call `train.set_epoch(ep)` once per epoch BEFORE the inner DataLoader
   loop.** Otherwise every epoch sees the same subset.
3. **Do NOT shuffle inside `RotatingSubsetDataset` and also inside
   DataLoader.** The rotation already drew a fresh subset; let DataLoader
   shuffle WITHIN that subset if you want batch-level diversity (pass a
   torch.Generator seeded from `(cfg["seed"], ep)` for full reproducibility).
4. **`n_per_epoch <= len(pool)` always.** Staging must produce pools
   large enough for the configured `train_n`; otherwise the rotation
   degenerates to drawing the same set every epoch.
5. **Pool ordering is the contract; don't rely on slice/patient locality.**
   Stagers may reshuffle their inputs before packing to break per-patient
   clustering — sample 0 and sample 1 are NOT guaranteed adjacent slices.
"""
from __future__ import annotations
import json
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

try:
    import h5py
except ImportError as e:
    raise ImportError(
        "h5py is required for StagedH5Dataset. Install with: pip install h5py"
    ) from e

# Register the lz4 HDF5 filter at import time so files packed with
# hdf5plugin LZ4 (the staging default) are readable. Without this, reads
# raise OSError "can't open directory (/usr/local/lib/plugin)".
try:
    import hdf5plugin                                                # noqa: F401
except ImportError:
    pass

from .geometry import FanBeamGeometry


class StagedH5Dataset(Dataset):
    """Sinogram + truth pairs from a staged HDF5 dataset.

    Returns `(sino, truth)` per item where:
      - sino:  float32 tensor, shape (A, D)
      - truth: float32 tensor, shape (H, W)

    The harness pipeline wraps each into batch + channel dims as needed.
    """

    def __init__(self, root: Path, split: str = "train",
                 n: int | None = None, *, dtype=torch.float32):
        self.root = Path(root)
        self.split = split
        self.dtype = dtype

        self.manifest_path = self.root / "manifest.json"
        if not self.manifest_path.exists():
            raise FileNotFoundError(
                f"No manifest.json in {self.root}. Run "
                f"`python data/fetch_<challenge>.py` first."
            )
        self.manifest = json.loads(self.manifest_path.read_text())
        self.geometry: dict = self.manifest["geometry"]
        splits: dict = self.manifest["splits"]
        if split not in splits:
            raise ValueError(
                f"split={split!r} not in manifest splits {list(splits.keys())}"
            )
        self.n_total = splits[split]
        self.n = self.n_total if n is None else min(int(n), self.n_total)

        self._sino_path = self.root / f"{split}_sinograms.h5"
        self._truth_path = self.root / f"{split}_truth.h5"
        for p in (self._sino_path, self._truth_path):
            if not p.exists():
                raise FileNotFoundError(p)

        # Worker-local file handles, opened lazily.
        self._sino_f: h5py.File | None = None
        self._truth_f: h5py.File | None = None

    def _open(self) -> Tuple[h5py.File, h5py.File]:
        if self._sino_f is None:
            self._sino_f = h5py.File(self._sino_path, "r", libver="latest")
            self._truth_f = h5py.File(self._truth_path, "r", libver="latest")
        return self._sino_f, self._truth_f

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, i: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if i < 0 or i >= self.n:
            raise IndexError(i)
        sf, tf = self._open()
        sino_ds = sf["sino"] if "sino" in sf else next(iter(sf.values()))
        img_ds = tf["image"] if "image" in tf else next(iter(tf.values()))
        sino = torch.from_numpy(np.asarray(sino_ds[i])).to(self.dtype)
        truth = torch.from_numpy(np.asarray(img_ds[i])).to(self.dtype)
        return sino, truth


class StagedTruthDataset(Dataset):
    """Truth-only variant: returns just the ground-truth image per item.

    Use this with datasets whose challenge geometry differs from the
    sparse-view geometry our solvers train against — the harness
    forward-projects each truth through `FanBeamGeometryFromManifest` at
    train time, simulating its own noisy sinograms. Mayo and DL-Spectral
    are typical examples.
    """

    def __init__(self, root: Path, split: str = "train",
                 *, dtype=torch.float32):
        self.root = Path(root)
        self.split = split
        self.dtype = dtype

        self.manifest_path = self.root / "manifest.json"
        if not self.manifest_path.exists():
            raise FileNotFoundError(
                f"No manifest.json in {self.root}. Run "
                f"`python data/fetch_<challenge>.py` first."
            )
        self.manifest = json.loads(self.manifest_path.read_text())
        self.geometry: dict = self.manifest["geometry"]
        splits: dict = self.manifest["splits"]
        if split not in splits:
            raise ValueError(
                f"split={split!r} not in manifest splits {list(splits.keys())}"
            )
        self.n_total = splits[split]

        self._truth_path = self.root / f"{split}_truth.h5"
        if not self._truth_path.exists():
            raise FileNotFoundError(self._truth_path)
        self._truth_f: h5py.File | None = None

    def _open(self) -> h5py.File:
        if self._truth_f is None:
            self._truth_f = h5py.File(self._truth_path, "r", libver="latest")
        return self._truth_f

    def __len__(self) -> int:
        return self.n_total

    def __getitem__(self, i: int) -> torch.Tensor:
        if i < 0 or i >= self.n_total:
            raise IndexError(i)
        tf = self._open()
        img_ds = tf["image"] if "image" in tf else next(iter(tf.values()))
        return torch.from_numpy(np.asarray(img_ds[i])).to(self.dtype)


class RotatingSubsetDataset(Dataset):
    """Wraps a pool dataset and exposes a different `n_per_epoch` subset
    per epoch, drawn deterministically from `(seed, epoch)`.

    The subset is selected without replacement WITHIN an epoch and is
    independent across epochs. Two runs with the same `seed` produce
    identical sequences of subsets — this is the reproducibility contract
    the harness depends on.

    Notes
    -----
    * `set_epoch(ep)` must be called before each epoch's DataLoader pass.
      The constructor sets `epoch=0`.
    * The wrapped `base` is consulted lazily by index; works with both
      `StagedH5Dataset` and `StagedTruthDataset`.
    * If `n_per_epoch >= len(base)`, every epoch reuses the whole pool
      in a permuted order (still seeded), and the warning is logged once.
    """

    def __init__(self, base: Dataset, n_per_epoch: int, seed: int):
        self.base = base
        self.n_per_epoch = int(n_per_epoch)
        self.seed = int(seed)
        self._pool_size = len(base)
        if self.n_per_epoch > self._pool_size:
            print(
                f"[RotatingSubsetDataset] n_per_epoch={self.n_per_epoch} > "
                f"pool size={self._pool_size}; falling back to whole-pool "
                f"permutation per epoch.",
                flush=True,
            )
        self.epoch = 0
        self._idx: np.ndarray | None = None
        self._refresh()

    def set_epoch(self, ep: int) -> None:
        self.epoch = int(ep)
        self._refresh()

    def _refresh(self) -> None:
        rng = np.random.default_rng(np.uint64(self.seed * 1_000_003 + self.epoch))
        if self.n_per_epoch >= self._pool_size:
            self._idx = rng.permutation(self._pool_size)
        else:
            self._idx = rng.choice(self._pool_size, size=self.n_per_epoch,
                                   replace=False)

    def __len__(self) -> int:
        return int(min(self.n_per_epoch, self._pool_size))

    def __getitem__(self, i):
        assert self._idx is not None
        return self.base[int(self._idx[i])]

    @property
    def pool_size(self) -> int:
        return self._pool_size


def FanBeamGeometryFromManifest(manifest_path: Path, *,
                                device=None) -> FanBeamGeometry:
    """Build a `FanBeamGeometry` from the geometry block in a staged manifest.

    Keeps the geometry definition in one place (the manifest produced at
    fetch time) and avoids re-typing it in every solver.
    """
    m = json.loads(Path(manifest_path).read_text())
    g = m["geometry"]
    return FanBeamGeometry(
        image_size=int(g["image_size"]),
        pixel_spacing=float(g["pixel_spacing"]),
        n_angles=int(g["n_angles"]),
        n_det=int(g["n_det"]),
        det_spacing=float(g["det_spacing"]),
        sod=float(g["sod"]),
        sdd=float(g["sdd"]),
    )
