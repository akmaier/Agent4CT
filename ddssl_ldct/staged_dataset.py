"""Read pentathlon challenge data from the staged HDF5 layout.

Layout (produced by `data/fetch_<challenge>.py`):

    data/<challenge>/staged/
        train_sinograms.h5      # dataset 'sino', shape (N, A, D), float32
        train_truth.h5          # dataset 'image', shape (N, H, W), float32
        val_*.h5
        test_*.h5
        manifest.json           # {geometry, splits, files[], fetched_at_utc}

Usage:

    from ddssl_ldct.staged_dataset import StagedH5Dataset, FanBeamGeometryFromManifest
    train = StagedH5Dataset(
        root=Path("/cluster/maier/Agent4CT/data/dl_sparse_view/staged"),
        split="train", n=400,
    )
    geom = FanBeamGeometryFromManifest(train.manifest_path, device=device)
    loader = DataLoader(train, batch_size=1, num_workers=4)

Design notes (cf. docs/performance.md):

- HDF5 files are opened LAZILY in each DataLoader worker, not in `__init__`.
  Opening in `__init__` would share the file handle across forked workers
  and crash on parallel reads.
- Chunks are `(1, A, D)` / `(1, H, W)`, so each `__getitem__` is a single
  chunk read — O(1) random access on the page cache.
- `n` selects the FIRST n cases deterministically. The harness uses a fixed
  seed and depends on consistent subset selection across iter and stage
  runs.
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
