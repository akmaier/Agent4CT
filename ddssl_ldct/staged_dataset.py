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


# ---------------------------------------------------------------------------
# Thin helpers for the demo_dl_reference TPE pipeline (Track B/C in
# docs/workplan_real_datasets.md). The classes above are for training-loop
# DataLoaders; the helpers below load a whole val split into memory in one
# go, which is what the solvers' main() functions need.
#
# Geometry table keyed by dataset_kind. Each entry pins everything the
# search agent needs to override in CONFIG so the existing solvers can run
# against real staged data without changing per-solver constants.

import os as _os

# DatasetInfo lives in `challenges/_common.py` so the challenges folder is
# the single source of truth for per-challenge geometry. Re-exported here
# for backwards compat with `from ddssl_ldct.staged_dataset import DatasetInfo`.
from challenges._common import DatasetInfo  # noqa: F401 (re-export)
from challenges.demo_dl.geometry import GEOMETRY as _DEMO_DL
from challenges.dl_sparse_view.geometry import GEOMETRY as _DL_SPARSE_VIEW


_DEFAULT_DATA_ROOT = Path(_os.environ.get(
    "AGENT4CT_DATA", "/cluster/maier/Agent4CT/data"))


GEOMETRIES: dict[str, DatasetInfo] = {
    "phantoms":   _DEMO_DL,           # synthetic ellipse-phantom fallback
    "breast_ct":  _DL_SPARSE_VIEW,    # Sidky 2021 DL-Sparse-View challenge
    # mayo_ldct (helix2fan-rebinned 2D fan-beam, Wagner split). Real LD
    # sinograms paired with HD reconstructed truth images. Solvers
    # consume LOWDOSE sino as the input by convention (LDCT denoising
    # task); override `sino_file_tmpl` to fulldose if you want the
    # baseline noise floor instead. Path matches the existing
    # `data/mayo_ldct/staged/` directory (where the truth h5s live).
    # The aggregated per-split sino h5s are produced by
    # `data/stage_mayo_sinos.py` after the helix2fan bulk rebin.
    # Geometry from FanBeamGeometry.mayo_ldct_fitted() — the Powell 5-param
    # fit on L014 (job 762284). DO NOT replace with MAYO_LDCT_SSR_DEFAULTS:
    # this is the FBP-step geometry, not the SSR-step geometry; they're
    # independent and only the FBP one belongs here. See findings.md
    # 2026-05-27 "FBP sod ≠ SSR sod" for context.
    "mayo_ldct_2d": DatasetInfo(
        image_size=512, pixel_spacing=0.700857,    # was 0.5859375 (stale)
        n_angles=2304, n_det=736, det_spacing=1.285044,   # was 1.2858
        sod=595.362, sdd=1086.803,                  # was 595.0 / 1085.6 (DICOM)
        display_min=0.0, display_max=0.05,
        has_real_sino=True,
        # CANONICAL per-sample data (2026-06-14): lossless angle-rolled sinos +
        # native truth + per-slice ps. Uniform angle_start=0 so only ps varies;
        # load_val_split(..., return_ps=True) returns the per-slice ps and
        # solvers reconstruct per-sample via mayo_proj_cache(). pixel_spacing
        # below is just the default/fallback for the pipeline init.
        #
        # ⚠️ REBUILD THIS DIR WITH `data/stage_mayo_canonical.py` ONLY.
        # It writes the truth dataset keyed "truth" + the per-slice "ps" array
        # (ps_eff), patient-ordered, with sinos in the CANONICAL frame
        # (roll + u-flip + slab, per patient) that this uniform-angle loader
        # expects. Do NOT rebuild with `fetch_mayo_ldct.py` (writes key "image",
        # NO "ps", shuffled) or `stage_mayo_sinos.py` (legacy non-canonical
        # packing) — both produce data this loader silently mis-reads. To
        # re-create from the surviving raw/ + staged_helix2fan_v3/:
        #   python data/stage_mayo_canonical.py --force --validate --subdir staged_helix2fan_v3
        # (--validate FBPs val LD vs GT, expect SSIM ~0.81). See data/README.md.
        staged_dir=_DEFAULT_DATA_ROOT / "mayo_ldct" / "staged_canonical",
        sino_file_tmpl="{split}_sino_lowdose.h5",
    ),
}


def get_dataset_kind(cfg: dict | None = None) -> str:
    """Resolve dataset kind from env (`AGENT4CT_DATASET`) → cfg → default.
    Env wins so an sbatch wrapper can flip the dataset without editing JSON."""
    env = _os.environ.get("AGENT4CT_DATASET")
    if env:
        return env
    if cfg and "dataset_kind" in cfg:
        return cfg["dataset_kind"]
    return "phantoms"


def geometry_overrides(kind: str) -> dict:
    """Return the cfg-key overrides for the named dataset (image_size,
    pixel_spacing, n_angles, n_det, det_spacing, sod, sdd, display_min/max)."""
    info = GEOMETRIES[kind]
    return dict(
        image_size=info.image_size, pixel_spacing=info.pixel_spacing,
        n_angles=info.n_angles, n_det=info.n_det,
        det_spacing=info.det_spacing, sod=info.sod, sdd=info.sdd,
        display_min=info.display_min, display_max=info.display_max,
    )


def load_val_split(kind: str, split: str, n: int, *, device,
                   seed: int = 1042, noise_i0: float = 1e5,
                   noise_sigma_e: float = 10.0,
                   geom: FanBeamGeometry | None = None,
                   return_ps: bool = False
                   ) -> tuple:
    """Return (phantoms, clean, noisy) tensors on `device`.

    For kind="phantoms": synthetic ellipse phantoms + simulated low-dose sino
    (backwards-compatible with the existing solvers' build_dataset).
    For kind="breast_ct" or "mayo_ldct_2d": load truth and REAL sinograms
    from staged HDF5; `clean` is None (no clean sino available).

    Always returns `(N, 1, H, W)` for images and `(N, 1, A, D)` for sinos.
    """
    info = GEOMETRIES[kind]
    if not info.has_real_sino:
        # Backwards-compat synthetic path.
        from ddssl_ldct.phantoms import random_ellipses_phantom
        from ddssl_ldct.simulate import simulate_low_dose
        from ddssl_ldct.pyronn_projector import PyronnFanBeamProjector
        assert geom is not None, "phantoms kind needs a geom (built from CONFIG)"
        proj = PyronnFanBeamProjector(geom).to(device)
        phantoms = torch.stack([
            random_ellipses_phantom(size=geom.image_size, n_ellipses=10,
                                     seed=seed + i)[0]
            for i in range(n)
        ]).to(device)
        with torch.no_grad():
            clean = proj.forward_project(phantoms)
            noisy = simulate_low_dose(clean, i0=noise_i0,
                                       sigma_e=noise_sigma_e,
                                       seed=seed + 10_000)
        return (phantoms, clean, noisy, None) if return_ps else (phantoms, clean, noisy)

    assert info.staged_dir is not None, f"{kind!r} has no staged_dir"
    truth_path = info.staged_dir / info.truth_file_tmpl.format(split=split)
    sino_path  = info.staged_dir / info.sino_file_tmpl .format(split=split)
    # Presentation-only TEST showcase (AGENT4CT_SHOWCASE): replace the Mayo val
    # load (single patient L277) with one CENTRAL slice from each of the 5
    # held-out TEST patients, for diverse leaderboard figures. Only the val
    # call is redirected (split=="train" untouched) -> the model still TRAINS on
    # the train patients; these slices are reconstructed only for the figure.
    # Indices are the per-patient central slice in Wagner test order; boundaries
    # verified vs ps transitions (247/457) + counts (L014 154, L056 93, L058 210,
    # L075 137, L123 151 = 745).
    import os as _os
    # Presentation-only VAL+TEST showcase (AGENT4CT_SHOWCASE=valtest): 6 scenes =
    # L277 central (val) + the 5 test-patient centrals, full 512² (NO FOV mask).
    # Gated on the exact value "valtest" + early-return, so the live search metric
    # and the older test-only showcase ("1") are untouched.
    if _os.environ.get("AGENT4CT_SHOWCASE") == "valtest" and kind == "mayo_ldct_2d" and split == "val":
        sd = info.staged_dir
        _tk = lambda f: (info.truth_dataset if info.truth_dataset in f else "truth")
        _sk = lambda f: (info.sino_dataset if info.sino_dataset in f else ("sino" if "sino" in f else list(f.keys())[0]))
        def _rd(spl, idxs):
            with h5py.File(sd / info.truth_file_tmpl.format(split=spl), "r") as f:
                tr = f[_tk(f)][idxs]; pv = f["ps"][idxs] if "ps" in f else None
            with h5py.File(sd / info.sino_file_tmpl.format(split=spl), "r") as f:
                si = f[_sk(f)][idxs]
            return tr, si, pv
        with h5py.File(sd / info.truth_file_tmpl.format(split="val"), "r") as f:
            _nv = f[_tk(f)].shape[0]
        _vt, _vs, _vp = _rd("val", [_nv // 2])                      # L277 central
        _tt, _ts, _tp = _rd("test", [77, 200, 352, 525, 668])       # 5 test centrals
        truth = np.concatenate([_vt, _tt], 0); sino = np.concatenate([_vs, _ts], 0)
        ps_arr = (np.concatenate([_vp, _tp], 0) if (_vp is not None and _tp is not None) else None)
        truth_t = torch.from_numpy(np.ascontiguousarray(truth)).to(device=device, dtype=torch.float32)
        sino_t  = torch.from_numpy(np.ascontiguousarray(sino )).to(device=device, dtype=torch.float32)
        if truth_t.dim() == 3: truth_t = truth_t.unsqueeze(1)
        if sino_t.dim() == 3:  sino_t = sino_t.unsqueeze(1)
        print(f"[staged] SHOWCASE=valtest: L277-central(idx {_nv//2}) + 5 test-central "
              f"-> {truth_t.shape[0]} scenes (full 512, no FOV)", flush=True)
        if return_ps:
            return truth_t, sino_t, sino_t, ps_arr
        return truth_t, sino_t, sino_t
    _sel_override = None
    if _os.environ.get("AGENT4CT_SHOWCASE") and kind == "mayo_ldct_2d" and split == "val":
        truth_path = info.staged_dir / info.truth_file_tmpl.format(split="test")
        sino_path  = info.staged_dir / info.sino_file_tmpl .format(split="test")
        _sel_override = [77, 200, 352, 525, 668]   # central slice / test patient
        print(f"[staged] SHOWCASE: 5 central TEST slices {_sel_override} "
              f"(L014 L056 L058 L075 L123) in place of val", flush=True)
    if not truth_path.exists():
        raise FileNotFoundError(f"{kind} {split} truth missing: {truth_path}")
    if not sino_path.exists():
        raise FileNotFoundError(f"{kind} {split} sino missing: {sino_path}")
    with h5py.File(truth_path, "r") as f:
        tkey = info.truth_dataset if info.truth_dataset in f else "truth"   # canonical uses "truth"
        n_truth = f[tkey].shape[0]
        n_eff = min(n, n_truth)
        # Mayo SUBSET (the agentic loop's "200 stratified" train_n): the canonical
        # h5 is patient-ordered, so the first n_eff slices would be ~1 patient.
        # Round-robin across ps-groups (≈ per-patient display-FOV) so the subset
        # spans patients -> better generalisation to the held-out val patient.
        # Full-set (TPE: n_eff==n_truth) and single-ps val fall through to first-n.
        sel = _sel_override if _sel_override is not None else slice(0, n_eff)
        if _sel_override is None and kind == "mayo_ldct_2d" and "ps" in f and 0 < n_eff < n_truth:
            allps = np.round(np.asarray(f["ps"][:], dtype=float), 5)
            groups: dict = {}
            for i, p in enumerate(allps):
                groups.setdefault(float(p), []).append(i)
            order = sorted(groups)
            if len(order) > 1:   # only stratify when >1 group present
                picks, ptr = [], {p: 0 for p in order}
                while len(picks) < n_eff:
                    advanced = False
                    for p in order:
                        if ptr[p] < len(groups[p]):
                            picks.append(groups[p][ptr[p]]); ptr[p] += 1; advanced = True
                            if len(picks) >= n_eff:
                                break
                    if not advanced:
                        break
                sel = sorted(picks)
                print(f"[staged] mayo stratified subset: {n_eff}/{n_truth} across "
                      f"{len(order)} ps-groups {order}", flush=True)
            else:   # single ps-group (val = L277): evenly-space across the FULL
                    # volume, NOT the first-n boundary slices (top-of-volume,
                    # near-empty -> unrepresentative metric + repeated figure rows).
                sel = sorted(set(np.linspace(0, n_truth - 1, n_eff)
                                 .round().astype(int).tolist()))
                print(f"[staged] mayo val evenly-spaced: {len(sel)}/{n_truth} "
                      f"across the volume (was first-{n_eff} boundary)", flush=True)
        truth = f[tkey][sel][...]
        ps_arr = f["ps"][sel][...] if "ps" in f else None   # per-slice recon ps (canonical)
    with h5py.File(sino_path, "r") as f:
        sino = f[info.sino_dataset][sel][...]
    truth_t = torch.from_numpy(truth).to(device=device, dtype=torch.float32)
    sino_t  = torch.from_numpy(sino ).to(device=device, dtype=torch.float32)
    if truth_t.dim() == 3:   # (N, H, W) -> (N, 1, H, W)
        truth_t = truth_t.unsqueeze(1)
    if sino_t.dim() == 3:    # (N, A, D) -> (N, 1, A, D)
        sino_t = sino_t.unsqueeze(1)
    # Apply per-dataset sinogram start-angle shift to align the gantry-
    # rotation convention with PyronnFanBeamProjector's (angle 0 = source
    # at +x, CCW positive). For breast_ct this is +32 of 128 views.
    if info.sino_angle_shift != 0:
        sino_t = torch.roll(sino_t, shifts=int(info.sino_angle_shift), dims=-2)
    # Alias `clean = noisy` (SAME tensor, no copy) so downstream solvers that
    # compute a "noiseless reference" via `proj.fbp(val_clean)` for the
    # comparison figure don't NPE on real-sino datasets. There is no separate
    # clean measurement available; the reference panel shows FBP(real_sino)
    # twice. `clean` is only ever read (one FBP for the panel) — never mutated —
    # so sharing the tensor is safe and the metric pipeline (truth + pred only)
    # is unaffected. The previous `.clone()` doubled sino GPU memory; with the
    # all-slices Mayo footprint (train_n=579 + val_n=214) that clone is ~5 GB
    # and triggered CUDA OOM. Dropping it keeps behaviour identical.
    if return_ps:
        return truth_t, sino_t, sino_t, ps_arr
    return truth_t, sino_t, sino_t


def mayo_proj_cache(ps_array, n_angles, n_det, device, *,
                    det_spacing: float = 1.285044, sod: float = 595.362,
                    sdd: float = 1086.803) -> dict:
    """Per-ps projector cache for the canonical Mayo data. Angle is uniform
    (angle_start=0, baked into the canonical sinos); only the recon pixel-
    spacing varies per patient (4 distinct values). Returns
    ``{round(ps,5): PyronnFanBeamProjector}``. Each projector auto-applies
    MAYO_LDCT_DET_OFFSET + MAYO_LDCT_TRUNCATION via the
    AGENT4CT_DATASET=mayo_ldct_2d env hard-wiring. Solvers (batch_size=1) look
    up ``cache[round(float(ps[i]), 5)]`` per sample and swap it into the
    pipeline's FBP projector before the forward pass."""
    import math
    from .pyronn_projector import PyronnFanBeamProjector
    cache = {}
    for u in np.unique(np.round(np.asarray(ps_array, dtype=float), 5)):
        geom = FanBeamGeometry(image_size=512, pixel_spacing=float(u),
                               n_angles=int(n_angles), n_det=int(n_det),
                               det_spacing=det_spacing, sod=sod, sdd=sdd,
                               angle_start=0.0, angle_end=2 * math.pi)
        cache[round(float(u), 5)] = PyronnFanBeamProjector(geom).to(device)
    print(f"[staged] mayo_proj_cache: {sorted(cache)} ps values", flush=True)
    return cache


def mayo_per_sample_setup(train_ps, val_ps, cfg, device):
    """Set up per-sample-ps reconstruction for the canonical Mayo data.

    The canonical sinos are angle-uniform (angle_start=0) but each slice must be
    reconstructed at its OWN recon pixel-spacing ``ps_eff`` so the FBP lands on
    the NATIVE (un-resampled) truth grid — preserving the native HD SSIM ~0.95
    (the per-patient FOV varies 0.66-0.78; a single fixed ps mis-scales every
    off-nominal slice by ~5%). Build a per-ps projector cache (≤4 distinct ps)
    and return rounded ps keys so a batch_size=1 solver can swap its projector
    per slice: ``model.<proj_attr> = projs[float(trk[idx[0]])]``.

    Returns ``(per_ps, projs, trk, vrk)``. ``per_ps`` is False when ps is None
    (non-Mayo datasets) so the caller falls back to its single fixed geometry.
    """
    if train_ps is None:
        return False, None, None, None
    projs = mayo_proj_cache(np.concatenate([train_ps, val_ps]),
                            cfg["n_angles"], cfg["n_det"], device)
    return (True, projs,
            np.round(np.asarray(train_ps, float), 5),
            np.round(np.asarray(val_ps, float), 5))


def mayo_per_sample_fbp(projs, keys, noisy, image_size=512):
    """Per-ps-group LD-FBP baseline of ``noisy`` (B,1,A,D) -> (B,1,H,W).

    Reconstructs each ps-group with its cached projector (the canonical Mayo
    baseline = the headroom-scoring 'low-dose FBP, no denoising' endpoint).
    ``keys`` = ``np.round(ps, 5)`` per slice (the ``vrk``/``trk`` from
    :func:`mayo_per_sample_setup`)."""
    import torch
    out = torch.empty(noisy.shape[0], 1, image_size, image_size,
                      device=noisy.device)
    for u in np.unique(keys):
        ii = np.where(keys == u)[0]
        out[ii] = projs[float(u)].fbp(noisy[ii]).clamp(min=0.0)
    return out


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
