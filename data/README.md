# Pentathlon data fetchers

Scripts to mirror per-challenge data onto the LME cluster in the staged HDF5
layout that `docs/performance.md` describes. Each fetcher does three things:

1. Download the raw archive from its canonical source (Zenodo / AAPM / Box / TCIA).
2. Verify the published checksum.
3. Convert to `data/<challenge>/staged/{train,val,test}_{sinograms,truth}.h5`
   plus a `manifest.json` recording geometry + provenance.

The five-minute iteration harness and the one-hour stage harness both read
the same HDF5 files — they only differ in `train_n` / `val_n` / `epochs`
overrides (see `cluster/slurm/dl_sparse_view*_5min.sbatch` vs the
matching `*_stage.sbatch`). So you stage once per challenge, not twice.

## Where the raw data lives on the cluster (state as of 2026-05-15)

All challenge data lives under `/cluster/maier/Agent4CT/data/<challenge>/`
on the LME cluster (`maier@cluster.i5.informatik.uni-erlangen.de`).
Subsequent agents can find what's already on disk here:

| Challenge | On-cluster path | Raw | Staged HDF5 | Splits (train/val/test) | Notes |
|---|---|---:|---:|---:|---|
| `ct_mar` | `data/ct_mar/raw/` + `data/ct_mar/staged/` | 110 GB | **6.0 GB** ✅ | 4000 / 1000 / 1000 | 15 tar.gz from RPI Box <https://rpi.app.box.com/s/7p8tkqj5ewhtdad2h8kx975i9qg6b7a4> (developer-token API, 14k cases). `stage_h5()` streams Target/no-metal images from tar.gz (no extraction) and converts HU→μ mm⁻¹. Sinograms are forward-projected at train time via the challenge geometry. |
| `dl_spectral` | `data/dl_spectral/raw/` + `staged/` | 3.3 GB | **2.1 GB** ✅ | 800 / 100 / 100 | 1000 cases from Zenodo 14262737. `stage_h5()` packs multi-channel truth `(N,3,H,W)` + sinograms `(N,2,A,D)`. Channels = adipose / fibroglandular / calcification (truth) and high-kVp / low-kVp (sino). Decompresses .npy.gz to a temp dir then mmaps to keep RAM under 30 MB. |
| `mayo_ldct` | `data/mayo_ldct/raw/` + `staged/` | 69 GB | **0.8 GB** ✅ | 672 / 99 / 607 | Wagner 10-patient subset: `L004, L033, L064, L107, L143, L186, L221, L260, L288, L299` (L067 doesn't exist in TCIA's `LDCT-and-Projection-data` collection — replaced with L064). `stage_h5()` parses Full-Dose Images DICOM series and converts HU→μ mm⁻¹. Sinograms are forward-projected at train time (the projection-data series shipped in TCIA is per-frame DICOM-CT-PD and would need a separate parser if anyone needs measured-noise sinograms). Re-running the fetcher is idempotent. |
| `dl_sparse_view` | — | — | — | — | **BLOCKED**. CodaLab-gated <https://dl-sparse-view-ct-challenge.eastus.cloudapp.azure.com/competitions/1>. No public Zenodo mirror (the README's old "Zenodo 13882980" claim is wrong; that record is DL-Spectral info, 0 files). |
| `truect` | `data/truect/` (raw zips) | **174 GiB** ✅ | — *(not staged yet)* | — | **Acquired 2026-06-18** via rclone from private B2 bucket `cvit:truect22` (organiser-provided), SHA1-verified (5 files, 0 diffs). Files: `dcmproj_copd.zip` 69.1 GB, `dcmproj_liver.zip` 48.2 GB, `dcmproj_lung_lesion.zip` 68.7 GB, `reference.zip` 359 MB, `TrueCT-Documentation.pdf`. 200 phantoms (COPD/liver/lung-lesion). Raw zips NOT yet extracted/staged. See `data/INVESTIGATE_truect.md`. |

All three staged datasets share **μ mm⁻¹** as the image-value convention
(water = 0.02 mm⁻¹, matching `ddssl_ldct.phantoms.random_ellipses_phantom`).
DL-Spectral truth is the 3-channel material decomposition `(adipose,
fibroglandular, calcification)` in `[0, 1]` and uses a separate convention
that future spectral solvers must convert as needed.

To re-stage from scratch (e.g. after editing `stage_h5()`):

```bash
ssh maier@cluster.i5.informatik.uni-erlangen.de
cd /cluster/maier/Agent4CT && source .venv/bin/activate
rm -rf data/<challenge>/staged
python data/fetch_<challenge>.py --skip-download   # uses existing raw/
```

Note the iter solvers currently still train on the synthetic phantoms from
`ddssl_ldct/phantoms.py`. To switch to real data, swap the solver's
`build_dataset()` call for `StagedTruthDataset + RotatingSubsetDataset`
(see the example in the next section).

Verify what's on the cluster:

```bash
ssh maier@cluster.i5.informatik.uni-erlangen.de \
    "du -sh /cluster/maier/Agent4CT/data/*/raw 2>/dev/null"
```

## Social storage budget

Shared `/cluster` is at ~95 % full lab-wide. Stay under **~500 GB** total
under `/cluster/maier/` so the rest of the lab has working room. Verify with
`df -h /cluster` before any large pull — the headline figure drifts.

| Dataset | Estimated size | % of typical ~1.7 TB free | Pull? |
|---|---:|---:|---|
| HTC 2022 | <1 GB | ε | ✅ |
| DL-Sparse-View 2021 | ~10–20 GB | <1 % | ✅ |
| DL-Spectral 2022 | ~30–50 GB | 2–3 % | ✅ |
| LoDoPaB-CT | 55 GB | 3 % | ✅ |
| TrueCT | **174 GiB (acquired 2026-06-18)** | ~10 % | ✅ on cluster at `data/truect/` |
| Mayo LDCT (Wagner 10-scan subset) | ~100–200 GB | 6–12 % | ✅ if needed for Mayo challenge |
| Mayo LDCT (full) | 1.32 TB | 77 % | ❌ antisocial |
| CT-MAR | ~150–300 GB (estimate) | 9–18 % | ⚠️ confirm post-challenge access |

## Usage

```bash
# One-off per dataset, run from the cluster. Most scripts pre-exist;
# fetch_htc2022.py / fetch_lodopab.py / fetch_truect.py are not yet written.
python data/fetch_dl_sparse_view.py    # BLOCKED — CodaLab gating
python data/fetch_dl_spectral.py       # ~3.3 GB actual (was estimate 30-50 GB)
python data/fetch_mayo_ldct.py         # ~40 GB, Wagner 10-patient subset, anonymous TCIA
python data/fetch_ct_mar.py            # ~110 GB actual, requires BOX_TOKEN env var
```

Every script writes to `${AGENT4CT_DATA:-/cluster/maier/Agent4CT/data}/<challenge>/`.

## Layout per challenge

```
data/<challenge>/
    raw/                            # untouched archive(s)
    staged/
        train_truth.h5              # (N_train_pool, H, W) float32 (or (N,C,H,W) for spectral)
        train_sinograms.h5          # (N_train_pool, A, D) float32 — only when the dataset
                                    #   ships measured sinograms in a geometry the harness
                                    #   can use directly. Otherwise omitted.
        val_truth.h5                # (N_val_pool, ...)
        val_sinograms.h5            #   (held-out)
        test_truth.h5               # (N_test_pool, ...)
        test_sinograms.h5           #   (held-out)
        manifest.json               # geometry, splits, checksums, dates
```

`N_*_pool` is the FULL pool that staging emits — typically much larger
than the per-epoch budget (`train_n=400` in current sbatch settings). The
loader rotates which `train_n` samples each epoch sees; see "Per-epoch
subset rotation" below.

HDF5 chunks are sized `(1, A, D)` for sinograms and `(1, H, W)` for images so
`__getitem__(i)` is a single chunk read. Compression: `lz4` if available, else
`gzip-1`. The harness expects this layout; if you stage by hand, follow it
exactly.

## Per-epoch subset rotation (use this for agentic training)

Within a single 5-min iter run (`epochs ≈ 6`), the goal is to (a) see at
most `train_n=400` cases per epoch (compute budget), (b) over many
epochs visit different subsets of the pool, (c) keep everything
deterministic so two runs with the same `cfg["seed"]` are bitwise-equal.

Use `RotatingSubsetDataset` in `ddssl_ldct.staged_dataset`:

```python
from torch.utils.data import DataLoader
from ddssl_ldct.staged_dataset import (
    StagedTruthDataset, RotatingSubsetDataset, FanBeamGeometryFromManifest,
)
from ddssl_ldct.geometry import PyronnFanBeamProjector

# 1. Open the full pool (lazy: just metadata at this point).
pool = StagedTruthDataset(root=DATA_ROOT/"mayo_ldct/staged", split="train")
geom = FanBeamGeometryFromManifest(pool.manifest_path)         # challenge geom

# 2. Wrap with rotation; pass cfg["seed"] from the solver.
train = RotatingSubsetDataset(pool, n_per_epoch=cfg["train_n"], seed=cfg["seed"])

# 3. Per-epoch rotation: call set_epoch() BEFORE the inner DataLoader loop.
proj = PyronnFanBeamProjector(geom).to(device)
for ep in range(cfg["epochs"]):
    train.set_epoch(ep)
    loader = DataLoader(train, batch_size=cfg["batch_size"], shuffle=True,
                        generator=torch.Generator().manual_seed(cfg["seed"]+ep))
    for truth_batch in loader:
        truth = truth_batch.to(device)                   # (B, H, W) μ mm^-1
        clean_sino = proj.forward_project(truth)         # forward through challenge geom
        noisy = simulate_low_dose(clean_sino,
                                  i0=cfg["noise_i0"], sigma_e=cfg["noise_sigma_e"],
                                  seed=cfg["seed"] + 10_000 * ep)
        # ... train step ...
```

### Reproducibility rules — please read before reusing the loader

| Rule | Why |
|---|---|
| **Fix `cfg["seed"]`.** Same seed → same per-epoch subsets, same shuffles, same simulated noise. | Two iters with identical configs must give identical results — that's the variance-budget claim every same-config comparison relies on. |
| **`train.set_epoch(ep)` BEFORE the inner loop.** | Forgetting it means every epoch reuses the epoch-0 subset → the rotation does nothing. |
| **Pass `seed + N*ep` to noise-sim calls, not a single global seed.** | Otherwise the same simulated low-dose noise is applied every epoch — defeats SWA / averaging variance reduction. |
| **Don't shuffle in BOTH `RotatingSubsetDataset` and DataLoader.** Rotation already drew a fresh subset; DataLoader shuffles WITHIN it. | Double-shuffling is fine functionally but blocks future debugging of "which sample was at index k". |
| **`n_per_epoch ≤ len(pool)`.** Stagers must produce enough samples per split. | If you accidentally over-request, the wrapper falls back to whole-pool permutation per epoch and prints a one-time warning. |
| **Treat sample order in the pool as a contract.** Stagers shuffle once at packing time, so sample 0 and sample 1 are NOT guaranteed to be neighbouring slices of the same patient. | Future agents can rely on locality being broken — slice-correlation isn't a confound for batch statistics. |

A `StagedTruthDataset` is a single-image lookup; a `StagedH5Dataset` returns
`(sino, truth)`. Spectral (multi-channel) needs a small custom wrapper —
the file layout is `(N, C, H, W)` / `(N, 2, A, D)` and the existing classes
will return the full channel stack as one tensor.

## Verifying a stage

After a fetch script finishes:

```bash
python data/verify_staged.py <challenge>
```

This script (also in this directory) re-opens each HDF5, asserts shapes
match the manifest, samples a few cases, and prints SHA-256 of each file
so you can compare to git-versioned `manifest.json`.

## Adding a new fetcher

Copy `data/fetch_dl_sparse_view.py` as a template. The script should:

- Accept `--data-root` (default `${AGENT4CT_DATA:-/cluster/maier/Agent4CT/data}`).
- Skip already-staged files (idempotent — re-running is free).
- Write `staged/manifest.json` with `{source, sha256, geometry, splits,
  fetched_at_utc}`.
- Print a one-line summary at the end (case counts, total size).
- Exit non-zero if any checksum mismatches.
