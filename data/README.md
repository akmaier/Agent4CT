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
| TrueCT | ~40–80 GB (estimate; not published) | 3–5 % | ⚠️ verify size after manifest |
| Mayo LDCT (Wagner 10-scan subset) | ~100–200 GB | 6–12 % | ✅ if needed for Mayo challenge |
| Mayo LDCT (full) | 1.32 TB | 77 % | ❌ antisocial |
| CT-MAR | ~150–300 GB (estimate) | 9–18 % | ⚠️ confirm post-challenge access |

## Usage

```bash
# One-off per dataset, run from the cluster:
python data/fetch_dl_sparse_view.py    # ~10-20 GB, ~15 min wall
python data/fetch_htc2022.py           # <1 GB, <1 min
python data/fetch_dl_spectral.py       # ~30-50 GB
python data/fetch_lodopab.py           # 55 GB
python data/fetch_truect.py            # est. 40-80 GB
# Mayo + CT-MAR: see notes in individual scripts (gated / large).
```

Every script writes to `${AGENT4CT_DATA:-/cluster/maier/Agent4CT/data}/<challenge>/`.

## Layout per challenge

```
data/<challenge>/
    raw/                            # untouched archive(s)
    staged/
        train_sinograms.h5          # (N_train, A, D) float32
        train_truth.h5              # (N_train, H, W) float32
        val_sinograms.h5            # (N_val,   A, D) float32
        val_truth.h5                # (N_val,   H, W) float32
        test_sinograms.h5           # (N_test,  A, D) float32 — held-out
        test_truth.h5               # (N_test,  H, W) float32 — held-out
        manifest.json               # geometry, splits, checksums, dates
```

HDF5 chunks are sized `(1, A, D)` for sinograms and `(1, H, W)` for images so
`__getitem__(i)` is a single chunk read. Compression: `lz4` if available, else
`gzip-1`. The harness expects this layout; if you stage by hand, follow it
exactly.

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
