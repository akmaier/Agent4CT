# Agent4CT — Benchmark challenges

Five CT reconstruction / denoising / artefact-reduction benchmarks that the
Agent4CT autoresearch loop will target. Each subfolder has an overview of the
challenge, the data format, download instructions, and the recommended
train / validation / test split for our pipeline.

| Subfolder | Challenge | Year | Public data | Why we picked it |
|---|---|:---:|:---:|---|
| [`mayo_ldct/`](mayo_ldct/) | AAPM Low Dose CT Grand Challenge | 2016 | ✅ | The dataset our recon backbone (Wagner et al. 2023) trains on |
| [`dl_sparse_view/`](dl_sparse_view/) | AAPM DL-Sparse-View CT | 2021 | ⚠️ partial | Synthetic phantom with perfectly-known ground truth — RMSE-clean leaderboard target |
| [`truect/`](truect/) | AAPM Truth-based CT Reconstruction | 2022 | ✅ | 200 virtual patients with mono-energetic truth, sinogram-in / image-out |
| [`ct_mar/`](ct_mar/) | AAPM CT Metal Artifact Reduction | 2024 | ✅ via XCIST | Ships sinogram **and** image pairs — perfect fit for our dual-domain pipeline |
| [`dl_spectral/`](dl_spectral/) | AAPM DL-Spectral CT | 2022 | ✅ Zenodo | Multi-energy recon; lets us probe spectral generalisation |

## Conventions used across the subfolders

- **Train / val / test split.** Each `README.md` proposes a split the
  Agent4CT pipeline will adopt. Where the challenge already defines a split
  we follow it verbatim; where it does not, we pick a held-out test set and
  document the choice.
- **Where the data lives on the cluster.** Since 2026-07-24 the data lives in
  the lab-wide share at `/cluster/shared_dataset/Agent4CT/<challenge>/`, with
  `/cluster/maier/Agent4CT/data/<challenge>` a symlink into it — so the paths
  below still resolve unchanged. The convention is
  `/cluster/maier/Agent4CT/data/<challenge>/raw/` for the untouched download
  and `/cluster/maier/Agent4CT/data/<challenge>/staged/` for the
  pre-processed (re-binned, packed) form we feed PYRO-NN. Hot copies move to
  `/scratch/maier/<challenge>/` at job start — see
  [`docs/performance.md`](../docs/performance.md) for why.
- **Loader contract.** Each subfolder will have a loader that yields
  `(sinogram, optional_image_target, optional_phantom)` tensors on CUDA in
  the shape PYRO-NN expects (`(B,1,A,D)` for sinograms; `(B,1,H,W)` for
  images). The geometry parameters in `FanBeamGeometry` are dataset-specific
  and listed in the per-challenge README.

## Size budget

The cluster's shared `/cluster` volume is 95 % full (1.7 TB free). The five
challenges combined fit comfortably if we mirror Wagner's subset of Mayo
rather than the full 1.32 TB:

| Challenge | Storage footprint we target |
|---|---:|
| Mayo LDCT (Wagner 10-scan subset, rebinned) | ~150 GB |
| DL-Sparse-View | ~15 GB |
| TrueCT | ~60 GB (est.) |
| CT-MAR (XCIST Box mirror) | ~200 GB (est.) |
| DL-Spectral | ~40 GB |
| **total** | **≤ 500 GB** |

That stays well under "be a good neighbour on the shared volume".
