# DL-Sparse-View CT — AAPM Grand Challenge (2021)

Sparse-view fan-beam reconstruction of a 2D simulated breast phantom with
**perfectly-known ground truth**. The cleanest leaderboard for "what's the
state-of-the-art on a well-posed inverse problem".

## Overview

| | |
|---|---|
| Year | 2021 (open Mar 17 – Jun 1) |
| Organisers | AAPM; Sidky & Pan group, University of Chicago |
| Report | Sidky et al., *Med. Phys.* 2022, doi:[10.1002/mp.15489](https://doi.org/10.1002/mp.15489) — [PMC PDF](https://pmc.ncbi.nlm.nih.gov/articles/PMC9314462/) |
| Page | <https://www.aapm.org/GrandChallenge/DL-sparse-view-CT/> |
| Phantom | 2D simulated breast — random fibro-glandular structure + high-contrast specks |
| Geometry | Fan-beam, 128 views (sparse) over 2π |
| Metric | RMSE against truth |

## Task

Reconstruct a 2D image from **128-view sparse sinograms** of a simulated
breast phantom such that RMSE against the known truth is minimised. Because
truth is exact, this is the cleanest sandbox we have for the autoresearch
loop — no FBP-reference fuzz, no domain gap.

## Data composition

| Split | Cases | Per case |
|---|---:|---|
| train | 4000 | truth image, 128-view sinogram, 128-view FBP |
| validation | held by organisers during the challenge | — |
| test | held by organisers during the challenge | — |

For our pipeline:

- **Train**: 3600 of the 4000 training cases.
- **Validation**: 400 cases held out from the 4000 (we don't have the
  organiser's val set anymore).
- **Test**: an additional 400-case held-out split if the data dump permits.

## Geometry

Single source of truth: [`challenges/dl_sparse_view/geometry.py`](geometry.py).
`GEOMETRY` is a `DatasetInfo`; `DEFAULTS` is the solver-CONFIG-shaped dict
that `ddssl_ldct.staged_dataset.geometry_overrides("breast_ct")` returns.

| Knob | Value | Provenance |
|---|---|---|
| `image_size` | 512 | Sidky 2022 §II.B |
| `pixel_spacing` | 0.3516 mm | 180 mm / 512 = paper FOV (18 cm) / image dim |
| `n_angles` | 128 | Sidky 2022 §II.B (128 views over 2π) |
| `n_det` | 1024 | Sidky 2022 §II.B |
| `det_spacing` | **0.35754 mm** | **NOT in paper**; pinned empirically (see below) |
| `sod` | 500 mm | Sidky 2022 §II.B (50 cm source-to-iso) |
| `sdd` | 1000 mm | Sidky 2022 §II.B (100 cm source-to-detector) |
| `display_max` | 0.5 | μ range [0, 0.33] 1/cm + headroom for microcalcs |
| `sino_angle_shift` | +32 (= +90°) | aligns Sidky gantry origin with pyronn (angle 0 = source at +x, CCW) |

### `det_spacing` derivation

The paper specifies image dims, channel count, SOD, SDD — but not the
detector pitch. We pinned it down by sweeping `det_spacing = 0.35156 · c`
over `c ∈ [0.94, 1.06]` and minimising the forward-projection L2 of the
matched-pair Siddon projector against the released `val_sinograms[0]`
(see `scripts/debug_breast_ct_detspacing_sweep.py`, SLURM job 761480):

- Best `c = 1.017` → `det_spacing = 0.35754 mm` → forward L2 ≈ 7.9 × 10⁻⁴
  (essentially exact agreement).
- Detector iso-projection covers `0.35754 · 1024 · 0.5 = 18.31 cm` — a
  1.7 % margin over the 18 cm image FOV (sensible CT design choice).

### FOV mask

Sidky's released `val_fbp128.h5` is masked outside the inscribed circle
of the 512×512 grid (radius 256 px = 9 cm). 21.46 % of pixels are
exactly 0 (= `1 − π/4`). Our metrics apply the same mask via
`ddssl_ldct.metrics.fov_mask(size=512)`.

## Download

**Confirmed 2026-05-15: dataset is CodaLab-gated, no Zenodo mirror exists.**
The "Zenodo 13882980" reference previously listed here was wrong — that's
the DL-Spectral info record (0 files). The real download flow requires
per-user CodaLab registration:

1. AAPM Grand Challenge page (overview only): <https://www.aapm.org/GrandChallenge/DL-sparse-view-CT/>
2. CodaLab competition portal (registration required):
   <https://dl-sparse-view-ct-challenge.eastus.cloudapp.azure.com/competitions/1>
3. Report-paper supplementary: see the data-availability statement in Sidky
   et al. 2022.

The training-set archive is ~10–20 GB.

## Storage layout on the cluster

```
/cluster/maier/Agent4CT/data/dl_sparse_view/
    raw/                          # untouched challenge archive
    staged/
        train_sinograms.h5        # (3600, 128, n_det)
        train_truth.h5            # (3600, 512, 512)
        val_*.h5
        test_*.h5
        parameters.json           # geometry block
```

## Citation

```
@article{sidky2022aapm,
  title={Report on the AAPM deep-learning sparse-view CT (DL-sparse-view CT) Grand Challenge},
  author={Sidky, Emil Y. and Pan, Xiaochuan},
  journal={Medical Physics},
  volume={49},
  number={8},
  pages={4986--5004},
  year={2022},
  doi={10.1002/mp.15489}
}
```
