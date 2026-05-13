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

## Geometry to pass into `FanBeamGeometry`

The challenge geometry is published in the report (Sec. 2.1):

```python
FanBeamGeometry(
    image_size=512,
    pixel_spacing=...,   # TODO — confirm from challenge manifest
    n_angles=128,
    n_det=...,           # TODO — confirm
    det_spacing=...,
    sod=...,
    sdd=...,
    angle_start=0.0,
    angle_end=2 * math.pi,
)
```

The data-staging pass will fill the TODOs from the challenge `parameters.txt`
that ships with the download.

## Download

Two routes — try the AAPM page first, fall back to the report if files are
gated:

1. AAPM Grand Challenge page → "data" link:
   <https://www.aapm.org/GrandChallenge/DL-sparse-view-CT/>
2. Zenodo mirror (some files restricted; request access if needed):
   <https://zenodo.org/records/13882980>
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
