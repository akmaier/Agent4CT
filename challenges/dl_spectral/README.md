# DL-Spectral CT — AAPM Grand Challenge (2022)

Multi-energy (spectral / dual-energy) CT reconstruction on simulated breast
phantoms. Extends our pipeline beyond monochromatic LDCT.

## Overview

| | |
|---|---|
| Year | 2022 |
| Organisers | AAPM; Sidky & Pan group, University of Chicago |
| Report | Sidky et al., *Med. Phys.* 2024, doi:[10.1002/mp.16363](https://doi.org/10.1002/mp.16363) |
| AAPM page | <https://www.aapm.org/GrandChallenge/DL-spectral-CT/> |
| Data DOI | [Zenodo 14262737](https://zenodo.org/records/14262737) (training + supporting), [Zenodo 13882980](https://zenodo.org/records/13882980) (challenge info) |
| Phantom | Simulated 2D breast phantom (adipose / fibroglandular / calcification tissue maps) |
| Spectra | Two source spectra (kVp settings) |
| Geometry | Fan-beam, fully published in the report |

## Task

Given the **two-energy** sparse fan-beam sinograms, jointly recover the
tissue-class density images (adipose, fibroglandular, calcification). The
challenge metric is RMSE against the truth tissue maps.

## Data composition

From the Zenodo record:

- `Phantom_<Tissue>.npy.gz` — `(1000, 512, 512)` arrays. One per tissue
  class. Truth.
- Sinograms — two sets per case (one per kVp), released alongside the truth.
- Forward-model / geometry definition shipped with the data.

| Split | Cases |
|---|---:|
| Phase-1 train (public) | 1000 |
| Validation | organiser-held |
| Test | organiser-held |

## Train / val / test split for Agent4CT

Since the organiser-held splits aren't available, we partition the 1000
public cases:

| Split | Cases |
|---|---:|
| train | 800 |
| val | 100 |
| test | 100 |

Random seed pinned (`np.random.RandomState(0).permutation(1000)`).

## Geometry

The report (Sidky et al. 2024) and the Zenodo README publish the
configuration. Fill into `FanBeamGeometry` after staging:

```python
FanBeamGeometry(
    image_size=512,
    pixel_spacing=...,  # TODO confirm from challenge config
    n_angles=...,
    n_det=...,
    det_spacing=...,
    sod=...,
    sdd=...,
)
```

Two spectra → two separate `FanBeamGeometry` instances or one with an extra
spectrum index passed to the projection step (PYRO-NN's `FanProjection2D` is
monochromatic; the spectral modelling happens upstream of the projector).

## Download

```bash
# On the cluster (zenodo is happy with direct wget):
ssh lme bash -c '
  cd /cluster/maier/Agent4CT/data/dl_spectral/raw
  for fn in $(curl -s https://zenodo.org/api/records/14262737 | jq -r ".files[].links.self"); do
    wget -c "$fn"
  done
'
```

Estimated total ~30–50 GB.

## Storage layout on the cluster

```
/cluster/maier/Agent4CT/data/dl_spectral/
    raw/                                  # Zenodo archive as downloaded
    staged/
        train_sino_low_kvp.h5             # (800, A, D)
        train_sino_high_kvp.h5            # (800, A, D)
        train_truth_adipose.h5            # (800, 512, 512)
        train_truth_fibroglandular.h5
        train_truth_calcification.h5
        val_*.h5
        test_*.h5
        geometry.json                     # spectral + fan-beam config
```

## Citation

```
@article{sidky2024spectral,
  title={Report on the AAPM deep-learning spectral CT Grand Challenge},
  author={Sidky, Emil Y. and Pan, Xiaochuan and others},
  journal={Medical Physics},
  year={2024},
  doi={10.1002/mp.16363}
}
```
