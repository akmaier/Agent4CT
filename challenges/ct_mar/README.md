# CT-MAR — AAPM CT Metal Artifact Reduction Grand Challenge (2024)

The dataset that fits our dual-domain pipeline architecturally: every case
ships **both sinogram and image** versions, with and without metal, plus a
metal mask.

## Overview

| | |
|---|---|
| Years | Oct 2023 — Jul 2024 (closed); report published 2025 |
| Organisers | GE HealthCare Technology and Innovation Center, Massachusetts General Hospital, Rensselaer Polytechnic Institute, under AAPM |
| Report | Haneda et al., *Med. Phys.* 2025, doi:[10.1002/mp.70050](https://doi.org/10.1002/mp.70050) |
| AAPM page | <https://www.aapm.org/GrandChallenge/CT-MAR/> · [winners](https://www.aapm.org/GrandChallenge/CT-MAR/winners.asp) |
| Public data mirror | [xcist/example/tree/main/AAPM_datachallenge](https://github.com/xcist/example/tree/main/AAPM_datachallenge) |
| Challenge dataset (RPI Box) | <https://rpi.app.box.com/s/7p8tkqj5ewhtdad2h8kx975i9qg6b7a4> |
| Simulator | [XCIST](https://github.com/xcist/main) (CatSim) |
| Anatomy | Lung, abdomen, liver, head, pelvis |

## Task

Reconstruct metal-artefact-free CT images (512 × 512) from sinograms
contaminated with high-density metal objects (implants, clips, screws). The
training set is a hybrid of real patient anatomy + virtual metal objects;
the final test is on real clinical cases.

## Why it is the best fit for Agent4CT

Each training case bundles **five tensors**:

1. Uncorrected sinogram (with metal)
2. Metal-free sinogram (ground truth, projection domain)
3. Uncorrected reconstructed image (with metal artefacts)
4. Metal-free reconstructed image (ground truth, image domain)
5. Metal mask

→ This matches our `DualDomainPipeline` exactly: `D_proj` learns on
sinogram pairs, `D_img` on image pairs, and the metal mask is a free
auxiliary label for an ablation / loss reweighting study.

## Train / val / test split

The challenge ran three phases:

| Phase | Purpose | Cases |
|---|---|---:|
| 1 (training) | 14 000 simulated cases — XCIST hybrid (real anatomy + virtual metal) | 14 000 |
| 2 (validation) | 1 000 held-out cases for PSNR/SSIM during the challenge | 1 000 (not in the public mirror) |
| 3 (scoring) | 29 real clinical cases from MGH, sinogram + image domain | 29 |

For Agent4CT we adopt:

| Split | Source | Cases |
|---|---|---:|
| train | Phase-1 mirror, first 12 000 cases | 12 000 |
| val | Phase-1 mirror, remaining 2 000 cases (anatomy-stratified) | 2 000 |
| test | **Phase-3 scoring set** (29 real clinical cases) | 29 |

Stratify val by anatomy (lung/abdomen/liver/head/pelvis) so we get
per-anatomy metrics.

## Geometry

XCIST simulation parameters are documented in the per-case `.cfg` files
shipped with the data. Typical configuration for this challenge:

- 512 × 512 image
- Fan-beam, equiangular detector
- ~720–1024 views per rotation
- ~900 detector channels

The data-staging pass should produce a single
`FanBeamGeometry` per anatomy group (they're constant within an anatomy in
this challenge).

## Download

Two Box links from the XCIST mirror:

| Bundle | URL | Notes |
|---|---|---|
| Phase 1 training/validation | <https://rpi.box.com/s/7p8tkqj5ewhtdad2h8kx975i9qg6b7a4> | 14 000 cases, ~150–250 GB (verify on landing page) |
| Phase 3 scoring benchmark | <https://rpi.box.com/s/p8aayubdww9tav66urn9tvpsv2bwyxar> | 29 clinical cases |

Both pages include a README with format details.

```bash
# Recommended: download on the laptop, copy to cluster (Box may be flaky
# from FAU IP space). Or use rclone with a Box remote.
wget -O ct_mar_phase1.tar https://rpi.box.com/s/7p8tkqj5ewhtdad2h8kx975i9qg6b7a4
wget -O ct_mar_phase3.tar https://rpi.box.com/s/p8aayubdww9tav66urn9tvpsv2bwyxar
scp ct_mar_phase*.tar maier@cluster.i5.informatik.uni-erlangen.de:/cluster/maier/Agent4CT/data/ct_mar/raw/
ssh lme bash -c '
  cd /cluster/maier/Agent4CT/data/ct_mar/raw
  tar -xf ct_mar_phase1.tar
  tar -xf ct_mar_phase3.tar
'
```

If Box throttles you, fall back to the [XCIST mirror page](https://github.com/xcist/example/tree/main/AAPM_datachallenge)
to look for any Zenodo / S3 mirrors that may have appeared after the report
was published.

## Storage layout on the cluster

```
/cluster/maier/Agent4CT/data/ct_mar/
    raw/                                 # untouched .tar downloads
    staged/
        train_sinos_uncorrected.h5       # (12000, A, D)
        train_sinos_metalfree.h5         # (12000, A, D)
        train_imgs_uncorrected.h5        # (12000, 512, 512)
        train_imgs_metalfree.h5          # (12000, 512, 512)
        train_metal_masks.h5             # (12000, 512, 512) uint8
        val_*.h5                         # same five tensors, 2000 cases
        test_*.h5                        # Phase-3 29 clinical cases
        geometries.json                  # per-anatomy FanBeamGeometry blocks
```

## Citation

```
@article{haneda2025ctmar,
  title={AAPM CT metal artifact reduction grand challenge},
  author={Haneda, Eri and others},
  journal={Medical Physics},
  year={2025},
  doi={10.1002/mp.70050}
}
```
