# TrueCT — AAPM Truth-based CT Reconstruction Grand Challenge

Reconstruct from sinograms with a **mono-energetic, voxel-level ground truth**
established from computational phantoms.

## Overview

| | |
|---|---|
| Year | 2022 (presented at AAPM 2022; report published 2025) |
| Organisers | AAPM + Center for Virtual Imaging Trials (CVIT), Duke (PI: Ehsan Abadi) |
| Report | Abadi et al., *Med. Phys.* 2025, doi:[10.1002/mp.17619](https://doi.org/10.1002/mp.17619) — [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC11973969/) |
| AAPM page | <https://www.aapm.org/GrandChallenge/TrueCT/> |
| CVIT page | <https://cvit.duke.edu/truth-based-ct-truect-reconstruction-challenge/> |
| Phantom | 200 computational phantoms (XCAT / MASH derivative) |
| Pathology | 67 COPD, 67 lung-nodule, 66 abdominal |
| Format | Sinograms in the Mayo-standard DICOM-CT-PD format |
| Truth | Mono-energetic phantom representation (voxel-level) |

## Task

Reconstruct CT slices in DICOM format from the challenge's simulated
sinograms. Evaluation is similarity to the mono-energetic phantom truth.

## Why we include it

- Same sinogram format as Mayo LDCT → the loader is 95 % shared with
  `mayo_ldct/`.
- The mono-energetic truth is voxel-exact, so PSNR / SSIM mean something
  (unlike comparing against an artefact-laden clinical reference).
- Three pathology classes lets us measure task-specific image quality
  downstream (lesion detection, density quantification) once the basic
  recon is solid.

## Train / val / test split

The challenge had its own organiser-held test set. For Agent4CT we adopt
the patient-disjoint split:

| Split | Cases | Pathology mix |
|---|---:|---|
| train | 140 (~70 %) | 47 COPD, 47 nodule, 46 abdomen |
| val | 30 | 10 / 10 / 10 |
| test | 30 | 10 / 10 / 10 |

Patient-ID-disjoint so no leakage of anatomical priors.

## Geometry

Per the CVIT distribution, geometry is per-case and embedded in the Mayo-style
header. The data-staging pass should:

1. Parse the per-case manifest into a `geometries.json` (one
   `FanBeamGeometry` per scan).
2. Group scans with matching geometry into shards for batch projection.

## Download

The CVIT distribution page is the canonical source:
<https://cvit.duke.edu/truth-based-ct-truect-reconstruction-challenge/>

CVIT typically requires a short registration form (institution + use
statement). After approval you receive a download link.

Once registered:

```bash
# On the laptop:
wget -O truect_data.zip "<signed CVIT URL>"
scp truect_data.zip maier@cluster.i5.informatik.uni-erlangen.de:/cluster/maier/Agent4CT/data/truect/raw/

# On the cluster:
ssh lme bash -c '
  cd /cluster/maier/Agent4CT/data/truect/raw
  unzip truect_data.zip
'
```

Estimated total size **~40–80 GB** (200 phantoms × multi-dose simulations
+ truth volumes); confirm against the CVIT release notes when you register.

## Storage layout on the cluster

```
/cluster/maier/Agent4CT/data/truect/
    raw/                                   # CVIT distribution as downloaded
    staged/
        sinograms.h5                       # rebinned / packed per geometry-shard
        truth.h5                           # mono-energetic ground-truth volumes
        manifest.json                      # patient ID, geometry block, split
```

## Citation

```
@article{abadi2025truect,
  title={AAPM Truth-based CT (TrueCT) reconstruction grand challenge},
  author={Abadi, Ehsan and Segars, W. Paul and others},
  journal={Medical Physics},
  year={2025},
  doi={10.1002/mp.17619}
}
```
