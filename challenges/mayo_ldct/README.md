# Mayo LDCT — AAPM Low Dose CT Grand Challenge (2016)

The dataset Wagner et al. 2023 trains on, and our primary target.

## Overview

| | |
|---|---|
| Year | 2016 (challenge); data later expanded in 2021 |
| Organisers | AAPM / Mayo Clinic (PI: Cynthia McCollough) |
| Reference | McCollough et al., *Med. Phys.* 2021 — "Low-Dose CT Image and Projection Data" |
| Data DOI | [10.7937/9NPB-2637](https://doi.org/10.7937/9NPB-2637) (TCIA) |
| Anatomy | Abdomen, chest, head |
| Format | DICOM-CT-PD (projection) + DICOM (image), Siemens SOMATOM Definition AS |
| Dose | Normal-dose reference + 25 % simulated low-dose |
| Full size | **1.32 TB**, 299 cases |

## Why it matters here

This is the dataset behind Wagner *et al.* 2023 (arXiv:2211.01111) and 2022
(Med. Phys.). Our `FanBeamGeometry` defaults are precisely the rebinned
geometry of these scanners (see `ddssl_ldct/geometry.py`):

- 512 × 512 image at 0.7 mm voxel pitch
- 1152 views over 2π, 736 detector channels at 1.2858 mm
- SOD 595 mm, SDD 1085.6 mm

## Subset we mirror (Wagner abdomen split)

Wagner trains on 4 train + 1 val + 5 test abdomen scans. We mirror that split
to keep storage manageable and the comparison fair.

| Role | Scans | Reconstructed slices (≈) |
|---|---|---:|
| train | L067 L096 L143 L192 | 4 × 100 = 400 |
| val | L286 | 1 × 100 = 100 |
| test | L291 L310 L333 L506 L067* | 5 × 100 = 500 |

`*` if a sixth test scan is needed; pick a non-overlapping ID. The exact IDs
in Wagner's paper are not enumerated in the published manuscript — we adopt
the patient IDs commonly used in the LDCT literature on this dataset.

## Storage layout on the cluster

```
/cluster/maier/Agent4CT/data/mayo_ldct/
    raw/                       # untouched DICOM-CT-PD (helical projections)
    staged/
        train_sinograms.h5     # rebinned fan-beam sinograms (1152, 736), full + half splits
        train_images.h5        # high-dose reference reconstruction (512, 512)
        val_*.h5
        test_*.h5
        manifest.json          # patient IDs → slice indices, geometry block
```

The rebinning from helical DICOM-CT-PD to fan-beam happens once via
[faebstn96/helix2fan](https://github.com/faebstn96/helix2fan) and is
checkpointed to `staged/`. We never touch `raw/` during training.

## Download

1. **Register on TCIA** (free, ORCID or email): <https://www.cancerimagingarchive.net/>
2. Install the [NBIA Data Retriever](https://wiki.cancerimagingarchive.net/display/NBIA/Downloading+TCIA+Images).
3. From the collection page <https://www.cancerimagingarchive.net/collection/ldct-and-projection-data/>,
   click *Download* → select **Projection Data (DICOM-CT-PD)** for the
   abdomen patient IDs above. Each scan is ~10–20 GB.
4. Or via CLI manifest (recommended for cluster):

   ```bash
   # On the laptop, get the .tcia manifest from TCIA after selecting cases.
   scp Manifest_LDCT-and-Projection-data_*.tcia maier@lme:/cluster/maier/Agent4CT/data/mayo_ldct/
   ssh lme bash -c '
     module load java 2>/dev/null || true
     cd /cluster/maier/Agent4CT/data/mayo_ldct
     ~/nbia-data-retriever -cli Manifest_*.tcia -d raw/
   '
   ```

5. **Rebin** to fan-beam (run on a GPU compute node — the rebinning is
   helped by torch):

   ```bash
   sbatch cluster/slurm/rebin_mayo.sbatch   # to be authored by the data-staging pass
   ```

## Loader (planned)

```python
from ddssl_ldct.data.mayo_ldct import MayoLDCTDataset
ds = MayoLDCTDataset(split="train", root="/cluster/maier/Agent4CT/data/mayo_ldct/staged")
sino_ld, sino_nd, img_nd = ds[0]  # (1,1152,736), (1,1152,736), (1,512,512)
```

## Citation

```
@article{mccollough2021ldct,
  title={Low-dose CT image and projection data (LDCT-and-Projection-data) (Version 5)},
  author={McCollough, C and Chen, B and Holmes, D and Duan, X and Yu, Z and Yu, L and Leng, S and Fletcher, J},
  journal={The Cancer Imaging Archive},
  year={2020},
  doi={10.7937/9NPB-2637}
}
```
