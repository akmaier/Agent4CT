# TrueCT — partial fetch design

## Status

**ACQUIRED 2026-06-18.** The full dataset (5 files, **174 GiB**) is on the cluster at
`/cluster/maier/Agent4CT/data/truect/`, fetched via `rclone` from a private
Backblaze B2 bucket (`cvit:truect22`) provided by the challenge organisers, and
**SHA1-verified** (`rclone check` → 5 matching files, 0 differences).

| File | Size |
|---|---:|
| `dcmproj_copd.zip` | 69.1 GB |
| `dcmproj_liver.zip` | 48.2 GB |
| `dcmproj_lung_lesion.zip` | 68.7 GB |
| `reference.zip` | 359 MB |
| `TrueCT-Documentation.pdf` | 400 KB |

The raw `.zip` archives are **NOT yet extracted or staged** into the harness HDF5
layout — that's the next step before any TrueCT solver work (read
`TrueCT-Documentation.pdf` for the geometry/format first). The size estimate
below (~600 GB–1 TB) was wrong — the published release is 174 GiB. The
CodaLab-gating notes that follow are now **historical**, kept for provenance.

## CVIT-Duke page (checked 2026-05-15)

The challenge page <https://cvit.duke.edu/truth-based-ct-truect-reconstruction-challenge/>
says:

- Access is via **CodaLab registration** (links from the challenge page);
  no direct file URLs are exposed.
- Format is "Mayo Clinic standard sinogram format" with supplemental scan
  geometry files.
- The page claims "datasets made public" August 2022 but does NOT link to
  a permanent mirror (no DOI, no Zenodo, no S3).
- Contact: `cvit-inquire@duke.edu`.

**Net:** there's no scraped-able fetch route. Two options:

1. Register at CodaLab, accept the rules, download whatever the portal
   serves. If it's a per-case archive list, we can subset; if it's a
   single 600 GB tarball, we can't.
2. Email `cvit-inquire@duke.edu` and ask for per-case URLs / DOI mirror.

Either path requires a human in the loop before we can write a working
fetch script.

## Where the data lives

- **AAPM challenge page:** [aapm.org/GrandChallenge/TrueCT](https://www.aapm.org/GrandChallenge/TrueCT/)
- **CVIT Duke page:** [cvit.duke.edu/truth-based-ct-truect-reconstruction-challenge](https://cvit.duke.edu/truth-based-ct-truect-reconstruction-challenge/)
- **Report:** [Abadi 2025](https://doi.org/10.1002/mp.17619)
- **Format:** Sinograms in Mayo-standard DICOM-CT-PD; truth as XCAT/MASH
  computational phantom voxel volumes.
- **Cases:** 200 (67 COPD / 67 lung-nodule / 66 abdominal).

## Can we fetch only parts?

**Probably yes, but the route is not obvious.** Three things to verify:

### 1. Is the data publicly downloadable at all?

Both pages link to a registration form for the challenge itself. The
**post-challenge data release** policy is the open question. Some AAPM
challenges (DL-Sparse-View, DL-Spectral) post the data to Zenodo
immediately; others (Mayo LDCT) require ongoing access agreements via
TCIA. TrueCT is published as of 2025 but the data link in the report
points to CVIT-Duke, not a permanent DOI mirror at the time of writing.

**Action:** email the corresponding author (Ehsan Abadi, Duke) and the
AAPM challenge contact to ask:
  - Is the dataset publicly downloadable?
  - Is there a per-case URL pattern, or only a single-archive download?
  - Are there checksums published?

### 2. If hosted as one archive

If the dataset is shipped as a single tarball (`truect_2022.tar.gz` or
similar), **no partial fetch is possible** without their cooperation.
Either:
  - download the whole archive (only feasible if ≤200 GB) and stage a
    subset locally, or
  - ask the authors to publish per-case bundles.

### 3. If hosted as per-case directories

If the CVIT portal serves one URL per phantom (e.g.
`cvit.duke.edu/truect/data/phantom-001.tar.gz`), the script just iterates
a subset of IDs:

```python
SUBSET = ["copd-001", "lung-001", "abdomen-001", ...]  # ~15 cases
for case_id in SUBSET:
    download(f"https://cvit.duke.edu/truect/data/{case_id}.tar.gz",
             raw_dir / f"{case_id}.tar.gz")
```

This is the same pattern as the Mayo TCIA approach.

## Recommended subset (when access is sorted)

To match our Wagner-style 4/1/5 split per pathology and keep size <50 GB:
~15 cases total (5 per pathology, 4 train + 1 val + … well, sample whatever
balances the three pathologies given our compute budget).

## Decision needed before implementing

1. **Email Abadi / AAPM** to clarify access and per-case URL availability.
2. If single-archive only: decide whether 200-case full download fits
   under our 500 GB total budget once other staging is added.
3. **Geometry confirmation.** The report says Mayo-standard DICOM-CT-PD,
   but the rebinned view count / detector layout may differ from our
   `FanBeamGeometry` defaults. Read the report Sec. 2 / their data
   parameter file before staging.
