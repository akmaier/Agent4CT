# Mayo LDCT — partial fetch design

## Status

Full dataset is **1.32 TB / 299 cases** at TCIA — antisocial to mirror on
our shared `/cluster` volume. Must subset. The Wagner-et-al. 2023 subset
(10 scans) is the working target.

## Where the data lives

- **Portal:** [TCIA — Low-Dose CT Image and Projection Data](https://www.cancerimagingarchive.net/collection/ldct-and-projection-data/)
- **DOI:** [10.7937/9NPB-2637](https://doi.org/10.7937/9NPB-2637)
- **Format:** DICOM-CT-PD (projection data) + standard DICOM (images),
  organised per-case: one directory per patient.
- **Per case:** ~4 GB on disk (1 projection-data series + 1 reconstructed
  image series).

## Can we fetch only parts? Yes.

TCIA exposes three partial-fetch routes:

### 1. NBIA REST API + manifest (preferred)

TCIA's Imaging Server API (`https://services.cancerimagingarchive.net/services/v4/TCIA/query/`)
accepts:

- `getCollectionValues` → list of collections
- `getPatientStudy?Collection=LDCT-and-Projection-Data` → list of studies
- `getSeries?Collection=LDCT-and-Projection-Data&PatientID=<id>` → series UIDs
- `getImage?SeriesInstanceUID=<uid>` → returns a ZIP of that single series

By calling `getImage` only on the SeriesInstanceUIDs you want, you pull a
single patient's projection + image data in ~4 GB instead of the full
1.32 TB. **This is the right design.**

Wagner et al. specify the case IDs in their paper supplementary; we'd hard-
code those PatientIDs in `fetch_mayo_ldct.py`.

### 2. NBIA Data Retriever CLI

TCIA ships a Java GUI/CLI called *NBIA Data Retriever* that consumes a
`.tcia` manifest file (exported from the portal). It also does per-series
downloads, but adds a Java install dependency we don't need.

### 3. Pre-built S3 bucket (when available)

Some TCIA collections are mirrored to public S3 buckets for `aws s3 cp`
download. Check `aws s3 ls s3://tcia-public/<collection>/` — if the LDCT
collection is mirrored, this is the fastest route. Not all collections are
mirrored.

## Recommended subset

Wagner et al. 2023 used **10 patient scans** split:
- 4 train, 1 val, 5 test
- Patient IDs published in the paper's supplementary table.

Size on disk after rebinning to our `FanBeamGeometry` (1152 views × 736
det × 512² image): ~100–200 GB. Within the social budget.

## Skeleton `fetch_mayo_ldct.py` (not yet implemented)

```python
WAGNER_PATIENT_IDS = [
    # From Wagner et al. 2023 Sec. 5 — verify against their supplementary
    # before committing checksums.
    "L067",  # placeholder — replace with the actual IDs
    ...
]

def fetch_raw(raw_dir):
    api = "https://services.cancerimagingarchive.net/services/v4/TCIA/query"
    for pid in WAGNER_PATIENT_IDS:
        series = http_json(
            f"{api}/getSeries?Collection=LDCT-and-Projection-Data&PatientID={pid}"
        )
        for s in series:
            uid = s["SeriesInstanceUID"]
            zip_path = raw_dir / pid / f"{uid}.zip"
            if not zip_path.exists():
                download(
                    f"{api}/getImage?SeriesInstanceUID={uid}",
                    zip_path,
                )
            unzip(zip_path, raw_dir / pid / uid)
```

Conversion (`stage_h5`) is the hard part: DICOM-CT-PD projection data needs
the parser from PYRO-NN or a hand-rolled one based on the format spec
(Mayo-PD doc shipped with the dataset). Rebinning to our geometry is what
`ddssl_ldct.geometry` is for.

## Decision needed before implementing

1. **Confirm the exact Wagner patient IDs.** Currently a placeholder above.
2. **TCIA credentials.** Most LDCT collections require a Limited Access
   agreement clicked through the TCIA portal once. Confirm whether `getImage`
   works anonymously or needs a session token (`api_key` header).
3. **Disk budget for rebinned cache.** ~150 GB on top of raw DICOMs. Tight
   if we also stage other datasets — may want to delete `raw/` after staging.
