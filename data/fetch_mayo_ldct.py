"""Fetch + stage the Mayo LDCT Wagner-subset (10 patients).

Full dataset is 1.32 TB across 299 patients at TCIA. We pull only the
subset Wagner et al. 2023 (arXiv:2211.01111) train + val + test on, via the
NBIA REST API's per-series getImage endpoint. Per-patient downloads run
~4 GB each, so 10 patients is ~40 GB raw -- well inside the social budget.

Source: TCIA collection `LDCT-and-Projection-Data`,
        DOI: https://doi.org/10.7937/9NPB-2637

Public-data NBIA endpoints require NO authentication (chest "C*" and
abdomen "L*" subjects are CC BY 4.0; the head "N*" cases are NIH-controlled
and would need a separate access agreement -- this script does NOT pull
those).

Layout produced:

    mayo_ldct/
        raw/
            <PatientID>/<SeriesInstanceUID>/*.dcm     # extracted ZIPs
        staged/
            train_sinograms.h5  (N_train, A_rebin, D_rebin) float32
            train_truth.h5      (N_train, H, W)             float32
            val_*.h5
            test_*.h5
            manifest.json

Run from the cluster:
    python data/fetch_mayo_ldct.py
"""

from __future__ import annotations
import argparse
import json
import sys
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path

from _common import (
    assert_budget,
    data_root,
    download,
    write_manifest,
)

CHALLENGE = "mayo_ldct"
TCIA_COLLECTION = "LDCT-and-Projection-data"  # case-sensitive ("data" lowercase per API)
TCIA_API = "https://services.cancerimagingarchive.net/nbia-api/services/v1"

# -----------------------------------------------------------------------------
# Wagner subset patient IDs.
#
# Wagner et al. 2023 ("Self-supervised dual-domain image denoising for
# low-dose CT", arXiv:2211.01111) and the companion repo
# https://github.com/faebstn96/helix2fan together describe the *methodology*
# (four training, one validation, five test abdomen scans) but do NOT pin
# specific TCIA PatientIDs. The choice of which 10 L* (abdomen) cases is
# left to the implementer. The ten below are a sensible default — any ten
# L* cases will reproduce Wagner's setup, since all L-cases share the same
# Siemens scanner geometry and reconstruction parameters.
#
# To change the subset: replace this list with any 10 PatientIDs that
# exist in https://www.cancerimagingarchive.net/collection/ldct-and-projection-data/
# (filter on Subject ID starts with "L"). Re-running with a new list will
# only redownload the missing patients; existing ones are cached.
# Valid L* IDs in the collection (100 abdomen cases) confirmed via
# getPatient API 2026-05-15. Picked 10 spread across the ID range.
WAGNER_PATIENT_IDS = [
    "L004", "L033", "L064", "L107", "L143",
    "L186", "L221", "L260", "L288", "L299",
]
# Verified 2026-05-15: every ID above appears in the
# getPatient?Collection=LDCT-and-Projection-data response. L067 looked
# right but does NOT exist (valid L* IDs skip non-monotonically: L064 is
# present, L067 is not, L071 is next). Collection name is case-sensitive:
# "...data" not "...Data".

# Wagner's split per the paper: 4 train, 1 val, 5 test.
WAGNER_SPLITS = {
    "train": WAGNER_PATIENT_IDS[:4],
    "val":   WAGNER_PATIENT_IDS[4:5],
    "test":  WAGNER_PATIENT_IDS[5:],
}

# Mayo Siemens SOMATOM Definition AS, rebinned to the fan-beam geometry our
# pipeline uses. See ddssl_ldct/geometry.py for cross-references.
GEOMETRY = {
    "image_size": 512,
    "pixel_spacing": 0.7,
    "n_angles": 1152,
    "n_det": 736,
    "det_spacing": 1.2858,
    "sod": 595.0,
    "sdd": 1085.6,
}


# -----------------------------------------------------------------------------
def http_json(url: str) -> list | dict:
    with urllib.request.urlopen(url) as r:
        return json.load(r)


def list_series_for_patient(patient_id: str) -> list[dict]:
    """Return the series records for one patient in our collection."""
    url = (f"{TCIA_API}/getSeries?"
           f"Collection={urllib.parse.quote(TCIA_COLLECTION)}"
           f"&PatientID={urllib.parse.quote(patient_id)}"
           f"&format=json")
    return http_json(url)


def fetch_patient(patient_id: str, patient_raw_dir: Path) -> list[Path]:
    """Pull every series for one patient. Returns the extracted dir paths."""
    patient_raw_dir.mkdir(parents=True, exist_ok=True)
    series = list_series_for_patient(patient_id)
    if not series:
        raise RuntimeError(
            f"No series found for PatientID={patient_id}. Either the ID is "
            f"a placeholder (see WAGNER_PATIENT_IDS) or the collection name "
            f"is wrong. Confirm at https://services.cancerimagingarchive.net"
            f"/nbia-api/services/v1/getCollectionValues"
        )
    extracted = []
    for s in series:
        uid = s["SeriesInstanceUID"]
        zip_path = patient_raw_dir / f"{uid}.zip"
        extracted_dir = patient_raw_dir / uid
        if extracted_dir.exists() and any(extracted_dir.iterdir()):
            print(f"[fetch] {patient_id}/{uid} already extracted, skip.")
            extracted.append(extracted_dir)
            continue
        download(f"{TCIA_API}/getImage?SeriesInstanceUID={uid}", zip_path)
        with zipfile.ZipFile(zip_path) as z:
            extracted_dir.mkdir(parents=True, exist_ok=True)
            z.extractall(extracted_dir)
        zip_path.unlink()  # the on-disk DICOMs are what we need
        extracted.append(extracted_dir)
    return extracted


def fetch_raw(raw_dir: Path) -> dict[str, list[Path]]:
    """Pull all Wagner patients, return {patient_id: [series_dirs]}."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    out: dict[str, list[Path]] = {}
    for pid in WAGNER_PATIENT_IDS:
        print(f"\n[fetch] === patient {pid} ===")
        out[pid] = fetch_patient(pid, raw_dir / pid)
    return out


# -----------------------------------------------------------------------------
def stage_h5(raw_per_patient: dict[str, list[Path]], staged_dir: Path) -> None:
    """Convert Mayo DICOM-CT-PD per-patient series into the staged layout.

    Each patient ships two series:
      (a) reconstructed images (regular DICOM, axial slices)
      (b) projection data (DICOM-CT-PD, one frame per view; helical fan-beam)

    We need to:
      1. Identify which series is which (Modality tag: 'CT' for images,
         'CT' with manufacturer-specific tags for projection data).
      2. Rebin the (a) image series to (H, W) = (512, 512).
      3. Rebin the (b) projection data to the (n_angles, n_det) of our
         FanBeamGeometry via either:
           - a published rebinning function (Sidky's `mayo_rebin.py` if we
             vendor it), or
           - a forward-project of the reconstructed image through
             FanBeamGeometry (the simpler route for the iter-phase
             "surrogate" mode — used for initial integration).
      4. Pack into HDF5 with chunks=(1, A, D) / (1, H, W).

    Per-patient output: one ndarray per slice → many slices per H5.
    """
    raise NotImplementedError(
        "Mayo DICOM-CT-PD parsing is the hard half of this script and is not "
        "yet implemented. Two implementations are possible:\n"
        "  (a) Direct DICOM-CT-PD: requires the Mayo-PD parser (see Sidky's "
        "    code or the AAPM Mayo doc shipped with the dataset).\n"
        "  (b) Surrogate path: read only the reconstructed-image DICOMs and "
        "    forward-project through FanBeamGeometry to synthesise sinograms.\n"
        "Recommendation: implement (b) first to unblock training, then add "
        "(a) when we want the real sinogram noise distribution."
    )


# -----------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", default=None)
    p.add_argument("--skip-download", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args(argv)

    root = data_root(args.data_root)
    challenge_dir = root / CHALLENGE
    raw_dir = challenge_dir / "raw"
    staged_dir = challenge_dir / "staged"

    print(f"[plan] AGENT4CT_DATA = {root}")
    print(f"[plan] subset = Wagner 10 patients = {WAGNER_PATIENT_IDS}")
    print(f"[plan] splits = {WAGNER_SPLITS}")
    print(f"[plan] estimated raw  ~ 40 GB at {raw_dir}")
    print(f"[plan] estimated stage ~ 150 GB at {staged_dir} (rebinned)")
    if args.dry_run:
        return 0

    assert_budget(root, need_gb=200.0)

    if not args.skip_download:
        raw_per_patient = fetch_raw(raw_dir)
    else:
        raw_per_patient = {}
        for pid in WAGNER_PATIENT_IDS:
            p_dir = raw_dir / pid
            raw_per_patient[pid] = sorted(p_dir.iterdir()) if p_dir.exists() else []
        if not any(raw_per_patient.values()):
            print("--skip-download but raw/ is empty.", file=sys.stderr)
            return 1

    if staged_dir.exists() and (staged_dir / "manifest.json").exists():
        print(f"[stage] {staged_dir}/manifest.json present — skip.")
    else:
        stage_h5(raw_per_patient, staged_dir)
        write_manifest(
            staged_dir,
            source=f"https://doi.org/10.7937/9NPB-2637 ({TCIA_COLLECTION})",
            geometry=GEOMETRY,
            splits={k: len(v) for k, v in WAGNER_SPLITS.items()},
            extra={"wagner_patient_ids": WAGNER_PATIENT_IDS,
                   "wagner_splits": WAGNER_SPLITS},
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
