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

import numpy as np

from _common import (
    assert_budget,
    data_root,
    download,
    pack_h5,
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
# pipeline uses. See ddssl_ldct/geometry.py for cross-references. The staged
# truth images are 512x512 with PixelSpacing ~0.586 mm — DICOM-native; the
# harness forward-projects through its OWN sparse-view geometry at train time.
GEOMETRY = {
    "image_size": 512,
    "pixel_spacing_dicom_mm": 0.5859375,
    "n_angles": 1152,
    "n_det": 736,
    "det_spacing": 1.2858,
    "sod": 595.0,
    "sdd": 1085.6,
}

# DICOM SOP Class UID for axial CT Image Storage. The projection-data series
# (1.2.840.10008.5.1.4.1.1.66 = Raw Data Storage) is also present but we use
# the reconstructed full-dose images as the clean truth — sparse-view
# sinograms are simulated by the harness via forward projection through
# `ddssl_ldct.geometry.FanBeamGeometry`.
SOP_CT_IMAGE = "1.2.840.10008.5.1.4.1.1.2"
MU_WATER_PER_MM = 0.02       # convention shared by ddssl_ldct.phantoms


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
def _hu_to_mu(hu: np.ndarray) -> np.ndarray:
    """Hounsfield Units -> linear attenuation (mm^-1). Convention matches
    ddssl_ldct.phantoms.random_ellipses_phantom (water = 0.02 mm^-1)."""
    return MU_WATER_PER_MM * (1.0 + hu.astype(np.float32) / 1000.0)


def _find_fulldose_image_series(patient_dir: Path):
    """Return (sorted_dicom_paths, sample_ds) for the Full-Dose Images
    series, or (None, None) if not present. Sort key is ImagePositionPatient[2]
    (z) so slices come out in axial order."""
    import pydicom
    for series_dir in sorted(patient_dir.iterdir()):
        if not series_dir.is_dir():
            continue
        # Read the first file to identify the series.
        sample = next(series_dir.iterdir(), None)
        if sample is None:
            continue
        try:
            ds = pydicom.dcmread(str(sample), stop_before_pixels=True)
        except Exception:
            continue
        if getattr(ds, "SOPClassUID", "") != SOP_CT_IMAGE:
            continue
        desc = getattr(ds, "SeriesDescription", "").lower()
        if "full" not in desc or "image" not in desc:
            continue
        # Sort slices by z position.
        files: list[tuple[float, Path]] = []
        for fp in series_dir.iterdir():
            try:
                meta = pydicom.dcmread(str(fp), stop_before_pixels=True)
                z = float(meta.ImagePositionPatient[2])
            except Exception:
                continue
            files.append((z, fp))
        files.sort()
        return [fp for _, fp in files], ds
    return None, None


def _load_slice_mu(path: Path) -> np.ndarray:
    """Read one DICOM slice and return (H, W) float32 μ in mm^-1."""
    import pydicom
    ds = pydicom.dcmread(str(path))
    pixels = ds.pixel_array.astype(np.float32)
    slope = float(getattr(ds, "RescaleSlope", 1.0))
    intercept = float(getattr(ds, "RescaleIntercept", 0.0))
    hu = pixels * slope + intercept
    return _hu_to_mu(hu)


def _collect_split(patient_ids: list[str], raw_dir: Path,
                   max_slices_per_patient: int | None = None
                   ) -> list[tuple[str, int, Path]]:
    """For each patient in a split, list the (pid, slice_idx, path) triples
    we will pack."""
    out: list[tuple[str, int, Path]] = []
    for pid in patient_ids:
        pdir = raw_dir / pid
        files, ds0 = _find_fulldose_image_series(pdir)
        if not files:
            print(f"[stage] {pid}: no Full-Dose Images series found, skip.",
                  flush=True)
            continue
        if max_slices_per_patient is not None:
            files = files[:max_slices_per_patient]
        print(f"[stage] {pid}: {len(files)} full-dose slices", flush=True)
        for k, fp in enumerate(files):
            out.append((pid, k, fp))
    return out


def stage_h5(raw_per_patient: dict[str, list[Path]], staged_dir: Path, *,
             shuffle_seed: int = 20260516) -> dict:
    """Pack full-dose recon images (in μ mm^-1) into per-split truth HDF5s.

    Sinograms are NOT pre-computed — the harness forward-projects each
    truth through its challenge geometry at train time. This keeps the
    staged size small (~1 GB total) and lets the same staged data serve
    multiple sparse-view geometries.

    Returns the splits dict suitable for `write_manifest`.
    """
    staged_dir.mkdir(parents=True, exist_ok=True)
    splits_out: dict[str, int] = {}
    for split, pids in WAGNER_SPLITS.items():
        triples = _collect_split(pids, raw_per_patient_to_raw_dir(raw_per_patient))
        if not triples:
            raise RuntimeError(f"split={split!r}: no slices collected")
        rng = np.random.default_rng(np.uint64(shuffle_seed + hash(split) % (1 << 31)))
        perm = rng.permutation(len(triples))
        triples_ordered = [triples[i] for i in perm]

        def emitter(t=triples_ordered):
            for i, (_pid, _k, fp) in enumerate(t):
                yield i, _load_slice_mu(fp)

        out = staged_dir / f"{split}_truth.h5"
        pack_h5(out, name="image",
                shape=(len(triples_ordered), 512, 512), dtype="float32",
                cases=emitter())
        splits_out[split] = len(triples_ordered)
        print(f"[stage] {split}: {len(triples_ordered)} slices -> {out.name}",
              flush=True)
    return splits_out


def raw_per_patient_to_raw_dir(raw_per_patient: dict) -> Path:
    """Pull the parent raw/ dir out of an arbitrary raw_per_patient entry."""
    for series_list in raw_per_patient.values():
        for s in series_list:
            return Path(s).parent.parent
    raise RuntimeError("raw_per_patient is empty; cannot derive raw_dir")


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
        splits = stage_h5(raw_per_patient, staged_dir)
        write_manifest(
            staged_dir,
            source=f"https://doi.org/10.7937/9NPB-2637 ({TCIA_COLLECTION})",
            geometry=GEOMETRY,
            splits=splits,
            extra={"wagner_patient_ids": WAGNER_PATIENT_IDS,
                   "wagner_splits": WAGNER_SPLITS,
                   "layout": "truth-only μ (mm^-1); harness forward-projects",
                   "sop_class_used": SOP_CT_IMAGE,
                   "series_description_filter": "Full Dose Images"},
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
