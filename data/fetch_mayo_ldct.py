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
import os
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

# Track A (A1/A2) of docs/workplan_real_datasets.md: helical-to-fan rebinning
# of the DICOM-CT-PD projection series. The actual NumPy implementation lives
# in ddssl_ldct.helix2fan; this module orchestrates the per-patient pass and
# writes the staged HDF5 outputs.
SOP_RAW_DATA = "1.2.840.10008.5.1.4.1.1.66"   # DICOM-CT-PD Raw Data Storage

CHALLENGE = "mayo_ldct"
TCIA_COLLECTION = "LDCT-and-Projection-data"  # case-sensitive ("data" lowercase per API)
TCIA_API = "https://services.cancerimagingarchive.net/nbia-api/services/v1"

# -----------------------------------------------------------------------------
# Wagner subset patient IDs.
#
# Wagner et al. 2023 ("Self-supervised dual-domain image denoising for
# low-dose CT", arXiv:2211.01111) and the companion repo
# The exact Wagner split (Wagner et al. 2022/2023) is pinned by patient ID:
#   train: L145, L186, L209, L219      (4 patients)
#   val:   L277                         (1 patient)
#   test:  L014, L056, L058, L075, L123 (5 patients)
# User-supplied 2026-05-19. Earlier defaults (L004 / L033 / L064 / L107 / L143
# / L186 / L221 / L260 / L288 / L299) were a placeholder set that happened to
# match the *structure* but not the *exact* IDs in Wagner's experiments. They
# have been retired in favor of the pinned IDs above.
WAGNER_SPLITS = {
    "train": ["L145", "L186", "L209", "L219"],
    "val":   ["L277"],
    "test":  ["L014", "L056", "L058", "L075", "L123"],
}
WAGNER_PATIENT_IDS = (WAGNER_SPLITS["train"] + WAGNER_SPLITS["val"] +
                     WAGNER_SPLITS["test"])

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
# Track A (A1/A2): helical -> 2D fan-beam rebinning of the projection series.
# -----------------------------------------------------------------------------

def _find_projection_series(patient_dir: Path,
                            description_filter: str) -> Path | None:
    """Return the raw-data DICOM-CT-PD series dir whose SeriesDescription
    contains `description_filter` (case-insensitive), or None if not present.

    Two series of interest per patient: "Full dose projections" and
    "Low dose projections" (both SOP `1.2.840.10008.5.1.4.1.1.66`).
    """
    import pydicom
    target = description_filter.lower()
    for series_dir in sorted(patient_dir.iterdir()):
        if not series_dir.is_dir():
            continue
        sample = next(series_dir.iterdir(), None)
        if sample is None:
            continue
        try:
            ds = pydicom.dcmread(str(sample), stop_before_pixels=True)
        except Exception:
            continue
        if getattr(ds, "SOPClassUID", "") != SOP_RAW_DATA:
            continue
        desc = getattr(ds, "SeriesDescription", "").lower()
        if target in desc:
            return series_dir
    return None


SOP_CT_IMAGE = "1.2.840.10008.5.1.4.1.1.2"


def _find_image_series(patient_dir: Path,
                       description_filter: str) -> Path | None:
    """Return the truth-image (CT-image SOP) DICOM-series directory whose
    SeriesDescription contains ``description_filter`` (case-insensitive),
    or ``None`` if not present. Companion to ``_find_projection_series``
    (which filters on the raw-data SOP).
    """
    import pydicom
    target = description_filter.lower()
    for series_dir in sorted(patient_dir.iterdir()):
        if not series_dir.is_dir():
            continue
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
        if target in desc:
            return series_dir
    return None


def _truth_z_anchor_source(image_series_dir: Path) -> float:
    """Return the source-frame z anchor for the Mayo HFS sign-flip
    convention: take the maximum ``ImagePositionPatient[2]`` across all
    truth slices (the most-superior patient z) and negate it.

    The choice of anchor is arbitrary (any truth slice works since
    they're on a regular 3-mm grid in patient frame), but the
    most-superior slice is unambiguously the smallest source z, which
    keeps the alignment shift small.
    """
    import pydicom
    zs = []
    for fp in image_series_dir.iterdir():
        try:
            m = pydicom.dcmread(str(fp), stop_before_pixels=True)
            zs.append(float(m.ImagePositionPatient[2]))
        except Exception:
            continue
    if not zs:
        raise RuntimeError(
            f"no usable truth DICOMs found in {image_series_dir}"
        )
    return -max(zs)


def _rebin_patient_series(series_dir: Path, out_h5: Path, out_zgrid: Path,
                          out_geom: Path, *,
                          dv_rebinned: float, n_jobs: int,
                          z_truth_anchor: float | None = None) -> dict:
    """Run the full curved->flat->fan pipeline on one DICOM-CT-PD series.

    If ``z_truth_anchor`` (source frame, mm) is provided, the output
    z_start is shifted so the sino z-grid lands on the Mayo truth z
    grid (every dv_rebinned-th slice hits a truth slice exactly).

    Writes `out_h5` (key "sino", shape (rotview, nu, nz) float32),
    `out_zgrid` (npy: per-output-slice z positions),
    `out_geom` (json: geometry dict — pydicom-derived).

    Returns a small status dict for the manifest entry.
    """
    # Import here so the top-level fetch path doesn't drag in joblib/pydicom
    # at module-import time on non-rebinning runs.
    import h5py  # type: ignore

    REPO = Path(__file__).resolve().parents[1]
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))
    from ddssl_ldct.helix2fan import (
        read_dicom_ctpd, rebin_curved_to_flat, rebin_helical_to_fan,
    )

    print(f"[rebin] reading {series_dir}", flush=True)
    proj_curved, geom = read_dicom_ctpd(series_dir)
    print(f"[rebin]   proj_curved shape={proj_curved.shape} "
          f"sdd={geom['sdd']:.3f} sod={geom['sod']:.3f} du={geom['du']:.4f} "
          f"dv={geom['dv']:.4f} total_rot={geom['total_rotations']:.3f}",
          flush=True)

    # Optional override: replace the DICOM-nominal SSR sod/sdd (and the
    # newly-added z-scaling s_z) with the v3 multi-GT fitted optimum
    # (SLURM 763384 → MAYO_LDCT_SSR_DEFAULTS in ddssl_ldct/geometry.py).
    # Gated by env var so the legacy DICOM-nominal path stays the default
    # for backwards compatibility. See findings.md 2026-06-12 entry for
    # the v3 / s_z rationale.
    if os.environ.get("HELIX2FAN_SSR_FITTED", "0") in ("1", "true", "True"):
        from ddssl_ldct.geometry import MAYO_LDCT_SSR_DEFAULTS
        sod_old, sdd_old = float(geom["sod"]), float(geom["sdd"])
        sod_new = float(MAYO_LDCT_SSR_DEFAULTS["sod"])
        sdd_new = float(MAYO_LDCT_SSR_DEFAULTS["sdd"])
        s_z_new = float(MAYO_LDCT_SSR_DEFAULTS.get("s_z", 1.0))
        print(f"[rebin]   SSR sod {sod_old:.3f} → {sod_new:.3f} "
              f"(Δ = {sod_new - sod_old:+.3f} mm; v3)", flush=True)
        print(f"[rebin]   SSR sdd {sdd_old:.3f} → {sdd_new:.3f} "
              f"(Δ = {sdd_new - sdd_old:+.3f} mm; v3)", flush=True)
        print(f"[rebin]   SSR s_z 1.0 → {s_z_new:.6f} "
              f"(Δ = {(s_z_new - 1.0) * 100:+.4f} % z-scaling; v3)",
              flush=True)
        geom["sod"] = sod_new
        geom["sdd"] = sdd_new
        geom["s_z"] = s_z_new

    print(f"[rebin]   curved -> flat (n_jobs={n_jobs}) ...", flush=True)
    proj_flat = rebin_curved_to_flat(proj_curved, geom, n_jobs=n_jobs)
    del proj_curved  # free ~3 GB

    z_pos = np.asarray(geom["z_positions"], dtype=np.float64)
    z_min, z_max = float(np.nanmin(z_pos)), float(np.nanmax(z_pos))
    pitch = abs(float(geom["pitch_mm"]))
    # Output covers (z_min + 0.5*pitch, z_max - 0.5*pitch) so every output
    # slice has a full helical window of contributing readouts.
    z_start = z_min + 0.5 * pitch
    z_end = z_max - 0.5 * pitch

    # Optional truth-grid z alignment. The Mayo truth DICOMs sit at
    # patient-frame z = z_truth_first - k * 3 mm for k = 0, 1, 2, ...
    # After our sign-flip mapping (patient_z = -source_z), the truth
    # z's land in the source frame at z_truth_anchor + k * 3 mm. To
    # have every dv_rebinned-th sino slice hit a truth slice exactly,
    # we shift z_start so that (z_start - z_truth_anchor) is an
    # integer multiple of dv_rebinned. Shift is ≤ dv_rebinned/2 in
    # either direction, then bumped up by dv_rebinned if it would
    # fall below z_min + 0.5*pitch.
    # `z_truth_anchor` is a function-level argument (see signature)
    if z_truth_anchor is not None:
        shift_mod = (z_truth_anchor - z_start) % dv_rebinned
        if shift_mod <= dv_rebinned / 2.0:
            z_start_new = z_start + shift_mod
        else:
            z_start_new = z_start - (dv_rebinned - shift_mod)
        while z_start_new < z_min + 0.5 * pitch - 1e-6:
            z_start_new += dv_rebinned
        print(f"[rebin]   z_start aligned to truth grid: "
              f"{z_start:.4f} → {z_start_new:.4f} mm (anchor={z_truth_anchor:.4f}, "
              f"shift={z_start_new - z_start:+.4f} mm)", flush=True)
        z_start = z_start_new

    nz_rebinned = max(1, int(np.floor((z_end - z_start) / dv_rebinned)) + 1)
    print(f"[rebin]   helical -> fan SSR (rotview~{int(round(proj_flat.shape[0] / geom['total_rotations']))}, "
          f"nz={nz_rebinned}, dv_rebinned={dv_rebinned})", flush=True)
    # Read FFS-drho correction toggle from env (set by sbatch); default off
    # for backwards compatibility with the legacy bulk-rebin pipeline.
    import os as _os
    ffs_correct_drho = _os.environ.get("HELIX2FAN_FFS_DRHO", "0") in ("1", "true", "True")
    if ffs_correct_drho:
        print(f"[rebin]   FFS-drho correction ENABLED (per-readout effective "
              f"sod/sdd from ffs_drho tag)", flush=True)
    rebinned, z_grid = rebin_helical_to_fan(
        proj_flat, geom,
        dv_rebinned=dv_rebinned,
        nz_rebinned=nz_rebinned,
        z_start=z_start,
        n_jobs=n_jobs,
        ffs_correct_drho=ffs_correct_drho,
    )
    del proj_flat

    out_h5.parent.mkdir(parents=True, exist_ok=True)
    print(f"[rebin]   writing {out_h5} shape={rebinned.shape}", flush=True)
    with h5py.File(out_h5, "w", libver="latest") as f:
        f.create_dataset("sino", data=rebinned, dtype="float32",
                         chunks=(1, rebinned.shape[1], rebinned.shape[2]),
                         compression="gzip", compression_opts=1)
    np.save(out_zgrid, z_grid)
    # geometry sidecar: strip the large per-readout arrays out of the dict
    # before json-encoding.
    geom_json = {k: (v if not isinstance(v, np.ndarray) else None)
                 for k, v in geom.items()}
    geom_json["n_proj"] = int(geom["n_proj"])
    geom_json["rotview"] = int(rebinned.shape[0])
    geom_json["nu"] = int(rebinned.shape[1])
    geom_json["nz_rebinned"] = int(rebinned.shape[2])
    geom_json["dv_rebinned"] = float(dv_rebinned)
    geom_json["z_start"] = float(z_start)
    out_geom.write_text(json.dumps(geom_json, indent=2, default=float))
    return {
        "series": str(series_dir),
        "h5": str(out_h5),
        "rotview": int(rebinned.shape[0]),
        "nu": int(rebinned.shape[1]),
        "nz": int(rebinned.shape[2]),
        "z_start": float(z_start),
        "z_step": float(dv_rebinned),
    }


def stage_h5_with_sino(raw_dir: Path, staged_dir: Path, *,
                       dv_rebinned: float = 1.0, n_jobs: int = -1) -> dict:
    """Iterate Wagner patients, rebin Full + Low dose projection series.

    Writes per-patient files into `staged_dir`:
      L<NNN>_sino_fulldose.h5    (key "sino", (rotview, nu, nz))
      L<NNN>_sino_fulldose_z_grid.npy
      L<NNN>_sino_fulldose_geometry.json
      ... and the lowdose triple.
    """
    staged_dir.mkdir(parents=True, exist_ok=True)
    report: dict = {"patients": {}}
    for pid in WAGNER_PATIENT_IDS:
        pdir = raw_dir / pid
        if not pdir.exists():
            print(f"[rebin] {pid}: no raw/ dir, skip.", flush=True)
            continue
        report["patients"][pid] = {}
        for kind, desc in (("fulldose", "full dose projections"),
                           ("lowdose",  "low dose projections")):
            series_dir = _find_projection_series(pdir, desc)
            if series_dir is None:
                print(f"[rebin] {pid}: no '{desc}' series, skip.", flush=True)
                continue
            # When HELIX2FAN_Z_ALIGN=1 we emit `*_aligned.h5` so the
            # old (un-aligned) bulk-rebin outputs stay around for
            # comparison. We also locate the matching truth-image
            # series for this patient/dose to extract its z anchor.
            import os as _os
            z_align = _os.environ.get("HELIX2FAN_Z_ALIGN", "0") in ("1", "true", "True")
            suffix = "_aligned" if z_align else ""
            out_h5 = staged_dir / f"{pid}_sino_{kind}{suffix}.h5"
            out_z = staged_dir / f"{pid}_sino_{kind}{suffix}_z_grid.npy"
            out_g = staged_dir / f"{pid}_sino_{kind}{suffix}_geometry.json"
            if out_h5.exists():
                print(f"[rebin] {pid}/{kind}: {out_h5.name} exists, skip.",
                      flush=True)
                continue

            z_truth_anchor = None
            if z_align:
                # Truth image-series DICOM tag whose ImagePositionPatient[2]
                # gives the first (most-superior) reconstructed slice in
                # patient frame. We sign-flip to the source frame because
                # that's where z_start lives.
                truth_desc = f"{'full' if kind == 'fulldose' else 'low'} dose image"
                truth_series = _find_image_series(pdir, truth_desc)
                if truth_series is None:
                    print(f"[rebin] {pid}/{kind}: WARN — no '{truth_desc}' "
                          f"series found; cannot derive z anchor, falling "
                          f"back to unaligned z_start.", flush=True)
                else:
                    z_truth_anchor = _truth_z_anchor_source(truth_series)
                    print(f"[rebin] {pid}/{kind}: z truth anchor "
                          f"(source frame) = {z_truth_anchor:.4f} mm "
                          f"from {truth_series.name}", flush=True)

            try:
                info = _rebin_patient_series(
                    series_dir, out_h5, out_z, out_g,
                    dv_rebinned=dv_rebinned, n_jobs=n_jobs,
                    z_truth_anchor=z_truth_anchor,
                )
                report["patients"][pid][kind] = info
            except Exception as e:  # pragma: no cover  (logged + skip)
                print(f"[rebin] {pid}/{kind} FAILED: {e}", flush=True)
                report["patients"][pid][kind] = {"error": str(e)}
    (staged_dir / "rebin_manifest.json").write_text(
        json.dumps(report, indent=2, default=str)
    )
    return report


# -----------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", default=None)
    p.add_argument("--skip-download", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--force", action="store_true",
                   help="Re-stage even if staged/manifest.json exists "
                        "(overwrites the existing HDF5 files in place).")
    p.add_argument("--rebin-only", action="store_true",
                   help="Skip download + truth-staging; only run the "
                        "helix2fan rebinning over the existing raw/ tree "
                        "into <data-root>/mayo_ldct/staged_helix2fan/.")
    p.add_argument("--dv-rebinned", type=float, default=1.0,
                   help="Axial output slice spacing in mm (default 1.0).")
    p.add_argument("--n-jobs", type=int, default=-1,
                   help="joblib n_jobs for the rebinning passes (default -1).")
    args = p.parse_args(argv)

    root = data_root(args.data_root)
    challenge_dir = root / CHALLENGE
    raw_dir = challenge_dir / "raw"
    staged_dir = challenge_dir / "staged"
    # Output subdir for the helix2fan-rebinned sinos. Default
    # "staged_helix2fan/" matches the legacy DICOM-nominal rebin output;
    # set STAGED_HELIX2FAN_SUBDIR to e.g. "staged_helix2fan_ssr_fitted"
    # to write the new SSR-fitted rebin into a sibling directory without
    # clobbering the existing 86 GB of legacy data.
    staged_sino_subdir = os.environ.get("STAGED_HELIX2FAN_SUBDIR",
                                          "staged_helix2fan")
    staged_sino_dir = challenge_dir / staged_sino_subdir

    print(f"[plan] AGENT4CT_DATA = {root}")
    print(f"[plan] subset = Wagner 10 patients = {WAGNER_PATIENT_IDS}")
    print(f"[plan] splits = {WAGNER_SPLITS}")
    print(f"[plan] estimated raw  ~ 40 GB at {raw_dir}")
    print(f"[plan] estimated stage ~ 150 GB at {staged_dir} (rebinned)")
    if args.dry_run:
        return 0

    if args.rebin_only:
        if not raw_dir.exists():
            print(f"--rebin-only: raw/ tree not found at {raw_dir}",
                  file=sys.stderr)
            return 1
        print(f"[rebin] mode=--rebin-only target={staged_sino_dir}")
        stage_h5_with_sino(raw_dir, staged_sino_dir,
                           dv_rebinned=args.dv_rebinned,
                           n_jobs=args.n_jobs)
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

    if staged_dir.exists() and (staged_dir / "manifest.json").exists() and not args.force:
        print(f"[stage] {staged_dir}/manifest.json present — pass --force to re-stage.")
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
