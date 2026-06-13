#!/usr/bin/env python -u
"""Probe per-patient truth-image geometry for all 10 Wagner Mayo patients.

The 2026-06-13 GT/HD/LD comparison showed every patient's FBP recon is a
valid reconstruction but at a different in-plane ZOOM and SHIFT relative
to its truth image. This script reads the reconstructed "Full Dose
Images" DICOM headers (NO pixel data, fast, login-node safe) and reports,
per patient, the two quantities that set zoom + shift:

  * voxel size (zoom): PixelSpacing[0]  + ReconstructionDiameter
  * in-plane shift: the recon FOV was panned off the scanner isocentre.
        DataCollectionCenterPatient   (0018,9313) = isocentre (FBP axis)
        ReconstructionTargetCenterPatient (0018,9318) = truth image centre
        shift = ReconTargetCenter - DataCollectionCenter   (x, y)
    Cross-check via ImagePositionPatient:
        image_centre = IPP[0:2] + ((N-1)/2) * ps
        shift_ipp = image_centre - DataCollectionCenter

Output: results/breast_debug/wagner_truth_geometry.json + a printed table.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pydicom

REPO = Path(__file__).resolve().parents[1]

WAGNER_ALL = ["L145", "L186", "L209", "L219", "L277",
              "L014", "L056", "L058", "L075", "L123"]
SPLIT = {**{p: "train" for p in ["L145", "L186", "L209", "L219"]},
         "L277": "val",
         **{p: "test" for p in ["L014", "L056", "L058", "L075", "L123"]}}
SOP_CT = "1.2.840.10008.5.1.4.1.1.2"


def _vec(v):
    return [float(x) for x in v]


def probe_patient(patient_dir: Path) -> dict:
    """Read one Full-Dose-Images slice header and extract geometry tags."""
    series = None
    for sd in sorted(patient_dir.iterdir()):
        if not sd.is_dir():
            continue
        f0 = next(sd.iterdir(), None)
        if f0 is None:
            continue
        try:
            h = pydicom.dcmread(str(f0), stop_before_pixels=True)
        except Exception:
            continue
        desc = getattr(h, "SeriesDescription", "").lower()
        if getattr(h, "SOPClassUID", "") == SOP_CT and "full" in desc and "image" in desc:
            series = sd
            break
    if series is None:
        return {"error": "no Full Dose Images series"}

    f0 = next(series.iterdir())
    ds = pydicom.dcmread(str(f0), stop_before_pixels=True)
    ps = float(ds.PixelSpacing[0])
    rows, cols = int(ds.Rows), int(ds.Columns)
    ipp = _vec(ds.ImagePositionPatient)

    def tag(t1, t2):
        try:
            return _vec(ds[t1, t2].value)
        except KeyError:
            return None

    dcc = tag(0x0018, 0x9313)   # DataCollectionCenterPatient (isocentre)
    rtc = tag(0x0018, 0x9318)   # ReconstructionTargetCenterPatient
    recon_diam = None
    try:
        recon_diam = float(ds[0x0018, 0x1100].value)
    except KeyError:
        pass

    # image centre in patient coords (x, y)
    img_cx = ipp[0] + (cols - 1) / 2.0 * ps
    img_cy = ipp[1] + (rows - 1) / 2.0 * ps

    out = {
        "series_desc": str(ds.SeriesDescription),
        "pixel_spacing": ps,
        "rows": rows, "cols": cols,
        "fov_mm": ps * cols,
        "recon_diameter": recon_diam,
        "ipp": ipp,
        "image_center_xy": [img_cx, img_cy],
        "data_collection_center": dcc,
        "recon_target_center": rtc,
    }
    if dcc is not None and rtc is not None:
        out["shift_tag_xy"] = [rtc[0] - dcc[0], rtc[1] - dcc[1]]
    if dcc is not None:
        out["shift_ipp_xy"] = [img_cx - dcc[0], img_cy - dcc[1]]
    return out


def main() -> int:
    import os
    root = Path(os.environ.get("AGENT4CT_DATA",
                                "/cluster/maier/Agent4CT/data")) / "mayo_ldct"
    raw = root / "raw"

    results = {}
    print(f"{'pat':5} {'spl':5} {'ps':>7} {'FOV':>6} {'reconD':>7} "
          f"{'shift_tag(x,y)':>18} {'shift_ipp(x,y)':>18}", flush=True)
    for pat in WAGNER_ALL:
        pdir = raw / pat
        if not pdir.exists():
            results[pat] = {"error": "no raw dir"}
            print(f"{pat:5} {SPLIT[pat]:5}  NO RAW DIR", flush=True)
            continue
        info = probe_patient(pdir)
        info["split"] = SPLIT[pat]
        results[pat] = info
        if "error" in info:
            print(f"{pat:5} {SPLIT[pat]:5}  ERROR: {info['error']}", flush=True)
            continue
        st = info.get("shift_tag_xy")
        si = info.get("shift_ipp_xy")
        st_s = f"({st[0]:+6.1f},{st[1]:+6.1f})" if st else "       n/a        "
        si_s = f"({si[0]:+6.1f},{si[1]:+6.1f})" if si else "       n/a        "
        print(f"{pat:5} {info['split']:5} {info['pixel_spacing']:7.4f} "
              f"{info['fov_mm']:6.1f} "
              f"{(info['recon_diameter'] or -1):7.1f} {st_s:>18} {si_s:>18}",
              flush=True)

    out_json = REPO / "results" / "breast_debug" / "wagner_truth_geometry.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(results, indent=2))
    print(f"\n[probe] wrote {out_json}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
