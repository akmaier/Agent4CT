#!/usr/bin/env python -u
"""Cache the post-curved-to-flat helical-projection data for L014/fulldose
so the downstream torch-differentiable helical→fan rebin fit doesn't have
to re-load 37k DICOMs every iteration.

For the L014 peak slice (target_pZ = −254.50, source_z = +256.86), we
only need the helix readouts whose z_src is within ±0.5·pitch of the
target, plus a one-rotation safety margin. That's roughly one full
rotation worth of data (≈ 2300 readouts × 64 × 736 × 4 bytes ≈ 430 MB).

Saved tensors:
  L014_proj_flat_peak.pt   — dict with:
    'proj_flat'       — (n_sub, nv, nu) float32 helix subset
    'z_positions'     — (n_sub,) source-frame z per readout
    'gantry_angles_corrected' — (n_sub,) gantry angle per readout
    'original_indices'        — (n_sub,) full-helix index for each subset element
    'rotview'         — int (2304)
    'n_proj_full'     — int (37982)
    'nu', 'nv', 'du', 'dv', 'sod', 'sdd', 'u0', 'v0', 'pitch_mm' — scalars
    'target_source_z' — float (256.86)
    'window_mm'       — float (mm of z covered by the saved subset)
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
if str(REPO / "data") not in sys.path:
    sys.path.insert(0, str(REPO / "data"))

from ddssl_ldct.helix2fan import (
    read_dicom_ctpd, rebin_curved_to_flat,
)
from data.fetch_mayo_ldct import _find_projection_series


def main() -> int:
    import os
    root = Path(os.environ.get("AGENT4CT_DATA",
                                "/cluster/maier/Agent4CT/data")) / "mayo_ldct"
    raw_dir = root / "raw" / "L014"
    out_dir = root / "staged_helix2fan"

    series_dir = _find_projection_series(raw_dir, "full dose projections")
    if series_dir is None:
        print(f"[cache] no fulldose projection series in {raw_dir}", file=sys.stderr)
        return 1
    print(f"[cache] reading {series_dir}", flush=True)
    proj_curved, geom = read_dicom_ctpd(series_dir)
    print(f"[cache] proj_curved shape={proj_curved.shape} "
          f"sdd={geom['sdd']:.3f} sod={geom['sod']:.3f} du={geom['du']:.4f} "
          f"dv={geom['dv']:.4f} pitch_mm={geom['pitch_mm']:.4f}",
          flush=True)

    print(f"[cache] curved -> flat (n_jobs=-1) …", flush=True)
    proj_flat = rebin_curved_to_flat(proj_curved, geom, n_jobs=-1)
    del proj_curved
    print(f"[cache] proj_flat shape={proj_flat.shape}", flush=True)

    z_positions = np.asarray(geom["z_positions"], dtype=np.float64)
    gantry_angles = np.asarray(geom.get("gantry_angles_corrected",
                                          geom.get("gantry_angles")), dtype=np.float64)

    rotview = int(round(proj_flat.shape[0] / geom["total_rotations"]))
    n_proj_full = int(proj_flat.shape[0])
    nu, nv = proj_flat.shape[2], proj_flat.shape[1]
    pitch_mm = float(geom["pitch_mm"])

    # Window around the L014 peak slice (target patient_z = -254.50)
    # In source frame this is +254.50. Take ±0.6·pitch (just above the
    # cone-beam window) so the SSR has room on both sides + a one-rotation
    # margin.
    target_source_z = 256.86       # = +z_grid[nz_middle + 4] in source frame
    half_window = max(1.2 * pitch_mm, 50.0)   # at least 50 mm to be safe
    mask = (z_positions >= target_source_z - half_window) & \
           (z_positions <= target_source_z + half_window)
    indices = np.where(mask)[0]
    print(f"[cache] subset: {indices.size} readouts in z ∈ "
          f"[{target_source_z - half_window:.2f}, {target_source_z + half_window:.2f}] mm "
          f"(half_window={half_window:.1f} mm = {half_window/pitch_mm:.2f}·pitch)",
          flush=True)

    proj_sub = proj_flat[indices].astype(np.float32)
    z_sub = z_positions[indices].astype(np.float64)
    angles_sub = gantry_angles[indices].astype(np.float64)
    print(f"[cache] subset proj_flat shape={proj_sub.shape} "
          f"= {proj_sub.nbytes / 1e9:.2f} GB", flush=True)

    out_pt = out_dir / "L014_proj_flat_peak.pt"
    blob = {
        "proj_flat": torch.from_numpy(proj_sub),
        "z_positions": torch.from_numpy(z_sub),
        "gantry_angles_corrected": torch.from_numpy(angles_sub),
        "original_indices": torch.from_numpy(indices.astype(np.int64)),
        "rotview": rotview,
        "n_proj_full": n_proj_full,
        "nu": nu, "nv": nv,
        "du": float(geom["du"]),
        "dv": float(geom["dv"]),
        "sod": float(geom["sod"]),
        "sdd": float(geom["sdd"]),
        "u0": float(geom.get("u0", (nu - 1) / 2.0)),
        "v0": float(geom.get("v0", (nv - 1) / 2.0)),
        "pitch_mm": pitch_mm,
        "target_source_z": target_source_z,
        "half_window_mm": half_window,
        "angle_start_corrected": float(geom.get("angle_start_corrected", 0.0)),
    }
    torch.save(blob, out_pt)
    print(f"[cache] wrote {out_pt}  "
          f"({out_pt.stat().st_size / 1e9:.2f} GB)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
