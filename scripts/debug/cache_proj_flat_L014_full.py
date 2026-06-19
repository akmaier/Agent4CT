#!/usr/bin/env python -u
"""Cache the curved-to-flat helical projections for L014/fulldose over the
FULL helical sweep (not the 100-mm peak slab that cache_proj_flat_L014.py
writes).

Needed by the v2 end-to-end geometry fit (scripts/fit_rebin_end2end_L014_v2.py)
which samples 10 GT slices uniformly across the full 154-slice "Full Dose
Images" series (patient-z ≈ [−482.5, −23.6] mm) instead of the central 10.

Output:
  L014_proj_flat_full.pt — same blob schema as L014_proj_flat_peak.pt but
  with the FULL helix (no z-window mask).  ~7.15 GB on disk.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
if str(REPO / "data") not in sys.path:
    sys.path.insert(0, str(REPO / "data"))

from ddssl_ldct.helix2fan import read_dicom_ctpd, rebin_curved_to_flat
from data.fetch_mayo_ldct import _find_projection_series


def main() -> int:
    import os
    root = Path(os.environ.get("AGENT4CT_DATA",
                                "/cluster/maier/Agent4CT/data")) / "mayo_ldct"
    raw_dir = root / "raw" / "L014"
    out_dir = root / "staged_helix2fan"
    out_dir.mkdir(parents=True, exist_ok=True)

    series_dir = _find_projection_series(raw_dir, "full dose projections")
    if series_dir is None:
        print(f"[cache-full] no fulldose projection series in {raw_dir}",
              file=sys.stderr)
        return 1
    print(f"[cache-full] reading {series_dir}", flush=True)
    proj_curved, geom = read_dicom_ctpd(series_dir)
    print(f"[cache-full] proj_curved shape={proj_curved.shape} "
          f"sdd={geom['sdd']:.3f} sod={geom['sod']:.3f} du={geom['du']:.4f} "
          f"dv={geom['dv']:.4f} pitch_mm={geom['pitch_mm']:.4f}",
          flush=True)

    print(f"[cache-full] curved -> flat (n_jobs=-1) …", flush=True)
    proj_flat = rebin_curved_to_flat(proj_curved, geom, n_jobs=-1)
    del proj_curved
    print(f"[cache-full] proj_flat shape={proj_flat.shape}  "
          f"dtype={proj_flat.dtype}", flush=True)

    z_positions = np.asarray(geom["z_positions"], dtype=np.float64)
    gantry_angles = np.asarray(geom.get("gantry_angles_corrected",
                                          geom.get("gantry_angles")),
                                dtype=np.float64)
    rotview = int(round(proj_flat.shape[0] / geom["total_rotations"]))
    n_proj_full = int(proj_flat.shape[0])
    nu, nv = proj_flat.shape[2], proj_flat.shape[1]
    pitch_mm = float(geom["pitch_mm"])

    # FULL helix — no subset mask.  original_indices is identity.
    indices = np.arange(n_proj_full, dtype=np.int64)
    proj_sub = proj_flat.astype(np.float32)
    z_sub = z_positions.astype(np.float64)
    angles_sub = gantry_angles.astype(np.float64)
    print(f"[cache-full] FULL proj_flat: {proj_sub.shape} = "
          f"{proj_sub.nbytes / 1e9:.2f} GB on disk", flush=True)
    print(f"[cache-full] z range = [{z_sub.min():.2f}, {z_sub.max():.2f}] mm "
          f"(span {z_sub.max() - z_sub.min():.2f} mm)", flush=True)

    ffs_dz = np.asarray(geom.get("ffs_dz",
                                  np.zeros(n_proj_full)), dtype=np.float64)
    ffs_dphi = np.asarray(geom.get("ffs_dphi",
                                    np.zeros(n_proj_full)), dtype=np.float64)
    ffs_drho = np.asarray(geom.get("ffs_drho",
                                    np.zeros(n_proj_full)), dtype=np.float64)
    print(f"[cache-full] FFS: dz [{ffs_dz.min():.4f}, {ffs_dz.max():.4f}], "
          f"dphi [{ffs_dphi.min():.3e}, {ffs_dphi.max():.3e}], "
          f"drho [{ffs_drho.min():.4f}, {ffs_drho.max():.4f}]", flush=True)

    blob = {
        "proj_flat": torch.from_numpy(proj_sub),
        "z_positions": torch.from_numpy(z_sub),
        "gantry_angles_corrected": torch.from_numpy(angles_sub),
        "original_indices": torch.from_numpy(indices),
        "ffs_dz":   torch.from_numpy(ffs_dz),
        "ffs_dphi": torch.from_numpy(ffs_dphi),
        "ffs_drho": torch.from_numpy(ffs_drho),
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
        # Kept for schema compatibility with the peak blob; the v2 fit
        # ignores these since it picks its own per-slice target z's.
        "target_source_z": 0.0,
        "half_window_mm": float(z_sub.max() - z_sub.min()) / 2,
        "angle_start_corrected": float(geom.get("angle_start_corrected", 0.0)),
    }
    out_pt = out_dir / "L014_proj_flat_full.pt"
    torch.save(blob, out_pt)
    print(f"[cache-full] wrote {out_pt}  "
          f"({out_pt.stat().st_size / 1e9:.2f} GB)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
