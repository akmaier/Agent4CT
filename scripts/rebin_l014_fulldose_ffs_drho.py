"""One-off driver: re-rebin L014 fulldose with FFS-drho correction enabled.

Outputs to L014_sino_fulldose_ffs_drho.h5 so the baseline
L014_sino_fulldose.h5 (no FFS-drho correction) stays available for
side-by-side validator comparison.

Usage (on the cluster, env HELIX2FAN_FFS_DRHO=1 already set by sbatch):
    python scripts/rebin_l014_fulldose_ffs_drho.py

The script imports `_find_projection_series` and `_rebin_patient_series`
from `data/fetch_mayo_ldct.py` and writes to a custom output path.

The FFS-drho correction itself is driven by the env var that
`_rebin_patient_series` already consults; this script just supplies the
custom output filenames.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
# Two paths needed: REPO (so `from data.fetch_mayo_ldct ...` works) AND the
# `data/` directory itself (because fetch_mayo_ldct.py uses bare imports like
# `from _common import ...`, which only resolve when data/ is on sys.path).
for p in (str(REPO), str(REPO / "data")):
    if p not in sys.path:
        sys.path.insert(0, p)

from data.fetch_mayo_ldct import (  # noqa: E402
    _find_projection_series,
    _rebin_patient_series,
)


def main() -> int:
    # Make sure FFS-drho really is enabled (sbatch should already export it,
    # but a script-level safety net is cheap insurance).
    if os.environ.get("HELIX2FAN_FFS_DRHO", "0") not in ("1", "true", "True"):
        print("[rebin-l014-ffs-drho] HELIX2FAN_FFS_DRHO not set; aborting "
              "(the whole point of this driver is to exercise that branch).",
              flush=True)
        return 2

    data_root = Path(os.environ.get(
        "AGENT4CT_DATA",
        "/cluster/maier/Agent4CT/data",
    ))
    raw_dir = data_root / "mayo_ldct" / "raw" / "L014"
    staged_dir = data_root / "mayo_ldct" / "staged_helix2fan"

    if not raw_dir.exists():
        print(f"[rebin-l014-ffs-drho] raw dir not found: {raw_dir}",
              flush=True)
        return 3

    series_dir = _find_projection_series(raw_dir, "full dose projections")
    if series_dir is None:
        print(f"[rebin-l014-ffs-drho] no 'full dose projections' series under "
              f"{raw_dir}", flush=True)
        return 4

    out_h5 = staged_dir / "L014_sino_fulldose_ffs_drho.h5"
    out_z = staged_dir / "L014_sino_fulldose_ffs_drho_z_grid.npy"
    out_g = staged_dir / "L014_sino_fulldose_ffs_drho_geometry.json"

    if out_h5.exists():
        print(f"[rebin-l014-ffs-drho] output already exists: {out_h5} — "
              f"refusing to overwrite. Rename or delete it first.",
              flush=True)
        return 5

    print(f"[rebin-l014-ffs-drho] series_dir={series_dir}", flush=True)
    print(f"[rebin-l014-ffs-drho] out_h5    ={out_h5}", flush=True)

    info = _rebin_patient_series(
        series_dir, out_h5, out_z, out_g,
        dv_rebinned=1.0, n_jobs=int(os.environ.get("N_JOBS", "16")),
    )
    print(f"[rebin-l014-ffs-drho] done. info={info}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
