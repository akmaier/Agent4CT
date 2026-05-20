"""Shared per-challenge geometry datatype.

Each challenge under `challenges/<name>/` exposes a `GEOMETRY` (a
`DatasetInfo` instance) describing the scan geometry + display range +
where the staged HDF5 files live. Solvers consume the bundled
`DEFAULTS` (in `challenges/<name>/geometry.py`) which is a
solver-CONFIG-shaped dict pre-populated with these values plus the
simulation knobs that don't live in the projector geometry (noise,
splits, seed).

The same `DatasetInfo` dataclass used to live in
`ddssl_ldct/staged_dataset.py`. It has moved here so the challenges
folder is the single source of truth and the staging code just consumes
it. `staged_dataset.py` re-exports the symbol for backwards compat.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DatasetInfo:
    """Per-dataset scan geometry + staging metadata.

    Distances are in mm (consistent with `ddssl_ldct.geometry.FanBeamGeometry`).
    Image attenuation μ is in 1/cm for `breast_ct` and 1/mm for `phantoms`
    (Wagner LDCT helical-rebin convention). When a projector needs μ and
    geometry in matching length units (e.g.
    `SiddonFanBeamProjector(length_unit_scale=…)`) the caller is responsible
    for passing the right scale.
    """
    image_size: int
    pixel_spacing: float
    n_angles: int
    n_det: int
    det_spacing: float
    sod: float
    sdd: float
    display_min: float
    display_max: float
    has_real_sino: bool             # True -> noisy is loaded; False -> simulated
    staged_dir: Path | None = None
    truth_file_tmpl: str = "{split}_truth.h5"
    truth_dataset: str = "image"
    sino_file_tmpl: str = "{split}_sinograms.h5"
    sino_dataset: str = "sino"
    # Per-dataset start-angle shift applied via np.roll along the angle
    # axis BEFORE the harness consumes the sino. Used to align the
    # dataset's gantry-rotation convention with PyronnFanBeamProjector's
    # (angle 0 = source at +x, CCW positive).
    sino_angle_shift: int = 0
