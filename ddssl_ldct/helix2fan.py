"""Helical-to-2D-fan-beam rebinning for the Mayo LDCT projection data.

Track A (A1) of docs/workplan_real_datasets.md: re-implement, in pure NumPy,
the helix2fan rebinning pipeline (https://github.com/faebstn96/helix2fan).

The TCIA Mayo LDCT-and-Projection-data collection ships per-readout DICOM
projections in a curved-detector helical geometry (Siemens SOMATOM Definition
AS/AS+). To feed those projections into our 2D fan-beam reconstruction
pipeline (ddssl_ldct.pyronn_projector) we need two consecutive remaps:

  1. Curved -> flat detector (per readout). For each virtual flat-detector
     pixel (i_u, i_v) we cast a ray from the source through that pixel,
     intersect it with the arc-shaped curved detector, and bilinearly sample
     the curved readout at the resulting (phi_curved, v) coordinate.

  2. Helical -> circular fan-beam single-slice rebinning (Noo 1999, full 2pi,
     no Parker). For each output rotview index s_angle and each output slice
     z_out, we gather the helical readouts that hit s_angle (mod rotview) and
     linearly interpolate along v to the helical z-window with the Noo Eq.(1)
     amplitude weight

         w = sqrt(u^2 + dsd^2) / sqrt(u^2 + v_precise^2 + dsd^2).

The functions are all pure NumPy; joblib is used to parallelise the
per-projection curved->flat pass when available. No torch dependency.

NB. Like the upstream helix2fan, we do NOT correct flying-focal-spot.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable

import numpy as np


# DICOM private-tag locations (DICOM-CT-PD format used by TCIA Mayo LDCT).
_TAG_DU = (0x7029, 0x1002)        # detector channel pitch (mm)
_TAG_DV = (0x7029, 0x1006)        # detector row pitch (mm)
_TAG_SOD = (0x7031, 0x1003)       # source-to-isocentre distance (mm)
_TAG_SDD = (0x7031, 0x1031)       # constant radial distance source-detector (mm)
_TAG_CENTRAL = (0x7031, 0x1033)   # central detector element (u0, v0)
_TAG_WATER_MU = (0x7041, 0x1001)  # water attenuation (optional)
_TAG_Z = (0x7031, 0x1002)         # detector focal-spot z position (mm)
_TAG_ANGLE = (0x7031, 0x1001)     # gantry angle (rad)
# Flying-focal-spot per-readout deflections (see literature/
# wagner_helix2fan_algorithm.md). Wagner reads these but never uses
# them; we apply a first-cut shift correction in rebin_helical_to_fan
# (use FFS-shifted phi and z when binning into rotview / SSR window).
_TAG_FFS_DPHI = (0x7033, 0x100B)  # azimuthal source-shift (rad)
_TAG_FFS_DZ   = (0x7033, 0x100C)  # axial source-shift (mm)
_TAG_FFS_DRHO = (0x7033, 0x100D)  # radial source-shift (mm)


# ---------------------------------------------------------------------------
# DICOM ingest
# ---------------------------------------------------------------------------

def _get_tag(ds, tag, default=None):
    """Look up a private DICOM tag, returning .value or default if absent."""
    if tag in ds:
        return ds[tag].value
    return default


def _bytes_to_floats(value):
    """If `value` is raw bytes (pydicom's fallback for private tags it
    doesn't know the VR of), unpack as packed little-endian float32s.

    DICOM-CT-PD private tags from the Siemens dataset use VR='UN'
    (unknown), so pydicom returns the bytes verbatim. The numeric
    payload is always a stream of float32 little-endian values whose
    count = len(bytes)/4 (1 for du/dv/sod/sdd; 2 for the (u0,v0) pair).
    """
    if not isinstance(value, (bytes, bytearray)):
        return None
    b = bytes(value)
    n = len(b) // 4
    if n * 4 != len(b) or n == 0:
        return None
    import struct
    return list(struct.unpack(f"<{n}f", b))


def _to_float(value, default=None):
    """Coerce a (potentially MultiValue / raw-bytes) DICOM value to float."""
    if value is None:
        return default
    # Private tag returned as raw bytes — unpack as float32.
    floats = _bytes_to_floats(value)
    if floats is not None:
        return float(floats[0])
    try:
        return float(value)
    except (TypeError, ValueError):
        # DICOM MultiValue with single element falls through here too.
        try:
            return float(value[0])
        except Exception:
            return default


def _to_float_pair(value):
    """Coerce a 2-element DICOM value to a (float, float) tuple."""
    if value is None:
        return None
    # Private tag returned as raw bytes — unpack as two float32.
    floats = _bytes_to_floats(value)
    if floats is not None and len(floats) >= 2:
        return float(floats[0]), float(floats[1])
    try:
        a, b = value
        return float(a), float(b)
    except Exception:
        arr = np.asarray(value, dtype=np.float64).ravel()
        if arr.size >= 2:
            return float(arr[0]), float(arr[1])
        raise


def read_dicom_ctpd(series_dir: Path) -> tuple[np.ndarray, dict]:
    """Stream-read one DICOM-CT-PD series into a curved-detector projection cube.

    `series_dir` is a directory of `.dcm` Raw Data Storage files
    (SOP `1.2.840.10008.5.1.4.1.1.66`) sorted alphabetically (= acquisition
    order in the helix sweep).

    Returns:
      proj_curved: `(n_proj, nv, nu)` float32, after Rescale slope/intercept
                   and channel flip (`arr[:, ::-1]` to match the right-handed
                   detector coordinate system used by helix2fan).
      geometry:    dict with keys du, dv, sod, sdd, u0, v0, water_mu (optional),
                   Rows, Columns, z_positions (n_proj,), gantry_angles (n_proj,),
                   pitch_mm, total_rotations.
    """
    try:
        import pydicom
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "pydicom is required for DICOM-CT-PD ingest; "
            "install via `pip install pydicom`."
        ) from e

    files = sorted(p for p in Path(series_dir).iterdir()
                   if p.is_file() and p.suffix.lower() == ".dcm")
    if not files:
        raise FileNotFoundError(f"No .dcm files in {series_dir}")

    # Re-sort by `InstanceNumber` (acquisition-time index). On Mayo
    # DICOM-CT-PD data the .dcm filenames are NOT in acquisition order
    # — alphabetic order shuffles InstanceNumber across the full
    # 1..n_proj range (verified on L014, 2026-05-23: file 00000001.dcm
    # had InstanceNumber=10298, 00000002.dcm had 23913, etc.).
    # Reading files in alphabetic order produced a sinogram whose
    # gantry angle and z source-position oscillate chaotically over
    # the entire scan extent rather than ramping monotonically. The
    # symptom: rebinned 2D fan-beam sino with no sinusoidal patient
    # signature, and an FBP that looks like a featureless disc.
    # Fix: re-sort by InstanceNumber before the main read loop. One
    # extra header pass over n_proj files (~1 s/1000 files on lme*).
    # See docs/findings.md 2026-05-23 entry.
    print(f"[helix2fan] sorting {len(files)} files by InstanceNumber "
          f"(Mayo DICOM-CT-PD alphabetic≠time order)…", flush=True)
    file_inst: list[tuple[int, Path]] = []
    for fp in files:
        head_only = pydicom.dcmread(str(fp), stop_before_pixels=True)
        inst = int(getattr(head_only, "InstanceNumber", -1))
        file_inst.append((inst, fp))
    file_inst.sort(key=lambda t: t[0])
    files = [fp for _, fp in file_inst]

    # --- geometry from the first file (after time-order sort) ---
    head = pydicom.dcmread(str(files[0]), stop_before_pixels=True)
    geom: dict = {}
    geom["du"] = _to_float(_get_tag(head, _TAG_DU))
    geom["dv"] = _to_float(_get_tag(head, _TAG_DV))
    geom["sod"] = _to_float(_get_tag(head, _TAG_SOD))
    geom["sdd"] = _to_float(_get_tag(head, _TAG_SDD))
    u0v0 = _to_float_pair(_get_tag(head, _TAG_CENTRAL))
    if u0v0 is None:
        raise RuntimeError("Could not read central detector element (0x7031,0x1033)")
    geom["u0"], geom["v0"] = u0v0
    water = _to_float(_get_tag(head, _TAG_WATER_MU))
    if water is not None:
        geom["water_mu"] = water
    geom["Rows"] = int(getattr(head, "Rows"))
    geom["Columns"] = int(getattr(head, "Columns"))
    # Mayo DICOM-CT-PD orientation (verified on L014, 2026-05-21): Rows is the
    # *fan* axis (736 channels) and Columns is the *cone* axis (64 rows). The
    # naïve convention "Rows = vertical = v" is wrong for this scanner. Per
    # the user (CT physicist): "for fan-beam we need to preserve the full 736
    # detector cells in u" — nu = the long axis = Rows.
    nu = geom["Rows"]
    nv = geom["Columns"]

    if any(geom[k] is None for k in ("du", "dv", "sod", "sdd")):
        missing = [k for k in ("du", "dv", "sod", "sdd") if geom[k] is None]
        raise RuntimeError(
            f"DICOM-CT-PD geometry tags missing on {files[0]}: {missing}"
        )

    n_proj = len(files)
    proj_curved = np.empty((n_proj, nv, nu), dtype=np.float32)
    z_positions = np.empty(n_proj, dtype=np.float64)
    gantry_angles = np.empty(n_proj, dtype=np.float64)
    ffs_dphi = np.zeros(n_proj, dtype=np.float64)
    ffs_dz   = np.zeros(n_proj, dtype=np.float64)
    ffs_drho = np.zeros(n_proj, dtype=np.float64)

    for i, fp in enumerate(files):
        ds = pydicom.dcmread(str(fp))
        slope = float(getattr(ds, "RescaleSlope", 1.0))
        intercept = float(getattr(ds, "RescaleIntercept", 0.0))
        # Pixel data is stored as (Columns, Rows) Fortran-order in DICOM-CT-PD.
        # For Mayo SOMATOM AS that buffer is (Columns=nv=64, Rows=nu=736) in
        # Fortran-order — i.e. equivalent to (nv, nu) in C-order, which is
        # exactly the layout the downstream rebin code expects per view.
        raw = np.frombuffer(ds.PixelData, dtype=ds.pixel_array.dtype)
        try:
            arr = raw.reshape((geom["Columns"], geom["Rows"]), order="F")
        except ValueError:
            # Some files might already be C-order (Rows, Columns) = (nu, nv);
            # fall back to that interpretation, then transpose to (nv, nu).
            arr = ds.pixel_array.astype(np.float32).T
        # arr now has shape (Columns, Rows) = (nv, nu) directly — no extra
        # transpose needed (was the bug pre-2026-05-21).
        arr = arr.astype(np.float32, copy=False) * slope + intercept
        # Flip the detector-channel (u) axis to right-handed coordinates.
        # NB. This flip is the reason the curved->flat remap centres the
        # virtual flat detector at index (nu - u0), not u0 — see
        # literature/wagner_helix2fan_algorithm.md (Bug 1).
        arr = arr[:, ::-1]
        proj_curved[i] = np.ascontiguousarray(arr, dtype=np.float32)
        z_positions[i] = _to_float(_get_tag(ds, _TAG_Z), default=np.nan)
        gantry_angles[i] = _to_float(_get_tag(ds, _TAG_ANGLE), default=np.nan)
        # Flying focal spot deflections (default 0 if tag absent).
        ffs_dphi[i] = _to_float(_get_tag(ds, _TAG_FFS_DPHI), default=0.0)
        ffs_dz[i]   = _to_float(_get_tag(ds, _TAG_FFS_DZ),   default=0.0)
        ffs_drho[i] = _to_float(_get_tag(ds, _TAG_FFS_DRHO), default=0.0)

    geom["z_positions"] = z_positions
    geom["gantry_angles"] = gantry_angles
    geom["ffs_dphi"] = ffs_dphi
    geom["ffs_dz"]   = ffs_dz
    geom["ffs_drho"] = ffs_drho
    # Wagner convention: angles += pi/2; angles = -unwrap(angles) - pi.
    # This makes them monotonically increasing from a negative start, which
    # matches the downstream FBP's expectation when angle_start is set to
    # angles_corrected[0] and angle_end = angle_start + 2*pi.
    if np.all(np.isfinite(gantry_angles)) and n_proj > 1:
        angles_corr = gantry_angles + (math.pi / 2.0)
        angles_corr = -np.unwrap(angles_corr) - math.pi
        geom["gantry_angles_corrected"] = angles_corr
        geom["angle_start_corrected"] = float(angles_corr[0])
    else:
        geom["gantry_angles_corrected"] = gantry_angles.copy()
        geom["angle_start_corrected"] = float("nan")
    # Derive helix pitch / total rotations from the per-readout metadata.
    if np.all(np.isfinite(z_positions)) and n_proj > 1:
        dz_per_view = float(np.median(np.diff(z_positions)))
    else:
        dz_per_view = float("nan")
    geom["dz_per_view"] = dz_per_view
    if np.all(np.isfinite(gantry_angles)) and n_proj > 1:
        # Unwrap, then total turns = (max - min) / (2 pi)
        unwrapped = np.unwrap(gantry_angles)
        total_rotations = float(abs(unwrapped[-1] - unwrapped[0]) / (2.0 * math.pi))
    else:
        total_rotations = float("nan")
    geom["total_rotations"] = total_rotations
    # Helix pitch = total z displacement per full rotation. Compute from
    # the actual z-span instead of `median(Δz) × n_proj/total_rotations`
    # — the median form under-reports the pitch when Δz is non-uniform
    # (e.g. Mayo SOMATOM AS+ with Z-FFS state interleaving gives Δz
    # values in {0.0, 0.01, 0.02, 0.03} with median=0.01 but mean=0.0133;
    # the median-based pitch was 23 mm while the true per-rotation
    # displacement is ~30.6 mm). The narrow SSR window from a
    # 23-mm pitch left a 25 % contiguous angular gap with zero
    # contributing readouts (the "flat band" on the rebinned sino).
    # Empirically this is the *only* pitch definition consistent with
    # the stride z-spacing the SSR code actually uses (= mean(Δz) ×
    # rotview = z_span / total_rotations).
    if (math.isfinite(total_rotations) and total_rotations > 0
            and np.all(np.isfinite(z_positions)) and n_proj > 1):
        geom["pitch_mm"] = float(z_positions[-1] - z_positions[0]) / total_rotations
    else:
        geom["pitch_mm"] = float("nan")
    geom["n_proj"] = n_proj

    return proj_curved, geom


# ---------------------------------------------------------------------------
# Curved -> flat detector
# ---------------------------------------------------------------------------

def _rebin_one_view_curved_to_flat(view_curved: np.ndarray,
                                   u_curved_idx: np.ndarray,
                                   v_curved_idx: np.ndarray,
                                   nv: int,
                                   nu: int) -> np.ndarray:
    """Bilinear curved→flat remap for a single ``(nv, nu)`` readout.

    Both ``u_curved_idx`` (shape ``(nu,)``) and ``v_curved_idx`` (shape
    ``(nv, nu)``) are fractional curved-detector indices, **already
    centred against the post-flip convention** (i.e. computed with
    ``(nu - u0)`` and ``(nv - v0)`` offsets — see Bug 1 in
    `literature/wagner_helix2fan_algorithm.md`). The v-axis index varies
    with column because the curved detector bulges in z as the ray
    leaves the central column (Wagner's ``p_on_curved_det[2]`` /
    ``dv`` term).
    """
    out = np.empty((nv, nu), dtype=np.float32)
    # Clamp u indices (kept 1D — same for every row).
    u_idx = np.clip(u_curved_idx, 0.0, float(nu - 2))
    u_floor = np.floor(u_idx).astype(np.int64)
    u_frac = (u_idx - u_floor).astype(np.float32)
    # Per-column gather (1D bilinear in u).
    cols0 = view_curved[:, u_floor]
    cols1 = view_curved[:, u_floor + 1]
    u_interp = cols0 * (1.0 - u_frac) + cols1 * u_frac   # (nv, nu)

    # Now interpolate along v. ``v_curved_idx`` has shape ``(nv, nu)``.
    # For each output (i_v, i_u) we want the v-row v_curved_idx[i_v, i_u]
    # of the curved detector after the u-bilinear step.
    v_idx = np.clip(v_curved_idx, 0.0, float(nv - 2))
    v_floor = np.floor(v_idx).astype(np.int64)
    v_frac = (v_idx - v_floor).astype(np.float32)
    # Gather rows from u_interp at fractional v positions. Per-column
    # vectorised (cheap since nu is at most ~736).
    col_idx = np.arange(nu)
    rows0 = u_interp[v_floor, col_idx[None, :]]          # (nv, nu)
    rows1 = u_interp[v_floor + 1, col_idx[None, :]]      # (nv, nu)
    out[:] = rows0 * (1.0 - v_frac) + rows1 * v_frac
    return out


def rebin_curved_to_flat(proj_curved: np.ndarray, geom: dict,
                         n_jobs: int = -1) -> np.ndarray:
    """Per-readout curved→flat detector rebinning.

    For each virtual flat-detector pixel ``(i_v, i_u)``, compute the
    fractional curved-detector indices ``(v_curved_idx, u_curved_idx)``
    at which the ray from the source through that flat pixel pierces
    the curved (cylindrical) detector. Bilinearly sample the curved
    readout at that position.

    Returns ``(n_proj, nv, nu)`` flat-detector projections. The u-axis
    pitch is still ``du`` (the curved channel pitch); the small
    ``(1 / cos(phi))`` distortion has been resolved by the remap.
    """
    n_proj, nv, nu = proj_curved.shape
    du = float(geom["du"])
    dv = float(geom["dv"])
    sdd = float(geom["sdd"])
    u0 = float(geom["u0"])
    v0 = float(geom["v0"])

    # Virtual flat-detector pixel (u, v) in mm, centred at (0, 0).
    # Wagner uses (i - n/2 + 0.5) * d — half-pixel offset so samples lie
    # at the centre of each cell, not the corner.
    i_u = np.arange(nu, dtype=np.float64)
    i_v = np.arange(nv, dtype=np.float64)
    u_mm_flat = (i_u - nu / 2.0 + 0.5) * du              # (nu,) mm
    v_mm_flat = (i_v - nv / 2.0 + 0.5) * dv              # (nv,) mm

    # Source at origin. Flat detector at y = sdd. Ray from origin through
    # (u_mm, sdd, v_mm) is parametrised; it hits the curved arc (radius
    # sdd) at length = sdd along the ray, i.e. the unit vector scaled by
    # sdd.
    # p_flat shape (nv, nu, 3): for each (i_v, i_u) pixel.
    norm_flat = np.sqrt(u_mm_flat**2 + sdd**2 + v_mm_flat[:, None]**2)  # (nv, nu)
    # Curved-detector hit point coordinates (per pixel).
    p_curved_x = (u_mm_flat / norm_flat) * sdd           # (nv, nu)
    p_curved_z = (v_mm_flat[:, None] / norm_flat) * sdd  # (nv, nu)
    # Arc-angle of the curved-detector hit (per pixel).
    phi = np.arcsin(p_curved_x / sdd)                    # (nv, nu) radians

    # Curved channel pitch in arc-angle.
    dphi_curved = 2.0 * math.atan(du / (2.0 * sdd))
    # Curved channel index. Wagner's offset is (nu - u0) because the data
    # was channel-flipped at load time (read_dicom_ctpd). u_curved_idx
    # has shape (nv, nu) — but the v-dependence is small enough that the
    # central v row dominates; we use the central row's u_curved_idx as
    # the 1D u-sampling pattern (consistent with Wagner who also
    # collapses (phi/dphi) to a 1D u-only lookup despite the formal
    # 2D index).
    u_curved_idx_1d = phi[nv // 2, :] / dphi_curved + (nu - u0)
    # v index on the curved detector for each (i_v, i_u) flat pixel.
    # Wagner uses p_curved_z / dv + (nv - v0).
    v_curved_idx_2d = p_curved_z / dv + (nv - v0)        # (nv, nu)

    def _one(i):
        return _rebin_one_view_curved_to_flat(
            proj_curved[i], u_curved_idx_1d, v_curved_idx_2d, nv, nu
        )

    try:
        from joblib import Parallel, delayed
        results = Parallel(n_jobs=n_jobs, prefer="threads")(
            delayed(_one)(i) for i in range(n_proj)
        )
        out = np.stack(results, axis=0).astype(np.float32, copy=False)
    except ImportError:
        out = np.empty_like(proj_curved)
        for i in range(n_proj):
            out[i] = _one(i)
    return out


# ---------------------------------------------------------------------------
# Helical -> circular fan-beam single-slice rebinning (Noo 1999)
# ---------------------------------------------------------------------------

def _rebin_one_sangle(s_angle: int,
                      rotview: int,
                      proj_flat: np.ndarray,
                      z_positions: np.ndarray,
                      z_out_grid: np.ndarray,
                      u_mm: np.ndarray,
                      v_grid: np.ndarray,
                      du: float,
                      dv: float,
                      dsd: float,
                      dso: float,
                      pitch_per_rot: float,
                      idx_helix: np.ndarray | None = None,
                      dsd_per_readout: np.ndarray | None = None,
                      dso_per_readout: np.ndarray | None = None) -> np.ndarray:
    """SSR for a single output view angle. See `rebin_helical_to_fan`.

    If ``idx_helix`` is provided, use it directly (e.g. for FFS-dphi
    corrected binning). Otherwise fall back to the simple stride
    ``arange(s_angle, n_proj, rotview)`` which assumes integer-rotation
    sampling and no FFS deflection.

    If ``dsd_per_readout`` / ``dso_per_readout`` are provided, use them
    per readout in Noo Eq.(1)/(2) for the radial-FFS correction
    (per-readout effective sod/sdd = nominal ± drho_i). The scalar
    ``dsd`` / ``dso`` args are still used as fallbacks.
    """
    n_proj, nv, nu = proj_flat.shape
    nz_rebinned = z_out_grid.size
    # Indices of helical readouts that hit this output view (one per turn).
    if idx_helix is None:
        idx_helix = np.arange(s_angle, n_proj, rotview)
    z_src_list = z_positions[idx_helix]

    out_view = np.zeros((nu, nz_rebinned), dtype=np.float32)
    # Track which output (i_u, i_z) cells have been written so we leave
    # unhit regions at exactly zero (the spec is an assignment, not an
    # accumulation: with pitch <= detector axial extent every (s_angle, z_out)
    # is covered by exactly one helical turn).
    half_pitch = 0.5 * pitch_per_rot

    for p_local, (helix_idx, z_src) in enumerate(zip(idx_helix, z_src_list)):
        if not math.isfinite(z_src):
            continue
        lo, hi = z_src - half_pitch, z_src + half_pitch
        # Output slice indices that fall in this readout's pitch window.
        # Clipped via searchsorted on the (sorted ascending) z_out_grid.
        if z_out_grid[0] <= z_out_grid[-1]:
            i_lo = int(np.searchsorted(z_out_grid, lo, side="left"))
            i_hi = int(np.searchsorted(z_out_grid, hi, side="right"))
        else:
            # descending
            i_lo = int(np.searchsorted(-z_out_grid, -hi, side="left"))
            i_hi = int(np.searchsorted(-z_out_grid, -lo, side="right"))
        if i_hi <= i_lo:
            continue

        # Per-readout effective sdd/sod (FFS-drho correction). Defaults
        # to the nominal scalars if no per-readout array was passed.
        dsd_i = float(dsd_per_readout[helix_idx]) if dsd_per_readout is not None else dsd
        dso_i = float(dso_per_readout[helix_idx]) if dso_per_readout is not None else dso

        proj_view = proj_flat[helix_idx]   # (nv, nu)
        for i_z in range(i_lo, i_hi):
            dZ = z_src - z_out_grid[i_z]
            # Per-detector-column SSR. Noo (1999) Eq.(1):
            # v_precise = dZ * (u^2 + dsd^2) / (dso * dsd)
            v_precise = dZ * (u_mm**2 + dsd_i**2) / (dso_i * dsd_i)
            # Linear interp along v axis. Out-of-range columns are skipped
            # (np.interp clamps to endpoint; we mask them instead).
            in_range = (v_precise >= v_grid[0]) & (v_precise <= v_grid[-1])
            if not np.any(in_range):
                continue
            # Vectorise the per-column interp via a small loop only over
            # the in-range u indices (nu = 736 is small enough not to
            # matter, but this saves us a python-level call per pixel).
            for i_u in np.nonzero(in_range)[0]:
                col = proj_view[:, i_u]
                v_val = np.interp(v_precise[i_u], v_grid, col)
                w = math.sqrt(u_mm[i_u]**2 + dsd_i**2) / math.sqrt(
                    u_mm[i_u]**2 + v_precise[i_u]**2 + dsd_i**2
                )
                out_view[i_u, i_z] = w * v_val

    return out_view


def rebin_helical_to_fan(proj_flat: np.ndarray,
                         geom: dict,
                         *,
                         dv_rebinned: float,
                         nz_rebinned: int,
                         z_start: float | None = None,
                         n_jobs: int = -1,
                         u_centering_mode: str = "u0",
                         ffs_correct_dphi: bool = False,
                         ffs_correct_drho: bool = False,
                         ) -> tuple[np.ndarray, np.ndarray]:
    """Helical -> circular fan-beam single-slice rebinning (Noo 1999).

    Inputs:
      proj_flat: (n_proj, nv, nu) flat-detector helical projections (output of
                 `rebin_curved_to_flat`).
      geom: geometry dict with du, dv, sod, sdd, u0, v0, z_positions
            (n_proj,), gantry_angles (n_proj,), total_rotations, pitch_mm.
      dv_rebinned: axial output spacing (mm/slice). Free parameter; typical
                   values are the original dv (=> 1:1 axial) down to ~0.6 mm.
      nz_rebinned: number of output axial slices.
      z_start: physical z of the first output slice. Defaults to the helix
               start (z_positions.min()) + half a pitch so every output slice
               has a full helical window.
      n_jobs: passed to joblib for the s_angle outer loop.
      u_centering_mode: how to compute the per-pixel ``u_mm`` / ``v_grid``
                        used in Noo Eq.(1)/(2). Options:
            "u0" (default; legacy): ``u_mm = (i_u − u0) · du``,
                ``v_grid = (i_v − v0) · dv``. The historical Wagner-style
                code uses ``u0/v0`` here, but after the curved→flat remap
                (which already centres the flat detector at ``nu − u0``)
                this leaves a residual offset of ``u0 − (nu/2 − 0.5)`` ≈
                4 mm on the L014 geometry — visible as faint shadow / sub-
                pixel ghosts in the FBP.
            "physical": ``u_mm = (i_u − (nu/2 − 0.5)) · du``,
                ``v_grid = (i_v − (nv/2 − 0.5)) · dv``. The flat detector's
                physical centre. Recommended when the curved→flat remap is
                already correct (Bug 1 patched).
      ffs_correct_dphi: if True, bin readouts into output ``s_angle`` slots
                        by their *effective* gantry angle
                        ``angles_eff = angles_corrected + ffs_dphi`` (mod
                        2π) instead of the simple stride
                        ``arange(s_angle, n_proj, rotview)``. Properly
                        accounts for the Siemens SOMATOM AS+ azimuthal FFS
                        deflection (≈ ±0.5 · 2π/rotview per readout). The
                        stride approach treats those deflections as if
                        they were on the nominal grid, leaving subtle
                        angular shadowing in the recon.
      ffs_correct_drho: if True, use per-readout effective ``sod`` and
                        ``sdd`` in Noo Eq.(1)/(2) instead of the nominal
                        scalars. For Mayo SOMATOM AS+ L014: drho
                        alternates {0, +5.45 mm} every readout (period-2
                        Z+radial FFS, characterised in
                        ``results/breast_debug/L014_ffs_pattern.png``).
                        Without correction the SSR averages two
                        magnifications (sdd/sod ≈ 1.8245 and ≈ 1.8170),
                        producing the faint shadow / ghost edges
                        visible after the geometry fix.

                        Convention (deviates from Wagner's literature
                        note in ``literature/wagner_helix2fan_algorithm.md``;
                        Wagner's item 1 reads "use sod + drho for sdd",
                        which appears to be a typo since that gives
                        sdd_eff ≈ 600 mm ≠ nominal sdd ≈ 1086 mm; here
                        we use the physically-consistent both-distances-
                        grow convention):
                            sod_eff_i = sod + ffs_drho[i]
                            sdd_eff_i = sdd + ffs_drho[i]
                        i.e. the source's perpendicular distance to
                        isocentre and to detector both grow by the
                        same drho — detector stays put, source moves
                        radially outward. If a future dataset has
                        opposite sign convention, flip the sign of
                        the ``+ ffs_drho`` term and re-validate.
    """
    n_proj, nv, nu = proj_flat.shape
    du = float(geom["du"])
    dv = float(geom["dv"])
    sdd = float(geom["sdd"])
    sod = float(geom["sod"])
    u0 = float(geom["u0"])
    v0 = float(geom["v0"])
    total_rot = float(geom["total_rotations"])
    pitch_mm = float(geom["pitch_mm"])
    z_positions = np.asarray(geom["z_positions"], dtype=np.float64)
    # FFS first-cut correction (literature/wagner_helix2fan_algorithm.md):
    # shift the per-readout z by dz and the per-readout phi by dphi
    # before binning into rotview / SSR windows. The radial drho
    # contribution to magnification is < 0.2 % at sod≈595 and ignored
    # in this first cut.
    ffs_dz   = np.asarray(geom.get("ffs_dz",   np.zeros(n_proj)), dtype=np.float64)
    ffs_dphi = np.asarray(geom.get("ffs_dphi", np.zeros(n_proj)), dtype=np.float64)
    ffs_drho = np.asarray(geom.get("ffs_drho", np.zeros(n_proj)), dtype=np.float64)
    # v3 (SLURM 763384, 2026-06-12): per-readout z-axis scaling. Multiplies
    # z_positions BEFORE the α_dz FFS shift; equivalent at leading order to
    # a ~0.17 % under-estimate of helical pitch. See findings.md 2026-06-12.
    # Default 1.0 preserves the pre-v3 behaviour for legacy callers.
    s_z = float(geom.get("s_z", 1.0))
    z_eff = s_z * z_positions + ffs_dz

    # Per-readout effective sod/sdd for the radial FFS correction.
    # Only built when `ffs_correct_drho` is True; otherwise the inner
    # loop uses the nominal scalars.
    if ffs_correct_drho:
        dso_per_readout = (sod + ffs_drho).astype(np.float64)
        dsd_per_readout = (sdd + ffs_drho).astype(np.float64)
        print(f"[helix2fan] ffs_correct_drho=True: per-readout sod range "
              f"[{dso_per_readout.min():.3f}, {dso_per_readout.max():.3f}], "
              f"sdd range [{dsd_per_readout.min():.3f}, "
              f"{dsd_per_readout.max():.3f}]", flush=True)
    else:
        dso_per_readout = None
        dsd_per_readout = None
    if not math.isfinite(total_rot) or total_rot < 0.5:
        raise RuntimeError(
            f"Implausible total_rotations={total_rot}; check the z and gantry tags."
        )

    # Determine rotview from the FFS-corrected gantry angles when available,
    # so the views span exactly one rotation in the *effective* angular
    # coordinate. Falls back to n_proj / total_rot if angles aren't usable.
    angles_corr = geom.get("gantry_angles_corrected")
    if angles_corr is not None and np.all(np.isfinite(angles_corr)):
        angles_eff = np.asarray(angles_corr, dtype=np.float64) + ffs_dphi
        # Number of views in one rotation: how many readouts fall in
        # [angles_eff[0], angles_eff[0] + 2π).
        rotview = int(round(n_proj / total_rot))
    else:
        angles_eff = None
        rotview = int(round(n_proj / total_rot))
    if rotview <= 0:
        raise RuntimeError(f"rotview <= 0: n_proj={n_proj}, total_rot={total_rot}")

    if z_start is None:
        z_min = float(np.nanmin(z_positions))
        z_start = z_min + 0.5 * abs(pitch_mm)
    z_out_grid = z_start + np.arange(nz_rebinned, dtype=np.float64) * dv_rebinned

    # Per-flat-pixel u/v grids in mm. See `u_centering_mode` docstring.
    i_u = np.arange(nu, dtype=np.float64)
    i_v = np.arange(nv, dtype=np.float64)
    if u_centering_mode == "physical":
        u_mm = (i_u - (nu / 2.0 - 0.5)) * du
        v_grid = (i_v - (nv / 2.0 - 0.5)) * dv
    elif u_centering_mode == "u0":
        u_mm = (i_u - u0) * du
        v_grid = (i_v - v0) * dv
    else:
        raise ValueError(f"unknown u_centering_mode={u_centering_mode!r}")

    # Precompute per-s_angle readout-index lists.
    if ffs_correct_dphi:
        if angles_corr is None or not np.all(np.isfinite(angles_corr)):
            raise RuntimeError(
                "ffs_correct_dphi=True requires geom['gantry_angles_corrected'] "
                "(Wagner-style +π/2 −unwrap −π). None found.")
        # Bin each readout by its effective angle (mod 2π) into rotview slots.
        # `angles_eff[0]` defines the s_angle=0 origin so the rebinned sino's
        # angular phase matches the legacy-stride version. Each output slot
        # receives ~total_rot readouts (one per helical turn).
        step = (2.0 * math.pi) / rotview
        bin_idx = np.round((angles_eff - angles_eff[0]) / step).astype(np.int64)
        # Map to [0, rotview); negative remainders need to wrap correctly
        # (Python % handles this for ints, but be explicit).
        bin_idx = np.mod(bin_idx, rotview)
        idx_helix_per_s = [np.where(bin_idx == s)[0] for s in range(rotview)]
    else:
        idx_helix_per_s = None   # _rebin_one_sangle will compute the stride.

    def _one(s):
        idx = idx_helix_per_s[s] if idx_helix_per_s is not None else None
        return _rebin_one_sangle(
            s, rotview, proj_flat, z_eff, z_out_grid,
            u_mm, v_grid, du, dv, sdd, sod, abs(pitch_mm), idx_helix=idx,
            dsd_per_readout=dsd_per_readout,
            dso_per_readout=dso_per_readout,
        )

    try:
        from joblib import Parallel, delayed
        views = Parallel(n_jobs=n_jobs, prefer="threads")(
            delayed(_one)(s) for s in range(rotview)
        )
        out = np.stack(views, axis=0).astype(np.float32, copy=False)
    except ImportError:
        out = np.empty((rotview, nu, nz_rebinned), dtype=np.float32)
        for s in range(rotview):
            out[s] = _one(s)
    return out, z_out_grid.astype(np.float32)


__all__ = [
    "read_dicom_ctpd",
    "rebin_curved_to_flat",
    "rebin_helical_to_fan",
]
