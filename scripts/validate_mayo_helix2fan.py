"""Validate one patient's helix2fan rebinning by FBP-vs-truth comparison.

Track A3 of docs/workplan_real_datasets.md.

Loads `L<NNN>_sino_fulldose.h5` written by `stage_h5_with_sino` (rotview, nu,
nz), picks the center z slice, builds a matching `PyronnFanBeamProjector`
with the staged-helix2fan geometry (n_angles=rotview, n_det=nu, sod=595,
sdd=1085.6, pixel_spacing=0.7, image_size=512), runs FBP, and compares
SSIM/PSNR against the matching truth slice from `staged/test_truth.h5` (or
whichever split contains the patient).

Run example:
    python scripts/validate_mayo_helix2fan.py --patient L014
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import torch  # noqa: E402

from ddssl_ldct.geometry import FanBeamGeometry  # noqa: E402
from ddssl_ldct.pyronn_projector import PyronnFanBeamProjector  # noqa: E402
from ddssl_ldct.metrics import (
    ssim as ssim_metric, psnr as psnr_metric, evaluate_calibrated,
)  # noqa: E402


# Wagner split lookup mirrors data/fetch_mayo_ldct.py.
WAGNER_SPLITS = {
    "train": ["L145", "L186", "L209", "L219"],
    "val":   ["L277"],
    "test":  ["L014", "L056", "L058", "L075", "L123"],
}


def find_split(patient: str) -> str:
    for split, pids in WAGNER_SPLITS.items():
        if patient in pids:
            return split
    raise ValueError(f"unknown patient {patient}; not in Wagner subset")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", default=None,
                   help="Override AGENT4CT_DATA root (defaults to env or "
                        "/cluster/maier/Agent4CT/data).")
    p.add_argument("--patient", default="L014",
                   help="Wagner-subset patient ID (e.g. L014, L056, ...).")
    p.add_argument("--dose", default="fulldose", choices=["fulldose", "lowdose"])
    p.add_argument("--out-png", default=None,
                   help="Where to drop the comparison png. Default: "
                        "scripts/_validate_<patient>_<dose>.png")
    p.add_argument("--device", default="cuda")
    p.add_argument("--z-offset-mm", type=float, default=3.5,
                   help="Shift the FBP slab centre by this many mm "
                        "(in source-z direction). Default +3.5 mm "
                        "= fine-z-sweep SSIM peak for L014 (job 762111). "
                        "Pass 0 to use the rebinned sino's middle slice.")
    p.add_argument("--slab-half", type=int, default=2,
                   help="Half-width of the FBP slab in 1-mm slices. "
                        "Default 2 (= 5-slice = 5-mm slab matching "
                        "the Mayo truth DICOM's SliceThickness=5 mm).")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    import os
    root = Path(args.data_root or os.environ.get("AGENT4CT_DATA",
                                                 "/cluster/maier/Agent4CT/data"))
    challenge = root / "mayo_ldct"
    sino_dir = challenge / "staged_helix2fan"
    truth_dir = challenge / "staged"

    sino_h5 = sino_dir / f"{args.patient}_sino_{args.dose}.h5"
    geom_path = sino_dir / f"{args.patient}_sino_{args.dose}_geometry.json"
    zgrid_path = sino_dir / f"{args.patient}_sino_{args.dose}_z_grid.npy"
    if not sino_h5.exists():
        print(f"[validate] missing {sino_h5}", file=sys.stderr)
        return 1

    geom_json = json.loads(geom_path.read_text())
    z_grid = np.load(zgrid_path)
    rotview = int(geom_json["rotview"])
    nu = int(geom_json["nu"])
    nz = int(geom_json["nz_rebinned"])
    du = float(geom_json["du"])

    print(f"[validate] {sino_h5.name}: rotview={rotview} nu={nu} nz={nz} "
          f"du={du:.4f}")

    # Load a 5-mm slab of the rebinned sino around the centre z (shifted
    # by `--z-offset-mm`) to match the Mayo truth DICOM's
    # SliceThickness=5 mm. dv_rebinned=1 mm in our helix2fan output, so
    # 5 adjacent slices = 5 mm. We FBP each individually and average the
    # recons — this matches the truth's 5-mm axial integration. Verified
    # 2026-05-24 on L014: slab-averaging added +0.034 calibrated SSIM
    # and +2.2 dB PSNR over single-slice; the +3.5 mm offset is the
    # SSIM peak from the fine-z-sweep (job 762111) — barely above noise
    # but worth keeping as the default until a per-patient drift study
    # informs otherwise.
    SLAB_HALF = int(args.slab_half)
    nz_middle = nz // 2
    nz_center = nz_middle + int(round(args.z_offset_mm))  # 1 mm/slice
    nz_center = max(0, min(nz - 1, nz_center))
    z_center = float(z_grid[nz_center])
    slab_lo = max(0, nz_center - SLAB_HALF)
    slab_hi = min(nz, nz_center + SLAB_HALF + 1)
    print(f"[validate] nz_middle={nz_middle}  z_offset={args.z_offset_mm:+.1f} mm  "
          f"→ nz_center={nz_center}  z_center={z_center:.2f} mm")
    with h5py.File(sino_h5, "r") as f:
        sino_slab = [
            np.asarray(f["sino"][:, :, j], dtype=np.float32)
            for j in range(slab_lo, slab_hi)
        ]
    # "Siemens flip" along the u (detector) axis, per
    # literature/wagner_helix2fan_algorithm.md (Bug 4). Wagner's
    # reco_example_fan_beam.py does `np.flip(projections[:, :, i], axis=1)`
    # before filtering — equivalent to axis=-1 for our 2D (rotview, nu)
    # slice. This undoes the curved-detector channel flip applied at
    # load time so the back-projection direction matches PYRO-NN's
    # right-handed convention.
    sino_slab = [np.ascontiguousarray(np.flip(s, axis=-1)) for s in sino_slab]
    print(f"[validate] z_center = {z_center:.2f} mm (centre slice {nz_center}/{nz}); "
          f"averaging slab [{slab_lo}..{slab_hi - 1}] = {len(sino_slab)} slices "
          f"(≈ {len(sino_slab)} mm thick to match truth SliceThk=5 mm); "
          f"Siemens-flipped u-axis")

    # Match truth slice via the staged manifest's z-ordered file list.
    # The truth h5 is shuffled by stage_h5; we need to recover (patient, z)
    # alignment. The cleanest path is to re-read the raw fulldose-image
    # DICOMs for this patient and pick the z that minimises |z - z_center|.
    # Returns a *z-interpolated* slice at the exact sino z_center, plus
    # the bracket pair for diagnostics.
    truth_info = _load_truth_slice_for_z(challenge / "raw" / args.patient,
                                          z_center)
    if truth_info is None:
        print(f"[validate] could not align truth slice for {args.patient}",
              file=sys.stderr)
        return 2
    truth_slice, truth_z_target_patient, truth_bracket = truth_info

    # Build the matching fan-beam geometry. Per Wagner, image is 512^2 @
    # 0.7 mm pixel spacing; sdd/sod default values are baked into the
    # FanBeamGeometry dataclass.
    # angle_start_corrected (Wagner's `+pi/2 -unwrap -pi` recipe) tells
    # us where the first rebinned view sits in the absolute gantry
    # frame; the full rotation spans 2π from there. See Bug 3 in
    # literature/wagner_helix2fan_algorithm.md.
    import math as _math
    angle_start = float(geom_json.get("angle_start_corrected", 0.0))
    angle_end = angle_start + 2.0 * _math.pi
    print(f"[validate] angle_start={angle_start:.4f} rad, "
          f"angle_end={angle_end:.4f} rad (1 full rotation)")
    geom = FanBeamGeometry(
        image_size=512,
        pixel_spacing=0.7,
        n_angles=rotview,
        n_det=nu,
        det_spacing=du,
        sod=float(geom_json.get("sod", 595.0)),
        sdd=float(geom_json.get("sdd", 1085.6)),
        angle_start=angle_start,
        angle_end=angle_end,
    )
    proj = PyronnFanBeamProjector(geom).to(args.device)

    # FBP each slab member individually, then average — equivalent to
    # the standard "thick-slice" reconstruction the scanner does at
    # 5 mm SliceThickness on B30f.
    fbp_slab_np = []
    for s in sino_slab:
        s_t = torch.from_numpy(s).to(args.device).float()[None, None]
        fbp_one = proj.fbp(s_t).detach()[0, 0].cpu().numpy()
        # PYRO-NN's 2D FBP output is in a coordinate system that needs a
        # 180° rotation (= flipud + fliplr) to match the Mayo
        # ImagePositionPatient DICOM convention (head-first supine, anterior
        # up, patient-right on image-left). flipud alone leaves the L/R
        # mirror; verified visually 2026-05-24 against L014 thorax slice.
        fbp_slab_np.append(np.fliplr(np.flipud(fbp_one)))
    fbp_np = np.ascontiguousarray(np.mean(np.stack(fbp_slab_np, axis=0), axis=0))
    fbp = torch.from_numpy(fbp_np).to(args.device).float()[None, None]

    truth_t = torch.from_numpy(truth_slice).to(args.device).float()[None, None]
    # data_range as configured for ddssl_ldct (water = 0.02 mm^-1; tissue ~0.05).
    dr = 0.05
    fbp_clipped = fbp.clamp(min=0.0)
    ssim_raw = float(ssim_metric(fbp_clipped, truth_t, dr).cpu())
    psnr_raw = float(psnr_metric(fbp_clipped, truth_t, dr).cpu())

    # ---- Intensity calibration (FOV-masked linear fit pred → truth) -------
    # `evaluate_calibrated` runs `intensity_calibrate` (two-point linear
    # bg/fg fit), applies the inscribed-circle FOV mask, then computes
    # PSNR/SSIM/RMSE on the calibrated `pred_cal` vs `truth`. This is the
    # exact same calibration the downstream training metric uses, so the
    # numbers compare apples-to-apples with what the dual-domain solvers
    # report. See ddssl_ldct.metrics.intensity_calibrate for the (a, bg)
    # affine details.
    metrics = evaluate_calibrated(
        fbp_clipped, truth_t,
        baseline=fbp_clipped,        # FBP IS our reference recon here
        display_min=0.0, display_max=dr,
    )
    pred_cal = metrics["pred_cal"]
    ssim_cal = float(metrics["val_ssim"])
    psnr_cal = float(metrics["val_psnr"])
    rmse_cal = float(metrics["val_rmse"])
    pred_cal_np = pred_cal[0, 0].cpu().numpy()
    cal_desc = metrics["calibration"]  # string description

    print(f"[validate] RAW         SSIM = {ssim_raw:.4f}  PSNR = {psnr_raw:.2f} dB")
    print(f"[validate] CALIBRATED  SSIM = {ssim_cal:.4f}  PSNR = {psnr_cal:.2f} dB  "
          f"RMSE = {rmse_cal:.5f}")
    print(f"[validate] calibration: {cal_desc}")
    print(f"[validate] truth slice    min={truth_slice.min():.4f}  "
          f"max={truth_slice.max():.4f}  mean={truth_slice.mean():.4f}")
    print(f"[validate] FBP   raw      min={fbp_np.min():.4f}  "
          f"max={fbp_np.max():.4f}  mean={fbp_np.mean():.4f}")
    print(f"[validate] FBP   cal      min={pred_cal_np.min():.4f}  "
          f"max={pred_cal_np.max():.4f}  mean={pred_cal_np.mean():.4f}")
    print(f"[validate] truth bracket  z_lo={truth_bracket[0]:.2f}  z_hi={truth_bracket[1]:.2f}  "
          f"weight_lo={truth_bracket[2]:.3f}  target={truth_z_target_patient:.2f}")

    out_png = Path(args.out_png) if args.out_png else (
        REPO / "scripts" / f"_validate_{args.patient}_{args.dose}.png"
    )
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    axes[0].imshow(truth_slice, cmap="gray", vmin=0, vmax=dr)
    axes[0].set_title(f"truth (z-interp)\npatient_z={truth_z_target_patient:.1f}\n"
                      f"bracket [{truth_bracket[0]:.1f}, {truth_bracket[1]:.1f}]",
                      fontsize=9)
    axes[1].imshow(fbp_np, cmap="gray", vmin=0, vmax=dr)
    axes[1].set_title(f"FBP raw\nSSIM={ssim_raw:.3f}  PSNR={psnr_raw:.1f} dB\n"
                      f"FBP mean={fbp_np.mean():.4f}", fontsize=9)
    axes[2].imshow(pred_cal_np, cmap="gray", vmin=0, vmax=dr)
    axes[2].set_title(f"FBP intensity-calibrated\n"
                      f"SSIM={ssim_cal:.3f}  PSNR={psnr_cal:.1f} dB  RMSE={rmse_cal:.4f}\n"
                      f"FBP_cal mean={pred_cal_np.mean():.4f}", fontsize=9)
    diff_cal = pred_cal_np - truth_slice
    axes[3].imshow(diff_cal, cmap="seismic", vmin=-0.02, vmax=0.02)
    axes[3].set_title(f"diff (cal − truth)\nmax|·| = {np.abs(diff_cal).max():.3f}",
                      fontsize=9)
    for ax in axes:
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(f"{args.patient}/{args.dose} helix2fan validation  "
                 f"(rotview={rotview}, pitch={float(geom_json['pitch_mm']):.2f} mm, "
                 f"z_center={z_center:.1f})")
    fig.tight_layout()
    fig.savefig(out_png, dpi=120)
    print(f"[validate] wrote {out_png}")

    return 0


def _load_truth_slice_for_z(patient_dir: Path, z_target: float):
    """Find the full-dose-image DICOM slice closest to z_target (mm) and
    **z-interpolate between the two bracketing truth slices** so the
    returned slice represents the *exact* sino z_target after the
    scanner→patient frame mapping.

    Re-reads the raw DICOMs directly rather than the staged h5 because the
    staged h5 is shuffled and discards per-slice z metadata.

    Returns:
        (truth_interp, target_patient_z, (z_lo, z_hi, weight_lo)) where:
            * truth_interp: (H, W) float32 μ image, interpolated.
            * target_patient_z: the patient-frame z the FBP should be
              compared against (= -z_target + offset).
            * (z_lo, z_hi, weight_lo): the bracketing truth z's and the
              linear-interpolation weight on the lower-z slice.
        Or None if no truth slice can be located.

    **Frame mismatch fix (2026-05-24).** The sino's z_target comes from
    the helix2fan SSR output, which is in the SCANNER-SOURCE frame
    (table position at acquisition; monotonically *increasing* over
    the helix sweep). The reconstructed truth-image DICOMs use
    `ImagePositionPatient[2]` in the PATIENT frame (head-positive
    DICOM convention). For a head-first supine CT, the two frames are
    related by a sign flip + constant table-position offset:

        patient_z ≈ -(source_z) + offset

    Empirically we detect the offset from data: collect all truth
    `ImagePositionPatient[2]` values and find the one closest to
    `-z_target + offset` for the offset that makes the truth-z range
    bracket the (sign-flipped) sino-z range. Falls back to the
    original `z` matching if the sign-flipped mapping doesn't bracket.
    """
    import pydicom
    SOP_CT_IMAGE = "1.2.840.10008.5.1.4.1.1.2"
    MU_WATER_PER_MM = 0.02

    if not patient_dir.exists():
        print(f"[validate] {patient_dir} not found", file=sys.stderr)
        return None

    truth_files: list[tuple[float, Path]] = []
    for series_dir in sorted(patient_dir.iterdir()):
        if not series_dir.is_dir():
            continue
        sample = next(series_dir.iterdir(), None)
        if sample is None:
            continue
        try:
            head = pydicom.dcmread(str(sample), stop_before_pixels=True)
        except Exception:
            continue
        if getattr(head, "SOPClassUID", "") != SOP_CT_IMAGE:
            continue
        desc = getattr(head, "SeriesDescription", "").lower()
        if "full" not in desc or "image" not in desc:
            continue
        for fp in series_dir.iterdir():
            try:
                meta = pydicom.dcmread(str(fp), stop_before_pixels=True)
                z = float(meta.ImagePositionPatient[2])
            except Exception:
                continue
            truth_files.append((z, fp))

    if not truth_files:
        return None
    truth_zs = np.array([z for z, _ in truth_files])
    z_lo, z_hi = float(truth_zs.min()), float(truth_zs.max())
    print(f"[validate] truth ImagePositionPatient[2] range = "
          f"[{z_lo:.1f}, {z_hi:.1f}] mm; sino z_target = {z_target:.1f}")

    # The Mayo head-first DICOM convention is `patient_z = -source_z`
    # (sign-flip, no constant offset). Verified empirically on L014:
    # this gives truth_match dz ≤ slice_thickness/2 for the centre
    # slice. Identity mapping (no flip) puts truth ~277 mm away.
    # Earlier versions of this script tried to auto-detect an additive
    # offset that mapped sino midpoint to truth midpoint, but that
    # produced a degenerate "always pick truth midpoint" behaviour
    # when the FBP slab was shifted to a non-midpoint z. The simple
    # sign-flip is the right invariant.
    cand_A_z = z_target                                  # identity
    cand_B_z = -z_target                                 # sign-flip
    cand_A = truth_zs[np.argmin(np.abs(truth_zs - cand_A_z))]
    cand_B = truth_zs[np.argmin(np.abs(truth_zs - cand_B_z))]
    dist_A = abs(cand_A - cand_A_z)
    dist_B = abs(cand_B - cand_B_z)
    print(f"[validate]   mapping A (identity):  cand_z={cand_A_z:+.1f}  "
          f"nearest truth z={cand_A:+.1f}  dist={dist_A:.1f} mm")
    print(f"[validate]   mapping B (sign-flip): cand_z={cand_B_z:+.1f}  "
          f"nearest truth z={cand_B:+.1f}  dist={dist_B:.1f} mm")
    if dist_B < dist_A:
        chosen_z = cand_B_z
        print(f"[validate]   → using sign-flipped mapping (Mayo head-first CT convention)")
    else:
        chosen_z = cand_A_z
        print(f"[validate]   → using identity mapping")

    # Find the two truth slices bracketing chosen_z and linearly interpolate.
    sorted_pairs = sorted(truth_files, key=lambda t: t[0])
    sorted_zs = np.array([z for z, _ in sorted_pairs])
    idx_above = int(np.searchsorted(sorted_zs, chosen_z, side="left"))
    if idx_above <= 0:
        # chosen_z below all truth slices; clamp to lowest
        lo_idx = hi_idx = 0
    elif idx_above >= len(sorted_pairs):
        # chosen_z above all truth slices; clamp to highest
        lo_idx = hi_idx = len(sorted_pairs) - 1
    else:
        lo_idx = idx_above - 1
        hi_idx = idx_above
    z_lo, fp_lo = sorted_pairs[lo_idx]
    z_hi, fp_hi = sorted_pairs[hi_idx]
    print(f"[validate] truth bracket: {fp_lo.name} (z={z_lo:.3f}) "
          f"+ {fp_hi.name} (z={z_hi:.3f}) "
          f"around target {chosen_z:.3f}")

    def _load_mu(fp):
        ds = pydicom.dcmread(str(fp))
        pix = ds.pixel_array.astype(np.float32)
        slope = float(getattr(ds, "RescaleSlope", 1.0))
        intercept = float(getattr(ds, "RescaleIntercept", 0.0))
        hu = pix * slope + intercept
        return MU_WATER_PER_MM * (1.0 + hu.astype(np.float32) / 1000.0)

    if lo_idx == hi_idx or z_hi == z_lo:
        weight_lo = 1.0
        truth_interp = _load_mu(fp_lo)
    else:
        # Linear interpolation weight on the lower-z slice.
        # weight_lo = 1 when chosen_z == z_lo, 0 when chosen_z == z_hi.
        weight_lo = float((z_hi - chosen_z) / (z_hi - z_lo))
        weight_lo = max(0.0, min(1.0, weight_lo))
        mu_lo = _load_mu(fp_lo)
        mu_hi = _load_mu(fp_hi)
        truth_interp = (weight_lo * mu_lo + (1.0 - weight_lo) * mu_hi).astype(np.float32)
    print(f"[validate] weight on lower-z slice = {weight_lo:.3f}")
    return truth_interp, chosen_z, (float(z_lo), float(z_hi), float(weight_lo))


if __name__ == "__main__":
    sys.exit(main())
