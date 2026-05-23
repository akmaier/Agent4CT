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
from ddssl_ldct.metrics import ssim as ssim_metric, psnr as psnr_metric  # noqa: E402


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

    # Load center z slice from rebinned sino: (rotview, nu, nz) -> (rotview, nu).
    nz_center = nz // 2
    z_center = float(z_grid[nz_center])
    with h5py.File(sino_h5, "r") as f:
        sino_center = np.asarray(f["sino"][:, :, nz_center], dtype=np.float32)
    # "Siemens flip" along the u (detector) axis, per
    # literature/wagner_helix2fan_algorithm.md (Bug 4). Wagner's
    # reco_example_fan_beam.py does `np.flip(projections[:, :, i], axis=1)`
    # before filtering — equivalent to axis=-1 for our 2D (rotview, nu)
    # slice. This undoes the curved-detector channel flip applied at
    # load time so the back-projection direction matches PYRO-NN's
    # right-handed convention.
    sino_center = np.ascontiguousarray(np.flip(sino_center, axis=-1))
    print(f"[validate] z_center = {z_center:.2f} mm (slice {nz_center}/{nz}); "
          f"Siemens-flipped u-axis")

    # Match truth slice via the staged manifest's z-ordered file list.
    # The truth h5 is shuffled by stage_h5; we need to recover (patient, z)
    # alignment. The cleanest path is to re-read the raw fulldose-image
    # DICOMs for this patient and pick the z that minimises |z - z_center|.
    truth_slice = _load_truth_slice_for_z(challenge / "raw" / args.patient,
                                          z_center)
    if truth_slice is None:
        print(f"[validate] could not align truth slice for {args.patient}",
              file=sys.stderr)
        return 2

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

    sino_t = torch.from_numpy(sino_center).to(args.device).float()[None, None]
    fbp = proj.fbp(sino_t).detach()  # (1, 1, H, W)
    fbp_np = fbp[0, 0].cpu().numpy()

    truth_t = torch.from_numpy(truth_slice).to(args.device).float()[None, None]
    # data_range as configured for ddssl_ldct (water = 0.02 mm^-1; tissue ~0.05).
    dr = 0.05
    ssim_val = float(ssim_metric(fbp.clamp(min=0.0), truth_t, dr).cpu())
    psnr_val = float(psnr_metric(fbp.clamp(min=0.0), truth_t, dr).cpu())

    print(f"[validate] SSIM = {ssim_val:.4f}")
    print(f"[validate] PSNR = {psnr_val:.2f} dB")
    print(f"[validate] truth slice  min={truth_slice.min():.4f} "
          f"max={truth_slice.max():.4f} mean={truth_slice.mean():.4f}")
    print(f"[validate] FBP   slice  min={fbp_np.min():.4f} "
          f"max={fbp_np.max():.4f} mean={fbp_np.mean():.4f}")

    out_png = Path(args.out_png) if args.out_png else (
        REPO / "scripts" / f"_validate_{args.patient}_{args.dose}.png"
    )
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(truth_slice, cmap="gray", vmin=0, vmax=dr)
    axes[0].set_title(f"truth z={z_center:.1f}")
    axes[1].imshow(fbp_np, cmap="gray", vmin=0, vmax=dr)
    axes[1].set_title(f"FBP(extract) SSIM={ssim_val:.3f}")
    diff = fbp_np - truth_slice
    axes[2].imshow(diff, cmap="seismic", vmin=-0.02, vmax=0.02)
    axes[2].set_title(f"diff PSNR={psnr_val:.1f} dB")
    for ax in axes:
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(f"{args.patient}/{args.dose} helix2fan validation")
    fig.tight_layout()
    fig.savefig(out_png, dpi=120)
    print(f"[validate] wrote {out_png}")

    return 0


def _load_truth_slice_for_z(patient_dir: Path, z_target: float) -> np.ndarray | None:
    """Find the full-dose-image DICOM slice closest to z_target (mm).

    Re-reads the raw DICOMs directly rather than the staged h5 because the
    staged h5 is shuffled and discards per-slice z metadata.
    """
    import pydicom
    SOP_CT_IMAGE = "1.2.840.10008.5.1.4.1.1.2"
    MU_WATER_PER_MM = 0.02

    best: tuple[float, Path] | None = None
    if not patient_dir.exists():
        print(f"[validate] {patient_dir} not found", file=sys.stderr)
        return None
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
            d = abs(z - z_target)
            if best is None or d < best[0]:
                best = (d, fp)
    if best is None:
        return None
    print(f"[validate] truth match: {best[1].name} dz={best[0]:.3f} mm")
    ds = pydicom.dcmread(str(best[1]))
    pix = ds.pixel_array.astype(np.float32)
    slope = float(getattr(ds, "RescaleSlope", 1.0))
    intercept = float(getattr(ds, "RescaleIntercept", 0.0))
    hu = pix * slope + intercept
    return MU_WATER_PER_MM * (1.0 + hu.astype(np.float32) / 1000.0)


if __name__ == "__main__":
    sys.exit(main())
