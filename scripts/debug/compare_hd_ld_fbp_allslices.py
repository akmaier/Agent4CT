"""All-slices HD vs LD FBP baseline for the Mayo Wagner split.

This establishes the **oracle (HD FBP)** and **baseline (LD FBP)** of the
headroom-scoring convention over EVERY reconstructable truth slice of all 10
Wagner patients — the foundation for rebuilding the Mayo leaderboard from
scratch after the bg->0 calibration bug (see findings.md 2026-06-13).

Everything is the v3-final production path, hard-wired:
  * rebin sinos      : staged_helix2fan_v3  (v3 SSR fit + s_z)
  * FBP geometry     : FanBeamGeometry.mayo_ldct_fitted()  (Powell 5-param)
  * detector offset  : MAYO_LDCT_DET_OFFSET
  * truncation corr  : MAYO_LDCT_TRUNCATION  (water-cylinder extrapolation)
  * calibration      : evaluate_calibrated(..., bg_target="truth")  (Mayo bg!=0)
  * per-patient FOV  : ps_eff = 0.700857 * truth_ps / 0.703125

Unlike compare_gt_hd_ld_fbp_wagner_trunc.py (one slab per patient), this loops
over ALL full-dose truth-image DICOM slices. For each truth slice at patient-z
pz, the matching fan slice is found by the validated sign-flip mapping
(source_z = -pz), a 5 mm slab (slab_half=2) is averaged in the SINOGRAM domain
(FBP is linear, so slab-mean-then-FBP == mean-of-slab-FBPs, 5x cheaper and the
physically-correct model of a 5 mm-thick truth slice), reconstructed once per
dose, and scored against that truth slice.

Outputs (out-dir default results/mayo_debug/allslices_hd_ld):
  <tag>_metrics.json                 per-slice + per-patient + per-split + overall
  <pat>_ssim_vs_z.png                HD/LD SSIM vs z, per patient
  <pat>_montage.png                  3 representative slices (min/med/max LD SSIM)
  summary.png                        per-patient + per-split SSIM box/bars (HD vs LD)
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from ddssl_ldct.geometry import (
    FanBeamGeometry, MAYO_LDCT_DET_OFFSET, MAYO_LDCT_TRUNCATION,
)
from ddssl_ldct.pyronn_projector import PyronnFanBeamProjector
from ddssl_ldct.metrics import evaluate_calibrated

WAGNER_SPLIT_OF = {
    **{p: "train" for p in ["L145", "L186", "L209", "L219"]},
    "L277": "val",
    **{p: "test" for p in ["L014", "L056", "L058", "L075", "L123"]},
}
WAGNER_ALL = ["L145", "L186", "L209", "L219", "L277",
              "L014", "L056", "L058", "L075", "L123"]
MU_WATER_PER_MM = 0.02
SOP_CT_IMAGE = "1.2.840.10008.5.1.4.1.1.2"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", default=None)
    p.add_argument("--tag", default="v3allslices")
    p.add_argument("--out-dir", default=None)
    p.add_argument("--patients", default=None,
                   help="comma list to restrict (default: all 10 Wagner).")
    p.add_argument("--slab-half", type=int, default=2,
                   help="half-width of the fan slab (5 mm = 2 -> 5x1 mm).")
    p.add_argument("--stride", type=int, default=1,
                   help="evaluate every Nth truth slice (1 = all).")
    p.add_argument("--device", default="cuda")
    p.add_argument("--display-max", type=float, default=0.05)
    return p.parse_args()


def _load_mu(fp):
    import pydicom
    ds = pydicom.dcmread(str(fp))
    pix = ds.pixel_array.astype(np.float32)
    slope = float(getattr(ds, "RescaleSlope", 1.0))
    intercept = float(getattr(ds, "RescaleIntercept", 0.0))
    hu = pix * slope + intercept
    return MU_WATER_PER_MM * (1.0 + hu.astype(np.float32) / 1000.0)


def enumerate_truth(patient_dir: Path):
    """Return (sorted [(patient_z, filepath)], meta) for the full-dose image
    series — same series-selection rule as validate_mayo_helix2fan."""
    import pydicom
    files = []
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
            files.append((z, fp))
    if not files:
        return None, None
    files.sort(key=lambda t: t[0])
    ds0 = pydicom.dcmread(str(files[0][1]), stop_before_pixels=True)
    meta = {
        "pixel_spacing": float(ds0.PixelSpacing[0]),
        "recon_kernel": str(getattr(ds0, "ConvolutionKernel", "?")),
        "slice_thickness": float(getattr(ds0, "SliceThickness", -1.0)),
        "n_slices": len(files),
    }
    return files, meta


def build_projector(geom_json, ps_eff, device):
    rotview = int(geom_json["rotview"])
    nu = int(geom_json["nu"])
    angle_start = float(geom_json.get("angle_start_corrected", 0.0))
    angle_end = angle_start + 2.0 * math.pi
    fitted = FanBeamGeometry.mayo_ldct_fitted(
        n_angles=rotview, n_det=nu, angle_start=angle_start, angle_end=angle_end)
    geom = FanBeamGeometry(
        image_size=512, pixel_spacing=ps_eff, n_angles=rotview, n_det=nu,
        det_spacing=fitted.det_spacing, sod=fitted.sod, sdd=fitted.sdd,
        angle_start=angle_start, angle_end=angle_end)
    return PyronnFanBeamProjector(
        geom, det_offset_mm=MAYO_LDCT_DET_OFFSET,
        truncation=MAYO_LDCT_TRUNCATION).to(device)


def fbp_slab_mean(sino, zgrid, source_z, slab_half, proj, device):
    """Sino-domain slab average -> single truncation-corrected FBP.

    ``sino`` is the FULL in-RAM (rotview, nu, nz) array. The staged h5 is
    chunked along the view axis ((1, nu, nz)), so reading sino[:,:,k] from
    disk re-reads the whole file; we load it into RAM once per patient in
    run_patient and index in memory here.
    """
    nz = zgrid.shape[0]
    j = int(np.argmin(np.abs(zgrid - source_z)))
    lo = max(0, j - slab_half)
    hi = min(nz, j + slab_half + 1)
    # mean over the slab (z axis), then flip the detector (u) axis
    slab_mean = np.ascontiguousarray(
        np.flip(sino[:, :, lo:hi].mean(axis=2), axis=-1))
    s_t = torch.from_numpy(slab_mean).to(device).float()[None, None]
    fbp = proj.fbp(s_t).detach()[0, 0].cpu().numpy()
    return np.ascontiguousarray(np.fliplr(np.flipud(fbp))), float(zgrid[j]), j


def cal(fbp_np, truth_t, dr, device):
    fbp_t = torch.from_numpy(fbp_np).to(device).float()[None, None].clamp(min=0.0)
    c = evaluate_calibrated(fbp_t, truth_t, baseline=fbp_t,
                            display_min=0.0, display_max=dr, fov=False,
                            bg_target="truth")
    return (c["pred_cal"][0, 0].cpu().numpy().astype(np.float32),
            float(c["val_ssim"]), float(c["val_psnr"]), float(c["val_rmse"]))


def run_patient(patient, sino_dir, truth_root, args):
    split = WAGNER_SPLIT_OF[patient]
    t0 = time.time()
    paths = {d: (sino_dir / f"{patient}_sino_{d}.h5",
                 sino_dir / f"{patient}_sino_{d}_geometry.json",
                 sino_dir / f"{patient}_sino_{d}_z_grid.npy")
             for d in ("fulldose", "lowdose")}
    for d, (h5, gj, zg) in paths.items():
        for p in (h5, gj, zg):
            if not p.exists():
                return {"patient": patient, "split": split, "error": f"missing {p.name}"}
    truth_files, tmeta = enumerate_truth(truth_root / patient)
    if not truth_files:
        return {"patient": patient, "split": split, "error": "no truth"}

    truth_ps = tmeta["pixel_spacing"]
    ps_eff = 0.700857 * (truth_ps / 0.703125)
    dr = float(args.display_max)

    geom = {d: json.loads(paths[d][1].read_text()) for d in paths}
    zgrid = {d: np.load(paths[d][2]) for d in paths}
    proj = build_projector(geom["fulldose"], ps_eff, args.device)  # HD/LD share geom

    # Load each sino FULLY into RAM once (the h5 is chunked along the view
    # axis -> reading one z-plane from disk re-reads the whole file).
    sino = {}
    for d in paths:
        with h5py.File(paths[d][0], "r") as f:
            sino[d] = np.ascontiguousarray(np.asarray(f["sino"][...], dtype=np.float32))
    print(f"[allslices] {patient}: loaded HD{sino['fulldose'].shape} "
          f"LD{sino['lowdose'].shape} into RAM", flush=True)

    # only truth slices whose source_z falls inside the fan z-range
    zmin, zmax = float(zgrid["fulldose"].min()), float(zgrid["fulldose"].max())

    rows = []
    for si, (pz, fp) in enumerate(truth_files):
        if si % args.stride != 0:
            continue
        source_z = -pz  # Mayo head-first sign flip
        if not (zmin - 5.0 <= source_z <= zmax + 5.0):
            continue
        truth = _load_mu(fp)
        truth_t = torch.from_numpy(truth).to(args.device).float()[None, None]
        rec = {"patient": patient, "split": split, "truth_z": pz,
               "source_z": source_z, "kernel": tmeta["recon_kernel"]}
        imgs = {}
        for d, dose in (("fulldose", "hd"), ("lowdose", "ld")):
            fbp, fanz, j = fbp_slab_mean(sino[d], zgrid[d], source_z,
                                         args.slab_half, proj, args.device)
            cimg, ss, ps, rm = cal(fbp, truth_t, dr, args.device)
            rec[f"{dose}_ssim"] = ss
            rec[f"{dose}_psnr"] = ps
            rec[f"{dose}_rmse"] = rm
            rec[f"{dose}_zres"] = abs(fanz - source_z)
            imgs[dose] = cimg
        rec["_truth"] = truth
        rec["_imgs"] = imgs
        rows.append(rec)
    del sino

    if not rows:
        return {"patient": patient, "split": split, "error": "no reconstructable slices"}
    hd = np.array([r["hd_ssim"] for r in rows])
    ld = np.array([r["ld_ssim"] for r in rows])
    print(f"[allslices] {patient} ({split:5}) ps={truth_ps:.4f} "
          f"n={len(rows)}/{tmeta['n_slices']}  "
          f"HD SSIM {hd.mean():.4f}+-{hd.std():.4f}  "
          f"LD SSIM {ld.mean():.4f}+-{ld.std():.4f}  "
          f"[{time.time()-t0:.0f}s]", flush=True)
    return {"patient": patient, "split": split, "truth_ps": truth_ps,
            "kernel": tmeta["recon_kernel"], "n_truth": tmeta["n_slices"],
            "rows": rows}


def agg(arr):
    a = np.asarray(arr, dtype=np.float64)
    return {"mean": float(a.mean()), "std": float(a.std()),
            "median": float(np.median(a)), "min": float(a.min()),
            "max": float(a.max()), "n": int(a.size)}


def plot_patient(pat_res, out_dir, dr):
    rows = pat_res["rows"]
    pat = pat_res["patient"]
    z = np.array([r["truth_z"] for r in rows])
    order = np.argsort(z)
    z = z[order]
    hd = np.array([r["hd_ssim"] for r in rows])[order]
    ld = np.array([r["ld_ssim"] for r in rows])[order]
    # SSIM vs z
    fig, ax = plt.subplots(figsize=(9, 3.2))
    ax.plot(z, hd, "-", color="tab:blue", label=f"HD FBP (oracle)  mean {hd.mean():.3f}")
    ax.plot(z, ld, "-", color="tab:red", label=f"LD FBP (baseline) mean {ld.mean():.3f}")
    ax.set_xlabel("truth z (patient frame, mm)")
    ax.set_ylabel("calibrated SSIM")
    ax.set_title(f"{pat} ({pat_res['split']}) — per-slice SSIM, all {len(rows)} slices")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / f"{pat}_ssim_vs_z.png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    # montage: min / median / max LD SSIM
    ld_all = np.array([r["ld_ssim"] for r in rows])
    idxs = {"min LD": int(np.argmin(ld_all)),
            "median LD": int(np.argsort(ld_all)[len(ld_all) // 2]),
            "max LD": int(np.argmax(ld_all))}
    fig, ax = plt.subplots(len(idxs), 4, figsize=(16, 4.0 * len(idxs)))
    for ri, (lbl, i) in enumerate(idxs.items()):
        r = rows[i]
        cols = [(r["_truth"], f"{pat} GT  z={r['truth_z']:.1f}", "gray", 0, dr),
                (r["_imgs"]["hd"], f"HD  SSIM={r['hd_ssim']:.3f}\nPSNR={r['hd_psnr']:.1f}", "gray", 0, dr),
                (r["_imgs"]["ld"], f"LD ({lbl})  SSIM={r['ld_ssim']:.3f}\nPSNR={r['ld_psnr']:.1f}", "gray", 0, dr),
                (r["_imgs"]["ld"] - r["_truth"], f"LD - GT\nRMSE={r['ld_rmse']:.4f}", "seismic", -0.015, 0.015)]
        for a, (img, ttl, cm, vmn, vmx) in zip(ax[ri], cols):
            a.imshow(img, cmap=cm, vmin=vmn, vmax=vmx); a.set_title(ttl, fontsize=9)
            a.set_xticks([]); a.set_yticks([])
    fig.suptitle(f"{pat} ({pat_res['split']}) — representative slices "
                 f"[v3 geom, trunc-corr, bg_target=truth]", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(out_dir / f"{pat}_montage.png", dpi=110, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    args = parse_args()
    root = Path(args.data_root or os.environ.get(
        "AGENT4CT_DATA", "/cluster/maier/Agent4CT/data")) / "mayo_ldct"
    sino_dir = root / os.environ.get("STAGED_HELIX2FAN_SUBDIR", "staged_helix2fan_v3")
    truth_root = root / "raw"
    out_dir = Path(args.out_dir) if args.out_dir else REPO / "results" / "mayo_debug" / "allslices_hd_ld"
    out_dir.mkdir(parents=True, exist_ok=True)
    pats = args.patients.split(",") if args.patients else WAGNER_ALL
    dr = float(args.display_max)
    print(f"[allslices] sino_dir={sino_dir} patients={pats} slab_half={args.slab_half} "
          f"stride={args.stride}", flush=True)

    results = []
    for pat in pats:
        try:
            r = run_patient(pat, sino_dir, truth_root, args)
        except Exception as e:
            import traceback; traceback.print_exc()
            r = {"patient": pat, "split": WAGNER_SPLIT_OF[pat], "error": repr(e)}
        if "error" in r:
            print(f"[allslices] {pat} ERROR: {r['error']}", flush=True)
        else:
            plot_patient(r, out_dir, dr)
        results.append(r)

    valid = [r for r in results if "error" not in r]

    # ---- aggregate per patient / split / overall ----
    def collect(rows, key):
        return [x for r in rows for x in [rr[key] for rr in r["rows"]]]
    summary = {"tag": args.tag, "slab_half": args.slab_half, "stride": args.stride,
               "patients": {}, "splits": {}, "overall": {}}
    for r in valid:
        summary["patients"][r["patient"]] = {
            "split": r["split"], "truth_ps": r["truth_ps"], "kernel": r["kernel"],
            "n_truth": r["n_truth"], "n_eval": len(r["rows"]),
            "hd_ssim": agg([x["hd_ssim"] for x in r["rows"]]),
            "ld_ssim": agg([x["ld_ssim"] for x in r["rows"]]),
            "hd_psnr": agg([x["hd_psnr"] for x in r["rows"]]),
            "ld_psnr": agg([x["ld_psnr"] for x in r["rows"]]),
            "zres_mean": float(np.mean([x["hd_zres"] for x in r["rows"]])),
        }
    for sp in ("train", "val", "test"):
        rs = [r for r in valid if r["split"] == sp]
        if not rs:
            continue
        summary["splits"][sp] = {
            "patients": [r["patient"] for r in rs],
            "n_eval": sum(len(r["rows"]) for r in rs),
            "hd_ssim": agg(collect(rs, "hd_ssim")), "ld_ssim": agg(collect(rs, "ld_ssim")),
            "hd_psnr": agg(collect(rs, "hd_psnr")), "ld_psnr": agg(collect(rs, "ld_psnr")),
        }
    if valid:
        summary["overall"] = {
            "n_eval": sum(len(r["rows"]) for r in valid),
            "hd_ssim": agg(collect(valid, "hd_ssim")), "ld_ssim": agg(collect(valid, "ld_ssim")),
            "hd_psnr": agg(collect(valid, "hd_psnr")), "ld_psnr": agg(collect(valid, "ld_psnr")),
        }
    summary["errors"] = {r["patient"]: r["error"] for r in results if "error" in r}
    oj = out_dir / f"{args.tag}_metrics.json"
    oj.write_text(json.dumps(summary, indent=2))
    print(f"[allslices] wrote {oj}", flush=True)

    # ---- summary figure: per-patient SSIM box (HD vs LD) ----
    if valid:
        fig, (axb, axs) = plt.subplots(1, 2, figsize=(17, 5),
                                       gridspec_kw={"width_ratios": [3, 1]})
        labels = [r["patient"] for r in valid]
        hd_data = [[x["hd_ssim"] for x in r["rows"]] for r in valid]
        ld_data = [[x["ld_ssim"] for x in r["rows"]] for r in valid]
        pos = np.arange(len(labels))
        bp1 = axb.boxplot(hd_data, positions=pos - 0.18, widths=0.3, patch_artist=True,
                          boxprops=dict(facecolor="tab:blue", alpha=0.6), showfliers=False)
        bp2 = axb.boxplot(ld_data, positions=pos + 0.18, widths=0.3, patch_artist=True,
                          boxprops=dict(facecolor="tab:red", alpha=0.6), showfliers=False)
        axb.set_xticks(pos)
        axb.set_xticklabels([f"{r['patient']}\n({r['split']})" for r in valid], fontsize=8)
        axb.set_ylabel("calibrated SSIM")
        axb.set_title("Per-patient per-slice SSIM — HD FBP (blue) vs LD FBP (red)")
        axb.legend([bp1["boxes"][0], bp2["boxes"][0]], ["HD (oracle)", "LD (baseline)"], fontsize=9)
        axb.grid(alpha=0.3, axis="y")
        # split-level bars
        sps = [s for s in ("train", "val", "test") if s in summary["splits"]]
        x = np.arange(len(sps))
        hdm = [summary["splits"][s]["hd_ssim"]["mean"] for s in sps]
        ldm = [summary["splits"][s]["ld_ssim"]["mean"] for s in sps]
        axs.bar(x - 0.2, hdm, 0.4, color="tab:blue", alpha=0.7, label="HD")
        axs.bar(x + 0.2, ldm, 0.4, color="tab:red", alpha=0.7, label="LD")
        for xi, (h, l) in enumerate(zip(hdm, ldm)):
            axs.text(xi - 0.2, h + 0.005, f"{h:.3f}", ha="center", fontsize=7)
            axs.text(xi + 0.2, l + 0.005, f"{l:.3f}", ha="center", fontsize=7)
        axs.set_xticks(x); axs.set_xticklabels(sps)
        axs.set_ylim(0, 1.0); axs.set_ylabel("mean SSIM")
        axs.set_title("Split means"); axs.legend(fontsize=8); axs.grid(alpha=0.3, axis="y")
        fig.suptitle(f"Mayo Wagner all-slices HD vs LD FBP  "
                     f"[tag={args.tag}, n={summary['overall']['n_eval']} slices]", fontsize=12)
        fig.tight_layout(rect=[0, 0, 1, 0.97])
        fig.savefig(out_dir / "summary.png", dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"[allslices] wrote {out_dir/'summary.png'}", flush=True)

    # ---- console summary ----
    print("\n[allslices] ===== SUMMARY (calibrated SSIM) =====", flush=True)
    for pat, d in summary["patients"].items():
        print(f"  {pat} ({d['split']:5}) n={d['n_eval']:3}  "
              f"HD {d['hd_ssim']['mean']:.4f}  LD {d['ld_ssim']['mean']:.4f}  "
              f"gap {d['hd_ssim']['mean']-d['ld_ssim']['mean']:.4f}", flush=True)
    for sp, d in summary["splits"].items():
        print(f"  [{sp:5}] n={d['n_eval']:4}  HD {d['hd_ssim']['mean']:.4f}  "
              f"LD {d['ld_ssim']['mean']:.4f}", flush=True)
    if valid:
        o = summary["overall"]
        print(f"  [OVERALL] n={o['n_eval']}  HD {o['hd_ssim']['mean']:.4f}  "
              f"LD {o['ld_ssim']['mean']:.4f}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
