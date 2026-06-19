"""Mayo FOV-masked evaluation experiment (central slices, val + test).

The current Mayo HD/LD-FBP baseline + every solver score via
``evaluate_calibrated(..., fov=False)`` — NO FOV mask. The patient table and
the recon periphery therefore contaminate SSIM/RMSE and dominate the diff
images, even though they carry no diagnostic signal.

This script does NOT change the production eval. It is a *measurement* run to
show the effect BEFORE we commit to a masked convention:

  * For val (L277) + the 5 test patients, take the CENTRAL reconstructable
    truth slice (median source_z in the fan z-range).
  * Reconstruct HD-FBP and LD-FBP on the exact production path
    (mayo_ldct_fitted geom, MAYO_LDCT_DET_OFFSET, MAYO_LDCT_TRUNCATION,
    per-patient ps_eff, 5 mm sino-domain slab, bg_target="truth").
  * Sweep a set of centred circular FOV radii (fraction of the 256-px
    inscribed circle) + the unmasked baseline, and report calibrated
    SSIM / RMSE for GT-vs-HD and GT-vs-LD at each.
  * QUANTITATIVELY localise the residual (no eyeballing — CT vision is
    unreliable here): radial RMSE-contribution profile + row-wise mean |LD-GT|
    profile (a table band shows up as a spike at high row index).
  * Render FOV-cropped comparison figures (GT | HD | LD | LD-GT diff), full
    vs masked, with the chosen FOV circle overlaid, for the user to inspect.

Outputs (results/mayo_debug/fov_eval/):
  fov_eval_metrics.json          per-slice + mean metrics for every FOV radius
  fov_sweep.png                  SSIM/RMSE vs FOV radius + radial & row profiles
  <patient>_fov_compare.png      GT|HD|LD|diff, full(+circle) vs FOV-masked
"""
from __future__ import annotations
import argparse, json, math, os, sys, time
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from ddssl_ldct.metrics import evaluate_calibrated
# reuse the exact production data-loading + FBP path
from scripts.compare_hd_ld_fbp_allslices import (
    WAGNER_SPLIT_OF, enumerate_truth, _load_mu, build_projector, fbp_slab_mean,
)

VAL_TEST = ["L277", "L014", "L056", "L058", "L075", "L123"]


def fov_radius_mm(g) -> float:
    """Physical measurement-FOV radius of a fan-beam geometry: the largest
    circle about isocentre seen by every view. R = SOD * sin(gamma_max),
    gamma_max = atan(half_detector_width / SDD). FOV-independent (does not
    depend on the recon pixel spacing). ~237.5 mm for the fitted Mayo geom."""
    half_w = 0.5 * g.n_det * g.det_spacing
    return g.sod * math.sin(math.atan(half_w / g.sdd))


def circ_mask(size: int, radius_pix: float, device, dtype=torch.float32):
    coords = torch.arange(size, device=device, dtype=dtype) - (size - 1) / 2.0
    yy, xx = torch.meshgrid(coords, coords, indexing="ij")
    return ((xx * xx + yy * yy) <= radius_pix * radius_pix).to(dtype)


def cal_with_fov(fbp_np, truth_t, dr, device, fov):
    """Calibrated SSIM/RMSE of a single FBP vs truth under a given fov arg
    (False, True, or a (H,W) mask tensor). Returns (ssim, rmse, pred_cal_np)."""
    fbp_t = torch.from_numpy(fbp_np).to(device).float()[None, None].clamp(min=0.0)
    c = evaluate_calibrated(fbp_t, truth_t, baseline=None,
                            display_min=0.0, display_max=dr, fov=fov,
                            bg_target="truth")
    return float(c["val_ssim"]), float(c["val_rmse"]), c["pred_cal"][0, 0].cpu().numpy()


def central_slice(patient, sino_dir, truth_root, args):
    """Reconstruct HD/LD FBP for the CENTRAL reconstructable truth slice."""
    split = WAGNER_SPLIT_OF[patient]
    paths = {d: (sino_dir / f"{patient}_sino_{d}.h5",
                 sino_dir / f"{patient}_sino_{d}_geometry.json",
                 sino_dir / f"{patient}_sino_{d}_z_grid.npy")
             for d in ("fulldose", "lowdose")}
    truth_files, tmeta = enumerate_truth(truth_root / patient)
    truth_ps = tmeta["pixel_spacing"]
    ps_eff = 0.700857 * (truth_ps / 0.703125)
    dr = float(args.display_max)
    geom = {d: json.loads(paths[d][1].read_text()) for d in paths}
    zgrid = {d: np.load(paths[d][2]) for d in paths}
    proj = build_projector(geom["fulldose"], ps_eff, args.device)
    R_FOV_mm = fov_radius_mm(proj.geom)
    sino = {}
    for d in paths:
        with h5py.File(paths[d][0], "r") as f:
            sino[d] = np.ascontiguousarray(np.asarray(f["sino"][...], dtype=np.float32))
    zmin, zmax = float(zgrid["fulldose"].min()), float(zgrid["fulldose"].max())
    recon = [(pz, fp) for (pz, fp) in truth_files if zmin - 5 <= -pz <= zmax + 5]
    pz, fp = recon[len(recon) // 2]          # central reconstructable slice
    truth = _load_mu(fp)
    truth_t = torch.from_numpy(truth).to(args.device).float()[None, None]
    out = {"patient": patient, "split": split, "truth_z": pz, "ps_eff": ps_eff,
           "truth_ps": truth_ps, "fov_mm": R_FOV_mm, "fov_pix": R_FOV_mm / ps_eff,
           "_truth": truth, "_fbp": {}}
    for d, dose in (("fulldose", "hd"), ("lowdose", "ld")):
        fbp, _, _ = fbp_slab_mean(sino[d], zgrid[d], -pz, args.slab_half, proj, args.device)
        out["_fbp"][dose] = fbp
    out["_truth_t"] = truth_t
    out["_dr"] = dr
    del sino
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--slab-half", type=int, default=2)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--display-max", type=float, default=0.05)
    args = ap.parse_args()

    root = Path(args.data_root or os.environ.get(
        "AGENT4CT_DATA", "/cluster/maier/Agent4CT/data")) / "mayo_ldct"
    sino_dir = root / os.environ.get("STAGED_HELIX2FAN_SUBDIR", "staged_helix2fan_v3")
    truth_root = root / "raw"
    out_dir = Path(args.out_dir) if args.out_dir else REPO / "results" / "mayo_debug" / "fov_eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    dev = args.device
    dr = float(args.display_max)

    slices = []
    for pat in VAL_TEST:
        t0 = time.time()
        s = central_slice(pat, sino_dir, truth_root, args)
        slices.append(s)
        print(f"[fov] {pat} ({s['split']}) central z={s['truth_z']:.1f} "
              f"ps_eff={s['ps_eff']:.4f}  [{time.time()-t0:.0f}s]", flush=True)

    # FOV candidates: none (current production), geom (physical measurement
    # FOV, per-patient R_FOV_mm/ps_eff), inscribed-256 (reference / other-
    # dataset convention).
    fov_kinds = ["none", "geom", "inscribed"]
    metrics = {"display_max": dr, "fovs": {},
               "fov_mm": float(np.mean([s["fov_mm"] for s in slices]))}

    for name in fov_kinds:
        per = []
        for s in slices:
            if name == "none":
                fov, rpx = False, float("nan")
            elif name == "geom":
                rpx = float(s["fov_pix"]); fov = circ_mask(512, rpx, dev)
            else:  # inscribed circle (radius 256 px)
                rpx = 256.0; fov = circ_mask(512, rpx, dev)
            hd_ss, hd_rm, _ = cal_with_fov(s["_fbp"]["hd"], s["_truth_t"], dr, dev, fov)
            ld_ss, ld_rm, _ = cal_with_fov(s["_fbp"]["ld"], s["_truth_t"], dr, dev, fov)
            per.append({"patient": s["patient"], "split": s["split"], "radius_px": rpx,
                        "hd_ssim": hd_ss, "hd_rmse": hd_rm,
                        "ld_ssim": ld_ss, "ld_rmse": ld_rm})
        def mean(split, key):
            v = [p[key] for p in per if (split is None or p["split"] == split)]
            return float(np.mean(v)) if v else float("nan")
        metrics["fovs"][name] = {
            "mean_radius_px": float(np.nanmean([p["radius_px"] for p in per])),
            "val":  {k: mean("val", k)  for k in ("hd_ssim", "hd_rmse", "ld_ssim", "ld_rmse")},
            "test": {k: mean("test", k) for k in ("hd_ssim", "hd_rmse", "ld_ssim", "ld_rmse")},
            "all":  {k: mean(None, k)   for k in ("hd_ssim", "hd_rmse", "ld_ssim", "ld_rmse")},
            "per_slice": per,
        }

    # ---- residual localisation (quantitative; LD-GT, calibrated, no mask) ----
    # radial RMSE-contribution + row-mean |LD-GT|, averaged over all 6 slices.
    coords = np.arange(512) - 255.5
    YY, XX = np.meshgrid(coords, coords, indexing="ij")
    RR = np.sqrt(XX * XX + YY * YY)
    rbins = np.linspace(0, 362, 41)            # 0..~corner radius
    rad_ld = np.zeros(len(rbins) - 1); rad_hd = np.zeros(len(rbins) - 1)
    rad_cnt = np.zeros(len(rbins) - 1)
    row_ld = np.zeros(512)
    for s in slices:
        _, _, ld_cal = cal_with_fov(s["_fbp"]["ld"], s["_truth_t"], dr, dev, False)
        _, _, hd_cal = cal_with_fov(s["_fbp"]["hd"], s["_truth_t"], dr, dev, False)
        d_ld = (ld_cal - s["_truth"]); d_hd = (hd_cal - s["_truth"])
        row_ld += np.abs(d_ld).mean(axis=1)
        idx = np.digitize(RR.ravel(), rbins) - 1
        for b in range(len(rbins) - 1):
            m = idx == b
            if m.any():
                rad_ld[b] += (d_ld.ravel()[m] ** 2).sum()
                rad_hd[b] += (d_hd.ravel()[m] ** 2).sum()
                rad_cnt[b] += m.sum()
    row_ld /= len(slices)
    rad_ld_rmse = np.sqrt(rad_ld / np.maximum(rad_cnt, 1))
    rad_hd_rmse = np.sqrt(rad_hd / np.maximum(rad_cnt, 1))
    rcent = 0.5 * (rbins[:-1] + rbins[1:])
    metrics["residual_profiles"] = {
        "radius_px": rcent.tolist(),
        "ld_rmse_by_radius": rad_ld_rmse.tolist(),
        "hd_rmse_by_radius": rad_hd_rmse.tolist(),
        "row_mean_abs_ld_resid": row_ld.tolist(),
        "inscribed_radius_px": 256.0,
    }
    (out_dir / "fov_eval_metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"[fov] wrote {out_dir/'fov_eval_metrics.json'}", flush=True)

    # ---- summary figure ----
    names = fov_kinds
    rgeom = metrics["fovs"]["geom"]["mean_radius_px"]
    lbls = {"none": "none (current)", "geom": f"geom-FOV (~{rgeom:.0f}px)",
            "inscribed": "inscribed (256px)"}
    xpos = np.arange(len(names))
    fig, ax = plt.subplots(2, 2, figsize=(14, 9))
    for off, sp, c in ((-0.18, "val", "tab:green"), (0.18, "test", "tab:purple")):
        ax[0, 0].bar(xpos + off, [metrics["fovs"][n][sp]["ld_ssim"] for n in names], 0.32,
                     color=c, alpha=0.85, label=f"{sp} LD")
        ax[0, 0].plot(xpos + off, [metrics["fovs"][n][sp]["hd_ssim"] for n in names], "k_",
                      ms=18, mew=2, label=f"{sp} HD" if off < 0 else None)
        ax[0, 1].bar(xpos + off, [metrics["fovs"][n][sp]["ld_rmse"] for n in names], 0.32,
                     color=c, alpha=0.85, label=f"{sp} LD")
    for a in (ax[0, 0], ax[0, 1]):
        a.set_xticks(xpos); a.set_xticklabels([lbls[n] for n in names], fontsize=8)
        a.grid(alpha=0.3, axis="y"); a.legend(fontsize=8)
    ax[0, 0].set_title("Calibrated LD-FBP SSIM (bars) + HD (ticks)"); ax[0, 0].set_ylabel("SSIM")
    ax[0, 1].set_title("Calibrated LD-FBP RMSE"); ax[0, 1].set_ylabel("RMSE (μ⁻¹)")
    ax[1, 0].plot(rcent, rad_ld_rmse, "o-", color="tab:red", label="LD-GT")
    ax[1, 0].plot(rcent, rad_hd_rmse, "s--", color="tab:blue", label="HD-GT")
    ax[1, 0].axvline(256, color="k", ls=":", alpha=0.5, label="inscribed (256)")
    ax[1, 0].axvline(rgeom, color="tab:green", ls="-", alpha=0.7, label=f"geom-FOV ({rgeom:.0f})")
    ax[1, 0].set_xlabel("radius from centre (px)"); ax[1, 0].set_ylabel("RMSE in radial bin")
    ax[1, 0].set_title("WHERE the residual lives (radial)"); ax[1, 0].grid(alpha=0.3); ax[1, 0].legend(fontsize=8)
    ax[1, 1].plot(row_ld, np.arange(512), "-", color="tab:red")
    ax[1, 1].invert_yaxis()
    ax[1, 1].set_ylabel("row index (0=top, 511=bottom)"); ax[1, 1].set_xlabel("mean |LD-GT| in row")
    ax[1, 1].set_title("WHERE the residual lives (rows; table=bottom spike)"); ax[1, 1].grid(alpha=0.3)
    fig.suptitle(f"Mayo geometry-FOV eval — val(L277)+5 test central slices  "
                 f"[R_FOV={metrics['fov_mm']:.1f}mm flat-detector]", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_dir / "fov_sweep.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"[fov] wrote {out_dir/'fov_sweep.png'}", flush=True)

    # ---- per-slice compare figures (full + geometry-FOV circle, vs masked) ----
    for s in slices:
        rpix = float(s["fov_pix"])        # physical measurement FOV for this slice
        m = circ_mask(512, rpix, dev)
        _, _, ld_cal = cal_with_fov(s["_fbp"]["ld"], s["_truth_t"], dr, dev, False)
        _, _, hd_cal = cal_with_fov(s["_fbp"]["hd"], s["_truth_t"], dr, dev, False)
        mn = m.cpu().numpy()
        gt = s["_truth"]
        rows = [
            ("FULL (unmasked)", gt, hd_cal, ld_cal, ld_cal - gt, False),
            (f"FOV-masked r={int(rpix)}px", gt * mn, hd_cal * mn, ld_cal * mn,
             (ld_cal - gt) * mn, True),
        ]
        fig, ax = plt.subplots(2, 4, figsize=(16, 8.4))
        for ri, (lbl, g, h, l, dff, masked) in enumerate(rows):
            panels = [(g, "GT", "gray", 0, dr), (h, "HD-FBP", "gray", 0, dr),
                      (l, "LD-FBP", "gray", 0, dr),
                      (dff, "LD − GT", "seismic", -0.015, 0.015)]
            for ci, (img, ttl, cm, vmn, vmx) in enumerate(panels):
                a = ax[ri, ci]
                a.imshow(img, cmap=cm, vmin=vmn, vmax=vmx)
                a.set_title(f"{lbl}\n{ttl}", fontsize=9); a.set_xticks([]); a.set_yticks([])
                if not masked:
                    a.add_patch(Circle((255.5, 255.5), rpix, fill=False,
                                       color="lime", lw=1.2, ls="--"))
        fig.suptitle(f"{s['patient']} ({s['split']}) central z={s['truth_z']:.1f} — "
                     f"geometry FOV r={int(rpix)}px (R={s['fov_mm']:.1f}mm / ps_eff={s['ps_eff']:.4f}). "
                     f"top=full(+FOV circle), bottom=inside-FOV only", fontsize=11)
        fig.tight_layout(rect=[0, 0, 1, 0.97])
        fn = out_dir / f"{s['patient']}_fov_compare.png"
        fig.savefig(fn, dpi=115, bbox_inches="tight")
        plt.close(fig)
        print(f"[fov] wrote {fn}", flush=True)

    # ---- console summary ----
    print(f"\n[fov] ===== calibrated SSIM by FOV (R_FOV={metrics['fov_mm']:.1f}mm flat) =====", flush=True)
    print(f"  {'FOV':>12} {'r_px':>6} | {'val_HD':>7} {'val_LD':>7} | {'test_HD':>7} {'test_LD':>7}", flush=True)
    for n in fov_kinds:
        f = metrics["fovs"][n]
        print(f"  {n:>12} {f['mean_radius_px']:6.0f} | "
              f"{f['val']['hd_ssim']:7.4f} {f['val']['ld_ssim']:7.4f} | "
              f"{f['test']['hd_ssim']:7.4f} {f['test']['ld_ssim']:7.4f}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
