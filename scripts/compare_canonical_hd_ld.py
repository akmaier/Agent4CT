"""HD vs LD FBP comparison on the CANONICAL re-staged Mayo data (the user gate
before any solver runs). Reads data/mayo_ldct/staged_canonical/{split}_* and
reconstructs with ONE uniform geometry (angle_start=0, ps=0.700857) — the same
path the solvers will use. If the canonical re-stage is right, HD/LD should
match the per-patient baseline (train HD0.9501/LD0.8659, val HD0.9331/LD0.8078,
test HD0.9528/LD0.8848).

Per-split + per-patient calibrated SSIM/PSNR (full-image, bg_target=truth),
per-patient montages (min/med/max LD-SSIM: GT|HD|LD|LD−GT), summary figure.
"""
from __future__ import annotations
import json, math, os, sys
from pathlib import Path
import h5py, numpy as np, torch
import hdf5plugin  # noqa: F401 — registers the LZ4 filter for the canonical h5s
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

os.environ.setdefault("AGENT4CT_DATASET", "mayo_ldct_2d")
REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
from ddssl_ldct.geometry import FanBeamGeometry
from ddssl_ldct.pyronn_projector import PyronnFanBeamProjector
from ddssl_ldct.metrics import evaluate_calibrated
from scripts.compare_hd_ld_fbp_allslices import enumerate_truth

WAGNER = {"train": ["L145", "L186", "L209", "L219"], "val": ["L277"],
          "test": ["L014", "L056", "L058", "L075", "L123"]}
CANON = Path("/cluster/maier/Agent4CT/data/mayo_ldct/staged_canonical")
RAW = Path("/cluster/maier/Agent4CT/data/mayo_ldct/raw")
DR = 0.05
dev = "cuda"


def slot_patients(split):
    """Re-derive slot->patient in the SAME order stage_mayo_canonical used
    (WAGNER[split] patients, each patient's truth slices sorted by z)."""
    labels = []
    for pid in WAGNER[split]:
        tfiles, _ = enumerate_truth(RAW / pid)
        labels += [pid] * len(tfiles)
    return labels


def cal(fbp, truth):
    c = evaluate_calibrated(fbp, truth, baseline=fbp, display_min=0.0,
                            display_max=DR, fov=False)
    return (c["pred_cal"], float(c["val_ssim"]), float(c["val_psnr"]), float(c["val_rmse"]))


def run_split(split, proj, out_dir):
    with h5py.File(CANON / f"{split}_truth.h5", "r") as f:
        truth = torch.from_numpy(f["truth"][:]).to(dev).float().unsqueeze(1)
    res = {}
    for dose in ("fulldose", "lowdose"):
        with h5py.File(CANON / f"{split}_sino_{dose}.h5", "r") as f:
            sino = torch.from_numpy(f["sino"][:]).to(dev).float().unsqueeze(1)
        fbp = proj.fbp(sino).clamp(min=0.0)   # fbp() batch-chunks internally
        ss = np.array([cal(fbp[i:i+1], truth[i:i+1])[1] for i in range(fbp.shape[0])])
        ps = np.array([float(evaluate_calibrated(fbp[i:i+1], truth[i:i+1], baseline=fbp[i:i+1],
                       display_min=0.0, display_max=DR, fov=False)["val_psnr"]) for i in range(fbp.shape[0])])
        res[dose] = {"ssim": ss, "psnr": ps, "fbp": fbp}
        del sino
    labels = slot_patients(split)
    n = truth.shape[0]
    if len(labels) != n:
        labels = labels[:n] + ["?"] * max(0, n - len(labels))
    return {"split": split, "labels": labels, "truth": truth,
            "hd": res["fulldose"], "ld": res["lowdose"], "n": n}


def main():
    out_dir = REPO / "results" / "mayo_debug" / "canonical_hd_ld"
    out_dir.mkdir(parents=True, exist_ok=True)
    with h5py.File(CANON / "val_sino_lowdose.h5", "r") as f:
        rot, nu = f["sino"].shape[-2], f["sino"].shape[-1]
    geom = FanBeamGeometry(image_size=512, pixel_spacing=0.700857, n_angles=rot, n_det=nu,
                           det_spacing=1.285044, sod=595.362, sdd=1086.803,
                           angle_start=0.0, angle_end=2 * math.pi)
    proj = PyronnFanBeamProjector(geom).to(dev)   # env applies det_offset + truncation

    summary = {"splits": {}, "patients": {}}
    splits_data = {}
    for split in ("train", "val", "test"):
        if not (CANON / f"{split}_truth.h5").exists():
            print(f"[canon-cmp] {split}: not staged yet, skip", flush=True); continue
        d = run_split(split, proj, out_dir)
        splits_data[split] = d
        hd_ss, ld_ss = d["hd"]["ssim"], d["ld"]["ssim"]
        summary["splits"][split] = {
            "n": d["n"], "hd_ssim": float(hd_ss.mean()), "ld_ssim": float(ld_ss.mean()),
            "hd_psnr": float(d["hd"]["psnr"].mean()), "ld_psnr": float(d["ld"]["psnr"].mean()),
            "gap": float(hd_ss.mean() - ld_ss.mean())}
        labels = np.array(d["labels"])
        for pid in WAGNER[split]:
            m = labels == pid
            if m.sum():
                summary["patients"][pid] = {
                    "split": split, "n": int(m.sum()),
                    "hd_ssim": float(hd_ss[m].mean()), "ld_ssim": float(ld_ss[m].mean()),
                    "gap": float(hd_ss[m].mean() - ld_ss[m].mean())}
        print(f"[canon-cmp] {split}: HD {hd_ss.mean():.4f} LD {ld_ss.mean():.4f} "
              f"gap {hd_ss.mean()-ld_ss.mean():.4f} (n={d['n']})", flush=True)
        # per-patient montage (use first patient of split as representative)
        for pid in WAGNER[split]:
            idx = np.where(labels == pid)[0]
            if len(idx) == 0:
                continue
            lss = d["ld"]["ssim"][idx]
            picks = {"min LD": idx[int(np.argmin(lss))],
                     "med LD": idx[int(np.argsort(lss)[len(lss)//2])],
                     "max LD": idx[int(np.argmax(lss))]}
            fig, ax = plt.subplots(3, 4, figsize=(16, 12))
            for ri, (lbl, i) in enumerate(picks.items()):
                t = d["truth"][i, 0].cpu().numpy()
                hd = d["hd"]["fbp"][i, 0].cpu().numpy(); ld = d["ld"]["fbp"][i, 0].cpu().numpy()
                # recompute calibrated for display
                hdc = cal(d["hd"]["fbp"][i:i+1], d["truth"][i:i+1])[0][0,0].cpu().numpy()
                ldc = cal(d["ld"]["fbp"][i:i+1], d["truth"][i:i+1])[0][0,0].cpu().numpy()
                cols = [(t, f"{pid} GT", "gray", 0, DR),
                        (hdc, f"HD SSIM={d['hd']['ssim'][i]:.3f}", "gray", 0, DR),
                        (ldc, f"LD ({lbl}) SSIM={d['ld']['ssim'][i]:.3f}", "gray", 0, DR),
                        (ldc - t, "LD-GT", "seismic", -0.015, 0.015)]
                for a, (im, ti, cm, vmn, vmx) in zip(ax[ri], cols):
                    a.imshow(im, cmap=cm, vmin=vmn, vmax=vmx); a.set_title(ti, fontsize=9)
                    a.set_xticks([]); a.set_yticks([])
            fig.suptitle(f"{pid} ({split}) canonical HD/LD FBP [uniform geom, ps=0.700857]", fontsize=11)
            fig.tight_layout(rect=[0,0,1,0.98])
            fig.savefig(out_dir / f"{pid}_montage.png", dpi=110, bbox_inches="tight"); plt.close(fig)

    # overall
    allhd = np.concatenate([splits_data[s]["hd"]["ssim"] for s in splits_data])
    allld = np.concatenate([splits_data[s]["ld"]["ssim"] for s in splits_data])
    summary["overall"] = {"n": int(allhd.size), "hd_ssim": float(allhd.mean()),
                          "ld_ssim": float(allld.mean()), "gap": float(allhd.mean()-allld.mean())}
    (out_dir / "canonical_hd_ld_metrics.json").write_text(json.dumps(summary, indent=2))

    # summary bar: per-split HD vs LD, canonical vs baseline
    base = {"train": (0.9501, 0.8659), "val": (0.9331, 0.8078), "test": (0.9528, 0.8848)}
    sps = [s for s in ("train","val","test") if s in summary["splits"]]
    fig, ax = plt.subplots(figsize=(10, 5)); x = np.arange(len(sps)); w = 0.2
    ax.bar(x-1.5*w, [summary["splits"][s]["hd_ssim"] for s in sps], w, label="HD canon", color="tab:blue")
    ax.bar(x-0.5*w, [base[s][0] for s in sps], w, label="HD baseline", color="tab:blue", alpha=0.4)
    ax.bar(x+0.5*w, [summary["splits"][s]["ld_ssim"] for s in sps], w, label="LD canon", color="tab:red")
    ax.bar(x+1.5*w, [base[s][1] for s in sps], w, label="LD baseline", color="tab:red", alpha=0.4)
    ax.set_xticks(x); ax.set_xticklabels(sps); ax.set_ylim(0,1); ax.set_ylabel("calibrated SSIM (full-image)")
    ax.set_title("Canonical re-stage HD/LD FBP vs per-patient baseline"); ax.legend(fontsize=8); ax.grid(alpha=0.3,axis="y")
    fig.tight_layout(); fig.savefig(out_dir / "summary.png", dpi=120, bbox_inches="tight"); plt.close(fig)
    print(f"[canon-cmp] wrote {out_dir}/summary.png + metrics + montages", flush=True)
    for s in sps:
        d = summary["splits"][s]
        print(f"  {s:5} HD {d['hd_ssim']:.4f} (base {base[s][0]}) | LD {d['ld_ssim']:.4f} (base {base[s][1]})", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
