#!/usr/bin/env python
"""Verify the re-staged `staged_canonical` with FBP(HD) and FBP(LD) vs GT —
NOT a solver. Reads the PACKED staged_canonical (truth + sino_{fulldose,lowdose}
+ per-slice ps), so a packing/slot misalignment shows up directly as FBP not
matching GT. Reconstructs each volume's CENTRAL slice on the production FBP path
(`mayo_proj_cache` per-ps projector; det-offset + truncation via the
`AGENT4CT_DATASET=mayo_ldct_2d` hard-wiring; Hann filter), evaluates calibrated
SSIM/RMSE vs GT over the FULL 512² view (no FOV mask, per user 2026-06-18 — and
matching the production eval). A FOV may be applied later, but it is the
DETECTOR-geometry measurement FOV `R = SOD·sin(atan(0.5·n_det·det_spacing/SDD))`
(a scanner property, ~237.5 mm, larger than the image inscribed circle), NOT a
slice-derived ReconstructionDiameter/PixelSpacing circle. Renders a
`GT | HD-FBP | LD-FBP | HD-GT | LD-GT` panel per patient.

val = L277; test = L014/L056/L058/L075/L123 (patient-ordered central-slice
indices, same convention as the showcase). Outputs docs/_restage_verify/.
"""
from __future__ import annotations
import json
import os
import sys
from pathlib import Path

import h5py
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
os.environ.setdefault("AGENT4CT_DATASET", "mayo_ldct_2d")     # det-offset + truncation
os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

from ddssl_ldct.staged_dataset import mayo_proj_cache          # noqa: E402
from ddssl_ldct.metrics import evaluate_calibrated             # noqa: E402

DATA = Path(os.environ.get("AGENT4CT_DATA", "/cluster/maier/Agent4CT/data")) \
    / "mayo_ldct" / "staged_canonical"
OUT = REPO / "docs" / "_restage_verify"
OUT.mkdir(parents=True, exist_ok=True)
DR = 0.05
DEV = "cuda" if torch.cuda.is_available() else "cpu"

# Central-slice index per test patient (patient-ordered test stack; counts
# L014 154 / L056 93 / L058 210 / L075 137 / L123 151 — same as the showcase).
TEST_IDX = [77, 200, 352, 525, 668]
TEST_PAT = ["L014", "L056", "L058", "L075", "L123"]


def _first_key(f, prefer):
    if prefer in f:
        return prefer
    cand = [k for k in f.keys() if k != "ps"]
    return cand[0] if cand else list(f.keys())[0]


def read_split(split):
    with h5py.File(DATA / f"{split}_truth.h5", "r") as f:
        truth = f[_first_key(f, "truth")][...]
        ps = f["ps"][...] if "ps" in f else None
    def sino(dose):
        with h5py.File(DATA / f"{split}_sino_{dose}.h5", "r") as f:
            return f[_first_key(f, "sino")][...]
    return truth, ps, sino("lowdose"), sino("fulldose")


def circ_mask(size, r, dev):
    c = torch.arange(size, device=dev, dtype=torch.float32) - (size - 1) / 2.0
    yy, xx = torch.meshgrid(c, c, indexing="ij")
    return ((xx * xx + yy * yy) <= r * r).float()


def fbp_one(sino_2d, ps_val):
    projs = mayo_proj_cache(np.array([ps_val], dtype=float), 2304, 736, DEV)
    s = torch.from_numpy(np.ascontiguousarray(sino_2d, dtype=np.float32))
    s = s.to(DEV)[None, None]                       # (1,1,A,D)
    return projs[round(float(ps_val), 5)].fbp(s).clamp(min=0.0)   # (1,1,H,W)


def cal(fbp_t, truth_t, mask):
    c = evaluate_calibrated(fbp_t, truth_t, baseline=None, display_min=0.0,
                            display_max=DR, fov=mask, bg_target="truth")
    return float(c["val_ssim"]), float(c["val_rmse"]), c["pred_cal"][0, 0].cpu().numpy()


def do(patient, split, idx, rows):
    truth, ps, ld, hd = rows[split]
    truth_2d = np.asarray(truth[idx], dtype=np.float32)
    ps_val = float(ps[idx])
    truth_t = torch.from_numpy(truth_2d).to(DEV)[None, None]
    # FULL 512² VIEW — no FOV mask (user, 2026-06-18). When we DO mask later, the
    # FOV is the DETECTOR-geometry measurement FOV — a scanner property,
    # independent of the slice:
    #     R_FOV = SOD * sin(atan(0.5 * n_det * det_spacing / SDD))
    # (~237.5 mm radius for the fitted Mayo geom; LARGER than the image inscribed
    # circle). It is NOT ReconstructionDiameter/PixelSpacing. `circ_mask` is kept
    # for that future use. fov=False here also matches the production eval.
    hd_ss, hd_rm, hd_cal = cal(fbp_one(hd[idx], ps_val), truth_t, False)
    ld_ss, ld_rm, ld_cal = cal(fbp_one(ld[idx], ps_val), truth_t, False)

    fig, ax = plt.subplots(1, 5, figsize=(20, 4.5))
    ax[0].imshow(truth_2d, cmap="gray", vmin=0, vmax=DR)
    ax[0].set_title(f"{patient} ({split}) GT  ps_eff={ps_val:.4f}  (full 512² view)")
    ax[1].imshow(hd_cal, cmap="gray", vmin=0, vmax=DR)
    ax[1].set_title(f"HD-FBP\nSSIM {hd_ss:.4f}  RMSE {hd_rm:.2e}")
    ax[2].imshow(ld_cal, cmap="gray", vmin=0, vmax=DR)
    ax[2].set_title(f"LD-FBP\nSSIM {ld_ss:.4f}  RMSE {ld_rm:.2e}")
    dlim = DR * 0.5
    ax[3].imshow(hd_cal - truth_2d, cmap="bwr", vmin=-dlim, vmax=dlim)
    ax[3].set_title("HD − GT")
    ax[4].imshow(ld_cal - truth_2d, cmap="bwr", vmin=-dlim, vmax=dlim)
    ax[4].set_title("LD − GT")
    for a in ax:
        a.axis("off")
    fig.tight_layout()
    fig.savefig(OUT / f"{patient}_fbp_vs_gt.png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"[verify] {patient} ({split}) idx={idx} ps_eff={ps_val:.4f}  FULL-VIEW  "
          f"HD {hd_ss:.4f}/{hd_rm:.2e}  LD {ld_ss:.4f}/{ld_rm:.2e}", flush=True)
    return {"patient": patient, "split": split, "idx": idx, "ps": ps_val,
            "fov": "none (full 512x512)",
            "hd_ssim": hd_ss, "hd_rmse": hd_rm, "ld_ssim": ld_ss, "ld_rmse": ld_rm}


def main():
    rows = {"val": read_split("val"), "test": read_split("test")}
    res = [do("L277", "val", rows["val"][0].shape[0] // 2, rows)]
    for pat, idx in zip(TEST_PAT, TEST_IDX):
        res.append(do(pat, "test", idx, rows))
    (OUT / "restage_verify_metrics.json").write_text(json.dumps(res, indent=1))
    hd = float(np.mean([r["hd_ssim"] for r in res]))
    ld = float(np.mean([r["ld_ssim"] for r in res]))
    print(f"[verify] MEAN (full 512² view)  HD-FBP SSIM {hd:.4f}  "
          f"LD-FBP SSIM {ld:.4f}", flush=True)


if __name__ == "__main__":
    main()
