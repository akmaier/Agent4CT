"""score_mayo_from_recons.py — score a TRAIN-ONCE / EVAL-ALL Mayo run from its
persisted raw reconstruction, with ZERO retraining.

A solver run launched with

    AGENT4CT_DATASET=mayo_ldct_2d
    AGENT4CT_EVAL_PATIENT=all          # loader: val(L277) ++ 5 test patients = N slices
    AGENT4CT_SAVE_RECON=<dir>          # metrics: dump raw pred/truth/baseline -> recon_raw.npz

trains ONCE on the train patients and reconstructs the WHOLE held-out set
(val L277 then the 5 Wagner test patients, concatenated). The raw recon is
persisted as <dir>/recon_raw.npz. This script reads that npz and, for EACH
held-out patient INDEPENDENTLY (per-patient two-point calibration, exactly the
ddssl_ldct.metrics convention), recomputes the FIXED metric (no upper clamp;
SSIM/PSNR data_range = truth's range) and aggregates the 5 TEST patients into
mean ± std (ddof=1) — the same final.json schema score_mayo_testset.py emits.

Because it reads a saved recon, re-scoring under any future metric is a parse,
not a re-run. It also writes a per-patient comparison.png (auditability: every
reported number has a supporting image).

Patient boundaries (Wagner test order, slices within the TEST block; the VAL
block L277 is the first n_val = N-745 slices):
    L014 [0:154]  L056 [154:247]  L058 [247:457]  L075 [457:594]  L123 [594:745]

Usage (run on the CLUSTER, where the recon + data live):
    AGENT4CT_DATASET=mayo_ldct_2d \\
    python scripts/score_mayo_from_recons.py \\
        --recon  runs/<slug>-evalall/recon_raw.npz \\
        --slug   <run-id> \\
        --outdir runs/<slug>-evalall            # final.json + figures land here
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

# The metric hard-wires Mayo calibration (bg_target="truth") + the Mayo FOV mask
# on this env, so it MUST be set before importing/using evaluate_calibrated.
os.environ.setdefault("AGENT4CT_DATASET", "mayo_ldct_2d")

import numpy as np  # noqa: E402
import torch        # noqa: E402

# Test-block sub-ranges (Wagner order), identical to staged_dataset.MAYO_TEST_PATIENT_RANGES.
TEST_RANGES = {
    "L014": (0, 154), "L056": (154, 247), "L058": (247, 457),
    "L075": (457, 594), "L123": (594, 745),
}
TEST_PATIENTS = ["L014", "L056", "L058", "L075", "L123"]
N_TEST = 745
FINAL_SCHEMA = "mayo_testset_final_v2_from_recons"
DISPLAY_MIN, DISPLAY_MAX = 0.0, 0.09   # figure window only (scoring ignores it)


def _mean_std(vals):
    finite = [float(v) for v in vals
              if isinstance(v, (int, float)) and v == v and abs(v) != float("inf")]
    if not finite:
        return None, None
    m = sum(finite) / len(finite)
    if len(finite) < 2:
        return m, 0.0
    var = sum((x - m) ** 2 for x in finite) / (len(finite) - 1)
    return m, var ** 0.5


def _score_block(pred, truth, base, *, label, outdir, device, make_fig=True):
    """Per-patient: calibrate+score the FIXED metric, render a comparison figure."""
    from ddssl_ldct.metrics import evaluate_calibrated, make_4panel_comparison
    p = torch.from_numpy(np.ascontiguousarray(pred)).to(device=device, dtype=torch.float32)
    t = torch.from_numpy(np.ascontiguousarray(truth)).to(device=device, dtype=torch.float32)
    b = (torch.from_numpy(np.ascontiguousarray(base)).to(device=device, dtype=torch.float32)
         if base is not None else None)
    if p.dim() == 3:
        p = p.unsqueeze(1); t = t.unsqueeze(1)
        if b is not None:
            b = b.unsqueeze(1)
    m = evaluate_calibrated(p, t, baseline=b,
                            display_min=DISPLAY_MIN, display_max=DISPLAY_MAX, fov=True)
    rec = {
        "headroom": m.get("headroom"),
        "ssim": m["val_ssim"], "psnr": m["val_psnr"], "rmse": m["val_rmse"],
        "ssim_std": m["val_ssim_std"], "psnr_std": m["val_psnr_std"],
        "rmse_std": m["val_rmse_std"],
        "baseline_rmse": m.get("baseline_rmse"), "baseline_ssim": m.get("baseline_ssim"),
        "n_slices": int(m["val_n_slices"]),
    }
    if make_fig and b is not None:
        fig_dir = Path(outdir) / label
        fig_dir.mkdir(parents=True, exist_ok=True)
        try:
            make_4panel_comparison(
                truth=t, fbp=m["baseline_cal"], recon=m["pred_cal"],
                out_path=fig_dir / "comparison.png",
                display_min=DISPLAY_MIN, display_max=DISPLAY_MAX,
                solver_label=f"{label}", headroom=m.get("headroom"))
            (fig_dir / "result.json").write_text(json.dumps(rec, indent=2))
        except Exception as e:
            print(f"[from-recons] WARN: figure for {label} failed: {e}", flush=True)
    print(f"[from-recons] {label}: hr={rec['headroom']} ssim={rec['ssim']:.4f} "
          f"psnr={rec['psnr']:.2f} rmse={rec['rmse']:.5f} (n={rec['n_slices']})", flush=True)
    return rec


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--recon", required=True, help="path to recon_raw.npz (train-once eval-all run)")
    ap.add_argument("--slug", required=True, help="run-id slug (for final.json metadata)")
    ap.add_argument("--outdir", required=True, help="where final.json + per-patient figures are written")
    ap.add_argument("--solver-key", default=None)
    ap.add_argument("--no-fig", action="store_true", help="skip per-patient figures (numbers only)")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    z = np.load(args.recon)
    pred, truth = z["pred"], z["truth"]
    base = z["baseline"] if "baseline" in z.files else None
    N = pred.shape[0]
    n_val = N - N_TEST
    if n_val < 1:
        print(f"[from-recons] ERROR: pred has {N} slices < {N_TEST} test slices; "
              f"not a train-once eval-all recon?", flush=True)
        return 2
    print(f"[from-recons] {args.recon}: N={N} -> n_val(L277)={n_val} + test={N_TEST}; "
          f"device={device}", flush=True)
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    def _sl(lo, hi):
        b = base[lo:hi] if base is not None else None
        return pred[lo:hi], truth[lo:hi], b

    # VAL (L277) — first n_val slices.
    vp, vt, vb = _sl(0, n_val)
    val_rec = _score_block(vp, vt, vb, label="L277_val", outdir=outdir,
                           device=device, make_fig=not args.no_fig)

    # 5 TEST patients (offset by n_val).
    per_patient = {}
    for pat in TEST_PATIENTS:
        lo, hi = TEST_RANGES[pat]
        pp, tp, bp = _sl(n_val + lo, n_val + hi)
        per_patient[pat] = _score_block(pp, tp, bp, label=pat, outdir=outdir,
                                        device=device, make_fig=not args.no_fig)

    def col(metric):
        return [per_patient[p].get(metric) for p in TEST_PATIENTS]
    hr_m, hr_s = _mean_std(col("headroom"))
    ss_m, ss_s = _mean_std(col("ssim"))
    ps_m, ps_s = _mean_std(col("psnr"))
    rm_m, rm_s = _mean_std(col("rmse"))

    final = {
        "schema": FINAL_SCHEMA,
        "run_id": args.slug,
        "solver_key": args.solver_key,
        "method": "train-once eval-all; per-patient two-point calibration; "
                  "FIXED metric (no upper clamp, data_range=truth-range); "
                  "scored from persisted recon_raw.npz (zero retraining)",
        "recon_src": str(args.recon),
        "n_val_slices": n_val,
        "test_n_patients": len(TEST_PATIENTS),
        "val_L277": val_rec,
        "patients": per_patient,
        "test_hr_mean": hr_m, "test_hr_std": hr_s,
        "test_ssim_mean": ss_m, "test_ssim_std": ss_s,
        "test_psnr_mean": ps_m, "test_psnr_std": ps_s,
        "test_rmse_mean": rm_m, "test_rmse_std": rm_s,
    }
    (outdir / "final.json").write_text(json.dumps(final, indent=2))
    print(f"[from-recons] wrote {outdir/'final.json'}", flush=True)
    print(f"[from-recons] TEST hr = {hr_m:.4f} ± {hr_s:.4f} (n=5)  |  "
          f"val L277 hr = {val_rec['headroom']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
