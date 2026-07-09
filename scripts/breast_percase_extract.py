"""breast_percase_extract.py — dump PER-CASE test metrics for the top-N breast
solvers (their best-by-test-hr iter) so a paired significance analysis (like Mayo)
can run. Reads each best iter's recon_raw.npz (pred/truth/baseline), computes the
per-case calibrated, FOV-masked, frozen-metric hr/ssim/psnr/rmse over the 200 test
cases with the BATCH-wide SSIM/PSNR data_range (matches the live board), and writes
a compact JSON {solver: {iter, hr:[...], ssim:[...], psnr:[...], rmse:[...]}}.

Run on the cluster login node (per-case loop is memory-light; no full-batch SSIM).
  python scripts/breast_percase_extract.py --top 10 --out docs/runs/breast_percase_top.json
"""
from __future__ import annotations
import argparse, json, os, sys, gc
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
RUNS = Path(os.environ.get("AGENT4CT_RUNS", str(REPO / "runs")))
SEL = REPO / "docs" / "runs" / "breast_testsweep_selection.json"


def recon_path(run_id: str, it: int) -> Path:
    slug = f"{run_id}-itertest__iter-{it:04d}-breasttest"
    return RUNS / slug / "test" / "recon_raw.npz"


def per_case(npz: Path):
    import numpy as np, torch
    from ddssl_ldct.metrics import evaluate_calibrated, ssim as ssim_fn, psnr as psnr_fn
    d = np.load(str(npz))
    key = "pred" if "pred" in d.files else "recon"
    def nchw(a):
        t = torch.from_numpy(np.ascontiguousarray(a)).float()
        return t[:, None] if t.dim() == 3 else (t[None, None] if t.dim() == 2 else t)
    pred, truth = nchw(d[key]), nchw(d["truth"])
    base = nchw(d["baseline"]) if "baseline" in d.files else None
    n = pred.shape[0]
    batch_dr = max(float(truth.max() - truth.min()), 1e-6)
    hr, ss, ps, rm = [], [], [], []
    for i in range(n):
        b_i = base[i:i + 1] if base is not None else None
        eb = os.environ.pop("AGENT4CT_SAVE_RECON", None)
        try:
            m = evaluate_calibrated(pred[i:i + 1], truth[i:i + 1], baseline=b_i, display_min=0.0, display_max=0.5)
        finally:
            if eb is not None:
                os.environ["AGENT4CT_SAVE_RECON"] = eb
        pcal = m["pred_cal"]; fm = m.get("fov_mask"); tc = truth[i:i + 1].to(pcal.device)
        if fm is not None:
            pcal = pcal * fm; tc = tc * fm
        ss.append(float(ssim_fn(pcal, tc, data_range=batch_dr).cpu()))
        ps.append(float(psnr_fn(pcal, tc, data_range=batch_dr).cpu()))
        rm.append(float(m["val_rmse"]))
        if b_i is not None and m.get("baseline_rmse"):
            hr.append(max(0.0, 1.0 - m["val_rmse"] / max(m["baseline_rmse"], 1e-12)))
        else:
            hr.append(m.get("headroom"))
    del d, pred, truth, base; gc.collect()
    return {"hr": hr, "ssim": ss, "psnr": ps, "rmse": rm}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=10)
    ap.add_argument("--out", default=str(REPO / "docs/runs/breast_percase_top.json"))
    a = ap.parse_args()
    sel = json.loads(SEL.read_text())
    sel = [s for s in sel if s.get("test_best_hr_mean") is not None]
    sel.sort(key=lambda s: -s["test_best_hr_mean"])
    out = {}
    for s in sel[:a.top]:
        it = s["test_best_iter"]; npz = recon_path(s["run_id"], it)
        if not npz.exists():
            print(f"  MISSING recon: {s['solver']} iter-{it} -> {npz}", flush=True); continue
        print(f"  {s['solver']} iter-{it} ...", flush=True)
        pc = per_case(npz)
        out[s["solver"]] = {"iter": it, "hr_mean": s["test_best_hr_mean"], **pc}
        print(f"    n={len(pc['hr'])} hr_mean={sum(pc['hr'])/len(pc['hr']):.4f}", flush=True)
    Path(a.out).write_text(json.dumps(out))
    print(f"wrote {a.out}  ({len(out)} solvers)", flush=True)


if __name__ == "__main__":
    main()
