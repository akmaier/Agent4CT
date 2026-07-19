"""build_breast_retrain_board.py — assemble docs/runs/index/breast_ct_noise_retrain.json.

The retrained-noisy board = the no-retrain BreastCT-Noise board with the TRAINABLE
solvers' test metrics REPLACED by their retrained-on-noise scores (each solver's
clean-best config, retrained on the Poisson-noised train split, scored on the noisy
test set — docs/runs/<run>-itertest-noise100000-retrain/iter-*/final.json). Classical /
per-scene / zero-shot solvers are carried over unchanged (they have no supervised
weights). Rows are then re-ranked by test hr (test_ssim tiebreak), exactly like the
other test-ranked boards (see registry_lib.rank_fields).

This is a surgical, paper-facing builder: it derives the new board from the two boards
the paper already reads and does NOT rebuild or perturb any other index file.

Usage:  python scripts/build_breast_retrain_board.py
"""
from __future__ import annotations
import json, glob, re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
IDX = REPO / "docs" / "runs" / "index"
RUNS = REPO / "docs" / "runs"
NOISE_I0 = 100000
_TEST_KEYS = ("test_hr_mean", "test_hr_std", "test_ssim_mean", "test_ssim_std",
              "test_psnr_mean", "test_psnr_std", "test_rmse_mean", "test_rmse_std")


def load_retrain():
    """dashed solver key -> retrain final.json dict."""
    out = {}
    for p in glob.glob(str(RUNS / f"*-itertest-noise{NOISE_I0}-retrain" / "iter-*" / "final.json")):
        d = json.loads(Path(p).read_text())
        key = re.sub(r"breast-ct-claude-agentic-|-search.*", "", Path(p).parts[-3])
        out[key] = d
    return out


def finite(x):
    return isinstance(x, (int, float)) and x == x and abs(x) != float("inf")


def main() -> int:
    base = json.loads((IDX / "breast_ct_noise.json").read_text())
    rt = load_retrain()
    rows = [dict(r) for r in base["leaderboard"]["rows"]]

    n_retrained = 0
    for r in rows:
        k = r["solver_key"]
        r["retrained"] = k in rt
        if k in rt:
            d = rt[k]
            for kk in _TEST_KEYS:
                r[kk] = d.get(kk)
            if finite(d.get("elapsed_s")):
                r["elapsed_s"] = d.get("elapsed_s")
            n_retrained += 1

    # Re-rank by test hr (test_ssim tiebreak); exclude hr<=0 / non-finite / DNF.
    for r in rows:
        hr = r.get("test_hr_mean")
        rm = hr if finite(hr) else None
        if rm is None:
            r["excluded_reason"] = "non-finite" if hr is not None else "pending-test"
        elif rm <= 0:
            r["excluded_reason"] = "hr<=0"
        else:
            r["excluded_reason"] = None
        r["rank_metric"] = rm
        r["rank_tiebreak"] = r.get("test_ssim_mean")

    ranked = sorted([r for r in rows if r["excluded_reason"] is None],
                    key=lambda r: (r["rank_metric"], r.get("rank_tiebreak") or -1),
                    reverse=True)
    for i, r in enumerate(ranked, 1):
        r["rank"] = i
    for r in rows:
        if r["excluded_reason"] is not None:
            r["rank"] = None

    out = dict(base)
    out["challenge"] = "breast_ct_noise_retrain"
    out["label"] = "BreastCT-Noise-Retrained"
    out["updated"] = base.get("updated")
    out["note"] = ("Retrained-noisy board: each trainable solver's clean-best config "
                   "retrained on the Poisson-noised (I0=1e5) train split, scored on the "
                   "noisy test set (n=200); non-trainable solvers carried over from the "
                   "no-retrain BreastCT-Noise board.")
    out["leaderboard"] = dict(base["leaderboard"])
    out["leaderboard"]["rows"] = rows
    (IDX / "breast_ct_noise_retrain.json").write_text(json.dumps(out, indent=1))

    nranked = len(ranked)
    print(f"wrote breast_ct_noise_retrain.json: {len(rows)} solvers "
          f"({n_retrained} retrained, {len(rows)-n_retrained} carried over), "
          f"{nranked} ranked.")
    print("top 6:")
    for r in ranked[:6]:
        print(f"  {r['rank']:>2} {r['solver_key']:28s} hr={r['test_hr_mean']:.4f} "
              f"ssim={r['test_ssim_mean']:.4f} {'[retrained]' if r['retrained'] else '[carry]'}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
