"""breast_board_rank_corr.py — Spearman rank correlation across the three breast
boards, for the paper's "does the clean ranking survive / return?" claims.

Boards (docs/runs/index/*.json leaderboard rows, ranked by held-out test hr):
  - breast_ct               noiseless
  - breast_ct_noise         noisy (I0=1e5), NO retraining (frozen clean weights)
  - breast_ct_noise_retrain noisy (I0=1e5), weights RETRAINED on the noisy train split
    (per-solver: docs/runs/<run>-itertest-noise100000-retrain/iter-*/final.json;
     non-trainable solvers carry over their no-retrain noisy score)

Reports Spearman rho over the solvers RANKED on the noiseless board (rank != None)
that also have a finite hr on the compared board — the honest "did the clean
leaderboard predict this board" set. Prints rho and rho^2 (variance explained).

Usage:  python scripts/breast_board_rank_corr.py
"""
from __future__ import annotations
import json, glob, re
from pathlib import Path
from scipy.stats import spearmanr

REPO = Path(__file__).resolve().parents[1]
IDX = REPO / "docs" / "runs" / "index"
RUNS = REPO / "docs" / "runs"
NOISE_I0 = 100000


def board(name):
    rows = json.loads((IDX / f"{name}.json").read_text())["leaderboard"]["rows"]
    return {r["solver_key"]: r for r in rows}


def retrain_hr():
    """Per-solver retrained-noisy hr, keyed by dashed solver key; carry over the
    no-retrain noisy hr for solvers that were not retrained."""
    nz = board("breast_ct_noise")
    rt = {}
    for p in glob.glob(str(RUNS / f"*-itertest-noise{NOISE_I0}-retrain" / "iter-*" / "final.json")):
        d = json.loads(Path(p).read_text())
        key = re.sub(r"breast-ct-claude-agentic-|-search.*", "", Path(p).parts[-3])
        rt[key] = d.get("test_hr_mean")
    hr = {}
    for k, r in nz.items():
        hr[k] = rt.get(k, r.get("test_hr_mean"))
    for k, v in rt.items():
        hr.setdefault(k, v)
    return hr


def _hr(b, k):
    r = b.get(k)
    return r.get("test_hr_mean") if r else None


def corr(nl, other_hr, label):
    xs, ys = [], []
    for k, r in nl.items():
        if r.get("rank") is None:          # noiseless-ranked solvers only
            continue
        a = r.get("test_hr_mean"); b = other_hr(k)
        if a is None or b is None:
            continue
        xs.append(a); ys.append(b)
    rho, p = spearmanr(xs, ys)
    print(f"  noiseless vs {label:22s}: rho={rho:+.3f}  rho^2={rho*rho:.3f} "
          f"({rho*rho*100:.0f}% of variance)  n={len(xs)}  p={p:.3g}")
    return rho


def main():
    nl = board("breast_ct"); nz = board("breast_ct_noise"); rt = retrain_hr()
    print("Spearman rank correlation vs the noiseless breast board:")
    corr(nl, lambda k: _hr(nz, k), "noisy (no retrain)")
    corr(nl, lambda k: rt.get(k), "noisy (retrained)")


if __name__ == "__main__":
    main()
