#!/usr/bin/env python
"""Diverse TEST-set showcase figures for the Mayo leaderboard.

For each leaderboard solver, re-run its BEST agentic config with
``AGENT4CT_SHOWCASE=1`` so the figure montage shows **one central slice from
each of the 5 held-out Wagner test patients** (L014/L056/L058/L075/L123) instead
of 4 adjacent slices of the single val patient L277. The model still TRAINS on
the train patients — only the 5 figure slices come from test, so the agentic
SEARCH metric (val L277) is never tuned on test; these figures are
presentation-only. Writes ``docs/runs/<search-slug>/test_showcase.png``
co-located with each search run (so the existing pinned rsync + publish picks it
up; no extra dashboard run dirs).

Driver mode (no ``SOLVER`` env): iterate all leaderboard solvers, one subprocess
per solver (fresh CUDA context each). Worker mode (``SOLVER`` + ``SEARCH_SLUG``
env): run exactly one solver.

  # one job loops every solver:
  python scripts/make_test_showcase.py
"""
from __future__ import annotations
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
RUNS = REPO / "docs" / "runs"
RUNID = "search-20260619-01"   # pinned rebuild run-id (never widen). The
# 2026-06-14 rebuild was purged in the 2026-06-19 reset; the live campaign is
# search-20260619-01.

# (solver_key, slug_dash). slug_dash defaults to solver_key.replace("_","-").
# ALL 19 Mayo solvers get the 6-central-patient valtest montage. The 9
# inference/per-image solvers were previously excluded; their main() reconstructs
# the same 6 valtest scenes (L277-central + 5 test centrals) via the dataset-level
# AGENT4CT_SHOWCASE=valtest redirect. Two solver/slug mismatches need an explicit
# slug_dash override: ram (key ram_zeroshot, slug ...-ram-...) and the two
# diffusion solvers (key diffusion_recon_*, slug ...-diff-recon-*).
KEYS = [
    # 10 trainers
    ("uswin",), ("itnet_v3",), ("itnet",), ("itnet_v2",),
    ("dual_domain_supervised",), ("dual_domain_bilateral_supervised",),
    ("learned_primal_dual",), ("hammernik_2017",), ("hammernik_vn",),
    ("wu_2015_trainable",),
    # 9 inference / per-image
    ("naf",), ("ram_zeroshot", "ram"), ("r2gaussian",),
    ("tv_iterative",), ("tv_iterative_supervised",),
    ("dual_domain_n2i",), ("dual_domain_bilateral_n2i",),
    ("diffusion_recon_dcstep_constrained_mayo_v4", "diff-recon-dcstep-constrained-mayo-v4"),
    ("diffusion_recon_dcstep_unconstrained_mayo_v4", "diff-recon-dcstep-unconstrained-mayo-v4"),
]


def last_iter_cfg(slug: str):
    """LAST iter (max iter index) for a search run + its cfg_full from
    observation.json. The dashboard shows each solver's FINAL-iteration result
    (iter-20 mandate), not the metric-best — so the test showcase must use the
    same iter the val figure does."""
    tsv = RUNS / slug / "results.tsv"
    if not tsv.exists():
        return None
    last = None
    for ln in tsv.read_text().splitlines()[1:]:
        c = ln.split("\t")
        try:
            it = int(c[0])
        except (ValueError, IndexError):
            continue
        if last is None or it > last:
            last = it
    if last is None:
        return None
    obs = RUNS / slug / "iterations" / f"iter-{last:04d}" / "observation.json"
    if not obs.exists():
        return None
    return last, (json.loads(obs.read_text()).get("cfg_full") or {})


def run_worker(solver: str, slug: str) -> None:
    from scripts.claude_agentic_one_iter import SOLVER_MAP
    if solver not in SOLVER_MAP:
        print(f"[skip] {solver}: not in SOLVER_MAP", flush=True)
        return
    bc = last_iter_cfg(slug)
    if not bc:
        print(f"[skip] {solver}: no last-iter cfg in {slug}", flush=True)
        return
    it, cfg = bc
    os.environ.update(AGENT4CT_DATASET="mayo_ldct_2d",
                      AGENT4CT_SHOWCASE="valtest", AGENT4CT_FIG_NSHOW="6")
    solver_file, _ = SOLVER_MAP[solver]
    spec = importlib.util.spec_from_file_location("scs_" + solver, REPO / solver_file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    tmp = Path(tempfile.mkdtemp(prefix=f"showcase_{solver}_"))
    try:
        res = mod.main(tmp, cfg)
        src = tmp / "comparison.png"
        if src.exists():
            dst = RUNS / slug / "valtest_showcase.png"
            shutil.copy(src, dst)
            ss = res.get("val_ssim", float("nan")) if isinstance(res, dict) else float("nan")
            print(f"[ok] {solver}: last iter-{it}  valtest6 (L277+5test) SSIM={ss:.4f}  -> "
                  f"{dst.relative_to(REPO)}", flush=True)
        else:
            print(f"[nofig] {solver}: main() wrote no comparison.png", flush=True)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main() -> int:
    solver = os.environ.get("SOLVER")
    if solver:  # worker
        run_worker(solver, os.environ["SEARCH_SLUG"])
        return 0
    # driver: one subprocess per solver so a crash/OOM in one doesn't abort the
    # rest and CUDA state is fresh each time.
    for entry in KEYS:
        k = entry[0]
        dash = entry[1] if len(entry) > 1 else k.replace("_", "-")
        slug = f"mayo-ldct-claude-agentic-{dash}-{RUNID}"
        if not (RUNS / slug).exists():
            print(f"[skip] {k}: no run dir {slug}", flush=True)
            continue
        if (RUNS / slug / "valtest_showcase.png").exists() and not os.environ.get("SHOWCASE_FORCE"):
            print(f"[skip] {k}: valtest_showcase.png exists (SHOWCASE_FORCE=1 to regen)", flush=True)
            continue
        print(f"=== showcase {k} ===", flush=True)
        env = {**os.environ, "SOLVER": k, "SEARCH_SLUG": slug}
        subprocess.run([sys.executable, __file__], env=env)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
