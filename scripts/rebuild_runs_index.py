"""Standalone, torch-free regenerator of docs/runs/runs-index.json.

Mirrors `learned_solver_search_agent.update_index()` (which is gated behind a
torch import, so it can't run on the laptop). Run this LOCALLY after rsyncing
the cluster's `docs/runs/<slug>/` dirs back, before committing for GitHub Pages
— the dashboard reads `runs-index.json` to list runs.

Difference vs the original: the per-run `challenge` is derived from the slug
prefix (`mayo-ldct-*` -> `mayo_ldct`, `breast-ct-*` -> `breast_ct`, …) rather
than the manifest field, because `claude_agentic_one_iter.py` hardcodes
`challenge="dl_sparse_view"` (it was written for the breast-ct rollout). The
dashboard groups by challenge, so deriving from the slug keeps Mayo runs out of
the wrong bucket without post-hoc patching every manifest.
"""
from __future__ import annotations
import json
import math
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DOCS_RUNS = REPO / "docs" / "runs"

_PREFIX_TO_CHALLENGE = [
    ("mayo-ldct", "mayo_ldct"),
    ("breast-ct", "breast_ct"),
    ("demo-dl", "demo_dl"),
    ("dl-sparse-view", "dl_sparse_view"),
    ("dl-spectral", "dl_spectral"),
    ("ct-mar", "ct_mar"),
    ("truect", "truect"),
]


def challenge_from_slug(slug: str, fallback: str | None) -> str | None:
    for prefix, ch in _PREFIX_TO_CHALLENGE:
        if slug.startswith(prefix):
            return ch
    return fallback


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def update_index() -> int:
    runs = []
    for run_dir in sorted(DOCS_RUNS.iterdir()):
        if not run_dir.is_dir():
            continue
        m_path = run_dir / "manifest.json"
        if not m_path.exists():
            continue
        manifest = json.loads(m_path.read_text())
        results = run_dir / "results.tsv"
        n_iter = 0
        best_score = None
        best_hr = None
        if results.exists():
            rows = [r for r in results.read_text().splitlines()[1:] if r]
            n_iter = len(rows)
            for r in rows:
                cells = r.split("\t")
                try:
                    v = float(cells[2])
                    if math.isfinite(v) and (best_score is None or v > best_score):
                        best_score = v
                    h = float(cells[3])
                    if math.isfinite(h) and (best_hr is None or h > best_hr):
                        best_hr = h
                except (ValueError, IndexError):
                    pass
        parts = run_dir.name.split("-")
        m_short = parts[-2] + "-" + parts[-1] if len(parts) >= 2 else run_dir.name
        runs.append({
            "slug": run_dir.name, "short_id": m_short,
            "challenge": challenge_from_slug(run_dir.name, manifest.get("challenge")),
            "started": manifest.get("started"),
            "status": manifest.get("status", "running"),
            "n_iterations": n_iter,
            "best_score": best_score,
            "best_headroom": best_hr,
            "agent": manifest.get("agent"),
            "model": manifest.get("model"),
        })
    (DOCS_RUNS / "runs-index.json").write_text(json.dumps(
        {"schema_version": 1, "updated": utc_now_iso(), "runs": runs},
        indent=2, allow_nan=False))
    print(f"[index] wrote runs-index.json with {len(runs)} runs")
    return len(runs)


if __name__ == "__main__":
    update_index()
