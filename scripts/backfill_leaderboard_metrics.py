"""Add PSNR (dB) / RMSE / time (s) columns to the hand-curated leaderboard
tables in docs/leaderboards/{breast_ct,demo_dl}.md WITHOUT disturbing the
curated rows (Variant strings, rank annotations, prose).

For every markdown table that has an exact `hr` column, this inserts three new
columns immediately after `hr`, sourced per-row from the run+iter the row
already links to:
  - PSNR (dB) = observation.json `val_psnr`
  - RMSE      = observation.json `val_rmse`
  - time (s)  = observation.json `elapsed_s` (the per-iter wall the agentic
                runner records)
Rows whose link can't be resolved (deprioritised / SLURM-job-only rows, or a
run dir not present locally) get `—` for the three cells. Idempotent: a table
that already has a `PSNR (dB)` column is left untouched.

Usage:  python3 scripts/backfill_leaderboard_metrics.py
"""
from __future__ import annotations
import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RUNS = REPO / "docs" / "runs"
BOARDS = [REPO / "docs" / "leaderboards" / "breast_ct.md",
          REPO / "docs" / "leaderboards" / "demo_dl.md"]

NEW_HEADERS = ["PSNR (dB)", "RMSE", "time (s)"]
NEW_SEPS = ["---:", "---:", "---:"]


def cells_of(line: str) -> list[str]:
    return [c.strip() for c in line.strip().strip("|").split("|")]


def render(cells: list[str]) -> str:
    return "| " + " | ".join(cells) + " |"


def is_sep(cells: list[str]) -> bool:
    return bool(cells) and all(re.fullmatch(r":?-{2,}:?", c) for c in cells)


def metrics_for_row(line: str):
    """Resolve (psnr, rmse, time) strings from the run+iter the row links to."""
    mslug = re.search(r"runs/([^/)\s]+)/", line)
    miter = re.search(r"iter-(\d+)", line)
    if not (mslug and miter):
        return "—", "—", "—"
    obs = (RUNS / mslug.group(1) / "iterations"
           / f"iter-{int(miter.group(1)):04d}" / "observation.json")
    if not obs.exists():
        return "—", "—", "—"
    try:
        o = json.loads(obs.read_text())
    except (json.JSONDecodeError, OSError):
        return "—", "—", "—"
    p, r, t = o.get("val_psnr"), o.get("val_rmse"), o.get("elapsed_s")
    return (f"{p:.2f}" if isinstance(p, (int, float)) else "—",
            f"{r:.2e}" if isinstance(r, (int, float)) else "—",
            f"{t:.0f}" if isinstance(t, (int, float)) else "—")


def process(md: str):
    out, lines = [], md.splitlines()
    hr_idx = None            # active table's hr column index, or None
    expect_sep = False
    n_rows = n_resolved = 0
    for line in lines:
        s = line.strip()
        if not (s.startswith("|") and s.endswith("|")):
            hr_idx, expect_sep = None, False
            out.append(line)
            continue
        cells = cells_of(line)
        # header row of a fresh table?
        if hr_idx is None:
            if "hr" in cells and ("Solver" in cells or "SSIM" in cells):
                if "PSNR (dB)" in cells:          # already migrated
                    out.append(line)
                    continue
                hr_idx = cells.index("hr")
                cells[hr_idx + 1:hr_idx + 1] = NEW_HEADERS
                expect_sep = True
                out.append(render(cells))
            else:
                out.append(line)                  # a table we don't touch
            continue
        # inside a target table
        if expect_sep and is_sep(cells):
            cells[hr_idx + 1:hr_idx + 1] = NEW_SEPS
            expect_sep = False
            out.append(render(cells))
            continue
        # data row
        n_rows += 1
        psnr, rmse, tsec = metrics_for_row(line)
        if psnr != "—":
            n_resolved += 1
        cells[hr_idx + 1:hr_idx + 1] = [psnr, rmse, tsec]
        out.append(render(cells))
    return "\n".join(out) + ("\n" if md.endswith("\n") else ""), n_rows, n_resolved


def main() -> int:
    for board in BOARDS:
        md = board.read_text()
        new, n_rows, n_resolved = process(md)
        board.write_text(new)
        print(f"[backfill] {board.name}: {n_rows} data rows, "
              f"{n_resolved} resolved from observation.json, "
              f"{n_rows - n_resolved} left as '—'")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
