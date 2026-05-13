"""Run / iteration writer for the Agent4CT autoresearch loop.

The harness owns the filesystem conventions described in
`docs/agents.md`:

  docs/runs/
    runs-index.json                   # auto-maintained list of all runs
    observations.jsonl                # cross-run append-only scratch pad
    <run-slug>/
      manifest.json                   # one per run
      results.tsv                     # iteration journal (one row per iter)
      stages.tsv                      # stage-check journal (every 30 iter)
      iterations/
        iter-0001/
          observation.json
          comparison.png
          solver.py.txt
          stdout.log
        iter-0002/
        ...

This module is intentionally side-effect-only — it does not call out to
Slurm, the cluster, or git. Drive it from a thin wrapper (see
`scripts/agent4ct_record.py`) which is what the LLM agent actually invokes
after each 5-minute Slurm job completes.

Versioning:
- Run slug:  ``<challenge-slug>-YYYYMMDD-NN`` (NN zero-padded, per-day seq).
- Iteration: ``iter-NNNN`` (NNNN zero-padded).
"""
from __future__ import annotations
import json
import shutil
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional


# --- conventions ---------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[1]
DOCS_RUNS = REPO_ROOT / "docs" / "runs"

RESULTS_HEADER = (
    "iter\tcommit\tval_score\theadroom\tstatus\tchange_class\trationale\n"
)
STAGES_HEADER = (
    "iter\tstage_val_score\tstage_headroom\tgap\tverdict\tnotes\n"
)


# --- helpers -------------------------------------------------------------

def utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def today_yyyymmdd() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d")


def _ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def _atomic_write_text(path: Path, content: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(content)
    tmp.replace(path)


def next_sequence_for_today(prefix: str, root: Path = DOCS_RUNS) -> int:
    """Return the next NN for ``<prefix>-YYYYMMDD-NN``."""
    if not root.exists():
        return 1
    today = today_yyyymmdd()
    needle = f"{prefix}-{today}-"
    used = []
    for d in root.iterdir():
        if not d.is_dir() or not d.name.startswith(needle):
            continue
        tail = d.name[len(needle):]
        if tail.isdigit():
            used.append(int(tail))
    return (max(used) + 1) if used else 1


def make_slug(prefix: str, root: Path = DOCS_RUNS) -> str:
    seq = next_sequence_for_today(prefix, root=root)
    return f"{prefix}-{today_yyyymmdd()}-{seq:02d}"


# --- data classes --------------------------------------------------------

@dataclass
class Observation:
    ts: str
    run_id: str
    iter: int
    challenge: str
    change_class: str
    rationale: str
    val_score: Optional[float] = None
    headroom: Optional[float] = None
    delta_vs_best: Optional[float] = None
    kept: Optional[bool] = None
    status: str = "keep"        # keep | discard | crash | timeout
    params_M: Optional[float] = None
    train_n: Optional[int] = None
    agent: Optional[str] = None
    model: Optional[str] = None
    comparison_image: Optional[str] = None
    advice_for_others: Optional[str] = None
    commit: Optional[str] = None
    extras: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        d = asdict(self)
        d.update(d.pop("extras"))
        return {k: v for k, v in d.items() if v is not None or k in ("kept",)}


# --- Run handle ----------------------------------------------------------

class Run:
    """One agentic run targeting one challenge.

    Use ``Run.create()`` to start a new run, or ``Run.load()`` to resume.
    """

    def __init__(self, slug: str, root: Path = DOCS_RUNS):
        self.slug = slug
        self.root = root
        self.dir = root / slug
        self.iter_root = self.dir / "iterations"

    @classmethod
    def create(cls, *, challenge: str, slug_prefix: str,
               agent: str = "claude", model: str = "claude-sonnet-4.5",
               notes: str = "", root: Path = DOCS_RUNS) -> "Run":
        _ensure_dir(root)
        slug = make_slug(slug_prefix, root=root)
        run = cls(slug, root=root)
        _ensure_dir(run.dir)
        _ensure_dir(run.iter_root)
        manifest = {
            "slug": slug,
            "challenge": challenge,
            "slug_prefix": slug_prefix,
            "started": utc_now_iso(),
            "agent": agent,
            "model": model,
            "status": "running",
            "notes": notes,
        }
        _atomic_write_text(run.dir / "manifest.json",
                           json.dumps(manifest, indent=2))
        _atomic_write_text(run.dir / "results.tsv", RESULTS_HEADER)
        _atomic_write_text(run.dir / "stages.tsv", STAGES_HEADER)
        run._touch_index()
        return run

    @classmethod
    def load(cls, slug: str, root: Path = DOCS_RUNS) -> "Run":
        run = cls(slug, root=root)
        if not run.dir.exists():
            raise FileNotFoundError(f"no run dir at {run.dir}")
        return run

    # ---- manifest helpers ----

    @property
    def manifest(self) -> dict[str, Any]:
        return json.loads((self.dir / "manifest.json").read_text())

    def update_manifest(self, **changes) -> None:
        m = self.manifest
        m.update(changes)
        _atomic_write_text(self.dir / "manifest.json", json.dumps(m, indent=2))
        self._touch_index()

    def finalize(self, status: str = "done", **extra) -> None:
        self.update_manifest(status=status, ended=utc_now_iso(), **extra)

    # ---- iteration recording ----

    def iteration_dir(self, iter_n: int) -> Path:
        return self.iter_root / f"iter-{iter_n:04d}"

    def record_iteration(
        self,
        iter_n: int,
        observation: Observation,
        *,
        comparison_png: Path | str | None = None,
        solver_src: Path | str | None = None,
        stdout_log: str | None = None,
    ) -> Path:
        """Write one iteration's artefacts + append a row to results.tsv +
        append a line to the cross-run scratch pad. Returns the iteration dir.
        """
        d = _ensure_dir(self.iteration_dir(iter_n))

        # Make the observation's comparison_image path relative to docs/ so
        # the dashboard JS can use it as-is.
        rel_img: Optional[str] = None
        if comparison_png is not None:
            comparison_png = Path(comparison_png)
            assert comparison_png.exists(), f"no such file: {comparison_png}"
            dst = d / "comparison.png"
            if comparison_png.resolve() != dst.resolve():
                shutil.copyfile(comparison_png, dst)
            rel_img = f"runs/{self.slug}/iterations/iter-{iter_n:04d}/comparison.png"
            observation.comparison_image = rel_img

        if solver_src is not None:
            solver_src = Path(solver_src)
            if solver_src.exists():
                (d / "solver.py.txt").write_text(solver_src.read_text())

        if stdout_log is not None:
            (d / "stdout.log").write_text(stdout_log)

        # Per-iteration observation.
        _atomic_write_text(d / "observation.json",
                           json.dumps(observation.to_json(), indent=2))

        # Append to results.tsv.
        with (self.dir / "results.tsv").open("a") as f:
            f.write("\t".join(str(c) for c in [
                iter_n,
                observation.commit or "",
                "" if observation.val_score is None else f"{observation.val_score:.6g}",
                "" if observation.headroom is None else f"{observation.headroom:.6g}",
                observation.status or "",
                observation.change_class or "",
                _tsv_safe(observation.rationale or ""),
            ]) + "\n")

        # Append to shared scratch pad.
        scratch = self.root / "observations.jsonl"
        with scratch.open("a") as f:
            f.write(json.dumps(observation.to_json()) + "\n")

        self._touch_index()
        return d

    def record_stage(
        self,
        *,
        iter_n: int,
        stage_val_score: float,
        stage_headroom: float,
        iter_val_score: float,
        verdict: str,        # "ok" | "overfit"
        notes: str = "",
    ) -> None:
        gap = iter_val_score - stage_val_score
        with (self.dir / "stages.tsv").open("a") as f:
            f.write("\t".join(str(c) for c in [
                iter_n,
                f"{stage_val_score:.6g}",
                f"{stage_headroom:.6g}",
                f"{gap:.6g}",
                verdict,
                _tsv_safe(notes),
            ]) + "\n")
        self._touch_index()

    # ---- index ----

    def _touch_index(self) -> None:
        """Recompute runs-index.json based on what's on disk."""
        _ensure_dir(self.root)
        idx_path = self.root / "runs-index.json"
        runs = []
        for run_dir in sorted(self.root.iterdir()):
            if not run_dir.is_dir():
                continue
            m_path = run_dir / "manifest.json"
            if not m_path.exists():
                continue
            try:
                manifest = json.loads(m_path.read_text())
            except Exception:
                continue
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
                        if best_score is None or v > best_score:
                            best_score = v
                    except (ValueError, IndexError):
                        pass
                    try:
                        h = float(cells[3])
                        if best_hr is None or h > best_hr:
                            best_hr = h
                    except (ValueError, IndexError):
                        pass
            runs.append({
                "slug": manifest.get("slug", run_dir.name),
                "challenge": manifest.get("challenge"),
                "started": manifest.get("started"),
                "status": manifest.get("status", "running"),
                "n_iterations": n_iter,
                "best_score": best_score,
                "best_headroom": best_hr,
            })
        index = {
            "schema_version": 1,
            "updated": utc_now_iso(),
            "runs": runs,
        }
        _atomic_write_text(idx_path, json.dumps(index, indent=2))


def _tsv_safe(s: str) -> str:
    return s.replace("\t", " ").replace("\n", " ").replace("\r", " ")
