"""Wu 2015 hyperparameter search agent (standalone, no ddssl_ldct imports).

Random-search over the five solver_wu_2015.py knobs, recording results to
``docs/runs/demo-dl-wu-search-<date>-NN/`` in the same shape as
``tv_search_agent_standalone.py`` so the dashboard picks the run up
automatically.

Usage:
    python scripts/wu_search_agent_standalone.py --iterations 20
"""
from __future__ import annotations
import argparse
import json
import math
import os
import random
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DOCS_RUNS = REPO_ROOT / "docs" / "runs"


def utc_now_iso():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def make_slug(prefix):
    today = datetime.now(timezone.utc).strftime("%Y%m%d")
    seq = 1
    if DOCS_RUNS.exists():
        for d in DOCS_RUNS.iterdir():
            if d.is_dir() and d.name.startswith(f"{prefix}-{today}-"):
                try:
                    tail = int(d.name.split("-")[-1])
                    seq = max(seq, tail + 1)
                except ValueError:
                    pass
    return f"{prefix}-{today}-{seq:02d}"


def create_run(challenge, slug_prefix, agent, model, notes=""):
    DOCS_RUNS.mkdir(parents=True, exist_ok=True)
    slug = make_slug(slug_prefix)
    run_dir = DOCS_RUNS / slug
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "iterations").mkdir(exist_ok=True)
    manifest = {
        "slug": slug, "challenge": challenge, "slug_prefix": slug_prefix,
        "started": utc_now_iso(), "agent": agent, "model": model,
        "status": "running", "notes": notes,
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    (run_dir / "results.tsv").write_text(
        "iter\tcommit\tval_score\theadroom\tstatus\tchange_class\tagent\tmodel\trationale\n"
    )
    (run_dir / "stages.tsv").write_text(
        "iter\tstage_val_score\tstage_headroom\tgap\tverdict\tnotes\n"
    )
    print(f"[agent] Created run: {slug}")
    return slug, run_dir


def record_iteration(run_dir, iter_n, params, result, out_dir=None):
    headroom = result.get("headroom", 0)
    val_score = result.get("val_score", 0)
    iter_dir = run_dir / "iterations" / f"iter-{iter_n:04d}"
    iter_dir.mkdir(parents=True, exist_ok=True)

    comparison_image = None
    if out_dir is not None:
        src_img = Path(out_dir) / "comparison.png"
        if src_img.exists():
            dst_img = iter_dir / "comparison.png"
            import shutil
            shutil.copy2(src_img, dst_img)
            comparison_image = (
                f"runs/{run_dir.name}/iterations/iter-{iter_n:04d}/comparison.png"
            )
            print(f"[agent] Copied comparison image to {dst_img}")

    rationale = (
        f"Wu 2015 search: n_bands={params['wu_n_bands']}, "
        f"n_outer={params['wu_n_outer']}, "
        f"motion_range={params['wu_motion_range']}, "
        f"motion_window={params['wu_motion_window']}, "
        f"soft_thresh={params['wu_soft_thresh']:.5f}"
    )
    obs = {
        "ts": utc_now_iso(),
        "run_id": run_dir.name,
        "iter": iter_n,
        "challenge": "dl_sparse_view",
        "change_class": "optimizer",
        "rationale": rationale,
        "val_score": val_score,
        "headroom": headroom,
        "kept": headroom > 0,
        "status": "keep" if headroom > 0 else "discard",
        "params_M": 0.0,
        "train_n": 0,
        "agent": "wu-search",
        "model": "random-search",
        "advice_for_others": (
            f"Wu 2015 bands={params['wu_n_bands']} outer={params['wu_n_outer']} "
            f"mrange={params['wu_motion_range']} thresh={params['wu_soft_thresh']:.4f} "
            f"-> hr={headroom:.4f}"
        ),
    }
    if comparison_image:
        obs["comparison_image"] = comparison_image

    (iter_dir / "observation.json").write_text(json.dumps(obs, indent=2))

    with (run_dir / "results.tsv").open("a") as f:
        f.write(
            f"{iter_n}\t\t{val_score:.6g}\t{headroom:.6g}\t"
            f"{'keep' if headroom > 0 else 'discard'}\toptimizer\t"
            f"wu-search\trandom-search\t{rationale.replace(chr(9), ' ')}\n"
        )

    scratch = DOCS_RUNS / "observations.jsonl"
    with scratch.open("a") as f:
        f.write(json.dumps(obs) + "\n")

    print(f"[agent] Recorded iter {iter_n}: headroom={headroom:.4f}")


def update_index():
    idx_path = DOCS_RUNS / "runs-index.json"
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
                    if best_score is None or v > best_score:
                        best_score = v
                    h = float(cells[3])
                    if best_hr is None or h > best_hr:
                        best_hr = h
                except (ValueError, IndexError):
                    pass
        parts = run_dir.name.split("-")
        m_short = parts[-2] + "-" + parts[-1] if len(parts) >= 2 else run_dir.name
        runs.append({
            "slug": run_dir.name, "short_id": m_short,
            "challenge": manifest.get("challenge"),
            "started": manifest.get("started"),
            "status": manifest.get("status", "running"),
            "n_iterations": n_iter,
            "best_score": best_score,
            "best_headroom": best_hr,
            "agent": manifest.get("agent"),
            "model": manifest.get("model"),
        })
    index = {"schema_version": 1, "updated": utc_now_iso(), "runs": runs}
    idx_path.write_text(json.dumps(index, indent=2))
    print(f"[agent] Updated index with {len(runs)} runs")


# ---------------------------------------------------------------------------
SEARCH_SPACE = {
    "wu_n_bands":       ([4, 6, 8, 12], "choice"),
    "wu_n_outer":       ([1, 2, 3], "choice"),
    "wu_motion_range":  ([3, 5, 8, 12], "choice"),
    "wu_motion_window": ([1, 2, 4], "choice"),
    "wu_soft_thresh":   ((5e-4, 5e-3), "log"),
}


def sample_params(rng: random.Random):
    params = {}
    for key, (spec, mode) in SEARCH_SPACE.items():
        if mode == "choice":
            params[key] = int(rng.choice(spec))
        elif mode == "log":
            lo, hi = spec
            params[key] = float(math.exp(rng.uniform(math.log(lo), math.log(hi))))
        elif mode == "linear":
            lo, hi = spec
            params[key] = float(rng.uniform(lo, hi))
        else:
            raise ValueError(mode)
    return params


def run_wu_solver(params, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    config_file = out_dir / "wu_config.json"
    config_file.write_text(json.dumps(params))

    cmd = [
        sys.executable,
        str(REPO_ROOT / "pentathlon" / "demo_dl_reference" / "solver_wu_2015.py"),
        str(out_dir),
    ]
    env = dict(os.environ)
    env["WU_CONFIG_PATH"] = str(config_file)
    env["PYTHONPATH"] = (
        str(REPO_ROOT)
        + (":" + env["PYTHONPATH"] if "PYTHONPATH" in env else "")
    )

    res = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if res.returncode != 0:
        print(f"[agent] Solver failed (rc={res.returncode}):", file=sys.stderr)
        print(res.stderr, file=sys.stderr)
        return None
    rp = out_dir / "result.json"
    if not rp.exists():
        print(f"[agent] No result.json at {rp}")
        return None
    return json.loads(rp.read_text())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--iterations", type=int, default=20)
    p.add_argument("--seed", type=int, default=20260516)
    p.add_argument("--notes", default="Wu 2015 hyperparameter random search")
    args = p.parse_args()

    rng = random.Random(args.seed)

    slug, run_dir = create_run(
        challenge="dl_sparse_view",
        slug_prefix="demo-dl-wu-search",
        agent="wu-search",
        model="random-search",
        notes=args.notes,
    )

    print(f"[agent] Starting Wu 2015 search: {slug}")
    print(f"[agent] Running {args.iterations} iterations")

    best_headroom = 0.0
    best_params = None
    for i in range(1, args.iterations + 1):
        print(f"\n[agent] === Iteration {i}/{args.iterations} ===")
        params = sample_params(rng)
        print(f"[agent] Params: {json.dumps(params)}")

        out_dir = Path(f"/tmp/wu-search-{slug}-{i:04d}")
        result = run_wu_solver(params, out_dir)
        if result is None:
            print(f"[agent] iter {i} failed, skipping")
            continue

        headroom = result.get("headroom", 0)
        print(
            f"[agent] hr={headroom:.4f}  SSIM={result.get('val_score', 0):.4f}  "
            f"time={result.get('train_time_s', 0):.1f}s"
        )
        if headroom > best_headroom:
            best_headroom = headroom
            best_params = params.copy()
            print(f"[agent] *** NEW BEST: hr={best_headroom:.4f} ***")
        record_iteration(run_dir, i, params, result, out_dir=out_dir)

    # Mark run done in manifest.
    manifest_path = run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["status"] = "done"
    manifest["finished"] = utc_now_iso()
    if best_params is not None:
        manifest["best_params"] = best_params
        manifest["best_headroom"] = best_headroom
    manifest_path.write_text(json.dumps(manifest, indent=2))

    update_index()

    print("\n" + "=" * 60)
    print(f"[agent] SEARCH COMPLETE — best hr={best_headroom:.4f}")
    if best_params:
        for k, v in best_params.items():
            print(f"  {k}: {v}")
    print("=" * 60)


if __name__ == "__main__":
    main()
