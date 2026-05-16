"""Generic 20-iter random-search agent for the learned demo-dl-reference solvers.

Currently supports three solvers via --solver:
  - itnet_v2      → solver_itnet_v2.py        (ITNET_CONFIG_PATH)
  - itnet_v3      → solver_itnet_v3.py        (ITNET_CONFIG_PATH)
  - hammernik     → solver_hammernik_2017.py  (HAMMERNIK_CONFIG_PATH)

Each iter samples a hyperparameter point, runs the solver via subprocess,
copies its `comparison.png` into the iter dir, and writes
`observation.json` + a `results.tsv` row in the autoresearch shape that
the dashboard reads.

Usage:
    python scripts/learned_solver_search_agent.py --solver itnet_v3 --iterations 20
"""
from __future__ import annotations
import argparse
import json
import math
import os
import random
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DOCS_RUNS = REPO_ROOT / "docs" / "runs"

# ---------------------------------------------------------------------------
SOLVERS = {
    "itnet_v2": {
        "solver": "pentathlon/demo_dl_reference/solver_itnet_v2.py",
        "env_var": "ITNET_CONFIG_PATH",
        "slug_prefix": "demo-dl-itnet-v2-search",
        "agent_name": "itnet-v2-search",
        "space": {
            "pretrain_epochs":     (3, 8, "int"),
            "pretrain_lr":         (1e-4, 5e-3, "log"),
            "itnet_k":             (3, 8, "int"),
            "itnet_alpha_init":    (1e-3, 5e-2, "log"),
            "residual_learning":   ([True, False], "choice"),
        },
    },
    "itnet_v3": {
        "solver": "pentathlon/demo_dl_reference/solver_itnet_v3.py",
        "env_var": "ITNET_CONFIG_PATH",
        "slug_prefix": "demo-dl-itnet-v3-search",
        "agent_name": "itnet-v3-search",
        "space": {
            "epochs":         (5, 15, "int"),
            "lr":             (1e-4, 2e-3, "log"),
            "batch_size":     ([10, 20, 40], "choice"),
            "unet_c":         ([8, 12, 16], "choice"),
            "itnet_k":        ([2, 3, 4], "choice"),
            "alpha_init":     (1e-3, 1e-2, "log"),
        },
    },
    "hammernik": {
        "solver": "pentathlon/demo_dl_reference/solver_hammernik_2017.py",
        "env_var": "HAMMERNIK_CONFIG_PATH",
        "slug_prefix": "demo-dl-hammernik-search",
        "agent_name": "hammernik-search",
        "space": {
            "epochs":          (10, 30, "int"),
            "lr":              (1e-4, 2e-3, "log"),
            "vn_T":            ([3, 5, 7], "choice"),
            "vn_n_filters":    ([16, 24, 32], "choice"),
            "vn_kernel":       ([7, 9, 11, 13], "choice"),
            "vn_lambda_init":  (1e-4, 1e-2, "log"),
        },
    },
}


# ---------------------------------------------------------------------------
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


def create_run(slug_prefix, agent_name, notes):
    DOCS_RUNS.mkdir(parents=True, exist_ok=True)
    slug = make_slug(slug_prefix)
    run_dir = DOCS_RUNS / slug
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "iterations").mkdir(exist_ok=True)
    manifest = {
        "slug": slug, "challenge": "dl_sparse_view",
        "slug_prefix": slug_prefix,
        "started": utc_now_iso(), "agent": agent_name, "model": "random-search",
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


def sample_params(space, rng):
    out = {}
    for key, spec in space.items():
        if isinstance(spec, tuple) and len(spec) == 3:
            lo, hi, mode = spec
            if mode == "int":
                out[key] = int(rng.randint(lo, hi))
            elif mode == "log":
                out[key] = float(math.exp(rng.uniform(math.log(lo), math.log(hi))))
            elif mode == "linear":
                out[key] = float(rng.uniform(lo, hi))
            else:
                raise ValueError(mode)
        elif isinstance(spec, tuple) and len(spec) == 2 and spec[1] == "choice":
            out[key] = rng.choice(spec[0])
        else:
            raise ValueError(f"Bad spec for {key}: {spec}")
    return out


def run_solver(solver_path, env_var, params, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg_file = out_dir / "config.json"
    cfg_file.write_text(json.dumps(params))
    cmd = [sys.executable, str(REPO_ROOT / solver_path), str(out_dir)]
    env = dict(os.environ)
    env[env_var] = str(cfg_file)
    env["PYTHONPATH"] = (
        str(REPO_ROOT) + (":" + env["PYTHONPATH"] if "PYTHONPATH" in env else "")
    )
    res = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=900)
    if res.returncode != 0:
        print(f"[agent] solver failed (rc={res.returncode}):", file=sys.stderr)
        print(res.stderr[-2000:], file=sys.stderr)
        return None
    rp = out_dir / "result.json"
    if not rp.exists():
        return None
    return json.loads(rp.read_text())


def record_iteration(run_dir, iter_n, params, result, agent_name, out_dir=None):
    headroom = result.get("headroom", 0.0)
    val_score = result.get("val_score", 0.0)
    iter_dir = run_dir / "iterations" / f"iter-{iter_n:04d}"
    iter_dir.mkdir(parents=True, exist_ok=True)

    comparison_image = None
    if out_dir is not None:
        src = Path(out_dir) / "comparison.png"
        if src.exists():
            dst = iter_dir / "comparison.png"
            shutil.copy2(src, dst)
            comparison_image = (
                f"runs/{run_dir.name}/iterations/iter-{iter_n:04d}/comparison.png"
            )
            print(f"[agent] Copied {dst}")

    param_str = ", ".join(
        f"{k}={v:.5g}" if isinstance(v, float) else f"{k}={v}"
        for k, v in params.items()
    )
    obs = {
        "ts": utc_now_iso(),
        "run_id": run_dir.name,
        "iter": iter_n,
        "challenge": "dl_sparse_view",
        "change_class": "architecture",
        "rationale": f"{agent_name}: {param_str}",
        "val_score": val_score,
        "headroom": headroom,
        "kept": headroom > 0,
        "status": "keep" if headroom > 0 else "discard",
        "params_M": result.get("params_M", 0),
        "train_n": result.get("train_n", 0),
        "agent": agent_name,
        "model": "random-search",
        "advice_for_others": f"{agent_name}: {param_str} -> hr={headroom:.4f}",
    }
    if comparison_image:
        obs["comparison_image"] = comparison_image
    (iter_dir / "observation.json").write_text(json.dumps(obs, indent=2))

    with (run_dir / "results.tsv").open("a") as f:
        f.write(
            f"{iter_n}\t\t{val_score:.6g}\t{headroom:.6g}\t"
            f"{'keep' if headroom > 0 else 'discard'}\tarchitecture\t"
            f"{agent_name}\trandom-search\t{obs['rationale'].replace(chr(9), ' ')}\n"
        )
    with (DOCS_RUNS / "observations.jsonl").open("a") as f:
        f.write(json.dumps(obs) + "\n")
    print(f"[agent] Recorded iter {iter_n}: hr={headroom:.4f}")


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
    idx_path.write_text(json.dumps(
        {"schema_version": 1, "updated": utc_now_iso(), "runs": runs}, indent=2))
    print(f"[agent] Updated index with {len(runs)} runs")


# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--solver", required=True, choices=list(SOLVERS.keys()))
    p.add_argument("--iterations", type=int, default=20)
    p.add_argument("--seed", type=int, default=20260516)
    p.add_argument("--notes", default="")
    p.add_argument("--out-base", default="/cluster/maier/Agent4CT/runs")
    args = p.parse_args()

    spec = SOLVERS[args.solver]
    rng = random.Random(args.seed)
    slug, run_dir = create_run(
        spec["slug_prefix"], spec["agent_name"],
        args.notes or f"{args.solver} hyperparameter random search"
    )

    print(f"[agent] === {args.solver} search → {slug} ===")
    print(f"[agent] space keys: {list(spec['space'].keys())}")

    best_hr = 0.0
    best_params = None
    for i in range(1, args.iterations + 1):
        params = sample_params(spec["space"], rng)
        print(f"\n[agent] iter {i}/{args.iterations}: {json.dumps(params)}", flush=True)
        out_dir = Path(args.out_base) / f"{slug}-iter-{i:04d}"
        result = run_solver(spec["solver"], spec["env_var"], params, out_dir)
        if result is None:
            print(f"[agent] iter {i} FAILED, continuing")
            continue
        hr = result.get("headroom", 0.0)
        print(f"[agent] hr={hr:.4f} SSIM={result.get('val_score', 0):.4f} "
              f"params_M={result.get('params_M', 0):.3f} "
              f"time={result.get('train_time_s', 0):.1f}s")
        if hr > best_hr:
            best_hr = hr
            best_params = params.copy()
            print(f"[agent] *** NEW BEST: hr={best_hr:.4f} ***")
        record_iteration(run_dir, i, params, result, spec["agent_name"], out_dir)

    manifest_path = run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["status"] = "done"
    manifest["finished"] = utc_now_iso()
    if best_params is not None:
        manifest["best_params"] = best_params
        manifest["best_headroom"] = best_hr
    manifest_path.write_text(json.dumps(manifest, indent=2))

    update_index()
    print("\n" + "=" * 60)
    print(f"[agent] {args.solver} SEARCH COMPLETE — best hr={best_hr:.4f}")
    if best_params:
        for k, v in best_params.items():
            print(f"  {k}: {v}")
    print("=" * 60)


if __name__ == "__main__":
    main()
