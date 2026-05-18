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
        "slug_prefix": "demo-fair-itnet-v2-search",
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
        "slug_prefix": "demo-fair-itnet-v3-search",
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
    # ----- Fair re-runs for solvers that previously had standalone agents ------
    "dual_domain": {
        "solver": "pentathlon/demo_dl_reference/solver_dual_ddomain.py",
        "env_var": "DD_CONFIG_PATH",
        "slug_prefix": "demo-fair-dual-domain-search",
        "agent_name": "dual-domain-search",
        "space": {
            "epochs":     (3, 8, "int"),
            "lr":         (1e-4, 5e-3, "log"),
            "batch_size": ([1, 2, 4], "choice"),
            "unet_c":     ([8, 16, 24], "choice"),
        },
    },
    "dual_domain_bilateral": {
        "solver": "pentathlon/demo_dl_reference/solver_dual_ddomain_bilateral.py",
        "env_var": "DD_BF_CONFIG_PATH",
        "slug_prefix": "demo-fair-dual-domain-bf-search",
        "agent_name": "dual-domain-bf-search",
        "space": {
            "epochs":     (10, 30, "int"),
            "lr":         (1e-3, 1e-2, "log"),
            "batch_size": ([1, 2, 4], "choice"),
            "proj_kernel": ([3, 5, 7], "choice"),
            "img_kernel":  ([5, 7, 9], "choice"),
            "proj_sx":    (0.5, 2.0, "linear"),
            "proj_sy":    (1.0, 3.0, "linear"),
            "proj_sr":    (0.01, 0.05, "log"),
            "img_sx":     (1.0, 3.0, "linear"),
            "img_sr":     (0.01, 0.05, "log"),
        },
    },
    "tv_iterative": {
        "solver": "pentathlon/demo_dl_reference/solver_tv_search.py",
        "env_var": "TV_CONFIG_PATH",
        "slug_prefix": "demo-fair-tv-search",
        "agent_name": "tv-search",
        "space": {
            "tv_lambda":     (1e-4, 1e-2, "log"),
            "tv_iterations": (50, 500, "int"),
            "tv_lr":         (1e-3, 1e-1, "log"),
            "tv_clip_max":   (0.03, 0.08, "linear"),
            "tv_decay":      (0.0, 0.05, "linear"),
        },
    },
    "wu_2015": {
        "solver": "pentathlon/demo_dl_reference/solver_wu_2015.py",
        "env_var": "WU_CONFIG_PATH",
        "slug_prefix": "demo-fair-wu-search",
        "agent_name": "wu-search",
        "space": {
            "wu_n_bands":        ([4, 6, 8, 12], "choice"),
            "wu_n_outer":        ([1, 2, 3], "choice"),
            "wu_motion_range":   ([3, 5, 8, 12], "choice"),
            "wu_motion_window":  ([1, 2, 4], "choice"),
            "wu_soft_thresh":    (5e-4, 5e-3, "log"),
        },
    },
    "hammernik": {
        "solver": "pentathlon/demo_dl_reference/solver_hammernik_2017.py",
        "env_var": "HAMMERNIK_CONFIG_PATH",
        "slug_prefix": "demo-fair-hammernik-search",
        "agent_name": "hammernik-search",
        "space": {
            "epochs":          (10, 30, "int"),
            "lr":              (1e-4, 2e-3, "log"),
            # v3: batch_size=2 forced (default 4 OOMs even on q6000/24GB
            # because the RBF-bump activation loop keeps T·N_k·n_bumps
            # tensors live for autograd).
            "batch_size":      ([2], "choice"),
            "vn_T":            ([3, 5], "choice"),
            "vn_n_filters":    ([16, 24], "choice"),
            "vn_kernel":       ([7, 9, 11], "choice"),
            "vn_lambda_init":  (1e-4, 1e-2, "log"),
        },
    },
    "hammernik_vn": {
        "solver": "pentathlon/demo_dl_reference/solver_hammernik_vn.py",
        "env_var": "HAMMERNIK_VN_CONFIG_PATH",
        "slug_prefix": "demo-fair-hammernik-vn-search",
        "agent_name": "hammernik-vn-search",
        "space": {
            "epochs":          (10, 18, "int"),
            "lr":              (5e-5, 3e-4, "log"),       # tightened: avoid divergence
            "batch_size":      ([2], "choice"),           # gradient-ckpt fits batch=2
            "vn_T":            ([3, 5, 7], "choice"),     # capped at 7 (T=10 OOMs)
            "vn_n_filters":    ([16, 24, 32], "choice"),  # capped at 32 (N_k=48 OOMs)
            "vn_kernel":       ([7, 9, 11], "choice"),
            "vn_lambda_init":  (1e-4, 1e-2, "log"),
            "vn_init":         (["fbp"], "choice"),       # backproj diverges; fix to fbp
        },
    },
    "uswin": {
        "solver": "pentathlon/demo_dl_reference/solver_uswin.py",
        "env_var": "USWIN_CONFIG_PATH",
        "slug_prefix": "demo-fair-uswin-search",
        "agent_name": "uswin-search",
        "space": {
            "epochs":      (6, 14, "int"),
            "lr":          (1e-4, 1e-3, "log"),
            "batch_size":  ([2, 4, 8], "choice"),
            "uswin_c":     ([16, 24, 32], "choice"),
            "swin_window": ([4, 8, 16], "choice"),
            "swin_heads":  ([2, 4, 8], "choice"),
        },
    },
    "naf": {
        "solver": "pentathlon/demo_dl_reference/solver_naf.py",
        "env_var": "NAF_CONFIG_PATH",
        "slug_prefix": "demo-fair-naf-search",
        "agent_name": "naf-search",
        "space": {
            "naf_n_freqs":      ([6, 8, 10, 12, 14], "choice"),
            "naf_hidden":       ([128, 192, 256], "choice"),
            "naf_layers":       ([4, 5, 6], "choice"),
            "naf_n_iter":       (1500, 4000, "int"),       # was [300, 1000]
            "naf_lr":           (5e-4, 1e-2, "log"),       # narrower for stability
            "naf_tv_weight":    (1e-6, 1e-3, "log"),       # extend low end (was 1e-5)
            "naf_outer_wall_s": ([2400], "choice"),        # 40 min per outer iter
        },
    },
    "r2gaussian": {
        "solver": "pentathlon/demo_dl_reference/solver_r2gaussian.py",
        "env_var": "R2G_CONFIG_PATH",
        "slug_prefix": "demo-fair-r2gaussian-search",
        "agent_name": "r2gaussian-search",
        "space": {
            "gs_n_gaussians": ([512, 1024, 2048], "choice"),
            "gs_n_iter":      (300, 800, "int"),
            "gs_lr_pos":      (1e-3, 2e-2, "log"),
            "gs_lr_scale":    (5e-3, 5e-2, "log"),
            "gs_lr_amp":      (5e-3, 5e-2, "log"),
            "gs_amp_init":    (5e-3, 5e-2, "log"),
            "gs_scale_init":  (0.02, 0.10, "linear"),
            "gs_tv_weight":   (1e-5, 1e-3, "log"),
        },
    },
    "diffusion": {
        "solver": "pentathlon/demo_dl_reference/solver_diffusion.py",
        "env_var": "DIFFUSION_CONFIG_PATH",
        "slug_prefix": "demo-dl-diffusion-search",
        "agent_name": "diffusion-search",
        "space": {
            "diff_mode":          (["dps", "mcg"], "choice"),
            "diff_sample_steps":  ([30, 50, 80, 120], "choice"),
            "diff_eta":           (0.1, 5.0, "log"),
            "diff_init":          (["fbp", "noise"], "choice"),
        },
    },
    # ----- DDPM training (unconstrained variant — pick best hyperparams here) ----
    "ddpm": {
        "solver": "pentathlon/demo_dl_reference/solver_ddpm.py",
        "env_var": "DDPM_CONFIG_PATH",
        "slug_prefix": "demo-dl-ddpm-search",
        "agent_name": "ddpm-search",
        "space": {
            "ddpm_mode":         (["unconstrained"], "choice"),  # constrained reuses
                                                                  # best hyperparams later
            "ddpm_n_train":      ([1000, 2000, 3000], "choice"),
            "ddpm_ch":           ([24, 32, 48], "choice"),
            "ddpm_n_steps":      ([500, 1000], "choice"),
            "ddpm_epochs":       (15, 40, "int"),
            "ddpm_batch":        ([4, 8, 16], "choice"),
            "ddpm_lr":           (5e-5, 5e-4, "log"),
            "ddpm_weight_decay": ([0.0, 1e-5, 1e-4], "choice"),
            # Cap each iter's training to 20 min so 20 iters fit in 7 h.
            "ddpm_train_wall_s": ([1200], "choice"),
        },
    },
    # ----- Diffusion-recon searches: load a frozen DDPM, vary sampling hyperparams ---
    "diffusion_recon_unconstrained": {
        "solver": "pentathlon/demo_dl_reference/solver_diffusion_recon.py",
        "env_var": "DIFFUSION_RECON_CONFIG_PATH",
        "slug_prefix": "demo-dl-diffusion-recon-unconstrained-search",
        "agent_name": "diffusion-recon-unconstrained-search",
        "space": {
            "recon_ckpt":          (["/cluster/maier/Agent4CT/checkpoints/ddpm_unconstrained_final.pt"], "choice"),
            "recon_mode":          (["dps", "mcg"], "choice"),
            "recon_sample_steps":  ([100, 200, 500], "choice"),  # DPS needs many steps
            # v3 adaptive ζ_t = eta / ‖residual‖ (Chung 2023 eq. 12). eta now
            # in [0.1, 10] range — original DPS paper used 1.0 with this scaling.
            "recon_eta":           (0.1, 10.0, "log"),
            "recon_init":          (["fbp", "noise"], "choice"),
            # No mid-trajectory clamp (default False); optional soft per-step
            # displacement cap as ablation knob.
            "recon_eta_clamp":     ([False, True], "choice"),
        },
    },
    "diffusion_recon_constrained": {
        "solver": "pentathlon/demo_dl_reference/solver_diffusion_recon.py",
        "env_var": "DIFFUSION_RECON_CONFIG_PATH",
        "slug_prefix": "demo-dl-diffusion-recon-constrained-search",
        "agent_name": "diffusion-recon-constrained-search",
        "space": {
            "recon_ckpt":          (["/cluster/maier/Agent4CT/checkpoints/ddpm_constrained_final.pt"], "choice"),
            "recon_mode":          (["dps", "mcg"], "choice"),
            "recon_sample_steps":  ([100, 200, 500], "choice"),
            "recon_eta":           (0.1, 10.0, "log"),
            "recon_init":          (["fbp", "noise"], "choice"),
            "recon_eta_clamp":     ([False, True], "choice"),
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


def create_run(slug_prefix, agent_name, notes, model="random-search"):
    DOCS_RUNS.mkdir(parents=True, exist_ok=True)
    slug = make_slug(slug_prefix)
    run_dir = DOCS_RUNS / slug
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "iterations").mkdir(exist_ok=True)
    manifest = {
        "slug": slug, "challenge": "dl_sparse_view",
        "slug_prefix": slug_prefix,
        "started": utc_now_iso(), "agent": agent_name, "model": model,
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
    """Random-search sampler (legacy)."""
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


def _optuna_suggest(trial, space):
    """Translate our search-space spec into Optuna trial.suggest_* calls.
    Mirrors sample_params semantics so the random and TPE runs are
    drawing from the exact same support."""
    out = {}
    for key, spec in space.items():
        if isinstance(spec, tuple) and len(spec) == 3:
            lo, hi, mode = spec
            if mode == "int":
                out[key] = trial.suggest_int(key, int(lo), int(hi))
            elif mode == "log":
                out[key] = trial.suggest_float(key, float(lo), float(hi), log=True)
            elif mode == "linear":
                out[key] = trial.suggest_float(key, float(lo), float(hi))
            else:
                raise ValueError(mode)
        elif isinstance(spec, tuple) and len(spec) == 2 and spec[1] == "choice":
            out[key] = trial.suggest_categorical(key, list(spec[0]))
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
    # Subprocess timeout must exceed the solver's internal per-iter wall.
    # NAF / R2Gaussian use 40-min outer walls; 1 h subprocess cap leaves margin.
    res = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=3600)
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
    p.add_argument("--sampler", choices=["random", "tpe"], default="random",
                   help="random = legacy uniform/log; tpe = Optuna's "
                        "Tree-structured Parzen Estimator (adaptive)")
    p.add_argument("--tpe-storage",
                   default="/cluster/maier/Agent4CT/optuna",
                   help="Directory for the per-study SQLite db (TPE only)")
    p.add_argument("--tpe-startup", type=int, default=5,
                   help="Random startup trials before TPE kicks in")
    args = p.parse_args()

    spec = SOLVERS[args.solver]
    # When using a non-default sampler, suffix both slug prefix and agent
    # name so the runs are clearly tagged but still group under the same
    # chart on the dashboard (chartGroupKey uses first 2 hyphen segments).
    if args.sampler == "tpe":
        spec = dict(spec)
        # demo-fair-uswin-search -> demo-fair-tpe-uswin-search
        spec["slug_prefix"] = spec["slug_prefix"].replace(
            "demo-fair-", "demo-fair-tpe-")
        spec["agent_name"] = spec["agent_name"] + "-tpe"
        model_label = "optuna-tpe"
    else:
        model_label = "random-search"

    rng = random.Random(args.seed)
    slug, run_dir = create_run(
        spec["slug_prefix"], spec["agent_name"],
        args.notes or f"{args.solver} hyperparameter {args.sampler} search",
        model=model_label,
    )

    print(f"[agent] === {args.solver} {args.sampler} search → {slug} ===")
    print(f"[agent] space keys: {list(spec['space'].keys())}")

    # Lazy-import Optuna so the random-search path doesn't require it.
    study = None
    if args.sampler == "tpe":
        try:
            import optuna
        except ImportError as e:
            raise SystemExit(
                "TPE sampler requested but Optuna is not installed. "
                "Run: pip install optuna"
            ) from e
        Path(args.tpe_storage).mkdir(parents=True, exist_ok=True)
        storage = f"sqlite:///{args.tpe_storage}/{slug}.db"
        study = optuna.create_study(
            study_name=slug, storage=storage, direction="maximize",
            sampler=optuna.samplers.TPESampler(
                seed=args.seed, n_startup_trials=args.tpe_startup),
            load_if_exists=True,
        )
        print(f"[agent] Optuna TPE study persisted at {storage} "
              f"(n_startup={args.tpe_startup}, seed={args.seed})", flush=True)

    best_hr = 0.0
    best_params = None
    for i in range(1, args.iterations + 1):
        if args.sampler == "tpe":
            trial = study.ask()
            params = _optuna_suggest(trial, spec["space"])
        else:
            params = sample_params(spec["space"], rng)
        print(f"\n[agent] iter {i}/{args.iterations}: {json.dumps(params)}",
              flush=True)
        out_dir = Path(args.out_base) / f"{slug}-iter-{i:04d}"
        result = run_solver(spec["solver"], spec["env_var"], params, out_dir)
        hr = (result or {}).get("headroom", 0.0)
        if result is None:
            print(f"[agent] iter {i} FAILED")
        else:
            print(f"[agent] hr={hr:.4f} SSIM={result.get('val_score', 0):.4f} "
                  f"params_M={result.get('params_M', 0):.3f} "
                  f"time={result.get('train_time_s', 0):.1f}s")
            if hr > best_hr:
                best_hr = hr; best_params = params.copy()
                print(f"[agent] *** NEW BEST: hr={best_hr:.4f} ***")
            record_iteration(run_dir, i, params, result, spec["agent_name"],
                              out_dir)
        if args.sampler == "tpe":
            study.tell(trial, hr)        # 0.0 on failure tells TPE this region is bad

    manifest_path = run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["status"] = "done"
    manifest["finished"] = utc_now_iso()
    manifest["sampler"] = args.sampler
    if best_params is not None:
        manifest["best_params"] = best_params
        manifest["best_headroom"] = best_hr
    manifest_path.write_text(json.dumps(manifest, indent=2))

    update_index()
    print("\n" + "=" * 60)
    print(f"[agent] {args.solver} {args.sampler.upper()} SEARCH COMPLETE — "
          f"best hr={best_hr:.4f}")
    if best_params:
        for k, v in best_params.items():
            print(f"  {k}: {v}")
    print("=" * 60)


if __name__ == "__main__":
    main()
