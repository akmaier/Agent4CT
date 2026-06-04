#!/usr/bin/env python -u
"""Pre-stage iter-(N+1) configs for the autoresearch loop.

Given a slug's prior iter result, this helper:
  1. Loads the prior `observation.json` / config snapshot.
  2. Builds an iter-(N+1) config with the **short-budget** defaults
     required by solver_plan.md Step 2 (epochs ≤ 3, val_n ≤ 10,
     train_n ≤ 100) for hypothesis tests.
  3. Requires a `--hypothesis` string ("if I do X, I expect Y because
     Z") to be passed; refuses to write a config without it (this is
     the protocol checklist item #4 made enforceable).
  4. Writes the config JSON to
     `/cluster/maier/Agent4CT/agentic_cfgs/<solver>_iter_NN.json`
     and prints the sbatch command to dispatch it.

This is the path of least resistance for iter-2+ dispatches — using
the bare `mayo_first_iter.sbatch` with full epochs is the slow-grid-
search anti-pattern flagged in solver_plan.md.

Example:
    python scripts/propose_next_iter.py \\
        --slug mayo-ldct-claude-agentic-learned-primal-dual-search-20260603-01 \\
        --prev-iter 2 --solver learned_primal_dual \\
        --hypothesis "if I scale lpd_hidden 32→48 (still fits memory) I expect SSIM up ~0.01 because iter-2 used capacity below the breast-CT champion sweet spot" \\
        --knob lpd_hidden=48
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path


# Short-budget defaults required for iter-2+ per solver_plan.md Step 2.
# These OVERRIDE any prior-iter values unless the caller explicitly
# passes them via --knob.
SHORT_BUDGET_DEFAULTS = {
    "epochs":  3,
    "val_n":   5,
    "train_n": 100,
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--slug", required=True,
                   help="Autoresearch run slug, e.g. mayo-ldct-claude-agentic-"
                        "learned-primal-dual-search-20260603-01")
    p.add_argument("--prev-iter", type=int, required=True,
                   help="Iter number to read the prior result from "
                        "(iter-(N+1) will be the next).")
    p.add_argument("--solver", required=True,
                   help="Solver key (must be in claude_agentic_one_iter.py:"
                        "SOLVER_MAP).")
    p.add_argument("--hypothesis", required=True,
                   help="Named hypothesis for the next iter. Format: "
                        "\"if I do X, I expect Y because Z\". The script "
                        "refuses to write a config without one — this is "
                        "solver_plan.md Step 2 checklist item #4.")
    p.add_argument("--knob", action="append", default=[],
                   help="Single knob change as KEY=VALUE (repeatable). "
                        "Step 2 forbids multi-knob iters; this script does "
                        "NOT enforce that, but if you pass more than one "
                        "you are violating the protocol.")
    p.add_argument("--ignore-short-budget", action="store_true",
                   help="DANGEROUS: skip the epochs/val_n/train_n caps. "
                        "Use only for iter-1 feasibility tests.")
    p.add_argument("--docs-runs",
                   default="/cluster/maier/Agent4CT/docs/runs",
                   help="Path to docs/runs (on the cluster).")
    p.add_argument("--cfgs",
                   default="/cluster/maier/Agent4CT/agentic_cfgs",
                   help="Where to write the new iter-(N+1) config JSON.")
    return p.parse_args()


def coerce_value(v: str):
    """Parse 'true'/'false'/numbers/lists from CLI strings."""
    if v in ("true", "True"):  return True
    if v in ("false", "False"): return False
    try:    return int(v)
    except ValueError: pass
    try:    return float(v)
    except ValueError: pass
    if v.startswith("[") and v.endswith("]"):
        return [coerce_value(x.strip()) for x in v[1:-1].split(",")]
    return v


def main() -> int:
    args = parse_args()
    if len(args.knob) > 1:
        print(f"⚠️  WARNING: {len(args.knob)} knobs being changed in one "
              f"iter. Step 2 forbids multi-knob iters — the journal becomes "
              f"uninterpretable. Continuing anyway, but FIX THIS in iter-"
              f"{args.prev_iter+2} by reverting all but one.", file=sys.stderr)

    obs_path = (Path(args.docs_runs) / args.slug / "iterations"
                / f"iter-{args.prev_iter:04d}" / "observation.json")
    if not obs_path.exists():
        print(f"❌ {obs_path} not found. Did the prior iter actually land?",
              file=sys.stderr)
        return 1
    obs = json.loads(obs_path.read_text())
    prev_hr = obs.get("headroom", 0.0)
    prev_ssim = obs.get("val_score", 0.0)
    print(f"[propose] prior iter-{args.prev_iter}: hr={prev_hr:.4f} "
          f"ssim={prev_ssim:.4f}", flush=True)

    # Reconstruct prior config from rationale (best-effort) — the
    # caller should normally have the full config JSON already.
    # Here we just start from short-budget defaults + apply knob
    # overrides. The user is expected to have already inspected the
    # prior config (checklist item #1).
    cfg: dict = dict(SHORT_BUDGET_DEFAULTS)
    if args.ignore_short_budget:
        cfg.clear()
        print("[propose] ⚠️  --ignore-short-budget set; no epoch/val/train cap.",
              file=sys.stderr)

    for kv in args.knob:
        if "=" not in kv:
            print(f"❌ --knob must be KEY=VALUE, got {kv!r}", file=sys.stderr)
            return 1
        k, v = kv.split("=", 1)
        cfg[k.strip()] = coerce_value(v.strip())

    cfg["rationale"] = args.hypothesis
    cfg["prev_iter"]   = args.prev_iter
    cfg["prev_hr"]     = prev_hr
    cfg["prev_ssim"]   = prev_ssim

    next_iter = args.prev_iter + 1
    out_path = Path(args.cfgs) / f"{args.solver}_iter_{next_iter:02d}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(cfg, indent=2))

    print(f"[propose] wrote {out_path}")
    print(f"[propose] hypothesis: {args.hypothesis}")
    print(f"[propose] knobs: {args.knob}")
    print()
    print("[propose] Dispatch with:")
    print(f"  sbatch --export=ALL,SOLVER={args.solver},SLUG={args.slug},"
          f"ITER={next_iter},CFG_JSON={out_path} \\")
    print(f"         cluster/slurm/mayo_agentic_iter.sbatch")
    return 0


if __name__ == "__main__":
    sys.exit(main())
