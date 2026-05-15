#!/usr/bin/env python3
"""Standalone script to record demo reference results without importing torch."""
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# We MUST NOT import ddssl_ldct because it imports torch
DOCS_RUNS = Path(__file__).resolve().parents[1] / "docs" / "runs"

def utc_now_iso():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

def make_slug(prefix):
    today = datetime.now(timezone.utc).strftime("%Y%m%d")
    seq = 1
    # Find existing sequence number
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
        "slug": slug,
        "challenge": challenge,
        "slug_prefix": slug_prefix,
        "started": utc_now_iso(),
        "agent": agent,
        "model": model,
        "status": "running",
        "notes": notes,
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    (run_dir / "results.tsv").write_text(
        "iter\tcommit\tval_score\theadroom\tstatus\tchange_class\tagent\tmodel\trationale\n"
    )
    (run_dir / "stages.tsv").write_text(
        "iter\tstage_val_score\tstage_headroom\tgap\tverdict\tnotes\n"
    )
    print(f"Created run: {slug}")
    return slug, run_dir

def record_iteration(run_dir, iter_n, kind, val_score, headroom, params_M, train_n, rationale, advice):
    iter_dir = run_dir / "iterations" / f"iter-{iter_n:04d}"
    iter_dir.mkdir(parents=True, exist_ok=True)
    
    obs = {
        "ts": utc_now_iso(),
        "run_id": run_dir.name,
        "iter": iter_n,
        "challenge": "dl_sparse_view",
        "change_class": "reference",
        "rationale": rationale,
        "val_score": val_score,
        "headroom": headroom,
        "kept": True,
        "status": "keep",
        "params_M": params_M,
        "train_n": train_n,
        "agent": "reference",
        "model": kind,
        "advice_for_others": advice,
    }
    (iter_dir / "observation.json").write_text(json.dumps(obs, indent=2))
    
    # Append to results.tsv
    with (run_dir / "results.tsv").open("a") as f:
        f.write(f"{iter_n}\t\t{val_score:.6g}\t{headroom:.6g}\tkeep\treference\treference\t{kind}\t{rationale.replace(chr(9), ' ')}\n")
    
    # Append to observations.jsonl
    scratch = DOCS_RUNS / "observations.jsonl"
    with scratch.open("a") as f:
        f.write(json.dumps(obs) + "\n")
    
    print(f"Recorded {kind} as iter {iter_n}: headroom={headroom:.4f}")

def update_index():
    """Rebuild runs-index.json."""
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
        
        m_short = run_dir.name.split("-")[-2] + "-" + run_dir.name.split("-")[-1] if "-" in run_dir.name else run_dir.name
        runs.append({
            "slug": run_dir.name,
            "short_id": m_short,
            "challenge": manifest.get("challenge"),
            "started": manifest.get("started"),
            "status": manifest.get("status", "running"),
            "n_iterations": n_iter,
            "best_score": best_score,
            "best_headroom": best_hr,
            "agent": manifest.get("agent"),
            "model": manifest.get("model"),
        })
    
    index = {
        "schema_version": 1,
        "updated": utc_now_iso(),
        "runs": runs,
    }
    idx_path.write_text(json.dumps(index, indent=2))
    print(f"Updated index with {len(runs)} runs")

def main():
    slug, run_dir = create_run(
        challenge="dl_sparse_view",
        slug_prefix="demo-dl-reference",
        agent="reference",
        model="demo-implementations",
        notes="Reference reconstructions on synthetic phantoms"
    )
    
    # FBP baseline
    record_iteration(
        run_dir, 1, "FBP-baseline",
        val_score=0.4454, headroom=0.0, params_M=0.0, train_n=0,
        rationale="Pure PYRO-NN FBP baseline (no learning). Establishes headroom=0 reference point.",
        advice="FBP baseline on synthetic phantoms: SSIM=0.445, RMSE=0.0139. Any learning must beat this.",
    )
    
    # Dual Domain
    record_iteration(
        run_dir, 2, "Dual-Domain",
        val_score=0.3055, headroom=0.5831, params_M=0.466, train_n=400,
        rationale="Dual-Domain Denoising (Wagner 2023): U-Net(c=16) in projection + image domain, Noise2Inverse self-supervised training.",
        advice="Dual-Domain N2I achieves headroom=0.583 on synthetic phantoms. Strong improvement over FBP. Uses dual-domain pipeline with split projections.",
    )
    
    # ItNet
    record_iteration(
        run_dir, 3, "ItNet",
        val_score=0.1369, headroom=0.0, params_M=0.233, train_n=400,
        rationale="ItNet-style iterative recon with data consistency (Sidky 2022 winner approach). Pretrained U-Net + 5 DC iterations. Underperforms on synthetic data - needs investigation.",
        advice="Naive ItNet: headroom=0 on synthetic phantoms. Likely issues: (1) DC gradient step too aggressive, (2) pretraining on noisy FBP poor, (3) needs proper geometry estimation. Real challenge data may behave differently.",
    )
    
    update_index()
    print(f"\nDone! Run recorded: {slug}")
    print("View at: https://akmaier.github.io/Agent4CT/dashboard.html")

if __name__ == "__main__":
    main()
