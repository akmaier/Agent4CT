"""Regenerate the agentic-results table in docs/leaderboards/mayo_ldct.md from
the run dirs — best iter per solver (max headroom, tie-break SSIM), ranked.

Run after each agentic wave + rsync, before commit. Replaces the content between
the `<!-- AGENTIC_TABLE_START -->` / `<!-- AGENTIC_TABLE_END -->` markers so the
table always reflects the true best-per-solver from the data (no manual tracking,
no stale rows). Torch-free — runs on the laptop after rsyncing the run dirs.
"""
from __future__ import annotations
import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RUNS = REPO / "docs" / "runs"
LB = REPO / "docs" / "leaderboards" / "mayo_ldct.md"

NAMES = {
    "dual-domain-supervised": "DD-UNet supervised L2",
    "itnet-v3": "ITNet v3",
    "itnet-v2": "ITNet v2",
    "itnet": "ITNet v1",
    "dual-domain-bilateral-supervised": "DD-BF supervised L2",
    "dual-domain-n2i": "DD-UNet N2I",
    "dual-domain-bilateral-n2i": "DD-BF N2I",
    "learned-primal-dual": "Learned Primal-Dual",
    "hammernik-2017": "Hammernik VN (2017)",
    "hammernik-vn": "Hammernik VN (MRI port)",
    "uswin": "U-Swin",
    "wu-2015-trainable": "Wu 2015 trainable",
    "ram": "RAM (zero-shot)",
    "naf": "NAF",
    "r2gaussian": "R2-Gaussian",
    "tv-iterative": "TV-iterative",
    "tv-iterative-supervised": "TV-iterative (unrolled)",
    "diff-recon-dcstep-constrained-mayo-v4": "Diffusion (constrained DPS+DC)",
    "diff-recon-dcstep-unconstrained-mayo-v4": "Diffusion (unconstrained DPS)",
}

# cfg keys that are common plumbing / geometry — omit from the "variant" string.
BOILER = {
    "train_n", "val_n", "batch_size", "val_chunk", "grad_clip", "seed",
    "lambda_neg", "max_train_s", "rationale", "noise_i0", "noise_sigma_e",
    "optimizer", "dataset_kind", "image_size", "pixel_spacing", "n_angles",
    "n_det", "det_spacing", "sod", "sdd", "display_min", "display_max",
    # diffusion_recon plumbing (keep only the tuned knobs recon_eta / recon_dcstep_every)
    "recon_ckpt", "recon_init", "recon_mode", "recon_sample_steps",
    "recon_eta_clamp", "recon_dcstep_n_cg", "recon_dcstep_warmup", "recon_dcstep_relax",
}


def variant(cfg: dict) -> str:
    parts = [f"{k}={cfg[k]}" for k in sorted(cfg)
             if k not in BOILER and not k.startswith("display")]
    return ", ".join(parts[:6])


def fmt_params(pm) -> str:
    if pm is None:
        return ""
    return f"{pm:.3f} M" if pm >= 0.001 else str(int(round(pm * 1e6)))


def best_row(slug_dir: Path):
    tsv = slug_dir / "results.tsv"
    if not tsv.exists():
        return None
    best = None
    for line in tsv.read_text().splitlines()[1:]:
        c = line.split("\t")
        try:
            it, ss, hr = int(c[0]), float(c[2]), float(c[3])
        except (ValueError, IndexError):
            continue
        # Select the best iter by SSIM (stable), not hr: the in-solver LD-FBP
        # baseline carries tiny GPU-atomic backprojection noise, so hr can flip
        # a lower-SSIM iter above a higher one. SSIM is the primary metric and
        # gives the same cross-solver order (all share the val baseline).
        if best is None or (ss, hr) > (best[2], best[1]):
            best = (it, hr, ss)
    return best


def main() -> int:
    rows = []
    # Pinned to the CURRENT rebuild run-id (2026-06-14) so the obsolete
    # pre-rebuild agentic runs (20260603/20260610/phase4) never leak back in.
    for d in sorted(RUNS.glob("mayo-ldct-claude-agentic-*-search-20260614-01")):
        m = re.match(r"mayo-ldct-claude-agentic-(.+)-search-\d{8}-\d+$", d.name)
        if not m:
            continue
        dash = m.group(1)
        name = NAMES.get(dash, dash)
        b = best_row(d)
        if not b:
            continue
        it, hr, ss = b
        var = f"iter-{it}"
        params = ""
        obs = d / "iterations" / f"iter-{it:04d}" / "observation.json"
        if obs.exists():
            o = json.loads(obs.read_text())
            params = fmt_params(o.get("params_M"))
            v = variant(o.get("cfg_full") or {})
            if v:
                var = f"iter-{it} ({v})"
        # Link the best iter's val (L277) comparison.png — always regenerated
        # by the solver at run time, so it stays in lock-step with the metric
        # after every rsync. (The held-out-test montages (test_showcase.png)
        # were a nicer view but went stale at 2026-06-15 and require a slow
        # retrain to refresh; regenerate them FOV-cropped as a follow-up.)
        img = f"../runs/{d.name}/iterations/iter-{it:04d}/comparison.png"
        res = f"../runs/{d.name}/results.tsv"
        rows.append((hr, ss, name, var, params, res, img))

    rows.sort(key=lambda r: (-r[1], -r[0]))   # by SSIM (r[1]), hr tiebreak
    out = ["| Rank | Solver | Best iter | SSIM | hr | params | Source | Comparison |",
           "|---:|---|---|---:|---:|---:|---|---|"]
    for i, (hr, ss, name, var, params, res, img) in enumerate(rows, 1):
        b = "**" if i == 1 else ""
        out.append(f"| {i} | {b}{name}{b} | {var} | {ss:.4f} | {hr:.4f} | "
                   f"{params} | [results]({res}) | [![{name}]({img})]({img}) |")
    table = "\n".join(out)

    md = LB.read_text()
    if "<!-- AGENTIC_TABLE_START -->" not in md:
        raise SystemExit("AGENTIC_TABLE markers not found in mayo_ldct.md")
    md = re.sub(r"<!-- AGENTIC_TABLE_START -->.*?<!-- AGENTIC_TABLE_END -->",
                "<!-- AGENTIC_TABLE_START -->\n" + table + "\n<!-- AGENTIC_TABLE_END -->",
                md, flags=re.S)
    LB.write_text(md)
    if rows:
        print(f"[leaderboard] {len(rows)} solvers; champion {rows[0][2]} "
              f"SSIM {rows[0][1]:.4f} (hr {rows[0][0]:.4f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
