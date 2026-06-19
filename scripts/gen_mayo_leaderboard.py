"""Regenerate the agentic-results table in docs/leaderboards/mayo_ldct.md from
the run dirs — best iter per solver (max SSIM, tie-break headroom), ranked.

Run after each agentic wave + rsync, before commit. Replaces the content between
the `<!-- AGENTIC_TABLE_START -->` / `<!-- AGENTIC_TABLE_END -->` markers so the
table always reflects the true best-per-solver from the data (no manual tracking,
no stale rows). Torch-free — runs on the laptop after rsyncing the run dirs.

Columns: Rank | Solver | Best iter | params (M) | SSIM | hr | PSNR (dB) | RMSE |
time (s) | Source | Comparison. PSNR/RMSE come from each iter's
`observation.json` (val_psnr / val_rmse); compute time is the per-iter wall
(`elapsed_s`) the agentic runner records.
"""
from __future__ import annotations
import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RUNS = REPO / "docs" / "runs"
LB = REPO / "docs" / "leaderboards" / "mayo_ldct.md"

# Pinned to the CURRENT Mayo campaign run-id. The 2026-06-14 rebuild
# (search-20260614-01) was purged in the 2026-06-19 reset (invalid val metric);
# the fresh search is search-20260619-01. NEVER widen this glob — it keeps the
# obsolete pre-rebuild agentic runs from leaking back in as extra rows.
RUNID = "search-20260619-01"

NAMES = {
    "dual-domain-supervised": "DD-UNet supervised L2",
    "itnet-v3": "ITNet v3",
    "itnet-v2": "ITNet v2",
    "itnet": "ITNet v1",
    "dual-domain-bilateral-supervised": "DD-BF supervised L2",
    "dual-domain-n2i": "DD-UNet N2I (per-image)",
    "dual-domain-bilateral-n2i": "DD-BF N2I (per-image)",
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
        return "—"
    return f"{pm:.3f}" if pm >= 0.001 else str(int(round(pm * 1e6)))


def fmt_psnr(p) -> str:
    return f"{p:.2f}" if isinstance(p, (int, float)) else "—"


def fmt_rmse(r) -> str:
    return f"{r:.2e}" if isinstance(r, (int, float)) else "—"


def fmt_time(t) -> str:
    return f"{t:.0f}" if isinstance(t, (int, float)) else "—"


# Authoritative param-count backstop for runs whose config is no longer on disk.
_SP = REPO / "docs" / "leaderboards" / "solver_params.json"
SOLVER_PARAMS = {k: v for k, v in json.loads(_SP.read_text()).items()
                 if not k.startswith("_")} if _SP.exists() else {}


def trainable_from_cfg(dash: str, cfg: dict):
    """Verified trainable-param formulas for the families that historically did
    not emit params_M. bilateral = 3*(proj_n_bf+img_n_bf) (3 per BF, default 1/1);
    wu_2015_trainable = wu_n_bands + 2 + 2*wu_n_outer. Returns None otherwise."""
    if dash in ("dual-domain-bilateral-supervised", "dual-domain-bilateral-n2i"):
        return 3 * (int(cfg.get("proj_n_bf", 1)) + int(cfg.get("img_n_bf", 1)))
    if dash == "wu-2015-trainable":
        return int(cfg.get("wu_n_bands", 0)) + 2 + 2 * int(cfg.get("wu_n_outer", 0))
    return None


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
    for d in sorted(RUNS.glob(f"mayo-ldct-claude-agentic-*-{RUNID}")):
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
        params = psnr = rmse = tsec = "—"
        obs = d / "iterations" / f"iter-{it:04d}" / "observation.json"
        if obs.exists():
            o = json.loads(obs.read_text())
            pm = o.get("params_M")
            if pm is not None:
                params = fmt_params(pm)
            else:                                  # solver omitted params_M: recompute
                cnt = trainable_from_cfg(dash, o.get("cfg_full") or {})
                if cnt is None:
                    cnt = SOLVER_PARAMS.get(d.name)
                params = str(cnt) if cnt is not None else "—"
            psnr = fmt_psnr(o.get("val_psnr"))
            rmse = fmt_rmse(o.get("val_rmse"))
            tsec = fmt_time(o.get("elapsed_s"))
            v = variant(o.get("cfg_full") or {})
            if v:
                var = f"iter-{it} ({v})"
        # Link the best iter's val (L277) comparison.png — always regenerated
        # by the solver at run time, so it stays in lock-step with the metric
        # after every rsync.
        img = f"../runs/{d.name}/iterations/iter-{it:04d}/comparison.png"
        res = f"../runs/{d.name}/results.tsv"
        rows.append((hr, ss, name, var, params, psnr, rmse, tsec, res, img))

    # Rank by HEADROOM (hr) — the canonical "fraction of the FBP→oracle gap
    # closed" metric, so the actual champion sits at #1. SSIM (shown alongside)
    # is the tiebreak and is also what selects the best iter within a run.
    rows.sort(key=lambda r: (-r[0], -r[1]))
    out = ["| Rank | Solver | Best iter | params (M) | SSIM | hr | PSNR (dB) | RMSE | time (s) | Source | Comparison |",
           "|---:|---|---|---:|---:|---:|---:|---:|---:|---|---|"]
    for i, (hr, ss, name, var, params, psnr, rmse, tsec, res, img) in enumerate(rows, 1):
        b = "**" if i == 1 else ""
        out.append(f"| {i} | {b}{name}{b} | {var} | {params} | {ss:.4f} | {hr:.4f} | "
                   f"{psnr} | {rmse} | {tsec} | [results]({res}) | [![{name}]({img})]({img}) |")
    if rows:
        c = rows[0]
        max_it = max((int(re.match(r"iter-(\d+)", r[3]).group(1)) for r in rows
                      if re.match(r"iter-(\d+)", r[3])), default=0)
        summary = (f"**🟢 LIVE — `{RUNID}` in progress** ({len(rows)} solvers onboarded, "
                   f"iters up to {max_it}/20). **Current champion by headroom: "
                   f"{c[2]} — hr {c[0]:.4f}, SSIM {c[1]:.4f}** ({c[3].split(' (')[0]}). "
                   f"Ranked by headroom; updated every wave.\n")
        table = summary + "\n".join(out)
    else:
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
