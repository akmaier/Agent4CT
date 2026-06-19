"""Build the dashboard data layer: per-dataset run indices + a per-dataset,
capped scratchpad — so the live dashboard loads ONE small index for the dataset
being viewed instead of fetching `results.tsv` for all 100+ runs and rendering
1000+ scratchpad cards.

Outputs under `docs/runs/`:
  index/datasets.json        per-dataset summary for the landing page
  index/<challenge>.json     one dataset's runs (summary + precomputed
                             best-headroom `curve` + best-iter val/test images)
  scratch/<challenge>.jsonl  that dataset's observations, newest-first, capped
  runs-index.json            legacy single index (back-compat; no curves)

Torch-free — runs locally after rsyncing the cluster's `docs/runs/<slug>/` dirs.
The per-run `challenge` is derived from the slug prefix (the agentic harness
hardcodes manifest `challenge="dl_sparse_view"`), so Mayo/breast/demo runs land
in the right bucket without patching manifests.
"""
from __future__ import annotations
import json
import math
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DOCS_RUNS = REPO / "docs" / "runs"
SCRATCH_CAP = 200          # most-recent observations kept per dataset

# Order matters: first matching prefix wins. The trailing ("demo-", demo_dl)
# is a catch-all for the demo_dl experiment families (demo-fair-*,
# demo-intensity-calibrated-*) that don't carry the "demo-dl-" prefix.
_PREFIX_TO_CHALLENGE = [
    ("mayo-ldct", "mayo_ldct"),
    ("breast-ct", "breast_ct"),
    ("dl-sparse-view", "dl_sparse_view"),
    ("dl-spectral", "dl_spectral"),
    ("ct-mar", "ct_mar"),
    ("truect", "truect"),
    ("demo-dl", "demo_dl"),
    ("demo-", "demo_dl"),
]
DATASET_LABELS = {
    "mayo_ldct": "Mayo-LDCT", "breast_ct": "Breast-CT", "demo_dl": "Demo-DL",
    "dl_sparse_view": "DL-Sparse-View", "dl_spectral": "DL-Spectral",
    "ct_mar": "CT-MAR", "truect": "TrueCT",
}


def challenge_from_slug(slug: str, fallback: str | None = None) -> str | None:
    for prefix, ch in _PREFIX_TO_CHALLENGE:
        if (slug or "").startswith(prefix):
            return ch
    return fallback


def display_name(slug: str) -> str:
    """Human-friendly run label = the solver name, recovered from the slug by
    stripping the trailing run-id, the `-search` tag, the harness infix
    (`claude-agentic-` / `calibrated-tpe-` / `calibrated-`), and the dataset
    dash-prefix. `mayo-ldct-claude-agentic-uswin-search-20260614-01` -> `uswin`;
    `breast-ct-calibrated-tpe-lpd-search-20260524-01` -> `lpd`."""
    import re
    s = re.sub(r"-\d{8}-\d{2}$", "", slug)
    s = re.sub(r"-search$", "", s)
    for infix in ("claude-agentic-", "calibrated-tpe-", "calibrated-"):
        i = s.find(infix)
        if i >= 0:
            s = s[i + len(infix):]
            break
    for dash, _ in _PREFIX_TO_CHALLENGE:
        if s.startswith(dash):
            s = s[len(dash):]
            break
    return s or slug


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_results(path: Path) -> list[dict]:
    """results.tsv -> rows of {iter, val_score, headroom, status}.
    Columns: 0 iter | 1 commit | 2 val_score | 3 headroom | 4 status."""
    rows = []
    for line in path.read_text().splitlines()[1:]:
        c = line.split("\t")
        if len(c) < 4:
            continue
        try:
            it, vs, hr = int(c[0]), float(c[2]), float(c[3])
        except (ValueError, IndexError):
            continue
        rows.append({"iter": it, "val_score": vs, "headroom": hr,
                     "status": (c[4].strip() if len(c) > 4 else "")})
    return rows


def summarize_run(run_dir: Path) -> dict:
    manifest = json.loads((run_dir / "manifest.json").read_text())
    slug = run_dir.name
    ch = challenge_from_slug(slug, manifest.get("challenge"))
    res = run_dir / "results.tsv"
    rows = parse_results(res) if res.exists() else []

    finite = lambda xs: [x for x in xs if math.isfinite(x)]
    best_score = max(finite(r["val_score"] for r in rows), default=None)
    best_hr = max(finite(r["headroom"] for r in rows), default=None)
    # rank_headroom drives the CHAMPION pick (datasets.json). Unlike best_hr it
    # excludes discarded rows and non-positive / non-finite headroom — a
    # below-baseline solver (status=discard / hr<=0) must NEVER be crowned. This
    # is the ONE canonical ranking (headroom, val_ssim tiebreak) that the
    # leaderboard uses too, so the dashboard and the board agree on the winner.
    # (Phase 0 of the result-register refactor — result_register_refactor_plan.md §6.)
    # val_score >= 0.1 sanity floor: the legacy uncalibrated `demo-dl-*` runs
    # report a DIFFERENT scoring rule (val_score is a negative loss, headroom
    # ~0.99) and are not comparable to the calibrated SSIM convention — without
    # the floor a hallucinated near-1.0 headroom from an uncalibrated DDPM run
    # crowns demo_dl. Phase 1's allowlist (CURRENT_RUNIDS.json) supersedes this.
    rankable = [r for r in rows
                if (r["status"] or "").strip().lower() != "discard"
                and math.isfinite(r["headroom"]) and r["headroom"] > 0
                and math.isfinite(r["val_score"]) and r["val_score"] >= 0.1]
    rank_headroom = max((r["headroom"] for r in rankable), default=None)

    # Compact running-best-headroom curve for the overview chart: [[iter, best], …]
    curve, run = [], -math.inf
    for r in rows:
        if math.isfinite(r["headroom"]) and r["headroom"] > run:
            run = r["headroom"]
        curve.append([r["iter"], None if run == -math.inf else round(run, 4)])

    # best_iter = max val_score (the metric). val_image = the highest-val_score
    # iter that ACTUALLY has a comparison.png — the harness only saved an image
    # on some iters (new-best / every-N), so the metric-best iter often has none.
    best_iter, val_image, test_image = None, None, None
    if rows:
        best_iter = max(rows, key=lambda r: r["val_score"]
                        if math.isfinite(r["val_score"]) else -math.inf)["iter"]
        # Mayo agentic runs show the LAST iteration's val figure (final state /
        # iter-20 mandate, per user request); other datasets (TPE searches) keep
        # the metric-best iter. Walk in the chosen order, take the first iter that
        # actually saved a comparison.png (the harness skips images on some iters).
        if ch == "mayo_ldct":
            order = sorted(rows, key=lambda r: r["iter"], reverse=True)
        else:
            order = sorted(rows, key=lambda r: r["val_score"]
                           if math.isfinite(r["val_score"]) else -math.inf, reverse=True)
        for r in order:
            it = r["iter"]
            if (run_dir / "iterations" / f"iter-{it:04d}" / "comparison.png").exists():
                val_image = f"runs/{slug}/iterations/iter-{it:04d}/comparison.png"
                break
    # The 6-patient val+test montage (valtest_showcase.png: L277 + 5 test centrals,
    # GT|FBP|recon|diff, full 512² no-FOV) is the canonical result image when
    # present — prefer it over the per-iter val comparison.
    if (run_dir / "valtest_showcase.png").exists():
        val_image = f"runs/{slug}/valtest_showcase.png"
    if (run_dir / "test_showcase.png").exists():
        test_image = f"runs/{slug}/test_showcase.png"

    parts = slug.split("-")
    short = parts[-2] + "-" + parts[-1] if len(parts) >= 2 else slug
    return {
        "slug": slug, "short_id": short, "name": display_name(slug), "challenge": ch,
        "started": manifest.get("started"),
        "status": manifest.get("status", "running"),
        "n_iterations": len(rows), "best_score": best_score,
        "best_headroom": best_hr, "rank_headroom": rank_headroom,
        "best_iter": best_iter,
        "agent": manifest.get("agent"), "model": manifest.get("model"),
        "curve": curve, "val_image": val_image, "test_image": test_image,
    }


def split_scratchpad() -> dict[str, list]:
    """Bucket observations.jsonl by challenge_from_slug(run_id), newest-first,
    capped per dataset. The file's `challenge` field is unreliable (hardcoded),
    so the run_id slug is authoritative."""
    src = DOCS_RUNS / "observations.jsonl"
    if not src.exists():
        return {}
    buckets: dict[str, list] = {}
    for line in src.read_text().splitlines():
        if not line.strip():
            continue
        try:
            o = json.loads(line)
        except Exception:
            continue
        ch = challenge_from_slug(o.get("run_id", ""), None)
        if ch is None:
            continue
        buckets.setdefault(ch, []).append(o)
    return {ch: list(reversed(entries))[:SCRATCH_CAP]
            for ch, entries in buckets.items()}


def main() -> int:
    runs = []
    for run_dir in sorted(DOCS_RUNS.iterdir()):
        if not run_dir.is_dir() or not (run_dir / "manifest.json").exists():
            continue
        try:
            runs.append(summarize_run(run_dir))
        except Exception as e:
            print(f"[index] skip {run_dir.name}: {e}")

    by_ds: dict[str, list] = {}
    for r in runs:
        by_ds.setdefault(r["challenge"] or "other", []).append(r)

    idx_dir = DOCS_RUNS / "index"; idx_dir.mkdir(exist_ok=True)
    scr_dir = DOCS_RUNS / "scratch"; scr_dir.mkdir(exist_ok=True)

    datasets = []
    for ch in sorted(by_ds):
        rs = sorted(by_ds[ch], key=lambda r: (r["started"] or ""), reverse=True)
        (idx_dir / f"{ch}.json").write_text(json.dumps(
            {"schema_version": 2, "challenge": ch,
             "label": DATASET_LABELS.get(ch, ch), "updated": utc_now_iso(),
             "runs": rs}, indent=1, allow_nan=False))
        # Champion = single canonical ranking: headroom (val_ssim tiebreak),
        # excluding discard/non-finite/hr<=0 (those have rank_headroom=None ->
        # sort last). Agrees with the leaderboard's rank-1 by construction, so
        # the dashboard never crowns a below-baseline hr=0 solver again (R1 fix).
        champ = max(rs, key=lambda r: (
            r["rank_headroom"] if r["rank_headroom"] is not None else -math.inf,
            r["best_score"] if r["best_score"] is not None else -math.inf))
        # champion_score is the headroom that won it (None if no run cleared
        # baseline); champion_ssim keeps the SSIM visible alongside.
        datasets.append({
            "challenge": ch, "label": DATASET_LABELS.get(ch, ch),
            "n_runs": len(rs), "n_iterations": sum(r["n_iterations"] for r in rs),
            "champion_slug": champ["slug"],
            "champion_score": champ["rank_headroom"],
            "champion_ssim": champ["best_score"],
            "thumbnail": champ.get("val_image"),
        })
    datasets.sort(key=lambda d: -d["n_runs"])
    (idx_dir / "datasets.json").write_text(json.dumps(
        {"schema_version": 2, "updated": utc_now_iso(), "datasets": datasets},
        indent=1, allow_nan=False))

    scratch = split_scratchpad()
    for ch, entries in scratch.items():
        (scr_dir / f"{ch}.jsonl").write_text(
            "\n".join(json.dumps(e) for e in entries) + "\n")

    # Legacy single index (back-compat) — drop the per-run curve to keep it small.
    legacy = [{k: v for k, v in r.items() if k != "curve"} for r in runs]
    (DOCS_RUNS / "runs-index.json").write_text(json.dumps(
        {"schema_version": 1, "updated": utc_now_iso(), "runs": legacy},
        indent=2, allow_nan=False))

    ds_summary = ", ".join(f"{d['challenge']}:{d['n_runs']}" for d in datasets)
    scr_summary = ", ".join(f"{k}:{len(v)}" for k, v in scratch.items())
    print(f"[index] {len(runs)} runs -> {len(datasets)} datasets "
          f"({ds_summary}); scratch {{{scr_summary}}}")
    return len(runs)


if __name__ == "__main__":
    main()
