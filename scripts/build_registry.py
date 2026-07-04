"""build_registry.py — the ONE deterministic aggregator (no LLM, torch-free).

Walks the allowlisted per-iter `observation.json` records (allowlist =
docs/runs/CURRENT_RUNIDS.json) and materializes every downstream view:

  docs/runs/index/
    registry.jsonl        flat canonical rollup, 1 line / run-iter (git-diffable)
    registry.meta.json    schema_version, builder_git_sha, built_at, content_hash,
                          allowlist_sha   <- drives the staleness gate
    <challenge>.json      per-dataset materialized view (dashboard fetches this)
    leaderboard.json      the ONE ranked surface per dataset, ALL solvers
    datasets.json         landing summary (champion = leaderboard rank-1, by hr)
    scratch/<challenge>.jsonl   capped recent observations for the advice cards
  README REGISTRY_TABLE   the <!--REGISTRY_TABLE-->…<!--/REGISTRY_TABLE--> block
                          in README.md regenerated in the same run (so the no-JS
                          GitHub front page cannot drift).

Canonical ranking is per-dataset (registry_lib.metric_basis): test datasets (Mayo)
rank by test_hr_mean (test_ssim tiebreak, n=5 patients), others by val headroom
(val_ssim tiebreak). status=discard / pending-test (no final.json) /
non-finite / hr<=0 are EXCLUDED from the rank but still emitted (excluded_reason
set) so every solver row is rendered — top-N is structurally unexpressible
(no slicing anywhere). No field from observation.json is dropped.

Run: python3 scripts/build_registry.py   (then validate_registry.py gates it).
See result_register_refactor_plan.md §3-§4.
"""
from __future__ import annotations

import hashlib
import json
import math
import subprocess
from pathlib import Path

import registry_lib as R

REPO = R.REPO
DOCS_RUNS = R.DOCS_RUNS
IDX = DOCS_RUNS / "index"
SCR = DOCS_RUNS / "scratch"
ALLOWLIST = DOCS_RUNS / "CURRENT_RUNIDS.json"
README = REPO / "README.md"
SOLVER_PARAMS_BACKSTOP = REPO / "docs" / "leaderboards" / "solver_params.json"
SCRATCH_CAP = 200

# Fields copied verbatim from observation.json into each registry line's metrics.
# headroom_std is the per-case val-headroom spread (populated by
# scripts/rescore_val_std.py from the saved recon_raw.npz) so the val board can
# render hr as mean ± std alongside the SSIM/PSNR/RMSE per-slice stds.
_METRIC_KEYS = ("val_ssim", "headroom", "val_psnr", "val_rmse",
                "val_ssim_std", "val_psnr_std", "val_rmse_std", "headroom_std")


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=REPO,
            text=True).strip()
    except Exception:
        return "unknown"


def _load_backstop() -> dict:
    if not SOLVER_PARAMS_BACKSTOP.exists():
        return {}
    return {k: v for k, v in R.load_json(SOLVER_PARAMS_BACKSTOP).items()
            if not k.startswith("_")}


def _images_for(slug: str, it: int, obs: dict) -> dict:
    """Resolve the per-iter comparison + run-level showcase image paths that
    actually exist on disk. Paths are repo-relative-to-docs (the `runs/...` form
    the dashboard/JS fetch)."""
    run_dir = DOCS_RUNS / slug
    images = {"comparison": None, "valtest_showcase": None, "test_showcase": None}
    cmp_rel = obs.get("comparison_image") or f"runs/{slug}/iterations/iter-{it:04d}/comparison.png"
    if (DOCS_RUNS / cmp_rel.replace("runs/", "", 1)).exists():
        images["comparison"] = cmp_rel
    if (run_dir / "valtest_showcase.png").exists():
        images["valtest_showcase"] = f"runs/{slug}/valtest_showcase.png"
    if (run_dir / "test_showcase.png").exists():
        images["test_showcase"] = f"runs/{slug}/test_showcase.png"
    return images


def _iter_observations(slug: str):
    """Yield (iter_int, obs_dict, obs_path) for every observation.json under a
    run dir, in iteration order."""
    iters_dir = DOCS_RUNS / slug / "iterations"
    if not iters_dir.is_dir():
        return
    for d in sorted(iters_dir.iterdir()):
        op = d / "observation.json"
        if not op.exists():
            continue
        try:
            obs = R.load_json(op)
        except Exception as e:
            print(f"[registry] skip {op}: {e}")
            continue
        try:
            it = int(obs.get("iter", int(d.name.split("-")[-1])))
        except Exception:
            continue
        yield it, obs, op


def build_registry_lines(allow: dict, backstop: dict):
    """-> (lines, exclude_iters_by_run). One line per allowlisted run-iter, in a
    stable order (challenge, slug, iter)."""
    lines = []
    for ch in ("mayo_ldct", "breast_ct", "demo_dl"):
        ds = allow["datasets"].get(ch, {})
        excl = ds.get("exclude_iters", {})
        for slug in ds.get("run_ids", []):
            key = R.solver_key(slug)
            for it, obs, op in _iter_observations(slug):
                if it in (excl.get(slug) or []):
                    continue
                pm, psrc = R.resolve_params_M(key, obs, backstop, slug)
                metrics = {k: obs.get(k) for k in _METRIC_KEYS}
                line = {
                    "schema_version": R.SCHEMA_VERSION,
                    "run_id": slug,
                    "challenge": ch,                       # from slug, NOT manifest
                    "campaign": R.campaign_from_slug(slug),
                    "solver_key": key,
                    "solver_name": R.display_name(slug),
                    "iter": it,
                    "ts": obs.get("ts"),
                    "git_commit": obs.get("commit") or obs.get("git_commit"),
                    "metrics": metrics,
                    "params_M": pm,
                    "params_source": psrc,
                    "runtime": {"elapsed_s": obs.get("elapsed_s")},
                    "split": {"train_n": obs.get("train_n"), "val_n": obs.get("val_n")},
                    "status": obs.get("status"),
                    "kept": obs.get("kept"),
                    "change_class": obs.get("change_class"),
                    "agent": obs.get("agent"),
                    "model": obs.get("model"),
                    "rationale": obs.get("rationale"),
                    "advice_for_others": obs.get("advice_for_others"),
                    "cfg_full": obs.get("cfg_full"),
                    "images": _images_for(slug, it, obs),
                    "obs_path": str(op.relative_to(DOCS_RUNS)),
                    "obs_sha1": hashlib.sha1(op.read_bytes()).hexdigest(),
                }
                lines.append(line)
    lines.sort(key=lambda l: (l["challenge"], l["run_id"], l["iter"]))
    return lines


# Phase 1B: per-patient TEST-set aggregates (mean ± std over the 5 held-out
# Wagner test patients), written by scripts/score_mayo_testset.py to
# docs/runs/<slug>/final.json. Surfaced on the leaderboard row alongside the
# val (L277) metrics. Null everywhere until a run's final.json exists, so this
# is graceful for the whole inventory before the test-set jobs run.
_TESTSET_KEYS = ("test_hr_mean", "test_hr_std", "test_ssim_mean", "test_ssim_std",
                 "test_psnr_mean", "test_psnr_std", "test_rmse_mean", "test_rmse_std")


def _testset_aggregates(slug: str) -> dict:
    """Read docs/runs/<slug>/final.json and return the test_* aggregate fields
    (None for each if no final.json yet, or it's malformed / incomplete). The
    additions are deterministic from committed final.json content, so the
    registry staleness gate (fresh-rebuild vs committed content_hash) still
    passes."""
    out = {k: None for k in _TESTSET_KEYS}
    fp = DOCS_RUNS / slug / "final.json"
    if not fp.exists():
        return out
    try:
        obj = R.load_json(fp)
    except Exception as e:
        print(f"[registry] skip final.json {fp}: {e}")
        return out
    for k in _TESTSET_KEYS:
        v = obj.get(k)
        out[k] = v if R.finite(v) else None
    return out


_TEST_PATIENTS = ("L014", "L056", "L058", "L075", "L123")

# Held-out TEST-set size per test-ranked dataset, for the "(test, n=…)" basis tag
# (Mayo = 5 Wagner patients; breast = 200 i.i.d. held-out cases, paper §5.0).
_TEST_N_BY_CHALLENGE = {"mayo_ldct": 5, "breast_ct": 200}


def _itertest_base(slug: str) -> Path:
    """Per-iter TEST-sweep namespace for a run. Mayo param-efficient (code-evolving)
    was scored into pe-iter-testeval/; every other solver (Mayo or breast) into
    <slug>-itertest/ by scripts/score_mayo_alliters.py / score_breast_alliters.py."""
    return DOCS_RUNS / ("pe-iter-testeval" if "param-efficient" in slug
                        else f"{slug}-itertest")


def _test_final_complete(obj: dict) -> bool:
    """A per-iter TEST final.json is complete when its whole held-out set scored.
    Mayo: all 5 patients present in the `patients` dict. Breast (i.i.d. cases, no
    patients): the `complete` flag its worker sets (breast_testset_final_v1). Both
    then require a finite test_hr_mean (checked by the caller)."""
    if "patients" in obj:                       # Mayo per-patient final.json
        pats = obj.get("patients") or {}
        return all(pats.get(p) is not None for p in _TEST_PATIENTS)
    return bool(obj.get("complete"))            # breast per-case final.json


def _test_best_iter(slug: str):
    """Pick a run's leaderboard iter by MAX test_hr_mean (test_ssim_mean tiebreak)
    over the per-iter TEST-sweep aggregates. Complete iters only, so incomplete /
    refused iters are skipped. Returns (iter:int, aggr:dict over _TESTSET_KEYS) or
    None if the run has no complete sweep result (then the caller falls back to the
    val-headroom iter)."""
    base = _itertest_base(slug)
    if not base.is_dir():
        return None
    best = None
    for fp in sorted(base.glob("iter-*/final.json")):
        try:
            obj = R.load_json(fp)
        except Exception:
            continue
        if not _test_final_complete(obj):
            continue
        hrv = obj.get("test_hr_mean")
        if not R.finite(hrv):
            continue
        try:
            it = int(fp.parent.name.split("-")[-1])
        except Exception:
            continue
        ssv = obj.get("test_ssim_mean")
        key = (hrv, ssv if R.finite(ssv) else -math.inf)
        if best is None or key > best[0]:
            best = (key, it, {k: (obj.get(k) if R.finite(obj.get(k)) else None)
                              for k in _TESTSET_KEYS})
    return (best[1], best[2]) if best else None


def best_iter_row(slug_lines: list[dict]) -> dict | None:
    """Pick the leaderboard row for ONE run. On a TEST-ranked dataset (Mayo) the
    representing iter is the one with the MAX per-patient test headroom over the
    full per-iter TEST sweep (score_mayo_alliters.py) — the final leaderboard is
    the best test result each solver reached. On a val dataset (demo_dl/breast_ct,
    no held-out test set) it stays the best-by-val-headroom iter that has a
    comparison image. Returns a leaderboard row dict."""
    if not slug_lines:
        return None

    def hr(l):
        h = l["metrics"].get("headroom")
        return h if R.finite(h) else -math.inf

    def ss(l):
        s = l["metrics"].get("val_ssim")
        return s if R.finite(s) else -math.inf

    run_id = slug_lines[0]["run_id"]
    ch = R.challenge_from_slug(run_id)
    is_test = R.metric_basis(ch) == "test"

    test_aggr = {k: None for k in _TESTSET_KEYS}
    has_final = False
    best = None
    if is_test:
        tb = _test_best_iter(run_id)
        if tb is not None:
            tb_iter, test_aggr = tb
            has_final = True
            best = next((l for l in slug_lines if l["iter"] == tb_iter), None)
    if best is None:
        # val datasets, OR a test run with no usable sweep result: best-by-val-
        # headroom iter that has a comparison image (imageless fallback so the
        # solver still shows).
        with_img = [l for l in slug_lines if l["images"].get("comparison")]
        pool = with_img or slug_lines
        best = max(pool, key=lambda l: (hr(l), ss(l)))
        if is_test and not has_final:      # legacy single-final.json (pre-sweep)
            test_aggr = _testset_aggregates(run_id)
            has_final = (DOCS_RUNS / run_id / "final.json").exists()

    m = best["metrics"]
    row = {
        "solver_key": best["solver_key"],
        "solver_name": best["solver_name"],
        "run_id": best["run_id"],
        "best_iter": best["iter"],
        "params_M": best["params_M"],
        "params_source": best["params_source"],
        "val_ssim": m.get("val_ssim"),
        "headroom": m.get("headroom"),
        "val_psnr": m.get("val_psnr"),
        "val_rmse": m.get("val_rmse"),
        "val_ssim_std": m.get("val_ssim_std"),
        "val_psnr_std": m.get("val_psnr_std"),
        "val_rmse_std": m.get("val_rmse_std"),
        "headroom_std": m.get("headroom_std"),
        "elapsed_s": best["runtime"].get("elapsed_s"),
        "image": best["images"].get("comparison"),
    }
    # Fold in the per-patient test-set mean±std for the CHOSEN iter: on a test
    # dataset that is the test-best iter's aggregate (computed above); on a val
    # dataset it is empty (None). `has_final` (set above) marks whether a complete
    # test result exists, so a run with no sweep result is 'pending-test' (dimmed)
    # rather than val-fallback-ranked.
    row.update(test_aggr)
    row["metric_basis"] = R.metric_basis(ch)
    # On a TEST-ranked dataset (Mayo, held-out test set) the reported metrics are
    # the per-patient TEST mean±std (folded in above). Validation (L277) is the
    # search-time signal, NOT a result — drop the val_* display fields entirely so
    # no validation number can surface anywhere on a test board. Datasets WITHOUT a
    # held-out test set (demo_dl, breast_ct) keep their native val_* metrics: those
    # legitimately ARE their reported results.
    if row["metric_basis"] == "test":
        for _k in ("val_ssim", "headroom", "val_psnr", "val_rmse",
                   "val_ssim_std", "val_psnr_std", "val_rmse_std", "headroom_std"):
            row[_k] = None
    rm, tb, reason = R.rank_fields(row, best["status"], ch, has_final)
    row["rank_metric"] = rm
    row["rank_tiebreak"] = tb
    row["excluded_reason"] = reason
    return row


def build_leaderboard(ch_lines: list[dict], ch: str | None = None) -> dict:
    """All solvers for a dataset -> one ranked board (best iter per RUN).
    Rankable first (rank_metric desc, ssim tiebreak), excluded dimmed below.
    Test-ranked datasets (Mayo) rank by test_hr_mean + show the test mean±std
    columns; others rank by val headroom + show the val columns."""
    by_run: dict[str, list] = {}
    for l in ch_lines:
        by_run.setdefault(l["run_id"], []).append(l)
    rows = [r for r in (best_iter_row(v) for v in by_run.values()) if r]
    rows.sort(key=R.rank_sort_key)
    for i, r in enumerate(rows, 1):
        r["rank"] = i if not r["excluded_reason"] else None
    if ch is None and rows:
        ch = R.challenge_from_slug(rows[0]["run_id"])
    basis = R.metric_basis(ch)
    test_n = _TEST_N_BY_CHALLENGE.get(ch) if basis == "test" else None
    if basis == "test":
        unit = "patients" if ch == "mayo_ldct" else "cases"
        ranking_metric = f"test hr (mean over {test_n} {unit})" if test_n else "test hr"
        tiebreak = "test_ssim"
    else:
        ranking_metric, tiebreak = "headroom", "val_ssim"
    return {"ranking_metric": ranking_metric, "tiebreak": tiebreak,
            "metric_basis": basis, "test_n": test_n, "rows": rows}


# Views that carry an "updated" wall-clock field (with the json.dump indent they
# were written with) — restored to the prior timestamp on an unchanged build so
# the output is byte-stable (publish.sh idempotency). registry.jsonl / scratch
# carry no build stamp and are already stable.
_TIMESTAMPED_VIEWS = {
    "datasets.json": 1, "leaderboard.json": 1,
    "mayo_ldct.json": 1, "breast_ct.json": 1, "demo_dl.json": 1,
}


def _restore_timestamps(prior_built_at: str) -> None:
    for name, indent in _TIMESTAMPED_VIEWS.items():
        p = IDX / name
        if not p.exists():
            continue
        obj = json.loads(p.read_text())
        if isinstance(obj, dict) and "updated" in obj:
            obj["updated"] = prior_built_at
            p.write_text(json.dumps(obj, indent=indent, allow_nan=False))


def main() -> int:
    allow = R.load_json(ALLOWLIST)
    backstop = _load_backstop()
    IDX.mkdir(parents=True, exist_ok=True)
    SCR.mkdir(parents=True, exist_ok=True)

    # Capture the prior build's content_hash + timestamp so the output stays
    # BYTE-STABLE when nothing substantive changed (publish.sh idempotency: a
    # re-run with no new data must be a no-op commit, not a timestamp churn).
    meta_p = IDX / "registry.meta.json"
    prior_meta = R.load_json(meta_p) if meta_p.exists() else {}
    prior_hash = prior_meta.get("content_hash")
    prior_built_at = prior_meta.get("built_at")
    # ONE timestamp for the whole build (every view `updated` + meta `built_at`
    # share it) so restore-on-unchanged-content is byte-exact.
    build_ts = R.utc_now_iso()

    lines = build_registry_lines(allow, backstop)

    # registry.jsonl — flat canonical rollup, stable order.
    reg_text = "\n".join(json.dumps(l, sort_keys=True, allow_nan=False) for l in lines) + "\n"
    (IDX / "registry.jsonl").write_text(reg_text)

    # Per-dataset views + the ONE leaderboard per dataset.
    by_ch: dict[str, list] = {}
    for l in lines:
        by_ch.setdefault(l["challenge"], []).append(l)

    leaderboards: dict[str, dict] = {}
    datasets_summary = []
    for ch in ("mayo_ldct", "breast_ct", "demo_dl"):
        ch_lines = by_ch.get(ch, [])
        lb = build_leaderboard(ch_lines, ch)
        leaderboards[ch] = lb
        ds_meta = allow["datasets"].get(ch, {})

        # <challenge>.json : the dataset's full per-run summary + curves (what the
        # dashboard fetches). One entry per allowlisted run.
        runs_view = []
        by_run: dict[str, list] = {}
        for l in ch_lines:
            by_run.setdefault(l["run_id"], []).append(l)
        for slug, rl in sorted(by_run.items()):
            rl = sorted(rl, key=lambda x: x["iter"])
            # running-best-headroom curve over kept/non-discard finite iters
            curve, run = [], -math.inf
            for l in rl:
                h = l["metrics"].get("headroom")
                if R.finite(h) and (l["status"] or "").lower() != "discard" and h > run:
                    run = h
                curve.append([l["iter"], None if run == -math.inf else round(run, 4)])
            br = best_iter_row(rl)
            basis_run = br["metric_basis"] if br else "val"
            is_test_run = (basis_run == "test")
            # On a test dataset (Mayo) the run card's headline numbers are the
            # per-patient TEST means (mean over the 5 held-out test patients), NOT
            # validation — the val_* fields were dropped on the row above. On a val
            # dataset (demo_dl, breast_ct) they stay the single-patient val metrics.
            head_hr = (br["test_hr_mean"] if br and is_test_run else
                       (br["headroom"] if br else None))
            head_ssim = (br["test_ssim_mean"] if br and is_test_run else
                         (br["val_ssim"] if br else None))
            runs_view.append({
                "slug": slug,
                "solver_key": R.solver_key(slug),
                "name": R.display_name(slug),
                "challenge": ch,
                "campaign": R.campaign_from_slug(slug),
                "n_iterations": len(rl),
                "best_iter": br["best_iter"] if br else None,
                "metric_basis": basis_run,
                "best_headroom": head_hr,
                "best_ssim": head_ssim,
                # schema-2 aliases so the current dashboard.js keeps working in
                # Phase 1 (additive). best_score = the SSIM it historically read.
                "best_score": head_ssim,
                # Test-set mean±std (test datasets only; None elsewhere) so the
                # dashboard run card can render "mean ± std" for held-out test runs.
                "test_hr_mean": br["test_hr_mean"] if br else None,
                "test_hr_std": br["test_hr_std"] if br else None,
                "test_ssim_mean": br["test_ssim_mean"] if br else None,
                "test_ssim_std": br["test_ssim_std"] if br else None,
                "test_psnr_mean": br["test_psnr_mean"] if br else None,
                "test_psnr_std": br["test_psnr_std"] if br else None,
                "test_rmse_mean": br["test_rmse_mean"] if br else None,
                "test_rmse_std": br["test_rmse_std"] if br else None,
                "started": rl[-1].get("ts"),
                "status": ("excluded" if (br and br["excluded_reason"]) else "ranked"),
                "params_M": br["params_M"] if br else None,
                "excluded_reason": br["excluded_reason"] if br else None,
                "agent": rl[-1].get("agent"),
                "model": rl[-1].get("model"),
                "ts": rl[-1].get("ts"),
                "curve": curve,
                "val_image": (br["image"] if br and br["image"] else None),
                "test_image": rl[-1]["images"].get("test_showcase"),
            })
        (IDX / f"{ch}.json").write_text(json.dumps({
            "schema_version": R.SCHEMA_VERSION, "challenge": ch,
            "label": R.DATASET_LABELS.get(ch, ch),
            "campaign": ds_meta.get("campaign"),
            "updated": build_ts,
            "leaderboard": lb,
            "runs": runs_view,
        }, indent=1, allow_nan=False))

        # datasets.json champion = leaderboard rank-1 (so the two surfaces are
        # structurally identical). None if no run cleared baseline. The headline
        # hr/SSIM follow the board's metric basis: test mean (n=5) for Mayo, val
        # for the others — so the landing card matches the board it links to.
        rank1 = next((r for r in lb["rows"] if r.get("rank") == 1), None)
        basis = lb.get("metric_basis", "val")
        if rank1:
            champ_hr = rank1.get("rank_metric")
            champ_ssim = (rank1.get("test_ssim_mean") if basis == "test"
                          else rank1.get("val_ssim"))
        else:
            champ_hr = champ_ssim = None
        datasets_summary.append({
            "challenge": ch, "label": R.DATASET_LABELS.get(ch, ch),
            "campaign": ds_meta.get("campaign"),
            "n_runs": len(by_run),
            "n_iterations": len(ch_lines),
            "n_solvers": len(lb["rows"]),
            "champion_slug": rank1["run_id"] if rank1 else None,
            "champion_name": rank1["solver_name"] if rank1 else None,
            "champion_headroom": champ_hr,
            "champion_ssim": champ_ssim,
            "champion_metric_basis": basis,
            # test-set size for the "(test, n=…)" tag (Mayo 5 / breast 200); None
            # on a val dataset.
            "champion_test_n": (_TEST_N_BY_CHALLENGE.get(ch) if basis == "test"
                                else None),
            # schema-2 alias: champion_score was the metric the dashboard card
            # printed — the ranking metric (test hr for Mayo, val headroom else).
            "champion_score": champ_hr,
            "thumbnail": rank1["image"] if rank1 else None,
        })

    (IDX / "leaderboard.json").write_text(json.dumps({
        "schema_version": R.SCHEMA_VERSION, "updated": build_ts,
        # ranking is per-dataset now (datasets[ch].ranking_metric / metric_basis):
        # test hr for Mayo, val headroom for breast/demo. leaderboard.js reads the
        # per-dataset label, not these wrapper keys.
        "ranking_metric": "per-dataset", "tiebreak": "per-dataset",
        "datasets": leaderboards,
    }, indent=1, allow_nan=False))

    (IDX / "datasets.json").write_text(json.dumps({
        "schema_version": R.SCHEMA_VERSION, "updated": build_ts,
        "datasets": datasets_summary,
    }, indent=1, allow_nan=False))

    # scratch/<ch>.jsonl — capped recent observations (newest-first) for advice.
    for ch in ("mayo_ldct", "breast_ct", "demo_dl"):
        ch_lines = sorted(by_ch.get(ch, []), key=lambda l: (l["ts"] or "", l["iter"]))
        recent = list(reversed(ch_lines))[:SCRATCH_CAP]
        cards = [{
            "ts": l["ts"], "run_id": l["run_id"], "iter": l["iter"],
            "agent": l["agent"], "model": l["model"],
            "change_class": l["change_class"], "status": l["status"],
            "kept": l["kept"], "val_score": l["metrics"].get("val_ssim"),
            "headroom": l["metrics"].get("headroom"),
            "params_M": l["params_M"], "train_n": l["split"].get("train_n"),
            "rationale": l["rationale"], "advice_for_others": l["advice_for_others"],
            "comparison_image": l["images"].get("comparison"),
        } for l in recent]
        (SCR / f"{ch}.jsonl").write_text(
            "\n".join(json.dumps(c) for c in cards) + ("\n" if cards else ""))

    # README REGISTRY_TABLE block (no-JS GitHub front page).
    write_readme_block(datasets_summary)

    # registry.meta.json — content hash over the materialized views (the gate
    # recomputes it from a fresh build and fails if it differs).
    content_hash = compute_content_hash()
    allow_sha = hashlib.sha1(ALLOWLIST.read_bytes()).hexdigest()

    # Byte-stability: if the substantive content is identical to the prior build,
    # reuse the prior build-provenance stamps everywhere (views' `updated`, meta's
    # `built_at` AND `builder_git_sha`) so the files are byte-for-byte unchanged
    # and publish.sh commits nothing. content_hash already excludes the volatile
    # keys, so an unchanged hash == an unchanged registry — only the build stamps
    # (incl. the HEAD sha, which advances on every commit) would otherwise churn.
    unchanged = (prior_hash == content_hash and prior_built_at)
    if unchanged:
        _restore_timestamps(prior_built_at)
        built_at = prior_built_at
        builder_sha = prior_meta.get("builder_git_sha", _git_sha())
    else:
        built_at = build_ts
        builder_sha = _git_sha()

    (IDX / "registry.meta.json").write_text(json.dumps({
        "schema_version": R.SCHEMA_VERSION,
        "builder_git_sha": builder_sha,
        "built_at": built_at,
        "content_hash": content_hash,
        "allowlist_sha": allow_sha,
        "n_runs": sum(d["n_runs"] for d in datasets_summary),
        "n_iterations": len(lines),
    }, indent=1, allow_nan=False))

    print(f"[registry] {len(lines)} run-iters -> "
          + ", ".join(f"{d['challenge']}:{d['n_solvers']} solvers" for d in datasets_summary))
    print(f"[registry] content_hash={content_hash[:12]}  builder_sha={_git_sha()}")
    return 0


# --------------------------------------------------------------------------
# Content hash: stable digest over the rendered views (NOT meta.json itself).
# The gate rebuilds and compares this to catch committed-view-vs-fresh-build drift.
# --------------------------------------------------------------------------
_HASHED_VIEWS = ["registry.jsonl", "datasets.json", "leaderboard.json",
                 "mayo_ldct.json", "breast_ct.json", "demo_dl.json"]
# Volatile build-stamp keys excluded from the content hash so two builds of the
# SAME data hash identically (the gate compares substance, not the wall clock).
_VOLATILE_KEYS = {"updated", "built_at"}


def _canonical_bytes(name: str, raw: bytes) -> bytes:
    """Strip top-level volatile timestamp keys, return a canonical JSON blob.
    registry.jsonl / *.jsonl are line-delimited and carry no build stamp, so
    they pass through unchanged."""
    if name.endswith(".jsonl"):
        return raw
    try:
        obj = json.loads(raw)
        if isinstance(obj, dict):
            obj = {k: v for k, v in obj.items() if k not in _VOLATILE_KEYS}
        return json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()
    except Exception:
        return raw


def compute_content_hash() -> str:
    h = hashlib.sha256()
    for name in _HASHED_VIEWS:
        p = IDX / name
        h.update(name.encode())
        h.update(_canonical_bytes(name, p.read_bytes()) if p.exists() else b"")
    for ch in ("mayo_ldct", "breast_ct", "demo_dl"):
        p = SCR / f"{ch}.jsonl"
        rel = f"scratch/{ch}.jsonl"
        h.update(rel.encode())
        h.update(_canonical_bytes(rel, p.read_bytes()) if p.exists() else b"")
    return h.hexdigest()


# --------------------------------------------------------------------------
# README generated table — the only numbers on the no-JS GitHub front page.
# --------------------------------------------------------------------------
_BEGIN = "<!--REGISTRY_TABLE-->"
_END = "<!--/REGISTRY_TABLE-->"


def _fmt(x, digits=4):
    return f"{x:.{digits}f}" if R.finite(x) else "—"


def write_readme_block(datasets_summary: list[dict]) -> None:
    if not README.exists():
        return
    rows = ["| Dataset | Champion solver | SSIM | hr | Leaderboard |",
            "|---|---|---:|---:|---|"]
    label_link = {
        "mayo_ldct": ("Mayo-LDCT** (Wagner split, real helical)", "docs/leaderboards/mayo_ldct.md"),
        "breast_ct": ("Breast-CT** (128-view sparse)", "docs/leaderboards/breast_ct.md"),
        "demo_dl": ("Demo-DL** (Sidky ellipse, 128-view sparse)", "docs/leaderboards/demo_dl.md"),
    }
    order = {"breast_ct": 0, "demo_dl": 1, "mayo_ldct": 2}
    for d in sorted(datasets_summary, key=lambda x: order.get(x["challenge"], 9)):
        lab, link = label_link.get(d["challenge"], (d["label"], "#"))
        champ = d.get("champion_name") or "—"
        # basis tag so a reader never compares a test mean against a val number
        # under the shared "SSIM | hr" header (test set n varies per dataset:
        # Mayo n=5 patients, breast n=200 cases; others are val).
        if d.get("champion_metric_basis") == "test":
            _tn = d.get("champion_test_n")
            basis_tag = f" (test, n={_tn})" if _tn else " (test)"
        else:
            basis_tag = " (val)"
        rows.append(f"| **{lab} | {champ} | {_fmt(d.get('champion_ssim'))} "
                    f"| **{_fmt(d.get('champion_headroom'))}**{basis_tag} | [`{link}`]({link}) |")
    block = (_BEGIN + "\n"
             + "<!-- AUTO-GENERATED by scripts/build_registry.py — do not edit by hand. -->\n"
             + "\n".join(rows) + "\n" + _END)
    txt = README.read_text()
    if _BEGIN in txt and _END in txt:
        import re
        txt = re.sub(re.escape(_BEGIN) + r".*?" + re.escape(_END), block, txt, flags=re.S)
        README.write_text(txt)


if __name__ == "__main__":
    raise SystemExit(main())
