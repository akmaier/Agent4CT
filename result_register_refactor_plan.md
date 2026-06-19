# Result Register + Repo Refactor — Plan

**Status:** proposed (2026-06-19). **Decisions resolved (§7); execution NOT started — held for user review.**
**Goal:** one **scripted, drift-proof** path from "a result is produced on the
cluster" → "a number/image is rendered", killing the markdown/HTML staleness that
has confused this repo for months. This plan is the synthesis of a read-only audit
(5 mappers) + 3 independent architecture proposals.

---

## 1. Why things keep going stale (root causes, all verified)

| # | Root cause | Evidence |
|---|---|---|
| R1 | **Two contradictory rankings ship together.** `rebuild_runs_index.py` ranks champions by **SSIM**; `gen_mayo_leaderboard.py` ranks by **headroom**. The dashboard crowns a *below-baseline hr=0* solver while the leaderboard crowns another. | `rebuild_runs_index.py:204` (best_score=val_ssim) vs `gen_mayo_leaderboard.py:165` (hr); `index/datasets.json` mayo champ = `bilateral-n2i` (hr 0) vs board = `DD-UNet` (hr 0.339) |
| R2 | **The same numbers live in 6+ stores that each drop fields.** Only `observation.json` has the full set (params_M, ssim, hr, psnr, rmse, elapsed_s, cfg). `results.tsv`, `index/*.json`, `runs-index.json`, the 3 `.md` boards, `solver_params.json`, `datasets.json`, `observations.jsonl` are all partial, derived, and drift independently. | `results.tsv` has 9 cols, no psnr/rmse/time/params; `backfill_leaderboard_metrics.py` patches the patch |
| R3 | **Numbers are typed into markdown by hand / by the agent.** 30+ `publish X iter-N` hand-commits; `sync_summaries()` regex-rewrites README rows; the boards restate numbers in prose that goes stale. | `publish_mayo_wave.sh` pinned to the **purged** run-id `search-20260614-01`; README Demo-DL row contradicts its own board |
| R4 | **Provenance fields are wrong at the source.** All 79 `manifest.json` say `challenge:"dl_sparse_view"`; agent carries `-breast-ct` on Mayo runs; model string frozen at `claude-opus-4.7-1m`; `commit` column written empty. | `claude_agentic_one_iter.py:162-166,209-239` hardcodes these |
| R5 | **Debug scratch + dead stores pollute the tree.** ~60M of `docs/_*debug/` served-tree junk, `results/` 233M regenerable, 3 separate result roots (`docs/runs`, `runs/`, `results/`), ~80 one-off scripts, 6 ghost purged run dirs. | folder audit |

---

## 2. Design principles

1. **One canonical record, written once, never edited:** the per-iter
   `observation.json` (already exists). Git history is its audit trail.
2. **One deterministic builder** turns the canonical records into **one
   registry** + materialized views. No field is ever dropped. No LLM in the
   numbers path after the solver exits.
3. **Render tables on the fly in JS** from the registry. Markdown stops carrying
   numbers — boards become thin prose + a `<div>` the JS fills. **Nothing a human
   types can go stale, because humans no longer type numbers.**
4. **One ranking, stated in-band:** `headroom` (hr), `val_ssim` tiebreak.
   Below-baseline / discarded / non-finite runs are **excluded from the rank but
   still rendered** (dimmed, below) — so every solver always shows (never top-N).
5. **A staleness gate** (content hash) makes "committed view ≠ fresh build" a
   hard CI/pre-commit error. Drift becomes impossible to commit.
6. **Non-disruptive:** the live campaign keeps running throughout; cluster changes
   are values-only (no file-layout change).

---

## 3. The data model (`schema_version: 3`)

```
observation.json  (CANONICAL, immutable, 1 per iter — already written by the cluster)
   │   full fields: val_ssim, headroom, val_psnr, val_rmse, params_M,
   │   elapsed_s, train_n, val_n, cfg_full, change_class, kept, status,
   │   rationale, advice_for_others, comparison_image, ts, run_id, iter
   ▼   ── build_registry.py (deterministic, the ONLY aggregator) ──
docs/runs/index/
   registry.jsonl        flat canonical rollup, 1 line/run-iter (diffable in git)
   registry.meta.json    provenance: schema_version, builder_git_sha, built_at,
                         content_hash, allowlist_sha   ← drives the staleness gate
   <challenge>.json      per-dataset materialized view  ← what the dashboard fetches
   leaderboard.json      the ONE ranked surface, all solvers, hr-ranked
   datasets.json         landing summary (champion by hr, not ssim)
   scratch/<challenge>.jsonl   capped recent observations for the advice cards
docs/runs/CURRENT_RUNIDS.json   the single allowlist (active campaign/dataset +
                                purge list + per-run excludes) — a reset edits ONE file
```

**`registry.jsonl` line (one per run-iter):**
```json
{"schema_version":3,"run_id":"mayo-ldct-claude-agentic-dual-domain-supervised-search-20260619-01",
 "challenge":"mayo_ldct","campaign":"search-20260619-01","solver_key":"dual_domain_supervised",
 "solver_name":"DD-UNet supervised L2","iter":2,"ts":"2026-06-19T16:49:47Z","git_commit":"5ced1ec9",
 "metrics":{"val_ssim":0.90976,"headroom":0.33898,"val_psnr":37.67,"val_rmse":0.000654},
 "params_M":0.46576,"params_source":"observation","runtime":{"elapsed_s":893.06},
 "split":{"train_n":200,"val_n":214},"status":"keep","kept":true,"change_class":"architecture",
 "agent":"claude-agentic","model":"claude-opus-4-8-1m","rationale":"...","advice_for_others":"...",
 "cfg_full":{...},"images":{"comparison":"runs/<slug>/iterations/iter-0002/comparison.png",
 "valtest_showcase":"runs/<slug>/valtest_showcase.png","test_showcase":null},
 "obs_path":"runs/<slug>/iterations/iter-0002/observation.json","obs_sha1":"<hash>"}
```

**`leaderboard.json`** (per dataset, the render source): `ranking_metric:"headroom"`,
`tiebreak:"val_ssim"`, then `rows:[{rank, solver_key, solver_name, run_id, best_iter,
params_M, val_ssim, headroom, val_psnr, val_rmse, elapsed_s, image, excluded_reason}]`
— **every** solver, `excluded_reason` ("hr<=0" | "discard" | "uncalibrated" | null)
controls dimmed placement, **never** a slice. `datasets.json` champion = leaderboard
rank-1 of each dataset (so the two surfaces are structurally identical).

**Params resolution (once, in the builder):** `observation.params_M` →
`trainable_from_cfg(cfg_full)` (the bilateral/Wu formulas, now the only copy) →
`solver_params.json` backstop. Records `params_source`. Then `solver_params.json`
and `backfill_leaderboard_metrics.py` are **deleted**.

---

## 4. Script-owned pipeline (no agent in the numbers path)

| Stage | Owner | Kind | What it does |
|---|---|---|---|
| produce result | `solver_*.py` | cluster | writes `result.json` + `comparison.png` (unchanged) |
| record (canonical) | `claude_agentic_one_iter.py` **[patched]** | cluster | writes immutable `observation.json` + appends `results.tsv`. **Fix:** challenge from slug, agent/model/train_n from env, git SHA → commit col, psnr/rmse/elapsed_s/params_M → tsv |
| figures | `make_test_showcase.py` | cluster | re-renders `valtest_showcase.png`; run-id from `CURRENT_RUNIDS.json` (no hand-pin) |
| **build registry** | `build_registry.py` **[new]** | laptop/CI | walks allowlisted `observation.json` → `registry.jsonl` + meta + all views. Deterministic; one ranking; drops no field; **fails if a champion iter has no image** |
| **gate** | `validate_registry.py` **[new]** | pre-commit + CI | recomputes content hash; **fails** if committed views ≠ fresh build; asserts datasets champ == leaderboard rank-1, every image path resolves, row count == inventory (kills top-N by test) |
| **publish** | `publish.sh <subject?>` **[new]** | laptop | the ONLY orchestration the agent runs: rsync (run-id from allowlist) → build → validate → commit → push. Idempotent. Replaces the 30+ hand-commits + the dead `publish_mayo_wave.sh` |
| render | `dashboard.js` + `table.js` **[new]** | browser | pure JS over the registry — dashboard cards/curves + every leaderboard table. No `.slice()` anywhere → top-N is unexpressible |

The agent's only remaining roles: (1) author the next-iter cfg (a research
decision — legitimately agentic), (2) call `publish.sh`. It never edits a table,
re-ranks, or hand-writes a metric.

**Markdown after migration:** `docs/leaderboards/{mayo_ldct,breast_ct,demo_dl}.md`
keep only prose + `<div data-leaderboard="mayo_ldct"></div>`; `leaderboard.js`
fills it from `leaderboard.json` at view time. README/`docs/index.md` champion rows
are replaced by either a JS snippet (Pages) or, for the GitHub repo front page
(no JS), a `<!--REGISTRY_TABLE-->` block **generated by `build_registry.py`** in the
same run — so it cannot drift. `sync_summaries()`/`_sub_line()` regex-patching deleted.

---

## 5. Target folder structure

```
Agent4CT/
  README.md  CLAUDE.md  solver_plan.md  result_register_refactor_plan.md
  ddssl_ldct/                 reusable backbone            (keep)
  pentathlon/
    demo_dl_reference/        solver impls + design docs   (keep)
    dl_sparse_view_*/         stale first-experiment variants (ARCHIVE)
  scripts/
    pipeline/                 the ~12 production scripts (build_registry, publish.sh,
                              validate_registry, make_test_showcase, the patched driver…)
    debug/                    the ~80 one-off debug_/fit_/investigate_/z_sweep_ scripts (MOVE)
  cluster/slurm/              production sbatch (~30)
  cluster/slurm/debug/        one-off ablation/sanity/verify/compare sbatch (~46, MOVE)
  data/  config/  challenges/  agentic_cfgs/  literature/   (keep)
  docs/                       GitHub Pages site ONLY — no debug junk
    index.md  dashboard.html  leaderboards/*.md (thin prose + JS mount)
    assets/{dashboard.js, table.js, leaderboard.js, dashboard.css}
    runs/
      <slug>/…                immutable run dirs (observation.json canonical)
      index/                  registry.jsonl, registry.meta.json, <ch>.json,
                              leaderboard.json, datasets.json, scratch/
      CURRENT_RUNIDS.json     the single allowlist
```

**Removed from the tree (git history retains them):**
- `docs/_breast_geom_debug/` (46M), `_fov_eval/` (8M), `_restage_verify/` (4.6M),
  `_mayo_figure_fix/`, `_showcase_debug/`, `_n2i_smoke/`, `_n2i_onboard/` — debug
  scratch in the served tree → `git rm` + `.gitignore` (the `docs/_*` glob).
- `results/` (233M, regenerable calibration/geometry PNGs/npz) → `git rm` + gitignore.
- `runs/` (top-level, 3.9M, read by nothing) → archive/remove.
- 6 ghost `mayo-ldct-*-search-20260614-01/` dirs (purged campaign, partial delete) → remove.
- `docs/leaderboards/baseline_2026-06-14/` (17–34M static baseline PNGs) → archive.
- `runs-index.json`, `solver_params.json`, `backfill_leaderboard_metrics.py`,
  `propose_next_iter.py` short-budget defaults — superseded by the builder/allowlist.

Net: docs/ payload drops by ~60M (debug) and the repo by ~230M (results/), the
Pages site serves only the product, and the result numbers live in exactly one place.

---

## 6. Migration order (each step ships value; the live loop never pauses)

**Phase 0 — stop the bleeding (no data move, immediate):**
- Unify the ranking NOW: point `rebuild_runs_index.py` champion at **headroom**
  (ssim tiebreak, exclude discard/non-finite) so the dashboard stops crowning the
  hr=0 solver. (Top-5 grids already removed today.)

**Phase 1 — the register (additive, nothing deleted yet):**
- Write `build_registry.py` (+ shared `naming.py`/`ranking.py` helpers) → emit
  `registry.jsonl` + meta + `<ch>.json` + `leaderboard.json` + `datasets.json`
  **alongside** the current `index/*.json` (back-compat).
- Write `validate_registry.py`. Add `CURRENT_RUNIDS.json`.

**Phase 2 — flip the renderers:**
- `dashboard.js` reads `<ch>.json`/`datasets.json` from the new builder.
- Add `table.js`/`leaderboard.js`; convert the 3 boards + README/index rows to JS
  mounts / generated block. Delete the numbers from markdown. Retire
  `gen_mayo_leaderboard.py` + `sync_summaries`.

**Phase 3 — the publish script + cluster recorder fix:**
- Land `publish.sh`; switch the cron tick to call it instead of the bash blob.
- Patch `claude_agentic_one_iter.py` (values-only: challenge/agent/model/train_n/
  commit/tsv-metrics). Applies to *new* iters; the builder backfills old runs from
  observation.json regardless.

**Phase 4 — cleanup + gate:**
- **Tag `archive/pre-refactor-2026-06-19`** (named restore point for the whole
  current tree incl. the debug artefacts) **before any removal.**
- `git rm` the debug dirs + `results/` + ghost runs; add `.gitignore` globs.
- Move `scripts/debug/` + `cluster/slurm/debug/`.
- Wire `validate_registry.py` as **both** a pre-commit hook **and** a GitHub
  Action (the full content-hash staleness gate — staleness blocks the commit
  locally and the PR in CI).
- Delete `runs-index.json`, `solver_params.json`, `backfill_leaderboard_metrics.py`.

**Verification per phase:** local `python -m http.server` in `docs/` → confirm one
index fetch (not 108 tsv fetches), all solvers render, champion agrees across
dashboard+board, image links resolve; `du -sh docs/runs results` before/after;
`gh api .../pages/builds` confirms Pages serves it.

---

## 7. Decisions (resolved 2026-06-19)

1. **Drift gate → FULL.** `build_registry.py` stamps a `content_hash` into
   `registry.meta.json`; `validate_registry.py` runs as **both** a local
   pre-commit hook and a GitHub Action, recomputing the hash and **failing**
   commit/PR if any committed view ≠ a fresh build, if dashboard champion ≠
   leaderboard rank-1, or if any rendered image link is dead. Staleness becomes
   impossible to commit. (Paper-grade — matches "these numbers go in a paper".)
2. **Cleanup → ARCHIVE-TAG FIRST.** Create tag `archive/pre-refactor-2026-06-19`
   capturing the full current tree, **then** `git rm` the debug dirs / `results/`
   / ghost runs and add `.gitignore` globs. Named restore point + slim tree.
3. **Execution → PLAN ONLY, REVIEW FIRST.** No code/data changes yet (beyond the
   already-shipped top-5 removal). The user reviews this doc and signals when/what
   to start. Phase 0 (unify dashboard ranking to hr) + Phase 1 (build the register
   additively) are the safe first steps once green-lit.
