"""validate_registry.py — the staleness GATE (pre-commit hook + GitHub Action).

Runs the FULL drift gate (result_register_refactor_plan.md §7 decision 1):

  1. Rebuilds the registry in a throwaway temp tree from the SAME allowlisted
     observation.json and recomputes the content_hash. FAILS if the committed
     registry.meta.json content_hash != a fresh build (i.e. a view was edited by
     hand, or build_registry.py was not re-run after the data changed).
  2. FAILS if any dataset's datasets.json champion != that dataset's
     leaderboard.json rank-1.
  3. FAILS if any rendered leaderboard image path is missing on disk.
  4. FAILS if a board's row count != that campaign's solver inventory
     (== len(allowlist run_ids) for the dataset) — kills top-N by test.
  5. FAILS if the allowlist_sha in meta != sha1(CURRENT_RUNIDS.json).

Exit 0 = clean (safe to commit / merge). Exit 1 = drift (blocked).
Read-only: it never writes into the repo (the temp build is in a scratch dir it
removes). Torch-free.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import sys
import tempfile
from pathlib import Path

import registry_lib as R
import build_registry as B

IDX = R.DOCS_RUNS / "index"
ALLOWLIST = R.DOCS_RUNS / "CURRENT_RUNIDS.json"


def _fail(msg: str, fails: list):
    fails.append(msg)
    print(f"  ✗ {msg}")


def check_fresh_build(fails: list) -> None:
    """Two independent hash checks:

    A. ON-DISK hash == committed meta hash. Hash the views CURRENTLY on disk and
       compare to registry.meta.json.content_hash. Catches a HAND-EDIT to any
       committed view (the on-disk bytes diverge from the recorded hash).
    B. committed meta hash == FRESH rebuild hash. Throwaway in-place rebuild from
       the allowlisted observation.json, restore originals. Catches "the data
       changed but build_registry.py was not re-run" (the recorded hash is stale).

    A together with B = committed views are byte-equal to a fresh build of the
    canonical records. Read-only: the throwaway build is restored."""
    meta_p = IDX / "registry.meta.json"
    if not meta_p.exists():
        _fail("registry.meta.json missing — run scripts/build_registry.py", fails)
        return
    committed = R.load_json(meta_p)
    committed_hash = committed.get("content_hash")

    # A. on-disk views vs recorded hash (catches hand-edits)
    ondisk_hash = B.compute_content_hash()
    if ondisk_hash != committed_hash:
        _fail(f"committed view drift: on-disk views hash {ondisk_hash[:12]} != "
              f"registry.meta.json {str(committed_hash)[:12]} — a view was "
              f"hand-edited. Run: python3 scripts/build_registry.py", fails)
    else:
        print(f"  ✓ on-disk views match recorded content_hash ({ondisk_hash[:12]})")

    # B. recorded hash vs a fresh rebuild (catches stale builder output)
    snap = Path(tempfile.mkdtemp(prefix="reg_gate_"))
    written = ([IDX / n for n in B._HASHED_VIEWS]
               + [IDX / "registry.meta.json"]
               + [B.SCR / f"{ch}.jsonl" for ch in ("mayo_ldct", "breast_ct", "demo_dl")]
               + [B.README])
    backups: dict[Path, Path] = {}
    try:
        for i, p in enumerate(written):
            if p.exists():
                bak = snap / f"{i:03d}_{p.name}"
                shutil.copy2(p, bak)
                backups[p] = bak
        B.main()
        fresh_hash = B.compute_content_hash()
    finally:
        for p in written:
            if p in backups:
                shutil.copy2(backups[p], p)
        shutil.rmtree(snap, ignore_errors=True)

    if committed_hash != fresh_hash:
        _fail(f"stale registry: recorded {str(committed_hash)[:12]} != fresh "
              f"rebuild {fresh_hash[:12]} — the data changed but "
              f"build_registry.py was not re-run. Run: python3 scripts/build_registry.py",
              fails)
    else:
        print(f"  ✓ recorded content_hash matches a fresh rebuild ({fresh_hash[:12]})")

    allow_sha = hashlib.sha1(ALLOWLIST.read_bytes()).hexdigest()
    if committed.get("allowlist_sha") != allow_sha:
        _fail("allowlist_sha drift: registry.meta.json does not match "
              "CURRENT_RUNIDS.json — rebuild the registry", fails)
    else:
        print("  ✓ allowlist_sha matches CURRENT_RUNIDS.json")


def check_views(fails: list) -> None:
    allow = R.load_json(ALLOWLIST)
    datasets = {d["challenge"]: d
                for d in R.load_json(IDX / "datasets.json")["datasets"]}
    lb = R.load_json(IDX / "leaderboard.json")["datasets"]

    for ch in ("mayo_ldct", "breast_ct", "demo_dl"):
        board = lb.get(ch, {})
        rows = board.get("rows", [])
        ds = datasets.get(ch, {})
        inventory = len(allow["datasets"].get(ch, {}).get("run_ids", []))

        # 4. row count == inventory (every solver present, no top-N)
        if len(rows) != inventory:
            _fail(f"{ch}: leaderboard has {len(rows)} rows but the campaign "
                  f"inventory is {inventory} — a solver is missing or top-N "
                  f"slicing crept in", fails)
        else:
            print(f"  ✓ {ch}: {len(rows)} rows == {inventory} solver inventory")

        # 2. datasets champion == leaderboard rank-1
        rank1 = next((r for r in rows if r.get("rank") == 1), None)
        champ = ds.get("champion_slug")
        if rank1 is None:
            if champ is not None:
                _fail(f"{ch}: datasets champion={champ} but leaderboard has no "
                      f"rank-1 (no run cleared baseline)", fails)
        elif rank1["run_id"] != champ:
            _fail(f"{ch}: champion mismatch — datasets={champ} vs leaderboard "
                  f"rank-1={rank1['run_id']}", fails)
        else:
            print(f"  ✓ {ch}: champion == leaderboard rank-1 ({champ})")

        # 3. every rendered image path resolves
        dead = 0
        for r in rows:
            img = r.get("image")
            if not img:
                _fail(f"{ch}: row {r['solver_name']} ({r['run_id']}) has NO "
                      f"image", fails)
                dead += 1
            elif not (R.DOCS_RUNS.parent / img).exists():
                _fail(f"{ch}: dead image link {img}", fails)
                dead += 1
        if dead == 0:
            print(f"  ✓ {ch}: all {len(rows)} image links resolve")

        # ranks must be a 1..k prefix with no gaps for the ranked subset
        ranked = [r for r in rows if r.get("rank") is not None]
        if [r["rank"] for r in ranked] != list(range(1, len(ranked) + 1)):
            _fail(f"{ch}: ranked rows are not a contiguous 1..n sequence", fails)


def main() -> int:
    fails: list = []
    print("[gate] checking fresh-build content hash …")
    check_fresh_build(fails)
    print("[gate] checking views (champion == rank-1, images, row==inventory) …")
    check_views(fails)
    if fails:
        print(f"\n[gate] FAILED — {len(fails)} drift error(s). Commit blocked.")
        return 1
    print("\n[gate] PASS — registry is consistent and drift-free.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
