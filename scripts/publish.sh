#!/usr/bin/env bash
# publish.sh — the ONE orchestration the agent/cron runs to publish results.
#
#   rsync (ONLY the allowlisted run-ids from docs/runs/CURRENT_RUNIDS.json,
#          NEVER a blanket docs/runs glob) cluster -> local
#   -> python3 scripts/build_registry.py     (deterministic, no LLM)
#   -> python3 scripts/validate_registry.py  (the staleness gate — ABORTS on drift)
#   -> git add (registry views + run dirs + the markdown boards) -> commit -> push
#
# Idempotent: re-running with no new data rebuilds identically (the gate's
# content_hash is deterministic) and commits nothing. The gate runs BEFORE the
# commit, so a drifted/broken registry can never be pushed.
#
# Usage:
#   scripts/publish.sh ["commit subject"]
# Env:
#   CLUSTER_HOST   ssh host for the cluster run dirs   (default: lme-bastion)
#   CLUSTER_RUNS   remote docs/runs path               (default: /cluster/maier/Agent4CT/docs/runs)
#   NO_RSYNC=1     skip the rsync (build/validate/commit from the local tree only)
#   NO_PUSH=1      build + validate + commit but do not push
#
# Replaces the 30+ hand-commits and the dead scripts/publish_mayo_wave.sh.
set -euo pipefail
cd "$(dirname "$0")/.."

CLUSTER_HOST="${CLUSTER_HOST:-lme-bastion}"
CLUSTER_RUNS="${CLUSTER_RUNS:-/cluster/maier/Agent4CT/docs/runs}"
ALLOWLIST="docs/runs/CURRENT_RUNIDS.json"
SUBJECT="${1:-Publish: registry + comparison images}"

if [[ ! -f "$ALLOWLIST" ]]; then
  echo "publish: $ALLOWLIST missing — cannot determine which run-ids to sync." >&2
  exit 1
fi

# --- 1. rsync ONLY the allowlisted run-ids (one rsync per run-id, no glob) -----
if [[ "${NO_RSYNC:-0}" != "1" ]]; then
  # Read run_ids + purge from the allowlist (python keeps this robust vs jq absence).
  mapfile -t RUN_IDS < <(python3 - "$ALLOWLIST" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
ids = []
for ds in d.get("datasets", {}).values():
    ids += ds.get("run_ids", [])
# de-dup, stable
seen = set()
for r in ids:
    if r not in seen:
        seen.add(r); print(r)
PY
)
  echo "publish: rsyncing ${#RUN_IDS[@]} allowlisted run-ids from ${CLUSTER_HOST}:${CLUSTER_RUNS}"
  for rid in "${RUN_IDS[@]}"; do
    # Per-run-id rsync. --relative keeps the slug dir; trailing path is exact, so
    # NO blanket docs/runs glob can re-bloat the index. Missing remote dirs are
    # tolerated (a run may not have produced output yet).
    rsync -e ssh -a --mkpath \
      "${CLUSTER_HOST}:${CLUSTER_RUNS}/${rid}/" "docs/runs/${rid}/" \
      2>/dev/null || echo "  (skip ${rid}: not on cluster yet)"
  done
else
  echo "publish: NO_RSYNC=1 — building from the local tree."
fi

# --- 2. build the registry (deterministic) -------------------------------------
echo "publish: building registry …"
python3 scripts/build_registry.py

# --- 3. GATE: validate before staging anything ---------------------------------
echo "publish: validating registry (staleness gate) …"
if ! python3 scripts/validate_registry.py; then
  echo "publish: GATE FAILED — registry is inconsistent. Nothing staged, nothing pushed." >&2
  exit 2
fi

# --- 4. stage + commit ---------------------------------------------------------
git add docs/runs/index docs/runs/scratch docs/runs/CURRENT_RUNIDS.json \
        docs/leaderboards README.md \
        $(git ls-files --modified --others --exclude-standard docs/runs | grep -E '^docs/runs/[^/]+/' || true) \
        2>/dev/null || true
# Also stage the synced run dirs (new comparison.png / observation.json).
git add docs/runs 2>/dev/null || true

if git diff --cached --quiet; then
  echo "publish: nothing to commit (idempotent — registry unchanged)."
  exit 0
fi

git commit -q -m "${SUBJECT}

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
echo "publish: committed."

# --- 5. push -------------------------------------------------------------------
if [[ "${NO_PUSH:-0}" == "1" ]]; then
  echo "publish: NO_PUSH=1 — committed locally, not pushed."
  exit 0
fi
# pull --rebase --autostash tolerates the other agents racing on the shared tree.
git pull --rebase --autostash -q 2>/dev/null || true
git push -q 2>&1 | tail -1 || { echo "publish: push failed." >&2; exit 3; }
echo "publish: PUSHED."
