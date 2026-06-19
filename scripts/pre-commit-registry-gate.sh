#!/usr/bin/env bash
# The result-register pre-commit gate. Installed (never silently) by
#   scripts/install-registry-hook.sh
# into .git/hooks/pre-commit. Runs validate_registry.py and BLOCKS the commit if
# the staged registry views disagree with a fresh build, if a dashboard champion
# != its leaderboard rank-1, if a rendered image is missing, or if a board's row
# count != its solver inventory. Skip a one-off commit with: git commit --no-verify.
set -euo pipefail

# Only gate when something in the registry's input/output surface is staged —
# a docs-only or code-only commit elsewhere should not pay the gate cost.
if git diff --cached --name-only | grep -qE \
    '^(docs/runs/|scripts/(build_registry|validate_registry|registry_lib)\.py|docs/runs/CURRENT_RUNIDS\.json)'; then
  echo "[pre-commit] running registry staleness gate …"
  if ! python3 scripts/validate_registry.py; then
    echo "[pre-commit] BLOCKED — registry is stale/inconsistent." >&2
    echo "[pre-commit] Fix: python3 scripts/build_registry.py && git add docs/runs/index docs/runs/scratch README.md" >&2
    echo "[pre-commit] (or, to bypass for this commit only: git commit --no-verify)" >&2
    exit 1
  fi
fi
exit 0
