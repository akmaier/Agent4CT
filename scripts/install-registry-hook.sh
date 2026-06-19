#!/usr/bin/env bash
# One-line installer for the result-register pre-commit gate. We do NOT silently
# write into .git/hooks — you run this explicitly to opt in:
#
#   bash scripts/install-registry-hook.sh
#
# It symlinks .git/hooks/pre-commit -> ../../scripts/pre-commit-registry-gate.sh
# (so the hook tracks the committed script). Re-run any time; it is idempotent.
# Uninstall with: rm .git/hooks/pre-commit
set -euo pipefail
cd "$(dirname "$0")/.."

GIT_DIR="$(git rev-parse --git-dir)"
HOOKS="$GIT_DIR/hooks"
mkdir -p "$HOOKS"
TARGET="$HOOKS/pre-commit"

if [[ -e "$TARGET" && ! -L "$TARGET" ]]; then
  echo "install-registry-hook: $TARGET already exists and is not our symlink." >&2
  echo "  Back it up / remove it first, then re-run." >&2
  exit 1
fi

# Relative symlink from .git/hooks/ back to the tracked script.
ln -sf "../../scripts/pre-commit-registry-gate.sh" "$TARGET"
chmod +x scripts/pre-commit-registry-gate.sh
echo "install-registry-hook: installed $TARGET -> scripts/pre-commit-registry-gate.sh"
echo "  The registry staleness gate now runs before every commit that touches"
echo "  docs/runs/ or the registry scripts. Bypass once with: git commit --no-verify"
