#!/usr/bin/env bash
# One-line installer for the result-register pre-commit gate. We do NOT silently
# write into .git/hooks — you run this explicitly to opt in:
#
#   bash scripts/install-registry-hook.sh
#
# It symlinks the repo's pre-commit hook -> scripts/pre-commit-registry-gate.sh
# (so the hook always tracks the committed script). Idempotent; re-run any time.
# Uninstall with:  rm "$(git rev-parse --git-common-dir)/hooks/pre-commit"
#
# NOTE on git worktrees: git consults hooks in the COMMON git dir (the main
# checkout's .git/hooks), not the per-worktree git dir — so we install there via
# `git rev-parse --git-common-dir`. An absolute symlink target keeps it resolving
# regardless of how deep the worktree lives.
set -euo pipefail
cd "$(dirname "$0")/.."

REPO_ROOT="$(git rev-parse --show-toplevel)"
COMMON_GIT_DIR="$(git rev-parse --git-common-dir)"
# git-common-dir may be relative to the worktree; normalise to absolute.
case "$COMMON_GIT_DIR" in
  /*) : ;;
  *) COMMON_GIT_DIR="$(cd "$COMMON_GIT_DIR" && pwd)" ;;
esac
HOOKS="$COMMON_GIT_DIR/hooks"
mkdir -p "$HOOKS"
TARGET="$HOOKS/pre-commit"
SRC="$REPO_ROOT/scripts/pre-commit-registry-gate.sh"

if [[ -e "$TARGET" && ! -L "$TARGET" ]]; then
  echo "install-registry-hook: $TARGET already exists and is not our symlink." >&2
  echo "  Back it up / remove it first, then re-run." >&2
  exit 1
fi

chmod +x "$SRC"
ln -sf "$SRC" "$TARGET"
echo "install-registry-hook: installed"
echo "  $TARGET"
echo "  -> $SRC"
echo "  The registry staleness gate now runs before every commit that touches"
echo "  docs/runs/ or the registry scripts. Bypass once with: git commit --no-verify"
