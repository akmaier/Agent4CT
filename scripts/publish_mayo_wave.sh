#!/bin/bash
# One-shot Mayo agentic publish, run after each wave of iters completes:
#   rsync the new run dirs cluster->local, rebuild the dashboard index,
#   regenerate the leaderboard table from the data, then commit + push to main
#   so GitHub Pages serves the new comparison images.
# Pass the commit subject as $1 (the in-flight note in mayo_ldct.md is edited
# by hand before running this, since it states the NEXT hypotheses).
set -euo pipefail
cd "$(dirname "$0")/.."
rsync -e ssh -a "lme-bastion:/cluster/maier/Agent4CT/docs/runs/mayo-ldct-claude-agentic-*-search-*" docs/runs/ 2>/dev/null || true
python3 scripts/rebuild_runs_index.py | tail -1
python3 scripts/gen_mayo_leaderboard.py
git add docs/runs docs/leaderboards/mayo_ldct.md README.md scripts/gen_mayo_leaderboard.py scripts/rebuild_runs_index.py scripts/publish_mayo_wave.sh 2>/dev/null || true
if git diff --cached --quiet; then
  echo "nothing to commit"
else
  git commit -q -m "${1:-Mayo agentic wave: results + comparison images}

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
  git push -q origin main 2>&1 | tail -1
  echo "PUSHED"
fi
