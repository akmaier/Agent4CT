#!/bin/bash
# Refresh the LIVE Breast-CT leaderboard on the GitHub Pages dashboard.
# Pulls the latest per-iter results from the cluster, rebuilds the registry,
# validates, and commits+pushes. All local git + build_registry (no LLM calls),
# so it runs fine even during Anthropic API outages. The 300MB per-iter
# recon_raw.npz / model_ckpt.pt are gitignored and never committed.
#
# Run hourly (from the laptop repo root):  bash scripts/publish_leaderboard.sh
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
REPO_C=/cluster/maier/Agent4CT
SSH="ssh -o BatchMode=yes -o ConnectTimeout=12"

# 1. Discover breast run dirs on the cluster.
slugs=$($SSH lme-bastion "ls -d $REPO_C/docs/runs/breast-ct-claude-agentic-*-search-20260703-01 2>/dev/null | sed 's#.*/##'" 2>/dev/null)
[ -z "$slugs" ] && { echo "publish_leaderboard: no breast run dirs on cluster"; exit 0; }

# 2. Rsync each (observation.json / comparison.png / results.tsv / manifest.json;
#    NEVER the big npz/pt or the per-JOBID DONE sentinels).
for d in $slugs; do
  rsync -az -q --exclude='*.npz' --exclude='*.pt' --exclude='DONE.*' -e "$SSH" \
    "lme-bastion:$REPO_C/docs/runs/$d/" "docs/runs/$d/" 2>/dev/null || true
done

# 3. Allowlist exactly the present breast run-ids.
python3 - "$slugs" <<'PY'
import json,sys
slugs=sorted(sys.argv[1].split())
p="docs/runs/CURRENT_RUNIDS.json"; d=json.load(open(p))
d["datasets"]["breast_ct"]["run_ids"]=slugs
json.dump(d,open(p,"w"),indent=2)
print(f"publish_leaderboard: {len(slugs)} breast run-ids allowlisted")
PY

# 4. Rebuild + validate (abort the publish if the drift gate fails).
PYTHONPATH=scripts python3 scripts/build_registry.py >/dev/null 2>&1 || { echo "build_registry FAILED"; exit 1; }
if ! PYTHONPATH=scripts python3 scripts/validate_registry.py >/dev/null 2>&1; then
  echo "validate_registry FAILED — not publishing"; exit 1
fi

# 5. Commit + push (npz/pt are gitignored so they can't be staged).
git add docs/runs/index docs/runs/scratch docs/runs/CURRENT_RUNIDS.json README.md \
        docs/runs/breast-ct-claude-agentic-*-search-20260703-01 2>/dev/null || true
if git diff --cached --quiet; then echo "publish_leaderboard: no change"; exit 0; fi
git commit -q -m "Live Breast-CT leaderboard refresh $(date -u +%Y-%m-%dT%H:%MZ)"
git push origin main 2>&1 | tail -2
