#!/bin/bash
#SBATCH --job-name=breast-test-score
#SBATCH --partition=main
#SBATCH --qos=turbo
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --exclude=lme49,lme50,lme51,lme52,lme170   # same exclude as the Mayo scorers (too-small / down / blacklisted nodes)
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=/cluster/maier/Agent4CT/results/slurm/%x-%j.out
#SBATCH --error=/cluster/maier/Agent4CT/results/slurm/%x-%j.err

# Breast-CT per-CASE TEST-set scorer (one job per breast solver). Runs the
# solver's best-iter config ONCE on the 200 held-out test cases via the loader
# redirect (AGENT4CT_EVAL_SPLIT=test), persists the raw recon, re-scores per case,
# and writes docs/runs/<SLUG>/final.json with per-case mean ± std (incl.
# headroom_std). Breast cases are i.i.d. (no patients) -> a clean per-case spread.
#
# --time=00:30:00 = the strict 20-min budget + figure/IO headroom (breast 128-view
# fan is far cheaper than Mayo's rebinned helical). Dispatched by
# scripts/score_breast_testset.py (--all / --solver).
#
# Required exported variables (set by the dispatcher's --export):
#   SLUG      — breast run-id slug (e.g. breast-ct-claude-agentic-itnet-v2-search-…)
#   SOLVER    — SOLVER_MAP key (e.g. itnet_v2, learned_primal_dual, …)
#   CFG_JSON  — path to the best-iter cfg JSON the dispatcher wrote
#   SPLIT     — held-out split to score (default test)
#   SOLVER_SRC— optional per-iter solver-src snapshot (code-evolving solvers)

set -euo pipefail
: "${SLUG:?need SLUG (breast run-id)}"
: "${SOLVER:?need SOLVER (SOLVER_MAP key)}"
: "${CFG_JSON:?need CFG_JSON path}"
SPLIT="${SPLIT:-test}"

REPO=/cluster/maier/Agent4CT
cd "$REPO"
VENV="$REPO/.venv"
export CUDA_HOME="$VENV/cuda-pip-bundle"
export PATH="$CUDA_HOME/bin:$PATH"
source "$VENV/bin/activate"
export PYTHONPATH="$REPO:${PYTHONPATH:-}"
export HDF5_USE_FILE_LOCKING=FALSE
python -c "import hdf5plugin" 2>/dev/null || true   # register blosc/zstd filter for staged breast h5

echo "[breast-test] host=$(hostname) job=$SLURM_JOB_ID"
echo "[breast-test] slug=$SLUG  solver=$SOLVER  split=$SPLIT  cfg=$CFG_JSON"
nvidia-smi -L
echo
cat "$CFG_JSON"; echo

python scripts/score_breast_testset.py --worker \
    --slug "$SLUG" --solver "$SOLVER" --cfg "$CFG_JSON" \
    --split "$SPLIT" --solver-src "${SOLVER_SRC:-}"

echo "[breast-test] done."
