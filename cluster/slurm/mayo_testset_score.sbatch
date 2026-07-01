#!/bin/bash
#SBATCH --job-name=mayo-test-score
#SBATCH --partition=main
#SBATCH --qos=turbo
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --exclude=lme49,lme50,lme51,lme52,lme170   # 2026-06-29: lme53 (v100 16GB) re-enabled after driver update + smoke test; lme49 still down (unk*), lme50/51/52 too small (gtx1080ti/titanxp 11-12GB), lme170 blacklisted
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/cluster/maier/Agent4CT/results/slurm/%x-%j.out
#SBATCH --error=/cluster/maier/Agent4CT/results/slurm/%x-%j.err

# Phase 1B per-patient TEST-set scorer (one job per Mayo solver). Loops the 5
# held-out Wagner test patients (L014 L056 L058 L075 L123), runs the solver's
# best-iter config on each via the loader override (AGENT4CT_EVAL_PATIENT), and
# writes docs/runs/<SLUG>/final.json with mean ± std over the 5 patients.
#
# ~20 min/patient (GPU recon) x 5 patients ~= 100 min; --time=3h has headroom.
# Dispatched by scripts/score_mayo_testset.py (--all / --solver). NOT a search
# iter — does not touch results.tsv / observation.json.
#
# Required exported variables (set by the dispatcher's --export):
#   SLUG      — Mayo run-id slug (e.g. mayo-ldct-claude-agentic-itnet-search-…)
#   SOLVER    — SOLVER_MAP key (e.g. itnet, ram_zeroshot, naf, …)
#   CFG_JSON  — path to the best-iter cfg JSON the dispatcher wrote
#
# Example (the dispatcher does this for you):
#   sbatch --job-name=mayo-test-itnet \
#          --export=ALL,SLUG=mayo-ldct-claude-agentic-itnet-search-20260619-01,\
#SOLVER=itnet,CFG_JSON=/cluster/maier/Agent4CT/agentic_cfgs/<slug>_testset_best_iter_0016.json \
#          cluster/slurm/mayo_testset_score.sbatch

set -euo pipefail
: "${SLUG:?need SLUG (Mayo run-id)}"
: "${SOLVER:?need SOLVER (SOLVER_MAP key)}"
: "${CFG_JSON:?need CFG_JSON path}"

REPO=/cluster/maier/Agent4CT
cd "$REPO"
VENV="$REPO/.venv"
export CUDA_HOME="$VENV/cuda-pip-bundle"
export PATH="$CUDA_HOME/bin:$PATH"
source "$VENV/bin/activate"
export PYTHONPATH="$REPO:${PYTHONPATH:-}"
export HDF5_USE_FILE_LOCKING=FALSE

echo "[testset] host=$(hostname) job=$SLURM_JOB_ID"
echo "[testset] slug=$SLUG  solver=$SOLVER  cfg=$CFG_JSON"
nvidia-smi -L
echo
cat "$CFG_JSON"; echo

python scripts/score_mayo_testset.py --worker \
    --slug "$SLUG" --solver "$SOLVER" --cfg "$CFG_JSON" \
    --solver-src "${SOLVER_SRC:-}"

echo "[testset] done."
