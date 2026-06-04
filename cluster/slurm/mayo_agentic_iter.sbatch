#!/bin/bash
#SBATCH --job-name=mayo-agentic-iter
#SBATCH --partition=main
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --exclude=lme49,lme53,lme170,lme171
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=/cluster/maier/Agent4CT/results/slurm/%x-%j.out
#SBATCH --error=/cluster/maier/Agent4CT/results/slurm/%x-%j.err

# Generic Mayo agentic iter dispatcher — for iter-2+ in the autoresearch
# loop. Use mayo_first_iter.sbatch for iter-1 (full feasibility config,
# 30-60 min wall). This one uses the SHORT (5-15 min) budget required
# by solver_plan.md Step 2 for iter-2+ hypothesis tests.
#
# Required exported variables:
#   SOLVER    — solver key from claude_agentic_one_iter.py:SOLVER_MAP
#   SLUG      — autoresearch slug (e.g. mayo-ldct-claude-agentic-lpd-search-…)
#   ITER      — iter number (≥2)
#   CFG_JSON  — path to the per-iter config JSON Claude wrote
#
# Typical iter-2+ wall: 5-15 minutes. If your iter is taking longer
# than 30 minutes, your config is too rich for an autoresearch
# hypothesis test — drop val_n / epochs / train_n further.
#
# Example:
#   sbatch --export=ALL,SOLVER=learned_primal_dual,\
#          SLUG=mayo-ldct-claude-agentic-lpd-search-20260603-01,\
#          ITER=3,\
#          CFG_JSON=/cluster/maier/Agent4CT/agentic_cfgs/mayo_lpd_iter_03.json \
#          cluster/slurm/mayo_agentic_iter.sbatch

set -euo pipefail
: "${SOLVER:?need SOLVER (e.g. learned_primal_dual)}"
: "${SLUG:?need SLUG (e.g. mayo-ldct-claude-agentic-<solver>-search-…)}"
: "${ITER:?need ITER (≥2)}"
: "${CFG_JSON:?need CFG_JSON path}"

REPO=/cluster/maier/Agent4CT
cd "$REPO"
VENV="$REPO/.venv"
export CUDA_HOME="$VENV/cuda-pip-bundle"
export PATH="$CUDA_HOME/bin:$PATH"
source "$VENV/bin/activate"
export PYTHONPATH="$REPO:${PYTHONPATH:-}"
export HDF5_USE_FILE_LOCKING=FALSE

echo "[agentic-iter] host=$(hostname) job=$SLURM_JOB_ID"
echo "[agentic-iter] solver=$SOLVER  slug=$SLUG  iter=$ITER  cfg=$CFG_JSON"
nvidia-smi -L
echo
cat "$CFG_JSON"; echo

SLUG="$SLUG" ITER_N="$ITER" SOLVER="$SOLVER" CFG_JSON="$CFG_JSON" \
  AGENT4CT_DATASET=mayo_ldct_2d \
  python scripts/claude_agentic_one_iter.py

echo "[agentic-iter] done."
