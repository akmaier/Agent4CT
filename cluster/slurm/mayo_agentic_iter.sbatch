#!/bin/bash
#SBATCH --job-name=mayo-agentic-iter
#SBATCH --partition=main
#SBATCH --qos=turbo
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --exclude=lme49,lme50,lme51,lme52,lme170   # 2026-06-29: lme53 (v100 16GB) re-enabled after driver update + smoke test; lme49 still down (unk*), lme50/51/52 too small (gtx1080ti/titanxp 11-12GB), lme170 blacklisted
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH --output=/cluster/maier/Agent4CT/results/slurm/%x-%j.out
#SBATCH --error=/cluster/maier/Agent4CT/results/slurm/%x-%j.err

# Generic Mayo agentic iter dispatcher (autoresearch loop).
#
# BUDGET (2026-06-19 reset — see README + solver_plan.md): each iteration has a
# HARD 20-MINUTE COMPUTE BUDGET for train + val-score. The driving agent MUST
# size the config (epochs / n_iter / per-image steps) so train+score fits
# <= 20 min. The val metric scores ALL 214 L277 slices with the detector-
# geometry FOV (321px). Per-image solvers set outer_wall_s=1200 as a backstop.
# The val FIGURE is rendered AFTER scoring and is NOT part of the 20-min budget;
# the heavy 6-patient leaderboard montage is a SEPARATE job (mayo_showcase.sbatch).
# --time below = 20-min budget + headroom for the figure + I/O. NO per-solver
# exceptions (the old LPD 5h --time override is gone — LPD must fit 20 min too).
#
# Required exported variables:
#   SOLVER    — solver key from claude_agentic_one_iter.py:SOLVER_MAP
#   SLUG      — autoresearch slug (e.g. mayo-ldct-claude-agentic-lpd-search-…)
#   ITER      — iter number
#   CFG_JSON  — path to the per-iter config JSON Claude wrote
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
