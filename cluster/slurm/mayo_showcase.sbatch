#!/bin/bash
#SBATCH --job-name=mayo-showcase
#SBATCH --partition=main
#SBATCH --qos=turbo
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --exclude=lme49,lme50,lme51,lme52,lme170   # 2026-06-29: lme53 (v100 16GB) re-enabled after driver update + smoke test; lme49 still down (unk*), lme50/51/52 too small (gtx1080ti/titanxp 11-12GB), lme170 blacklisted
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/cluster/maier/Agent4CT/results/slurm/%x-%j.out
#SBATCH --error=/cluster/maier/Agent4CT/results/slurm/%x-%j.err

# Regenerate the diverse TEST-set showcase figures for every Mayo leaderboard
# solver (one central slice per held-out test patient). One job loops all
# solvers (subprocess each); writes docs/runs/<search-slug>/test_showcase.png.
# Uses one of the 4 QOS slots for ~30-50 min; the search loop keeps the rest.

set -euo pipefail
REPO=/cluster/maier/Agent4CT
cd "$REPO"
VENV="$REPO/.venv"
export CUDA_HOME="$VENV/cuda-pip-bundle"
export PATH="$CUDA_HOME/bin:$PATH"
source "$VENV/bin/activate"
export PYTHONPATH="$REPO:${PYTHONPATH:-}"
export HDF5_USE_FILE_LOCKING=FALSE

echo "[showcase] host=$(hostname) job=$SLURM_JOB_ID"
nvidia-smi -L
echo

python scripts/make_test_showcase.py

echo "[showcase] done."
