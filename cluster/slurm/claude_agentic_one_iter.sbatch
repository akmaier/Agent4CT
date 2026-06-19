#!/bin/bash
#SBATCH --job-name=cl-ag-iter
#SBATCH --partition=main
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --exclude=lme49,lme52,lme53,lme170,lme171
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH --output=/cluster/maier/Agent4CT/results/slurm/%x-%j.out
#SBATCH --error=/cluster/maier/Agent4CT/results/slurm/%x-%j.err

# One iteration of a Claude-driven agentic parameter search.
# Required env: SLUG, ITER_N, SOLVER, CFG_JSON, AGENT4CT_DATASET.

set -euo pipefail
REPO=/cluster/maier/Agent4CT
cd "$REPO"
VENV="$REPO/.venv"
export CUDA_HOME="$VENV/cuda-pip-bundle"
export PATH="$CUDA_HOME/bin:$PATH"
source "$VENV/bin/activate"
export PYTHONPATH="$REPO:${PYTHONPATH:-}"
# Disable HDF5 file locking: the staged H5s live on NFS and concurrent
# agentic jobs (e.g. two breast_ct N2I solvers) opening the same truth.h5
# read-only otherwise fail with "errno 37, No locks available". The Mayo
# sbatch already sets this; the generic dispatcher must too.
export HDF5_USE_FILE_LOCKING=FALSE

: "${SLUG:?need SLUG}"
: "${ITER_N:?need ITER_N}"
: "${SOLVER:?need SOLVER}"
: "${CFG_JSON:?need CFG_JSON}"
: "${AGENT4CT_DATASET:?need AGENT4CT_DATASET}"

echo "[claude-agentic] host=$(hostname) job=$SLURM_JOB_ID slug=$SLUG iter=$ITER_N solver=$SOLVER dataset=$AGENT4CT_DATASET"
nvidia-smi -L || true
python scripts/claude_agentic_one_iter.py
echo "[claude-agentic] done."
