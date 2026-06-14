#!/bin/bash
#SBATCH --job-name=stage-canon
#SBATCH --partition=main
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=40G
#SBATCH --time=03:00:00
#SBATCH --exclude=lme49,lme50,lme51,lme52,lme53,lme170,lme171,lme221
#SBATCH --output=/cluster/maier/Agent4CT/results/slurm/%x-%j.out
#SBATCH --error=/cluster/maier/Agent4CT/results/slurm/%x-%j.err
set -uo pipefail
REPO=/cluster/maier/Agent4CT; cd "$REPO"
export CUDA_HOME="$REPO/.venv/cuda-pip-bundle"; export PATH="$CUDA_HOME/bin:$PATH"
source "$REPO/.venv/bin/activate"
export PYTHONPATH="$REPO:${PYTHONPATH:-}"; export HDF5_USE_FILE_LOCKING=FALSE
export AGENT4CT_DATASET=mayo_ldct_2d
nvidia-smi -L || true
python data/stage_mayo_canonical.py --splits "${SPLITS:-val}" --doses "${DOSES:-lowdose,fulldose}" ${VALIDATE:+--validate} ${FORCE:+--force} ${TRUTH_ONLY:+--truth-only}
echo "[stage-canon] done."
