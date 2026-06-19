#!/bin/bash
#SBATCH --job-name=debug-breast-ct-geom
#SBATCH --partition=main
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --exclude=lme49,lme53
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH --output=/cluster/maier/Agent4CT/results/slurm/%x-%j.out
#SBATCH --error=/cluster/maier/Agent4CT/results/slurm/%x-%j.err

set -euo pipefail
REPO=/cluster/maier/Agent4CT
cd "$REPO"
VENV="$REPO/.venv"
export CUDA_HOME="$VENV/cuda-pip-bundle"
export PATH="$CUDA_HOME/bin:$PATH"
source "$VENV/bin/activate"
export PYTHONPATH="$REPO:${PYTHONPATH:-}"

echo "[debug] host=$(hostname) job=$SLURM_JOB_ID"
nvidia-smi -L
python scripts/debug_breast_ct_geometry.py
ls -la /tmp/_breast_debug/
echo "[debug] done."
