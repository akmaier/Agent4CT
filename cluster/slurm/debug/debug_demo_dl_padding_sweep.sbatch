#!/bin/bash
#SBATCH --job-name=debug-padsweep
#SBATCH --partition=main
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --exclude=lme49,lme53,lme171
#SBATCH --mem=24G
#SBATCH --time=00:20:00
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

echo "[padsweep] host=$(hostname) job=$SLURM_JOB_ID"
nvidia-smi -L || true
python scripts/debug_demo_dl_padding_sweep.py
echo "[padsweep] done."
