#!/bin/bash
#SBATCH --job-name=demo-dl-itnet-v2-search
#SBATCH --partition=main
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:q8000:1
#SBATCH --exclude=lme171,lme222,lme223,lme49,lme50,lme51,lme52
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --output=/cluster/maier/Agent4CT/results/slurm/%x-%j.out
#SBATCH --error=/cluster/maier/Agent4CT/results/slurm/%x-%j.err

set -euo pipefail
REPO=/cluster/maier/Agent4CT
cd "$REPO"
mkdir -p results/slurm

VENV="$REPO/.venv"
export CUDA_HOME="$VENV/cuda-pip-bundle"
export PATH="$CUDA_HOME/bin:$PATH"
source "$VENV/bin/activate"
export PYTHONPATH="$REPO:${PYTHONPATH:-}"

echo "[itnet-v2-search] host=$(hostname) job=$SLURM_JOB_ID"
nvidia-smi -L
python -c "import torch; print(f'torch={torch.__version__} cuda={torch.cuda.is_available()}')"

python scripts/learned_solver_search_agent.py --solver itnet_v2 --iterations 20 \
    --notes "ItNet v2 hyperparameter random search: pretrain_epochs, lr, k, unet_c, alpha, residual"

echo "[itnet-v2-search] done."
