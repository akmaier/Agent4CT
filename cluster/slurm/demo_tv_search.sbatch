#!/bin/bash
#SBATCH --job-name=demo-dl-tv-search
#SBATCH --partition=main
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --exclude=lme49
#SBATCH --mem=24G
#SBATCH --time=00:30:00
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

echo "[tv-search] host=$(hostname) job=$SLURM_JOB_ID"
nvidia-smi -L
python -c "import torch; print(f'torch={torch.__version__} cuda={torch.cuda.is_available()}')"

# Run the TV search agent for 20 iterations
python scripts/tv_search_agent_standalone.py --iterations 20 \
    --notes "TV hyperparameter random search: lambda, lr, iters, clip, decay"

echo "[tv-search] done."
