#!/bin/bash
#SBATCH --job-name=demo-dl-hammernik-search
#SBATCH --partition=main
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --exclude=lme49
#SBATCH --mem=24G
#SBATCH --time=02:30:00
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

echo "[hammernik-search] host=$(hostname) job=$SLURM_JOB_ID"
nvidia-smi -L
python -c "import torch; print(f'torch={torch.__version__} cuda={torch.cuda.is_available()}')"

python scripts/learned_solver_search_agent.py --solver hammernik --iterations 20 \
    --notes "Hammernik 2017 variational network hyperparameter search: T, N_k, kernel, lr, lambda_init, epochs"

echo "[hammernik-search] done."
