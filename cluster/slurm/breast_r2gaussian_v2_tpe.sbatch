#!/bin/bash
#SBATCH --job-name=breast-r2g-v2-tpe
#SBATCH --partition=main
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --exclude=lme49,lme53,lme50,lme51,lme52
#SBATCH --mem=24G
#SBATCH --time=10:00:00
#SBATCH --output=/cluster/maier/Agent4CT/results/slurm/%x-%j.out
#SBATCH --error=/cluster/maier/Agent4CT/results/slurm/%x-%j.err

# Re-run R²-Gaussian TPE on breast-CT with extended iteration budget.
# Previous TPE on breast-ct (slug breast-ct-calibrated-tpe-r2gaussian-
# search-20260521-01) hit hr=0 across all 20 trials, but the inner
# n_iter was capped at 800. Paper reports 20k-40k iters for X3D-class
# phantoms. Updated bounds: gs_n_iter ∈ [10000, 40000].
# Slug: `breast-ct-calibrated-tpe-r2gaussian-v2-search-…`.

set -euo pipefail
REPO=/cluster/maier/Agent4CT
cd "$REPO"
mkdir -p results/slurm
VENV="$REPO/.venv"
export CUDA_HOME="$VENV/cuda-pip-bundle"
export PATH="$CUDA_HOME/bin:$PATH"
source "$VENV/bin/activate"
export PYTHONPATH="$REPO:${PYTHONPATH:-}"

echo "[breast-r2g-v2-tpe] host=$(hostname) job=$SLURM_JOB_ID"
nvidia-smi -L

python scripts/learned_solver_search_agent.py \
    --solver r2gaussian \
    --sampler tpe \
    --calibrated \
    --dataset breast_ct \
    --iterations 20 \
    --tpe-startup 5 \
    --notes "R²-Gaussian TPE on breast-CT, extended n_iter bound to [10000, 40000] (was [300, 800])"

echo "[breast-r2g-v2-tpe] done."
