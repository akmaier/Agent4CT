#!/bin/bash
#SBATCH --job-name=demo-dd-unet-sup-tpe
#SBATCH --partition=main
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --exclude=lme49,lme53,lme170
#SBATCH --mem=24G
#SBATCH --time=08:00:00
#SBATCH --output=/cluster/maier/Agent4CT/results/slurm/%x-%j.out
#SBATCH --error=/cluster/maier/Agent4CT/results/slurm/%x-%j.err

# Demo-DL DD-UNet supervised-L2 TPE search to fill the gap vs breast-ct
# (where this is #2 at hr=0.836). Demo-DL only had the N2I-loss variant
# before. Per-trial wall ~15-20 min × 20 trials = ~7 h.

set -euo pipefail
REPO=/cluster/maier/Agent4CT
cd "$REPO"
mkdir -p results/slurm
VENV="$REPO/.venv"
export CUDA_HOME="$VENV/cuda-pip-bundle"
export PATH="$CUDA_HOME/bin:$PATH"
source "$VENV/bin/activate"
export PYTHONPATH="$REPO:${PYTHONPATH:-}"

echo "[demo-dd-unet-sup-tpe] host=$(hostname) job=$SLURM_JOB_ID"
nvidia-smi -L

python scripts/learned_solver_search_agent.py \
    --solver dual_domain_supervised \
    --sampler tpe \
    --calibrated \
    --iterations 20 \
    --tpe-startup 5 \
    --notes "DD-UNet supervised L2 TPE on demo-DL; fills gap vs breast-ct (where this is hr=0.836)"

echo "[demo-dd-unet-sup-tpe] done."
