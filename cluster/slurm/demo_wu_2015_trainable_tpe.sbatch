#!/bin/bash
#SBATCH --job-name=demo-wu-trainable-tpe
#SBATCH --partition=main
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --exclude=lme49,lme53,lme170
#SBATCH --mem=24G
#SBATCH --time=08:00:00
#SBATCH --output=/cluster/maier/Agent4CT/results/slurm/%x-%j.out
#SBATCH --error=/cluster/maier/Agent4CT/results/slurm/%x-%j.err

# Demo-DL Wu 2015 trainable TPE search to fill the gap vs breast-ct
# (where the 10-trainable-scalar Wu hit hr=0.22). Demo-DL only had the
# non-trainable Wu before.

set -euo pipefail
REPO=/cluster/maier/Agent4CT
cd "$REPO"
mkdir -p results/slurm
VENV="$REPO/.venv"
export CUDA_HOME="$VENV/cuda-pip-bundle"
export PATH="$CUDA_HOME/bin:$PATH"
source "$VENV/bin/activate"
export PYTHONPATH="$REPO:${PYTHONPATH:-}"

echo "[demo-wu-trainable-tpe] host=$(hostname) job=$SLURM_JOB_ID"
nvidia-smi -L

python scripts/learned_solver_search_agent.py \
    --solver wu_2015_trainable \
    --sampler tpe \
    --calibrated \
    --iterations 20 \
    --tpe-startup 5 \
    --notes "Wu 2015 trainable TPE on demo-DL; fills gap vs breast-ct (where it hit hr=0.22 at 10 params)"

echo "[demo-wu-trainable-tpe] done."
