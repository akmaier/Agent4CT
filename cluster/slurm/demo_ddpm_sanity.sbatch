#!/bin/bash
#SBATCH --job-name=demo-ddpm-sanity
#SBATCH --partition=main
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --exclude=lme49,lme53
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/cluster/maier/Agent4CT/results/slurm/%x-%j.out
#SBATCH --error=/cluster/maier/Agent4CT/results/slurm/%x-%j.err

# Sample 6 images unconditionally from both DDPM checkpoints,
# save side-by-side PNGs. Quick — only ~5-10 min.

set -euo pipefail
REPO=/cluster/maier/Agent4CT
cd "$REPO"
mkdir -p results/slurm
VENV="$REPO/.venv"
export CUDA_HOME="$VENV/cuda-pip-bundle"
export PATH="$CUDA_HOME/bin:$PATH"
source "$VENV/bin/activate"
export PYTHONPATH="$REPO:${PYTHONPATH:-}"

echo "[ddpm-sanity] host=$(hostname) job=$SLURM_JOB_ID"
nvidia-smi -L

OUT=results/ddpm-sanity-${SLURM_JOB_ID}
mkdir -p "$OUT/unconstrained" "$OUT/constrained"

python scripts/ddpm_sanity_sample.py \
    /cluster/maier/Agent4CT/checkpoints/ddpm_unconstrained_final.pt \
    "$OUT/unconstrained" 100

python scripts/ddpm_sanity_sample.py \
    /cluster/maier/Agent4CT/checkpoints/ddpm_constrained_final.pt \
    "$OUT/constrained" 100

echo "[ddpm-sanity] done. PNGs in $OUT/{unconstrained,constrained}/"
