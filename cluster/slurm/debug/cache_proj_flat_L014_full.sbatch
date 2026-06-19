#!/bin/bash
#SBATCH --job-name=cache-proj-flat-L014-full
#SBATCH --partition=main
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:0
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH --exclude=lme49,lme50,lme51,lme52,lme53,lme170,lme171
#SBATCH --output=/cluster/maier/Agent4CT/results/slurm/%x-%j.out
#SBATCH --error=/cluster/maier/Agent4CT/results/slurm/%x-%j.err

set -euo pipefail
REPO=/cluster/maier/Agent4CT
cd "$REPO"
VENV="$REPO/.venv"
. "$VENV/bin/activate"
export PYTHONPATH="$REPO:${PYTHONPATH:-}"

echo "[cache-full] host=$(hostname)"
python -u scripts/cache_proj_flat_L014_full.py
echo "[cache-full] done."
ls -la /cluster/maier/Agent4CT/data/mayo_ldct/staged_helix2fan/L014_proj_flat_full.pt
