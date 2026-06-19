#!/bin/bash
#SBATCH --job-name=ssim-validate
#SBATCH --partition=main
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --mem=12G
#SBATCH --time=00:05:00
#SBATCH --output=/cluster/maier/Agent4CT/results/slurm/%x-%j.out

set -euo pipefail
cd /cluster/maier/Agent4CT
source .venv/bin/activate
export PYTHONPATH="/cluster/maier/Agent4CT${PYTHONPATH:+:$PYTHONPATH}"

python scripts/validate_ssim.py > results/slurm/ssim-validate-${SLURM_JOB_ID}.txt
cat results/slurm/ssim-validate-${SLURM_JOB_ID}.txt
echo "[validate] done."
