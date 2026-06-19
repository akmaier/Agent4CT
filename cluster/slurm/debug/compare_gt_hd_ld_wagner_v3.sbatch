#!/bin/bash
#SBATCH --job-name=cmp-wagner-v3
#SBATCH --partition=main
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --exclude=lme49,lme50,lme51,lme52,lme53,lme170,lme171
#SBATCH --output=/cluster/maier/Agent4CT/results/slurm/%x-%j.out
#SBATCH --error=/cluster/maier/Agent4CT/results/slurm/%x-%j.err

# GT vs HD-FBP vs LD-FBP central-slice comparison across all 10 Wagner
# Mayo patients, using the v3-final geometry. Reads from the v3-rebinned
# per-patient sinos at staged_helix2fan_v3/ and produces a 10-row × 3-col
# panel at results/mayo_debug/wagner_gt_hd_ld_fbp_v3.png + a JSON of
# per-patient SSIM / PSNR. Dispatched as a SLURM dependency of the
# rebin-v3 job so it fires automatically once v3 staging completes.

set -euo pipefail
REPO=/cluster/maier/Agent4CT
cd "$REPO"
mkdir -p results/slurm results/mayo_debug
VENV="$REPO/.venv"
export CUDA_HOME="$VENV/cuda-pip-bundle"
export PATH="$CUDA_HOME/bin:$PATH"
. "$VENV/bin/activate"
export PYTHONPATH="$REPO:${PYTHONPATH:-}"
export HDF5_USE_FILE_LOCKING=FALSE
export STAGED_HELIX2FAN_SUBDIR=staged_helix2fan_v3

echo "[cmp-wagner-v3] host=$(hostname) job=$SLURM_JOB_ID"
nvidia-smi -L

python -u scripts/compare_gt_hd_ld_fbp_wagner.py --tag v3
echo "[cmp-wagner-v3] done."
