#!/bin/bash
#SBATCH --job-name=validate-mayo-L014-ffs-drho
#SBATCH --partition=main
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --exclude=lme49,lme53,lme171
#SBATCH --mem=24G
#SBATCH --time=00:20:00
#SBATCH --output=/cluster/maier/Agent4CT/results/slurm/%x-%j.out
#SBATCH --error=/cluster/maier/Agent4CT/results/slurm/%x-%j.err

# Validate the FFS-drho-corrected L014 fulldose rebin
# (L014_sino_fulldose_ffs_drho.h5) at the same +3.5-mm slab anchor that
# the baseline used (SLURM 762115: SSIM 0.9445 / PSNR 37.23 dB).

set -euo pipefail
REPO=/cluster/maier/Agent4CT
cd "$REPO"
VENV="$REPO/.venv"
export CUDA_HOME="$VENV/cuda-pip-bundle"
export PATH="$CUDA_HOME/bin:$PATH"
source "$VENV/bin/activate"
export PYTHONPATH="$REPO:${PYTHONPATH:-}"

echo "[validate-ffs-drho] host=$(hostname) job=$SLURM_JOB_ID"
nvidia-smi -L || true
mkdir -p "$REPO/results/breast_debug"
python scripts/validate_mayo_helix2fan.py --patient L014 --dose fulldose \
    --sino-suffix _ffs_drho \
    --out-png "$REPO/results/breast_debug/validate_L014_fulldose_ffs_drho.png"
ls -la "$REPO/results/breast_debug/validate_L014_fulldose_ffs_drho.png"
echo "[validate-ffs-drho] done."
