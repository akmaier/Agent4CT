#!/bin/bash
#SBATCH --job-name=breast-diff-recon-dcstep-tpe
#SBATCH --partition=main
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --exclude=lme49,lme53,lme170,lme171
#SBATCH --mem=24G
#SBATCH --time=10:00:00
#SBATCH --output=/cluster/maier/Agent4CT/results/slurm/%x-%j.out
#SBATCH --error=/cluster/maier/Agent4CT/results/slurm/%x-%j.err

# TPE search over DPS + DC-step diff-recon hyperparameters on breast-CT
# using the BREAST-TRAINED DDPM checkpoint (not the demo-phantom one
# the earlier diff-recon-dcstep TPE runs mistakenly used). Pass
# --variant=unconstrained or --variant=constrained to pick the
# checkpoint family.
#
# Usage:
#   sbatch --export=ALL,VARIANT=unconstrained breast_diff_recon_dcstep_tpe.sbatch
#   sbatch --export=ALL,VARIANT=constrained   breast_diff_recon_dcstep_tpe.sbatch

set -euo pipefail
: "${VARIANT:?need VARIANT=unconstrained|constrained}"
case "$VARIANT" in
  unconstrained) SOLVER=diffusion_recon_dcstep_unconstrained_breast ;;
  constrained)   SOLVER=diffusion_recon_dcstep_constrained_breast   ;;
  *) echo "VARIANT must be unconstrained|constrained, got $VARIANT" >&2; exit 1 ;;
esac

REPO=/cluster/maier/Agent4CT
cd "$REPO"
mkdir -p results/slurm
VENV="$REPO/.venv"
export CUDA_HOME="$VENV/cuda-pip-bundle"
export PATH="$CUDA_HOME/bin:$PATH"
source "$VENV/bin/activate"
export PYTHONPATH="$REPO:${PYTHONPATH:-}"

echo "[breast-diff-recon-dcstep-tpe] host=$(hostname) job=$SLURM_JOB_ID variant=$VARIANT solver=$SOLVER"
nvidia-smi -L

python scripts/learned_solver_search_agent.py \
    --solver "$SOLVER" \
    --sampler tpe \
    --calibrated \
    --dataset breast_ct \
    --iterations 20 \
    --tpe-startup 5 \
    --notes "Breast-DDPM diff-recon dcstep TPE ($VARIANT) on breast-CT; uses the breast-trained checkpoint instead of the demo one the earlier TPE runs mistakenly used."

echo "[breast-diff-recon-dcstep-tpe] done."
