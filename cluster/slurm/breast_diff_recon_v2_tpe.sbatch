#!/bin/bash
#SBATCH --job-name=breast-diff-recon-v2-tpe
#SBATCH --partition=main
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --exclude=lme49,lme53,lme170
#SBATCH --mem=24G
#SBATCH --time=14:00:00
#SBATCH --output=/cluster/maier/Agent4CT/results/slurm/%x-%j.out
#SBATCH --error=/cluster/maier/Agent4CT/results/slurm/%x-%j.err

# Re-run breast-ct diff-recon TPE against the v2 DDPM checkpoints
# (ch=64, 80 epochs; SLURM 762624, 2026-06-01). The v1 ckpts (ch=32,
# 25 epochs) gave SSIM ~0.46 — under-trained. The v2 ckpts have 4× the
# params and the val_eps_loss improved 0.005 → 0.003 — expected SSIM
# uplift to clear the FBP baseline (0.957).
#
# Runs both constrained and unconstrained in sequence in one sbatch
# (one TPE study each, 20 trials × ~25 min ≈ 8 h per variant → ~14h
# total; close to the wall limit).

set -euo pipefail
REPO=/cluster/maier/Agent4CT
cd "$REPO"
mkdir -p results/slurm
VENV="$REPO/.venv"
export CUDA_HOME="$VENV/cuda-pip-bundle"
export PATH="$CUDA_HOME/bin:$PATH"
source "$VENV/bin/activate"
export PYTHONPATH="$REPO:${PYTHONPATH:-}"

echo "[breast-diff-recon-v2-tpe] host=$(hostname) job=$SLURM_JOB_ID"
nvidia-smi -L

echo "=== UNCONSTRAINED-v2 ==="
python scripts/learned_solver_search_agent.py \
    --solver diffusion_recon_dcstep_unconstrained_breast_v2 \
    --sampler tpe \
    --calibrated \
    --dataset breast_ct \
    --iterations 20 \
    --tpe-startup 5 \
    --notes "breast-ct diff-recon DC-step TPE against v2 unconstrained DDPM (ch=64, 80ep, SLURM 762624)"

echo "=== CONSTRAINED-v2 ==="
python scripts/learned_solver_search_agent.py \
    --solver diffusion_recon_dcstep_constrained_breast_v2 \
    --sampler tpe \
    --calibrated \
    --dataset breast_ct \
    --iterations 20 \
    --tpe-startup 5 \
    --notes "breast-ct diff-recon DC-step TPE against v2 constrained DDPM (ch=64, 80ep, SLURM 762624)"

echo "[breast-diff-recon-v2-tpe] done."
