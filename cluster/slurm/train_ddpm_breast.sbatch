#!/bin/bash
#SBATCH --job-name=train-ddpm-breast
#SBATCH --partition=main
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --exclude=lme49,lme53
#SBATCH --mem=24G
#SBATCH --time=06:00:00
#SBATCH --output=/cluster/maier/Agent4CT/results/slurm/%x-%j.out
#SBATCH --error=/cluster/maier/Agent4CT/results/slurm/%x-%j.err

# Track B.2: train DDPM checkpoints on Sidky 2021 breast phantoms.
# Two checkpoints saved sequentially in one job:
#   1) ddpm_breast_unconstrained_final.pt  (all 3600 train images)
#   2) ddpm_breast_constrained_final.pt    (200 train images subset)
# Each at ~2-3 h on Q8000; 6 h walltime is generous.

set -euo pipefail
REPO=/cluster/maier/Agent4CT
cd "$REPO"
VENV="$REPO/.venv"
export CUDA_HOME="$VENV/cuda-pip-bundle"
export PATH="$CUDA_HOME/bin:$PATH"
source "$VENV/bin/activate"
export PYTHONPATH="$REPO:${PYTHONPATH:-}"
export AGENT4CT_DATASET=breast_ct

CKPT=/cluster/maier/Agent4CT/checkpoints
mkdir -p "$CKPT"
nvidia-smi -L

# 1) UNCONSTRAINED — all 3600 breast phantoms
CFG1=/tmp/_ddpm_breast_uncon_${SLURM_JOB_ID}.json
cat > "$CFG1" <<JEOF
{"ddpm_mode":"unconstrained","ddpm_n_train":3600,"ddpm_n_train_constrained":200,"ddpm_n_val":40,"ddpm_ch":32,"ddpm_n_steps":1000,"ddpm_epochs":25,"ddpm_batch":8,"ddpm_lr":2e-4,"ddpm_weight_decay":0,"ddpm_train_wall_s":9000,"ddpm_save_path":"$CKPT/ddpm_breast_unconstrained_final.pt"}
JEOF
echo "[ddpm-breast] === UNCONSTRAINED ==="
OUT1=$REPO/runs/train-ddpm-breast-uncon-${SLURM_JOB_ID}
mkdir -p "$OUT1"
DDPM_CONFIG_PATH=$CFG1 python pentathlon/demo_dl_reference/solver_ddpm.py "$OUT1"

# 2) CONSTRAINED — 200-image subset
CFG2=/tmp/_ddpm_breast_con_${SLURM_JOB_ID}.json
cat > "$CFG2" <<JEOF
{"ddpm_mode":"constrained","ddpm_n_train":3600,"ddpm_n_train_constrained":200,"ddpm_n_val":40,"ddpm_ch":32,"ddpm_n_steps":1000,"ddpm_epochs":25,"ddpm_batch":8,"ddpm_lr":2e-4,"ddpm_weight_decay":0,"ddpm_train_wall_s":9000,"ddpm_save_path":"$CKPT/ddpm_breast_constrained_final.pt"}
JEOF
echo "[ddpm-breast] === CONSTRAINED ==="
OUT2=$REPO/runs/train-ddpm-breast-con-${SLURM_JOB_ID}
mkdir -p "$OUT2"
DDPM_CONFIG_PATH=$CFG2 python pentathlon/demo_dl_reference/solver_ddpm.py "$OUT2"

ls -la "$CKPT" | grep breast
echo "[ddpm-breast] done."
