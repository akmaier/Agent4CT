#!/bin/bash
#SBATCH --job-name=train-ddpm-breast-v2
#SBATCH --partition=main
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --exclude=lme49,lme53,lme50,lme51,lme52
#SBATCH --mem=32G
#SBATCH --time=20:00:00
#SBATCH --output=/cluster/maier/Agent4CT/results/slurm/%x-%j.out
#SBATCH --error=/cluster/maier/Agent4CT/results/slurm/%x-%j.err

# Retrain breast-DDPM checkpoints with LONGER schedule + BIGGER U-Net.
# The original 2026-05-20 checkpoints (ddpm_breast_*_final.pt) at ch=32,
# 25 epochs gave SSIM ~0.46 in the diff-recon TPE — visibly under-trained.
# Diagnosis in findings.md 2026-05-23 ("Breast-trained DDPM checkpoints
# EXIST but the diff-recon TPE-seed-config gives hr=0").
#
# This v2 retrain:
#   * ddpm_ch: 32 → 64 (2× channels, 4× params; ~15 M total)
#   * ddpm_epochs: 25 → 80 (3.2× training)
#   * ddpm_train_wall_s: 9000 → 32400 (9 h per variant; total 18 h)
#   * Output: ddpm_breast_{,un}constrained_v2.pt (new files; legacy v1
#     stay on disk for A/B comparison)
#
# After this lands, re-run breast-ct diff-recon TPE pointing at the v2
# checkpoints. Slug `breast-ct-calibrated-tpe-diff-recon-dcstep-
# {,un}constrained-breast-v2-search-...`.

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

# 1) UNCONSTRAINED v2 — all 3600 breast phantoms, ch=64, 80 epochs
CFG1=/tmp/_ddpm_breast_uncon_v2_${SLURM_JOB_ID}.json
cat > "$CFG1" <<JEOF
{"ddpm_mode":"unconstrained","ddpm_n_train":3600,"ddpm_n_train_constrained":200,"ddpm_n_val":40,"ddpm_ch":64,"ddpm_n_steps":1000,"ddpm_epochs":80,"ddpm_batch":8,"ddpm_lr":2e-4,"ddpm_weight_decay":0,"ddpm_train_wall_s":32400,"ddpm_ckpt":"$CKPT/ddpm_breast_unconstrained_v2.pt"}
JEOF
echo "[ddpm-breast-v2] === UNCONSTRAINED (ch=64, ep=80) ==="
OUT1=$REPO/runs/train-ddpm-breast-uncon-v2-${SLURM_JOB_ID}
mkdir -p "$OUT1"
DDPM_CONFIG_PATH=$CFG1 python pentathlon/demo_dl_reference/solver_ddpm.py "$OUT1"

# 2) CONSTRAINED v2 — 200-image subset, ch=64, 80 epochs
CFG2=/tmp/_ddpm_breast_con_v2_${SLURM_JOB_ID}.json
cat > "$CFG2" <<JEOF
{"ddpm_mode":"constrained","ddpm_n_train":3600,"ddpm_n_train_constrained":200,"ddpm_n_val":40,"ddpm_ch":64,"ddpm_n_steps":1000,"ddpm_epochs":80,"ddpm_batch":8,"ddpm_lr":2e-4,"ddpm_weight_decay":0,"ddpm_train_wall_s":32400,"ddpm_ckpt":"$CKPT/ddpm_breast_constrained_v2.pt"}
JEOF
echo "[ddpm-breast-v2] === CONSTRAINED (ch=64, ep=80) ==="
OUT2=$REPO/runs/train-ddpm-breast-con-v2-${SLURM_JOB_ID}
mkdir -p "$OUT2"
DDPM_CONFIG_PATH=$CFG2 python pentathlon/demo_dl_reference/solver_ddpm.py "$OUT2"

ls -la "$CKPT" | grep -E "ddpm_breast_(un)?constrained_v2"
echo "[ddpm-breast-v2] done."
