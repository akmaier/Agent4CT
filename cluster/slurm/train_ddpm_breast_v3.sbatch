#!/bin/bash
#SBATCH --job-name=train-ddpm-breast-v3
#SBATCH --partition=main
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --exclude=lme49,lme53,lme50,lme51,lme52,lme170
#SBATCH --mem=40G
#SBATCH --time=22:00:00
#SBATCH --output=/cluster/maier/Agent4CT/results/slurm/%x-%j.out
#SBATCH --error=/cluster/maier/Agent4CT/results/slurm/%x-%j.err

# Agentic-iter HYPOTHESIS: the breast-DDPM ε-loss converges fine
# (train_loss 0.79 → 0.003 at ch=64), but the SAMPLES that come out
# of DPS/DC posterior sampling are still SSIM ~0.4 — well below
# baseline FBP 0.957. Both v1 (ch=32 / 25 ep) and v2 (ch=64 / 80 ep)
# show the same failure, so the issue is NOT under-training of a
# given architecture — it's the architecture/capacity itself.
#
# v3 hypothesis: SmallDDPM at ch=64 = 3.8M params is still too small
# for breast-phantom complexity (~100× more structure than the
# ellipse-phantom DDPM that succeeds at hr=0.45). Going to ch=128
# is 4× more channels = ~15M params (still tiny by DDPM standards).
# If diff-recon STILL gives hr=0 against this v3 ckpt, the problem
# is not capacity at all and we'll need a structurally different
# prior (e.g. score-SDE, U-ViT, or train a CT-specific prior).
#
# Per-variant wall: ch=128 ≈ 4× v2 wall per epoch + 60 ep budget.
# Both variants together fit in ~20h on Q5000/Q6000.

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

# 1) UNCONSTRAINED v3 — all 3600 breast phantoms, ch=128, 60 epochs
CFG1=/tmp/_ddpm_breast_uncon_v3_${SLURM_JOB_ID}.json
cat > "$CFG1" <<JEOF
{"ddpm_mode":"unconstrained","ddpm_n_train":3600,"ddpm_n_train_constrained":200,"ddpm_n_val":40,"ddpm_ch":128,"ddpm_n_steps":1000,"ddpm_epochs":60,"ddpm_batch":4,"ddpm_lr":1e-4,"ddpm_weight_decay":0,"ddpm_train_wall_s":36000,"ddpm_ckpt":"$CKPT/ddpm_breast_unconstrained_v3.pt"}
JEOF
echo "[ddpm-breast-v3] === UNCONSTRAINED (ch=128, ep=60, batch=4, lr=1e-4) ==="
OUT1=$REPO/runs/train-ddpm-breast-uncon-v3-${SLURM_JOB_ID}
mkdir -p "$OUT1"
DDPM_CONFIG_PATH=$CFG1 python pentathlon/demo_dl_reference/solver_ddpm.py "$OUT1"

# 2) CONSTRAINED v3 — 200-image subset, ch=128, 60 epochs
CFG2=/tmp/_ddpm_breast_con_v3_${SLURM_JOB_ID}.json
cat > "$CFG2" <<JEOF
{"ddpm_mode":"constrained","ddpm_n_train":3600,"ddpm_n_train_constrained":200,"ddpm_n_val":40,"ddpm_ch":128,"ddpm_n_steps":1000,"ddpm_epochs":60,"ddpm_batch":4,"ddpm_lr":1e-4,"ddpm_weight_decay":0,"ddpm_train_wall_s":36000,"ddpm_ckpt":"$CKPT/ddpm_breast_constrained_v3.pt"}
JEOF
echo "[ddpm-breast-v3] === CONSTRAINED (ch=128, ep=60, batch=4, lr=1e-4) ==="
OUT2=$REPO/runs/train-ddpm-breast-con-v3-${SLURM_JOB_ID}
mkdir -p "$OUT2"
DDPM_CONFIG_PATH=$CFG2 python pentathlon/demo_dl_reference/solver_ddpm.py "$OUT2"

ls -la "$CKPT" | grep -E "ddpm_breast_(un)?constrained_v3"
echo "[ddpm-breast-v3] done."
