#!/bin/bash
#SBATCH --job-name=train-fast-diffusion-mayo
#SBATCH --partition=main
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --exclude=lme49,lme50,lme51,lme52,lme53,lme170,lme171,lme221
#SBATCH --mem=40G
#SBATCH --time=12:00:00
#SBATCH --output=/cluster/maier/Agent4CT/results/slurm/%x-%j.out
#SBATCH --error=/cluster/maier/Agent4CT/results/slurm/%x-%j.err

# Train ONE fast-diffusion (flow-matching) prior variant on Mayo GT.
#
# The 4 advertised solvers = 4 checkpoints = this script run 4 times with
# different env:
#   FD_DOMAIN ∈ {pixel, wavelet}   x   FD_MODE ∈ {unconstrained, constrained}
#
# Env knobs (all optional, defaults below):
#   FD_DOMAIN   pixel | wavelet                          (default: pixel)
#   FD_MODE     unconstrained | constrained              (default: unconstrained)
#   FD_CKPT     output checkpoint path                   (default: derived)
#   FD_CONFIG   path to a JSON cfg to merge on top       (default: written here)
#               (equivalently set FAST_DIFFUSION_CONFIG_PATH directly)
#   FD_EPOCHS / FD_CH / FD_BATCH / FD_LR / FD_N_TRAIN / FD_N_TRAIN_CON /
#   FD_N_VAL / FD_EMA_DECAY / FD_WALL_S — override individual cfg fields.
#
# Example (submit all four):
#   for d in pixel wavelet; do for m in unconstrained constrained; do
#     FD_DOMAIN=$d FD_MODE=$m \
#     FD_CKPT=/cluster/maier/Agent4CT/checkpoints/fast_diffusion_${d}_${m}.pt \
#       sbatch cluster/slurm/train_fast_diffusion_mayo.sbatch
#   done; done

set -euo pipefail
REPO=/cluster/maier/Agent4CT
cd "$REPO"
VENV="$REPO/.venv"
export CUDA_HOME="$VENV/cuda-pip-bundle"
export PATH="$CUDA_HOME/bin:$PATH"
source "$VENV/bin/activate"
export PYTHONPATH="$REPO:${PYTHONPATH:-}"
export AGENT4CT_DATASET=mayo_ldct_2d

# ---- variant selection (env, with defaults) ------------------------------
FD_DOMAIN="${FD_DOMAIN:-pixel}"
FD_MODE="${FD_MODE:-unconstrained}"

CKPTDIR=/cluster/maier/Agent4CT/checkpoints
mkdir -p "$CKPTDIR"
FD_CKPT="${FD_CKPT:-$CKPTDIR/fast_diffusion_${FD_DOMAIN}_${FD_MODE}.pt}"

# Individual cfg field overrides (fall back to solver CONFIG defaults).
FD_EPOCHS="${FD_EPOCHS:-30}"
FD_CH="${FD_CH:-32}"
FD_BATCH="${FD_BATCH:-8}"
FD_LR="${FD_LR:-2e-4}"
FD_N_TRAIN="${FD_N_TRAIN:-3000}"
FD_N_TRAIN_CON="${FD_N_TRAIN_CON:-200}"
FD_N_VAL="${FD_N_VAL:-100}"
FD_EMA_DECAY="${FD_EMA_DECAY:-0.999}"
FD_WALL_S="${FD_WALL_S:-36000}"

nvidia-smi -L || true   # non-fatal: a driver/NVML hiccup must not abort under `set -e`

# ---- assemble the config -------------------------------------------------
# If FD_CONFIG (or FAST_DIFFUSION_CONFIG_PATH) is provided, use it verbatim;
# otherwise write a tmp cfg from the env fields above.
if [ -n "${FD_CONFIG:-}" ]; then
  CFG="$FD_CONFIG"
elif [ -n "${FAST_DIFFUSION_CONFIG_PATH:-}" ]; then
  CFG="$FAST_DIFFUSION_CONFIG_PATH"
else
  CFG="/tmp/_fast_diffusion_${FD_DOMAIN}_${FD_MODE}_${SLURM_JOB_ID}.json"
  cat > "$CFG" <<JEOF
{"fd_domain":"$FD_DOMAIN","fd_mode":"$FD_MODE","fd_n_train":$FD_N_TRAIN,"fd_n_train_constrained":$FD_N_TRAIN_CON,"fd_n_val":$FD_N_VAL,"fd_ch":$FD_CH,"fd_epochs":$FD_EPOCHS,"fd_batch":$FD_BATCH,"fd_lr":$FD_LR,"fd_ema_decay":$FD_EMA_DECAY,"fd_train_wall_s":$FD_WALL_S,"fd_ckpt":"$FD_CKPT"}
JEOF
fi

echo "[fast-diffusion] domain=$FD_DOMAIN mode=$FD_MODE ckpt=$FD_CKPT cfg=$CFG"
cat "$CFG"

OUT=$REPO/runs/train-fast-diffusion-${FD_DOMAIN}-${FD_MODE}-${SLURM_JOB_ID}
mkdir -p "$OUT"

FAST_DIFFUSION_CONFIG_PATH="$CFG" \
  python pentathlon/demo_dl_reference/solver_fast_diffusion.py "$OUT"

ls -la "$CKPTDIR" | grep -E "fast_diffusion_${FD_DOMAIN}_${FD_MODE}" || true
echo "[fast-diffusion] done."
