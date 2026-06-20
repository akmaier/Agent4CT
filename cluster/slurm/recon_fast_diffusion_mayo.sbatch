#!/bin/bash
#SBATCH --job-name=recon-fast-diffusion-mayo
#SBATCH --partition=main
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --exclude=lme49,lme50,lme51,lme52,lme53,lme170,lme171,lme221
#SBATCH --mem=40G
#SBATCH --time=02:00:00
#SBATCH --output=/cluster/maier/Agent4CT/results/slurm/%x-%j.out
#SBATCH --error=/cluster/maier/Agent4CT/results/slurm/%x-%j.err
#
# Few-step DC-guided reconstruction eval from a trained fast-diffusion prior.
# Runs solver_fast_recon.py on the Mayo L277 val split, writes result.json +
# comparison.png to an out dir. Env:
#   FAST_RECON_CKPT   path to a fast_diffusion_*.pt checkpoint (REQUIRED)
#   FR_VAL_N          number of L277 val slices (default 214)
#   FR_STEPS          Euler steps (default 8)
#   FR_DC             0|1 hard CG DC-step (default 1)
#   FR_INIT_T         FBP-embed flow time (default 0.7)
#   FR_ETA            DPS guidance scale (default 1.0)
#   FR_TAG            label for the out dir (default derived from ckpt name)

set -euo pipefail
REPO=/cluster/maier/Agent4CT
cd "$REPO"
VENV="$REPO/.venv"
export CUDA_HOME="$VENV/cuda-pip-bundle"
export PATH="$CUDA_HOME/bin:$PATH"
source "$VENV/bin/activate"
export PYTHONPATH="$REPO:${PYTHONPATH:-}"
export AGENT4CT_DATASET=mayo_ldct_2d

CKPT="${FAST_RECON_CKPT:?set FAST_RECON_CKPT to a fast_diffusion_*.pt}"
FR_VAL_N="${FR_VAL_N:-214}"
FR_STEPS="${FR_STEPS:-8}"
FR_DC="${FR_DC:-1}"
FR_INIT_T="${FR_INIT_T:-0.7}"
FR_ETA="${FR_ETA:-1.0}"
TAG="${FR_TAG:-$(basename "$CKPT" .pt)}"

nvidia-smi -L || true

CFG="/tmp/_fast_recon_${TAG}_${SLURM_JOB_ID}.json"
cat > "$CFG" <<JEOF
{"fd_recon_ckpt":"$CKPT","val_n":$FR_VAL_N,"fd_recon_steps":$FR_STEPS,"fd_recon_dc":$FR_DC,"fd_recon_init":"fbp","fd_recon_init_t":$FR_INIT_T,"fd_recon_eta":$FR_ETA}
JEOF
echo "[fast-recon] ckpt=$CKPT val_n=$FR_VAL_N steps=$FR_STEPS dc=$FR_DC"
cat "$CFG"

OUT=$REPO/runs/recon-fast-diffusion-${TAG}-${SLURM_JOB_ID}
mkdir -p "$OUT"

FAST_RECON_CONFIG_PATH="$CFG" FAST_RECON_CKPT="$CKPT" \
  python pentathlon/demo_dl_reference/solver_fast_recon.py "$OUT"

echo "[fast-recon] result:"; cat "$OUT/result.json" 2>/dev/null | python3 -c "import json,sys; d=json.load(sys.stdin); print({k:d[k] for k in ('fd_domain','fd_mode','val_ssim','val_psnr','val_rmse','headroom','val_n') if k in d})" 2>/dev/null || true
echo "[fast-recon] done -> $OUT"
