#!/bin/bash
#SBATCH --job-name=compare-gt-hd-ld-fbp
#SBATCH --partition=main
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=24G
#SBATCH --time=00:25:00
#SBATCH --exclude=lme49,lme50,lme51,lme52,lme53,lme170,lme171
#SBATCH --output=/cluster/maier/Agent4CT/results/slurm/%x-%j.out
#SBATCH --error=/cluster/maier/Agent4CT/results/slurm/%x-%j.err

# Runs the canonical validator (Powell FBP geometry, MAYO_LDCT_DET_OFFSET,
# Siemens flip, angle_start_corrected, 5-mm slab averaging, fliplr/flipud,
# intensity-calibrate) once per dose against the SSR-fitted staged data,
# then composes a single GT | HD FBP | LD FBP comparison.

set -euo pipefail
REPO=/cluster/maier/Agent4CT
cd "$REPO"
VENV="$REPO/.venv"
export CUDA_HOME="$VENV/cuda-pip-bundle"
export PATH="$CUDA_HOME/bin:$PATH"
. "$VENV/bin/activate"
export PYTHONPATH="$REPO:${PYTHONPATH:-}"
export HDF5_USE_FILE_LOCKING=FALSE
export STAGED_HELIX2FAN_SUBDIR="staged_helix2fan_ssr_fitted"

OUT_DIR="$REPO/results/mayo_debug"
mkdir -p "$OUT_DIR"

echo "[compare] host=$(hostname)  using STAGED_HELIX2FAN_SUBDIR=$STAGED_HELIX2FAN_SUBDIR"
nvidia-smi -L

# HD pass — writes individual validator PNG + an npz of (truth, fbp_cal, fbp_raw)
python scripts/validate_mayo_helix2fan.py \
    --patient L014 --dose fulldose \
    --out-png "$OUT_DIR/L014_validator_fulldose.png" \
    --geometry fitted

# LD pass
python scripts/validate_mayo_helix2fan.py \
    --patient L014 --dose lowdose \
    --out-png "$OUT_DIR/L014_validator_lowdose.png" \
    --geometry fitted

# Compose GT | HD | LD on one figure from the two validator outputs.
python scripts/compose_gt_hd_ld_fbp.py \
    --hd-png "$OUT_DIR/L014_validator_fulldose.png" \
    --ld-png "$OUT_DIR/L014_validator_lowdose.png" \
    --out-png "$OUT_DIR/L014_gt_hd_ld_fbp.png"

echo "[compare] done. Output PNGs in $OUT_DIR"
ls -la "$OUT_DIR"
