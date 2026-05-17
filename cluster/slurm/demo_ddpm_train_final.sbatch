#!/bin/bash
#SBATCH --job-name=demo-dl-ddpm-final
#SBATCH --partition=main
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:q8000:1
#SBATCH --exclude=lme171,lme222,lme223,lme49,lme50,lme51,lme52
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --output=/cluster/maier/Agent4CT/results/slurm/%x-%j.out
#SBATCH --error=/cluster/maier/Agent4CT/results/slurm/%x-%j.err

# Final DDPM training driver. Reads the best config from a JSON file
# named in $BEST_CFG_JSON (set when launching). Writes the checkpoint
# to the path in cfg.ddpm_ckpt.
#
# Usage:
#   sbatch --export=ALL,BEST_CFG_JSON=/path/to/best.json,OUT=/path/to/out \\
#          cluster/slurm/demo_ddpm_train_final.sbatch

set -euo pipefail
REPO=/cluster/maier/Agent4CT
cd "$REPO"
mkdir -p results/slurm
VENV="$REPO/.venv"
export CUDA_HOME="$VENV/cuda-pip-bundle"
export PATH="$CUDA_HOME/bin:$PATH"
source "$VENV/bin/activate"
export PYTHONPATH="$REPO:${PYTHONPATH:-}"

: "${BEST_CFG_JSON:?need BEST_CFG_JSON}"
: "${OUT:?need OUT}"
mkdir -p "$OUT"

echo "[ddpm-train-final] host=$(hostname) job=$SLURM_JOB_ID  cfg=$BEST_CFG_JSON  out=$OUT"
nvidia-smi -L

export DDPM_CONFIG_PATH="$BEST_CFG_JSON"
python pentathlon/demo_dl_reference/solver_ddpm.py "$OUT"
cat "$OUT/result.json"
echo "[ddpm-train-final] done."
