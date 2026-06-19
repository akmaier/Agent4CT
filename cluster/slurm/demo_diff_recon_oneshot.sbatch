#!/bin/bash
#SBATCH --job-name=demo-diff-recon-oneshot
#SBATCH --partition=main
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --exclude=lme49,lme53
#SBATCH --mem=24G
#SBATCH --time=00:45:00
#SBATCH --output=/cluster/maier/Agent4CT/results/slurm/%x-%j.out
#SBATCH --error=/cluster/maier/Agent4CT/results/slurm/%x-%j.err

# Single-iter diffusion recon, config picked by the Claude agent.
# Required env: CFG_JSON  OUT  (driver script supplies both)

set -euo pipefail
REPO=/cluster/maier/Agent4CT
cd "$REPO"
mkdir -p results/slurm
VENV="$REPO/.venv"
export CUDA_HOME="$VENV/cuda-pip-bundle"
export PATH="$CUDA_HOME/bin:$PATH"
source "$VENV/bin/activate"
export PYTHONPATH="$REPO:${PYTHONPATH:-}"

: "${CFG_JSON:?need CFG_JSON}"
: "${OUT:?need OUT}"
mkdir -p "$OUT"

echo "[oneshot] host=$(hostname) job=$SLURM_JOB_ID  cfg=$CFG_JSON  out=$OUT"
nvidia-smi -L

export DIFFUSION_RECON_CONFIG_PATH="$CFG_JSON"
python pentathlon/demo_dl_reference/solver_diffusion_recon.py "$OUT"
cat "$OUT/result.json"
echo "[oneshot] done."
