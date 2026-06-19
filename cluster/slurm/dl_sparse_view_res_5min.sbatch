#!/bin/bash
#SBATCH --job-name=ddssl-dlsv-res-iter
#SBATCH --partition=main
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=24G
#SBATCH --time=00:13:00            # iter-40 (12 epochs cosine ~ 9-10 min training) + slack
#SBATCH --exclude=lme222,lme223,lme49,lme50,lme51,lme52,lme171   # exclude slow/bad-driver nodes
#SBATCH --output=/cluster/maier/Agent4CT/results/slurm/%x-%j.out
#SBATCH --error=/cluster/maier/Agent4CT/results/slurm/%x-%j.err
# One DL-Sparse-View RESIDUAL-stack iteration (spawn agent B). Runs the
# agent-edited residual-denoiser solver, writes result.json +
# comparison.png into a per-job out_dir, and prints the headline metrics
# + the agent4ct_record CLI string the operator/agent needs to call next.

set -euo pipefail
REPO=/cluster/maier/Agent4CT
cd "$REPO"
mkdir -p results/slurm

VENV="$REPO/.venv"
export CUDA_HOME="$VENV/cuda-pip-bundle"
export PATH="$CUDA_HOME/bin:$PATH"
source "$VENV/bin/activate"
export PYTHONPATH="$REPO:${PYTHONPATH:-}"

OUT="$REPO/runs/iter-res-${SLURM_JOB_ID}"
mkdir -p "$OUT"

echo "[iter-res] host=$(hostname)  job=$SLURM_JOB_ID  out=$OUT"
nvidia-smi -L || echo "[iter-res] WARN: nvidia-smi failed (driver mismatch on this node?)"
python -c "import torch; print(f'torch={torch.__version__} cuda_ok={torch.cuda.is_available()} dev={torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"NONE\"}')"
if ! python -c "import torch; assert torch.cuda.is_available(), 'CUDA unavailable on this node'"; then
    echo "[iter-res] FATAL: CUDA unavailable; resubmitting on a working node would help."
    exit 3
fi

python pentathlon/dl_sparse_view_res/solver.py "$OUT"

if [ ! -f "$OUT/result.json" ]; then
    echo "[iter-res] FATAL: solver did not produce $OUT/result.json"
    exit 2
fi

echo
echo "==== METRICS (machine-readable) ===="
cat "$OUT/result.json"
echo
echo "[iter-res] done."
