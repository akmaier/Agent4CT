#!/bin/bash
#SBATCH --job-name=ddssl-dlsv-deepseek
#SBATCH --partition=main
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=24G
#SBATCH --time=00:10:00            # 5-min iteration + ~5 min slack
#SBATCH --exclude=lme222,lme223,lme49,lme50,lme51,lme52,lme171
#SBATCH --output=/cluster/maier/Agent4CT/results/slurm/%x-%j.out
#SBATCH --error=/cluster/maier/Agent4CT/results/slurm/%x-%j.err
# One DL-Sparse-View KIMI-driven iteration. Runs the agent-edited
# pentathlon/dl_sparse_view_deepseek/solver.py, writes result.json +
# comparison.png. The Deepseek agent (moonshotai/Deepseek-K2.6 via the RRZE
# LLM gateway) edits solver.py on the workstation and submits this
# sbatch.

set -euo pipefail
REPO=/cluster/maier/Agent4CT
cd "$REPO"
mkdir -p results/slurm

VENV="$REPO/.venv"
export CUDA_HOME="$VENV/cuda-pip-bundle"
export PATH="$CUDA_HOME/bin:$PATH"
source "$VENV/bin/activate"
export PYTHONPATH="$REPO:${PYTHONPATH:-}"

OUT="$REPO/runs/iter-deepseek-${SLURM_JOB_ID}"
mkdir -p "$OUT"

echo "[iter-deepseek] host=$(hostname)  job=$SLURM_JOB_ID  out=$OUT"
nvidia-smi -L || echo "[iter-deepseek] WARN: nvidia-smi failed"
python -c "import torch; print(f'torch={torch.__version__} cuda_ok={torch.cuda.is_available()} dev={torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"NONE\"}')"
if ! python -c "import torch; assert torch.cuda.is_available(), 'CUDA unavailable on this node'"; then
    echo "[iter-deepseek] FATAL: CUDA unavailable; resubmit needed."
    exit 3
fi

python pentathlon/dl_sparse_view_deepseek/solver.py "$OUT"

if [ ! -f "$OUT/result.json" ]; then
    echo "[iter-deepseek] FATAL: solver did not produce $OUT/result.json"
    exit 2
fi

echo
echo "==== METRICS (machine-readable) ===="
cat "$OUT/result.json"
echo
echo "[iter-deepseek] done."
