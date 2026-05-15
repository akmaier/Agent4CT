#!/bin/bash
#SBATCH --job-name=ddssl-dlsv-loss
#SBATCH --partition=main
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=24G
#SBATCH --time=00:15:00            # 5-min iteration + ~10 min slack (train_n up to 800)
#SBATCH --exclude=lme171,lme221,lme222,lme223,lme49,lme50,lme51,lme52,lme53  # lme221 added after C iter-87 TIMEOUT
#SBATCH --output=/cluster/maier/Agent4CT/results/slurm/%x-%j.out
#SBATCH --error=/cluster/maier/Agent4CT/results/slurm/%x-%j.err
# One DL-Sparse-View iteration on the LOSS-FUNCTION / DATA-STRATEGY branch
# (Spawn-agent C, run dl-sparse-view-loss-20260514-01). Runs the
# pentathlon/dl_sparse_view_loss/solver.py, writes result.json +
# comparison.png into a per-job out_dir, and prints the headline metrics
# + the agent4ct_record CLI string the agent needs to call next.

set -euo pipefail
REPO=/cluster/maier/Agent4CT
cd "$REPO"
mkdir -p results/slurm

VENV="$REPO/.venv"
export CUDA_HOME="$VENV/cuda-pip-bundle"
export PATH="$CUDA_HOME/bin:$PATH"
source "$VENV/bin/activate"
export PYTHONPATH="$REPO:${PYTHONPATH:-}"

OUT="$REPO/runs/iter-loss-${SLURM_JOB_ID}"
mkdir -p "$OUT"

echo "[iter-loss] host=$(hostname)  job=$SLURM_JOB_ID  out=$OUT"
nvidia-smi -L
python -c "import torch; print(f'torch={torch.__version__} cuda_ok={torch.cuda.is_available()} dev={torch.cuda.get_device_name(0)}')"

python pentathlon/dl_sparse_view_loss/solver.py "$OUT"

if [ ! -f "$OUT/result.json" ]; then
    echo "[iter-loss] FATAL: solver did not produce $OUT/result.json"
    exit 2
fi

echo
echo "==== METRICS (machine-readable) ===="
cat "$OUT/result.json"
echo
echo "==== NEXT (agent runs this after deciding keep/discard) ===="
echo "  python scripts/agent4ct_record.py record \\"
echo "    --slug dl-sparse-view-loss-20260514-01 --iter <N> \\"
echo "    --val-score \$(jq -r .val_score $OUT/result.json) \\"
echo "    --headroom  \$(jq -r .headroom  $OUT/result.json) \\"
echo "    --params-M  \$(jq -r .params_M  $OUT/result.json) \\"
echo "    --train-n   \$(jq -r .train_n   $OUT/result.json) \\"
echo "    --change-class <architecture|optimizer|loss|augmentation|other> \\"
echo "    --rationale  '<one-line reasoning>' \\"
echo "    --advice     '<one-line generalisable advice for other agents>' \\"
echo "    --kept <true|false> --commit \$(git rev-parse --short HEAD) \\"
echo "    --comparison $OUT/comparison.png \\"
echo "    --solver pentathlon/dl_sparse_view_loss/solver.py"
echo "[iter-loss] done."
