#!/bin/bash
#SBATCH --job-name=ddssl-dlsv-iter
#SBATCH --partition=main
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:q8000:1     # pin 48 GB; RTX 5000 / 16 GB OOMs at val_n=100 + BF
#SBATCH --exclude=lme171,lme222,lme223,lme49,lme50,lme51,lme52  # bad-driver + slow-GPU nodes
#SBATCH --mem=24G
#SBATCH --time=00:10:00            # was 8 min; iter-28 onward sits at 7:32 train, +30s val — too tight
#SBATCH --output=/cluster/maier/Agent4CT/results/slurm/%x-%j.out
#SBATCH --error=/cluster/maier/Agent4CT/results/slurm/%x-%j.err
# One DL-Sparse-View iteration. Runs the (agent-edited) solver, writes
# result.json + comparison.png into a per-job out_dir, and prints the
# headline metrics + the agent4ct_record CLI string the operator/agent
# needs to call next.

set -euo pipefail
REPO=/cluster/maier/Agent4CT
cd "$REPO"
mkdir -p results/slurm

VENV="$REPO/.venv"
export CUDA_HOME="$VENV/cuda-pip-bundle"
export PATH="$CUDA_HOME/bin:$PATH"
source "$VENV/bin/activate"
export PYTHONPATH="$REPO:${PYTHONPATH:-}"

# Use /scratch for transient I/O, /cluster for the final artifacts.
OUT="$REPO/runs/iter-${SLURM_JOB_ID}"
mkdir -p "$OUT"

echo "[iter] host=$(hostname)  job=$SLURM_JOB_ID  out=$OUT"
nvidia-smi -L
python -c "import torch; print(f'torch={torch.__version__} cuda_ok={torch.cuda.is_available()} dev={torch.cuda.get_device_name(0)}')"

python pentathlon/dl_sparse_view/solver.py "$OUT"

if [ ! -f "$OUT/result.json" ]; then
    echo "[iter] FATAL: solver did not produce $OUT/result.json"
    exit 2
fi

echo
echo "==== METRICS (machine-readable) ===="
cat "$OUT/result.json"
echo
echo "==== NEXT (agent runs this after deciding keep/discard) ===="
echo "  python scripts/agent4ct_record.py record \\"
echo "    --slug <RUN_SLUG> --iter <N> \\"
echo "    --val-score \$(jq -r .val_score $OUT/result.json) \\"
echo "    --headroom  \$(jq -r .headroom  $OUT/result.json) \\"
echo "    --params-M  \$(jq -r .params_M  $OUT/result.json) \\"
echo "    --train-n   \$(jq -r .train_n   $OUT/result.json) \\"
echo "    --change-class <architecture|optimizer|loss|augmentation|other> \\"
echo "    --rationale  '<one-line reasoning>' \\"
echo "    --advice     '<one-line generalisable advice for other agents>' \\"
echo "    --kept <true|false> --commit \$(git rev-parse --short HEAD) \\"
echo "    --comparison $OUT/comparison.png \\"
echo "    --solver pentathlon/dl_sparse_view/solver.py"
echo "[iter] done."
