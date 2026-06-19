#!/bin/bash
#SBATCH --job-name=ddssl-smoke
#SBATCH --partition=main
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --output=/cluster/maier/Agent4CT/results/slurm/%x-%j.out
#SBATCH --error=/cluster/maier/Agent4CT/results/slurm/%x-%j.err
# Smoke test for the Wagner dual-domain self-supervised LDCT reproduction
# on PYRO-NN. Small dataset, full Wagner geometry (512² / 1152 views / 736 dets).
# Runs the three-way comparison (low-dose FBP / image-only / dual-domain)
# and dumps comparison.png to runs/exp_pyronn/.
#
# Submit from /cluster/maier/Agent4CT:
#     sbatch cluster/slurm/ddssl_smoke.sbatch
#
# Verified bring-up environment: see cluster/setup.sh

set -euo pipefail

REPO=/cluster/maier/Agent4CT
cd "$REPO"

mkdir -p results/slurm runs/exp_pyronn

# Re-export pip-shipped CUDA toolkit dirs (see cluster/setup.sh § 4).
VENV="$REPO/.venv"
export CUDA_HOME="$VENV/cuda-pip-bundle"
export PATH="$CUDA_HOME/bin:$PATH"
# PyTorch ships its own CUDA libs; nothing needed in LD_LIBRARY_PATH.

source "$VENV/bin/activate"
export PYTHONPATH="$REPO:${PYTHONPATH:-}"
echo "[smoke] host=$(hostname) date=$(date)"
nvidia-smi -L
python -c "import torch; print(f'torch={torch.__version__} cuda_ok={torch.cuda.is_available()} dev={torch.cuda.get_device_name(0)}')"

# Quick sanity check first — abort the smoke run if FBP doesn't reconstruct
# a clean phantom correctly. Saves a comparison PNG either way.
python scripts/sanity_pyronn.py --out runs/exp_pyronn/sanity

# Full smoke training run at the Wagner geometry, small dataset.
python scripts/run_experiment.py \
    --size 512 \
    --angles 1152 \
    --dets 736 \
    --train_n 8 \
    --val_n 2 \
    --epochs 5 \
    --unet_c 16 \
    --lr 1e-3 \
    --out_dir runs/exp_pyronn

echo "[smoke] done. results in runs/exp_pyronn/"
