#!/bin/bash
#SBATCH --job-name=ddssl-train
#SBATCH --partition=main
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=/cluster/maier/Agent4CT/results/slurm/%x-%j.out
#SBATCH --error=/cluster/maier/Agent4CT/results/slurm/%x-%j.err
# Full Wagner dual-domain reproduction. Author this script *now*, submit
# only after the smoke run has been verified to converge.
#
#     sbatch cluster/slurm/ddssl_train.sbatch
#
# 64 phantoms × 50 epochs is the smallest training regime that matches the
# qualitative behaviour of the paper on synthetic ellipse / Shepp-Logan data.
# When real Mayo LDCT-and-Projection-data has been rebinned by helix2fan and
# wired into build_dataset(), bump --train_n / --val_n to the actual case
# split (Wagner et al. used 10 of the AAPM patients for train and 2 for val).

set -euo pipefail

REPO=/cluster/maier/Agent4CT
cd "$REPO"

mkdir -p results/slurm runs/exp_pyronn_train

VENV="$REPO/.venv"
export CUDA_HOME="$VENV/cuda-pip-bundle"
export PATH="$CUDA_HOME/bin:$PATH"

source "$VENV/bin/activate"
echo "[train] host=$(hostname) date=$(date)"
nvidia-smi | head -8
python -c "import torch; print(f'torch={torch.__version__} cuda_ok={torch.cuda.is_available()} dev={torch.cuda.get_device_name(0)}')"

python scripts/run_experiment.py \
    --size 512 \
    --angles 1152 \
    --dets 736 \
    --train_n 64 \
    --val_n 8 \
    --epochs 50 \
    --unet_c 32 \
    --lr 1e-4 \
    --i0 1e4 \
    --sigma_e 5.0 \
    --out_dir runs/exp_pyronn_train

echo "[train] done. results in runs/exp_pyronn_train/"
