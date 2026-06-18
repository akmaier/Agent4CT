#!/bin/bash
#SBATCH --job-name=restage-canonical-v3
#SBATCH --partition=main
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --exclude=lme49,lme50,lme51,lme52,lme53,lme170,lme171,lme221
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=/cluster/maier/Agent4CT/results/slurm/%x-%j.out
#SBATCH --error=/cluster/maier/Agent4CT/results/slurm/%x-%j.err

# RECOVERY (corrected): rebuild data/mayo_ldct/staged_canonical with the DEDICATED
# canonical stager — data/stage_mayo_canonical.py — which is what built the
# original (2026-06-14). It writes:
#   {split}_truth.h5     dataset "truth" (folded image-flip) + per-slice "ps"
#                        (ps_eff = 0.700857*native_ps/0.703125), patient-ordered
#   {split}_sino_{lowdose,fulldose}.h5  CANONICAL frame: roll(flip_u(slab_mean))
# from the surviving raw/ (truth) + staged_helix2fan_v3/ (per-patient v3 sinos).
# NOT fetch_mayo_ldct.py (wrong: "image" key, no ps, shuffled) + stage_mayo_sinos
# (wrong: legacy non-canonical packing) — my first attempt used those by mistake.
# --validate FBPs the val split per-sample and prints LD-FBP SSIM (expect ~0.81).

set -euo pipefail
REPO=/cluster/maier/Agent4CT
cd "$REPO"
mkdir -p results/slurm
VENV="$REPO/.venv"
export CUDA_HOME="$VENV/cuda-pip-bundle"
export PATH="$CUDA_HOME/bin:$PATH"
source "$VENV/bin/activate"
export PYTHONPATH="$REPO:${PYTHONPATH:-}"
export HDF5_USE_FILE_LOCKING=FALSE
export AGENT4CT_DATASET=mayo_ldct_2d        # validate() needs det-offset + truncation

echo "[restage] host=$(hostname) job=$SLURM_JOB_ID  $(date)"

# Clean slate: drop the INCORRECT first re-stage (wrong truth key / no ps / wrong sinos).
rm -rf data/mayo_ldct/staged_canonical

python data/stage_mayo_canonical.py --force --validate --subdir staged_helix2fan_v3

echo "[restage] done $(date). Contents:"
ls -la data/mayo_ldct/staged_canonical/
echo "[restage] truth h5 keys (expect truth + ps):"
python -c "import h5py; f=h5py.File('data/mayo_ldct/staged_canonical/val_truth.h5'); print(list(f.keys())); print('ps[:3]=', f['ps'][:3] if 'ps' in f else 'MISSING')"
