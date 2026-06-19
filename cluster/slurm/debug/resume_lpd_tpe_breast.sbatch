#!/bin/bash
# Complete the breast-CT LPD TPE study (slug ...-20260524-01) to 20 trials.
# It timed out at 17/20 (job 762081, 14h wall). Resumes the existing Optuna
# study + writes iter-0018..0020 into the existing run dir. partition=main, so
# it does NOT consume a Mayo 4-slot QOS slot (the live campaign is unaffected).
#SBATCH --job-name=bct-lpd-resume
#SBATCH --partition=main
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --exclude=lme49,lme52,lme53
#SBATCH --mem=24G
#SBATCH --time=06:00:00
#SBATCH --output=/cluster/maier/Agent4CT/results/slurm/%x-%j.out
#SBATCH --error=/cluster/maier/Agent4CT/results/slurm/%x-%j.err

set -euo pipefail
cd /cluster/maier/Agent4CT
VENV=/cluster/maier/Agent4CT/.venv
export CUDA_HOME=$VENV/cuda-pip-bundle
export PATH=$CUDA_HOME/bin:$PATH
source $VENV/bin/activate
export PYTHONPATH=/cluster/maier/Agent4CT:${PYTHONPATH:-}
export AGENT4CT_DATASET=breast_ct
export SEARCH_AGENT_SUBPROC_TIMEOUT_S=5400

echo "[bct-lpd-resume] host=$(hostname) job=${SLURM_JOB_ID:-?}"
nvidia-smi -L
python scripts/debug/resume_lpd_tpe_breast.py
echo "[bct-lpd-resume] done."
