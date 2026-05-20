#!/bin/bash
# Track B.4 (continued): submit dataset-agnostic DL solvers on the
# breast_ct staged data. None of these need the breast DDPM (only
# diffusion-recon does), so they can run while the DDPM ckpt-save bug
# is fixed.

set -euo pipefail
REPO=/cluster/maier/Agent4CT

# (solver, walltime, mem) — all GPU
JOBS=(
  "itnet_v3:04:00:00:24G"
  "uswin:04:00:00:24G"
  "hammernik_vn:04:00:00:24G"
  "hammernik:03:00:00:16G"
  "itnet_v2:03:00:00:16G"
  "dual_domain:03:00:00:16G"
  "dual_domain_bilateral:03:00:00:16G"
  "ram_zeroshot:01:30:00:24G"
)

for entry in "${JOBS[@]}"; do
  IFS=':' read -r SOLVER H M S MEM <<<"$entry"
  WALL="${H}:${M}:${S}"
  JOBNAME="bct-tpe-${SOLVER}"
  cat > /tmp/_bct_$$.sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=$JOBNAME
#SBATCH --partition=main
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --exclude=lme49,lme53
#SBATCH --mem=$MEM
#SBATCH --time=$WALL
#SBATCH --output=$REPO/results/slurm/%x-%j.out
#SBATCH --error=$REPO/results/slurm/%x-%j.err

set -euo pipefail
cd $REPO
VENV=$REPO/.venv
export CUDA_HOME=\$VENV/cuda-pip-bundle
export PATH=\$CUDA_HOME/bin:\$PATH
source \$VENV/bin/activate
export PYTHONPATH=$REPO:\${PYTHONPATH:-}
export AGENT4CT_DATASET=breast_ct

echo "[bct-tpe] solver=$SOLVER host=\$(hostname) job=\$SLURM_JOB_ID"
nvidia-smi -L
python scripts/learned_solver_search_agent.py \\
    --solver $SOLVER \\
    --sampler tpe \\
    --iterations 20 \\
    --calibrated \\
    --dataset breast_ct \\
    --notes "breast-ct calibrated TPE on Sidky 2021 real sinograms"
echo "[bct-tpe] done."
EOF
  sbatch /tmp/_bct_$$.sbatch
done
rm -f /tmp/_bct_$$.sbatch
