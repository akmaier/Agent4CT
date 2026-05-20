#!/bin/bash
# Post-geometry-fix TPE batch.
#
# Dispatches every learned/iterative solver via the central search agent at
# --sampler tpe --iterations 20 against BOTH:
#   1. synthetic phantoms (--calibrated; slug prefix demo-intensity-calibrated-*)
#   2. real Sidky breast_ct data (--dataset breast_ct --calibrated; slug prefix
#      breast-ct-calibrated-*)
#
# The diffusion-recon / DDPM / NAF solvers re-use their cached checkpoints
# under /cluster/maier/Agent4CT/checkpoints — no retraining.
#
# Each (solver, dataset) pair is a single Slurm job. learned_solver_search_agent
# rebuilds runs-index.json at the end of each job's 20-iter sweep, so the
# dashboard updates incrementally without any manual intervention.

set -euo pipefail
REPO=/cluster/maier/Agent4CT
mkdir -p $REPO/results/slurm

# (solver, walltime, mem, gpu?)
JOBS=(
  "itnet_v3:04:00:00:24G:1"
  "uswin:04:00:00:24G:1"
  "hammernik_vn:04:00:00:24G:1"
  "hammernik:03:00:00:16G:1"
  "itnet_v2:03:00:00:16G:1"
  "dual_domain:03:00:00:16G:1"
  "dual_domain_bilateral:03:00:00:16G:1"
  "tv_iterative:01:00:00:8G:0"
  "wu_2015:01:00:00:8G:0"
  "diffusion_recon_dcstep_unconstrained:06:00:00:24G:1"
  "diffusion_recon_dcstep_constrained:06:00:00:24G:1"
  "ram_zeroshot:01:30:00:24G:1"
  "naf:16:00:00:24G:1"
)

DATASETS=("phantoms" "breast_ct")

for ds in "${DATASETS[@]}"; do
  for entry in "${JOBS[@]}"; do
    IFS=':' read -r SOLVER H M S MEM GPU <<<"$entry"
    WALL="${H}:${M}:${S}"
    if [ "$ds" = "phantoms" ]; then
      DS_FLAG=""                       # default; agent prefixes as demo-intensity-calibrated-*
      JOBNAME="cal-tpe-${SOLVER}"
    else
      DS_FLAG="--dataset $ds"          # agent prefixes as ${ds//_/-}-calibrated-tpe-*
      JOBNAME="${ds//_/-}-tpe-${SOLVER}"
    fi
    if [ "$GPU" = "1" ]; then
      GRES_LINE="#SBATCH --gres=gpu:1"
    else
      GRES_LINE="#SBATCH --comment=cpu-only"
    fi
    cat > /tmp/_post_$$.sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=$JOBNAME
#SBATCH --partition=main
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
$GRES_LINE
#SBATCH --exclude=lme49,lme53,lme171
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

echo "[post-geomfix] solver=$SOLVER dataset=$ds host=\$(hostname) job=\$SLURM_JOB_ID"
if [ "$GPU" = "1" ]; then nvidia-smi -L; fi
python scripts/learned_solver_search_agent.py \\
    --solver $SOLVER \\
    --sampler tpe \\
    --iterations 20 \\
    --calibrated \\
    $DS_FLAG \\
    --notes "post-geometry-fix TPE (FOV mask + corrected det_spacing + 2N pad + H[0]/2 filter)"
echo "[post-geomfix] done."
EOF
    sbatch /tmp/_post_$$.sbatch
  done
done
rm -f /tmp/_post_$$.sbatch
