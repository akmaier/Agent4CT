#!/bin/bash
#SBATCH --job-name=breast-r2g-fbp-init-iter2
#SBATCH --partition=main
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --exclude=lme49,lme53,lme50,lme51,lme52,lme170
#SBATCH --mem=24G
#SBATCH --time=01:30:00
#SBATCH --output=/cluster/maier/Agent4CT/results/slurm/%x-%j.out
#SBATCH --error=/cluster/maier/Agent4CT/results/slurm/%x-%j.err

# Autoresearch iter-2 for the R²-Gaussian FBP-init hypothesis (breast-CT).
#
# Diagnosis of iter-1 (SLURM 762639):
#   - hr=0.0000 SSIM=0.8949 — looks like a failure, BUT
#   - log says `[fit] outer wall 600s hit at sample 1`: the
#     gs_outer_wall_s=600 default killed the job after processing only
#     1 of 20 val samples. The other 19 samples were never even
#     started, so val SSIM defaulted to ~baseline FBP.
#   - The FBP-init hypothesis was therefore NOT TESTED in iter-1.
#
# Iter-2 hypothesis: same config, but with a runtime budget that
# actually lets the solver process all 20 val samples.
#   - gs_n_iter 20000 → 5000  (4× faster per sample; sufficient to
#     show iter-1's expected signal differential vs cold init at
#     similar iter counts)
#   - gs_outer_wall_s 600 → 1800 (3× more time as a safety margin)
#   - Everything else identical to iter-1 so the only DIFFERENT axis
#     between iter-1 (cold init, this season's TPE) and iter-2 is the
#     init scheme.
#
# Expected outcome: if FBP warm-start meaningfully changes the loss
# landscape, hr should be measurably > 0 in iter-2. If iter-2 still
# gives hr=0 with a complete val pass, the FBP-init hypothesis is
# falsified and R²-Gaussian on dense-view breast is structurally
# bounded — file the verdict and end the loop.

set -euo pipefail
REPO=/cluster/maier/Agent4CT
cd "$REPO"
VENV="$REPO/.venv"
export CUDA_HOME="$VENV/cuda-pip-bundle"
export PATH="$CUDA_HOME/bin:$PATH"
source "$VENV/bin/activate"
export PYTHONPATH="$REPO:${PYTHONPATH:-}"

SLUG=breast-ct-claude-agentic-r2g-fbp-init-search-20260602-01
ITER=2
SOLVER=r2gaussian

mkdir -p /cluster/maier/Agent4CT/agentic_cfgs
CFG_JSON=/cluster/maier/Agent4CT/agentic_cfgs/breast_r2g_fbp_init_iter_02.json
cat > "$CFG_JSON" <<'JEOF'
{
  "gs_n_gaussians": 1024,
  "gs_n_iter":      5000,
  "gs_lr_pos":      2.5e-3,
  "gs_lr_scale":    1.2e-2,
  "gs_lr_amp":      6.5e-3,
  "gs_lr_rot":      1.0e-2,
  "gs_amp_init":    0.01,
  "gs_scale_init":  0.04,
  "gs_tv_weight":   1e-4,
  "gs_n_clip":      0.05,
  "gs_init_from_fbp": true,
  "gs_amp_init_max":  0.05,
  "gs_outer_wall_s":  1800,
  "val_n":          20
}
JEOF

echo "[iter2] host=$(hostname) job=$SLURM_JOB_ID slug=$SLUG iter=$ITER"
nvidia-smi -L
cat "$CFG_JSON"; echo

SLUG="$SLUG" ITER_N="$ITER" SOLVER="$SOLVER" CFG_JSON="$CFG_JSON" \
  AGENT4CT_DATASET=breast_ct \
  python scripts/claude_agentic_one_iter.py

echo "[iter2] done."
