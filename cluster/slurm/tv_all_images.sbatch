#!/bin/bash
#SBATCH --job-name=tv-all-images
#SBATCH --partition=main
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --mem=12G
#SBATCH --time=00:60:00
#SBATCH --output=/cluster/maier/Agent4CT/results/slurm/%x-%j.out

set -euo pipefail
cd /cluster/maier/Agent4CT
source .venv/bin/activate
export PYTHONPATH="/cluster/maier/Agent4CT${PYTHONPATH:+:$PYTHONPATH}"

# Regenerate comparison images for all 20 iterations of TV search
SEARCH_DIR="/cluster/maier/Agent4CT/runs/tv-all-images-${SLURM_JOB_ID}"
mkdir -p "$SEARCH_DIR"

# Iteration 1
OUT="$SEARCH_DIR/iter-0001"
mkdir -p "$OUT"
echo '{"tv_lambda":0.003007, "tv_iterations":75, "tv_lr":0.005082, "tv_clip_max":0.0556, "tv_decay":0.0323, "tv_optimizer":"gd", "tv_init":"fbp"}' > "$OUT/tv_config.json"
TV_CONFIG_PATH="$OUT/tv_config.json" python pentathlon/demo_dl_reference/solver_tv_search.py "$OUT" 2>&1 | tail -3

# Iteration 2
OUT="$SEARCH_DIR/iter-0002"
mkdir -p "$OUT"
echo '{"tv_lambda":0.001465, "tv_iterations":270, "tv_lr":0.004970, "tv_clip_max":0.0339, "tv_decay":0.0047, "tv_optimizer":"gd", "tv_init":"fbp"}' > "$OUT/tv_config.json"
TV_CONFIG_PATH="$OUT/tv_config.json" python pentathlon/demo_dl_reference/solver_tv_search.py "$OUT" 2>&1 | tail -3

# Iteration 3
OUT="$SEARCH_DIR/iter-0003"
mkdir -p "$OUT"
echo '{"tv_lambda":0.001262, "tv_iterations":174, "tv_lr":0.009254, "tv_clip_max":0.0376, "tv_decay":0.0235, "tv_optimizer":"gd", "tv_init":"fbp"}' > "$OUT/tv_config.json"
TV_CONFIG_PATH="$OUT/tv_config.json" python pentathlon/demo_dl_reference/solver_tv_search.py "$OUT" 2>&1 | tail -3

# Iteration 4
OUT="$SEARCH_DIR/iter-0004"
mkdir -p "$OUT"
echo '{"tv_lambda":0.000113, "tv_iterations":209, "tv_lr":0.002898, "tv_clip_max":0.0574, "tv_decay":0.0318, "tv_optimizer":"gd", "tv_init":"fbp"}' > "$OUT/tv_config.json"
TV_CONFIG_PATH="$OUT/tv_config.json" python pentathlon/demo_dl_reference/solver_tv_search.py "$OUT" 2>&1 | tail -3

# Iteration 5
OUT="$SEARCH_DIR/iter-0005"
mkdir -p "$OUT"
echo '{"tv_lambda":0.001452, "tv_iterations":302, "tv_lr":0.030860, "tv_clip_max":0.0793, "tv_decay":0.0224, "tv_optimizer":"gd", "tv_init":"fbp"}' > "$OUT/tv_config.json"
TV_CONFIG_PATH="$OUT/tv_config.json" python pentathlon/demo_dl_reference/solver_tv_search.py "$OUT" 2>&1 | tail -3

# Iteration 6
OUT="$SEARCH_DIR/iter-0006"
mkdir -p "$OUT"
echo '{"tv_lambda":0.000556, "tv_iterations":119, "tv_lr":0.005786, "tv_clip_max":0.0407, "tv_decay":0.0284, "tv_optimizer":"gd", "tv_init":"fbp"}' > "$OUT/tv_config.json"
TV_CONFIG_PATH="$OUT/tv_config.json" python pentathlon/demo_dl_reference/solver_tv_search.py "$OUT" 2>&1 | tail -3

# Iteration 7
OUT="$SEARCH_DIR/iter-0007"
mkdir -p "$OUT"
echo '{"tv_lambda":0.001645, "tv_iterations":315, "tv_lr":0.002591, "tv_clip_max":0.0634, "tv_decay":0.0287, "tv_optimizer":"gd", "tv_init":"fbp"}' > "$OUT/tv_config.json"
TV_CONFIG_PATH="$OUT/tv_config.json" python pentathlon/demo_dl_reference/solver_tv_search.py "$OUT" 2>&1 | tail -3

# Iteration 8
OUT="$SEARCH_DIR/iter-0008"
mkdir -p "$OUT"
echo '{"tv_lambda":0.000329, "tv_iterations":424, "tv_lr":0.026094, "tv_clip_max":0.0561, "tv_decay":0.0130, "tv_optimizer":"gd", "tv_init":"fbp"}' > "$OUT/tv_config.json"
TV_CONFIG_PATH="$OUT/tv_config.json" python pentathlon/demo_dl_reference/solver_tv_search.py "$OUT" 2>&1 | tail -3

# Iteration 9
OUT="$SEARCH_DIR/iter-0009"
mkdir -p "$OUT"
echo '{"tv_lambda":0.003723, "tv_iterations":408, "tv_lr":0.041828, "tv_clip_max":0.0714, "tv_decay":0.0463, "tv_optimizer":"gd", "tv_init":"fbp"}' > "$OUT/tv_config.json"
TV_CONFIG_PATH="$OUT/tv_config.json" python pentathlon/demo_dl_reference/solver_tv_search.py "$OUT" 2>&1 | tail -3

# Iteration 10
OUT="$SEARCH_DIR/iter-0010"
mkdir -p "$OUT"
echo '{"tv_lambda":0.005687, "tv_iterations":211, "tv_lr":0.002013, "tv_clip_max":0.0364, "tv_decay":0.0119, "tv_optimizer":"gd", "tv_init":"fbp"}' > "$OUT/tv_config.json"
TV_CONFIG_PATH="$OUT/tv_config.json" python pentathlon/demo_dl_reference/solver_tv_search.py "$OUT" 2>&1 | tail -3

echo "Done with first 10. Now doing 11-20..."

# Iteration 11
OUT="$SEARCH_DIR/iter-0011"
mkdir -p "$OUT"
echo '{"tv_lambda":0.000270, "tv_iterations":168, "tv_lr":0.008375, "tv_clip_max":0.0631, "tv_decay":0.0013, "tv_optimizer":"gd", "tv_init":"fbp"}' > "$OUT/tv_config.json"
TV_CONFIG_PATH="$OUT/tv_config.json" python pentathlon/demo_dl_reference/solver_tv_search.py "$OUT" 2>&1 | tail -3

# Iteration 12
OUT="$SEARCH_DIR/iter-0012"
mkdir -p "$OUT"
echo '{"tv_lambda":0.000371, "tv_iterations":372, "tv_lr":0.065501, "tv_clip_max":0.0508, "tv_decay":0.0263, "tv_optimizer":"gd", "tv_init":"fbp"}' > "$OUT/tv_config.json"
TV_CONFIG_PATH="$OUT/tv_config.json" python pentathlon/demo_dl_reference/solver_tv_search.py "$OUT" 2>&1 | tail -3

# Iteration 13
OUT="$SEARCH_DIR/iter-0013"
mkdir -p "$OUT"
echo '{"tv_lambda":0.000503, "tv_iterations":414, "tv_lr":0.014869, "tv_clip_max":0.0670, "tv_decay":0.0259, "tv_optimizer":"gd", "tv_init":"fbp"}' > "$OUT/tv_config.json"
TV_CONFIG_PATH="$OUT/tv_config.json" python pentathlon/demo_dl_reference/solver_tv_search.py "$OUT" 2>&1 | tail -3

# Iteration 14
OUT="$SEARCH_DIR/iter-0014"
mkdir -p "$OUT"
echo '{"tv_lambda":0.009582, "tv_iterations":145, "tv_lr":0.002234, "tv_clip_max":0.0458, "tv_decay":0.0372, "tv_optimizer":"gd", "tv_init":"fbp"}' > "$OUT/tv_config.json"
TV_CONFIG_PATH="$OUT/tv_config.json" python pentathlon/demo_dl_reference/solver_tv_search.py "$OUT" 2>&1 | tail -3

# Iteration 15
OUT="$SEARCH_DIR/iter-0015"
mkdir -p "$OUT"
echo '{"tv_lambda":0.004941, "tv_iterations":338, "tv_lr":0.002368, "tv_clip_max":0.0532, "tv_decay":0.0288, "tv_optimizer":"gd", "tv_init":"fbp"}' > "$OUT/tv_config.json"
TV_CONFIG_PATH="$OUT/tv_config.json" python pentathlon/demo_dl_reference/solver_tv_search.py "$OUT" 2>&1 | tail -3

# Iteration 16
OUT="$SEARCH_DIR/iter-0016"
mkdir -p "$OUT"
echo '{"tv_lambda":0.000715, "tv_iterations":250, "tv_lr":0.052273, "tv_clip_max":0.0654, "tv_decay":0.0016, "tv_optimizer":"gd", "tv_init":"fbp"}' > "$OUT/tv_config.json"
TV_CONFIG_PATH="$OUT/tv_config.json" python pentathlon/demo_dl_reference/solver_tv_search.py "$OUT" 2>&1 | tail -3

# Iteration 17
OUT="$SEARCH_DIR/iter-0017"
mkdir -p "$OUT"
echo '{"tv_lambda":0.008074, "tv_iterations":66, "tv_lr":0.039511, "tv_clip_max":0.0370, "tv_decay":0.0273, "tv_optimizer":"gd", "tv_init":"fbp"}' > "$OUT/tv_config.json"
TV_CONFIG_PATH="$OUT/tv_config.json" python pentathlon/demo_dl_reference/solver_tv_search.py "$OUT" 2>&1 | tail -3

# Iteration 18
OUT="$SEARCH_DIR/iter-0018"
mkdir -p "$OUT"
echo '{"tv_lambda":0.000452, "tv_iterations":455, "tv_lr":0.005030, "tv_clip_max":0.0731, "tv_decay":0.0156, "tv_optimizer":"gd", "tv_init":"fbp"}' > "$OUT/tv_config.json"
TV_CONFIG_PATH="$OUT/tv_config.json" python pentathlon/demo_dl_reference/solver_tv_search.py "$OUT" 2>&1 | tail -3

# Iteration 19
OUT="$SEARCH_DIR/iter-0019"
mkdir -p "$OUT"
echo '{"tv_lambda":0.000729, "tv_iterations":335, "tv_lr":0.045940, "tv_clip_max":0.0553, "tv_decay":0.0356, "tv_optimizer":"gd", "tv_init":"fbp"}' > "$OUT/tv_config.json"
TV_CONFIG_PATH="$OUT/tv_config.json" python pentathlon/demo_dl_reference/solver_tv_search.py "$OUT" 2>&1 | tail -3

# Iteration 20
OUT="$SEARCH_DIR/iter-0020"
mkdir -p "$OUT"
echo '{"tv_lambda":0.002683, "tv_iterations":322, "tv_lr":0.001772, "tv_clip_max":0.0688, "tv_decay":0.0028, "tv_optimizer":"gd", "tv_init":"fbp"}' > "$OUT/tv_config.json"
TV_CONFIG_PATH="$OUT/tv_config.json" python pentathlon/demo_dl_reference/solver_tv_search.py "$OUT" 2>&1 | tail -3

echo ""
echo "=== ALL IMAGES GENERATED ==="
echo "Location: $SEARCH_DIR"
echo ""
echo "Copy back with:"
echo "  scp -r maier@cluster.i5.informatik.uni-erlangen.de:$SEARCH_DIR/* docs/runs/demo-dl-tv-search-20260515-01/iterations/"
