#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=12:00:00

set -eo pipefail

if [ -z "$1" ]; then
    echo "usage: sbatch [-J name] -o LOG -e ERR run_gpt5_slurm.sh <yaml_path>" >&2
    exit 2
fi
CFG="$1"

# Activate the Python environment that has `shinka` installed before running.
# Example (edit for your cluster):
#   source ~/.bashrc
#   conda activate <your-env>

cd "$(dirname "$0")"

echo "=== GPT-5 run ==="
echo "host=$(hostname)  job=${SLURM_JOB_ID:-?}  start=$(date -Is)"
echo "cwd=$(pwd)"
echo "config=${CFG}"
echo "switches:"
grep -E "^(backward_search|complex_patch_actions|  enabled|  results_dir|  llm_models)" "$CFG" || true
echo "================="

exec python run_evo.py --config_path "$CFG"
