#!/bin/bash
# Standalone SLURM job: long-running E5 retriever server (FAISS over wiki-18).
# Submit this BEFORE the trainer so the URL file is ready when training starts.
#
# Usage:
#   sbatch scripts/run_retriever_slurm.sh
#
# The job publishes its URL to $DATA_ROOT/retriever.endpoint and stays up
# until killed.  One retriever can be shared across many trainer runs.
#
#SBATCH --job-name=bes-retriever
#SBATCH --gres=gpu:1
#SBATCH -c 16
#SBATCH --mem=200G
#SBATCH -t 2-00:00
#SBATCH -o slurm_logs/bes_retr_%j.log
#SBATCH -e slurm_logs/bes_retr_%j.log

# =============================================================================
# USER: set these once for your cluster / data layout.
# (Use `--account=...` / `--partition=...` on the sbatch CLI, or add #SBATCH
# lines above if your scheduler requires them.)
# =============================================================================
DATA_ROOT=${DATA_ROOT:-/path/to/Tree-GRPO-data}         # see README §2
CONDA_ENV=${CONDA_ENV:-<YOUR_CONDA_ENV>}                # env name OR abs path
RETRIEVER_PORT=${RETRIEVER_PORT:-17800}
# =============================================================================

set -euo pipefail
# Under SLURM, $0 points at the cached copy in spool, so use SLURM_SUBMIT_DIR.
REPO_ROOT=${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}
cd "$REPO_ROOT"
mkdir -p slurm_logs

# <USER:EDIT> — uncomment if your cluster uses environment modules.
# module load cuda gcc cmake intel-mkl

if [ -d "$CONDA_ENV" ]; then
    # Absolute path — inject directly so we don't depend on `conda activate`
    # being available inside a non-interactive SLURM batch shell.
    export PATH="$CONDA_ENV/bin:$PATH"
    export LD_LIBRARY_PATH="$CONDA_ENV/lib:${LD_LIBRARY_PATH:-}"
    unset PYTHONPATH
else
    # Treat as an env name; caller must have conda initialised.
    conda activate "$CONDA_ENV"
fi
export PYTHONNOUSERSITE=1

ENDPOINT_FILE=$DATA_ROOT/retriever.endpoint
HOST_IP=$(hostname -i 2>/dev/null | awk '{print $1}' || hostname)
echo "http://${HOST_IP}:${RETRIEVER_PORT}/retrieve" > "$ENDPOINT_FILE"
echo "Retriever endpoint: $(cat "$ENDPOINT_FILE")"
trap "rm -f '$ENDPOINT_FILE'; echo 'Removed endpoint file'" EXIT

ulimit -n 65535
python -s search_r1/search/retrieval_server.py \
    --index_path  "$DATA_ROOT/index/e5_Flat.index" \
    --corpus_path "$DATA_ROOT/corpus/wiki-18.jsonl" \
    --topk 3 --retriever_name e5 \
    --retriever_model "$DATA_ROOT/e5-model" \
    --port "$RETRIEVER_PORT" --faiss_gpu
