#!/bin/bash
# Standalone SLURM job: vLLM OpenAI-compatible "backward" server.
# Required only when training with --algo bes (BES's bidirectional rollout
# calls this endpoint to DECOMPOSE + VERIFY subgoals). Submit it BEFORE
# the trainer.
#
# Usage:
#   sbatch -A <account> -p <partition> scripts/run_backward_slurm.sh
#
# The job publishes its URL to $REPO_ROOT/backward_server_url.txt and
# stays up until killed. One backward server can be shared across many
# BES trainer runs.
#
#SBATCH --job-name=bes-backward
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=80G
#SBATCH -t 1-00:00
#SBATCH -o logs/bes_bw_%j.log
#SBATCH -e logs/bes_bw_%j.log

# =============================================================================
# USER: set these. (--account / --partition via sbatch CLI, or add #SBATCH
# lines above.)
# =============================================================================
BACKWARD_MODEL=${BACKWARD_MODEL:-google/gemma-3-1b-it}
CONDA_ENV=${CONDA_ENV:-<YOUR_CONDA_ENV>}
BACKWARD_PORT=${BACKWARD_PORT:-8235}
# =============================================================================

set -euo pipefail
# Under SLURM, $0 points at the cached copy in spool, so use SLURM_SUBMIT_DIR.
REPO_ROOT=${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}
cd "$REPO_ROOT"
mkdir -p logs

# <USER:EDIT> — uncomment if your cluster uses environment modules.
# module load cuda gcc cmake

if [ -d "$CONDA_ENV" ]; then
    export PATH="$CONDA_ENV/bin:$PATH"
    export LD_LIBRARY_PATH="$CONDA_ENV/lib:${LD_LIBRARY_PATH:-}"
    unset PYTHONPATH
else
    conda activate "$CONDA_ENV"
fi
export PYTHONNOUSERSITE=1

URL_FILE=$REPO_ROOT/backward_server_url.txt
HOST=$(hostname -f 2>/dev/null || hostname)
BACKWARD_URL="http://${HOST}:${BACKWARD_PORT}/v1"
trap "rm -f '$URL_FILE'; echo 'Removed URL file'" EXIT

# Launch vLLM OpenAI server — dedicated GPU, CUDA graphs, generous KV cache.
VLLM_USE_FLASHINFER_SAMPLER=0 python -m vllm.entrypoints.openai.api_server \
    --model "$BACKWARD_MODEL" \
    --dtype bfloat16 \
    --max-model-len 2048 \
    --max-num-batched-tokens 16384 \
    --max-num-seqs 256 \
    --host 0.0.0.0 \
    --port "$BACKWARD_PORT" \
    --gpu-memory-utilization 0.9 \
    --disable-log-requests \
    --trust-remote-code &
VLLM_PID=$!

# Wait up to ~20 min for vLLM to start serving, then publish URL.
for i in $(seq 1 240); do
    kill -0 $VLLM_PID 2>/dev/null || { echo "vLLM died during startup"; exit 1; }
    curl -sf "http://localhost:${BACKWARD_PORT}/v1/models" 2>/dev/null | grep -q model && break
    sleep 5
done
echo "$BACKWARD_URL" > "$URL_FILE"
echo "Backward URL published: $(cat "$URL_FILE")"
wait $VLLM_PID
