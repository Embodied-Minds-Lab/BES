#!/bin/bash
# Local single-node training tutorial — GRPO / Tree-GRPO / BES × 3B / 8B.
#
# Launches all components on one node as background processes:
#   GPU 0       → retriever            (always)
#   GPU 1       → backward server      (only when --algo bes)
#   GPU 2,3     → trainer              (2 GPUs)
#
# Usage:
#   bash scripts/run_bes_local.sh                     # default: --algo bes --model 8b
#   bash scripts/run_bes_local.sh --algo grpo --model 3b
#   bash scripts/run_bes_local.sh --algo tree-grpo --model 8b
#
set -euo pipefail
REPO_ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$REPO_ROOT"

# =============================================================================
# Defaults — override here or via env vars / CLI flags.
# =============================================================================
ALGO=bes                                                # grpo | tree-grpo | bes
MODEL=8b                                                # 3b | 8b
DRY_RUN=0
DATA_ROOT=${DATA_ROOT:-$REPO_ROOT/data}                 # see README §2
CKPT_DIR=${CKPT_DIR:-}                                  # default: $REPO_ROOT/checkpoints/<exp>
BASE_MODEL=${BASE_MODEL:-}                              # default: HF hub name for --model
BACKWARD_MODEL=${BACKWARD_MODEL:-meta-llama/Llama-3.1-8B-Instruct}
RETRIEVER_PORT=${RETRIEVER_PORT:-17800}
BACKWARD_PORT=${BACKWARD_PORT:-8236}
# =============================================================================

while [ $# -gt 0 ]; do
    case "$1" in
        --algo)    ALGO="$2";  shift 2 ;;
        --model)   MODEL="$2"; shift 2 ;;
        --dry-run) DRY_RUN=1;  shift ;;
        -h|--help) sed -n '2,18p' "$0"; exit 0 ;;
        *) echo "Unknown arg: $1"; exit 2 ;;
    esac
done

# Validate + derive per-model defaults.
case "$MODEL" in
    3b) MODEL_TAG=llama3.2-3b
        BASE_MODEL=${BASE_MODEL:-meta-llama/Llama-3.2-3B-Instruct} ;;
    8b) MODEL_TAG=llama3.1-8b
        BASE_MODEL=${BASE_MODEL:-meta-llama/Llama-3.1-8B-Instruct} ;;
    *)  echo "Bad --model: '$MODEL' (use 3b or 8b)"; exit 2 ;;
esac

# Validate + derive per-algorithm defaults.
case "$ALGO" in
    grpo)
        ENTRYPOINT=verl.trainer.main_ppo_format
        ADV=grpo; N_AGENT=4; NEED_BACKWARD=0 ;;
    tree-grpo|treegrpo|tree)
        ALGO=tree-grpo
        ENTRYPOINT=verl.trainer.main_ppo_format_ts
        ADV=tree; N_AGENT=1; NEED_BACKWARD=0 ;;
    bes)
        ENTRYPOINT=verl.trainer.main_ppo_format_ts
        ADV=grpo; N_AGENT=8; NEED_BACKWARD=1 ;;
    *) echo "Bad --algo: '$ALGO' (use grpo | tree-grpo | bes)"; exit 2 ;;
esac

EXPERIMENT_NAME=multihopqa-${ALGO}-${MODEL_TAG}-musique
CKPT_DIR=${CKPT_DIR:-$REPO_ROOT/checkpoints/$EXPERIMENT_NAME}

DATA_DIR=$DATA_ROOT/datasets/multihop_musique
ENDPOINT_FILE=$DATA_ROOT/retriever.endpoint
URL_FILE=$REPO_ROOT/musique_backward_server_url.txt
mkdir -p "$CKPT_DIR" "$REPO_ROOT/verl_log" /tmp/bes_local_logs

echo "=== ${ALGO^^} on ${MODEL_TAG} (experiment=$EXPERIMENT_NAME) ==="
echo "    entrypoint:    $ENTRYPOINT"
echo "    adv_estimator: $ADV"
echo "    n_agent:       $N_AGENT"
echo "    backward:      $([ $NEED_BACKWARD = 1 ] && echo yes || echo no)"
if [ "$DRY_RUN" = 1 ]; then
    echo "    (DRY RUN — printing config only, not launching anything)"
    echo "    base_model:    $BASE_MODEL"
    echo "    backward_model:$BACKWARD_MODEL"
    echo "    data_root:     $DATA_ROOT"
    echo "    ckpt_dir:      $CKPT_DIR"
    echo "    retriever_port:$RETRIEVER_PORT  backward_port:$BACKWARD_PORT"
    exit 0
fi

cleanup() {
    set +e
    [ -n "${RETRIEVER_PID:-}" ] && kill "$RETRIEVER_PID" 2>/dev/null
    [ -n "${BACKWARD_PID:-}"  ] && kill "$BACKWARD_PID"  2>/dev/null
    rm -f "$ENDPOINT_FILE" "$URL_FILE"
}
trap cleanup EXIT

# -----------------------------------------------------------------------------
# 1) Retriever (always)
# -----------------------------------------------------------------------------
HOST_IP=$(hostname -i 2>/dev/null | awk '{print $1}' || echo 127.0.0.1)
echo "[1] Launching retriever on GPU 0 (port $RETRIEVER_PORT)..."
echo "http://${HOST_IP}:${RETRIEVER_PORT}/retrieve" > "$ENDPOINT_FILE"
CUDA_VISIBLE_DEVICES=0 \
python search_r1/search/retrieval_server.py \
    --index_path  "$DATA_ROOT/index/e5_Flat.index" \
    --corpus_path "$DATA_ROOT/corpus/wiki-18.jsonl" \
    --topk 3 --retriever_name e5 \
    --retriever_model "$DATA_ROOT/e5-model" \
    --port "$RETRIEVER_PORT" --faiss_gpu \
    > /tmp/bes_local_logs/retriever.log 2>&1 &
RETRIEVER_PID=$!

# -----------------------------------------------------------------------------
# 2) Backward server (BES only)
# -----------------------------------------------------------------------------
if [ "$NEED_BACKWARD" = 1 ]; then
    echo "[2] Launching backward server on GPU 1 (port $BACKWARD_PORT, model=$BACKWARD_MODEL)..."
    echo "http://${HOST_IP}:${BACKWARD_PORT}/v1" > "$URL_FILE"
    CUDA_VISIBLE_DEVICES=1 \
    python -m vllm.entrypoints.openai.api_server \
        --model "$BACKWARD_MODEL" \
        --served-model-name "$BACKWARD_MODEL" \
        --port "$BACKWARD_PORT" \
        --gpu-memory-utilization 0.85 \
        > /tmp/bes_local_logs/backward.log 2>&1 &
    BACKWARD_PID=$!
else
    echo "[2] (skipping backward server — not needed for --algo $ALGO)"
fi

# -----------------------------------------------------------------------------
# 3) Wait for endpoints
# -----------------------------------------------------------------------------
echo "Waiting for retriever..."
for i in $(seq 1 120); do
    curl -sf -m 2 -X POST "$(cat "$ENDPOINT_FILE")" \
        -H 'Content-Type: application/json' \
        -d '{"queries":["ping"],"topk":1}' >/dev/null 2>&1 && { echo "  ok"; break; }
    sleep 5
done

if [ "$NEED_BACKWARD" = 1 ]; then
    echo "Waiting for backward server..."
    for i in $(seq 1 120); do
        curl -sf "http://localhost:${BACKWARD_PORT}/v1/models" 2>/dev/null | grep -q model \
            && { echo "  ok"; break; }
        sleep 5
    done
fi

# -----------------------------------------------------------------------------
# 4) Trainer
# -----------------------------------------------------------------------------
n_gpus_per_node=2
train_batch_size=$(( n_gpus_per_node * 64 ))
val_batch_size=$(( n_gpus_per_node * 16 ))
actor_ppo_mini_batch_size=$(( n_gpus_per_node * 8 ))
actor_ppo_micro_batch_size=$(( n_gpus_per_node * 4 ))
log_prob_micro_batch_size=$(( n_gpus_per_node * 4 ))
ulimit -n 65535

# Algorithm-specific extra args (only BES needs these).
BES_ARGS=()
if [ "$ALGO" = "bes" ]; then
    BACKWARD_URL=$(cat "$URL_FILE")
    BES_ARGS=(
        ++actor_rollout_ref.rollout.use_bes=true
        ++actor_rollout_ref.rollout.bes.k_parallel=4
        ++actor_rollout_ref.rollout.bes.budget=50
        ++actor_rollout_ref.rollout.bes.grpo_n=8
        ++actor_rollout_ref.rollout.bes.sim_threshold=0.6
        ++actor_rollout_ref.rollout.bes.backward_url="$BACKWARD_URL"
        ++actor_rollout_ref.rollout.bes.backward_model="$BACKWARD_MODEL"
        ++actor_rollout_ref.rollout.bes.embedder=sentence-transformers/all-MiniLM-L6-v2
        ++actor_rollout_ref.rollout.bes.embedder_device=cpu
        ++actor_rollout_ref.rollout.bes.no_span_abstract=false
        ++actor_rollout_ref.rollout.bes.think_prefix=true
    )
fi

echo "[3] Launching trainer on GPUs 2,3..."
CUDA_VISIBLE_DEVICES=2,3 \
python -s -m $ENTRYPOINT \
    data.train_files=$DATA_DIR/hard_train.parquet \
    data.val_files=$DATA_DIR/hard_test.parquet \
    data.train_batch_size=$train_batch_size \
    data.val_batch_size=$val_batch_size \
    data.max_prompt_length=4096 \
    data.max_response_length=2048 \
    data.max_start_length=2048 \
    data.max_obs_length=500 \
    data.shuffle_train_dataloader=True \
    algorithm.adv_estimator=$ADV \
    actor_rollout_ref.model.path=$BASE_MODEL \
    actor_rollout_ref.model.enable_gradient_checkpointing=true \
    actor_rollout_ref.model.use_remove_padding=true \
    actor_rollout_ref.actor.policy_loss=grpo \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.285 \
    actor_rollout_ref.actor.use_kl_loss=true \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.ppo_mini_batch_size=$actor_ppo_mini_batch_size \
    actor_rollout_ref.actor.ppo_micro_batch_size=$actor_ppo_micro_batch_size \
    actor_rollout_ref.actor.state_masking=true \
    actor_rollout_ref.actor.fsdp_config.param_offload=false \
    actor_rollout_ref.actor.fsdp_config.grad_offload=false \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=false \
    actor_rollout_ref.rollout.log_prob_micro_batch_size=$log_prob_micro_batch_size \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.temperature=1 \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.n_agent=$N_AGENT \
    ++actor_rollout_ref.rollout.disable_log_stats=true \
    ++actor_rollout_ref.rollout.enable_chunked_prefill=true \
    ++actor_rollout_ref.rollout.enable_prefix_caching=true \
    ++actor_rollout_ref.rollout.max_num_batched_tokens=32768 \
    '++actor_rollout_ref.rollout.stop=["</search>","</answer>"]' \
    ++actor_rollout_ref.rollout.include_stop_str_in_output=true \
    "${BES_ARGS[@]}" \
    actor_rollout_ref.ref.log_prob_micro_batch_size=$log_prob_micro_batch_size \
    actor_rollout_ref.ref.fsdp_config.param_offload=false \
    algorithm.no_think_rl=false \
    algorithm.use_kl_in_reward=false \
    reward_model.structure_format_score=0.2 \
    reward_model.final_format_score=0.1 \
    reward_model.retrieval_score=0 \
    do_search=true \
    max_turns=3 \
    retriever.url="$(cat "$ENDPOINT_FILE")" \
    retriever.topk=3 \
    trainer.logger="['console','wandb']" \
    ++trainer.val_only=false \
    ++trainer.val_before_train=false \
    trainer.default_hdfs_dir=null \
    trainer.n_gpus_per_node=$n_gpus_per_node \
    trainer.nnodes=1 \
    trainer.save_freq=20 \
    ++trainer.remove_previous_ckpt=true \
    trainer.test_freq=60 \
    trainer.project_name=Tree-GRPO \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.total_epochs=5 \
    trainer.total_training_steps=720 \
    trainer.default_local_dir=$CKPT_DIR \
    hydra.run.dir=/tmp/hydra_${ALGO}_${MODEL_TAG} \
    2>&1 | tee "$REPO_ROOT/verl_log/${EXPERIMENT_NAME}.log"
