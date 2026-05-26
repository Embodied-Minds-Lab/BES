#!/bin/bash
# Local single-node Logical-Reasoning training tutorial — GRPO / MaxRL / BES.
#
# Launches all components on one node as background processes:
#   GPU 0     → backward vLLM server  (only when --algo bes)
#   GPU 1,2   → trainer               (2 GPUs)
#
# Usage:
#   bash scripts/run_bes_local.sh                   # default: --algo bes
#   bash scripts/run_bes_local.sh --algo grpo
#   bash scripts/run_bes_local.sh --algo maxrl
#
set -euo pipefail
REPO_ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$REPO_ROOT"

# =============================================================================
# Defaults — override here or via env vars / CLI flags.
# =============================================================================
ALGO=bes                                                # grpo | maxrl | bes
DRY_RUN=0
DATA_DIR=${DATA_DIR:-$REPO_ROOT/../data}                 # BES/logical/data
CKPT_DIR=${CKPT_DIR:-}                                  # default: $REPO_ROOT/checkpoints/<exp>
MODEL_PATH=${MODEL_PATH:-Xkev/gemma-3-1b-it-kk}         # SFT checkpoint
BACKWARD_MODEL=${BACKWARD_MODEL:-google/gemma-3-1b-it}
BACKWARD_PORT=${BACKWARD_PORT:-8235}
# =============================================================================

while [ $# -gt 0 ]; do
    case "$1" in
        --algo)    ALGO="$2"; shift 2 ;;
        --dry-run) DRY_RUN=1; shift ;;
        -h|--help) sed -n '2,16p' "$0"; exit 0 ;;
        *) echo "Unknown arg: $1"; exit 2 ;;
    esac
done

case "$ALGO" in
    grpo)  ADV=grpo;       NEED_BACKWARD=0 ;;
    maxrl) ADV=grpo_maxrl; NEED_BACKWARD=0 ;;
    bes)   ADV=grpo_maxrl; NEED_BACKWARD=1 ;;
    *) echo "Bad --algo: '$ALGO' (use grpo | maxrl | bes)"; exit 2 ;;
esac

EXPERIMENT_NAME=logical-${ALGO}-gemma3-1b
CKPT_DIR=${CKPT_DIR:-$REPO_ROOT/checkpoints/$EXPERIMENT_NAME}
URL_FILE=$REPO_ROOT/backward_server_url.txt
mkdir -p "$CKPT_DIR" /tmp/bes_local_logs

echo "=== ${ALGO^^} on Gemma3-1B-IT (experiment=$EXPERIMENT_NAME) ==="
echo "    adv_estimator: $ADV"
echo "    backward:      $([ $NEED_BACKWARD = 1 ] && echo yes || echo no)"
echo "    base_model:    $MODEL_PATH"
echo "    train_files:   $DATA_DIR/train.parquet"
echo "    val_files:     $DATA_DIR/eval.parquet"
if [ "$DRY_RUN" = 1 ]; then
    echo "    (DRY RUN — printing config only, not launching anything)"
    exit 0
fi

cleanup() {
    set +e
    [ -n "${BACKWARD_PID:-}" ] && kill "$BACKWARD_PID" 2>/dev/null
    rm -f "$URL_FILE"
}
trap cleanup EXIT

# -----------------------------------------------------------------------------
# Backward server (BES only)
# -----------------------------------------------------------------------------
BACKWARD_URL=""
if [ "$NEED_BACKWARD" = 1 ]; then
    HOST_IP=$(hostname -i 2>/dev/null | awk '{print $1}' || echo 127.0.0.1)
    BACKWARD_URL="http://${HOST_IP}:${BACKWARD_PORT}/v1"
    echo "[1/2] Launching backward vLLM server on GPU 0 (port $BACKWARD_PORT, model=$BACKWARD_MODEL)..."
    VLLM_USE_FLASHINFER_SAMPLER=0 \
    CUDA_VISIBLE_DEVICES=0 \
    python -m vllm.entrypoints.openai.api_server \
        --model "$BACKWARD_MODEL" \
        --dtype bfloat16 \
        --max-model-len 2048 \
        --max-num-batched-tokens 16384 \
        --max-num-seqs 256 \
        --host 0.0.0.0 \
        --port "$BACKWARD_PORT" \
        --gpu-memory-utilization 0.9 \
        --disable-log-requests \
        --trust-remote-code \
        > /tmp/bes_local_logs/backward.log 2>&1 &
    BACKWARD_PID=$!

    echo "Waiting for backward server..."
    for i in $(seq 1 240); do
        kill -0 $BACKWARD_PID 2>/dev/null || { echo "  vLLM died"; exit 1; }
        curl -sf "http://localhost:${BACKWARD_PORT}/v1/models" 2>/dev/null | grep -q model \
            && { echo "  ok"; break; }
        sleep 5
    done
    echo "$BACKWARD_URL" > "$URL_FILE"
fi

# -----------------------------------------------------------------------------
# Trainer
# -----------------------------------------------------------------------------
echo "[2/2] Launching ${ALGO^^} trainer on GPUs 1,2..."

# Algorithm-specific Hydra overrides (only BES needs the search.* block).
BES_ARGS=()
if [ "$ALGO" = "bes" ]; then
    BES_ARGS=(
        actor_rollout_ref.rollout.agent.default_agent_loop=bes
        +actor_rollout_ref.rollout.search.budget=200
        +actor_rollout_ref.rollout.search.decompose_interval=10
        +actor_rollout_ref.rollout.search.backward_url="$BACKWARD_URL"
        +actor_rollout_ref.rollout.search.backward_model="$BACKWARD_MODEL"
    )
fi

ulimit -n 65535
CUDA_VISIBLE_DEVICES=1,2 \
python -W ignore -m verl.trainer.main_ppo \
    hydra.run.dir=/tmp/hydra_logical_${ALGO} \
    algorithm.adv_estimator=$ADV \
    data.train_files=$DATA_DIR/train.parquet \
    data.val_files=$DATA_DIR/eval.parquet \
    data.train_batch_size=32 \
    data.max_prompt_length=1024 \
    data.max_response_length=4096 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.0 \
    actor_rollout_ref.actor.clip_ratio_low=0.2 \
    actor_rollout_ref.actor.clip_ratio_high=0.2 \
    actor_rollout_ref.actor.grad_clip=1.0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.actor.fsdp_config.model_dtype=bf16 \
    actor_rollout_ref.actor.ppo_epochs=1 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.max_model_len=6144 \
    actor_rollout_ref.rollout.max_num_batched_tokens=8192 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.n=1 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.6 \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.95 \
    actor_rollout_ref.rollout.val_kwargs.top_k=-1 \
    actor_rollout_ref.rollout.multi_turn.enable=False \
    "${BES_ARGS[@]}" \
    algorithm.use_kl_in_reward=False \
    algorithm.kl_penalty=low_var_kl \
    algorithm.kl_ctrl.kl_coef=0.0 \
    reward_model.reward_manager=naive \
    trainer.balance_batch=True \
    trainer.critic_warmup=0 \
    trainer.val_before_train=True \
    trainer.val_only=False \
    trainer.logger=['console','wandb'] \
    trainer.project_name=BES_Logical \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.default_local_dir=$CKPT_DIR \
    trainer.n_gpus_per_node=2 \
    trainer.nnodes=1 \
    trainer.save_freq=50 \
    trainer.test_freq=50 \
    trainer.total_epochs=16 \
    2>&1 | tee "$REPO_ROOT/${EXPERIMENT_NAME}.log"
