#!/bin/bash
# SLURM trainer launcher — GRPO / Tree-GRPO / BES × 3B / 8B.
#
# This script submits ONLY the trainer SLURM job. The retriever and (for
# --algo bes) backward server are long-running services and should already
# be running before you call this script:
#
#   sbatch scripts/run_retriever_slurm.sh         # always
#   sbatch scripts/run_backward_slurm.sh          # only for --algo bes
#
# Then submit the trainer:
#
#   bash scripts/run_bes_slurm.sh                       # default: --algo bes --model 8b
#   bash scripts/run_bes_slurm.sh --algo grpo --model 3b
#   bash scripts/run_bes_slurm.sh --algo tree-grpo --model 8b
#
# The trainer polls retriever.endpoint and musique_backward_server_url.txt
# (up to 1 h each) and waits for both servers before starting verl.
#
# Optional:  --dry-run  prints the generated SBATCH file path and the
#            embedded python command without calling sbatch (use this to
#            sanity-check your config before burning queue time).
set -euo pipefail
REPO_ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$REPO_ROOT"

# =============================================================================
# Defaults — override at the top or via env vars / CLI flags.
# =============================================================================
ALGO=bes                                                # grpo | tree-grpo | bes
MODEL=8b                                                # 3b | 8b
DRY_RUN=0

SLURM_ACCOUNT=${SLURM_ACCOUNT:-<USER:EDIT>}
SLURM_PARTITION=${SLURM_PARTITION:-<USER:EDIT>}
SLURM_MAIL_USER=${SLURM_MAIL_USER:-}
CONDA_ENV=${CONDA_ENV:-<YOUR_CONDA_ENV>}                # name OR absolute path

DATA_ROOT=${DATA_ROOT:-/path/to/Tree-GRPO-data}         # see README §2
CKPT_DIR=${CKPT_DIR:-}                                  # default: $REPO_ROOT/checkpoints/<exp>
BASE_MODEL=${BASE_MODEL:-}                              # default: HF hub name for --model
BACKWARD_MODEL=${BACKWARD_MODEL:-meta-llama/Llama-3.1-8B-Instruct}
# =============================================================================

while [ $# -gt 0 ]; do
    case "$1" in
        --algo)    ALGO="$2";  shift 2 ;;
        --model)   MODEL="$2"; shift 2 ;;
        --dry-run) DRY_RUN=1;  shift ;;
        -h|--help) sed -n '2,22p' "$0"; exit 0 ;;
        *) echo "Unknown arg: $1"; exit 2 ;;
    esac
done

case "$MODEL" in
    3b) MODEL_TAG=llama3.2-3b
        BASE_MODEL=${BASE_MODEL:-meta-llama/Llama-3.2-3B-Instruct} ;;
    8b) MODEL_TAG=llama3.1-8b
        BASE_MODEL=${BASE_MODEL:-meta-llama/Llama-3.1-8B-Instruct} ;;
    *)  echo "Bad --model: '$MODEL' (use 3b or 8b)"; exit 2 ;;
esac
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
ENDPOINT_FILE=$DATA_ROOT/retriever.endpoint
URL_FILE=$REPO_ROOT/musique_backward_server_url.txt
mkdir -p "$CKPT_DIR" "$REPO_ROOT/slurm_logs" "$REPO_ROOT/verl_log"

echo "=== ${ALGO^^} on ${MODEL_TAG} (experiment=$EXPERIMENT_NAME) ==="
echo "    entrypoint:    $ENTRYPOINT"
echo "    adv_estimator: $ADV"
echo "    n_agent:       $N_AGENT"
echo "    backward:      $([ $NEED_BACKWARD = 1 ] && echo yes || echo no)"

mail_block() {
    if [ -n "$SLURM_MAIL_USER" ]; then
        printf "#SBATCH --mail-type=BEGIN,END,FAIL\n#SBATCH --mail-user=%s\n" "$SLURM_MAIL_USER"
    fi
}

if [ "$ALGO" = "bes" ]; then
    BES_ARGS_STR='    ++actor_rollout_ref.rollout.use_bes=true \
    ++actor_rollout_ref.rollout.bes.k_parallel=4 \
    ++actor_rollout_ref.rollout.bes.budget=50 \
    ++actor_rollout_ref.rollout.bes.grpo_n=8 \
    ++actor_rollout_ref.rollout.bes.sim_threshold=0.6 \
    ++actor_rollout_ref.rollout.bes.backward_url="$BACKWARD_URL" \
    ++actor_rollout_ref.rollout.bes.backward_model='"$BACKWARD_MODEL"' \
    ++actor_rollout_ref.rollout.bes.embedder=sentence-transformers/all-MiniLM-L6-v2 \
    ++actor_rollout_ref.rollout.bes.embedder_device=cpu \
    ++actor_rollout_ref.rollout.bes.no_span_abstract=false \
    ++actor_rollout_ref.rollout.bes.think_prefix=true \'
else
    BES_ARGS_STR=''
fi

TRAIN_SCRIPT=$(mktemp -t bes_train.XXXXXX.sh)
cat > "$TRAIN_SCRIPT" <<EOF
#!/bin/bash
#SBATCH --job-name=bes-train-${ALGO}-${MODEL_TAG}
#SBATCH --account=${SLURM_ACCOUNT}
#SBATCH --partition=${SLURM_PARTITION}
#SBATCH --gres=gpu:2
#SBATCH -c 32
#SBATCH --mem=512G
#SBATCH -t 2-00:00
#SBATCH -o ${REPO_ROOT}/slurm_logs/bes_train_${ALGO}_${MODEL_TAG}_%j.log
#SBATCH -e ${REPO_ROOT}/slurm_logs/bes_train_${ALGO}_${MODEL_TAG}_%j.log
$(mail_block)
set -euo pipefail
CONDA_ENV=${CONDA_ENV}

# <USER:EDIT> — uncomment if your cluster uses environment modules.
# module load cuda gcc cmake
if [ -d "\$CONDA_ENV" ]; then
    export PATH="\$CONDA_ENV/bin:\$PATH"
    export LD_LIBRARY_PATH="\$CONDA_ENV/lib:\${LD_LIBRARY_PATH:-}"
    unset PYTHONPATH
else
    conda activate "\$CONDA_ENV"
fi
export PYTHONNOUSERSITE=1
cd ${REPO_ROOT}

NEED_BACKWARD=${NEED_BACKWARD}
echo "Waiting for retriever endpoint at ${ENDPOINT_FILE}..."
for i in \$(seq 1 720); do
    RETRIEVER_URL=\$(cat ${ENDPOINT_FILE} 2>/dev/null || true)
    if [ -n "\$RETRIEVER_URL" ]; then
        curl -sf -m 3 -X POST "\$RETRIEVER_URL" \\
            -H 'Content-Type: application/json' \\
            -d '{"queries":["ping"],"topk":1}' >/dev/null 2>&1 && break
    fi
    sleep 5
done
echo "Retriever: \$RETRIEVER_URL"

BACKWARD_URL=""
if [ "\$NEED_BACKWARD" = 1 ]; then
    echo "Waiting for backward server URL at ${URL_FILE}..."
    for i in \$(seq 1 720); do
        BACKWARD_URL=\$(cat ${URL_FILE} 2>/dev/null || true)
        if [ -n "\$BACKWARD_URL" ]; then
            curl -sf "\$BACKWARD_URL/models" 2>/dev/null | grep -q model && break
        fi
        sleep 5
    done
    echo "Backward:  \$BACKWARD_URL"
fi

n_gpus_per_node=2
train_batch_size=\$(( n_gpus_per_node * 64 ))
val_batch_size=\$(( n_gpus_per_node * 16 ))
actor_ppo_mini_batch_size=\$(( n_gpus_per_node * 8 ))
actor_ppo_micro_batch_size=\$(( n_gpus_per_node * 4 ))
log_prob_micro_batch_size=\$(( n_gpus_per_node * 4 ))
ulimit -n 65535

python -s -m ${ENTRYPOINT} \\
    data.train_files=${DATA_ROOT}/datasets/multihop_musique/hard_train.parquet \\
    data.val_files=${DATA_ROOT}/datasets/multihop_musique/hard_test.parquet \\
    data.train_batch_size=\$train_batch_size \\
    data.val_batch_size=\$val_batch_size \\
    data.max_prompt_length=4096 \\
    data.max_response_length=2048 \\
    data.max_start_length=2048 \\
    data.max_obs_length=500 \\
    data.shuffle_train_dataloader=True \\
    algorithm.adv_estimator=${ADV} \\
    actor_rollout_ref.model.path=${BASE_MODEL} \\
    actor_rollout_ref.model.enable_gradient_checkpointing=true \\
    actor_rollout_ref.model.use_remove_padding=true \\
    actor_rollout_ref.actor.policy_loss=grpo \\
    actor_rollout_ref.actor.optim.lr=1e-6 \\
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.285 \\
    actor_rollout_ref.actor.use_kl_loss=true \\
    actor_rollout_ref.actor.kl_loss_coef=0.001 \\
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \\
    actor_rollout_ref.actor.ppo_mini_batch_size=\$actor_ppo_mini_batch_size \\
    actor_rollout_ref.actor.ppo_micro_batch_size=\$actor_ppo_micro_batch_size \\
    actor_rollout_ref.actor.state_masking=true \\
    actor_rollout_ref.actor.fsdp_config.param_offload=false \\
    actor_rollout_ref.actor.fsdp_config.grad_offload=false \\
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=false \\
    actor_rollout_ref.rollout.log_prob_micro_batch_size=\$log_prob_micro_batch_size \\
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \\
    actor_rollout_ref.rollout.name=vllm \\
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \\
    actor_rollout_ref.rollout.temperature=1 \\
    actor_rollout_ref.rollout.enforce_eager=False \\
    actor_rollout_ref.rollout.free_cache_engine=False \\
    actor_rollout_ref.rollout.n_agent=${N_AGENT} \\
    ++actor_rollout_ref.rollout.disable_log_stats=true \\
    ++actor_rollout_ref.rollout.enable_chunked_prefill=true \\
    ++actor_rollout_ref.rollout.enable_prefix_caching=true \\
    ++actor_rollout_ref.rollout.max_num_batched_tokens=32768 \\
    '++actor_rollout_ref.rollout.stop=["</search>","</answer>"]' \\
    ++actor_rollout_ref.rollout.include_stop_str_in_output=true \\
${BES_ARGS_STR}
    actor_rollout_ref.ref.log_prob_micro_batch_size=\$log_prob_micro_batch_size \\
    actor_rollout_ref.ref.fsdp_config.param_offload=false \\
    algorithm.no_think_rl=false \\
    algorithm.use_kl_in_reward=false \\
    reward_model.structure_format_score=0.2 \\
    reward_model.final_format_score=0.1 \\
    reward_model.retrieval_score=0 \\
    do_search=true \\
    max_turns=3 \\
    retriever.url="\$RETRIEVER_URL" \\
    retriever.topk=3 \\
    trainer.logger="['console','wandb']" \\
    ++trainer.val_only=false \\
    ++trainer.val_before_train=false \\
    trainer.default_hdfs_dir=null \\
    trainer.n_gpus_per_node=\$n_gpus_per_node \\
    trainer.nnodes=1 \\
    trainer.save_freq=20 \\
    ++trainer.remove_previous_ckpt=true \\
    trainer.test_freq=60 \\
    trainer.project_name=Tree-GRPO \\
    trainer.experiment_name=${EXPERIMENT_NAME} \\
    trainer.total_epochs=5 \\
    trainer.total_training_steps=720 \\
    trainer.default_local_dir=${CKPT_DIR} \\
    hydra.run.dir=/tmp/hydra_\${SLURM_JOB_ID}_${ALGO}_${MODEL_TAG} \\
    2>&1 | tee ${REPO_ROOT}/verl_log/${EXPERIMENT_NAME}_\${SLURM_JOB_ID}.log
EOF

if [ "$DRY_RUN" = 1 ]; then
    echo
    echo "=== DRY RUN — generated SBATCH script at: $TRAIN_SCRIPT ==="
    cat "$TRAIN_SCRIPT"
    exit 0
fi

if [ "$NEED_BACKWARD" = 1 ]; then
    echo
    echo "Note: --algo bes also needs scripts/run_backward_slurm.sh to be running."
    if [ ! -f "$URL_FILE" ]; then
        echo "WARNING: $URL_FILE not found. Did you sbatch scripts/run_backward_slurm.sh first?"
        echo "         The trainer will block until that URL appears (timeout 1 h)."
    fi
fi
if [ ! -f "$ENDPOINT_FILE" ]; then
    echo "WARNING: $ENDPOINT_FILE not found. Did you sbatch scripts/run_retriever_slurm.sh first?"
    echo "         The trainer will block until that endpoint appears (timeout 1 h)."
fi

TJ=$(sbatch --parsable "$TRAIN_SCRIPT")
echo "  trainer  → $TJ"
echo "Log:        ${REPO_ROOT}/slurm_logs/bes_train_${ALGO}_${MODEL_TAG}_${TJ}.log"
