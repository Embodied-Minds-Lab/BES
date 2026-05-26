#!/bin/bash
# SLURM trainer launcher — GRPO / MaxRL / BES on Gemma3-1B-IT.
#
# This script submits ONLY the trainer SLURM job. For --algo bes, the
# backward server is a long-running service and should already be
# running before you call this script:
#
#   sbatch -A <account> -p <partition> scripts/run_backward_slurm.sh
#
# Then submit the trainer:
#
#   bash scripts/run_bes_slurm.sh                   # default: --algo bes
#   bash scripts/run_bes_slurm.sh --algo grpo
#   bash scripts/run_bes_slurm.sh --algo maxrl
#
# The trainer polls backward_server_url.txt (up to 1 h) and waits for
# the server before starting verl.
#
# --dry-run prints the generated SBATCH file path and contents without
# calling sbatch.
set -euo pipefail
REPO_ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$REPO_ROOT"

# =============================================================================
# Defaults — override here or via env vars / CLI flags.
# =============================================================================
ALGO=bes                                                # grpo | maxrl | bes
DRY_RUN=0

SLURM_ACCOUNT=${SLURM_ACCOUNT:-<USER:EDIT>}
SLURM_PARTITION=${SLURM_PARTITION:-<USER:EDIT>}
SLURM_MAIL_USER=${SLURM_MAIL_USER:-}
CONDA_ENV=${CONDA_ENV:-<YOUR_CONDA_ENV>}

DATA_DIR=${DATA_DIR:-$REPO_ROOT/../data}                # BES/logical/data
CKPT_DIR=${CKPT_DIR:-}                                  # default: $REPO_ROOT/checkpoints/<exp>
MODEL_PATH=${MODEL_PATH:-Xkev/gemma-3-1b-it-kk}
BACKWARD_MODEL=${BACKWARD_MODEL:-google/gemma-3-1b-it}
# =============================================================================

while [ $# -gt 0 ]; do
    case "$1" in
        --algo)    ALGO="$2"; shift 2 ;;
        --dry-run) DRY_RUN=1; shift ;;
        -h|--help) sed -n '2,20p' "$0"; exit 0 ;;
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
mkdir -p "$CKPT_DIR" "$REPO_ROOT/logs"

echo "=== ${ALGO^^} on Gemma3-1B-IT (experiment=$EXPERIMENT_NAME) ==="
echo "    adv_estimator: $ADV"
echo "    backward:      $([ $NEED_BACKWARD = 1 ] && echo yes || echo no)"

mail_block() {
    if [ -n "$SLURM_MAIL_USER" ]; then
        printf "#SBATCH --mail-type=BEGIN,END,FAIL\n#SBATCH --mail-user=%s\n" "$SLURM_MAIL_USER"
    fi
}

if [ "$ALGO" = "bes" ]; then
    BES_ARGS_STR='    actor_rollout_ref.rollout.agent.default_agent_loop=bes \
    +actor_rollout_ref.rollout.search.budget=200 \
    +actor_rollout_ref.rollout.search.decompose_interval=10 \
    +actor_rollout_ref.rollout.search.backward_url="$BACKWARD_URL" \
    +actor_rollout_ref.rollout.search.backward_model='"$BACKWARD_MODEL"' \'
else
    BES_ARGS_STR=''
fi

TRAIN_SCRIPT=$(mktemp -t bes_logical_train.XXXXXX.sh)
cat > "$TRAIN_SCRIPT" <<EOF
#!/bin/bash
#SBATCH --job-name=bes-logical-${ALGO}
#SBATCH --account=${SLURM_ACCOUNT}
#SBATCH --partition=${SLURM_PARTITION}
#SBATCH --gres=gpu:2
#SBATCH -c 16
#SBATCH --mem=256G
#SBATCH -t 1-00:00
#SBATCH -o ${REPO_ROOT}/logs/bes_train_${ALGO}_%j.log
#SBATCH -e ${REPO_ROOT}/logs/bes_train_${ALGO}_%j.log
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
    echo "Backward: \$BACKWARD_URL"
fi

ulimit -n 65535
python -W ignore -m verl.trainer.main_ppo \\
    hydra.run.dir=/tmp/hydra_\${SLURM_JOB_ID}_${ALGO} \\
    algorithm.adv_estimator=${ADV} \\
    data.train_files=${DATA_DIR}/train.parquet \\
    data.val_files=${DATA_DIR}/eval.parquet \\
    data.train_batch_size=32 \\
    data.max_prompt_length=1024 \\
    data.max_response_length=4096 \\
    data.filter_overlong_prompts=True \\
    data.truncation='error' \\
    actor_rollout_ref.model.path=${MODEL_PATH} \\
    actor_rollout_ref.actor.optim.lr=1e-6 \\
    actor_rollout_ref.model.use_remove_padding=True \\
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \\
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=16 \\
    actor_rollout_ref.actor.use_kl_loss=False \\
    actor_rollout_ref.actor.kl_loss_coef=0.0 \\
    actor_rollout_ref.actor.clip_ratio_low=0.2 \\
    actor_rollout_ref.actor.clip_ratio_high=0.2 \\
    actor_rollout_ref.actor.grad_clip=1.0 \\
    actor_rollout_ref.model.enable_gradient_checkpointing=True \\
    actor_rollout_ref.actor.fsdp_config.param_offload=False \\
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \\
    actor_rollout_ref.actor.fsdp_config.model_dtype=bf16 \\
    actor_rollout_ref.actor.ppo_epochs=1 \\
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16 \\
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \\
    actor_rollout_ref.rollout.name=vllm \\
    actor_rollout_ref.rollout.max_model_len=6144 \\
    actor_rollout_ref.rollout.max_num_batched_tokens=8192 \\
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \\
    actor_rollout_ref.rollout.n=1 \\
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=16 \\
    actor_rollout_ref.ref.fsdp_config.param_offload=True \\
    actor_rollout_ref.rollout.val_kwargs.n=1 \\
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \\
    actor_rollout_ref.rollout.val_kwargs.temperature=0.6 \\
    actor_rollout_ref.rollout.val_kwargs.top_p=0.95 \\
    actor_rollout_ref.rollout.val_kwargs.top_k=-1 \\
    actor_rollout_ref.rollout.multi_turn.enable=False \\
${BES_ARGS_STR}
    algorithm.use_kl_in_reward=False \\
    algorithm.kl_penalty=low_var_kl \\
    algorithm.kl_ctrl.kl_coef=0.0 \\
    reward_model.reward_manager=naive \\
    trainer.balance_batch=True \\
    trainer.critic_warmup=0 \\
    trainer.val_before_train=True \\
    trainer.val_only=False \\
    trainer.logger=['console','wandb'] \\
    trainer.project_name=BES_Logical \\
    trainer.experiment_name=${EXPERIMENT_NAME} \\
    trainer.default_local_dir=${CKPT_DIR} \\
    trainer.n_gpus_per_node=2 \\
    trainer.nnodes=1 \\
    trainer.save_freq=50 \\
    trainer.test_freq=50 \\
    trainer.total_epochs=16
EOF

if [ "$DRY_RUN" = 1 ]; then
    echo
    echo "=== DRY RUN — generated SBATCH at: $TRAIN_SCRIPT ==="
    cat "$TRAIN_SCRIPT"
    exit 0
fi

if [ "$NEED_BACKWARD" = 1 ] && [ ! -f "$URL_FILE" ]; then
    echo "WARNING: $URL_FILE not found. Did you sbatch scripts/run_backward_slurm.sh first?"
    echo "         The trainer will block until that URL appears (timeout 1 h)."
fi

TJ=$(sbatch --parsable "$TRAIN_SCRIPT")
echo "  trainer  → $TJ"
echo "Log:        ${REPO_ROOT}/logs/bes_train_${ALGO}_${TJ}.log"
