#!/bin/bash
# Standalone SLURM job: SFT on Knights-and-Knaves with Gemma3-1B-IT.
# Re-creates the SFT base checkpoint published at Xkev/gemma-3-1b-it-kk.
#
# Usage:
#   sbatch -A <account> -p <partition> scripts/run_sft_slurm.sh
#
# Prepare a parquet with verl's SFT message schema (chat-formatted
# `messages` column) and point TRAIN_DATA below at it.
#
#SBATCH --job-name=bes-sft
#SBATCH --gres=gpu:1
#SBATCH -c 16
#SBATCH --mem=256G
#SBATCH -t 0-02:00
#SBATCH -o logs/sft_gemma3_1b_%j.log
#SBATCH -e logs/sft_gemma3_1b_%j.log

# =============================================================================
# USER: set these. (--account / --partition via sbatch CLI, or add #SBATCH
# lines above.)
# =============================================================================
TRAIN_DATA=${TRAIN_DATA:-/path/to/sft.parquet}
MODEL_PATH=${MODEL_PATH:-google/gemma-3-1b-it}
CONDA_ENV=${CONDA_ENV:-<YOUR_CONDA_ENV>}
CKPT_DIR=${CKPT_DIR:-}
# =============================================================================

set -euo pipefail
# Under SLURM, $0 points at the cached copy in spool, so use SLURM_SUBMIT_DIR.
REPO_ROOT=${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}
cd "$REPO_ROOT"
mkdir -p logs

EXPERIMENT_NAME=sft-gemma3-1b-kk
CKPT_DIR=${CKPT_DIR:-$REPO_ROOT/checkpoints/$EXPERIMENT_NAME}
mkdir -p "$CKPT_DIR"

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

torchrun --nproc_per_node=1 -m verl.trainer.sft_trainer \
    data.train_files="$TRAIN_DATA" \
    data.train_batch_size=8 \
    data.micro_batch_size_per_gpu=4 \
    data.max_length=1280 \
    data.truncation='error' \
    data.pad_mode=no_padding \
    data.use_dynamic_bsz=True \
    data.max_token_len_per_gpu=8192 \
    model.path="$MODEL_PATH" \
    model.use_remove_padding=True \
    model.enable_gradient_checkpointing=True \
    engine.model_dtype=bf16 \
    optim.lr=1e-5 \
    optim.weight_decay=0.01 \
    optim.lr_warmup_steps_ratio=0.1 \
    optim.lr_scheduler_type=cosine \
    trainer.total_epochs=3 \
    trainer.project_name=BES_Logical \
    trainer.experiment_name="$EXPERIMENT_NAME" \
    trainer.default_local_dir="$CKPT_DIR" \
    trainer.logger=['console','wandb'] \
    checkpoint.save_contents='["model","hf_model"]' \
    trainer.save_freq=-1 \
    trainer.test_freq=-1 \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    "$@"
