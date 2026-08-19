#!/usr/bin/env bash
#SBATCH --job-name=opdlm_ot3_06b
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --mem=128G
#SBATCH --chdir=/scratch/user/shubhamprshr_tamu.edu/OPDLM
#SBATCH --output=/scratch/user/shubhamprshr_tamu.edu/OPDLM/logs/%x-%j.out
#SBATCH --err=/scratch/user/shubhamprshr_tamu.edu/OPDLM/logs/%x-%j.err
#SBATCH --time=4:00:00
#SBATCH --partition=def
#SBATCH --qos=standard

set -euo pipefail

REPO_ROOT="/scratch/user/shubhamprshr_tamu.edu/OPDLM"
cd "$REPO_ROOT"

source ~/.bashrc
module load CUDA/12.6.0
eval "$(conda shell.bash hook)"
conda activate "${CONDA_ENV:-opdlm}"

export CUDACXX=$CUDA_HOME/bin/nvcc
export PATH=$CONDA_PREFIX/bin:$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export DS_SKIP_CUDA_CHECK=1
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-9.0}"

export RUN_TIMESTAMP=$(date +"%m%d_%H%M%S")

DATA_PATH=/scratch/user/shubhamprshr_tamu.edu/OPDLM/data

export HF_HOME=$DATA_PATH
export TRITON_CACHE_DIR="$REPO_ROOT/.triton_cache/${SLURM_JOB_ID:-local}"
# Keep this run isolated from any stale EXP_BASE exported by another checkout.
export EXP_BASE="$REPO_ROOT/experiments"
mkdir -p "$HF_HOME" "$TRITON_CACHE_DIR" "$EXP_BASE" logs

# Fresh A2D-converted Qwen3-0.6B initialization (not trained OPDLM weights).
STUDENT="${STUDENT:-$REPO_ROOT/pretrained_models/Qwen3-0.6B-a2d-init}"
TEACHER="${TEACHER:-$REPO_ROOT/pretrained_models/Qwen3-0.6B}"
TRAIN_DATA="$REPO_ROOT/data/OpenThoughts3.json"

JOB_TOKEN="${SLURM_JOB_ID:-0}"
PORT_OFFSET=$((JOB_TOKEN % 1000))
EXPERIMENT_PORT=$((22000 + PORT_OFFSET))
ROLLOUT_BASE_PORT=$((24000 + PORT_OFFSET))

NUM_GPUS="${NUM_GPUS:-4}"
PPO_BATCH_SIZE="${PPO_BATCH_SIZE:-8}"
BATCH_SIZE_LM="${BATCH_SIZE_LM:-2}"
LR_WARMUP_STEPS="${LR_WARMUP_STEPS:-5}"

GRADIENT_ACCUMULATION_STEPS=$((PPO_BATCH_SIZE / (BATCH_SIZE_LM * NUM_GPUS)))

DEEPSPEED_FILE="1_node_${NUM_GPUS}_gpus_deepspeed_zero3"

RUN_NAME="${RUN_NAME:-OpenThoughts3_A2D06B_bs8_len4k_ramp100_warmup${LR_WARMUP_STEPS}_${RUN_TIMESTAMP}}"

accelerate launch \
    --num_machines 1 \
    --machine_rank 0 \
    --main_process_ip 127.0.0.1 \
    --main_process_port $EXPERIMENT_PORT \
    --config_file accelerate_configs/$DEEPSPEED_FILE.yaml \
    rl.py \
    config=configs/rl_bd3lm.yaml \
    rollout.base_port=$ROLLOUT_BASE_PORT \
    rollout.num_task_per_step=128 \
    training.batch_size_lm=$BATCH_SIZE_LM \
    training.gradient_accumulation_steps=$GRADIENT_ACCUMULATION_STEPS \
    dataset.train_dataset=OpenThoughts3 \
    dataset.num_data_epochs=1 \
    dataset.skip_correctness=True \
    dataset.skip_code_correctness=True \
    evaluation.eval_dataset=GSM8K \
    evaluation.max_token=1000 \
    optimizer.params.learning_rate=1e-5 \
    lr_scheduler.params.warmup_steps=$LR_WARMUP_STEPS \
    max_token_schedule.end=4000 \
    max_token_schedule.ramp_steps=100 \
    model.pretrained_model=$STUDENT \
    model.teacher_model=$TEACHER \
    wandb.project=openthoughts3_OPDLM \
    wandb.group=QwenA2D0.6B_OpenThoughts3 \
    wandb.run_name=$RUN_NAME \
    training.one_state_per_block=True \
    dynamic_threshold_schedule.enabled=False
