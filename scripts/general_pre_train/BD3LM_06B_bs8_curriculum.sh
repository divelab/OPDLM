#!/usr/bin/env bash
# Curriculum training: continue from OPDLM 0.6B (trained at block_size=4)
# with block_size=8.  The teacher stays the same (Qwen3-0.6B ARM).
#
# Download the OPDLM 0.6B checkpoint first (replace repo id if HF name differs):
#   huggingface-cli download divelab/OPDLM-0.6B --local-dir $DATA_PATH/pretrained_models/OPDLM-0.6B
# Then set STUDENT below to that path.
#
# Run directly (no SBATCH):
#   bash scripts/general_pre_train/BD3LM_06B_bs8_curriculum.sh

source ~/.bashrc
eval "$(conda shell.bash hook)"
conda activate opdlm

rm -rf /dev/shm/torch_cache 2>/dev/null || true

export CUDA_HOME=/usr/local/cuda
export CUDACXX=$CUDA_HOME/bin/nvcc
export PATH=$CONDA_PREFIX/bin:$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}

set -eox pipefail
export RUN_TIMESTAMP=$(date +"%m%d_%H%M%S")

DATA_PATH=/home/shubhamprshr/OPDLM

export HF_HOME=$DATA_PATH
export TRITON_CACHE_DIR=$DATA_PATH/triton_cache
export EXP_BASE=/home/shubhamprshr/OPDLM/experiment

# Student: OPDLM 0.6B checkpoint (pretrained at block_size=4)
STUDENT=$DATA_PATH/pretrained_models/OPDLM-0.6B
TEACHER=$DATA_PATH/pretrained_models/Qwen/Qwen3-0.6B

PORT_OFFSET=74
EXPERIMENT_PORT=$((20200 + PORT_OFFSET))
ROLLOUT_BASE_PORT=$((20300 + PORT_OFFSET))

NUM_GPUS=8
PPO_BATCH_SIZE=8
BATCH_SIZE_LM=1
GRADIENT_ACCUMULATION_STEPS=$((PPO_BATCH_SIZE / (BATCH_SIZE_LM * NUM_GPUS)))
if [ $GRADIENT_ACCUMULATION_STEPS -lt 1 ]; then
  echo "Error: GRADIENT_ACCUMULATION_STEPS is less than 1."
  exit 1
fi

DEEPSPEED_FILE="1_node_${NUM_GPUS}_gpus_deepspeed_zero3"

BLOCK_SIZE=8
DENOISING_STEPS=8

RUN_NAME=s128b${BLOCK_SIZE}bs8_curriculum_from_bs4_lr1e-6cos_warm20_revkl_onestate_topk16
export DS_SKIP_CUDA_CHECK=1
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
    rollout.block_size=$BLOCK_SIZE \
    rollout.denoising_steps_per_block=$DENOISING_STEPS \
    training.block_size=$BLOCK_SIZE \
    training.batch_size_lm=$BATCH_SIZE_LM \
    training.gradient_accumulation_steps=$GRADIENT_ACCUMULATION_STEPS \
    training.one_state_per_block=True \
    training.top_k_logits=16 \
    evaluation.block_size=$BLOCK_SIZE \
    evaluation.denoising_steps_per_block=$DENOISING_STEPS \
    evaluation.eval_dataset=GSM8K \
    evaluation.max_token=1000 \
    dataset.train_dataset=opdlm_train \
    optimizer.params.learning_rate=1e-6 \
    lr_scheduler.params.warmup_steps=20 \
    training.reverse_kl_weight=1.0 \
    max_token_schedule.end=4000 \
    max_token_schedule.ramp_steps=100 \
    model.pretrained_model=$STUDENT \
    model.teacher_model=$TEACHER \
    wandb.project=opdlmv2 \
    wandb.group=QwenARM0.6B_bs8_curriculum \
    wandb.run_name=$RUN_NAME \
    dynamic_threshold_schedule.enabled=False \
