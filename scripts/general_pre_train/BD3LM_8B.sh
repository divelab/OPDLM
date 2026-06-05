#!/usr/bin/env bash
#SBATCH --job-name=fixed_train
#SBATCH --nodes=1
#SBATCH --gres=gpu:8             # Default. Override at submit time for multi-GPU runs.
#SBATCH --output=./logs/%x-%j.out  # Standard output (progress bars, prints)
#SBATCH --err=./logs/%x-%j.err     # Error logs (crashes, tracebacks)

#SBATCH --time=2-00:00:00        # Max time for the job (48h)
#SBATCH --partition=def
#SBATCH --qos=standard

#######
# On-policy ForKL, 8B, with top_k_logits=16 (Nemotron-style sparse KL
# restricted to teacher's top-16 tokens). 8B sibling of
# BD3LM_vision_4B_no_threshold_topk16.sh.
#
# USAGE:
# sbatch scripts/general_pre_train/vision/BD3LM_vision_8B_no_threshold_top16.sh
#######

source ~/.bashrc
eval "$(conda shell.bash hook)"
conda activate tracerl-vision

rm -rf /dev/shm/torch_cache 2>/dev/null || true  # clean stale triton cache from prior jobs

export CUDA_HOME=/sw/eb/sw/CUDA/12.8.0
export CUDACXX=$CUDA_HOME/bin/nvcc
export PATH=$CONDA_PREFIX/bin:$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}

set -eox pipefail
export RUN_TIMESTAMP=$(date +"%m%d_%H%M%S")

DATA_PATH=/scratch/user/xingyu.su_tamu.edu/traceRL

export HF_HOME=$DATA_PATH
export TRITON_CACHE_DIR=$DATA_PATH/triton_cache
export EXP_BASE=/scratch/project/prj-02-pi-shuiwang-ji/xingyu/traceRL/experiments

# student and teacher models
STUDENT=$DATA_PATH/pretrained_models/BD3LM/Qwen3-8B-a2d-init
TEACHER=$DATA_PATH/pretrained_models/Qwen/Qwen3-8B


PORT_OFFSET=63
EXPERIMENT_PORT=$((20200 + PORT_OFFSET))
ROLLOUT_BASE_PORT=$((20300 + PORT_OFFSET))


NUM_GPUS=8
PPO_BATCH_SIZE=8
BATCH_SIZE_LM=1
STEPS_PER_BLOCK=4
GRADIENT_ACCUMULATION_STEPS=$((PPO_BATCH_SIZE / (BATCH_SIZE_LM * NUM_GPUS)))
if [ $GRADIENT_ACCUMULATION_STEPS -lt 1 ]; then
  echo "Error: GRADIENT_ACCUMULATION_STEPS is less than 1. Please adjust PPO_BATCH_SIZE, BATCH_SIZE_LM, or NUM_GPUS."
  exit 1
fi


DEEPSPEED_FILE="1_node_${NUM_GPUS}_gpus_deepspeed_zero3"

RUN_NAME=s128b4bs8_ForKL_Tea8B_Stu8B_len4ks100_lr1e-5cos_onestate_fix128_top16

# BD3LM

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
    dataset.train_dataset=opdlm_train \
    evaluation.eval_dataset=GSM8K \
    evaluation.max_token=1000 \
    optimizer.params.learning_rate=1e-5 \
    max_token_schedule.end=4000 \
    max_token_schedule.ramp_steps=100 \
    model.pretrained_model=$STUDENT \
    model.teacher_model=$TEACHER \
    wandb.group=QwenARM8B_General \
    wandb.run_name=$RUN_NAME \
    training.one_state_per_block=True \
    dynamic_threshold_schedule.enabled=False \
    training.top_k_logits=16 \

