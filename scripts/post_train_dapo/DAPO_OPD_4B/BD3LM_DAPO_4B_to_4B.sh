#!/usr/bin/env bash
#SBATCH --job-name=fixed_train
#SBATCH --nodes=1
#SBATCH --gres=gpu:1             # Default. Override at submit time for multi-GPU runs.
#SBATCH --output=./logs/%x-%j.out  # Standard output (progress bars, prints)
#SBATCH --err=./logs/%x-%j.err     # Error logs (crashes, tracebacks)
##SBATCH --partition=short       # Choose your queue (e.g., short, medium, long)
##SBATCH --time=4:00:00          # Max time for the job

#SBATCH --partition=2xlong       # Choose your queue (e.g., short, medium, long)
#SBATCH --time=2-00:00:00          # Max time for the job


##SBATCH --partition=medium       # Choose your queue (e.g., short, medium, long)
##SBATCH --time=8:00:00          # Max time for the job


##SBATCH --time=16:00:00          # Max time for the job
##SBATCH --partition=long       # Choose your queue (e.g., short, medium, long)


#######
# USAGE:
# sbatch --gres=gpu:1 --export=NUM_DEVICES=1 run.sh
#######

source ~/.bashrc
eval "$(conda shell.bash hook)"
conda activate tracerl-vision

rm -rf /dev/shm/torch_cache 2>/dev/null || true  # clean stale triton cache from prior jobs

export CUDA_HOME=/sw/eb/sw/CUDA/12.8.0
export CUDACXX=$CUDA_HOME/bin/nvcc
export PATH=$CONDA_PREFIX/bin:$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
# export CUDA_DEVICE_ORDER=PCI_BUS_ID

set -eox pipefail

DATA_PATH=/scratch/user/xingyu.su_tamu.edu/traceRL

export HF_HOME=$DATA_PATH
export TRITON_CACHE_DIR=$DATA_PATH/triton_cache

# student and teacher models
STUDENT=$DATA_PATH/pretrained_models/BD3LM/Qwen3-4B-a2d-init
TEACHER=$DATA_PATH/pretrained_models/Qwen/Qwen3-4B


PORT_OFFSET=10
EXPERIMENT_PORT=$((20200 + PORT_OFFSET))
ROLLOUT_BASE_PORT=$((20300 + PORT_OFFSET))


NUM_GPUS=1
PPO_BATCH_SIZE=2
BATCH_SIZE_LM=2
STEPS_PER_BLOCK=4
GRADIENT_ACCUMULATION_STEPS=$((PPO_BATCH_SIZE / (BATCH_SIZE_LM * NUM_GPUS)))
if [ $GRADIENT_ACCUMULATION_STEPS -lt 1 ]; then
  echo "Error: GRADIENT_ACCUMULATION_STEPS is less than 1. Please adjust PPO_BATCH_SIZE, BATCH_SIZE_LM, or NUM_GPUS."
  exit 1
fi


DEEPSPEED_FILE="1_node_${NUM_GPUS}_gpus_deepspeed_zero3"

RUN_NAME=s128b4bs2_ForKL_Tea4B_Stu4B_len014ks100cos_inf

# exit 0

# BD3LM 
python rl.py \
    experiment.deepspeed_file=$DEEPSPEED_FILE \
    config=configs/rl_bd3lm.yaml \
    experiment.port=$EXPERIMENT_PORT \
    rollout.base_port=$ROLLOUT_BASE_PORT \
    rollout.num_task_per_step=128 \
    rollout.num_response_per_task=1 \
    training.batch_size_lm=$BATCH_SIZE_LM \
    training.gradient_accumulation_steps=$GRADIENT_ACCUMULATION_STEPS \
    training.exclude_im_end=True \
    rollout.max_token=-1 \
    training.max_gen_length=-1 \
    evaluation.max_token=1000 \
    wandb.group=QwenARM4B_DAPO \
    wandb.run_name=$RUN_NAME \
    dataset.train_dataset=DAPO_Math_17k \
    evaluation.eval_dataset=GSM8K \
    training.block_size=4 \
    rollout.block_size=4 \
    evaluation.block_size=4 \
    rollout.denoising_steps_per_block=$STEPS_PER_BLOCK \
    evaluation.denoising_steps_per_block=$STEPS_PER_BLOCK \
    experiment.total_step=-1 \
    lr_scheduler.scheduler=constant \
    model.pretrained_model=$STUDENT \
    model.teacher_model=$TEACHER \
    max_token_schedule.enabled=True \
    max_token_schedule.start=100 \
    max_token_schedule.end=4000 \
    max_token_schedule.ramp_steps=100 \
    max_token_schedule.type=cos 
    


