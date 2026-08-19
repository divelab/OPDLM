#!/usr/bin/env bash
#SBATCH --job-name=disagr-tcs-06b
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --mem=128G
#SBATCH --time=03:00:00
#SBATCH --partition=def
#SBATCH --qos=standard
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# Multi-hour, four-GPU run for disagreement-routed block-TCS correction.
# Run from the repository root with:
#   sbatch scripts/smoke/MC_TCSM_06B_4GPU.sh

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$REPO_ROOT"

source ~/.bashrc
module load CUDA/12.6.0
eval "$(conda shell.bash hook)"
conda activate "${CONDA_ENV:-opdlm}"

export CUDACXX="$CUDA_HOME/bin/nvcc"
export PATH="$CONDA_PREFIX/bin:$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export DS_SKIP_CUDA_CHECK=1
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-9.0}"

export RUN_TIMESTAMP="${RUN_TIMESTAMP:-$(date +%m%d_%H%M%S)}"
LEARNING_RATE="${LEARNING_RATE:-5e-6}"
RUN_NAME_BASE="${RUN_NAME_BASE:-opdlm_06b_disagreement_tcs}"
export HF_HOME="${HF_HOME:-$REPO_ROOT/.hf_cache}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$REPO_ROOT/.triton_cache/${SLURM_JOB_ID:-local}}"
export EXP_BASE="${EXP_BASE:-$REPO_ROOT/experiments}"
mkdir -p "$HF_HOME" "$TRITON_CACHE_DIR" "$EXP_BASE" logs
LIVE_LOG="$REPO_ROOT/logs/mc_tcsm_06b_4gpu_${RUN_TIMESTAMP}.log"
echo "Live log: $LIVE_LOG"
echo "Learning rate: $LEARNING_RATE"

STUDENT="$REPO_ROOT/pretrained_models/OPDLM-0.6B"
TEACHER="$REPO_ROOT/pretrained_models/Qwen3-0.6B"
TRAIN_DATA="$REPO_ROOT/data/opdlm_train.json"

for required_path in "$STUDENT" "$TEACHER" "$TRAIN_DATA"; do
    if [[ ! -e "$required_path" ]]; then
        echo "Missing smoke-test prerequisite: $required_path" >&2
        exit 1
    fi
done

JOB_TOKEN="${SLURM_JOB_ID:-0}"
PORT_OFFSET=$((JOB_TOKEN % 1000))
EXPERIMENT_PORT=$((22000 + PORT_OFFSET))
ROLLOUT_BASE_PORT=$((24000 + PORT_OFFSET))

accelerate launch \
    --num_machines 1 \
    --machine_rank 0 \
    --main_process_ip 127.0.0.1 \
    --main_process_port "$EXPERIMENT_PORT" \
    --config_file accelerate_configs/1_node_4_gpus_deepspeed_zero3.yaml \
    rl.py \
    config=configs/rl_bd3lm.yaml \
    experiment.total_step=100 \
    experiment.stop_RL_step=-1 \
    experiment.save_every=5 \
    experiment.eval_every=10 \
    dataset.num_data_epochs=-1 \
    dataset.train_dataset=opdlm_train \
    rollout.base_port="$ROLLOUT_BASE_PORT" \
    rollout.num_task_per_step=128 \
    rollout.num_response_per_task=1 \
    rollout.max_active=128 \
    rollout.max_token=4000 \
    rollout.gpu_memory_utilization=0.55 \
    training.max_gen_length=4000 \
    training.batch_size_lm=2 \
    training.gradient_accumulation_steps=1 \
    training.one_state_per_block=True \
    training.top_k_logits=16 \
    model.pretrained_model="$STUDENT" \
    model.teacher_model="$TEACHER" \
    max_token_schedule.enabled=True \
    max_token_schedule.start=100 \
    max_token_schedule.end=4000 \
    max_token_schedule.ramp_steps=100 \
    dynamic_threshold_schedule.enabled=False \
    evaluation.eval_dataset=GSM8K \
    evaluation.max_token=1000 \
    evaluation.run_before_training=True \
    optimizer.params.learning_rate="$LEARNING_RATE" \
    wandb.enabled=True \
    wandb.project=mc_tcs_06b \
    wandb.group=disagreement_tcs \
    wandb.run_name="$RUN_NAME_BASE" \
    tcsm.enabled=True \
    tcsm.lambda_max=1.0 \
    tcsm.ramp_steps=0 \
    tcsm.ramp_type=linear \
    tcsm.counterfactual_batch_size=16 \
    tcsm.show_progress=False \
    2>&1 | tee "$LIVE_LOG"
