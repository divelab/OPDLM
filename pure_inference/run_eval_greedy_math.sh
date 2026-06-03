#!/bin/bash
# Pure inference eval runner.
# Edit MODELS, MODEL_BASES, and DATASETS below, then run:
#   bash pure_inference/run_eval_greedy.sh

set -e

# Change to repo root (parent of pure_inference/)
cd "$(dirname "$0")/.."

# ══════════════════════════════════════════════════════════════════════
# Models + their bases. Arrays must be same length.
# model_base: "bd3lm" (diffusion) or "qwen" (autoregressive)
# ══════════════════════════════════════════════════════════════════════
MODELS=(
    # "pretrained_models/Qwen/Qwen3-0.6B"
    # "pretrained_models/Qwen/Qwen3-4B"
    # "pretrained_models/BD3LM/Qwen3-0.6B-a2d-dapo_sft_10epoch"
    # "pretrained_models/experiment_ckpts/s128b4bs8_len2k_temp1_ForKL_Tea0.6B_StuARM_alpha0_inf_0404_033145_epoch-100"
    "experiments/s128b4bs64_ForKL_Tea4B_Stu4B_len2ks100_lr1e-5cos_0416_225333/ckpt/epoch-80"
    "experiments/s128b4bs64_ForKL_Tea4B_Stu4B_len2ks100_lr1e-5cos_0416_225333/ckpt/epoch-70"
    "experiments/s128b4bs64_ForKL_Tea4B_Stu4B_len2ks100_lr1e-5cos_0416_225333/ckpt/epoch-60"
    "experiments/s128b4bs64_ForKL_Tea4B_Stu4B_len2ks100_lr1e-5cos_0416_225333/ckpt/epoch-50"
    "experiments/s128b4bs64_ForKL_Tea4B_Stu4B_len2ks100_lr1e-5cos_0416_225333/ckpt/epoch-40"
)
MODEL_BASES=(
    # "qwen"
    # "qwen"
    "bd3lm"
    "bd3lm"
    "bd3lm"
    "bd3lm"
    "bd3lm"
)

# ══════════════════════════════════════════════════════════════════════
# Datasets (must exist as data/{NAME}.json) + per-dataset max tokens.
# Arrays must be same length.
# ══════════════════════════════════════════════════════════════════════
DATASETS=(
    # "GSM8K"
    "MATH500"
    "AIME2024"

)
DATASET_MAX_TOKENS=(
    # 1000
    2000
    8000        # AIME2024

)

# ══════════════════════════════════════════════════════════════════════
# Generation settings
# TEMPERATURE=0.0 → argmax (evaluation)
# NUM_RESPONSE=1  → greedy is deterministic, no need for multiple samples
# ══════════════════════════════════════════════════════════════════════
BLOCK_SIZE=4                # BD3LM only
DENOISING_STEPS=4           # BD3LM only
# Greedy: T=0, top_k=1 (deterministic, single sample)
TEMPERATURE=0.0
TOP_P=1.0
TOP_K=1
MIN_P=0.0
NUM_RESPONSE=1
GPU_MEM_UTIL=0.8
MAX_ACTIVE=128
TP=1
BASE_PORT=30600
OUT_DIR="pure_inference/results"
TAG="greedy"                 # subdir suffix: "greedy" or "sample"

# ══════════════════════════════════════════════════════════════════════

python pure_inference/eval.py \
    --models "${MODELS[@]}" \
    --model_bases "${MODEL_BASES[@]}" \
    --datasets "${DATASETS[@]}" \
    --dataset_max_tokens "${DATASET_MAX_TOKENS[@]}" \
    --block_size ${BLOCK_SIZE} \
    --denoising_steps_per_block ${DENOISING_STEPS} \
    --temperature ${TEMPERATURE} \
    --top_p ${TOP_P} \
    --top_k ${TOP_K} \
    --min_p ${MIN_P} \
    --num_response_per_task ${NUM_RESPONSE} \
    --gpu_memory_utilization ${GPU_MEM_UTIL} \
    --max_active ${MAX_ACTIVE} \
    --tensor_parallel_size ${TP} \
    --base_port ${BASE_PORT} \
    --out_dir ${OUT_DIR} \
    --tag ${TAG}
