#!/bin/bash
# Pure inference eval runner.
# Edit MODELS, MODEL_BASES, and DATASETS below, then run:
#   bash pure_inference/run_eval_greedy_17B_thinking.sh
#
# Single-GPU mode (default):
#   CUDA_VISIBLE_DEVICES=0 bash pure_inference/run_eval_greedy_17B_thinking.sh
#
# Multi-GPU data-parallel mode — set GPUS to a comma-separated list:
#   GPUS=0,1,2,7 bash pure_inference/run_eval_greedy_17B_thinking.sh
# Datasets get round-robin distributed across the listed GPUs (modulo sharding).
# Each shard runs in the background, logs to
# pure_inference/results/parallel_logs/, and the parent waits for all to
# finish. Each shard gets its own BASE_PORT (offset by shard_index*10) and
# its own TAG suffix so outputs don't clash.
set -e

# Change to repo root (parent of pure_inference/)
cd "$(dirname "$0")/.."

# ══════════════════════════════════════════════════════════════════════
# Multi-GPU parent launcher.
# If GPUS is set (e.g. GPUS=0,1,2,7) AND we're not already inside a shard
# (SHARD_INDEX unset), fork N children with proper CUDA_VISIBLE_DEVICES
# and SHARD_INDEX/NUM_SHARDS env vars, then wait for all to finish.
# ══════════════════════════════════════════════════════════════════════
if [ -n "${GPUS:-}" ] && [ -z "${SHARD_INDEX:-}" ]; then
    IFS=',' read -r -a _GPU_ARR <<< "${GPUS}"
    NUM_SHARDS_LOCAL=${#_GPU_ARR[@]}
    LOG_DIR="pure_inference/results/parallel_logs"
    mkdir -p "${LOG_DIR}"
    STAMP="$(date +%Y%m%d_%H%M%S)"
    echo "[parent] GPUS=${GPUS}  NUM_SHARDS=${NUM_SHARDS_LOCAL}  log_dir=${LOG_DIR}"
    PIDS=()
    for i in "${!_GPU_ARR[@]}"; do
        GPU="${_GPU_ARR[$i]}"
        LOG="${LOG_DIR}/$(basename "$0" .sh)_shard${i}of${NUM_SHARDS_LOCAL}_gpu${GPU}_${STAMP}.log"
        echo "  shard ${i}/${NUM_SHARDS_LOCAL} -> CUDA_VISIBLE_DEVICES=${GPU}  log=${LOG}"
        SHARD_INDEX=${i} NUM_SHARDS=${NUM_SHARDS_LOCAL} CUDA_VISIBLE_DEVICES="${GPU}" \
            GPUS="" bash "$0" > "${LOG}" 2>&1 &
        PIDS+=("$!")
    done
    echo "[parent] PIDs: ${PIDS[*]} — waiting..."
    FAIL=0
    for pid in "${PIDS[@]}"; do
        if ! wait "$pid"; then
            echo "[parent] shard pid=$pid FAILED (see log)"
            FAIL=1
        fi
    done
    if [ "${FAIL}" -eq 0 ]; then
        echo "[parent] All shards finished OK."
    else
        echo "[parent] One or more shards failed; check logs in ${LOG_DIR}"
    fi

    # ──────────────────────────────────────────────────────────────────
    # Aggregate per-dataset results from all shard log files and print
    # a single table to stdout (visible in tmux).
    # Each shard log contains lines like:
    #   "  Dataset: ARC_C  (max_token=4096, is_code=False)"
    # followed by:
    #   "train step: 1  ...  acc: 0.7474  avg length: 43.0  avg tok/step: 1.000  [...]"
    # We pair each Dataset header with the next acc line that follows it.
    # ──────────────────────────────────────────────────────────────────
    echo
    echo "════════════════════════════════════════════════════════════════════"
    echo "  Aggregated results (${NUM_SHARDS_LOCAL} shards, GPUs=${GPUS})"
    echo "════════════════════════════════════════════════════════════════════"
    printf "  %-30s %-10s %-10s %-10s\n" "Dataset" "Acc" "AvgLen" "TokPerStep"
    echo   "  ------------------------------------------------------------------"
    awk '
        /^  Dataset:/ {
            # Extract dataset name (between "Dataset: " and the first space)
            line = $0
            sub(/^  Dataset: */, "", line)
            sub(/  *\(.*$/, "", line)
            current_ds = line
            next
        }
        /^train step: 1/ && current_ds != "" {
            acc = ""; al = ""; tps = ""
            n = split($0, kv, "  ")
            for (i = 1; i <= n; i++) {
                if (kv[i] ~ /^acc:/)         { sub(/^acc: */, "", kv[i]); acc = kv[i] }
                if (kv[i] ~ /^avg length:/)  { sub(/^avg length: */, "", kv[i]); al = kv[i] }
                if (kv[i] ~ /^avg tok\/step:/){ sub(/^avg tok\/step: */, "", kv[i]); tps = kv[i] }
            }
            # Pretty-format (4dp acc, 1dp len, 3dp tps when numeric)
            if (acc != "")  acc = sprintf("%.4f", acc + 0)
            if (al  != "")  al  = sprintf("%.1f",  al  + 0)
            if (tps != "")  tps = sprintf("%.3f",  tps + 0)
            printf "  %-30s %-10s %-10s %-10s\n", current_ds, acc, al, tps
            current_ds = ""
        }
    ' "${LOG_DIR}/$(basename "$0" .sh)"_shard*of"${NUM_SHARDS_LOCAL}"_*_"${STAMP}".log
    echo "════════════════════════════════════════════════════════════════════"
    [ "${FAIL}" -eq 0 ] || exit 1
    exit 0
fi

# ══════════════════════════════════════════════════════════════════════
# Models + their bases. Arrays must be same length.
# model_base: "bd3lm" (diffusion) or "qwen" (autoregressive)
# ══════════════════════════════════════════════════════════════════════
MODELS=(
    "pretrained_models/Qwen/Qwen3-4B"
    "pretrained_models/Qwen/Qwen3-8B"
    "pretrained_models/Qwen/Qwen3-32B"
)
MODEL_BASES=(
    "qwen"
    "qwen"
    "qwen"
)

# ══════════════════════════════════════════════════════════════════════
# Datasets (must exist as data/{NAME}.json) + per-dataset max tokens
# and num_responses. Arrays must be same length.
# ══════════════════════════════════════════════════════════════════════
DATASETS=(
    "GSM8K"
    # "CEval"
    # "MMLU_Redux"
    # "ZebraLogic"
    # "LiveBench"
    # "MT_AIME2024"
    # "PolyMath"
    # "MLogiQA"
    # "INCLUDE_Lite"
)
# DATASET_MAX_TOKENS=(
#     # 16000
#     # 16000
#     # 16000
#     # 16000
#     # 16000
#     # 8000


#     # 4000
#     # 4000
#     # 4096
#     # 2000



#     # 16000
#     # 8000
#     # 8000
#     # 8000
#     # 8000
#     # 8000
#     # 8000
#     # 4000

#     # 4000
#     # 4000
#     # 4000
#     # 4000
#     # 4000

# )


# thinking logic
DATASET_MAX_TOKENS=( $(printf '16000 %.0s' "${DATASETS[@]}") )



# ══════════════════════════════════════════════════════════════════════
# Generation settings
# TEMPERATURE=0.0 → argmax (evaluation)
# NUM_RESPONSE=1  → greedy is deterministic, no need for multiple samples
# ══════════════════════════════════════════════════════════════════════
BLOCK_SIZE=4                # BD3LM only
DENOISING_STEPS=4           # BD3LM only
# Greedy: T=0, top_k=1 (deterministic, single sample)
TEMPERATURE=1.0
TOP_P=1.0
TOP_K=1
MIN_P=0.0
NUM_RESPONSE=1
SEED=42
GPU_MEM_UTIL=0.9
MAX_ACTIVE=16
TP=1
BASE_PORT=17693
OUT_DIR="pure_inference/results"
TAG="greedy_Qwen"                 # subdir suffix: "greedy" or "sample"

# ══════════════════════════════════════════════════════════════════════
# Multi-GPU data-parallel sharding (optional).
# If NUM_SHARDS > 1, slice DATASETS / DATASET_MAX_TOKENS so this process
# only handles every NUM_SHARDS-th entry starting at SHARD_INDEX.
# ══════════════════════════════════════════════════════════════════════
NUM_SHARDS=${NUM_SHARDS:-1}
SHARD_INDEX=${SHARD_INDEX:-0}
if [ "${NUM_SHARDS}" -gt 1 ]; then
    _DS=()
    _MT=()
    for i in "${!DATASETS[@]}"; do
        if [ $((i % NUM_SHARDS)) -eq "${SHARD_INDEX}" ]; then
            _DS+=("${DATASETS[$i]}")
            _MT+=("${DATASET_MAX_TOKENS[$i]}")
        fi
    done
    DATASETS=("${_DS[@]}")
    DATASET_MAX_TOKENS=("${_MT[@]}")
    BASE_PORT=$((BASE_PORT + SHARD_INDEX * 10))
    TAG="${TAG}_shard${SHARD_INDEX}of${NUM_SHARDS}"
    if [ "${#DATASETS[@]}" -eq 0 ]; then
        echo "[shard ${SHARD_INDEX}/${NUM_SHARDS}] no datasets in this shard; exiting."
        exit 0
    fi
    echo "[shard ${SHARD_INDEX}/${NUM_SHARDS}] datasets=${DATASETS[*]}  port=${BASE_PORT}"
fi

# Thinking mode: false → disable_thinking (default; chat template injects an
# empty <think></think> block), true → enable_thinking (template lets the model
# reason first). Some diffusion models (e.g. SDAR) need true on MC tasks to
# avoid emitting <|im_end|> immediately. SDAR's official eval also uses true.
ENABLE_THINKING=false

REMASKING_STRATEGY="low_confidence_static"  # low_confidence_static, low_confidence_dynamic
DYNAMIC_THRESHOLD=0.1

# ══════════════════════════════════════════════════════════════════════
# Crash-safe chunking (see pure_inference/eval.py --num_chunks).
# Each (model, dataset) is split into N contiguous chunks; per-chunk
# JSON is saved before the next chunk starts, and reruns skip already-
# completed chunks. 1 = no chunking. Use 2-8 for long thinking-mode runs.
#
# Two ways to use chunking with this script:
# 1. Sequential (single-machine):  bash script.sh
#    eval.py loops over all N chunks, then merges + runs reward.
# 2. Parallel (multi-machine, env-driven shard): for each k in 0..N-1, run
#       NUM_CHUNKS=N CHUNK_INDEX=k CUDA_VISIBLE_DEVICES=<gpu> bash script.sh
#    Each writes its own chunk JSON; eval.py auto-detects env vars and
#    skips merge/reward. After all N chunks exist, run once more without
#    env vars to merge + run reward.
# ══════════════════════════════════════════════════════════════════════
NUM_CHUNKS=1   # Qwen vLLM is stable end-to-end; no chunking needed.

if [ "${REMASKING_STRATEGY}" = "low_confidence_dynamic" ]; then
    _THR_TAG=$(printf '%s' "${DYNAMIC_THRESHOLD}" | tr '.' 'p')
    TAG="${TAG}_dyn${_THR_TAG}"
fi

# Build optional flag for enable_thinking; eval.py's default is disabled.
THINKING_FLAG=""
if [ "${ENABLE_THINKING}" = "true" ] || [ "${ENABLE_THINKING}" = "1" ]; then
    THINKING_FLAG="--enable_thinking"
fi

export TRACERL_SEED=$SEED

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
    --tag ${TAG} \
    --remasking_strategy ${REMASKING_STRATEGY} \
    --dynamic_threshold ${DYNAMIC_THRESHOLD} \
    --num_chunks ${NUM_CHUNKS} \
    ${THINKING_FLAG}
