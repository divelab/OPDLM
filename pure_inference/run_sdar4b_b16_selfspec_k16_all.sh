#!/usr/bin/env bash
set -euo pipefail

# Full SDAR-4B-Chat-b16 evaluation suite:
#   1. K=16 self-speculative decoding
#   2. Native K=16 blockwise decoding with static remasking
#   3. Native K=16 blockwise decoding with dynamic remasking
#
# Each run uses all ten GPUs as independent shards, then merges and scores
# with pure_inference/eval.py before advancing to the next run.
#
# Usage:
#   bash pure_inference/run_sdar4b_b16_selfspec_k16_all.sh
#
# Optional overrides:
#   PYTHON_BIN=/path/to/python MODEL_PATH=/path/to/model \
#   GPU_IDS="0 1 2 3 4 5 6 7 8 9" DYNAMIC_THRESHOLD=0.9 \
#   bash pure_inference/run_sdar4b_b16_selfspec_k16_all.sh

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON_BIN:-/home/shubhamprshr/conda/envs/opdlm/bin/python}"
MODEL_PATH="${MODEL_PATH:-models/SDAR-4B-Chat-b16}"
GPU_IDS_STRING="${GPU_IDS:-0 1 2 3 4 5 6 7 8 9}"
DYNAMIC_THRESHOLD="${DYNAMIC_THRESHOLD:-0.9}"
read -r -a GPU_IDS_ARRAY <<< "$GPU_IDS_STRING"
NUM_SHARDS="${#GPU_IDS_ARRAY[@]}"

if [[ "$NUM_SHARDS" -eq 0 ]]; then
    echo "GPU_IDS must contain at least one GPU index." >&2
    exit 2
fi
if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "Python executable not found: $PYTHON_BIN" >&2
    exit 2
fi
if [[ ! -f "$MODEL_PATH/config.json" ]]; then
    echo "Model checkpoint not found: $MODEL_PATH" >&2
    exit 2
fi

export PATH="$(dirname "$PYTHON_BIN"):$PATH"

DATASETS=(
    AIME2024
    AIME2025
    MATH500
    GSM8K
    GPQA_Diamond_selfspec
    HumanEval
    MBPP
    LCB_v6
)
MAX_TOKENS=(
    8000
    8000
    4000
    2000
    4000
    8000
    1000
    8000
)
MODES=(
    self_speculative
    static
    dynamic
)

MODEL_ABS="$(realpath "$MODEL_PATH")"
MODEL_NAME="$(basename "$MODEL_ABS")"
MODEL_STEM="${MODEL_ABS//\//.}"
LOG_ROOT="pure_inference/results/sdar4b_b16_k16_all_modes_logs"
mkdir -p "$LOG_ROOT"

cleanup_children() {
    local pid
    for pid in $(jobs -pr); do
        kill "$pid" 2>/dev/null || true
    done
}
trap cleanup_children INT TERM

run_dataset() {
    local run_index="$1"
    local mode="$2"
    local dataset="$3"
    local max_tokens="$4"
    local threshold_tag="${DYNAMIC_THRESHOLD//./p}"
    local tag
    local -a mode_args
    case "$mode" in
        self_speculative)
            tag="selfspec_k16_${NUM_SHARDS}gpu_${max_tokens}tok_all_suite"
            mode_args=(
                --generation-mode self_speculative
                --draft-block-size 16
                --self-speculative-margin-threshold 0
                --remasking_strategy low_confidence_static
            )
            ;;
        static)
            tag="native_k16_static_${NUM_SHARDS}gpu_${max_tokens}tok_all_suite"
            mode_args=(
                --generation-mode blockwise
                --remasking_strategy low_confidence_static
            )
            ;;
        dynamic)
            tag="native_k16_dynamic_thr${threshold_tag}_${NUM_SHARDS}gpu_${max_tokens}tok_all_suite"
            mode_args=(
                --generation-mode blockwise
                --remasking_strategy low_confidence_dynamic
                --dynamic_threshold "$DYNAMIC_THRESHOLD"
            )
            ;;
        *)
            echo "Unknown mode: $mode" >&2
            exit 2
            ;;
    esac
    local project="pure_inference/results/${MODEL_NAME}_${dataset}_${tag}"
    local log_dir="${LOG_ROOT}/${mode}/${dataset}"
    local shard gpu port output_file pid status
    local -a pids=()
    local -a launched_shards=()

    mkdir -p "$log_dir"
    echo
    echo "======================================================================"
    echo "Mode: $mode | Dataset: $dataset | max tokens: $max_tokens"
    echo "Block size: 16 | shards: $NUM_SHARDS"
    echo "Output:  $project"
    echo "======================================================================"

    for shard in "${!GPU_IDS_ARRAY[@]}"; do
        gpu="${GPU_IDS_ARRAY[$shard]}"
        port=$((24000 + run_index * 100 + shard))
        output_file="${project}/temp_data/outputs-eval-${MODEL_STEM}-${dataset}_chunk${shard}of${NUM_SHARDS}.json"

        if [[ -s "$output_file" ]]; then
            echo "[resume] shard $shard already complete: $output_file"
            continue
        fi

        echo "[launch] shard $shard/$((NUM_SHARDS - 1)) on GPU $gpu"
        env CUDA_VISIBLE_DEVICES="$gpu" NUM_CHUNKS="$NUM_SHARDS" CHUNK_INDEX="$shard" \
            "$PYTHON_BIN" pure_inference/eval.py \
                --models "$MODEL_PATH" \
                --model_bases sdar \
                --datasets "$dataset" \
                --dataset_max_tokens "$max_tokens" \
                --dataset_num_responses 1 \
                --block_size 16 \
                --denoising_steps_per_block 16 \
                --max_active 8 \
                --num_chunks "$NUM_SHARDS" \
                --base_port "$port" \
                --tag "$tag" \
                "${mode_args[@]}" \
                > "${log_dir}/chunk${shard}.log" 2>&1 &
        pids+=("$!")
        launched_shards+=("$shard")
    done

    status=0
    for shard in "${!pids[@]}"; do
        pid="${pids[$shard]}"
        if ! wait "$pid"; then
            echo "[error] mode $mode dataset $dataset shard ${launched_shards[$shard]} failed." >&2
            tail -n 80 "${log_dir}/chunk${launched_shards[$shard]}.log" >&2 || true
            status=1
        fi
    done
    if [[ "$status" -ne 0 ]]; then
        echo "Fix the failed shard and rerun this script; completed shards will be reused." >&2
        exit "$status"
    fi

    echo "[merge+score] $mode / $dataset"
    "$PYTHON_BIN" pure_inference/eval.py \
        --models "$MODEL_PATH" \
        --model_bases sdar \
        --datasets "$dataset" \
        --dataset_max_tokens "$max_tokens" \
        --dataset_num_responses 1 \
        --block_size 16 \
        --denoising_steps_per_block 16 \
        --max_active 8 \
        --num_chunks "$NUM_SHARDS" \
        --tag "$tag" \
        "${mode_args[@]}" \
        2>&1 | tee "${log_dir}/merge_score.log"
}

run_index=0
for mode in "${MODES[@]}"; do
    for index in "${!DATASETS[@]}"; do
        run_dataset "$run_index" "$mode" "${DATASETS[$index]}" "${MAX_TOKENS[$index]}"
        run_index=$((run_index + 1))
    done
done

echo
echo "All SDAR-4B-Chat-b16 K=16 self-speculative/static/dynamic evaluations completed."
echo "Logs: $LOG_ROOT"
