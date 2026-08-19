#!/usr/bin/env bash
#SBATCH --job-name=idlm-eval
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --partition=def
#SBATCH --qos=standard
#SBATCH --chdir=/scratch/user/shubhamprshr_tamu.edu/OPDLM
#SBATCH --output=/scratch/user/shubhamprshr_tamu.edu/OPDLM/logs/%x-%j.out
#SBATCH --error=/scratch/user/shubhamprshr_tamu.edu/OPDLM/logs/%x-%j.err

set -euo pipefail

REPO_ROOT="/scratch/user/shubhamprshr_tamu.edu/OPDLM"
cd "$REPO_ROOT"

source ~/.bashrc
module load CUDA/12.6.0
eval "$(conda shell.bash hook)"
conda activate "${CONDA_ENV:-opdlm}"

export HF_HOME="$REPO_ROOT/data"
export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

mkdir -p "$REPO_ROOT/logs" "$REPO_ROOT/pure_inference/results"

MODEL="${MODEL:-$REPO_ROOT/pretrained_models/OPDLM-0.6B}"
STRIDE="${STRIDE:-4}"
MATH_MAX_TOKEN="${MATH_MAX_TOKEN:-2000}"
CODE_MAX_TOKEN="${CODE_MAX_TOKEN:-1024}"

# These are conservative throughput-oriented defaults for a 0.6B model on a
# single 141 GB H200. Override them at submission time if desired, e.g.
# `MATH_BATCH_SIZE=8 sbatch pure_inference/run_test_idlm_transfer.sh`.
MATH_BATCH_SIZE="${MATH_BATCH_SIZE:-16}"
CODE_BATCH_SIZE="${CODE_BATCH_SIZE:-32}"
PROMPT_STYLE="${PROMPT_STYLE:-idlm}"
OUT_DIR="${OUT_DIR:-pure_inference/results}"

EXTRA_ARGS=()
if [[ -n "${LIMIT:-}" ]]; then
    EXTRA_ARGS+=(--limit "$LIMIT")
fi

# Register the repo-local A2D architecture in the exact interpreter that runs
# each evaluation. This avoids relying on native Transformers support for the
# `a2d-qwen3` model type.
PYTHON_RUNNER="import runpy; runpy.run_path('pure_inference/test_idlm_transfer.py', run_name='__main__')"

run_dataset() {
    local dataset="$1"
    local max_token="$2"
    local batch_size="$3"

    echo "============================================================"
    echo "Dataset=$dataset max_token=$max_token batch_size=$batch_size"
    echo "============================================================"

    python -u -c "$PYTHON_RUNNER" \
        --dataset "$dataset" \
        --model "$MODEL" \
        --max_token "$max_token" \
        --batch_size "$batch_size" \
        --stride "$STRIDE" \
        --prompt_style "$PROMPT_STYLE" \
        --out_dir "$OUT_DIR" \
        "${EXTRA_ARGS[@]}"
}

run_dataset GSM8K    "$MATH_MAX_TOKEN" "$MATH_BATCH_SIZE"
run_dataset MATH500  "$MATH_MAX_TOKEN" "$MATH_BATCH_SIZE"
run_dataset HumanEval "$CODE_MAX_TOKEN" "$CODE_BATCH_SIZE"
run_dataset MBPP      "$CODE_MAX_TOKEN" "$CODE_BATCH_SIZE"
