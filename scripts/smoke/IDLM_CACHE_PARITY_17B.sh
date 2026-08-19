#!/usr/bin/env bash
#SBATCH --job-name=idlm-cache-17b
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:20:00
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
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

MODEL="$REPO_ROOT/pretrained_models/OPDLM-1.7B"
CACHED_ROOT="pure_inference/results/cache_smoke_17b_cached"
UNCACHED_ROOT="pure_inference/results/cache_smoke_17b_uncached"
RESULT_SUFFIX="OPDLM-1.7B_GSM8K_introspective_B4/outputs.json"

if [[ ! -f "$MODEL/config.json" || ! -f "$MODEL/model.safetensors" ]]; then
    echo "Missing OPDLM-1.7B checkpoint: $MODEL" >&2
    exit 1
fi

CUDA_VISIBLE_DEVICES=0 /usr/bin/time -f 'cached_wall_seconds=%e' \
    -o logs/idlm_cache_17b_cached_time_${SLURM_JOB_ID}.log \
    python -u pure_inference/test_idlm_transfer.py \
    --model "$MODEL" --dataset GSM8K --max_token 512 --batch_size 8 \
    --stride 4 --use_cache --limit 32 --out_dir "$CACHED_ROOT" \
    > logs/idlm_cache_17b_cached_${SLURM_JOB_ID}.log 2>&1 &
CACHED_PID=$!

CUDA_VISIBLE_DEVICES=1 /usr/bin/time -f 'uncached_wall_seconds=%e' \
    -o logs/idlm_cache_17b_uncached_time_${SLURM_JOB_ID}.log \
    python -u pure_inference/test_idlm_transfer.py \
    --model "$MODEL" --dataset GSM8K --max_token 512 --batch_size 8 \
    --stride 4 --no-use_cache --limit 32 --out_dir "$UNCACHED_ROOT" \
    > logs/idlm_cache_17b_uncached_${SLURM_JOB_ID}.log 2>&1 &
UNCACHED_PID=$!

FAILED=0
wait "$CACHED_PID" || FAILED=1
wait "$UNCACHED_PID" || FAILED=1
if [[ "$FAILED" -ne 0 ]]; then
    echo "Cached or uncached 1.7B generation failed; see per-path logs." >&2
    exit 1
fi

python -u pure_inference/check_idlm_cache_parity.py \
    "$CACHED_ROOT/$RESULT_SUFFIX" \
    "$UNCACHED_ROOT/$RESULT_SUFFIX" \
    --allow_drift --max_accuracy_gap 0.10

cat logs/idlm_cache_17b_cached_time_${SLURM_JOB_ID}.log
cat logs/idlm_cache_17b_uncached_time_${SLURM_JOB_ID}.log
