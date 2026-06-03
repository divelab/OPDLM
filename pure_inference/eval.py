"""
Pure inference evaluation — no experiment folder, no training state.
Supports BD3LM (diffusion) and Qwen (autoregressive) models.

Static remasking (argmax), top_k=1 for BD3LM; greedy decoding for Qwen.

Usage:
    python pure_inference/eval_bd3lm.py \
        --models pretrained_models/BD3LM/Qwen3-0.6B-a2d-dapo_sft_1epoch \
                 pretrained_models/Qwen/Qwen3-0.6B \
        --model_bases bd3lm qwen \
        --datasets MATH500 GSM8K \
        --max_token 2048 \
        --out_dir pure_inference/results
"""
import argparse
import os
import subprocess
import sys

from omegaconf import OmegaConf

# Make eval_utils importable so we can resolve per-dataset domain. Mirrors
# rl.py's _eval_dataset_is_code — code datasets (HumanEval/MBPP/LCB/...) need
# to go through rl_execute.py + rl_code_reward.py instead of rl_reward.py.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO_ROOT)
from eval_utils import DATASET_CONFIGS as _EVAL_DATASET_CONFIGS


def _eval_dataset_is_code(ds_name):
    cfg = _EVAL_DATASET_CONFIGS.get(ds_name, {})
    return cfg.get("domain") == "code"



def main():
    parser = argparse.ArgumentParser(description="Pure inference eval for BD3LM / Qwen models")
    parser.add_argument("--models", nargs="+", required=True, help="One or more model paths")
    parser.add_argument("--model_bases", nargs="+", required=True, help="Model base per path: 'bd3lm' or 'qwen' (one per model)")
    parser.add_argument("--datasets", nargs="+", required=True, help="One or more eval datasets (e.g. MATH500 GSM8K)")
    parser.add_argument("--max_token", type=int, default=None,
                        help="If set, applies globally to all datasets (overrides per-dataset).")
    parser.add_argument("--dataset_max_tokens", nargs="+", type=int, default=None,
                        help="Per-dataset max_tokens (same length as --datasets). "
                             "If omitted, falls back to defaults (GSM8K=1000, MATH500/AIME24=2000, else=2048).")
    parser.add_argument("--block_size", type=int, default=4)
    parser.add_argument("--denoising_steps_per_block", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="0.0 = argmax (default). Set >0 for stochastic sampling.")
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=1)
    parser.add_argument("--min_p", type=float, default=0.0)
    parser.add_argument("--num_response_per_task", type=int, default=3,
                        help="Global default. Overridden by --dataset_num_responses if set.")
    parser.add_argument("--dataset_num_responses", nargs="+", type=int, default=None,
                        help="Per-dataset num_response_per_task (same length as --datasets).")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.8)
    parser.add_argument("--max_active", type=int, default=128)
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--base_port", type=int, default=20500)
    parser.add_argument("--out_dir", type=str, default="pure_inference/results")
    parser.add_argument("--tag", type=str, default="",
                        help="Optional suffix appended to per-(model,dataset) output folder name.")
    parser.add_argument("--remasking_strategy", type=str, default="low_confidence_static",
                        choices=["low_confidence_static", "low_confidence_dynamic"],
                        help="BD3LM remasking strategy (default: low_confidence_static).")
    parser.add_argument("--dynamic_threshold", type=float, default=0.9,
                        help="Confidence threshold for low_confidence_dynamic (ignored for static). Default: 0.9.")
    parser.add_argument("--enable_thinking", action="store_true", default=False,
                        help="For Qwen3 models: enable <think> reasoning block (default: False).")
    parser.add_argument("--scorer", type=str, default="math_verify",
                        choices=["math_verify", "opencompass"],
                        help=("Math equality checker. 'math_verify' (default) uses "
                              "the math_verify library; 'opencompass' uses OpenCompass's "
                              "MATHEvaluator + math_postprocess_v2 (mirrors SDAR HF eval). "
                              "Affects math domain only; mc/code/ifeval ignore this."))
    parser.add_argument("--num_chunks", type=int, default=1,
                        help=("Split each (model, dataset) rollout into N contiguous chunks; "
                              "save per-chunk JSON before starting the next chunk. On rerun, "
                              "completed chunks are skipped. After all chunks finish, eval.py "
                              "merges them into the canonical outputs JSON before reward. "
                              "Use 2-8 for long thinking-mode runs to avoid losing all work "
                              "if a JetEngine worker dies on a tail prompt. Default: 1 (no chunking)."))
    args = parser.parse_args()

    assert len(args.models) == len(args.model_bases), "--models and --model_bases must have same length"
    for mb in args.model_bases:
        assert mb in ("bd3lm", "sdar", "qwen"), f"Unsupported model_base: {mb} (must be 'bd3lm', 'sdar', or 'qwen')"

    # Per-dataset max_token resolution: --max_token (global) > --dataset_max_tokens > defaults
    DATASET_MAX_TOKEN_DEFAULT = {
        "GSM8K": 1000,
        "MATH500": 2000,
        "AIME2024": 2000,
    }
    if args.dataset_max_tokens is not None:
        assert len(args.dataset_max_tokens) == len(args.datasets), \
            "--dataset_max_tokens and --datasets must have same length"
        _per_dataset = dict(zip(args.datasets, args.dataset_max_tokens))
    else:
        _per_dataset = {}
    def _get_max_token(dataset):
        if args.max_token is not None:
            return args.max_token
        if dataset in _per_dataset:
            return _per_dataset[dataset]
        return DATASET_MAX_TOKEN_DEFAULT.get(dataset, 2048)

    if args.dataset_num_responses is not None:
        assert len(args.dataset_num_responses) == len(args.datasets), \
            "--dataset_num_responses and --datasets must have same length"
        _per_dataset_nr = dict(zip(args.datasets, args.dataset_num_responses))
    else:
        _per_dataset_nr = {}
    def _get_num_response(dataset):
        return _per_dataset_nr.get(dataset, args.num_response_per_task)

    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.makedirs(os.path.join(root_dir, args.out_dir), exist_ok=True)

    for model_path, model_base in zip(args.models, args.model_bases):
        model_abspath = os.path.abspath(model_path)
        model_name = os.path.basename(model_abspath.rstrip("/"))

        for dataset in args.datasets:
            max_token = _get_max_token(dataset)
            num_response = _get_num_response(dataset)
            # Per-dataset code/non-code routing — mirrors rl.py eval_loop so
            # code datasets go through rl_execute.py + rl_code_reward.py
            # (and HumanEval/MBPP route through the evalplus deferred scorer
            # inside rl_execute.py).
            is_code_task = _eval_dataset_is_code(dataset)
            data_type = "code" if is_code_task else "math"
            print(f"\n{'='*70}")
            print(f"  Model: {model_name} ({model_base})")
            print(f"  Dataset: {dataset}  (max_token={max_token}, is_code={is_code_task})")
            if model_base in ("bd3lm", "sdar"):
                _strat_desc = args.remasking_strategy
                if args.remasking_strategy == "low_confidence_dynamic":
                    _strat_desc += f" (threshold={args.dynamic_threshold})"
                print(f"  Strategy: {_strat_desc}, top_k=1")
            else:
                print(f"  Strategy: greedy (top_k=1, temperature=0)")
            print(f"{'='*70}")

            # project_name points to a per-(model, dataset) workdir under out_dir
            _suffix = f"_{args.tag}" if args.tag else ""
            project_name = os.path.join(args.out_dir, f"{model_name}_{dataset}{_suffix}")
            project_abs = os.path.join(root_dir, project_name)
            os.makedirs(os.path.join(project_abs, "results"), exist_ok=True)
            os.makedirs(os.path.join(project_abs, "temp_data"), exist_ok=True)

            # Per-(model, dataset) config file to avoid overwrites when running in parallel
            config_path = os.path.join(project_abs, "_tmp_eval_config.yaml")

            cfg = {
                "wandb": {"enabled": False, "project": "pure_inference", "group": None, "run_name": "eval"},
                "experiment": {
                    "project": project_name,
                    "port": args.base_port,
                    "function": "evaluation",
                    "start_from_scratch": True,
                    "total_step": 1,
                    "save_every": 999,
                    "eval_every": 1,
                    "current_epoch": 1,  # current_epoch=1 → load model.pretrained_model directly
                    "deepspeed_file": "1gpu_debug",
                    "num_node": 1,
                    "node_index": 0,
                },
                "model": {
                    "pretrained_model": model_abspath,
                    "teacher_model": model_abspath,  # unused in eval, but must be set
                    "optimized_name": "optimized",
                    "model_base": model_base,
                },
                "dataset": {
                    "train_dataset": dataset,
                    "optimization_data": "rl_data",
                    "data_type": data_type,
                },
                "rollout": {
                    "tensor_parallel_size": args.tensor_parallel_size,
                    "gpu_memory_utilization": args.gpu_memory_utilization,
                    "max_active": args.max_active,
                    "num_task_per_step": 1,
                    "num_response_per_task": num_response,
                    "temperature": args.temperature,
                    "max_token": max_token,
                    "block_size": args.block_size,
                    "denoising_steps_per_block": args.denoising_steps_per_block,
                    "top_p": args.top_p,
                    "top_k": args.top_k,
                    "min_p": args.min_p,
                    "remasking_strategy": args.remasking_strategy,
                    "dynamic_threshold": args.dynamic_threshold,
                    # Single thinking knob: --enable_thinking sets start_with_think.
                    "start_with_think": args.enable_thinking,
                    "base_port": args.base_port,
                },
                "execute": {"num_chunk": 128},
                "training": {
                    "gradient_checkpointing_enable": True,
                    "gradient_accumulation_steps": 1,
                    "batch_size_lm": 1,
                    "mixed_precision": "bf16",
                    "enable_tf32": True,
                    "seed": 10086,
                    "num_train_epochs": 1,
                    "max_grad_norm": 1.0,
                    "method": "TraceRL",
                    "block_size": args.block_size,
                    "shrink": 1,
                    "post_num": 0,
                    "max_gen_length": max_token,
                    "max_prompt_len": 784,
                    "lower_p": 0.1,
                    "upper_p": 0.9,
                    "eps": 0.20,
                    "beta": 0.01,
                    "use_kl_estimator_k3": True,
                    "teacher_fill": "argmax_fill",
                    "exclude_im_end": False,
                    "loss_type": "kl",
                    "student_arm_shift": False,
                },
                "optimizer": {
                    "name": "adamw",
                    "params": {
                        "learning_rate": 1e-6, "scale_lr": False, "beta1": 0.9, "beta2": 0.999,
                        "weight_decay": 0.0, "epsilon": 1e-8,
                    },
                },
                "lr_scheduler": {
                    "scheduler": "cosine",
                    "params": {"learning_rate": 1e-6, "warmup_steps": 0, "min_lr_scale": 1.0},
                },
                "evaluation": {
                    "eval_dataset": dataset,
                    "data_type": data_type,
                    "tensor_parallel_size": args.tensor_parallel_size,
                    "gpu_memory_utilization": args.gpu_memory_utilization,
                    "max_active": args.max_active,
                    "num_response_per_task": num_response,
                    "temperature": args.temperature,
                    "max_token": max_token,
                    "block_size": args.block_size,
                    "denoising_steps_per_block": args.denoising_steps_per_block,
                    "top_p": args.top_p,
                    "top_k": args.top_k,
                    "min_p": args.min_p,
                    "remasking_strategy": args.remasking_strategy,
                    "dynamic_threshold": args.dynamic_threshold,
                    # Single thinking knob: --enable_thinking sets start_with_think.
                    "start_with_think": args.enable_thinking,
                    "run_before_training": True,
                    # Math scorer: 'math_verify' (default) or 'opencompass'.
                    # Read by reward/rl_reward.py to dispatch math equivalence check.
                    "scorer": args.scorer,
                },
            }
            cfg["config"] = os.path.relpath(config_path, os.path.join(root_dir, "sample"))

            # ── Standard generative path ────────────────────────
            # Run rollout (sample) — dispatch on model_base
            rollout_script = {
                "bd3lm": "bd3lm_rl_rollout.py",
                "sdar":  "sdar_rl_rollout.py",
                "qwen":  "qwen_rl_rollout.py",
            }.get(model_base, "qwen_rl_rollout.py")

            # Chunked rollout: split into N contiguous chunks. Each chunk's
            # rollout writes outputs-...-{dataset}_chunk{k}of{N}.json. After
            # all chunks finish we concatenate into the canonical
            # outputs-...-{dataset}.json and proceed to execute/reward as
            # usual. Reruns skip chunks whose JSON already exists.
            num_chunks = max(1, int(args.num_chunks))

            # Env-driven single-chunk mode: if NUM_CHUNKS/CHUNK_INDEX are set
            # in the parent environment, the rollout itself slices to that
            # chunk and writes the matching `_chunk{K}of{N}.json` file.
            # In that case we MUST NOT also drive eval.py's own chunk loop
            # (would cause double-chunking) and we MUST NOT merge/reward
            # (other chunks are produced by parallel invocations elsewhere).
            #
            # Workflow:
            #   # parallel single-chunk producers (env-driven, no merge):
            #   NUM_CHUNKS=8 CHUNK_INDEX=0 CUDA_VISIBLE_DEVICES=0 bash run_eval...sh
            #   ...
            #   NUM_CHUNKS=8 CHUNK_INDEX=7 CUDA_VISIBLE_DEVICES=7 bash run_eval...sh
            #   # then a single merge+reward call (no env vars):
            #   bash run_eval...sh           # picks --num_chunks 8, all chunks exist, merges, runs reward
            _env_num = os.environ.get("NUM_CHUNKS", "").strip()
            _env_idx = os.environ.get("CHUNK_INDEX", "").strip()
            _env_chunk_mode = bool(_env_num and _env_idx)

            # Compute the canonical output stem (mirrors rollout's naming).
            _outputs_stem = ("eval-" + os.path.abspath(model_path).replace("/", ".") + "-" + dataset)
            _canonical_out = os.path.join(project_abs, "temp_data", f"outputs-{_outputs_stem}.json")
            _chunk_outs = [os.path.join(project_abs, "temp_data",
                                        f"outputs-{_outputs_stem}_chunk{k}of{num_chunks}.json")
                           for k in range(num_chunks)]

            if _env_chunk_mode:
                # Single-chunk producer mode driven by env. Run rollout once,
                # let env-driven shard inside the rollout do the slicing,
                # skip merge/execute/reward (they happen in the final
                # no-env "merge & reward" pass once all chunks exist).
                print(f"\n--- Sampling ({rollout_script}) [env-shard NUM_CHUNKS={_env_num} CHUNK_INDEX={_env_idx}] ---")
                # Pass cfg WITHOUT yaml chunk_index/num_chunks so eval.py's
                # yaml-driven chunking branch in the rollout stays inactive.
                OmegaConf.save(OmegaConf.create(cfg), config_path)
                sample_cmd = (
                    f"python {rollout_script} "
                    f"config={os.path.relpath(config_path, os.path.join(root_dir, 'sample'))} "
                )
                subprocess.run(sample_cmd, shell=True, cwd=os.path.join(root_dir, "sample"), check=True)
                _expected = os.path.join(project_abs, "temp_data",
                                         f"outputs-{_outputs_stem}_chunk{_env_idx}of{_env_num}.json")
                if os.path.exists(_expected):
                    print(f"[env-chunk] wrote {_expected}")
                else:
                    print(f"[env-chunk] WARNING: expected {_expected} not found (rollout may have failed or env vars not honored)")
                # Cleanup tmp config and SKIP merge/execute/reward.
                if os.path.exists(config_path):
                    os.remove(config_path)
                continue

            print(f"\n--- Sampling ({rollout_script}) ---")
            for _k in range(num_chunks):
                if num_chunks > 1 and os.path.exists(_chunk_outs[_k]):
                    print(f"[chunk {_k+1}/{num_chunks}] already done -> {_chunk_outs[_k]}; skipping")
                    continue
                # Re-save config per chunk so chunk_index/num_chunks is set
                # only when num_chunks > 1 (no surprise to single-chunk runs).
                cfg_this = dict(cfg)
                cfg_this["dataset"] = dict(cfg_this.get("dataset", {}))
                if num_chunks > 1:
                    cfg_this["dataset"]["chunk_index"] = _k
                    cfg_this["dataset"]["num_chunks"] = num_chunks
                OmegaConf.save(OmegaConf.create(cfg_this), config_path)
                if num_chunks > 1:
                    print(f"[chunk {_k+1}/{num_chunks}] running...")
                sample_cmd = (
                    f"python {rollout_script} "
                    f"config={os.path.relpath(config_path, os.path.join(root_dir, 'sample'))} "
                )
                subprocess.run(sample_cmd, shell=True, cwd=os.path.join(root_dir, "sample"), check=True)

            # Merge per-chunk outputs into the canonical filename (only when
            # chunked). Each chunk JSON is the standard list-of-dicts that
            # rl_reward.py expects; concatenation in order preserves data
            # ordering since chunks are contiguous slices.
            if num_chunks > 1:
                import json as _json
                merged = []
                for p in _chunk_outs:
                    if not os.path.exists(p):
                        raise RuntimeError(f"missing chunk output: {p}")
                    with open(p) as fh:
                        merged.extend(_json.load(fh))
                os.makedirs(os.path.dirname(_canonical_out), exist_ok=True)
                with open(_canonical_out, "w", encoding="utf-8") as fh:
                    _json.dump(merged, fh, indent=2, ensure_ascii=False)
                print(f"[merge] wrote {len(merged)} prompts to {_canonical_out}")

            # Re-save cfg WITHOUT chunk keys before execute/reward so downstream
            # tools always see a clean config (and so the file is in a clean
            # state if you cat/inspect it after the run).
            OmegaConf.save(OmegaConf.create(cfg), config_path)

            # For code datasets, run rl_execute.py to populate `correctness`
            if is_code_task:
                print(f"\n--- Execute (rl_execute.py) ---")
                execute_cmd = (
                    f"python rl_execute.py "
                    f"config={os.path.relpath(config_path, os.path.join(root_dir, 'reward'))} "
                )
                subprocess.run(execute_cmd, shell=True, cwd=os.path.join(root_dir, "reward"), check=True)

            # Run reward — rl_code_reward.py for code; rl_reward.py otherwise.
            reward_script = "rl_code_reward.py" if is_code_task else "rl_reward.py"
            print(f"\n--- Reward ({reward_script}) ---")
            reward_cmd = (
                f"python {reward_script} "
                f"config={os.path.relpath(config_path, os.path.join(root_dir, 'reward'))} "
            )
            subprocess.run(reward_cmd, shell=True, cwd=os.path.join(root_dir, "reward"), check=True)

            # Print the result file
            results_dir = os.path.join(project_abs, "results")
            filter_str = args.remasking_strategy if model_base in ("bd3lm", "sdar") else ""
            for f in sorted(os.listdir(results_dir)):
                if dataset in f and "eval" in f:
                    fpath = os.path.join(results_dir, f)
                    print(f"\n--- {f} ---")
                    with open(fpath) as fh:
                        for line in fh:
                            if filter_str in line:
                                print(line.strip())

            # Cleanup tmp config
            if os.path.exists(config_path):
                os.remove(config_path)

    print(f"\n{'='*70}")
    print(f"  All done. Results under: {args.out_dir}/")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
