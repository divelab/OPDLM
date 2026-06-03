import os
import re
import sys
import json
import math
import shutil
import random
import time
from datetime import datetime
from termcolor import cprint
import wandb
import torch
import torch.distributed as dist

from omegaconf import DictConfig, ListConfig, OmegaConf

# Make eval_utils importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# Make train/ and reward/ importable
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'train'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'reward'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'sample'))

from eval_utils import DATASET_CONFIGS as _EVAL_DATASET_CONFIGS


def _eval_dataset_is_code(ds_name):
    cfg = _EVAL_DATASET_CONFIGS.get(ds_name, {})
    return cfg.get("domain") == "code"


def get_config():
    cli_conf = OmegaConf.from_cli()
    yaml_conf = OmegaConf.load(cli_conf.config)
    conf = OmegaConf.merge(yaml_conf, cli_conf)
    return conf


if __name__ == "__main__":
    config = get_config()

    # ── Distributed info (from accelerate launch) ──
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    is_main = (rank == 0)

    start_from_scratch = config.experiment.start_from_scratch
    _config_arg = next((a for a in sys.argv[1:] if a.startswith("config=")), None)
    config_name = _config_arg.split("=", 1)[1].replace("configs/", "").replace(".yaml", "") if _config_arg else config.experiment.project
    wandb_project = config.wandb.get("project") or config_name
    base_run_name = config.wandb.get("run_name") or config_name
    # Pull timestamp from env (set by the launch script) so all ranks agree.
    # Fallback to per-rank now() only if the env var isn't set — this can produce
    # divergent project dirs if ranks initialize >1 second apart (seen on dive9).
    timestamp = os.environ.get("RUN_TIMESTAMP") or datetime.now().strftime("%m%d_%H%M%S")
    if not start_from_scratch and config.experiment.get("project", None):
        # Resume: use existing experiment directory
        project_name = config.experiment.project
        run_name = os.path.basename(project_name)
    else:
        run_name = f"{base_run_name}_{timestamp}"
        _exp_base = os.environ.get("EXP_BASE", "experiments")
        project_name = f"{_exp_base}/{run_name}"           # unique output dir per run
    print(f"Config: {config_name} | Project dir: {project_name}")
    wandb_group = config.wandb.get("group", None)
    model_base = config.model.model_base

    # Set project_name in config so downstream code can find it
    config.experiment.project = project_name

    wandb_enabled = bool(config.wandb.get("enabled", True))
    run_id = None
    if wandb_enabled and is_main:
        run_id = wandb.util.generate_id()
        init_kwargs = dict(
            project=wandb_project,
            name=run_name,
            id=run_id,
            resume="never",
        )
        if wandb_group:
            init_kwargs["group"] = str(wandb_group)
        wandb_run = wandb.init(**init_kwargs)
        wandb.finish()

    # Propagate wandb env vars for training init
    if is_main:
        os.environ["WANDB_PROJECT"] = str(wandb_project)
        os.environ["WANDB_NAME"] = str(run_name)
        os.environ["WANDB_RESUME"] = str(config.wandb.get("resume", "allow"))
        if wandb_group:
            os.environ["WANDB_RUN_GROUP"] = str(wandb_group)
        if run_id is not None:
            os.environ["WANDB_RUN_ID"] = str(run_id)
        if not wandb_enabled:
            os.environ["WANDB_MODE"] = "disabled"

    # Propagate seed
    _global_seed = config.training.get("seed", None)
    if _global_seed is not None:
        os.environ["TRACERL_SEED"] = str(_global_seed)
        os.environ["PYTHONHASHSEED"] = str(_global_seed)

    from omegaconf import MISSING
    have_value_model = (OmegaConf.select(config, "model.value_base_model", default=MISSING) is not MISSING)

    def begin_with(file_name):
        with open(file_name, "w") as f:
            f.write("")

    if start_from_scratch and is_main:
        os.makedirs(f"{project_name}/results", exist_ok=True)
        optimized_model = project_name + "/ckpt/" + config.model.optimized_name
        begin_with(f"{project_name}/results/results-rl-" + optimized_model.replace("/", ".") + "-" + config.dataset.train_dataset + ".txt")
        _eval_ds = config.evaluation.eval_dataset
        if isinstance(_eval_ds, str):
            _eval_ds = [_eval_ds]
        elif isinstance(_eval_ds, ListConfig):
            _eval_ds = list(_eval_ds)
        for _ds in _eval_ds:
            begin_with(f"{project_name}/results/results-eval-" + optimized_model.replace("/", ".") + "-" + _ds + ".txt")

    # ── Compute total_step early (needed by LR scheduler in init_training) ──
    _num_train_epochs_early = config.dataset.get("num_data_epochs", -1)
    if _num_train_epochs_early >= 1:
        _train_data_path_early = f"data/{config.dataset.train_dataset}.json"
        with open(_train_data_path_early, 'r') as f:
            _n_samples_early = len(json.load(f))
        _chunk_size_early = config.rollout.num_task_per_step
        _steps_per_epoch_early = (_n_samples_early + _chunk_size_early - 1) // _chunk_size_early
        config.experiment.total_step = _steps_per_epoch_early * _num_train_epochs_early
        if is_main:
            print(f"Epoch mode (early): {_n_samples_early} samples, {_chunk_size_early}/step, "
                  f"{_steps_per_epoch_early} steps/epoch, {_num_train_epochs_early} epochs, "
                  f"{config.experiment.total_step} total RL steps")

    # ── Initialize persistent engines ──

    # Set CUDA device for this rank (before any CUDA operations)
    torch.cuda.set_device(local_rank)

    # 1. Initialize training engine FIRST (creates Accelerator which inits torch.distributed)
    if is_main:
        cprint("Initializing training engine...", "green")
    from rl_sdar import init_training, train_one_step
    training_state = init_training(config)
    if is_main:
        cprint("Training engine initialized.", "green")

    # 2. Initialize JetEngine (rollout) on each rank
    #    torch.distributed is now initialized by the Accelerator, so JetEngine
    #    will detect it and set _owns_process_group=False (colocate mode).
    if is_main:
        cprint("Initializing JetEngine rollout engine...", "green")
    # On resume, load JetEngine from the saved checkpoint (matches the training student).
    if config.experiment.current_epoch > 1 or not start_from_scratch:
        _ckpt_path = os.path.join(project_name, "ckpt", config.model.optimized_name)
        pretrained_model = _ckpt_path if os.path.exists(_ckpt_path) else config.model.pretrained_model
    else:
        pretrained_model = config.model.pretrained_model
    from bd3lm_rl_rollout import init_jetengine
    je_llm, je_tokenizer = init_jetengine(pretrained_model, config)
    if is_main:
        cprint("JetEngine initialized.", "green")

    # 3. Import reward function
    from rl_reward import compute_rewards

    # ── Epoch-based data iterator ──
    # When num_data_epochs >= 1: shuffle full dataset, iterate sequentially,
    # reshuffle each epoch. When num_data_epochs == -1: random sample each step (legacy).
    # NOTE: this is separate from training.num_train_epochs which controls
    # inner passes over the same rollout batch per RL step.
    _num_train_epochs = config.dataset.get("num_data_epochs", -1)
    _epoch_mode = (_num_train_epochs >= 1)
    _train_data_pool = None
    _train_data_idx = [0]
    _train_epoch = [1]
    _chunk_size = config.rollout.num_task_per_step

    if _epoch_mode:
        train_data_path = f"data/{config.dataset.train_dataset}.json"
        if is_main:
            print(f"Epoch mode: loading {train_data_path} ...")
        with open(train_data_path, 'r') as f:
            _train_data_pool = json.load(f)
        random.shuffle(_train_data_pool)
        _total_steps_per_epoch = (len(_train_data_pool) + _chunk_size - 1) // _chunk_size
        _total_rl_steps_epoch = _total_steps_per_epoch * _num_train_epochs
        # total_step already set in early block above; just reuse for logging
        if is_main:
            print(f"Epoch mode: {len(_train_data_pool)} samples, {_chunk_size}/step, {_total_steps_per_epoch} steps/epoch, {_num_train_epochs} epochs, {config.experiment.total_step} total RL steps")

    def _next_train_chunk():
        """Get next sequential chunk. Reshuffle when epoch exhausted."""
        start = _train_data_idx[0]
        end = start + _chunk_size
        if end <= len(_train_data_pool):
            chunk = _train_data_pool[start:end]
            _train_data_idx[0] = end
        else:
            # Wrap around: take remainder + start new epoch
            chunk = _train_data_pool[start:]
            _train_epoch[0] += 1
            random.shuffle(_train_data_pool)
            remainder = _chunk_size - len(chunk)
            if remainder > 0:
                chunk += _train_data_pool[:remainder]
            _train_data_idx[0] = remainder
            if is_main:
                print(f"Starting epoch {_train_epoch[0]}/{_num_train_epochs}")
        return chunk

    # Max token schedule
    _mt_sched = config.get("max_token_schedule", {})
    _mt_enabled = _mt_sched.get("enabled", False)
    _mt_start = _mt_sched.get("start", 500)
    _mt_end = _mt_sched.get("end", config.rollout.max_token)
    _mt_ramp_steps = _mt_sched.get("ramp_steps", 20)
    _mt_type = _mt_sched.get("type", "sin")

    def _get_scheduled_max_token(step_i):
        if not _mt_enabled:
            return int(config.rollout.max_token)
        progress = min(step_i / _mt_ramp_steps, 1.0)
        if _mt_type == "linear":
            frac = progress
        elif _mt_type == "sin":
            frac = math.sin(math.pi / 2 * progress)
        elif _mt_type == "cos":
            frac = 1.0 - math.cos(math.pi / 2 * progress)
        else:
            raise ValueError(f"Unknown max_token_schedule type: {_mt_type}")
        val = _mt_start + (_mt_end - _mt_start) * frac
        block_size = config.training.get("block_size", 4)
        val = int(round(val / block_size) * block_size)
        return max(val, block_size)

    # Dynamic-threshold schedule (hold → ramp → clamp)
    _dt_sched = config.get("dynamic_threshold_schedule", {})
    _dt_enabled = _dt_sched.get("enabled", False)
    _dt_start = _dt_sched.get("start", config.rollout.dynamic_threshold)
    _dt_end = _dt_sched.get("end", config.rollout.dynamic_threshold)
    _dt_hold = _dt_sched.get("hold_steps", 0)
    _dt_ramp_raw = _dt_sched.get("ramp_steps", -1)
    _dt_type = _dt_sched.get("type", "linear")

    # ramp_steps == -1 (or missing) → auto: ramp from hold_steps to end of training.
    # Requires experiment.total_step to be set (i.e. epoch mode / explicit total_step).
    if _dt_enabled and (_dt_ramp_raw is None or int(_dt_ramp_raw) <= 0):
        _total_step = int(OmegaConf.select(config, "experiment.total_step", default=-1) or -1)
        if _total_step > _dt_hold:
            _dt_ramp = _total_step - _dt_hold
            if is_main:
                print(f"  dynamic_threshold_schedule: ramp_steps=auto -> {_dt_ramp} "
                      f"(total_step={_total_step}, hold_steps={_dt_hold})")
        else:
            raise ValueError(
                "dynamic_threshold_schedule.ramp_steps=auto requires experiment.total_step "
                f"to be known and > hold_steps ({_dt_hold}); got total_step={_total_step}. "
                "Set ramp_steps explicitly or use dataset.num_data_epochs / experiment.total_step."
            )
    else:
        _dt_ramp = max(int(_dt_ramp_raw), 1)

    def _get_scheduled_dynamic_threshold(step_i):
        if not _dt_enabled:
            return float(config.rollout.dynamic_threshold)
        if step_i <= _dt_hold:
            return float(_dt_start)
        progress = min((step_i - _dt_hold) / _dt_ramp, 1.0)
        if _dt_type == "linear":
            frac = progress
        elif _dt_type == "sin":
            frac = math.sin(math.pi / 2 * progress)
        elif _dt_type == "cos":
            frac = 1.0 - math.cos(math.pi / 2 * progress)
        else:
            raise ValueError(f"Unknown dynamic_threshold_schedule type: {_dt_type}")
        return float(_dt_start + (_dt_end - _dt_start) * frac)

    # GPU hours tracking
    _cuda_devs = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    _num_gpus = len(_cuda_devs.split(",")) if _cuda_devs else 1
    _gpu_hours_file = os.path.join(project_name, "temp_data", "cumulative_gpu_hours.txt")
    if is_main:
        os.makedirs(os.path.dirname(_gpu_hours_file), exist_ok=True)

    def _read_gpu_hours():
        if os.path.exists(_gpu_hours_file):
            with open(_gpu_hours_file) as f:
                return float(f.read().strip())
        return 0.0

    def _write_gpu_hours(val):
        with open(_gpu_hours_file, "w") as f:
            f.write(str(val))

    if config.dataset.data_type == "code":
        is_code_task = True
    else:
        is_code_task = False

    is_process_reward = (OmegaConf.select(config, "model.process_reward_model", default=MISSING) is not MISSING
                         and config.model.process_reward_model is not None)

    # ── Rollout wrapper (uses persistent JetEngine) ──
    from bd3lm_rl_rollout import run_rollout, sleep_jetengine, reload_jetengine_weights

    def do_rollout(i, function_type, block_size=None, top_k=None, remasking_strategy=None,
                   eval_dataset=None, train_dataset=None, max_token=None, rollout_max_token=None,
                   rollout_dynamic_threshold=None, data_override=None):
        """Run rollout using persistent JetEngine. All ranks participate."""
        # Apply overrides to config
        cfg = OmegaConf.create(OmegaConf.to_container(config, resolve=True))
        cfg.experiment.current_epoch = i
        cfg.experiment.function = function_type
        cfg.experiment.project = project_name
        if eval_dataset is not None:
            cfg.evaluation.eval_dataset = eval_dataset
        if train_dataset is not None:
            cfg.dataset.train_dataset = train_dataset
        if max_token is not None:
            cfg.evaluation.max_token = max_token
        if rollout_max_token is not None:
            cfg.rollout.max_token = rollout_max_token
        if rollout_dynamic_threshold is not None:
            cfg.rollout.dynamic_threshold = rollout_dynamic_threshold
        if block_size is not None:
            cfg.evaluation.block_size = block_size
        if top_k is not None:
            cfg.evaluation.top_k = top_k
        if remasking_strategy is not None:
            cfg.evaluation.remasking_strategy = remasking_strategy

        run_rollout(je_llm, je_tokenizer, cfg, project_name, rank, world_size, data_override=data_override)

    # ── Reward wrapper (rank 0 only) ──
    def do_reward(i, function_type, is_code_task_flag, block_size=None, top_k=None,
                  remasking_strategy=None, eval_dataset=None, train_dataset=None,
                  rollout_dynamic_threshold=None):
        """Compute rewards. Runs on rank 0 only, other ranks wait."""
        if is_main:
            cfg = OmegaConf.create(OmegaConf.to_container(config, resolve=True))
            cfg.experiment.current_epoch = i
            cfg.experiment.function = function_type
            cfg.experiment.project = project_name
            if eval_dataset is not None:
                cfg.evaluation.eval_dataset = eval_dataset
            if train_dataset is not None:
                cfg.dataset.train_dataset = train_dataset
            if block_size is not None:
                cfg.evaluation.block_size = block_size
            if top_k is not None:
                cfg.evaluation.top_k = top_k
            if remasking_strategy is not None:
                cfg.evaluation.remasking_strategy = remasking_strategy
            if rollout_dynamic_threshold is not None:
                cfg.rollout.dynamic_threshold = rollout_dynamic_threshold

            # Get active wandb run if available
            active_wandb = wandb.run if wandb_enabled else None
            compute_rewards(cfg, project_name, wandb_run=active_wandb)
        dist.barrier()

    # ── Training wrapper (all ranks via DeepSpeed) ──
    def do_train(i, max_gen_length=None):
        """Train one step using persistent training state."""
        # Free all JetEngine GPU memory (params + KV cache + graphs) for training
        sleep_jetengine(je_llm)

        cfg = OmegaConf.create(OmegaConf.to_container(config, resolve=True))
        cfg.experiment.current_epoch = i
        cfg.experiment.project = project_name
        if max_gen_length is not None:
            cfg.training.max_gen_length = max_gen_length

        train_one_step(training_state, cfg)

        # Barrier: ensure rank 0 has finished writing the checkpoint before
        # other ranks try to load it for JetEngine.
        dist.barrier()

        # Reload JetEngine weights from the new checkpoint
        checkpoint_path = os.path.join(project_name, "ckpt", config.model.optimized_name)
        reload_jetengine_weights(je_llm, checkpoint_path)

    # ── Train loop ──
    def train_loop(train_i):
        if is_main:
            print(f"Starting training for step {train_i}...")
        _step_start = time.time()
        if _global_seed is not None:
            os.environ["TRACERL_SEED"] = str(_global_seed + train_i)

        _scheduled_max_token = _get_scheduled_max_token(train_i)
        _prev_max_token = _get_scheduled_max_token(train_i - 1) if train_i > 1 else None
        if is_main and _mt_enabled and _scheduled_max_token != _prev_max_token:
            print(f"  max_token={_scheduled_max_token} (step {train_i}, schedule={_mt_type})")

        _scheduled_dt = _get_scheduled_dynamic_threshold(train_i)
        _prev_dt = _get_scheduled_dynamic_threshold(train_i - 1) if train_i > 1 else None
        if is_main and _dt_enabled and _scheduled_dt != _prev_dt:
            print(f"  dynamic_threshold={_scheduled_dt:.4f} (step {train_i}, schedule={_dt_type})")

        rl_data_dest = f"{project_name}/temp_data/{config.dataset.optimization_data}.json"

        # Epoch mode: pass sequential chunk; legacy mode: rollout loads & samples
        train_chunk = _next_train_chunk() if _epoch_mode else None
        do_rollout(train_i, "train", rollout_max_token=_scheduled_max_token,
                   rollout_dynamic_threshold=_scheduled_dt, data_override=train_chunk)
        do_reward(train_i, "train", is_code_task, rollout_dynamic_threshold=_scheduled_dt)
        dist.barrier()

        do_train(train_i, max_gen_length=_scheduled_max_token)

        if is_main:
            _step_hours = (time.time() - _step_start) / 3600.0
            _total = _read_gpu_hours() + _num_gpus * _step_hours
            _write_gpu_hours(_total)
            print(f"Finished training for step {train_i}. Step GPU hours: {_num_gpus * _step_hours:.2f}, Cumulative: {_total:.2f}")

    # ── Eval loop ──
    def eval_loop(eval_i):
        if is_main:
            print(f"Starting evaluation for step {eval_i}...")

        eval_datasets = config.evaluation.eval_dataset
        if isinstance(eval_datasets, str):
            eval_datasets = [eval_datasets]
        elif isinstance(eval_datasets, ListConfig):
            eval_datasets = list(eval_datasets)

        eval_max_tokens = config.evaluation.max_token
        if not isinstance(eval_max_tokens, (list, ListConfig)):
            eval_max_tokens = [eval_max_tokens] * len(eval_datasets)
        else:
            eval_max_tokens = list(eval_max_tokens)

        for ds_idx, ds_name in enumerate(eval_datasets):
            ds_max_token = eval_max_tokens[ds_idx]
            ds_is_code = _eval_dataset_is_code(ds_name) or is_code_task
            if is_main:
                print(f"  Evaluating on {ds_name}... (is_code={ds_is_code})")

            if model_base in ("sdar", "qwen", "bd3lm"):
                remasking_strategy_list = config.evaluation.remasking_strategy
                top_k_list = config.evaluation.top_k
                block_size = config.evaluation.block_size
                for j in range(len(remasking_strategy_list)):
                    remasking_strategy = remasking_strategy_list[j]
                    top_k = top_k_list[j]
                    do_rollout(eval_i, "evaluation", block_size=block_size, top_k=top_k,
                               remasking_strategy=remasking_strategy, eval_dataset=ds_name, max_token=ds_max_token)
                    do_reward(eval_i, "evaluation", ds_is_code, block_size=block_size, top_k=top_k,
                              remasking_strategy=remasking_strategy, eval_dataset=ds_name)
            else:
                # Generic eval path
                do_rollout(eval_i, "evaluation", eval_dataset=ds_name, max_token=ds_max_token)
                do_reward(eval_i, "evaluation", ds_is_code, eval_dataset=ds_name)

        if is_main:
            print(f"Finished evaluation for step {eval_i}.")

    def _parse_best_eval_acc(eval_i):
        eval_datasets = config.evaluation.eval_dataset
        if isinstance(eval_datasets, str):
            eval_datasets = [eval_datasets]
        elif isinstance(eval_datasets, ListConfig):
            eval_datasets = list(eval_datasets)
        optimized_model = project_name + "/ckpt/" + config.model.optimized_name
        accs = []
        for ds in eval_datasets:
            results_file = f"{project_name}/results/results-eval-" + optimized_model.replace("/", ".") + f"-{ds}.txt"
            if not os.path.exists(results_file):
                continue
            best_acc_ds = None
            with open(results_file, 'r') as f:
                for line in f:
                    m = re.search(r'train step:\s*' + str(eval_i) + r'\b.*acc:\s*([0-9.]+)', line)
                    if m:
                        acc = float(m.group(1))
                        if best_acc_ds is None or acc > best_acc_ds:
                            best_acc_ds = acc
            if best_acc_ds is not None:
                accs.append(best_acc_ds)
        return sum(accs) / len(accs) if accs else None

    def _save_best_checkpoint(eval_i, acc):
        src = os.path.join(project_name, "ckpt", config.model.optimized_name)
        dst = os.path.join(project_name, "ckpt", "best")
        if os.path.exists(dst):
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
        meta = {"step": eval_i, "accuracy": acc}
        with open(os.path.join(dst, "best_meta.json"), "w") as f:
            json.dump(meta, f, indent=2)
        cprint(f"New best checkpoint saved: step {eval_i}, acc {acc:.4f} -> {dst}", "green")

    _best_eval_acc = 0.0

    # ── Main loop ──
    i = config.experiment.current_epoch

    if config.evaluation.get("run_before_training", True):
        eval_loop(i)

    def _should_stop(step_i):
        """Check termination: stop_RL_step (manual cap), total_step, or epoch limit."""
        # Manual early-stop knob — preserves LR schedule (which uses total_step) but
        # exits the RL loop early. Default -1 means unused.
        _stop_rl = config.experiment.get("stop_RL_step", -1)
        if _stop_rl not in (-1, None) and step_i > _stop_rl:
            return True
        if config.experiment.total_step not in (-1, None) and step_i > config.experiment.total_step:
            return True
        if _epoch_mode and _train_epoch[0] > _num_train_epochs:
            return True
        return False

    from tqdm.auto import tqdm
    _epoch_pbar = None
    if _epoch_mode and is_main:
        _epoch_pbar = tqdm(total=_total_steps_per_epoch, desc=f"Data epoch {_train_epoch[0]}/{_num_train_epochs}", dynamic_ncols=True)
    _prev_epoch = _train_epoch[0]

    while not _should_stop(i):
        train_loop(i)

        if _epoch_pbar is not None:
            _epoch_pbar.update(1)
            if _train_epoch[0] != _prev_epoch:
                _epoch_pbar.close()
                _prev_epoch = _train_epoch[0]
                _epoch_pbar = tqdm(total=_total_steps_per_epoch, desc=f"Data epoch {_train_epoch[0]}/{_num_train_epochs}", dynamic_ncols=True)

        if i % config.experiment.eval_every == 0:
            eval_loop(i)
            # if is_main:
            #     eval_acc = _parse_best_eval_acc(i)
            #     if eval_acc is not None and eval_acc > _best_eval_acc:
            #         _best_eval_acc = eval_acc
            #         _save_best_checkpoint(i, eval_acc)

        i += 1

    if _epoch_pbar is not None:
        _epoch_pbar.close()
