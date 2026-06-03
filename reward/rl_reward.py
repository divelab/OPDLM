import json
import os
import sys
import math_utils
import nest_asyncio
from scipy.stats import norm
from concurrent.futures import ThreadPoolExecutor
import asyncio
from termcolor import cprint
from omegaconf import MISSING
from omegaconf import DictConfig, ListConfig, OmegaConf
import wandb

# Import eval_utils for dataset-specific correctness checks
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from eval_utils import DATASET_CONFIGS

# Domain-based extract/check dispatch (math / mc / code).
from domain_reward import check_answer, get_domain, _classify_tests


def _is_scorable(sample, ds_cfg):
    """Is this sample's correctness a meaningful learning signal?

    Unscorable cases (excluded from reported train/eval acc, but still
    kept in the rollout set — KL distillation still trains on them):

      - chat domain (no verifiable answer — _noop_check always returns False)
      - code with malformed tests_json (harness will always crash regardless
        of model output). Currently the only detected case is taco's fn_call
        samples that shipped raw problem-statement strings as `args`
        (~171/2477 taco_fn samples); see debug_general_construct_data_v2/.
    """
    dom = get_domain(sample, ds_cfg)
    if dom == "chat":
        return False
    if dom == "code":
        fmt, parsed = _classify_tests(sample.get("tests_json"))
        if fmt == "unknown":
            return False
        if fmt == "fn_call" and isinstance(parsed, dict):
            cases = parsed.get("cases") or []
            if not cases or not all(isinstance(c.get("args"), list) for c in cases):
                return False
    return True


def _check_code_worker(args):
    """Module-level worker for multiprocessing pool (must be picklable)."""
    k, extracted, sample, ds_cfg, scorer = args
    return k, check_answer(extracted, sample, ds_cfg=ds_cfg, scorer=scorer)


def _z_score_normalize(lst):
    mean = sum(lst) / len(lst)
    std = (sum((x - mean) ** 2 for x in lst) / len(lst)) ** 0.5
    if std == 0:
        return [0 for x in lst]
    return [(x - mean) / std for x in lst]


def compute_rewards(config, project_name, wandb_run=None):
    """Compute rewards for rollout outputs. Importable from rl.py.

    Args:
        config: OmegaConf config (must have experiment.current_epoch, experiment.function set)
        project_name: path to experiment directory (e.g. "experiments/run_name")
        wandb_run: optional active wandb run for logging (None = skip wandb logging)
    """
    if config.experiment.current_epoch == 1:
        pretrained_model = config.model.pretrained_model
    else:
        pretrained_model = project_name + "/ckpt/" + config.model.optimized_name

    if config.experiment.function == "train":
        dataset = config.dataset.train_dataset
        outputs_name = "rl-" + pretrained_model.replace("/", ".") + "-" + dataset
    elif config.experiment.function == "evaluation":
        dataset = config.evaluation.eval_dataset
        outputs_name = "eval-" + pretrained_model.replace("/", ".") + "-" + dataset

    file_name = project_name + "/temp_data/outputs-" + outputs_name + ".json"

    with open(file_name, 'r') as f:
        data = json.load(f)

    index_list = []
    extracted_output_list = []
    sample_data_list = []    # per-rollout view of the source sample (for domain + gt + tests_json)
    response_length_list = []
    for i in range(len(data)):
        data[i]["correctness"] = []
        response_length_list = response_length_list + data[i]["response_length"]
        index_list = index_list + [i] * len(data[i]["extracted_output"])
        extracted_output_list = extracted_output_list + data[i]["extracted_output"]
        sample_data_list = sample_data_list + [data[i]] * len(data[i]["extracted_output"])

    nest_asyncio.apply()

    # Skip correctness check if configured (e.g. general SFT data has no verifiable answers)
    skip_correctness = config.dataset.get("skip_correctness", False)
    skip_code_correctness = config.dataset.get("skip_code_correctness", False)
    # Math scorer: "math_verify" (default, pure_inference legacy) or
    # "opencompass" (mirrors SDAR HF eval's MATHEvaluator + math_postprocess_v2).
    scorer = OmegaConf.select(config, "evaluation.scorer", default="math_verify")
    if skip_correctness and config.experiment.function == "train":
        correctness_list = [False] * len(index_list)
    else:
        ds_cfg = DATASET_CONFIGS.get(dataset, None)

        code_idx_set = {
            k for k in range(len(index_list))
            if get_domain(sample_data_list[k], ds_cfg) == "code"
        }
        correctness_list = [False] * len(index_list)

        # If using a non-default scorer for math, re-extract from full_output
        # so the extractor matches the checker (rollout used math_verify by default).
        if scorer == "opencompass":
            from domain_reward import extract_answer as _extract
            for k in range(len(index_list)):
                if k in code_idx_set:
                    continue
                _full = sample_data_list[k].get("full_output", [])
                # full_output / extracted_output are parallel lists per sample;
                # the offset within the sample equals position in extracted_output
                # for that sample. Recover it from the original index_list.
                # Simpler: re-extract from the *current* extracted text isn't
                # right (already postprocessed); use full_output instead.
                _per_sample_idx = sum(1 for _i in range(k)
                                      if index_list[_i] == index_list[k])
                if _per_sample_idx < len(_full):
                    extracted_output_list[k] = _extract(
                        _full[_per_sample_idx],
                        data_i=sample_data_list[k], ds_cfg=ds_cfg, scorer=scorer,
                    )

        # Inline pass for math/mc (cheap).
        for k in range(len(index_list)):
            if k in code_idx_set:
                continue
            correctness_list[k] = check_answer(
                extracted_output_list[k], sample_data_list[k], ds_cfg=ds_cfg,
                scorer=scorer,
            )

        # Pool pass for code.
        _skip_code_pool = (skip_code_correctness
                           and config.experiment.function == "train")
        if code_idx_set and not _skip_code_pool:
            import multiprocessing as mp
            jobs = [(k, extracted_output_list[k], sample_data_list[k], ds_cfg, scorer)
                    for k in code_idx_set]
            num_workers = min(32, len(jobs))
            with mp.Pool(num_workers) as pool:
                for k, ok in pool.imap_unordered(_check_code_worker, jobs):
                    correctness_list[k] = ok
    for i in range(len(index_list)):
        index_i = index_list[i]
        data[index_i]["correctness"].append(correctness_list[i])

    # Average tokens committed per denoising step (diffusion models only).
    # Computed here — before the per-prompt loop below clears step_map in
    # evaluation mode — so the metric is still available when the results
    # file is written (reward/rl_reward.py used to read step_map after it
    # had already been zeroed, which produced avg tok/step: 0.000).
    # step_map is a list-of-lists: one entry per response, each entry is the
    # same length as the response and labels each position with the denoising
    # step that unmasked it. tokens_per_step = len(sm) / num_distinct_steps.
    _ratios = []
    for _item in data:
        for _sm in _item.get("step_map", []):
            if _sm:
                _ratios.append(len(_sm) / len(set(_sm)))
    avg_tok_per_step = sum(_ratios) / len(_ratios) if _ratios else 0.0

    final_data = []
    num_prompts_total = 0
    num_prompts_kept = 0
    for i in range(len(data)):
        correctness = data[i]["correctness"]
        lengths = data[i]["response_length"]

        for j in range(len(lengths)):
            if OmegaConf.select(config, "rollout.max_gen_length", default=MISSING) is not MISSING and lengths[j] >= config.rollout.max_gen_length - 5:
                correctness[j] = False
            if OmegaConf.select(config, "rollout.max_token", default=MISSING) is not MISSING and lengths[j] >= config.rollout.max_token - 5:
                correctness[j] = False

        rewards = _z_score_normalize(correctness)

        data[i]["rewards"] = rewards

        if config.experiment.function == "train":
            num_prompts_total += 1

            proportion = sum(correctness) / len(correctness)
            _loss_type = getattr(config.training, "loss_type", "kl")
            if _loss_type in ("ppo", "grpo") and (proportion > 0.8 or proportion < 0.2):
                continue
            num_prompts_kept += 1

            for j in range(len(rewards)):
                data_i = {}
                data_i["prompt"] = data[i]["prompt"]
                data_i["question"] = data[i].get("question", "")
                data_i["reward"] = rewards[j]
                data_i["response"] = data[i]["full_output"][j]
                data_i["step_map"] = data[i]["step_map"][j]
                final_data.append(data_i)

        if config.experiment.function == "evaluation":
            data[i]["step_map"] = []

    if config.experiment.function == "train":
        cprint(f"Prompt filter (10%-90%): kept {num_prompts_kept}/{num_prompts_total} prompts "
               f"({num_prompts_kept/max(num_prompts_total,1):.1%}), "
               f"{len(final_data)} total samples", "green")
        with open(project_name + "/temp_data/" + config.dataset.optimization_data + ".json", "w", encoding="utf-8") as f:
            json.dump(final_data, f, indent=2, ensure_ascii=False)

    os.makedirs(os.path.dirname(file_name), exist_ok=True)
    with open(file_name, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    outputs_result_name = project_name + "/results/results-" + outputs_name + ".txt"
    os.makedirs(os.path.dirname(outputs_result_name), exist_ok=True)
    with open(outputs_result_name, "a") as f:
        def save_and_print(text):
            cprint("\n\n\n" + text, color="green")
            f.write(text + "\n")

        ds_cfg_for_acc = DATASET_CONFIGS.get(dataset, None)
        _skip_code = (skip_code_correctness
                      and config.experiment.function == "train")
        scorable_list = [
            _is_scorable(sample_data_list[k], ds_cfg_for_acc)
            and not (_skip_code and get_domain(sample_data_list[k], ds_cfg_for_acc) == "code")
            for k in range(len(index_list))
        ]

        per_domain = {"math": [0, 0], "mc": [0, 0], "code": [0, 0],
                      "ifeval": [0, 0], "triviaqa": [0, 0]}
        # LiveBench: separate per-category accumulator (math/reasoning/
        # data_analysis/coding/instruction_following/language). Active only
        # when the dataset is "LiveBench" (or rows carry livebench_category).
        per_lb_cat = {}
        # MT-AIME (and any other dataset with per-language `subdivision`):
        # accumulator keyed by subdivision string (e.g., "mt_aime_de").
        per_subdivision = {}
        n_correct_scorable = 0
        n_scorable = 0
        for k in range(len(index_list)):
            if not scorable_list[k]:
                continue
            n_scorable += 1
            if correctness_list[k]:
                n_correct_scorable += 1
            dom = get_domain(sample_data_list[k], ds_cfg_for_acc)
            if dom in per_domain:
                per_domain[dom][1] += 1
                if correctness_list[k]:
                    per_domain[dom][0] += 1
            # LiveBench per-category accumulator
            lb_cat = sample_data_list[k].get("livebench_category")
            if lb_cat:
                if lb_cat not in per_lb_cat:
                    per_lb_cat[lb_cat] = [0, 0]
                per_lb_cat[lb_cat][1] += 1
                if correctness_list[k]:
                    per_lb_cat[lb_cat][0] += 1
            # Per-subdivision accumulator (MT-AIME, MMLU subjects, etc.)
            sub = sample_data_list[k].get("subdivision")
            if sub:
                if sub not in per_subdivision:
                    per_subdivision[sub] = [0, 0]
                per_subdivision[sub][1] += 1
                if correctness_list[k]:
                    per_subdivision[sub][0] += 1

        acc = n_correct_scorable / max(n_scorable, 1)
        avg_len = sum(response_length_list)/len(response_length_list)

        # avg_tok_per_step is computed earlier — see the block just above the
        # per-prompt clear loop.

        per_domain_strs = []
        per_domain_metrics = {}
        for dom in ("math", "mc", "code", "ifeval", "triviaqa"):
            c, t = per_domain[dom]
            if t > 0:
                per_domain_strs.append(f"{dom}: {c/t:.4f} ({c}/{t})")
                per_domain_metrics[dom] = c / t
        # LiveBench per-category line — sorted for stable output
        for lb_cat in sorted(per_lb_cat.keys()):
            c, t = per_lb_cat[lb_cat]
            if t > 0:
                per_domain_strs.append(f"lb_{lb_cat}: {c/t:.4f} ({c}/{t})")
                per_domain_metrics[f"lb_{lb_cat}"] = c / t
        # Per-subdivision line (MT-AIME `mt_aime_<lang>`, etc.) — also sorted.
        # Only emit if there's more than one subdivision (otherwise it's just a
        # restatement of the global acc).
        if len(per_subdivision) > 1:
            for sub in sorted(per_subdivision.keys()):
                c, t = per_subdivision[sub]
                if t > 0:
                    per_domain_strs.append(f"{sub}: {c/t:.4f} ({c}/{t})")
                    per_domain_metrics[sub] = c / t
        per_domain_str = "  ".join(per_domain_strs)
        n_total = len(correctness_list)
        scorable_str = f"scorable: {n_scorable}/{n_total}"

        output_text = f"train step: {config.experiment.current_epoch}  "

        if config.experiment.function == "train":
            if config.model.model_base != "sdar" and config.model.model_base != "trado":
                output_text = output_text + f"remasking_strategy: {config.rollout.remasking_strategy}  block_size: {config.rollout.block_size}  acc: {acc}  avg length: {avg_len}  avg tok/step: {avg_tok_per_step:.3f}"
            else:
                output_text = output_text + f"remasking_strategy: {config.rollout.remasking_strategy}  top_k: {config.rollout.top_k}  acc: {acc}  avg length: {avg_len}  avg tok/step: {avg_tok_per_step:.3f}"
        else:
            if config.model.model_base != "sdar" and config.model.model_base != "trado":
                output_text = output_text + f"remasking_strategy: {config.evaluation.remasking_strategy}  block_size: {config.evaluation.block_size}  acc: {acc}  avg length: {avg_len}  avg tok/step: {avg_tok_per_step:.3f}"
            else:
                output_text = output_text + f"remasking_strategy: {config.evaluation.remasking_strategy}  top_k: {config.evaluation.top_k}  acc: {acc}  avg length: {avg_len}  avg tok/step: {avg_tok_per_step:.3f}"
        if per_domain_str:
            output_text += f"  [{scorable_str}  {per_domain_str}]"
        else:
            output_text += f"  [{scorable_str}]"
        save_and_print(output_text)

    if wandb_run is not None:
        flat_rewards = []
        for item in data:
            item_rewards = item.get("rewards", [])
            if isinstance(item_rewards, list):
                flat_rewards.extend(item_rewards)

        if config.experiment.function == "train":
            prefix = f"train/{config.rollout.remasking_strategy}"
        else:
            prefix = f"eval/{config.evaluation.remasking_strategy}/{dataset}"
        metrics = {
            f"{prefix}/accuracy": float(acc),
            f"{prefix}/avg_len": float(avg_len),
            f"{prefix}/scorable_frac": float(n_scorable / max(n_total, 1)),
            "train/current_epoch": config.experiment.current_epoch,
        }
        if config.experiment.function == "train":
            metrics["train/dynamic_threshold"] = float(config.rollout.dynamic_threshold)
            metrics["train/avg_tok_per_step"] = float(avg_tok_per_step)
        for dom, dom_acc in per_domain_metrics.items():
            metrics[f"{prefix}/accuracy_{dom}"] = float(dom_acc)
        if len(flat_rewards) > 0:
            metrics[f"{prefix}/reward_mean"] = float(sum(flat_rewards) / len(flat_rewards))
            metrics[f"{prefix}/reward_min"] = float(min(flat_rewards))
            metrics[f"{prefix}/reward_max"] = float(max(flat_rewards))
            mean_reward = metrics[f"{prefix}/reward_mean"]
            reward_var = sum((r - mean_reward) ** 2 for r in flat_rewards) / len(flat_rewards)
            metrics[f"{prefix}/reward_std"] = float(reward_var ** 0.5)

        wandb.log(metrics)


def get_config():
    cli_conf = OmegaConf.from_cli()
    yaml_conf = OmegaConf.load(cli_conf.config)
    conf = OmegaConf.merge(yaml_conf, cli_conf)
    return conf

if __name__ == "__main__":
    config = get_config()
    project_name = config.experiment.project

    wandb_enabled = bool(config.wandb.get("enabled", True))
    wandb_run = None
    if wandb_enabled:
        wandb_run_id = os.getenv("WANDB_RUN_ID", None)
        if wandb_run_id is None:
            raise ValueError("WANDB_RUN_ID environment variable is not set.")
        wandb_run = wandb.init(id=wandb_run_id, resume="must")
        wandb.define_metric("train/current_epoch")
        wandb.define_metric("*", step_metric="train/current_epoch")

    # Standalone mode runs from reward/ dir, so paths need "../" prefix.
    # Patch config paths for backward compat.
    if config.experiment.current_epoch > 1:
        config.model.pretrained_model = "../" + project_name + "/ckpt/" + config.model.optimized_name
    config.experiment.project = "../" + project_name

    compute_rewards(config, config.experiment.project, wandb_run=wandb_run)

    if wandb_run is not None:
        wandb.finish()
