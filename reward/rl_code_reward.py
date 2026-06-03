import json
import os
import math_utils
import nest_asyncio
from scipy.stats import norm
from concurrent.futures import ThreadPoolExecutor
import asyncio
from termcolor import cprint
from omegaconf import MISSING
from omegaconf import DictConfig, ListConfig, OmegaConf
import wandb
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
            raise ValueError("WANDB_RUN_ID environment variable is not set. Please set it to the desired run ID.")
        wandb_run = wandb.init(
            id=wandb_run_id,
            resume="must",
        )
        wandb.define_metric("train/current_epoch")
        wandb.define_metric("*", step_metric="train/current_epoch")

    if config.experiment.current_epoch == 1:
        pretrained_model = config.model.pretrained_model
    else:
        pretrained_model = "../" + project_name + "/ckpt/" + config.model.optimized_name
    

    if config.experiment.function == "train":
        shrink = config.training.shrink
        dataset = config.dataset.train_dataset
        outputs_name = "rl-" + pretrained_model.replace("/", ".") + "-" + dataset
        
    elif config.experiment.function == "evaluation":
        dataset = config.evaluation.eval_dataset
        outputs_name = "eval-" + pretrained_model.replace("/", ".") + "-" + dataset
    
    

    
    file_name = "../" + project_name + "/temp_data/outputs-" + outputs_name + ".json"

    with open(file_name, 'r') as f:
        data = json.load(f)



    def z_score_normalize(lst):
        mean = sum(lst) / len(lst)
        std = (sum((x - mean) ** 2 for x in lst) / len(lst)) ** 0.5
        if std == 0:
            return [0 for x in lst]
        return [(x - mean) / std for x in lst]






    def set_last_t(lst: list, t: int) -> None:
        new_lst = lst.copy()
        new_val = max(lst) + 1
        new_lst[-t:] = [new_val] * t
        return new_lst



    # avg_tok_per_step — computed before the per-prompt loop clears step_map
    # in evaluation mode (see rl_reward.py for the same pattern / fix).
    _ratios = []
    for _item in data:
        for _sm in _item.get("step_map", []):
            if _sm:
                _ratios.append(len(_sm) / len(set(_sm)))
    avg_tok_per_step = sum(_ratios) / len(_ratios) if _ratios else 0.0

    response_length_list = []
    num_task   = 0
    num_correct_task = 0
    final_data = []
    for i in range(len(data)):
        response_length_list = response_length_list + data[i]["response_length"]
        acc_list = []
        for x in data[i]["correctness"]:
            acc_list.append(sum(x))
            num_correct_task += all(x)
            num_task += 1
        lengths = data[i]["response_length"]

        for j in range(len(lengths)):
            if OmegaConf.select(config, "rollout.max_gen_length", default=MISSING) is not MISSING and lengths[j] >= config.rollout.max_gen_length - 5:
                acc_list[j] = 0
            if OmegaConf.select(config, "rollout.max_token", default=MISSING) is not MISSING and lengths[j] >= config.rollout.max_token - 5:
                acc_list[j] = 0

        rewards = z_score_normalize(acc_list)
        data[i]["rewards"] = rewards
        
        if config.experiment.function == "train":

            for j in range(len(rewards)):
                data_i = {}
                data_i["prompt"] = data[i]["prompt"]
                data_i["reward"] = rewards[j]
                data_i["response"] = data[i]["full_output"][j]
                data_i["step_map"] = data[i]["step_map"][j]
                final_data.append(data_i)
        
        if config.experiment.function == "evaluation":
            data[i]["step_map"] = []


    if config.experiment.function == "train":
        with open("../" + project_name + "/temp_data/" + config.dataset.optimization_data + ".json", "w", encoding="utf-8") as f:
            json.dump(final_data, f, indent=2, ensure_ascii=False)


    os.makedirs(os.path.dirname(file_name), exist_ok=True)
    with open(file_name, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


    outputs_result_name = "../" + project_name + "/results/results-" + outputs_name + ".txt"
    os.makedirs(os.path.dirname(outputs_result_name), exist_ok=True)
    with open(outputs_result_name, "a") as f:
        # Save + print
        def save_and_print(text):
            cprint("\n\n\n" + text, color="green")
            f.write(text + "\n")
        
        acc = num_correct_task / num_task if num_task else 0
        avg_len = sum(response_length_list)/len(response_length_list)

        # avg_tok_per_step computed earlier — see block above the per-prompt loop.

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
            "train/current_epoch": config.experiment.current_epoch,
        }
        if config.experiment.function == "train":
            metrics["train/dynamic_threshold"] = float(config.rollout.dynamic_threshold)
            metrics["train/avg_tok_per_step"] = float(avg_tok_per_step)
        # EvalPlus-scored datasets (HumanEval/MBPP) expose {base, plus}
        # pass@1 directly via rl_execute.py. The "accuracy" metric above
        # already reflects base pass@1 (correctness=[base_pass] per cand).
        # Surface plus pass@1 as a separate metric for completeness.
        _ep_pak = next(
            (item.get("evalplus_pass_at_k") for item in data
             if isinstance(item.get("evalplus_pass_at_k"), dict)),
            None,
        )
        if _ep_pak is not None:
            if _ep_pak.get("base") is not None:
                metrics[f"{prefix}/evalplus_base_pass@1"] = float(_ep_pak["base"])
            if _ep_pak.get("plus") is not None:
                metrics[f"{prefix}/evalplus_plus_pass@1"] = float(_ep_pak["plus"])
        if len(flat_rewards) > 0:
            metrics[f"{prefix}/reward_mean"] = float(sum(flat_rewards) / len(flat_rewards))
            metrics[f"{prefix}/reward_min"] = float(min(flat_rewards))
            metrics[f"{prefix}/reward_max"] = float(max(flat_rewards))
            mean_reward = metrics[f"{prefix}/reward_mean"]
            reward_var = sum((r - mean_reward) ** 2 for r in flat_rewards) / len(flat_rewards)
            metrics[f"{prefix}/reward_std"] = float(reward_var ** 0.5)

        wandb.log(metrics)
        wandb.finish()
