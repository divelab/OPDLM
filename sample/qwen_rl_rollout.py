"""
vLLM-based rollout for standard autoregressive models (Qwen3-0.6B, etc.)
Drop-in replacement for sdar_rl_rollout.py when model_base == "qwen".
"""

import os
import re
import json
import random
import math

# Seed from parent process for reproducibility
_seed_str = os.environ.get("TRACERL_SEED")
if _seed_str is not None:
    random.seed(int(_seed_str))

from termcolor import cprint
from jinja2 import Template
from omegaconf import OmegaConf

import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "reward"))
from eval_utils import DATASET_CONFIGS, reformat_choices, build_evalplus_prompt, build_lcb_prompt

# Domain-based extraction dispatch (math / mc / code).
# Per-sample `data_i["domain"]` > ds_cfg["domain"] > default "math".
from domain_reward import extract_answer


def get_config():
    cli_conf = OmegaConf.from_cli()
    yaml_conf = OmegaConf.load(cli_conf.config)
    return OmegaConf.merge(yaml_conf, cli_conf)

def _format_question(data_i):
    """Format a single question using ds_cfg rules (reformat_choices, prompt_template, etc.)."""
    q = data_i["question"]
    if _ds_cfg.get("reformat_choices"):
        q = reformat_choices(q)
    per_dom_tpl = _ds_cfg.get("per_domain_template")
    if per_dom_tpl is not None:
        dom = data_i.get("domain", "math")
        tpl = per_dom_tpl.get(dom, "{question}")
        return tpl.format(question=q)
    tpl = _ds_cfg.get("prompt_template")
    if callable(tpl):
        # Per-row dispatch (MathBench, LMB-Hard) — definitions live in
        # eval_utils.DATASET_CONFIGS. Signature: tpl(data_i, q) -> str.
        return tpl(data_i, q)
    if tpl is not None:
        return tpl.format(question=q)
    return q

def get_prompt(data_i):
    if _use_chat_template:
        # Prompt resolution (fail-loud): dataset MUST be registered in
        # DATASET_CONFIGS or we refuse to run, so we never silently train/eval
        # on a wrong prompt. Valid shapes:
        #   - chat_style == "evalplus_prefill" (HumanEval/MBPP): route through
        #     build_evalplus_prompt for the evalplus-style assistant-prefill
        #     prompt ending inside a ```python fence.
        #   - per_domain_template (dict) → pick by data_i["domain"] (opdlm_train)
        #   - prompt_template (str / callable) → apply it (GSM8K, MATH500, MathBench, LMB-Hard, ...)
        #   - all None → pass question as-is
        if _ds_cfg.get("chat_style") == "evalplus_prefill":
            return build_evalplus_prompt(data_i["question"], _tokenizer)
        if _ds_cfg.get("chat_style") == "lcb":
            # LCB: system message + pre-baked canonical user prompt; always
            # non-thinking (matches Qwen3 tech-report LCB numbers).
            return build_lcb_prompt(data_i["question"], _tokenizer,
                                    enable_thinking=False)

        content = _format_question(data_i)
        # `content` is a string (single user turn) or a list of
        # {"role","content"} dicts (multi-turn; e.g. SDAR MMLU 5-shot or
        # TriviaQA 1-shot send in-context examples as separate USER/ASSISTANT
        # rounds so the model sees the test query as the current turn).
        if isinstance(content, list):
            messages = content
        else:
            messages = [{"role": "user", "content": content}]
        return _tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=_enable_thinking,
        )
    return Template(system_prompts).render(problem=data_i["question"])


def extract_code(full_output):
    matches = re.findall(r"```python(.*?)```", full_output, re.DOTALL)
    if matches:
        return matches[-1].strip()
    return "We can not extract the code in the output. "


def get_data_chunk(data, num_node, node_idx):
    total = len(data)
    chunk_size = (total + num_node - 1) // num_node
    start_idx = node_idx * chunk_size
    end_idx = min((node_idx + 1) * chunk_size, total)
    return data[start_idx:end_idx]


def random_select(data_list, random_k):
    return random.sample(data_list, random_k)


if __name__ == "__main__":

    config = get_config()
    from transformers import AutoTokenizer

    # Thinking mode control via tokenizer's chat_template (enable_thinking parameter).
    # Single thinking knob: rollout.start_with_think — same convention as
    # bd3lm_rl_rollout.py (line 758) and sdar_rl_rollout.py (line 372).
    # Old code read `disable_thinking` (never set by eval.py) and defaulted
    # to True, ignoring the runner's ENABLE_THINKING=false.
    _enable_thinking = bool(getattr(config.rollout, "start_with_think", False))
    _use_chat_template = True
    _user_suffix = "Please reason step by step, and put your final answer within \\boxed{}."
    # Fallback for legacy hardcoded prompts
    system_prompts = '''<|im_start|>user\n{{problem}}\nPlease reason step by step, and put your final answer within \\boxed{}.<|im_end|>\n<|im_start|>assistant\n'''

    project_name = config.experiment.project

    if config.experiment.current_epoch == 1:
        pretrained_model = config.model.pretrained_model
    else:
        pretrained_model = "../" + project_name + "/ckpt/" + config.model.optimized_name

    code_task = False
    if config.experiment.function == "train":
        dataset = config.dataset.train_dataset
        k_sample = config.rollout.num_response_per_task

        if config.dataset.data_type == "code":
            code_task = True
            system_prompts_function = '''<|im_start|>user\n{{problem}}\nPlace your code within a single Python code block ```python ```. Do not include more than one code block. <|im_end|>\n<|im_start|>assistant\n'''
            system_prompts_stdio = '''<|im_start|>user\nThis is the problem:\n{{problem}}\nYou should put your code in ```python ```. Use input() to read input and print() to produce output in your script. <|im_end|>\n<|im_start|>assistant\n'''
            if config.rollout.start_with_think:
                system_prompts_stdio = '''<|im_start|>user\nThis is the problem:\n{{problem}}\nYou should put your code in ```python ```. Use input() to read input and print() to produce output in your script. <|im_end|>\n<|im_start|>assistant<think>\n'''

        outputs_name = "rl-" + pretrained_model.replace("/", ".") + "-" + dataset

    elif config.experiment.function == "evaluation":
        dataset = config.evaluation.eval_dataset
        if config.evaluation.data_type == "code":
            code_task = True
            system_prompts_function = '''<|im_start|>user\n{{problem}}\nPlace your code within a single Python code block ```python ```. Do not include more than one code block. <|im_end|>\n<|im_start|>assistant\n'''
            system_prompts_stdio = '''<|im_start|>user\nThis is the problem:\n{{problem}}\nYou should put your code in ```python ```. Use input() to read input and print() to produce output in your script. <|im_end|>\n<|im_start|>assistant\n'''
            if config.rollout.start_with_think:
                system_prompts_stdio = '''<|im_start|>user\nThis is the problem:\n{{problem}}\nYou should put your code in ```python ```. Use input() to read input and print() to produce output in your script. <|im_end|>\n<|im_start|>assistant<think>\n'''

        k_sample = config.evaluation.num_response_per_task

        config.rollout.tensor_parallel_size = config.evaluation.tensor_parallel_size
        config.rollout.max_active = config.evaluation.max_active
        config.rollout.max_token = config.evaluation.max_token
        config.rollout.temperature = config.evaluation.temperature
        config.rollout.top_p = config.evaluation.top_p
        config.rollout.top_k = config.evaluation.top_k
        config.rollout.gpu_memory_utilization = OmegaConf.select(
            config,
            "evaluation.gpu_memory_utilization",
            default=OmegaConf.select(config, "rollout.gpu_memory_utilization", default=0.5),
        )

        outputs_name = "eval-" + pretrained_model.replace("/", ".") + "-" + dataset

    # Look up dataset-specific config from eval_utils (prompt template + domain
    # dispatch for extraction). Consulted for BOTH training and evaluation so
    # that e.g. opdlm_train training also picks up prompt_template=None and
    # the per-sample domain mix.
    #
    # Fail loud if the dataset isn't registered — otherwise we'd silently run
    # with the wrong prompt / extraction and the user wouldn't notice.
    if dataset not in DATASET_CONFIGS:
        raise ValueError(
            f"Dataset {dataset!r} is not registered in eval_utils.DATASET_CONFIGS. "
            f"Add an entry with `domain` and `prompt_template` (use None if the "
            f"prompt is already encoded per-sample, e.g. opdlm_train). "
            f"Known datasets: {sorted(DATASET_CONFIGS.keys())}"
        )
    _ds_cfg = DATASET_CONFIGS[dataset]

    # Honor _ds_cfg["path"] so multiple registry entries can share one JSON
    # (e.g. ARC_C / ARC_C_sdar both read ARC_C.json).
    _ds_path = _ds_cfg.get("path", dataset + ".json")
    with open("../data/" + _ds_path, 'r') as f:
        data = json.load(f)

    num_node = config.experiment.num_node
    node_index = config.experiment.node_index
    if num_node > 1:
        if config.experiment.function == "train":
            random.shuffle(data)
        data = get_data_chunk(data, num_node, node_index)

    if config.experiment.function == "train":
        random_select_num = config.rollout.num_task_per_step
        random_select_num = int(random_select_num / num_node)
        random_select_num = min(random_select_num, len(data))
        data = random_select(data, random_select_num)

    # Optional chunking (eval-only): if dataset.{chunk_index, num_chunks}
    # are both set in the YAML config (driven by eval.py's --num_chunks
    # auto-iteration), take a contiguous slice of `data` and append
    # `_chunk{i}of{n}` to the output filename. Lets eval.py recover from
    # a worker crash without redoing already-completed chunks.
    _chunk_idx  = OmegaConf.select(config, "dataset.chunk_index", default=None)
    _num_chunks = OmegaConf.select(config, "dataset.num_chunks",  default=None)
    if (_chunk_idx is not None and _num_chunks is not None
            and config.experiment.function == "evaluation" and int(_num_chunks) > 1):
        _chunk_idx, _num_chunks = int(_chunk_idx), int(_num_chunks)
        _cs = (len(data) + _num_chunks - 1) // _num_chunks
        _s, _e = _chunk_idx * _cs, min((_chunk_idx + 1) * _cs, len(data))
        data = data[_s:_e]
        outputs_name = outputs_name + f"_chunk{_chunk_idx}of{_num_chunks}"
        print(f"[chunk] {_chunk_idx+1}/{_num_chunks} slice [{_s}:{_e}] of original ({len(data)} prompts)")

    # Env-driven manual sharding: NUM_CHUNKS=N CHUNK_INDEX=K → K-th of N
    # equal slices. Lets you run N single-GPU shells in parallel with
    # different CHUNK_INDEX + CUDA_VISIBLE_DEVICES.
    _en_num = os.environ.get("NUM_CHUNKS", "").strip()
    _en_idx = os.environ.get("CHUNK_INDEX", "").strip()
    if _en_num and _en_idx and config.experiment.function == "evaluation":
        _en_num, _en_idx = int(_en_num), int(_en_idx)
        _cs = (len(data) + _en_num - 1) // _en_num
        _s, _e = _en_idx * _cs, min((_en_idx + 1) * _cs, len(data))
        data = data[_s:_e]
        outputs_name = outputs_name + f"_chunk{_en_idx}of{_en_num}"
        print(f"[CHUNK] {_en_idx+1}/{_en_num} → data[{_s}:{_e}] ({len(data)} prompts)")

    num = len(data)

    model_path = os.path.expanduser(pretrained_model)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    _tokenizer = tokenizer  # used by get_prompt() for apply_chat_template

    rollout_gpu_memory_utilization = float(
        OmegaConf.select(config, "rollout.gpu_memory_utilization", default=0.5)
    )

    # Evalplus datasets (HumanEval/MBPP) route through build_evalplus_prompt
    # in get_prompt() and don't carry `test_method`/`prefix` — the instruction
    # + assistant prefill is handled entirely by the evalplus wrapper. Skip
    # the legacy code prompt setup for them.
    # LiveCodeBench (LCB_v5/v6) works the same way: build_lcb_prompt bakes
    # the canonical system message + pre-rendered user prompt, no per-item
    # test_method/prefix plumbing.
    _uses_evalplus = _ds_cfg.get("chat_style") == "evalplus_prefill"
    _uses_lcb = _ds_cfg.get("chat_style") == "lcb"
    _prompt_is_prebuilt = _uses_evalplus or _uses_lcb

    # Build prompts
    generation_prompts = []
    prefix_list = []
    index_list = []
    for i in range(num):
        if code_task and not _prompt_is_prebuilt:
            if data[i]["test_method"] == "stdio":
                system_prompts = system_prompts_stdio
                prefix_list = prefix_list + [None] * k_sample
            else:
                system_prompts = system_prompts_function + data[i]["prefix"]
                prefix_list = prefix_list + [data[i]["prefix"]] * k_sample
        elif code_task and _prompt_is_prebuilt:
            # No raw prefix — the prompt builder already includes everything.
            prefix_list = prefix_list + [None] * k_sample
        generation_prompts = generation_prompts + [get_prompt(data[i])] * k_sample
        index_list = index_list + [i] * k_sample
        data[i]["full_output"] = []
        data[i]["step_map"] = []
        data[i]["extracted_output"] = []
        data[i]["response_length"] = []
        data[i]["prompt"] = get_prompt(data[i])

    # ------------------- Generation with vLLM -------------------
    cprint("start generation (vLLM)...", "green")

    from vllm import LLM, SamplingParams

    # vLLM top_k: 0 or negative means disabled
    top_k_val = config.rollout.top_k
    # Handle OmegaConf ListConfig or plain list
    if hasattr(top_k_val, '__iter__') and not isinstance(top_k_val, str):
        top_k_val = list(top_k_val)
        top_k_val = top_k_val[0] if len(top_k_val) > 0 else -1
    top_k_val = int(top_k_val)
    if top_k_val == 0:
        top_k_val = -1  # vLLM convention: -1 = disabled

    tp = int(config.rollout.tensor_parallel_size)

    cprint(f"Loading vLLM engine (tp={tp}, gpu_mem={rollout_gpu_memory_utilization})...", "green")
    llm = LLM(
        model=model_path,
        tensor_parallel_size=tp,
        gpu_memory_utilization=rollout_gpu_memory_utilization,
        trust_remote_code=True,
        dtype="auto",
        max_model_len=int(config.rollout.max_token) + 1024,  # prompt + generation
    )

    # Per-domain max_token cap (only active if sample has a `domain` field, e.g. opdlm_train).
    # Effective per-sample max = min(scheduled rollout.max_token, cap[domain]).
    per_dom_max_cap = dict(config.rollout.get("per_domain_max_token", {}) or {})
    scheduled_max = int(config.rollout.max_token)

    def _max_tokens_for(data_i):
        dom = data_i.get("domain")
        if dom is None or dom not in per_dom_max_cap:
            return scheduled_max
        return min(scheduled_max, int(per_dom_max_cap[dom]))

    sp_common = dict(
        temperature=float(config.rollout.temperature),
        top_p=float(config.rollout.top_p),
        top_k=top_k_val,
        min_p=float(getattr(config.rollout, "min_p", 0.0)),
        stop=["<|im_end|>", "<|endoftext|>"],
    )

    if per_dom_max_cap:
        # Build per-prompt SamplingParams aligned with generation_prompts order.
        sampling_params = []
        for i in range(num):
            mt = _max_tokens_for(data[i])
            sampling_params += [SamplingParams(**sp_common, max_tokens=mt)] * k_sample
    else:
        sampling_params = SamplingParams(**sp_common, max_tokens=scheduled_max)

    cprint(f"Generating {len(generation_prompts)} prompts...", "green")
    outputs = llm.generate(generation_prompts, sampling_params, use_tqdm=True)

    # Extract results (vLLM returns in same order as input)
    restored_outputs = [o.outputs[0].text for o in outputs]

    cprint("generation job done!", "green")

    # ------------------- Post-processing -------------------
    def get_token_lengths(strings, tokenizer):
        pad_token = tokenizer.pad_token
        escaped = re.escape(pad_token)
        pattern = rf"(?:{escaped})+"
        remove_pattern = escaped
        collapse_re = re.compile(pattern)
        lengths = []
        for s in strings:
            s_clean = collapse_re.sub(lambda _: pad_token if isinstance(pad_token, str) else '', s)
            s_clean = re.sub(remove_pattern, '', s_clean)
            lengths.append(len(tokenizer.encode(s_clean, add_special_tokens=False)))
        return lengths

    response_length = get_token_lengths(restored_outputs, tokenizer)
    mean_response_length = sum(response_length) / len(response_length)

    i = 0
    for full_output in restored_outputs:
        index_i = index_list[i]
        if code_task:
            if _uses_evalplus:
                # evalplus scoring in rl_execute.py reads `full_output`
                # directly and runs evalplus.sanitize on it — we still
                # populate extracted_output for logging parity.
                extracted_output = extract_code(full_output)
            elif _uses_lcb:
                # LCB canonical extractor: keep only the last ``` block.
                # Respects ds_cfg["extract"] override (extract_lcb_code).
                extracted_output = _ds_cfg["extract"](full_output)
            elif data[int(i / k_sample)]["test_method"] == "function":
                extracted_output = extract_code(prefix_list[i] + full_output)
            else:
                extracted_output = extract_code(full_output)
        else:
            # Domain dispatch: per-sample data_i["domain"] > ds_cfg["domain"] > math.
            # ds_cfg["extract"] override (e.g. BBH, HellaSwag) still takes precedence.
            extracted_output = extract_answer(
                full_output, data_i=data[index_i], ds_cfg=_ds_cfg,
            )
        data[index_i]["full_output"].append(full_output)
        data[index_i]["step_map"].append([])  # AR: no step_map needed
        data[index_i]["extracted_output"].append(extracted_output)
        data[index_i]["response_length"].append(response_length[i])
        i += 1

    # Save output
    if num_node > 1:
        output_file_name = "../" + project_name + f"/temp_data/outputs-{node_index}-" + outputs_name + ".json"
    else:
        output_file_name = "../" + project_name + "/temp_data/outputs-" + outputs_name + ".json"
    os.makedirs(os.path.dirname(output_file_name), exist_ok=True)
    with open(output_file_name, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    cprint(f"Saved {len(data)} items to {output_file_name} (mean response length: {mean_response_length:.1f})", "green")

    # Kill vLLM worker processes and force-exit to free GPU memory
    import signal, multiprocessing
    for child in multiprocessing.active_children():
        try:
            child.kill()
            child.join(timeout=5)
        except Exception:
            pass
    try:
        import psutil
        for child in psutil.Process(os.getpid()).children(recursive=True):
            try:
                child.kill()
            except Exception:
                pass
    except ImportError:
        pass
    import sys; sys.stdout.flush(); sys.stderr.flush()
    os._exit(0)
