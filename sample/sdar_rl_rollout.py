import os as _os
_os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

# Use /dev/shm for fast triton cache; fall back to /tmp/torch_cache_<uid> if
# that path is owned by another user and thus unwritable to us (mirrors
# bd3lm_rl_rollout.py).
_cache_root = "/dev/shm/torch_cache"
try:
    _os.makedirs(_cache_root, exist_ok=True)
    _test_file = _os.path.join(_cache_root, f".write_test_{_os.getpid()}")
    open(_test_file, "w").close()
    _os.remove(_test_file)
except (PermissionError, OSError):
    _cache_root = _os.environ.get("TRITON_CACHE_DIR", _os.path.join("/tmp", f"torch_cache_{_os.getuid()}"))
    _os.makedirs(_cache_root, exist_ok=True)
_os.environ["TORCH_EXTENSIONS_DIR"] = _os.path.join(_cache_root, "torch_extensions")
_os.environ["TRITON_CACHE_DIR"]      = _os.path.join(_cache_root, "triton")
_os.environ["XDG_CACHE_HOME"]        = _cache_root
_os.environ.setdefault("CUDA_MODULE_LOADING", "LAZY")



_os.environ.setdefault("TORCH_NCCL_BLOCKING_WAIT", "1")
_os.environ.setdefault("TORCH_NCCL_ASYNC_ERROR_HANDLING", "1")
_os.environ.pop("NCCL_BLOCKING_WAIT", None)
_os.environ.pop("NCCL_ASYNC_ERROR_HANDLING", None)

import os
import re
import json
from termcolor import cprint
import random
import torch.multiprocessing as mp
from jinja2 import Template

# Seed from parent process for reproducibility
_seed_str = os.environ.get("TRACERL_SEED")
if _seed_str is not None:
    _seed = int(_seed_str)
    random.seed(_seed)
    import torch as _torch_seed
    _torch_seed.manual_seed(_seed)

from omegaconf import DictConfig, ListConfig, OmegaConf

# Import eval_utils for dataset-specific prompts, extraction, and gen settings.
# Mirrors bd3lm_rl_rollout.py so SDAR inference handles the same dataset mix.
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'reward'))
from eval_utils import DATASET_CONFIGS, reformat_choices, build_evalplus_prompt, build_lcb_prompt
# Domain-based extraction dispatch (math / mc / code).
# Per-sample `data_i["domain"]` > ds_cfg["domain"] > default "math".
from domain_reward import extract_answer

def get_config():
    cli_conf = OmegaConf.from_cli()
    yaml_conf = OmegaConf.load(cli_conf.config)
    conf = OmegaConf.merge(yaml_conf, cli_conf)
    return conf

def extract_final_boxed_answer(s: str):
    tag = r'\boxed{'
    start = s.rfind(tag)          # last \boxed{
    if start == -1:
        return "Can not extract the answer!"

    i = start + len(tag)
    depth = 1                    # we are already inside one '{'
    buf = []

    while i < len(s) and depth:
        ch = s[i]
        if ch == '{':
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0:       # matching '}' for the opening \boxed{
                break
        buf.append(ch)
        i += 1

    return ''.join(buf) if depth == 0 else "Can not extract the answer!"


# GSM8K extraction: follows dllm's lm-evaluation-harness gsm8k-cot.yaml
_GSM8K_STRICT_RE = re.compile(r"The answer is (\-?[0-9\.\,]+)\.")
_GSM8K_FLEXIBLE_RE = re.compile(r"(-?[$0-9.,]{2,})|(-?[0-9]+)")

def extract_gsm8k_answer(s: str):
    # Stage 1: strict-match "The answer is X."
    matches = _GSM8K_STRICT_RE.findall(s)
    if matches:
        return matches[0].strip()
    # Stage 2: flexible-extract, last number-like pattern
    matches = _GSM8K_FLEXIBLE_RE.findall(s)
    if matches:
        match = matches[-1]
        if isinstance(match, tuple):
            match = [m for m in match if m]
            if match:
                return match[0].strip()
        else:
            return match.strip()
    return "Can not extract the answer!"




def extract_code(full_output):
    matches = re.findall(r"```python(.*?)```", full_output, re.DOTALL)
    if matches:
        code_output = matches[-1].strip()
    else:
        code_output = "We can not extract the code in the output. "
    return code_output


def get_data_chunk(data, num_node, node_idx):
    total = len(data)
    chunk_size = (total + num_node - 1) // num_node 
    start_idx = node_idx * chunk_size
    end_idx = min((node_idx + 1) * chunk_size, total)
    return data[start_idx:end_idx]


import socket

def _patch_safe_destroy():
    import torch.distributed as dist
    _real_destroy = dist.destroy_process_group
    def _safe_destroy(group=None):
        try:
            if not dist.is_available():
                return
            try:
                if not dist.is_initialized():
                    return
            except Exception:
                return
            _real_destroy(group)
        except AssertionError:
            pass
    dist.destroy_process_group = _safe_destroy



def _llm_worker_run(args):
    (model_path, tp, block_size, sampling_kwargs, gpu_memory_utilization, vis_ids,
     prompts_slice, indices_slice, enforce_eager, max_active, store_port) = args

    import os
    # 1) Setup environment (critical for correct worker behavior)
    os.environ.setdefault("TORCH_NCCL_BLOCKING_WAIT", "1")
    os.environ.setdefault("TORCH_NCCL_ASYNC_ERROR_HANDLING", "1")
    os.environ.pop("NCCL_BLOCKING_WAIT", None)
    os.environ.pop("NCCL_ASYNC_ERROR_HANDLING", None)
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, vis_ids))
    #os.environ.setdefault("TORCH_EXTENSIONS_DIR", f"/tmp/torch_ext_worker_{store_port}")
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(store_port)
    os.environ["JETENGINE_PORT"] = str(store_port)
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    # 1.1) Create a per-worker `sitecustomize.py` and inject it into PYTHONPATH 
    # (this must be done before importing torch/jetengine)
    patch_dir = f"/tmp/je_site_{store_port}"
    os.makedirs(patch_dir, exist_ok=True)
    patch_file = os.path.join(patch_dir, "sitecustomize.py")
    # Important: the content must start at column 0 (no indentation)!
    with open(patch_file, "w") as _f:
        _f.write(
            "import os\n"
            "import torch.distributed as dist\n"
            "_real = dist.init_process_group\n"
            "def _wrapped(backend, init_method=None, *args, **kwargs):\n"
            "    port = os.environ.get('JE_TCP_PORT')\n"
            "    if port and isinstance(init_method, str) and init_method.startswith('tcp://localhost:2333'):\n"
            "        init_method = f'tcp://127.0.0.1:{port}'\n"
            "    return _real(backend, init_method, *args, **kwargs)\n"
            "dist.init_process_group = _wrapped\n"
        )
    os.environ["PYTHONPATH"] = patch_dir + (":" + os.environ["PYTHONPATH"] if "PYTHONPATH" in os.environ else "")
    os.environ["JE_TCP_PORT"] = str(store_port)

    # 2) Import torch and patch the current worker process
    import torch
    import torch.distributed as dist
    _patch_dist_port(store_port)   # Patch port binding for this process
    _patch_safe_destroy()          # Avoid AssertionError in destroy_process_group
    torch.cuda.set_device(0)       # From this worker’s perspective, cuda:0 is the first visible device

    # For debugging: print the worker’s CUDA_VISIBLE_DEVICES and assigned port
    print(f"[worker pid={os.getpid()}] CVD={os.environ['CUDA_VISIBLE_DEVICES']}, port={store_port}, prompts={len(prompts_slice)}", flush=True)

    # 3) Import jetengine and create the engine 
    # (child processes inherit the sitecustomize patch)
    from jetengine_ext.llm import LLM
    from jetengine_ext.sampling_params import SamplingParams

    llm = LLM(
        model_path,
        enforce_eager=enforce_eager,
        tensor_parallel_size=tp,
        mask_token_id=151669,
        block_length=block_size,
        gpu_memory_utilization=gpu_memory_utilization,
    )
    sp = SamplingParams(**sampling_kwargs)
    outs = llm.generate_streaming(prompts_slice, sp, max_active=max_active)
    #seq = [o["text"] for o in outs]
    #print(outs[0]["first_unmask_times"])
    triples = []
    for j, o in enumerate(outs):
        triples.append((
            indices_slice[j],          # Global index (used to restore original order)
            o["text"],                 # Generated text
            o.get("first_unmask_times", None)  # Optional time series aligned with completion tokens
        ))

    try:
        if hasattr(llm, "shutdown"):
            llm.shutdown()
    except Exception:
        pass

    return triples



def _llm_worker_entry(args, out_q):
    import traceback, os
    try:
        res = _llm_worker_run(args)
        out_q.put(("ok", res))
    except Exception:
        out_q.put(("err", {
            "pid": os.getpid(),
            "port": args[-1],  # store_port
            "traceback": traceback.format_exc(),
        }))


def _find_free_port():
    s = socket.socket(); s.bind(('', 0))
    p = s.getsockname()[1]; s.close()
    return p

def _patch_dist_port(port: int):
    import torch.distributed as _dist
    _real_init = _dist.init_process_group

    def _wrapped(backend, init_method=None, *args, **kwargs):
        # jetengine internally hardcodes "tcp://localhost:2333" — replace the port here
        if isinstance(init_method, str) and init_method.startswith("tcp://localhost:2333"):
            init_method = f"tcp://127.0.0.1:{port}"
        return _real_init(backend, init_method, *args, **kwargs)

    _dist.init_process_group = _wrapped


import random 
def random_select(data_list, random_k):
    data_list = random.sample(data_list, random_k)
    return data_list

if __name__ == "__main__":



    tp = int(get_config().rollout.tensor_parallel_size)  # Or check after loading config

    if tp == 1:
        os.environ.setdefault("TORCH_NCCL_ASYNC_ERROR_HANDLING", "1")
        os.environ.setdefault("TORCH_NCCL_BLOCKING_WAIT", "1")
        # These two are NCCL’s own variables, keep using the NCCL_ prefix
        os.environ.setdefault("NCCL_P2P_DISABLE", "1")
        os.environ.setdefault("NCCL_IB_DISABLE", "1")
    else:
        # For multi-GPU communication, do not disable P2P/IB;
        # also clean up related variables (both old and new names)
        for k in [
            "NCCL_P2P_DISABLE", "NCCL_IB_DISABLE",
            "TORCH_NCCL_BLOCKING_WAIT", "TORCH_NCCL_ASYNC_ERROR_HANDLING",
            "NCCL_BLOCKING_WAIT", "NCCL_ASYNC_ERROR_HANDLING",
        ]:
            os.environ.pop(k, None)



    from transformers import AutoTokenizer

    # --- graceful shutdown & unique port ---
    import os, sys, atexit, signal, torch.distributed as dist


    # 2) Automatically set compile architecture according to the local GPU
    # (do NOT hardcode 8.0)
    def _set_arch():
        try:
            if torch.cuda.is_available():
                major, minor = torch.cuda.get_device_capability(0)
                os.environ["TORCH_CUDA_ARCH_LIST"] = f"{major}.{minor}"
        except Exception:
            pass
    _set_arch()

    

    # 1) Use a new port at each startup to avoid conflicts with 2333
    if "MASTER_PORT" not in os.environ:
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = str(_find_free_port())
    # (If JetEngine hardcodes tcp://localhost:2333 instead of using env://,
    # see the “special case” section at the end)

    # 2) Intercept Ctrl-C/TERM to destroy distributed groups & engine gracefully
    _llm = None
    _child_ps = []    # If you create your own mp.Process/Pool, append objects here

    def _cleanup():
        # 2.1) Shutdown JetEngine engine (if API available)
        global _llm
        try:
            if _llm is not None and hasattr(_llm, "shutdown"):
                _llm.shutdown()
        except Exception:
            pass
        # 2.3) Kill/join child processes
        for p in _child_ps:
            try:
                if hasattr(p, "terminate"): p.terminate()
            except Exception:
                pass
        for p in _child_ps:
            try:
                if hasattr(p, "join"): p.join(timeout=2)
            except Exception:
                pass

    atexit.register(_cleanup)
    def _sig_handler(sig, frame):
        _cleanup()
        # 130: standard exit code for SIGINT; 143: for SIGTERM
        sys.exit(130 if sig == signal.SIGINT else 143)

    signal.signal(signal.SIGINT,  _sig_handler)
    signal.signal(signal.SIGTERM, _sig_handler)









    config = get_config()

    try:
        if mp.get_start_method(allow_none=True) != "spawn":
            mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    
    
    # Thinking mode control via tokenizer's chat_template (enable_thinking parameter).
    # Matches bd3lm_rl_rollout.py: per-dataset prompt building is driven by
    # DATASET_CONFIGS (chat_style / prompt_template / reformat_choices).
    # Single thinking knob: rollout.start_with_think=True → enable_thinking=True.
    enable_thinking = bool(config.rollout.start_with_think)

    project_name = config.experiment.project

    if config.experiment.current_epoch == 1:
        pretrained_model = config.model.pretrained_model
    else:
        pretrained_model = "../" + project_name + "/ckpt/" + config.model.optimized_name

    if config.experiment.function == "train":
        dataset = config.dataset.train_dataset
        k_sample = config.rollout.num_response_per_task
        outputs_name = "rl-" + pretrained_model.replace("/", ".") + "-" + dataset

    elif config.experiment.function == "evaluation":
        dataset = config.evaluation.eval_dataset
        k_sample = config.evaluation.num_response_per_task

        config.rollout.tensor_parallel_size = config.evaluation.tensor_parallel_size
        config.rollout.max_active = config.evaluation.max_active
        config.rollout.max_token = config.evaluation.max_token
        config.rollout.remasking_strategy = config.evaluation.remasking_strategy
        config.rollout.dynamic_threshold = config.evaluation.dynamic_threshold
        config.rollout.denoising_steps_per_block = config.evaluation.denoising_steps_per_block
        config.rollout.temperature = config.evaluation.temperature
        config.rollout.top_p = config.evaluation.top_p
        config.rollout.top_k = config.evaluation.top_k
        config.rollout.block_size = config.evaluation.block_size
        config.rollout.gpu_memory_utilization = OmegaConf.select(
            config,
            "evaluation.gpu_memory_utilization",
            default=OmegaConf.select(config, "rollout.gpu_memory_utilization", default=0.5),
        )

        outputs_name = "eval-" + pretrained_model.replace("/", ".") + "-" + dataset

    # Fail loud if the dataset isn't registered. Matches bd3lm_rl_rollout.py.
    if dataset not in DATASET_CONFIGS:
        raise ValueError(
            f"Dataset {dataset!r} is not registered in eval_utils.DATASET_CONFIGS. "
            f"Known datasets: {sorted(DATASET_CONFIGS.keys())}"
        )
    ds_cfg = DATASET_CONFIGS[dataset]

    # Honor ds_cfg["path"] so multiple registry entries can share one JSON
    # (e.g. ARC_C / ARC_C_sdar both read ARC_C.json).
    _ds_path = ds_cfg.get("path", dataset + ".json")
    with open("../data/" + _ds_path, 'r') as f:
        data = json.load(f)
    #data = [data[i] for i in range(8)]

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

    # Optional chunking (eval-only): if dataset.{chunk_index, num_chunks} are
    # both set and num_chunks > 1, take a contiguous slice of `data` and
    # append _chunk{i}of{n} to the output filename. Lets eval.py recover from
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

    # Env-driven manual sharding: NUM_CHUNKS=N CHUNK_INDEX=K → slice into
    # N equal chunks and process the K-th. Use case: avoid JetEngine's
    # multi-GPU teardown race by running N single-GPU shells, each with
    # a different CHUNK_INDEX + CUDA_VISIBLE_DEVICES. Mirrors the existing
    # yaml-driven chunking (`dataset.chunk_index`/`dataset.num_chunks`).
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
    # Initialize the LLM

    block_size = config.rollout.block_size
    rollout_gpu_memory_utilization = float(
        OmegaConf.select(config, "rollout.gpu_memory_utilization", default=0.5)
    )

    # Dataset-aware prompt builder, mirroring bd3lm_rl_rollout.py's __main__ path.
    # Shapes: chat_style="evalplus_prefill" (HumanEval/MBPP) | "lcb" (LiveCodeBench)
    #       | reformat_choices + prompt_template (MMLU variants, GPQA)
    #       | per_domain_template (opdlm_train)
    #       | plain prompt_template (GSM8K/MATH500/AIME)
    #       | no template → raw question (IFEval).
    def _build_prompt_from_template(content):
        # content can be either a string (legacy: single user turn) or a list
        # of {"role", "content"} dicts (multi-turn; e.g. SDAR MMLU 5-shot
        # sends the in-context examples as 5 user/assistant rounds).
        if isinstance(content, list):
            messages = content
        else:
            messages = [{"role": "user", "content": content}]
        return tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
            enable_thinking=enable_thinking,
        )

    def get_prompt(data_i):
        q = data_i["question"]
        if ds_cfg.get("chat_style") == "evalplus_prefill":
            return build_evalplus_prompt(q, tokenizer)
        if ds_cfg.get("chat_style") == "lcb":
            return build_lcb_prompt(q, tokenizer, enable_thinking=False)
        if ds_cfg.get("reformat_choices"):
            q = reformat_choices(q)
        per_dom_tpl = ds_cfg.get("per_domain_template")
        if per_dom_tpl is not None:
            dom = data_i.get("domain", "math")
            tpl = per_dom_tpl.get(dom, "{question}")
            body = tpl.format(question=q)
        else:
            tpl = ds_cfg.get("prompt_template")
            if callable(tpl):
                # Per-row dispatch driven by data_i (e.g. MathBench, LMB-Hard).
                # All prompt definitions live in eval_utils.DATASET_CONFIGS,
                # not in this file. Signature: tpl(data_i, question) -> str.
                body = tpl(data_i, q)
            elif tpl is None:
                body = q
            else:
                body = tpl.format(question=q)
        return _build_prompt_from_template(body)
    






    # initialization
    generation_prompts = []
    index_list = []
    for i in range(num):
        prompt_text = get_prompt(data[i])
        generation_prompts = generation_prompts + [prompt_text] * k_sample
        index_list = index_list + [i] * k_sample
        data[i]["full_output"] = []
        data[i]["step_map"] = []
        data[i]["extracted_output"] = []
        data[i]["response_length"] = []
        data[i]["prompt"] = prompt_text
    




    # --------------------------- 1. shuffle --------------------------
    cprint("start generation...", "green")

    all_prompts = generation_prompts
    N = len(all_prompts)

    shuffled_idx     = list(range(N))
    random.shuffle(shuffled_idx)
    shuffled_prompts = [all_prompts[i] for i in shuffled_idx]


    import torch, math
    print(f"[preflight] CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}")
    print(f"[preflight] parent sees torch.cuda.device_count()={torch.cuda.device_count()}")
    print(f"[preflight] rollout.gpu_memory_utilization={rollout_gpu_memory_utilization}")

    cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cvd:
        visible_gpus = [x.strip() for x in cvd.split(",") if x.strip() != ""]
        device_ids = [int(x) for x in visible_gpus]         
    else:
        device_ids = list(range(torch.cuda.device_count()))
    
    gpu_num = len(device_ids)     
    tp = int(config.rollout.tensor_parallel_size)
    assert gpu_num >= tp, f"Visible GPUs ({gpu_num}) < tensor_parallel_size ({tp})."
    assert gpu_num >= 1, "No GPU visible"
    if tp > 1:
        ngroups = 1
    else:
        ngroups = max(1, gpu_num // max(1, tp))
    
    groups = [ device_ids[i*tp : (i+1)*tp] for i in range(ngroups) ]

    sampling_kwargs = dict(
        temperature          = config.rollout.temperature,
        topk                 = config.rollout.top_k,
        topp                 = config.rollout.top_p,
        max_tokens           = config.rollout.max_token,
        remasking_strategy   = config.rollout.remasking_strategy,
        block_length         = block_size,
        denoising_steps      = config.rollout.denoising_steps_per_block,
        dynamic_threshold    = config.rollout.dynamic_threshold,
        stop_words           = [151645],  # <|im_end|>: stop at teacher's EOS
    )
    max_active_local = config.rollout.max_active

    def _chunk_by_groups(lst, ng):
        L = len(lst)
        if ng <= 1: return [lst]
        chunk_size = math.ceil(L / ng)
        return [ lst[i*chunk_size : min((i+1)*chunk_size, L)] for i in range(ng) ]

    prompt_chunks = _chunk_by_groups(shuffled_prompts, ngroups)
    index_chunks  = _chunk_by_groups(shuffled_idx,     ngroups)

    for a, b in zip(prompt_chunks, index_chunks):
        assert len(a) == len(b)

    seq_pairs = []

    base_port = int(config.rollout.get("base_port", 29000))

    if ngroups == 1:
        from jetengine_ext.llm import LLM
        from jetengine_ext.sampling_params import SamplingParams

        os.environ["JETENGINE_PORT"] = str(base_port)
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, groups[0]))
        import torch
        torch.cuda.set_device(0)

        if config.rollout.tensor_parallel_size > 1:
            enforce_eager = False
        else:
            enforce_eager = True
        llm = LLM(
            model_path,
            enforce_eager=enforce_eager,
            tensor_parallel_size=config.rollout.tensor_parallel_size,
            mask_token_id=151669,   # Optional: only needed for masked/diffusion models
            block_length=block_size,
            gpu_memory_utilization=rollout_gpu_memory_utilization,
        )
        _llm = llm

        # Set sampling/generation parameters
        sampling_params = SamplingParams(
            temperature=config.rollout.temperature,
            topk=config.rollout.top_k,
            topp=config.rollout.top_p,
            max_tokens=config.rollout.max_token,
            remasking_strategy=config.rollout.remasking_strategy,
            block_length=block_size,
            denoising_steps=config.rollout.denoising_steps_per_block,
            dynamic_threshold=config.rollout.dynamic_threshold,
            stop_words=[151645],  # <|im_end|>: stop at teacher's EOS
        )
        try:
            outputs = llm.generate_streaming(prompt_chunks[0], sampling_params, max_active=config.rollout.max_active)
            for j, o in enumerate(outputs):
                seq_pairs.append( (
                    index_chunks[0][j],
                    o["text"],
                    o.get("first_unmask_times", None)
                ) )
        finally:
            _cleanup()
    else:
        ctx = mp.get_context("spawn")
        enforce_eager_local = False if tp > 1 else True

        store_ports = [_find_free_port() for _ in range(ngroups)]

        out_q = ctx.Queue()
        procs = []
        for g in range(ngroups):
            if len(prompt_chunks[g]) == 0:
                continue
            args = (
                model_path, tp, block_size, sampling_kwargs, rollout_gpu_memory_utilization, groups[g],
                prompt_chunks[g], index_chunks[g],
                enforce_eager_local, max_active_local, store_ports[g],
            )
            p = ctx.Process(target=_llm_worker_entry, args=(args, out_q), daemon=False)
            p.start()
            procs.append(p)
            _child_ps.append(p)  

        import queue, time

        results_needed = len(procs)
        results_got = 0

        while results_got < results_needed:
            try:
                kind, payload = out_q.get(timeout=1800)
            except queue.Empty:
                dead = [p for p in procs if not p.is_alive()]
                if dead:
                    for p in dead:
                        print(f"[parent] worker pid={p.pid} exitcode={p.exitcode} (no result)", flush=True)
                    for p in procs:
                        if p.is_alive():
                            p.terminate()
                    for p in procs:
                        p.join(timeout=5)
                    raise RuntimeError("Some workers died without returning results. See logs above.")
                continue

            if kind == "ok":
                seq_pairs.extend(payload)
                results_got += 1
            else:  # "err"
                print(f"[parent] worker error on port {payload['port']} pid {payload['pid']}:\n{payload['traceback']}", flush=True)
                for p in procs:
                    if p.is_alive():
                        p.terminate()
                for p in procs:
                    p.join(timeout=5)
                raise RuntimeError("Worker failed. See traceback above.")

        for p in procs:
            p.join()


    # ------------------- 3. restore original order -------------------


    restored_outputs = [None] * N
    restored_steps   = [None] * N

    for item in seq_pairs:
        if len(item) == 2:
            gi, text = item
            steps = None
        else:
            gi, text, steps = item
        restored_outputs[gi] = text
        restored_steps[gi]   = steps


    for i in range(N):
        if restored_outputs[i] is None:
            restored_outputs[i] = ""
        if restored_steps[i] is None:
            restored_steps[i] = ""

    cprint("generation job done!", "green")






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

    # try:
    #     print("Prompt:\n", shuffled_prompts[0])
    #     print("Response:\n", restored_outputs[0])
    # except Exception as e:
    #     print(f"Error printing sample output: {e}")

    # process outputs — domain dispatch (math / mc / code) via extract_answer.
    # Per-sample data_i["domain"] > ds_cfg["domain"] > "math". ds_cfg["extract"]
    # still wins if a dataset wants a custom extractor (e.g. BBH, HellaSwag).
    for i, full_output in enumerate(restored_outputs):
        index_i = index_list[i]
        extracted_output = extract_answer(
            full_output, data_i=data[index_i], ds_cfg=ds_cfg,
        )
        data[index_i]["full_output"].append(full_output)
        step_map_i = restored_steps[i] if restored_steps[i] is not None else []
        data[index_i]["step_map"].append(step_map_i)
        data[index_i]["extracted_output"].append(extracted_output)
        data[index_i]["response_length"].append(response_length[i])

    # output the data
    if num_node > 1:
        output_file_name = "../" + project_name + f"/temp_data/outputs-{node_index}-" + outputs_name + ".json"
    else:
        output_file_name = "../" + project_name + "/temp_data/outputs-" + outputs_name + ".json"
    os.makedirs(os.path.dirname(output_file_name), exist_ok=True)
    with open(output_file_name, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


