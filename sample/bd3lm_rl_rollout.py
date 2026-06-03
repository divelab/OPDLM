import os as _os
_os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

# Use /dev/shm for fast triton cache; fall back to TRITON_CACHE_DIR or /tmp if not writable.
# Per-rank subdir avoids torch TritonBundler races (os.replace ENOTEMPTY when
# multiple ranks populate the same target dir concurrently).
# Per-job subdir ($SLURM_JOB_ID) avoids races between multiple slurm jobs
# co-tenant on the same node (each job has rank0=>same path otherwise).
_local_rank = _os.environ.get("LOCAL_RANK", _os.environ.get("SLURM_LOCALID", "0"))
_job_id = _os.environ.get("SLURM_JOB_ID", str(_os.getpid()))
_cache_root = f"/dev/shm/torch_cache_{_job_id}/rank{_local_rank}"
try:
    _os.makedirs(_cache_root, exist_ok=True)
    _test_file = _os.path.join(_cache_root, f".write_test_{_os.getpid()}")
    open(_test_file, "w").close()
    _os.remove(_test_file)
except (PermissionError, OSError):
    _fallback = _os.environ.get("TRITON_CACHE_DIR", _os.path.join("/tmp", f"torch_cache_{_os.getuid()}"))
    _cache_root = _os.path.join(_fallback, f"rank{_local_rank}")
    _os.makedirs(_cache_root, exist_ok=True)
_os.environ["TORCH_EXTENSIONS_DIR"] = _os.path.join(_cache_root, "torch_extensions")
_os.environ["TRITON_CACHE_DIR"]      = _os.path.join(_cache_root, "triton")
_os.environ["XDG_CACHE_HOME"]        = _cache_root
_os.environ.setdefault("CUDA_MODULE_LOADING", "LAZY")

import os
import re
import sys
import copy
import json
import math
import random
import numpy as np
import torch

# Seed from parent process for reproducibility
_seed_str = os.environ.get("TRACERL_SEED")
if _seed_str is not None:
    _seed = int(_seed_str)
    random.seed(_seed)
    np.random.seed(_seed)
    torch.manual_seed(_seed)
import torch.nn.functional as F
import torch.multiprocessing as mp
from termcolor import cprint

from omegaconf import DictConfig, ListConfig, OmegaConf

# Import eval_utils for dataset-specific prompts, extraction, and gen settings
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'reward'))
from eval_utils import DATASET_CONFIGS, reformat_choices, build_evalplus_prompt, build_lcb_prompt

# Domain-based extraction dispatch (math / mc / code).
# Per-sample `data_i["domain"]` > ds_cfg["domain"] > default "math".
from domain_reward import extract_answer

import transformers
from transformers import AutoTokenizer, AutoModelForMaskedLM
from transformers.cache_utils import DynamicCache
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.modeling_attn_mask_utils import _prepare_4d_attention_mask
from torch import nn


# ══════════════════════════════════════════════════════════════════════
# A2DQwen3 inline classes (transformers 4.52 compat)
# NOTE: Only the Config is registered at module level. The Model classes
# are defined lazily via _register_a2d_model_classes() because defining
# a class that inherits from Qwen3ForCausalLM at module level poisons
# the CUDA driver state in multiprocessing-spawn children, causing all
# workers to land on the same GPU.
# ══════════════════════════════════════════════════════════════════════
class A2DQwen3Config(transformers.Qwen3Config):
    model_type = "a2d-qwen3"

transformers.AutoConfig.register("a2d-qwen3", A2DQwen3Config)

_a2d_model_registered = False

def _register_a2d_model_classes():
    global _a2d_model_registered
    if _a2d_model_registered:
        return
    _a2d_model_registered = True

    class A2DQwen3Model(transformers.Qwen3Model):
        def forward(self, input_ids=None, attention_mask=None, position_ids=None,
                    past_key_values=None, inputs_embeds=None, use_cache=None,
                    cache_position=None, **kwargs):
            if (input_ids is None) ^ (inputs_embeds is not None):
                raise ValueError()
            if inputs_embeds is None:
                inputs_embeds = self.embed_tokens(input_ids)
            if use_cache and past_key_values is None:
                past_key_values = DynamicCache()
            if cache_position is None:
                past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
                cache_position = torch.arange(past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device)
            if position_ids is None:
                position_ids = cache_position.unsqueeze(0)
            # Bidirectional (padding-only) mask instead of causal mask
            if isinstance(attention_mask, dict):
                causal_mask_mapping = attention_mask
            else:
                if not (isinstance(attention_mask, torch.Tensor) and attention_mask.ndim == 4):
                    if attention_mask is None:
                        attention_mask = torch.ones(inputs_embeds.shape[:2],
                                                    device=inputs_embeds.device, dtype=torch.long)
                    attention_mask = _prepare_4d_attention_mask(attention_mask, self.dtype)
                causal_mask_mapping = {"full_attention": attention_mask}
                if hasattr(self, 'has_sliding_layers') and self.has_sliding_layers:
                    causal_mask_mapping["sliding_attention"] = attention_mask
            hidden_states = inputs_embeds
            position_embeddings = self.rotary_emb(hidden_states, position_ids)
            for layer in self.layers[:self.config.num_hidden_layers]:
                layer_mask = causal_mask_mapping.get(
                    getattr(layer, 'attention_type', 'full_attention'), attention_mask)
                out = layer(hidden_states, attention_mask=layer_mask, position_ids=position_ids,
                           past_key_value=past_key_values, use_cache=use_cache,
                           cache_position=cache_position, position_embeddings=position_embeddings, **kwargs)
                hidden_states = out[0] if isinstance(out, tuple) else out
            hidden_states = self.norm(hidden_states)
            return BaseModelOutputWithPast(last_hidden_state=hidden_states, past_key_values=past_key_values if use_cache else None)

    class A2DQwen3LMHeadModel(transformers.Qwen3ForCausalLM):
        config_class = A2DQwen3Config
        def __init__(self, config):
            transformers.Qwen3PreTrainedModel.__init__(self, config)
            self.model = A2DQwen3Model(config)
            self.vocab_size = config.vocab_size
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
            self.post_init()

    transformers.AutoModel.register(A2DQwen3Config, A2DQwen3LMHeadModel)
    transformers.AutoModelForMaskedLM.register(A2DQwen3Config, A2DQwen3LMHeadModel)


# ══════════════════════════════════════════════════════════════════════
# BD3LM sampling (inlined from dllm)
# ══════════════════════════════════════════════════════════════════════

def _add_gumbel_noise(logits, temperature):
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (-torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise


def _get_num_transfer_tokens(mask_index, steps):
    mask_num = mask_index.sum(dim=1, keepdim=True)
    num_transfer = torch.zeros(mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64)
    for i in range(mask_num.size(0)):
        mn = mask_num[i, 0].clone()
        for j, (t_val, s_val) in enumerate(zip(range(steps, 0, -1), range(steps - 1, -1, -1))):
            s = s_val / steps
            t = t_val / steps
            prob = 1.0 - (s / t)  # linear schedule reverse_mask_prob
            x = mn.to(torch.float64) * prob
            n = torch.round(x).to(torch.int64)
            n = torch.minimum(n, mn.to(torch.int64))
            num_transfer[i, j] = n
            mn -= n
            if mn.item() == 0:
                break
    rows = []
    max_len = 0
    for i in range(num_transfer.size(0)):
        nz = num_transfer[i][num_transfer[i] > 0]
        rows.append(nz)
        max_len = max(max_len, nz.numel())
    if max_len == 0:
        return torch.zeros(num_transfer.size(0), 1, device=mask_index.device, dtype=torch.int64)
    padded = []
    for r in rows:
        if r.numel() < max_len:
            r = torch.cat([r, torch.zeros(max_len - r.numel(), dtype=r.dtype, device=r.device)])
        padded.append(r)
    return torch.stack(padded, dim=0)


def _prepare_for_sampling(x, block_size, pad_token_id):
    B, T = x.shape
    device = x.device
    valid = x != pad_token_id
    pos_raw = torch.cumsum(valid.to(torch.long), dim=-1)
    logical_pos = pos_raw - 1
    position_ids = torch.where(valid, logical_pos, torch.zeros_like(logical_pos)).to(device=device, dtype=torch.long)
    pos = torch.arange(T, device=device)
    block_ids = torch.div(pos, block_size, rounding_mode="floor").view(1, T).expand(B, -1)
    block_ids = torch.where(valid, block_ids, torch.full_like(block_ids, -1))
    bid_q = block_ids.view(B, 1, T, 1)
    bid_k = block_ids.view(B, 1, 1, T)
    valid_q = bid_q >= 0
    valid_k = bid_k >= 0
    base_mask = bid_k <= bid_q
    attn_mask = base_mask & valid_q & valid_k
    return attn_mask, position_ids


def _diffusion_step_block(logits, x_block, mask_block, num_transfer_step, temperature, remasking):
    B, L, _ = logits.shape
    if not mask_block.any():
        return x_block
    logits_with_noise = _add_gumbel_noise(logits, temperature=temperature)
    x0 = torch.argmax(logits_with_noise, dim=-1)
    if remasking in ("low_confidence", "low_confidence_dynamic", "low_confidence_static"):
        p = F.softmax(logits.float(), dim=-1)
        x0_p = torch.gather(p, dim=-1, index=x0.unsqueeze(-1)).squeeze(-1)
    elif remasking == "random":
        x0_p = torch.rand((B, L), device=x0.device)
    else:
        raise NotImplementedError(remasking)
    x0 = torch.where(mask_block, x0, x_block)
    confidence = torch.where(mask_block, x0_p, torch.full_like(x0_p, -float("inf")))
    transfer = torch.zeros_like(x0, dtype=torch.bool)
    for j in range(B):
        k = int(num_transfer_step[j].item())
        if k <= 0:
            continue
        valid_count = (confidence[j] > -float("inf")).sum().item()
        if valid_count == 0:
            continue
        k = min(k, valid_count)
        _, sel = torch.topk(confidence[j], k)
        transfer[j, sel] = True
    x_block_new = x_block.clone()
    x_block_new[transfer] = x0[transfer]
    return x_block_new


def _truncate_kv_cache(kv_cache, target_len):
    """Truncate a DynamicCache back to target_len entries (in-place)."""
    for layer_idx in range(len(kv_cache.key_cache)):
        kv_cache.key_cache[layer_idx] = kv_cache.key_cache[layer_idx][:, :, :target_len, :]
        kv_cache.value_cache[layer_idx] = kv_cache.value_cache[layer_idx][:, :, :target_len, :]


@torch.no_grad()
def bd3lm_sample(model, tokenizer, inputs, max_new_tokens, block_size, steps,
                 temperature, remasking):
    """
    KV-cache-optimized BD3LM sampling.

    Instead of recomputing the full prefix from scratch for each new block (O(n^2)),
    this extends the KV cache incrementally (O(n)).

    For block_size=1 with 1 step/block, uses a merged forward trick: the previous
    block's actual token and the current [MASK] are forwarded together as a 2-token
    input. This halves the number of forward calls (n+1 instead of 2n).

    For block_size>1 or multi-step, the general path uses incremental cache extension
    with copy.deepcopy for the diffusion loop within each block.
    """
    mask_id = tokenizer.mask_token_id
    pad_id = tokenizer.pad_token_id
    eos_id = tokenizer.eos_token_id
    device = model.device

    if isinstance(inputs[0], list):
        inputs = [torch.as_tensor(p, dtype=torch.long, device=device) for p in inputs]

    prompt_lens = [p.shape[0] for p in inputs]
    B = len(inputs)
    max_prompt_len = max(prompt_lens)
    padded_prompt_len = ((max_prompt_len + block_size - 1) // block_size) * block_size

    x = torch.full((B, padded_prompt_len), pad_id, dtype=torch.long, device=device)
    for b, p in enumerate(inputs):
        L = prompt_lens[b]
        offset = padded_prompt_len - L
        x[b, offset:offset + L] = p

    done = torch.zeros((B,), dtype=torch.bool, device=device)
    num_blocks = math.ceil(max_new_tokens / block_size)
    steps_per_block = math.ceil(steps / num_blocks)
    generated = 0

    # Step 1: Forward pass on full prompt to build initial KV cache
    # Use inner model (skip lm_head) to avoid allocating [B, T, vocab] logits
    prefix_attn, prefix_pos = _prepare_for_sampling(x, block_size, pad_id)
    out_prefix = model.model(x, attention_mask=prefix_attn, position_ids=prefix_pos, use_cache=True)
    prefix_kv = out_prefix.past_key_values

    # Pre-compute valid mask for block_size=1 fast path
    valid_prefix = (x != pad_id)  # [B, padded_prompt_len]
    valid_counts = valid_prefix.sum(dim=1)  # [B]
    pending_token = None  # for merged forward trick

    # Step 2: Block-by-block generation with incremental cache extension
    for b_idx in range(num_blocks):
        if done.all():
            break

        T_prefix = x.shape[1]
        cur_block_len = min(block_size, max_new_tokens - generated)
        if cur_block_len <= 0:
            break

        new_block = torch.full((B, cur_block_len), mask_id, dtype=torch.long, device=device)
        x = torch.cat([x, new_block], dim=1)
        T_total = x.shape[1]

        if block_size == 1:
            # ── Fast path: block_size=1, 1 step ──
            # Merge cache extension + [MASK] prediction into single forward.
            past_len = prefix_kv.get_seq_length()
            mask_input = torch.full((B, 1), mask_id, dtype=torch.long, device=device)

            if pending_token is not None:
                # Merged forward: [prev_actual, MASK] as 2 tokens
                combined = torch.cat([pending_token, mask_input], dim=1)  # [B, 2]
                attn_block = torch.ones(B, 1, 2, past_len + 2, dtype=torch.bool, device=device)
                attn_block[:, :, :, :padded_prompt_len] = valid_prefix.view(B, 1, 1, padded_prompt_len)
                attn_block[:, :, 0, past_len + 1] = False  # prev_actual can't see MASK
                pos_block = torch.stack([
                    valid_counts + generated - 1,
                    valid_counts + generated,
                ], dim=1)
                out = model(combined, attention_mask=attn_block, position_ids=pos_block,
                            past_key_values=prefix_kv, use_cache=True)
                cond_logits = out.logits[:, 1:2, :]
                _truncate_kv_cache(prefix_kv, past_len + 1)  # keep prev_actual, remove MASK
            else:
                # Block 0: just forward [MASK]
                attn_block = torch.ones(B, 1, 1, past_len + 1, dtype=torch.bool, device=device)
                attn_block[:, :, :, :padded_prompt_len] = valid_prefix.view(B, 1, 1, padded_prompt_len)
                pos_block = (valid_counts + generated).unsqueeze(1)
                cond_logits = model(
                    mask_input, attention_mask=attn_block, position_ids=pos_block,
                    past_key_values=prefix_kv, use_cache=True,
                ).logits
                _truncate_kv_cache(prefix_kv, past_len)  # remove MASK

            # Sample token
            x_block = x[:, T_prefix:T_total]
            mask_block = x_block == mask_id
            num_transfer_tokens = _get_num_transfer_tokens(mask_block, 1)
            x_block_updated = _diffusion_step_block(
                logits=cond_logits, x_block=x_block, mask_block=mask_block,
                num_transfer_step=num_transfer_tokens[:, 0],
                temperature=temperature, remasking=remasking,
            )
            x[:, T_prefix:T_total] = x_block_updated
            pending_token = x[:, T_prefix:T_total].clone()
        else:
            # ── General path: block_size>1 or multi-step ──
            full_attn, full_pos = _prepare_for_sampling(x, block_size, pad_id)
            attn_block = full_attn[:, :, T_prefix:T_total, :]
            pos_block = full_pos[:, T_prefix:T_total]

            block_mask_index = x[:, -cur_block_len:] == mask_id
            num_transfer_tokens = _get_num_transfer_tokens(block_mask_index, steps_per_block)
            effective_steps = num_transfer_tokens.size(1)

            # Diffusion loop (uses accumulated prefix_kv, deepcopy per step)
            for i_step in range(effective_steps):
                x_block = x[:, T_prefix:T_total]
                mask_block = x_block == mask_id
                if not mask_block.any():
                    break
                cond_logits = model(
                    x_block, attention_mask=attn_block, position_ids=pos_block,
                    past_key_values=copy.deepcopy(prefix_kv), use_cache=False,
                ).logits
                x_block_updated = _diffusion_step_block(
                    logits=cond_logits, x_block=x_block, mask_block=mask_block,
                    num_transfer_step=num_transfer_tokens[:, i_step],
                    temperature=temperature, remasking=remasking,
                )
                x[:, T_prefix:T_total] = x_block_updated

            # Extend prefix KV cache with finalized block tokens
            finalized = x[:, T_prefix:T_total]
            out_ext = model(
                finalized, attention_mask=attn_block, position_ids=pos_block,
                past_key_values=prefix_kv, use_cache=True,
            )
            prefix_kv = out_ext.past_key_values

        # EOS check
        if eos_id is not None:
            eos_in_block = (x[:, T_prefix:T_total] == eos_id).any(dim=1)
            done = done | eos_in_block
        generated += cur_block_len

    # Extend cache with final pending token (block_size=1 merged path)
    if pending_token is not None:
        past_len = prefix_kv.get_seq_length()
        attn_final = torch.ones(B, 1, 1, past_len + 1, dtype=torch.bool, device=device)
        attn_final[:, :, :, :padded_prompt_len] = valid_prefix.view(B, 1, 1, padded_prompt_len)
        pos_final = (valid_counts + generated - 1).unsqueeze(1)
        model(pending_token, attention_mask=attn_final, position_ids=pos_final,
              past_key_values=prefix_kv, use_cache=True)

    return x, padded_prompt_len


# ══════════════════════════════════════════════════════════════════════
# Helpers (same as mdlm_rl_rollout.py)
# ══════════════════════════════════════════════════════════════════════

def get_config():
    cli_conf = OmegaConf.from_cli()
    yaml_conf = OmegaConf.load(cli_conf.config)
    conf = OmegaConf.merge(yaml_conf, cli_conf)
    return conf


def get_prompt(data_i):
    # Placeholder — overridden after tokenizer is loaded
    raise RuntimeError("get_prompt called before tokenizer initialization")


def extract_final_boxed_answer(s: str):
    tag = r'\boxed{'
    start = s.rfind(tag)
    if start == -1:
        return "Can not extract the answer!"
    i = start + len(tag)
    depth = 1
    buf = []
    while i < len(s) and depth:
        ch = s[i]
        if ch == '{':
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0:
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


# math-verify extraction (verl-style): unified extraction for all math datasets
from math_verify import parse as mv_parse, LatexExtractionConfig, ExprExtractionConfig

def extract_math_verify(s: str):
    """Extract answer using math-verify (verl-style). Works for both GSM8K and MATH."""
    try:
        result = mv_parse(s, extraction_config=[ExprExtractionConfig(), LatexExtractionConfig()])
        if result and len(result) > 1:
            return str(result[1])
        elif result:
            return str(result[0])
    except Exception:
        pass
    return "Can not extract the answer!"


def random_select(data_list, random_k):
    return random.sample(data_list, random_k)


# ══════════════════════════════════════════════════════════════════════
# PyTorch fallback helpers (block_size > 1)
# ══════════════════════════════════════════════════════════════════════

def _decode_batch(output_ids, padded_prompt_len, prompt_lens, tokenizer):
    """Decode a batch of output_ids into response texts."""
    mask_id = tokenizer.mask_token_id
    pad_id = tokenizer.pad_token_id
    eos_id = tokenizer.eos_token_id
    results = []
    for j in range(output_ids.shape[0]):
        response_ids = output_ids[j, padded_prompt_len:].tolist()
        cleaned = []
        for tid in response_ids:
            if tid == eos_id or tid == pad_id or tid == mask_id:
                break
            cleaned.append(tid)
        results.append(tokenizer.decode(cleaned, skip_special_tokens=False))
    return results


def _gpu_worker(rank, model_path, prompt_ids_chunk, sample_kwargs, result_dict):
    """Worker function for multi-GPU PyTorch sampling (block_size > 1 fallback)."""
    _register_a2d_model_classes()
    torch.cuda.set_device(rank)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForMaskedLM.from_pretrained(
        model_path, trust_remote_code=False, torch_dtype=torch.bfloat16,
    ).to(f"cuda:{rank}").eval()

    batch_size = sample_kwargs.pop("batch_size", 8)
    N = len(prompt_ids_chunk)
    outputs = [None] * N

    for start_idx in range(0, N, batch_size):
        end_idx = min(start_idx + batch_size, N)
        batch_inputs = [torch.tensor(ids, dtype=torch.long) for ids in prompt_ids_chunk[start_idx:end_idx]]
        batch_prompt_lens = [len(ids) for ids in prompt_ids_chunk[start_idx:end_idx]]

        output_ids, padded_prompt_len = bd3lm_sample(
            model, tokenizer, batch_inputs, **sample_kwargs,
        )

        texts = _decode_batch(output_ids, padded_prompt_len, batch_prompt_lens, tokenizer)
        for j, text in enumerate(texts):
            outputs[start_idx + j] = text

        if rank == 0:
            cprint(f"  [GPU {rank}] Generated {min(end_idx, N)}/{N} samples...", "cyan")

    result_dict[rank] = outputs


# ══════════════════════════════════════════════════════════════════════
# JetEngine helpers
# ══════════════════════════════════════════════════════════════════════

import socket
import signal
import atexit

def _find_free_port():
    s = socket.socket(); s.bind(('', 0))
    p = s.getsockname()[1]; s.close()
    return p

def _patch_dist_port(port: int):
    import torch.distributed as _dist
    _real_init = _dist.init_process_group
    def _wrapped(backend, init_method=None, *args, **kwargs):
        if isinstance(init_method, str) and init_method.startswith("tcp://localhost:2333"):
            init_method = f"tcp://127.0.0.1:{port}"
        return _real_init(backend, init_method, *args, **kwargs)
    _dist.init_process_group = _wrapped

def _patch_safe_destroy():
    import torch.distributed as dist
    _real_destroy = dist.destroy_process_group
    def _safe_destroy(group=None):
        try:
            if not dist.is_initialized():
                return
            _real_destroy(group)
        except AssertionError:
            pass
    dist.destroy_process_group = _safe_destroy


def _je_worker_run(args):
    """JetEngine worker for multi-GPU group."""
    (model_path, tp, block_size, sampling_kwargs, gpu_memory_utilization,
     vis_ids, prompts_slice, indices_slice, enforce_eager, max_active, store_port,
     arm_shift, max_tokens_slice) = args

    os.environ.setdefault("TORCH_NCCL_BLOCKING_WAIT", "1")
    os.environ.setdefault("TORCH_NCCL_ASYNC_ERROR_HANDLING", "1")
    os.environ.pop("NCCL_BLOCKING_WAIT", None)
    os.environ.pop("NCCL_ASYNC_ERROR_HANDLING", None)
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, vis_ids))
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(store_port)
    os.environ["JETENGINE_PORT"] = str(store_port)
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    # Create per-worker sitecustomize.py for dist port patching in child processes
    patch_dir = f"/tmp/je_site_{store_port}"
    os.makedirs(patch_dir, exist_ok=True)
    patch_file = os.path.join(patch_dir, "sitecustomize.py")
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

    import torch
    import torch.distributed as dist
    _patch_dist_port(store_port)
    _patch_safe_destroy()
    _register_a2d_model_classes()
    torch.cuda.set_device(0)

    from jetengine_ext.llm import LLM
    from jetengine_ext.sampling_params import SamplingParams

    llm = LLM(model_path, enforce_eager=enforce_eager, tensor_parallel_size=tp,
              mask_token_id=151669, block_length=block_size,
              gpu_memory_utilization=gpu_memory_utilization,
              arm_shift=arm_shift)
    if max_tokens_slice is not None:
        sp_arg = [SamplingParams(**{**sampling_kwargs, "max_tokens": mt}) for mt in max_tokens_slice]
    else:
        sp_arg = SamplingParams(**sampling_kwargs)
    outs = llm.generate_streaming(prompts_slice, sp_arg, max_active=max_active)

    triples = []
    for j, o in enumerate(outs):
        triples.append((indices_slice[j], o["text"], o.get("first_unmask_times", None), len(o["token_ids"])))

    try:
        if hasattr(llm, "shutdown"):
            llm.shutdown()
    except Exception:
        pass
    return triples


def _je_worker_entry(args, out_q):
    import traceback
    # Per-worker cache to avoid race conditions in triton/inductor
    _worker_cache = os.path.join(os.environ.get("TRITON_CACHE_DIR", "/tmp"), f"worker_{os.getpid()}")
    os.makedirs(_worker_cache, exist_ok=True)
    os.environ["TRITON_CACHE_DIR"] = _worker_cache
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = _worker_cache
    try:
        res = _je_worker_run(args)
        out_q.put(("ok", res))
    except Exception:
        out_q.put(("err", {
            "pid": os.getpid(),
            "port": args[10],  # store_port position in worker args tuple
            "traceback": traceback.format_exc(),
        }))


# ══════════════════════════════════════════════════════════════════════
# Importable API (for in-process use from rl.py under accelerate launch)
# ══════════════════════════════════════════════════════════════════════

def init_jetengine(model_path, config):
    """Initialize JetEngine LLM on the current GPU. Call once per rank.

    Returns (llm, tokenizer).
    """
    from jetengine_ext.llm import LLM
    _register_a2d_model_classes()

    block_size_cfg = int(config.rollout.block_size)
    gpu_memory_utilization = float(config.rollout.get("gpu_memory_utilization", 0.8))
    arm_shift = bool(getattr(config.training, "student_arm_shift", False))
    tp = int(config.rollout.tensor_parallel_size)
    enforce_eager = (tp == 1)

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    llm = LLM(
        model_path,
        enforce_eager=enforce_eager,
        tensor_parallel_size=tp,
        mask_token_id=151669,
        block_length=block_size_cfg,
        gpu_memory_utilization=gpu_memory_utilization,
        arm_shift=arm_shift,
    )
    return llm, tokenizer


def sleep_jetengine(llm):
    """Free all JetEngine GPU memory (params + KV cache + graphs) before training."""
    llm.sleep()


def reload_jetengine_weights(llm, checkpoint_path):
    """Reload JetEngine weights + KV cache + graphs from checkpoint after training.

    Loads safetensors directly to bypass DeepSpeed ZeRO-3 hooks that would
    create empty sharded params instead of full weights.
    """
    from jetengine_ext.utils.loader import load_model as je_load_model
    _register_a2d_model_classes()
    # Recreate model and load weights directly from safetensors (bypasses DeepSpeed)
    llm.wake_up_from_path(checkpoint_path)
    torch.cuda.empty_cache()


def _build_get_prompt(tokenizer, ds_cfg, enable_thinking):
    """Build a get_prompt closure for a given tokenizer and dataset config."""
    def _build_prompt_from_template(content):
        # content can be a string (single user turn — most datasets) or a
        # list of {"role","content"} dicts (multi-turn; e.g. SDAR MMLU 5-shot
        # or TriviaQA 1-shot send the in-context examples as separate
        # USER/ASSISTANT rounds so the model sees the test query as the
        # current turn, not as text appended to a long user message).
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

    def get_prompt_fn(data_i):
        q = data_i["question"]
        if ds_cfg.get("chat_style") == "evalplus_prefill":
            return build_evalplus_prompt(q, tokenizer)
        if ds_cfg.get("chat_style") == "lcb":
            # LCB: system message + pre-baked canonical user prompt, always
            # non-thinking (matches Qwen3 tech-report LCB numbers).
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
                # Per-row dispatch (MathBench, LMB-Hard) — definitions live
                # in eval_utils.DATASET_CONFIGS. Signature: tpl(data_i, q) -> str.
                body = tpl(data_i, q)
            elif tpl is None:
                body = q
            else:
                body = tpl.format(question=q)
        return _build_prompt_from_template(body)

    return get_prompt_fn


def run_rollout(llm, tokenizer, config, project_name, rank, world_size, data_override=None):
    """Run rollout generation using a persistent JetEngine LLM.

    Each rank generates its slice of prompts, then rank 0 aggregates and saves.
    Called from rl.py under accelerate launch.
    """
    import torch.distributed as dist
    from jetengine_ext.sampling_params import SamplingParams

    enable_thinking = bool(config.rollout.start_with_think)

    if config.experiment.current_epoch == 1:
        pretrained_model = config.model.pretrained_model
    else:
        pretrained_model = project_name + "/ckpt/" + config.model.optimized_name

    if config.experiment.function == "train":
        dataset = config.dataset.train_dataset
        k_sample = config.rollout.num_response_per_task
        outputs_name = "rl-" + pretrained_model.replace("/", ".") + "-" + dataset
    elif config.experiment.function == "evaluation":
        dataset = config.evaluation.eval_dataset
        k_sample = config.evaluation.num_response_per_task
        config.rollout.temperature = config.evaluation.temperature
        config.rollout.max_token = config.evaluation.max_token
        config.rollout.block_size = config.evaluation.block_size
        config.rollout.denoising_steps_per_block = config.evaluation.denoising_steps_per_block
        config.rollout.remasking_strategy = config.evaluation.remasking_strategy
        config.rollout.dynamic_threshold = config.evaluation.dynamic_threshold
        config.rollout.top_p = config.evaluation.top_p
        config.rollout.top_k = config.evaluation.top_k
        outputs_name = "eval-" + pretrained_model.replace("/", ".") + "-" + dataset

    if dataset not in DATASET_CONFIGS:
        raise ValueError(
            f"Dataset {dataset!r} is not registered in eval_utils.DATASET_CONFIGS. "
            f"Known datasets: {sorted(DATASET_CONFIGS.keys())}"
        )
    ds_cfg = DATASET_CONFIGS[dataset]

    if data_override is not None:
        data = data_override
    else:
        with open("data/" + dataset + ".json", 'r') as f:
            data = json.load(f)

        if config.experiment.function == "train":
            random_select_num = config.rollout.num_task_per_step
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
    # a different CHUNK_INDEX + CUDA_VISIBLE_DEVICES. Each writes a
    # distinct output file via the `_chunk{idx}of{num}` suffix; merge
    # after all finish. Mirrors the existing yaml-driven chunking
    # (`dataset.chunk_index`/`dataset.num_chunks`).
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
    get_prompt = _build_get_prompt(tokenizer, ds_cfg, enable_thinking)

    block_size_cfg = int(config.rollout.block_size)
    denoising_steps = int(config.rollout.denoising_steps_per_block)
    max_new_tokens = int(config.rollout.max_token)
    temperature_cfg = float(config.rollout.temperature)
    remasking_cfg = str(config.rollout.remasking_strategy)

    if config.experiment.function == "evaluation":
        max_new_tokens = int(config.evaluation.max_token)

    # Build all prompts (deterministic across ranks)
    generation_prompts = []
    index_list = []
    for i in range(num):
        prompt_text = get_prompt(data[i])
        generation_prompts += [prompt_text] * k_sample
        index_list += [i] * k_sample
        data[i]["full_output"] = []
        data[i]["step_map"] = []
        data[i]["extracted_output"] = []
        data[i]["response_length"] = []
        data[i]["prompt"] = prompt_text

    N = len(generation_prompts)
    if rank == 0:
        cprint(f"Starting BD3LM generation: {N} total samples ({num} prompts x {k_sample} responses), {world_size} ranks", "green")

    sampling_kwargs = dict(
        temperature=temperature_cfg,
        topk=int(config.rollout.top_k),
        topp=float(config.rollout.top_p),
        max_tokens=max_new_tokens,
        remasking_strategy=remasking_cfg,
        block_length=block_size_cfg,
        denoising_steps=denoising_steps,
        dynamic_threshold=float(config.rollout.dynamic_threshold),
    )

    # Per-domain max_token cap (only active if sample has a `domain` field, e.g. opdlm_train).
    # Effective per-sample max = min(scheduled max_token, per_domain_cap[domain]).
    per_dom_max_cap = dict(config.rollout.get("per_domain_max_token", {}) or {})

    def _max_tokens_for(idx):
        dom = data[idx].get("domain")
        if dom is None or dom not in per_dom_max_cap:
            return max_new_tokens
        return min(max_new_tokens, int(per_dom_max_cap[dom]))

    # Shuffle prompts deterministically
    shuffled_idx = list(range(N))
    random.shuffle(shuffled_idx)

    # Each rank takes its slice
    my_indices = shuffled_idx[rank::world_size]
    my_prompts = [generation_prompts[i] for i in my_indices]

    max_active_local = int(config.rollout.get("max_active", 256))
    if per_dom_max_cap:
        sp_list = []
        for gi in my_indices:
            mt = _max_tokens_for(index_list[gi])
            sp_list.append(SamplingParams(**{**sampling_kwargs, "max_tokens": mt}))
        outputs = llm.generate_streaming(my_prompts, sp_list, max_active=max_active_local)
    else:
        sp = SamplingParams(**sampling_kwargs)
        outputs = llm.generate_streaming(my_prompts, sp, max_active=max_active_local)

    my_results = []
    for j, o in enumerate(outputs):
        my_results.append((
            my_indices[j],
            o["text"],
            o.get("first_unmask_times", None),
            len(o["token_ids"]),
        ))

    if rank == 0:
        cprint(f"  Rank {rank}: generated {len(my_results)} samples", "cyan")

    # Gather results from all ranks to rank 0
    all_results_list = [None] * world_size
    dist.all_gather_object(all_results_list, my_results)

    if rank == 0:
        # Assemble in original order
        all_outputs = [""] * N
        all_steps = [None] * N
        all_token_lens = [0] * N
        for rank_results in all_results_list:
            for item in rank_results:
                gi, text = item[0], item[1]
                steps = item[2]
                tok_len = item[3]
                all_outputs[gi] = text
                all_steps[gi] = steps
                if tok_len is not None:
                    all_token_lens[gi] = tok_len

        cprint("Generation done!", "green")

        # Process outputs
        for i in range(N):
            full_output = all_outputs[i] if all_outputs[i] is not None else ""
            idx = index_list[i]
            extracted_output = extract_answer(
                full_output, data_i=data[idx], ds_cfg=ds_cfg,
            )
            data[idx]["full_output"].append(full_output)
            step_map_i = all_steps[i] if all_steps[i] is not None else []
            data[idx]["step_map"].append(step_map_i)
            data[idx]["extracted_output"].append(extracted_output)
            resp_len = all_token_lens[i] if all_token_lens[i] > 0 else len(tokenizer.encode(full_output, add_special_tokens=False))
            data[idx]["response_length"].append(resp_len)

        # Save output
        output_file_name = project_name + "/temp_data/outputs-" + outputs_name + ".json"
        os.makedirs(os.path.dirname(output_file_name), exist_ok=True)
        with open(output_file_name, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        # Track cumulative rollout tokens (train only)
        if config.experiment.function == "train":
            total_rollout_tokens = sum(resp_len for d in data for resp_len in d.get("response_length", []))
            rollout_token_file = project_name + "/temp_data/cumulative_rollout_tokens.txt"
            prev_rollout_tokens = 0
            if os.path.exists(rollout_token_file):
                with open(rollout_token_file) as f:
                    prev_rollout_tokens = int(f.read().strip())
            with open(rollout_token_file, "w") as f:
                f.write(str(prev_rollout_tokens + total_rollout_tokens))

        cprint(f"Saved rollout outputs to {output_file_name}", "green")

    # Sync all ranks before continuing
    dist.barrier()


# ══════════════════════════════════════════════════════════════════════
# Main (standalone subprocess mode — kept for backward compat)
# ══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    config = get_config()

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

        config.rollout.temperature = config.evaluation.temperature
        config.rollout.max_token = config.evaluation.max_token
        config.rollout.block_size = config.evaluation.block_size
        config.rollout.denoising_steps_per_block = config.evaluation.denoising_steps_per_block
        config.rollout.remasking_strategy = config.evaluation.remasking_strategy
        config.rollout.dynamic_threshold = config.evaluation.dynamic_threshold
        config.rollout.top_p = config.evaluation.top_p
        config.rollout.top_k = config.evaluation.top_k

        outputs_name = "eval-" + pretrained_model.replace("/", ".") + "-" + dataset

    # Look up dataset-specific config from eval_utils. Fail loud if the dataset
    # isn't registered — otherwise we'd silently run with the wrong prompt /
    # extraction and the user wouldn't notice.
    if dataset not in DATASET_CONFIGS:
        raise ValueError(
            f"Dataset {dataset!r} is not registered in eval_utils.DATASET_CONFIGS. "
            f"Add an entry with `domain` and `prompt_template` (use None if the "
            f"prompt is already encoded per-sample, e.g. opdlm_train). "
            f"Known datasets: {sorted(DATASET_CONFIGS.keys())}"
        )
    ds_cfg = DATASET_CONFIGS[dataset]

    # Honor ds_cfg["path"] so multiple registry entries can share one JSON
    # (e.g. ARC_C / ARC_C_sdar both read ARC_C.json).
    _ds_path = ds_cfg.get("path", dataset + ".json")
    with open("../data/" + _ds_path, 'r') as f:
        data = json.load(f)

    if config.experiment.function == "train":
        random_select_num = config.rollout.num_task_per_step
        random_select_num = min(random_select_num, len(data))
        data = random_select(data, random_select_num)

    # Optional chunking (eval-only) — see in-process path above for rationale.
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

    # Env-driven manual sharding (mirror of in-process path) — see comment
    # there for rationale. NUM_CHUNKS=N CHUNK_INDEX=K → K-th of N equal slices.
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
    cprint(f"Loading BD3LM model from {model_path}", "green")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    # Build get_prompt using tokenizer.apply_chat_template (matches dllm eval)
    def _build_prompt_from_template(content):
        """Use official chat template instead of hardcoded strings.

        Accepts either a string (single user turn) or a list of
        {"role","content"} dicts (multi-turn; see _build_get_prompt for
        the same logic).
        """
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

    # Prompt resolution (fail-loud): dataset is guaranteed registered at this
    # point. Valid shapes:
    #   - chat_style == "evalplus_prefill" (HumanEval/MBPP): route through
    #     build_evalplus_prompt which produces the evalplus-style chat
    #     prompt ending inside a ```python fence.
    #   - per_domain_template (dict) → pick by data_i["domain"] (opdlm_train)
    #   - prompt_template (format string) → apply it (GSM8K/MATH500/MMLU*)
    #   - both None → pass question as-is
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
                # Per-row dispatch (MathBench, LMB-Hard) — definitions live
                # in eval_utils.DATASET_CONFIGS. Signature: tpl(data_i, q) -> str.
                body = tpl(data_i, q)
            elif tpl is None:
                body = q
            else:
                body = tpl.format(question=q)
        return _build_prompt_from_template(body)

    block_size_cfg = int(config.rollout.block_size)
    denoising_steps = int(config.rollout.denoising_steps_per_block)
    max_new_tokens = int(config.rollout.max_token)
    temperature_cfg = float(config.rollout.temperature)
    remasking_cfg = str(config.rollout.remasking_strategy)

    if config.experiment.function == "evaluation":
        # evaluation.max_token is overridden per dataset by rl.py eval_loop
        max_new_tokens = int(config.evaluation.max_token)

    generation_prompts = []
    index_list = []
    for i in range(num):
        prompt_text = get_prompt(data[i])
        generation_prompts += [prompt_text] * k_sample
        index_list += [i] * k_sample
        data[i]["full_output"] = []
        data[i]["step_map"] = []
        data[i]["extracted_output"] = []
        data[i]["response_length"] = []
        data[i]["prompt"] = prompt_text

    N = len(generation_prompts)
    cprint(f"Starting BD3LM generation: {N} total samples ({num} prompts x {k_sample} responses)...", "green")

    # ══ JetEngine sampling ══
    cprint(f"  Using JetEngine (block_size={block_size_cfg})", "green")

    tp = int(config.rollout.tensor_parallel_size)
    if tp == 1:
        os.environ.setdefault("TORCH_NCCL_ASYNC_ERROR_HANDLING", "1")
        os.environ.setdefault("TORCH_NCCL_BLOCKING_WAIT", "1")
        os.environ.setdefault("NCCL_P2P_DISABLE", "1")
        os.environ.setdefault("NCCL_IB_DISABLE", "1")
    else:
        for k in ["NCCL_P2P_DISABLE", "NCCL_IB_DISABLE",
                   "TORCH_NCCL_BLOCKING_WAIT", "TORCH_NCCL_ASYNC_ERROR_HANDLING"]:
            os.environ.pop(k, None)

    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    gpu_memory_utilization = float(config.rollout.get("gpu_memory_utilization", 0.8))
    max_active_local = int(config.rollout.get("max_active", 256))
    base_port = int(config.rollout.get("base_port", 29000))

    cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cvd:
        device_ids = [int(x.strip()) for x in cvd.split(",") if x.strip()]
    else:
        device_ids = list(range(torch.cuda.device_count()))
    gpu_num = len(device_ids)

    if tp > 1:
        ngroups = 1
    else:
        ngroups = max(1, gpu_num // max(1, tp))
    groups = [device_ids[i*tp:(i+1)*tp] for i in range(ngroups)]

    sampling_kwargs = dict(
        temperature=temperature_cfg,
        topk=int(config.rollout.top_k),
        topp=float(config.rollout.top_p),
        max_tokens=max_new_tokens,
        remasking_strategy=remasking_cfg,
        block_length=block_size_cfg,
        denoising_steps=denoising_steps,
        dynamic_threshold=float(config.rollout.dynamic_threshold),
    )

    # Per-domain max_token cap (only active if sample has a `domain` field, e.g. opdlm_train).
    # Effective per-sample max = min(scheduled max_token, per_domain_cap[domain]).
    per_dom_max_cap = dict(config.rollout.get("per_domain_max_token", {}) or {})

    def _max_tokens_for(idx):
        dom = data[idx].get("domain")
        if dom is None or dom not in per_dom_max_cap:
            return max_new_tokens
        return min(max_new_tokens, int(per_dom_max_cap[dom]))

    # Per-prompt max_tokens aligned with generation_prompts (same length as N).
    per_prompt_max_tokens = [_max_tokens_for(index_list[i]) for i in range(N)]

    # Shuffle prompts (same as SDAR)
    shuffled_idx = list(range(N))
    random.shuffle(shuffled_idx)
    shuffled_prompts = [generation_prompts[i] for i in shuffled_idx]
    shuffled_max_tokens = [per_prompt_max_tokens[i] for i in shuffled_idx]

    def _chunk_by_groups(lst, ng):
        if ng <= 1:
            return [lst]
        cs = math.ceil(len(lst) / ng)
        return [lst[i*cs:min((i+1)*cs, len(lst))] for i in range(ng)]

    prompt_chunks = _chunk_by_groups(shuffled_prompts, ngroups)
    index_chunks = _chunk_by_groups(shuffled_idx, ngroups)
    max_tokens_chunks = _chunk_by_groups(shuffled_max_tokens, ngroups)

    cprint(f"  tp={tp}, ngroups={ngroups}, gpus={groups}, max_active={max_active_local}", "cyan")

    _llm = None
    _child_ps = []

    def _cleanup():
        global _llm
        try:
            if _llm is not None and hasattr(_llm, "shutdown"):
                _llm.shutdown()
        except Exception:
            pass
        for p in _child_ps:
            try:
                if hasattr(p, "terminate"):
                    p.terminate()
            except Exception:
                pass

    atexit.register(_cleanup)
    signal.signal(signal.SIGINT, lambda s, f: (_cleanup(), sys.exit(130)))
    signal.signal(signal.SIGTERM, lambda s, f: (_cleanup(), sys.exit(143)))

    seq_pairs = []

    arm_shift = bool(getattr(config.training, "student_arm_shift", False))

    if ngroups == 1:
        # Single group: run JetEngine in-process
        os.environ["JETENGINE_PORT"] = str(base_port)
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, groups[0]))
        _register_a2d_model_classes()
        import torch
        torch.cuda.set_device(0)

        if "MASTER_PORT" not in os.environ:
            os.environ["MASTER_ADDR"] = "127.0.0.1"
            os.environ["MASTER_PORT"] = str(_find_free_port())

        from jetengine_ext.llm import LLM
        from jetengine_ext.sampling_params import SamplingParams

        enforce_eager = (tp == 1)
        llm = LLM(model_path, enforce_eager=enforce_eager, tensor_parallel_size=tp,
                   mask_token_id=151669, block_length=block_size_cfg,
                   gpu_memory_utilization=gpu_memory_utilization,
                   arm_shift=arm_shift)
        _llm = llm

        if per_dom_max_cap:
            sp_arg = [SamplingParams(**{**sampling_kwargs, "max_tokens": mt})
                      for mt in max_tokens_chunks[0]]
        else:
            sp_arg = SamplingParams(**sampling_kwargs)
        try:
            outputs = llm.generate_streaming(prompt_chunks[0], sp_arg, max_active=max_active_local)
            for j, o in enumerate(outputs):
                seq_pairs.append((index_chunks[0][j], o["text"], o.get("first_unmask_times", None), len(o["token_ids"])))
        finally:
            _cleanup()
    else:
        # Multiple groups: spawn workers
        ctx = mp.get_context("spawn")
        enforce_eager_local = (tp == 1)
        store_ports = [base_port + g for g in range(ngroups)]

        out_q = ctx.Queue()
        procs = []
        for g in range(ngroups):
            if not prompt_chunks[g]:
                continue
            args = (model_path, tp, block_size_cfg, sampling_kwargs,
                    gpu_memory_utilization, groups[g],
                    prompt_chunks[g], index_chunks[g],
                    enforce_eager_local, max_active_local, store_ports[g],
                    arm_shift,
                    max_tokens_chunks[g] if per_dom_max_cap else None)
            p = ctx.Process(target=_je_worker_entry, args=(args, out_q), daemon=False)
            p.start()
            procs.append(p)
            _child_ps.append(p)

        import queue
        results_needed = len(procs)
        results_got = 0
        while results_got < results_needed:
            try:
                kind, payload = out_q.get(timeout=1800)
            except queue.Empty:
                dead = [p for p in procs if not p.is_alive()]
                if dead:
                    for p in procs:
                        if p.is_alive():
                            p.terminate()
                    for p in procs:
                        p.join(timeout=5)
                    raise RuntimeError("Some JetEngine workers died without returning results.")
                continue
            if kind == "ok":
                seq_pairs.extend(payload)
                results_got += 1
            else:
                cprint(f"Worker error on port {payload['port']}:\n{payload['traceback']}", "red")
                for p in procs:
                    if p.is_alive():
                        p.terminate()
                for p in procs:
                    p.join(timeout=5)
                raise RuntimeError("JetEngine worker failed. See traceback above.")
        for p in procs:
            p.join()

    # Restore original order
    all_outputs = [""] * N
    all_steps = [None] * N
    all_token_lens = [0] * N
    for item in seq_pairs:
        gi, text = item[0], item[1]
        steps = item[2] if len(item) > 2 else None
        tok_len = item[3] if len(item) > 3 else None
        all_outputs[gi] = text
        all_steps[gi] = steps
        if tok_len is not None:
            all_token_lens[gi] = tok_len

    cprint("Generation done!", "green")

    # Process outputs
    for i in range(N):
        full_output = all_outputs[i] if all_outputs[i] is not None else ""
        idx = index_list[i]
        # Domain dispatch: per-sample data_i["domain"] > ds_cfg["domain"] > math.
        # ds_cfg["extract"] override (e.g. BBH, HellaSwag) still takes precedence.
        extracted_output = extract_answer(
            full_output, data_i=data[idx], ds_cfg=ds_cfg,
        )
        data[idx]["full_output"].append(full_output)
        step_map_i = all_steps[i] if all_steps[i] is not None else []
        data[idx]["step_map"].append(step_map_i)
        data[idx]["extracted_output"].append(extracted_output)
        # Use actual completion token count from JetEngine (avoids re-tokenization mismatch)
        resp_len = all_token_lens[i] if all_token_lens[i] > 0 else len(tokenizer.encode(full_output, add_special_tokens=False))
        data[idx]["response_length"].append(resp_len)

    # Save output
    output_file_name = "../" + project_name + "/temp_data/outputs-" + outputs_name + ".json"
    os.makedirs(os.path.dirname(output_file_name), exist_ok=True)
    with open(output_file_name, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    # Track cumulative rollout tokens for FLOPs estimation (train only, exclude eval)
    if config.experiment.function == "train":
        total_rollout_tokens = sum(resp_len for d in data for resp_len in d.get("response_length", []))
        rollout_token_file = "../" + project_name + "/temp_data/cumulative_rollout_tokens.txt"
        prev_rollout_tokens = 0
        if os.path.exists(rollout_token_file):
            with open(rollout_token_file) as f:
                prev_rollout_tokens = int(f.read().strip())
        with open(rollout_token_file, "w") as f:
            f.write(str(prev_rollout_tokens + total_rollout_tokens))

    cprint(f"Saved rollout outputs to {output_file_name}", "green")
