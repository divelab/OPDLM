import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["TOKENIZERS_PARALLELISM"] = "true"
import json
import logging
import math
import random
import shutil
import time
from pathlib import Path
from typing import Union

import numpy as np
from PIL import Image
from omegaconf import OmegaConf
import wandb
from pprint import pprint
import torch
from torch.optim import AdamW

from transformers import AutoTokenizer, AutoModelForCausalLM
from accelerate import Accelerator
from accelerate.utils import DeepSpeedPlugin
from accelerate.logging import get_logger
from accelerate.utils import DistributedType, set_seed



from models import SDARForCausalLM
from train.prompting_utils import UniversalPrompting
from models.lr_schedulers import get_scheduler
from models.logging import set_verbosity_info, set_verbosity_error

from torch.utils.data import Dataset, DataLoader

SYSTEM_PROMPT_LEN = 28

from train.utils import get_config, flatten_omega_conf, AverageMeter

try:
    import apex

    is_apex_available = True
except ImportError:
    is_apex_available = False

logger = get_logger(__name__, log_level="INFO")

TRAINING_STATE_TAG = "student"


def _combine_rounds_one_state_per_block(
    per_prompt_ext, per_prompt_pm, clean_input_ids, L0, L1, block_size,
):
    """Stitch a trajectory's T denoising rounds into a single training row,
    independently sampling one round per block for the block's tail + pmask.
    The prompt and clean response are invariant across rounds.
    """
    ext = torch.stack(per_prompt_ext, dim=0)          # [T, L0+2*L1]
    pm = torch.stack(per_prompt_pm, dim=0)            # [T, L0+L1]

    tail = clean_input_ids[L0:L0 + L1].clone()
    out_pm = torch.zeros(L0 + L1, dtype=torch.bool, device=pm.device)

    for bs in range(0, L1, block_size):
        be = min(bs + block_size, L1)
        rounds = pm[:, L0 + bs:L0 + be].any(dim=1).nonzero().flatten().tolist()
        if not rounds:
            continue
        r = random.choice(rounds)
        out_pm[L0 + bs:L0 + be] = pm[r, L0 + bs:L0 + be]
        tail[bs:be] = ext[r, L0 + L1 + bs:L0 + L1 + be]

    return torch.cat([clean_input_ids, tail], dim=0), out_pm


def _combine_rounds_random_mask(
    clean_input_ids, L0, L1, block_size, mask_id,
):
    """Synthetic alternative to _combine_rounds_one_state_per_block: ignore the
    rollout trajectory and instead build one training row per prompt by
    independently random-masking each block of the clean response.
    For each block we sample a mask ratio t ~ Uniform(0,1) and Bernoulli-mask
    each position with that probability. Returns the same (extended_input_ids,
    p_mask) shapes as the trajectory-based variant so the rest of the training
    loop is unchanged.
    """
    device = clean_input_ids.device
    tail = clean_input_ids[L0:L0 + L1].clone()
    out_pm = torch.zeros(L0 + L1, dtype=torch.bool, device=device)

    for bs in range(0, L1, block_size):
        be = min(bs + block_size, L1)
        block_len = be - bs
        # Per-block independent uniform mask ratio.
        t = torch.rand((), device=device).item()
        mask = torch.rand(block_len, device=device) < t
        if not mask.any():
            # Ensure at least one masked position so the block contributes
            # gradient signal; pick a uniform random index.
            mask[torch.randint(0, block_len, (1,), device=device).item()] = True
        out_pm[L0 + bs:L0 + be] = mask
        # Apply mask to the tail (masked positions become mask_id, unmasked
        # positions stay as the clean token).
        tail_block = tail[bs:be]
        tail_block[mask] = mask_id
        tail[bs:be] = tail_block

    return torch.cat([clean_input_ids, tail], dim=0), out_pm




def _maybe_wrap_lora(model, config, project_name, adapter_resume_dir=None):
    """If config.training.lora.enabled, wrap `model` with PEFT LoRA.
    On resume (adapter_resume_dir given and exists), load the adapter from disk
    on top of `model` (which should be the base). Returns the (possibly-wrapped)
    model. Default-off: when lora.enabled is false, returns model unchanged.
    """
    lcfg = getattr(config.training, "lora", None)
    if not lcfg or not getattr(lcfg, "enabled", False):
        return model

    from peft import LoraConfig, get_peft_model, PeftModel
    if adapter_resume_dir and os.path.isdir(adapter_resume_dir):
        logger.info(f"[LoRA] resuming adapter from {adapter_resume_dir}")
        model = PeftModel.from_pretrained(model, adapter_resume_dir, is_trainable=True)
    else:
        peft_cfg = LoraConfig(
            r=int(lcfg.r),
            lora_alpha=int(lcfg.alpha),
            lora_dropout=float(lcfg.dropout),
            target_modules=list(lcfg.target_modules),
            bias=str(lcfg.bias),
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, peft_cfg)
        logger.info(f"[LoRA] wrapped student with r={lcfg.r}, alpha={lcfg.alpha}, "
                    f"target_modules={list(lcfg.target_modules)}")
    # Required when combining LoRA (frozen base) with gradient checkpointing:
    # the embedding output must require_grad=True so the checkpointed
    # recompute can backprop through the frozen base into the LoRA layers.
    # Without this, backward fails with
    #   "element 0 of tensors does not require grad and does not have a grad_fn".
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    if hasattr(model, "print_trainable_parameters"):
        model.print_trainable_parameters()
    return model


class TrainDataset(Dataset):
    def __init__(self, extended_input_ids, p_mask, tok_idx_ext, labels, reward):
        self.extended_input_ids = extended_input_ids
        self.p_mask = p_mask
        self.tok_idx_ext = tok_idx_ext
        self.labels = labels
        self.reward   = reward
        self.logp_old_tok = torch.full(
            (len(extended_input_ids), p_mask.shape[1]), 
            float('-inf')
        )

    def __len__(self):
        return len(self.extended_input_ids)

    def __getitem__(self, idx):
        return (
            idx,
            self.extended_input_ids[idx],
            self.p_mask[idx],
            self.tok_idx_ext[idx],
            self.labels[idx],
            self.reward[idx],
        )


def main():
    #########################
    # SETUP Accelerator     #
    #########################
    config = get_config()
    wandb_enabled = bool(config.wandb.get("enabled", True))

    project_name = config.experiment.project
    _lora_enabled = getattr(getattr(config.training, "lora", None), "enabled", False)
    _lora_adapter_resume = None
    if config.experiment.current_epoch == 1:
        pretrained_model = config.model.pretrained_model
    elif _lora_enabled:
        # LoRA resume: keep base = original pretrained model; load adapter
        # from the previous save's `adapter/` subdir.
        pretrained_model = config.model.pretrained_model
        _lora_adapter_resume = os.path.join(
            project_name, "ckpt", config.model.optimized_name, "adapter"
        )
    else:
        pretrained_model = project_name + "/ckpt/" + config.model.optimized_name

    # Enable TF32 on Ampere GPUs
    if config.training.enable_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

    config.experiment.logging_dir = str(Path(config.experiment.project) / "logs")
    _zero_stage = int(config.training.get("zero_stage", 3))
    _offload_opt = config.training.get("offload_optimizer_device", "cpu")
    _offload_param = config.training.get("offload_param_device", "cpu")
    _student_kwargs = dict(
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        gradient_clipping=config.training.max_grad_norm,
        zero_stage=_zero_stage,
        offload_optimizer_device=_offload_opt,
    )
    if _zero_stage == 3:
        _student_kwargs.update(
            offload_param_device=_offload_param,
            zero3_init_flag=True,
            zero3_save_16bit_model=True,
        )
    # Teacher is frozen (inference-only). Stage 2 requires an optimizer, so
    # force teacher to stage 3 regardless of student stage.
    _teacher_kwargs = dict(
        zero_stage=3,
        offload_param_device=_offload_param,
        zero3_init_flag=True,
    )
    deepspeed_plugins = {
        "student": DeepSpeedPlugin(**_student_kwargs),
        "teacher": DeepSpeedPlugin(**_teacher_kwargs),
    }
    accelerator = Accelerator(
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        mixed_precision=config.training.mixed_precision,
        log_with="wandb" if wandb_enabled else None,
        project_dir=config.experiment.logging_dir,
        split_batches=False,
        deepspeed_plugins=deepspeed_plugins,
    )

    #####################################
    # SETUP LOGGING, SEED and CONFIG    #
    #####################################
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        set_verbosity_info()
    else:
        set_verbosity_error()

    if accelerator.is_main_process and wandb_enabled:
        run_id = config.wandb.get("run_id", None) or os.getenv("WANDB_RUN_ID", None)
        if run_id is None:
            raise ValueError("WANDB_RUN_ID environment variable is not set. Please set it to the desired run ID.")

        wandb_init_kwargs = dict(
            id=run_id,
            resume="must",
        )
        wandb_config = {k: v for k, v in flatten_omega_conf(config, resolve=True)}
        wandb_config.pop("experiment.resume_from_checkpoint", None)

        wandb_project = config.wandb.get("project") or config.experiment.project
        accelerator.init_trackers(
            wandb_project,
            config=wandb_config,
            init_kwargs={"wandb": wandb_init_kwargs},
        )
        wandb.define_metric("train/current_epoch")
        wandb.define_metric("*", step_metric="train/current_epoch")

    if accelerator.is_main_process:
        os.makedirs(config.experiment.project, exist_ok=True)
        config_path = Path(config.experiment.project) / "config.yaml"
        logging.info(f"Saving config to {config_path}")
        OmegaConf.save(config, config_path)

    # If passed along, set the training seed now.
    # Prefer TRACERL_SEED (exported by rl.py as base_seed + current_epoch) so picks vary per RL step;
    # fall back to config.training.seed.
    _env_seed = os.environ.get("TRACERL_SEED")
    _step_seed = int(_env_seed) if _env_seed is not None else config.training.seed
    if _step_seed is not None:
        set_seed(_step_seed)

    #########################
    # MODELS and OPTIMIZER  #
    #########################
    logger.info("Loading models and optimizer")


    tokenizer = AutoTokenizer.from_pretrained(pretrained_model, trust_remote_code=True)
    uni_prompting = UniversalPrompting(tokenizer, max_prompt_len=config.training.max_prompt_len,
                                       max_gen_length=config.training.max_gen_length,
                                       ignore_id=-100)

    model_base = getattr(config.model, "model_base", "sdar")
    if model_base == "sdar":
        model = SDARForCausalLM.from_pretrained(pretrained_model, trust_remote_code=True, torch_dtype="auto")
    elif model_base == "bd3lm":
        # ── Inline A2DQwen3 registration (transformers 4.52 compat) ──
        import transformers as _tf
        from transformers.cache_utils import DynamicCache
        from transformers.modeling_outputs import BaseModelOutputWithPast
        from transformers.modeling_attn_mask_utils import _prepare_4d_attention_mask
        from torch import nn as _nn

        class _A2DQwen3Config(_tf.Qwen3Config):
            model_type = "a2d-qwen3"

        class _A2DQwen3Model(_tf.Qwen3Model):
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
                    _past = past_key_values.get_seq_length() if past_key_values is not None else 0
                    cache_position = torch.arange(_past, _past + inputs_embeds.shape[1], device=inputs_embeds.device)
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
                return BaseModelOutputWithPast(last_hidden_state=hidden_states,
                                               past_key_values=past_key_values if use_cache else None)

        class _A2DQwen3LMHeadModel(_tf.Qwen3ForCausalLM):
            config_class = _A2DQwen3Config
            def __init__(self, config):
                _tf.Qwen3PreTrainedModel.__init__(self, config)
                self.model = _A2DQwen3Model(config)
                self.vocab_size = config.vocab_size
                self.lm_head = _nn.Linear(config.hidden_size, config.vocab_size, bias=False)
                self.post_init()

        _tf.AutoConfig.register("a2d-qwen3", _A2DQwen3Config)
        _tf.AutoModel.register(_A2DQwen3Config, _A2DQwen3LMHeadModel)
        _tf.AutoModelForMaskedLM.register(_A2DQwen3Config, _A2DQwen3LMHeadModel)

        from transformers import AutoModelForMaskedLM
        model = AutoModelForMaskedLM.from_pretrained(pretrained_model, trust_remote_code=False, torch_dtype="auto")
    else:
        model = AutoModelForCausalLM.from_pretrained(pretrained_model, trust_remote_code=True, torch_dtype="auto")

    # calculate loss ourselves, needs logits，so aviod fuse CE
    if hasattr(model, "config"):
        model.config.fuse_cross_entropy = False
        if getattr(config.training, "student_arm_shift", False):
            model.config.arm_shift = True

    # Optionally wrap student with LoRA (default off; only the adapter is
    # trainable when enabled).
    model = _maybe_wrap_lora(model, config, project_name, adapter_resume_dir=_lora_adapter_resume)

    if config.training.gradient_checkpointing_enable:
        model.gradient_checkpointing_enable()
        if hasattr(model, "config"):
            model.config.use_cache = False
    else:
        model = model.to(accelerator.device)

    mask_id = tokenizer.mask_token_id
    if mask_id is None:
        # AR models (e.g. Qwen) have no mask token; use pad_id as placeholder
        # for data preparation (causal/GRPO forward paths don't rely on mask_id)
        mask_id = tokenizer.pad_token_id
    pad_id = tokenizer.pad_token_id

    ##################################
    #   Optimizer and LR scheduler   #
    #################################
    optimizer_config = config.optimizer.params

    # no decay on bias and layernorm and embedding
    no_decay = ["bias", "layer_norm.weight", "mlm_ln.weight", "embeddings.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if
                       p.requires_grad and not any(nd in n for nd in no_decay)],
            "weight_decay": optimizer_config.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if
                       p.requires_grad and any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]

    optimizer_type = config.optimizer.name
    if optimizer_type == "adamw":
        optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=optimizer_config.learning_rate,
            betas=(optimizer_config.beta1, optimizer_config.beta2),
            weight_decay=optimizer_config.weight_decay,
            eps=optimizer_config.epsilon,
        )
    else:
        raise ValueError(f"Optimizer {optimizer_type} not supported")




    def collapse_k_unique(lst, k: int):
        if k <= 0:
            raise ValueError("k must be > 0")
        uniq = sorted(set(lst))

        mapping = {}
        n = len(uniq)
        for idx, val in enumerate(uniq):
            group = idx // k
            end_idx = min((group + 1) * k - 1, n - 1)
            rep = uniq[end_idx]
            mapping[val] = rep
        return [mapping[x] for x in lst]
    
    


    ##################################
    #         DATALOADER             #
    #################################
    logger.info("Creating dataloaders and lr_scheduler")


    def simple_collate(batch):
        idx, extended_input_ids, p_mask, tok_idx_ext, labels, reward = zip(*batch)
        return {
            "ids":        torch.tensor(idx),
            "extended_input_ids":  torch.stack(extended_input_ids),
            "p_mask":  torch.stack(p_mask),
            "tok_idx_ext":  torch.stack(tok_idx_ext),
            "labels":  torch.stack(labels),
            "reward":     reward,
        }
    


    
    with open(project_name + "/temp_data/" + config.dataset.optimization_data + ".json", 'r') as f:
        dataset_load = json.load(f)
    #dataset_load = dataset_load[:2000]

    if len(dataset_load) == 0:
        logger.warning("No training data after filtering. Skipping this training step.")
        return

    prompt_list = []
    response_list = []
    step_map_list = []
    reward_list = []
    source_list = []
    for x in dataset_load:
        prompt_list.append(x["prompt"])
        response_list.append(x["response"])
        reward_list.append(x["reward"])
        source_list.append(x.get("source", "student"))
    
    input_ids_lm, _, start_pos, drop_num = uni_prompting((prompt_list, response_list))


    _, L = input_ids_lm.shape
    L0    = start_pos
    L1    = L - L0
    post_num = config.training.post_num


    for x in dataset_load:
        if "step_map" not in x.keys() or len(x["step_map"]) == 0:
            step_map_list.append([j for j in range(L1)])
        else:
            step_map_i = x["step_map"]
            if len(step_map_i) > L1:
                step_map_i = step_map_i[:L1]
            else:
                step_map_i = step_map_i + [max(step_map_i) + 1] * (L1 - len(step_map_i))
            step_map_list.append(step_map_i)

    
    
    def make_basic_block_attention(
        N: int,
        start_pos: int,            # = L0
        block_size: int,           # = b
    ) -> torch.Tensor:
        B = 1
        L0     = start_pos
        L1     = (N - L0) // 2          # N = L0 + 2·L1 
        assert L0 + 2 * L1 == N, "input length must be L0 + 2*L1"

        # all -inf first
        bias = torch.full((B, 1, N, N), 0)


        rows = torch.arange(L0 + L1, L0 + 2 * L1)              # (L1,)
        rows_token = torch.arange(L0, L0 + L1)              # (L1,)

        # update block by block
        for bi in range((L1 + block_size - 1) // block_size):
            #  [bi*b , min((bi+1)*b, L1))
            left_end   = L0 + min((bi) * block_size, L1)        
            right_start= L0 + L1 + (left_end - L0)

            i_start = bi * block_size
            i_end   = min((bi + 1) * block_size, L1)              # no i_end

            block_rows = rows[i_start:i_end]                    
            bias[:, :, block_rows.unsqueeze(-1), 0:left_end]   = 1
            bias[:, :, block_rows.unsqueeze(-1), right_start:(right_start + block_size)] = 1

            block_rows = rows_token[i_start:i_end]
            left_end   = L0 + min((bi + 1) * block_size, L1)
            bias[:, :, block_rows.unsqueeze(-1), 0:left_end]   = 1
        
        if L0 > 0:
            num_blocks_pre = (L0 + block_size - 1) // block_size
            for bi in range(num_blocks_pre):
                # row interval [row_start, row_end)
                row_end   = max(L0 - bi * block_size, 0)
                row_start = max(L0 - (bi + 1) * block_size, 0)
                if row_end > row_start:
                    block_rows = torch.arange(row_start, row_end)
                    bias[:, :, block_rows.unsqueeze(-1), 0:row_end] = 1
        
        return bias        # (B,1,N,N)
    
    
    

    basic_block_attention = make_basic_block_attention(L0 + 2 * L1, start_pos, config.training.block_size)
    basic_block_attention = basic_block_attention.cpu()


    def process_pad(attn, input_ids):
        N = L0 + 2 * L1
        device = input_ids.device

        cols = torch.arange(N, device=device)                  # (N,)
        key_mask = (cols < start_pos).unsqueeze(0) & (input_ids == pad_id)  # (B, N)

        # set -inf
        attn.masked_fill_(key_mask[:, None, None, :], 0)

        # aviod +-inf or none in forward
        A = attn[:, 0]  # (B, N, N)
        bad = (A.sum(dim=-1) == 0) & (torch.arange(A.size(1), device=A.device).unsqueeze(0) < start_pos)
        b, r = bad.nonzero(as_tuple=True)
        A[b, r, :] = 0; A[b, r, r] = 1

        attn = attn.bool()

        return attn





    def one_round_vectorized(input_ids_b, step_map_b, L0, L1, block_size, mask_id):
        """
        Perform a single "round" on one sample b:
        - For each block, take the minimum non -1 value in step_map.
        - Create pmask (positions equal to the block minimum).
        - Create a noise mask for the extended segment (positions >= block minimum).
        - Mark the chosen minimum positions in step_map as -1 for the next round.

        Returns:
        extended_input_ids_b : Tensor with duplicated + masked response segment
        pmask_b              : Boolean mask for tokens selected in this round
        new_step_map_b       : Updated step_map (selected positions set to -1)
        has_any              : Whether any position was selected in this round
        """
        device = input_ids_b.device
        NB = (L1 + block_size - 1) // block_size
        pad_len = NB * block_size - L1

        # Reshape step_map into [NB, block_size], fill last incomplete block with -1
        step_pad = torch.full((NB * block_size,), -1, dtype=torch.long, device=device)
        step_pad[:L1] = step_map_b
        step_blk = step_pad.view(NB, block_size)                      # [NB, Bk]

        valid = step_blk.ge(0)                                        # Valid positions (not -1)
        big = torch.iinfo(step_blk.dtype).max
        tmp = step_blk.masked_fill(~valid, big)                       # Fill invalid positions with a large value
        min_vals, _ = tmp.min(dim=1, keepdim=True)                    # Current minimum for each block

        # Select positions equal to block minimum (only valid positions)
        pmask_blk = step_blk.eq(min_vals) & valid                     
        if not pmask_blk.any():
            # No positions left to select in this round
            return None, None, step_map_b, False

        # Noise mask for extended segment: mark positions >= block minimum
        ge_mask_blk = step_blk.ge(min_vals) & valid                   # [NB, Bk]

        # Flatten back to length L1 (discard padding)
        pmask_tail = pmask_blk.view(-1)[:L1]                          # [L1]
        ge_mask_tail = ge_mask_blk.view(-1)[:L1]                      # [L1]

        # Construct pmask_b: [0:L0] = False, [L0:] = pmask_tail
        pmask_b = torch.zeros(L0 + L1, dtype=torch.bool, device=device)
        pmask_b[L0:] = pmask_tail

        # Build extended segment: duplicate response and replace noise positions with mask_id
        tail = input_ids_b[L0:L0+L1].clone()
        tail[ge_mask_tail] = mask_id

        
        extended_input_ids_b = torch.empty(L0 + L1 + L1, dtype=input_ids_b.dtype, device=device)
        extended_input_ids_b[:L0+L1] = input_ids_b
        extended_input_ids_b[L0+L1:] = tail

        # Update step_map: mark selected minimum positions as -1 for the next round
        new_step_map_b = step_map_b.clone()
        new_step_map_b[pmask_tail] = -1

        return extended_input_ids_b, pmask_b, new_step_map_b, True
    



    def collect_training_data(input_ids, step_map_list, reward, source_list=None):

        B, L = input_ids.shape
        L0    = start_pos
        L1    = L - L0
        block_size = config.training.block_size

        lower = config.training.lower_p
        upper = config.training.upper_p

        # Check if we have mixed sources (teacher + student)
        has_mixed = source_list is not None and any(s == "teacher" for s in source_list)

        if has_mixed:
            # Per-sample dispatch: teacher → random masking, student → TraceRL masking
            extended_input_ids_list, pmask_list, reward_list = [], [], []

            for b in range(B):
                if source_list[b] == "teacher":
                    # Random masking for teacher samples
                    reward_list.append(reward[b])
                    extended_input_ids_b = input_ids[b]
                    pmask_b = torch.zeros(start_pos, dtype=torch.bool)

                    for j in range(int((L1 - 1) / block_size) + 1):
                        start = j * block_size
                        end = min(L1, (j + 1) * block_size)
                        pmask_b_j = torch.rand(end - start) <= torch.empty(end - start).uniform_(lower, upper)
                        pmask_b = torch.cat([pmask_b, pmask_b_j], dim=0)
                        noise_b_j = input_ids[b, (L0 + start):(L0 + end)].clone()
                        noise_b_j = noise_b_j.masked_fill_(pmask_b_j, mask_id)
                        extended_input_ids_b = torch.cat([extended_input_ids_b, noise_b_j], dim=0)

                    extended_input_ids_list.append(extended_input_ids_b)
                    pmask_list.append(pmask_b)
                else:
                    # TraceRL masking for student samples
                    step_map_i = step_map_list[b]
                    for j in range(int((L1 - 1) / block_size) + 1):
                        s, e = j * block_size, min(L1, (j + 1) * block_size)
                        step_map_list[b][s:e] = collapse_k_unique(step_map_i[s:e], config.training.shrink)
                    step_b = torch.as_tensor(step_map_list[b], dtype=torch.long)
                    while True:
                        out = one_round_vectorized(input_ids[b], step_b, L0, L1, block_size, mask_id)
                        extended_b, pmask_b, step_b, has_any = out
                        if not has_any:
                            break
                        extended_input_ids_list.append(extended_b)
                        pmask_list.append(pmask_b)
                        reward_list.append(reward[b])

        elif config.training.method == "random_masking":

            extended_input_ids_list, pmask_list, reward_list = [], [], []

            for b in range(B):

                reward_list.append(reward[b])

                extended_input_ids_b = input_ids[b]
                pmask_b = torch.zeros(start_pos, dtype=torch.bool)

                for j in range(int((L1 - 1) / block_size) + 1):

                    start = j * block_size
                    end = min(L1, (j + 1) * block_size)

                    pmask_b_j = torch.rand(end - start) <= torch.empty(end - start).uniform_(lower, upper)
                    #pmask_b_j = torch.rand(end - start) <= torch.linspace(lower, upper, steps=end - start)
                    pmask_b = torch.cat([pmask_b, pmask_b_j], dim=0)

                    noise_b_j = input_ids[b, (L0 + start):(L0 + end)].clone()
                    noise_b_j = noise_b_j.masked_fill_(pmask_b_j, mask_id)

                    extended_input_ids_b = torch.cat([extended_input_ids_b, noise_b_j], dim=0)

                extended_input_ids_list.append(extended_input_ids_b)
                pmask_list.append(pmask_b)

        if config.training.method == "coupled" and not has_mixed:

            extended_input_ids_list, pmask_list, reward_list = [], [], []
            coupled_input_ids_list, coupled_pmask_list, coupled_reward_list = [], [], []

            for b in range(B):

                reward_list.append(reward[b])
                coupled_reward_list.append(reward[b])

                extended_input_ids_b = input_ids[b]
                pmask_b = torch.zeros(start_pos, dtype=torch.bool)

                coupled_input_ids_b = input_ids[b]
                coupled_pmask_b = torch.zeros(start_pos, dtype=torch.bool)

                for j in range(int((L1 - 1) / block_size) + 1):

                    start = j * block_size
                    end = min(L1, (j + 1) * block_size)

                    pmask_b_j = torch.rand(end - start) <= torch.empty(end - start).uniform_(lower, upper)
                    #pmask_b_j = torch.rand(end - start) <= torch.linspace(lower, upper, steps=end - start)
                    pmask_b = torch.cat([pmask_b, pmask_b_j], dim=0)
                    coupled_pmask_b = torch.cat([coupled_pmask_b, ~pmask_b_j], dim=0)

                    noise_b_j = input_ids[b, (L0 + start):(L0 + end)].clone()
                    noise_b_j = noise_b_j.masked_fill_(pmask_b_j, mask_id)

                    coupled_noise_b_j = input_ids[b, (L0 + start):(L0 + end)].clone()
                    coupled_noise_b_j = coupled_noise_b_j.masked_fill_(~pmask_b_j, mask_id)

                    extended_input_ids_b = torch.cat([extended_input_ids_b, noise_b_j], dim=0)
                    coupled_input_ids_b  = torch.cat([coupled_input_ids_b, coupled_noise_b_j], dim=0)

                extended_input_ids_list.append(extended_input_ids_b)
                pmask_list.append(pmask_b)

                coupled_input_ids_list.append(coupled_input_ids_b)
                coupled_pmask_list.append(coupled_pmask_b)

            extended_input_ids_list += coupled_input_ids_list
            pmask_list += coupled_pmask_list
            reward_list += coupled_reward_list

        elif config.training.method == "TraceRL" and not has_mixed:

            for b in range(B):
                step_map_i = step_map_list[b]

                for j in range(int((L1 - 1) / block_size) + 1):
                    start = j * block_size
                    end = min(L1, (j + 1) * block_size)
                    step_map_list[b][start:end] = collapse_k_unique(step_map_i[start:end], config.training.shrink)

            step_map = torch.as_tensor(step_map_list, dtype=torch.long)

            assert step_map.shape[1] == L1

            extended_input_ids_list, pmask_list, reward_list = [], [], []

            one_state_per_block = getattr(config.training, "one_state_per_block", False)
            random_mask = getattr(config.rollout, "random_mask", False)

            for b in range(B):
                # When random_mask is on, skip the trajectory roll-out entirely:
                # we synthesize one row directly from the clean response.
                if random_mask:
                    ext_b, pm_b = _combine_rounds_random_mask(
                        input_ids[b], L0, L1, block_size, mask_id,
                    )
                    extended_input_ids_list.append(ext_b)
                    pmask_list.append(pm_b)
                    reward_list.append(reward[b])
                    continue

                step_b = step_map[b]
                per_prompt_ext, per_prompt_pm = [], []
                while True:
                    out = one_round_vectorized(
                        input_ids_b=input_ids[b],
                        step_map_b=step_b,
                        L0=L0,
                        L1=L1,
                        block_size=block_size,
                        mask_id=mask_id,
                    )
                    extended_b, pmask_b, step_b, has_any = out
                    if not has_any:
                        break
                    per_prompt_ext.append(extended_b)
                    per_prompt_pm.append(pmask_b)

                if not per_prompt_ext:
                    continue

                if one_state_per_block:
                    ext_b, pm_b = _combine_rounds_one_state_per_block(
                        per_prompt_ext, per_prompt_pm, input_ids[b], L0, L1, block_size,
                    )
                    extended_input_ids_list.append(ext_b)
                    pmask_list.append(pm_b)
                    reward_list.append(reward[b])
                else:
                    extended_input_ids_list.extend(per_prompt_ext)
                    pmask_list.extend(per_prompt_pm)
                    reward_list.extend([reward[b]] * len(per_prompt_ext))

        extended_input_ids = torch.stack(extended_input_ids_list, dim=0)
        p_mask =  torch.stack(pmask_list, dim=0).to(torch.bool)
        
        pad_resp = (extended_input_ids[:, :L] == pad_id) & p_mask        
        if post_num is not None:
            cum_pad = torch.cumsum(pad_resp.int(), dim=1)
            p_mask &= ~(pad_resp & (cum_pad > post_num))
        
        labels = extended_input_ids[:, :L].clone()

        idx = torch.arange(L).unsqueeze(0).expand(extended_input_ids.shape[0], -1)
        valid = (idx >= start_pos) | extended_input_ids[:, :L].ne(pad_id)      
        tok_idx = valid.long().cumsum(dim=-1) - 1         
        tok_idx = tok_idx.masked_fill(~valid, 1)
        tok_idx_resp = tok_idx[:, start_pos:]  
        tok_idx_ext  = torch.cat([tok_idx, tok_idx_resp], dim=1)

        keep = p_mask.view(p_mask.size(0), -1).any(dim=1)
        idx  = keep.nonzero(as_tuple=True)[0]          # LongTensor of indices

        extended_input_ids = extended_input_ids[idx]
        p_mask            = p_mask[idx]
        tok_idx_ext       = tok_idx_ext[idx]
        labels            = labels[idx]

        reward_list = [reward_list[i] for i in idx.tolist()]

        return extended_input_ids, p_mask, tok_idx_ext, labels, reward_list
        

    
    extended_input_ids, p_mask, tok_idx_ext, labels, rewards = collect_training_data(input_ids_lm, step_map_list, reward_list, source_list=source_list)




    dataset_lm = TrainDataset(extended_input_ids, p_mask, tok_idx_ext, labels, rewards)
    logger.info(f"  Num training rows (after expand+filter) = {len(dataset_lm)} (from {input_ids_lm.shape[0]} responses)")

    total_batch_size_lm = config.training.batch_size_lm * accelerator.num_processes * config.training.gradient_accumulation_steps
    num_update_steps_per_epoch = math.ceil(len(dataset_lm) / total_batch_size_lm)
    num_train_epochs = config.training.num_train_epochs
    steps_per_rl_step = num_update_steps_per_epoch * num_train_epochs
    total_rl_steps = config.experiment.total_step
    _scheduler_name = str(config.lr_scheduler.scheduler).lower()
    if total_rl_steps > 0:
        max_train_steps = steps_per_rl_step * total_rl_steps + 1
    elif _scheduler_name in ("constant", "constant_with_warmup"):
        max_train_steps = None
    else:
        raise ValueError(
            f"lr_scheduler='{_scheduler_name}' needs a finite training horizon, "
            "but experiment.total_step<=0 and no epoch-mode override is active "
            "(set dataset.num_data_epochs>=1 or experiment.total_step>0, or use a constant scheduler)."
        )

    _decay_steps = getattr(config.lr_scheduler.params, "decay_steps", None)
    if _decay_steps is not None:
        _decay_steps = int(_decay_steps) * steps_per_rl_step  # convert RL steps to inner training steps
    # Convert warmup_steps from RL steps to inner optimizer steps
    _warmup_rl_steps = config.lr_scheduler.params.warmup_steps
    _warmup_inner_steps = int(_warmup_rl_steps) * steps_per_rl_step
    lr_scheduler = get_scheduler(
        config.lr_scheduler.scheduler,
        optimizer=optimizer,
        num_training_steps=max_train_steps,
        num_warmup_steps=_warmup_inner_steps,
        min_lr_scale=config.lr_scheduler.params.min_lr_scale,
        decay_steps=_decay_steps,
    )

    train_dataloader_lm = DataLoader(
        dataset_lm,
        batch_size=config.training.batch_size_lm,
        sampler=None,
        shuffle=True,
        collate_fn=simple_collate,
        num_workers=0
    )





    

    ##################################
    #       Prepare accelerator     #
    #################################
    logger.info("Preparing model, optimizer and dataloaders")
    model, optimizer, lr_scheduler, train_dataloader_lm = accelerator.prepare(
        model, optimizer, lr_scheduler, train_dataloader_lm
    )

    should_resume_training_state = (
        config.experiment.current_epoch > 1 or not config.experiment.start_from_scratch
    )
    if should_resume_training_state:
        load_training_state(model, config, config.model.optimized_name)

    if len(train_dataloader_lm) < accelerator.gradient_accumulation_steps:
        print(
            f"Number of batches ({len(train_dataloader_lm)}) is less than gradient accumulation steps ({accelerator.gradient_accumulation_steps}). "
            "Please reduce gradient accumulation steps or increase the number of training samples."
        )

    # Prepare teacher model AFTER student so accelerator.backward stays tied to student engine.
    # PPO doesn't need the teacher during training (reward is pre-computed).
    _loss_type = getattr(config.training, "loss_type", "kl")
    if _loss_type in ("kl", "jsd"):
        accelerator.state.select_deepspeed_plugin("teacher")
        accelerator.state.deepspeed_plugin.deepspeed_config["train_micro_batch_size_per_gpu"] = 1
        teacher_model = AutoModelForCausalLM.from_pretrained(
            config.model.teacher_model, trust_remote_code=True, torch_dtype="auto"
        )
        teacher_model.requires_grad_(False)
        teacher_model = accelerator.prepare(teacher_model)
        teacher_model.eval()
        accelerator.state.select_deepspeed_plugin("student")
    else:
        teacher_model = None
        logger.info(f"{_loss_type} mode: skipping teacher model loading (reward pre-computed)")

    # Cache full (unsharded) param counts for FLOPs logging (ds_numel for ZeRO-3)
    _student_total_params = sum(getattr(p, 'ds_numel', p.numel()) for p in model.parameters())
    _teacher_total_params = sum(getattr(p, 'ds_numel', p.numel()) for p in teacher_model.parameters()) if teacher_model is not None else 0
    logger.info(f"Param counts for FLOPs: student={_student_total_params/1e6:.1f}M, teacher={_teacher_total_params/1e6:.1f}M")


    import torch.nn.functional as F


    @torch.no_grad()
    def compute_logp_old_tok_parallel(
            accelerator,
            dataset,
            train_dataloader_lm,
            start_pos, pad_id,
            batch_size):

        model.eval()

        dl = train_dataloader_lm

        for batch in dl:
            ids        = batch["ids"]         
            extended_input_ids = batch["extended_input_ids"].to(accelerator.device)
            p_mask = batch["p_mask"].to(accelerator.device)
            tok_idx_ext = batch["tok_idx_ext"].to(accelerator.device)
            labels = batch["labels"].to(accelerator.device)

            B, L = p_mask.shape
            L0    = start_pos
            L1    = L - L0
            device = extended_input_ids.device

            attention_mask = basic_block_attention.clone()
            attention_mask = attention_mask.repeat_interleave(B, dim=0).to(device)
            attention_mask = process_pad(attention_mask, extended_input_ids)

            logits = model(input_ids = extended_input_ids, attention_mask = attention_mask, position_ids = tok_idx_ext).logits
            logits = torch.cat([logits[:, :L0, :], logits[:, L0 + L1 :, :]], dim=1)  # (B, L0+L1, V)

            log_probs = F.log_softmax(logits, dim=-1)
            logp_tok  = log_probs.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)

            dataset.logp_old_tok[ids] = logp_tok.float().cpu()

        accelerator.wait_for_everyone()

        model.train()


    #################################
    #             Inference         #
    #################################
    _loss_type = getattr(config.training, "loss_type", "kl")
    if config.training.block_size == 1:
        logger.info("***** Skipping old logp inference (causal mode) *****")
    elif _loss_type in ("kl", "jsd"):
        logger.info("***** Skipping old logp inference (KL/JSD distillation, logp_old not used) *****")
    else:
        logger.info("***** Running inference *****")
        compute_logp_old_tok_parallel(
            accelerator,
            dataset_lm,
            train_dataloader_lm,
            start_pos=start_pos,
            pad_id=pad_id,
            batch_size=config.training.batch_size_lm,
        )






    #################################
    #             Training          #
    #################################
    logger.info("***** Running training *****")
    
    logger.info(f"  Num response = {len(dataset_load)}")
    logger.info(f"  Num sample dropped = {drop_num}")
    logger.info(f"  Num training data = {input_ids_lm.shape[0]}")
    logger.info(f"  Num training steps = {max_train_steps}")
    logger.info(f"  Instantaneous batch size per device = {config.training.batch_size_lm}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size_lm}")
    logger.info(f"  Gradient Accumulation steps = {config.training.gradient_accumulation_steps}")

    first_epoch = 0
    data_time_m = AverageMeter()
    # Load cumulative loss token count from previous RL steps
    _token_count_file = os.path.join(config.experiment.project, "temp_data", "cumulative_loss_tokens.txt")
    if os.path.exists(_token_count_file):
        with open(_token_count_file) as f:
            cumulative_loss_tokens = int(f.read().strip())
    else:
        cumulative_loss_tokens = 0
    end = time.time()

    current_epoch = config.experiment.current_epoch

    if wandb_enabled and len(rewards) > 0:
        reward_arr = np.asarray(rewards, dtype=np.float32)
        accelerator.log(
            {
                "train/reward_mean": float(reward_arr.mean()),
                "train/reward_std": float(reward_arr.std()),
                "train/reward_min": float(reward_arr.min()),
                "train/reward_max": float(reward_arr.max()),
                "train/current_epoch": current_epoch,
            }
        )

    


    

    def forward_process_causal(extended_input_ids, p_mask, tok_idx_ext, labels, adv, logp_old_tok):
        """Pure causal KL distillation for AR student and AR teacher.

        Both models see [prompt | response] with standard causal attention.
        logits[t] predicts token t+1.  KL loss on response tokens only.
        No [MASK] tokens, no doubled sequence, no block attention.
        """

        # labels holds the clean [prompt | response] (= extended_input_ids[:, :L])
        B, L = labels.shape
        L0 = start_pos
        L1 = L - L0
        device = labels.device
        input_ids = labels  # [prompt | response], shape (B, L)

        pad_mask = input_ids.ne(pad_id)  # (B, L), bool
        if model_base == "qwen":
            # Qwen: 2D bool mask — transformers _update_causal_mask adds causal structure
            attn_mask = pad_mask  # [B, L] bool
        else:
            # SDAR/BD3LM: 4D bool mask — they don't auto-add causal structure
            attn_mask = torch.tril(torch.ones(L, L, dtype=torch.bool, device=device))  # [L, L]
            attn_mask = attn_mask[None, None, :, :] & pad_mask[:, None, None, :]  # [B, 1, L, L] bool

        # ── Student (causal) ──────────────────────────────────────
        student_logits = model(input_ids=input_ids, attention_mask=attn_mask).logits  # (B, L, V)
        # logits[t] predicts token t+1; shift so index t aligns with token t
        student_logprobs = F.log_softmax(student_logits[:, :-1, :].float(), dim=-1)  # (B, L-1, V)

        # ── Teacher (causal) ─────────────────────────────────────
        with torch.no_grad():
            teacher_logits = teacher_model(input_ids=input_ids, attention_mask=pad_mask).logits
            teacher_logprobs = F.log_softmax(teacher_logits[:, :-1, :].float(), dim=-1)  # (B, L-1, V)

        # ── KL per token ────────────────────────────────────────
        _divergence_type = getattr(config.training, "loss_type", "kl")
        if _divergence_type == "jsd":
            _jsd_alpha = getattr(config.training, "jsd_alpha", 0.5)
            teacher_probs = teacher_logprobs.exp()
            student_probs = student_logprobs.exp()
            M = _jsd_alpha * teacher_probs + (1.0 - _jsd_alpha) * student_probs
            log_M = M.log()
            kl_div = _jsd_alpha * (teacher_probs * (teacher_logprobs - log_M)).sum(dim=-1) \
                   + (1.0 - _jsd_alpha) * (student_probs * (student_logprobs - log_M)).sum(dim=-1)
        else:
            kl_div = (teacher_logprobs.exp() * (teacher_logprobs - student_logprobs)).sum(dim=-1)  # KL(teacher||student), (B, L-1)

        # ── Loss mask: response tokens only ──────────────────────
        # shifted labels correspond to tokens at positions 1..L-1
        # response starts at L0, so in shifted space it starts at L0-1
        response_start = max(L0 - 1, 0)
        loss_mask = torch.zeros(B, L - 1, dtype=torch.bool, device=device)
        loss_mask[:, response_start:] = True

        # exclude pad tokens (shifted)
        loss_mask &= input_ids[:, 1:].ne(pad_id)

        # exclude everything after first <|im_end|> in response
        im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
        is_im_end = input_ids[:, 1:].eq(im_end_id) & loss_mask
        im_end_cumsum = is_im_end.cumsum(dim=1)
        im_end_shifted = F.pad(im_end_cumsum[:, :-1], (1, 0))
        loss_mask &= im_end_shifted.eq(0)
        if getattr(config.training, "exclude_im_end", False):
            loss_mask &= ~is_im_end

        # ── Compute loss ─────────────────────────────────────────
        kl_masked = kl_div * loss_mask
        loss_unreduced = kl_masked.sum(dim=1) / loss_mask.sum(dim=1).clamp_min(1.0)

        num_response = loss_mask.sum(dim=1)
        loss_by_masking_rate = {}
        for n, loss_i in zip(num_response, loss_unreduced):
            metric_name = "train/loss_causal"
            num_rate, total_loss = loss_by_masking_rate.get(metric_name, (0, 0))
            loss_by_masking_rate[metric_name] = (num_rate + 1, total_loss + loss_i.item())

        loss = loss_unreduced.mean()
        # Count all non-padding tokens (prompt + response) for FLOPs estimation
        num_forward_tokens = pad_mask.sum().item()
        return loss, loss_by_masking_rate, num_forward_tokens

    def forward_process_ppo_causal(extended_input_ids, p_mask, tok_idx_ext, labels, adv, logp_old_tok):
        """PPO-clipped policy gradient for causal (AR) models.

        Same as dLLM-RL's original forward_process but for the causal path.
        Uses importance-weighted clipped surrogate + optional KL penalty.
        """
        B, L = labels.shape
        L0 = start_pos
        L1 = L - L0
        device = labels.device
        input_ids = labels  # [prompt | response], shape (B, L)
        adv = torch.as_tensor(adv, device=device, dtype=torch.float32).detach()

        pad_mask = input_ids.ne(pad_id)  # (B, L), bool
        if model_base == "qwen":
            attn_mask = pad_mask
        else:
            attn_mask = torch.tril(torch.ones(L, L, dtype=torch.bool, device=device))
            attn_mask = attn_mask[None, None, :, :] & pad_mask[:, None, None, :]

        # ── Student forward (causal) ─────────────────────────────
        student_logits = model(input_ids=input_ids, attention_mask=attn_mask).logits  # (B, L, V)
        student_logprobs = F.log_softmax(student_logits[:, :-1, :].float(), dim=-1)  # (B, L-1, V)

        # ── Response mask ────────────────────────────────────────
        response_start = max(L0 - 1, 0)
        loss_mask = torch.zeros(B, L - 1, dtype=torch.bool, device=device)
        loss_mask[:, response_start:] = True
        loss_mask &= input_ids[:, 1:].ne(pad_id)

        im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
        is_im_end = input_ids[:, 1:].eq(im_end_id) & loss_mask
        im_end_cumsum = is_im_end.cumsum(dim=1)
        im_end_shifted = F.pad(im_end_cumsum[:, :-1], (1, 0))
        loss_mask &= im_end_shifted.eq(0)
        if getattr(config.training, "exclude_im_end", False):
            loss_mask &= ~is_im_end

        # ── Per-token log probs ──────────────────────────────────
        logp_new_tok = student_logprobs.gather(dim=-1, index=input_ids[:, 1:].unsqueeze(-1)).squeeze(-1)  # (B, L-1)

        # ── PPO clipped surrogate ────────────────────────────────
        ratio = logp_new_tok - logp_old_tok[:, :logp_new_tok.shape[1]]
        ratio = torch.where(loss_mask, ratio, torch.zeros_like(ratio)).clamp(-10.0, 10.0)
        ratio = torch.exp(ratio)
        clipped = torch.clamp(ratio, 1 - config.training.eps, 1 + config.training.eps)

        adv_tok = adv.unsqueeze(1)
        surrogate_tok = torch.min(ratio * adv_tok, clipped * adv_tok)
        surrogate_tok = surrogate_tok * loss_mask
        surrogate_tok = surrogate_tok.sum(dim=1) / loss_mask.sum(dim=1).clamp_min(1.0)
        policy_loss = -(surrogate_tok.sum() / B)

        # ── KL penalty against old policy (optional) ─────────────
        kl_loss = torch.tensor(0.0, device=device)
        if config.training.beta > 0:
            kl_seq = logp_new_tok - logp_old_tok[:, :logp_new_tok.shape[1]]
            kl_seq = torch.where(loss_mask, kl_seq, torch.zeros_like(kl_seq))
            if config.training.use_kl_estimator_k3:
                t = (-kl_seq).clamp(-10.0, 10.0)
                kl_seq = t.exp() - 1.0 + kl_seq
            kl_seq = (kl_seq * loss_mask).sum(dim=1) / loss_mask.sum(dim=1).clamp_min(1.0)
            kl_loss = config.training.beta * kl_seq.sum() / B
            total_loss = policy_loss + kl_loss
        else:
            total_loss = policy_loss

        loss_by_masking_rate = {"train/loss_ppo_policy": (B, policy_loss.item() * B),
                                "train/loss_ppo_kl": (B, kl_loss.item() * B)}
        # Count all non-padding tokens (prompt + response) for FLOPs estimation
        num_forward_tokens = pad_mask.sum().item()
        return total_loss, loss_by_masking_rate, num_forward_tokens

    def forward_process(extended_input_ids, p_mask, tok_idx_ext, labels, adv, logp_old_tok):

        adv = torch.as_tensor(adv, device=extended_input_ids.device).detach()

        B, L = p_mask.shape
        L0    = start_pos
        L1    = L - L0
        device = extended_input_ids.device

        attention_mask = basic_block_attention.clone()
        attention_mask = attention_mask.repeat_interleave(B, dim=0).to(device)
        attention_mask = process_pad(attention_mask, extended_input_ids)

        full_logits = model(input_ids = extended_input_ids, attention_mask = attention_mask, position_ids = tok_idx_ext).logits
        logits = torch.cat([full_logits[:, :L0, :], full_logits[:, L0 + L1 :, :]], dim=1)  # (B, L0+L1, V)

        if getattr(config.training, "student_arm_shift", False):
            logits = logits.roll(dims=1, shifts=1)  # ARM next-token → current-token alignment

        log_probs = F.log_softmax(logits.float(), dim=-1)

        # try:
        #     res = {
        #         'L0': L0,
        #         'L1': L1,
        #         'attention_mask': attention_mask.detach(),
        #         'extended_input_ids': extended_input_ids.detach(),
        #         'tok_idx_ext': tok_idx_ext.detach(),
        #     }
        #     now = time.time()
        #     rank = accelerator.process_index    
        #     torch.save(res, f"./{project_name}/temp_data/debug_res_{int(now)}_rank{rank}.pt")
        # except Exception as e:
        #     logger.warning(f"Failed to save debug info: {e}")

        # out = tokenizer.decode(extended_input_ids.flatten())
        # print(out)


        # x0 only
        prompt_response = extended_input_ids[:, :L0 + L1]
        # attn_mask = prompt_response.ne(tokenizer.pad_token_id)
        # pos_ids = tok_idx_ext[:, :L0 + L1]

        # fill [MASK] tokens for teacher input
        extended_input_ids_sampled = extended_input_ids.clone()
        masked_input_tokens = extended_input_ids_sampled == tokenizer.mask_token_id
        if masked_input_tokens[:, :L0 + L1].any():
            print("Warning: mask token found in prompt or response, which may cause incorrect sampling. This should not happen if the data is prepared correctly.")
        teacher_fill = getattr(config.training, "teacher_fill", "argmax_fill")
        if teacher_fill == "argmax_fill":
            masked_logits = full_logits[masked_input_tokens]  # (num_masked_tokens, V)
            sampled_response_tokens = masked_logits.argmax(dim=-1)  # (num_masked_tokens,)
            extended_input_ids_sampled[masked_input_tokens] = sampled_response_tokens
        elif teacher_fill == "clean_fill":
            # fill [MASK] in xt with corresponding x0 tokens
            x0_repeated = torch.cat([extended_input_ids[:, :L0 + L1], extended_input_ids[:, L0:L0 + L1]], dim=1)  # (B, L0+2*L1)
            extended_input_ids_sampled[masked_input_tokens] = x0_repeated[masked_input_tokens]
        else:
            raise ValueError(f"Unknown teacher_fill method: {teacher_fill}")
        # out_noised = tokenizer.decode(extended_input_ids_sampled.flatten())
        # print(out_noised)
        causal_attention_mask_bool = attention_mask  # bool: True means attend
        prompt_and_x0_mask = torch.ones(1, 1, L0 + L1, L0 + L1, dtype=torch.bool, device=device)
        prompt_and_x0_mask = torch.tril(prompt_and_x0_mask, diagonal=0).repeat(B, 1, 1, 1)
        xt_mask = torch.ones(1, 1, L1, L1, dtype=torch.bool, device=device)
        xt_mask = torch.tril(xt_mask, diagonal=0).repeat(B, 1, 1, 1)
        causal_attention_mask_bool[:, :, :L0 + L1, :L0 + L1] &= prompt_and_x0_mask
        causal_attention_mask_bool[:, :, L0 + L1 : , L0 + L1 : ] &= xt_mask
        causal_attention_mask = torch.full(causal_attention_mask_bool.shape, float("-inf"), dtype=torch.bfloat16, device=device)
        causal_attention_mask[causal_attention_mask_bool] = 0.0


        with torch.no_grad():

            # x0 only
            # logits_teacher = teacher_model(
            #     input_ids=prompt_response, 
            #     attention_mask=attn_mask.bfloat16(),
            #     position_ids=pos_ids
            # ).logits
            # logits_teacher = logits_teacher.roll(dims=1, shifts=1)  # shift right to align with next-token prediction
            # teacher_logprobs = F.log_softmax(logits_teacher.float(), dim=-1)

            # with sampled
            block_size = config.training.block_size
            logits_teacher_full = teacher_model(
                input_ids=extended_input_ids_sampled, 
                attention_mask=causal_attention_mask.bfloat16(),
                position_ids=tok_idx_ext
            ).logits
            logits_teacher = torch.cat([logits_teacher_full[:, :L0, :], logits_teacher_full[:, L0 + L1 :, :]], dim=1)  # (B, L0+L1, V)
            replace_x0_indices = torch.arange(start=L0 + block_size - 1, end=L0 + L1, step=block_size)
            logits_teacher[:, replace_x0_indices] = logits_teacher_full[:, replace_x0_indices]
            logits_teacher = logits_teacher.roll(dims=1, shifts=1)  # shift right to align with next-token prediction
            teacher_logprobs = F.log_softmax(logits_teacher.float(), dim=-1)

            
        _divergence_type = getattr(config.training, "loss_type", "kl")
        if _divergence_type == "jsd":
            _jsd_alpha = getattr(config.training, "jsd_alpha", 0.5)
            teacher_probs = teacher_logprobs.exp()
            student_probs = log_probs.exp()
            M = _jsd_alpha * teacher_probs + (1.0 - _jsd_alpha) * student_probs
            log_M = M.log()
            kl_div = _jsd_alpha * (teacher_probs * (teacher_logprobs - log_M)).sum(dim=-1) \
                   + (1.0 - _jsd_alpha) * (student_probs * (log_probs - log_M)).sum(dim=-1)
        else:
            kl_div = (teacher_logprobs.exp() * (teacher_logprobs - log_probs)).sum(dim=-1)  # KL(teacher||student), (B, L0+L1)

        # prompt mask
        prompt = extended_input_ids[:, :L0]
        response_noised = extended_input_ids[:, L0 + L1 :]
        prompt_mask = torch.cat([torch.zeros_like(prompt), torch.ones_like(response_noised)], dim=1).bool()

        # we need a mask to rm everything after the '<|im_end|>' in the response:  out_kl = ...,('.\n', 0.3), ('$$', 0.1), ('<|im_end|>', 9.3), ('\n', 71.5), ('<|endoftext|>', 153.0)
        im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
        is_im_end = prompt_response.eq(im_end_id)
        is_response_im_end = is_im_end & prompt_mask
        im_end_cumsum = is_response_im_end.cumsum(dim=1)
        im_end_shifted = F.pad(im_end_cumsum[:, :-1], (1,0)) # right shift so that the first token after '<|im_end|>' in response is marked as 1
        im_end_mask = im_end_shifted.eq(0)  # True before the first '<|im_end|>' in response, False after
        # optionally exclude <|im_end|> itself from loss
        if getattr(config.training, "exclude_im_end", False):
            im_end_mask = im_end_mask & ~is_response_im_end

        # # the generation prompt has the following text at the end: <|im_end|>\n<|im_start|>assistant\n
        # # for some reason, the final newline is included as part of the response, and has high KL: 
        # # ...,('<|im_start|>', 5.5), ('assistant', 46.0), ('\n', 34.2), ('To', 7.8), (' figure', 1.0), ...
        # # create a mask for this first newline token in response
        # newline_id = tokenizer.encode("\n")
        # assert len(newline_id) == 1, "newline should be tokenized to a single token"
        # newline_id = newline_id[0]
        # is_newline = prompt_response.eq(newline_id)
        # is_response_newline = is_newline & prompt_mask
        # response_newline_positions = is_response_newline.nonzero()
        # response_first_newline_positions = []
        # for b in range(B):
        #     pos = response_newline_positions[response_newline_positions[:,0] == b][:,1]
        #     if len(pos) > 0:
        #         response_first_newline_positions.append(pos.min())
        #     else:
        #         raise ValueError("no newline in response, unexpected")
        # response_first_newline_positions = torch.stack(response_first_newline_positions)
        # first_newline_mask = torch.ones_like(im_end_mask, dtype=torch.bool)
        # arange = torch.arange(B).type_as(response_first_newline_positions)
        # first_newline_mask[arange, response_first_newline_positions] = False
        # assert first_newline_mask.logical_not().count_nonzero(dim=1).eq(1).all()

        # pad mask        
        prompt_noised_response = torch.cat([prompt, response_noised], dim=1)
        pad_mask = prompt_noised_response.ne(tokenizer.pad_token_id) 

        # masked token mask
        masked_token_mask = prompt_noised_response.eq(tokenizer.mask_token_id)

        response_mask = prompt_mask & pad_mask & im_end_mask # & first_newline_mask
        loss_mask = response_mask & masked_token_mask
        kl_div_mask = kl_div * loss_mask
        loss_unreduced = kl_div_mask.sum(dim=1) / loss_mask.sum(dim=1).clamp_min(1.0)  

        # if accelerator.is_main_process: 
        #     out = [tokenizer.decode(t) for t in prompt_response.flatten()]
        #     kl_div_mask_max = kl_div_mask.max()
        #     out_kl = [(f"***** {t} *****" if kl > 0.9 * kl_div_mask_max else t, round(kl.item(), 1)) for t, kl in zip(out, kl_div.flatten())]
        #     out_kl_filter = [out for out, mask in zip(out_kl, loss_mask.flatten()) if mask]
        #     out_kl_filter2 = [out if mask else (out[0],-1) for out, mask in zip(out_kl, response_mask.flatten())]
        #     pprint(out_kl_filter2)

        num_response_tokens = response_mask.sum(dim=1)
        # out_noised = tokenizer.decode(prompt_noised_response.flatten())
        # print(out_noised)
        num_masked_tokens = (response_mask & masked_token_mask).sum(dim=1)
        frac_masked = num_masked_tokens.float() / num_response_tokens.float().clamp(min=1)
        loss_by_masking_rate = {}
        for frac, loss_i in zip(frac_masked, loss_unreduced):
            metric_name = f"train/loss_masked_frac_{frac:.1f}"
            num_rate, total_loss = loss_by_masking_rate.get(metric_name, (0, 0))
            loss_by_masking_rate[metric_name] = (num_rate + 1, total_loss + loss_i.item())
        # print((loss_mask.float().mean().item(), kl_div[loss_mask].max().item(), loss.item()))
        loss = loss_unreduced.mean() # mean over batch dim
        # Count all non-padding tokens in prompt+response (L0+L1) for FLOPs estimation
        num_forward_tokens = prompt_response.ne(tokenizer.pad_token_id).sum().item()
        return loss, loss_by_masking_rate, num_forward_tokens

    def forward_process_ppo(extended_input_ids, p_mask, tok_idx_ext, labels, adv, logp_old_tok):
        """PPO-clipped policy gradient for diffusion models (original dLLM-RL).

        Uses importance-weighted clipped surrogate on masked tokens + optional
        KL penalty against the old policy.
        """
        adv = torch.as_tensor(adv, device=extended_input_ids.device).detach()

        B, L = p_mask.shape
        L0    = start_pos
        L1    = L - L0
        device = extended_input_ids.device

        attention_mask = basic_block_attention.clone()
        attention_mask = attention_mask.repeat_interleave(B, dim=0).to(device)
        attention_mask = process_pad(attention_mask, extended_input_ids)

        full_logits = model(input_ids=extended_input_ids, attention_mask=attention_mask, position_ids=tok_idx_ext).logits
        logits = torch.cat([full_logits[:, :L0, :], full_logits[:, L0 + L1:, :]], dim=1)  # (B, L0+L1, V)

        if getattr(config.training, "student_arm_shift", False):
            logits = logits.roll(dims=1, shifts=1)  # ARM next-token → current-token alignment

        log_probs = F.log_softmax(logits.float(), dim=-1)
        logp_new_tok = log_probs.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)  # (B, T)

        ratio = logp_new_tok - logp_old_tok
        ratio = torch.where(p_mask, ratio, torch.zeros_like(ratio)).clamp(-10.0, 10.0)
        ratio = torch.exp(ratio)
        clipped = torch.clamp(ratio, 1 - config.training.eps, 1 + config.training.eps)

        adv_tok = adv.unsqueeze(1)
        surrogate_tok = torch.min(ratio * adv_tok, clipped * adv_tok)
        surrogate_tok = surrogate_tok * p_mask
        surrogate_tok = surrogate_tok.sum(dim=1) / L1
        policy_loss = -(surrogate_tok.sum() / B)

        # KL penalty against old policy (optional)
        kl_loss = torch.tensor(0.0, device=device)
        if config.training.beta > 0:
            kl_seq = logp_new_tok - logp_old_tok
            kl_seq = torch.where(p_mask, kl_seq, torch.zeros_like(kl_seq))
            if config.training.use_kl_estimator_k3:
                t = (-kl_seq).clamp(-10.0, 10.0)
                kl_seq = t.exp() - 1.0 + kl_seq
            kl_seq = (kl_seq * p_mask).sum(dim=1) / L1
            kl_loss = config.training.beta * kl_seq.sum() / B
            total_loss = policy_loss + kl_loss
        else:
            total_loss = policy_loss

        loss_by_masking_rate = {"train/loss_ppo_policy": (B, policy_loss.item() * B),
                                "train/loss_ppo_kl": (B, kl_loss.item() * B)}
        # Count all non-padding tokens in prompt+response (L0+L1) for FLOPs estimation
        num_forward_tokens = extended_input_ids[:, :L0 + L1].ne(tokenizer.pad_token_id).sum().item()
        return total_loss, loss_by_masking_rate, num_forward_tokens






    from tqdm.auto import tqdm

    for epoch in range(first_epoch, num_train_epochs):
        
        model.train()
        
        progress_bar = tqdm(
            train_dataloader_lm,
            desc=f"Epoch {epoch+1}/{num_train_epochs}",
            disable=not accelerator.is_local_main_process,
            dynamic_ncols=True,          
            leave=True               
        )
        
        
        metrics = {}
        stepped = False
        avg_loss = AverageMeter()
        for step, batch in enumerate(progress_bar):

            # for loss calculation

            data_time_m.update(time.time() - end)

            extended_input_ids = batch["extended_input_ids"].to(accelerator.device)
            p_mask = batch["p_mask"].to(accelerator.device)
            tok_idx_ext = batch["tok_idx_ext"].to(accelerator.device)
            labels = batch["labels"].to(accelerator.device)
            reward = batch["reward"]
            old_lp = dataset_lm.logp_old_tok[batch["ids"].cpu()].to(accelerator.device)

            if config.training.block_size != 1 and _loss_type not in ("kl", "jsd") and torch.isneginf(old_lp).any().item():
                print(old_lp)

            _loss_type = getattr(config.training, "loss_type", "kl")
            if _loss_type == "ppo":
                _fwd_fn = forward_process_ppo_causal if config.training.block_size == 1 else forward_process_ppo
            elif _loss_type in ("kl", "jsd"):
                _fwd_fn = forward_process_causal if config.training.block_size == 1 else forward_process
            else:
                raise ValueError(f"Unknown loss_type: {_loss_type}")

            loss_lm, loss_metrics, batch_loss_tokens = _fwd_fn(
                    extended_input_ids=extended_input_ids,
                    p_mask=p_mask,
                    tok_idx_ext=tok_idx_ext,
                    labels=labels,
                    adv=reward,
                    logp_old_tok=old_lp
                )
            _batch_loss_tokens_tensor = torch.tensor(batch_loss_tokens, device=accelerator.device)
            torch.distributed.all_reduce(_batch_loss_tokens_tensor, op=torch.distributed.ReduceOp.SUM)
            cumulative_loss_tokens += _batch_loss_tokens_tensor.item()
            for loss_metric_name, (loss_metric_ct, loss_metric_value) in loss_metrics.items():
                if loss_metric_name not in metrics:
                    metrics[loss_metric_name] = AverageMeter()
                loss_metric_value = loss_metric_value / loss_metric_ct
                metrics[loss_metric_name].update(loss_metric_value, n=loss_metric_ct)
            avg_loss.update(loss_lm.item(), n=len(extended_input_ids))

            with accelerator.accumulate(model):
                accelerator.backward(loss_lm)
                if accelerator.sync_gradients:
                    stepped = True
                    _clip_val = config.training.max_grad_norm if config.training.max_grad_norm is not None else float("inf")
                    grad_norm_before_clip = accelerator.clip_grad_norm_(model.parameters(), _clip_val)

                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad(set_to_none=True)

                    float2tensor = lambda x: torch.tensor(x, device=accelerator.device)
                    all_sums = accelerator.gather(float2tensor(avg_loss.sum))
                    all_counts = accelerator.gather(float2tensor(avg_loss.count))
                    all_avg = all_sums.sum() / all_counts.sum().clamp_min(1)
                    # Sync metric keys across processes: encode local keys as a
                    # binary tensor so all processes agree on the full key set.
                    all_possible_keys = sorted({
                        f"train/loss_masked_frac_{round(f, 1)}"
                        for f in [i / 10 for i in range(11)]
                    })
                    local_flags = torch.tensor(
                        [1.0 if k in metrics else 0.0 for k in all_possible_keys],
                        device=accelerator.device,
                    )
                    global_flags = accelerator.gather(local_flags)
                    if global_flags.ndim > 1:
                        global_flags = global_flags.sum(dim=0)
                    all_metrics_keys = [k for k, f in zip(all_possible_keys, global_flags) if f > 0]

                    all_metrics = {}
                    for key in all_metrics_keys:
                        all_sums = accelerator.gather(float2tensor(metrics[key].sum if key in metrics else 0.0))
                        all_counts = accelerator.gather(float2tensor(metrics[key].count if key in metrics else 0))
                        if all_counts.sum() > 0:
                            all_metrics[key] = all_sums.sum() / all_counts.sum()

                    logged_metrics = {
                        "train/loss": all_avg.item(),
                        "train/lr": float(lr_scheduler.get_last_lr()[0]),
                        "train/epoch": float(epoch + 1),
                        **all_metrics,
                    }
                    logged_metrics["train/grad_norm"] = grad_norm_before_clip.item() if hasattr(grad_norm_before_clip, 'item') else float(grad_norm_before_clip)
                    logged_metrics["train/num_training_samples"] = input_ids_lm.shape[0]
                    logged_metrics["train/cumulative_loss_tokens"] = cumulative_loss_tokens
                    # FLOPs estimation (Kaplan et al. C≈6NT): tokens = all non-padding tokens (prompt+response)
                    # student (forward+backward) = 6*params*tokens, teacher (forward only) = 2*params*tokens
                    logged_metrics["train/student_cumulative_tflops"] = 6 * _student_total_params * cumulative_loss_tokens / 1e18
                    logged_metrics["train/teacher_cumulative_tflops"] = 2 * _teacher_total_params * cumulative_loss_tokens / 1e18
                    # Rollout FLOPs (student forward-only inference)
                    _rollout_token_file = os.path.join(config.experiment.project, "temp_data", "cumulative_rollout_tokens.txt")
                    _cumulative_rollout_tokens = 0
                    if os.path.exists(_rollout_token_file):
                        with open(_rollout_token_file) as _f:
                            _cumulative_rollout_tokens = int(_f.read().strip())
                    logged_metrics["train/rollout_cumulative_tokens"] = _cumulative_rollout_tokens
                    logged_metrics["train/rollout_cumulative_tflops"] = 2 * _student_total_params * _cumulative_rollout_tokens / 1e18
                    logged_metrics["train/total_cumulative_tflops"] = (
                        logged_metrics["train/student_cumulative_tflops"]
                        + logged_metrics["train/teacher_cumulative_tflops"]
                    )
                    # GPU hours (tracked by rl.py: rollout + reward + train, excludes eval)
                    _gpu_hours_file = os.path.join(config.experiment.project, "temp_data", "cumulative_gpu_hours.txt")
                    if os.path.exists(_gpu_hours_file):
                        with open(_gpu_hours_file) as _ghf:
                            logged_metrics["train/cumulative_gpu_hours"] = float(_ghf.read().strip())
                    else:
                        logged_metrics["train/cumulative_gpu_hours"] = 0.0
                    logged_metrics["train/current_epoch"] = current_epoch
                    # if accelerator.is_main_process:
                    #     loss_val = logged_metrics.get("train/loss", 0)
                    #     grad_val = logged_metrics.get("train/grad_norm", 0)
                    #     lr_val = logged_metrics.get("train/lr", 0)
                    #     gpu_hrs = logged_metrics.get("train/cumulative_gpu_hours", 0)
                    #     print(f"Step {step+1} | loss={loss_val:.4f} grad={grad_val:.2f} lr={lr_val:.1e} gpu_hrs={gpu_hrs:.1f}")
                    if wandb_enabled:
                        accelerator.log(logged_metrics)
                        metrics = {}
                    avg_loss.reset()

                    torch.cuda.empty_cache()
            

                

    if not stepped:
        logger.warning(f"Training ended with no optimizer step taken. This may be due to having fewer batches ({len(train_dataloader_lm)}) than gradient accumulation steps ({accelerator.gradient_accumulation_steps}). Consider reducing gradient accumulation steps or increasing the number of training samples.")
    # Persist cumulative loss tokens for next RL step
    token_count_file = os.path.join(config.experiment.project, "temp_data", "cumulative_loss_tokens.txt")
    os.makedirs(os.path.dirname(token_count_file), exist_ok=True)
    with open(token_count_file, "w") as f:
        f.write(str(cumulative_loss_tokens))
    accelerator.wait_for_everyone()

    # save checkpoint at the end of training
    # Always save model weights + training state for "optimized" (latest, for resuming)
    save_checkpoint(model, tokenizer, config, accelerator, config.model.optimized_name, save_training_state_flag=True, lr_scheduler=lr_scheduler)
    if config.experiment.current_epoch % config.experiment.save_every == 0:
        # Epoch checkpoints: only save model weights (no training state to save disk)
        save_checkpoint(model, tokenizer, config, accelerator, f"epoch-{config.experiment.current_epoch}", save_training_state_flag=False)

    accelerator.end_training()






def save_checkpoint(model, tokenizer, config, accelerator, name, save_training_state_flag=True, lr_scheduler=None):
    from pathlib import Path
    import time, json, shutil, os, glob, importlib, inspect

    output_dir = Path(config.experiment.project)
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoints_total_limit = config.experiment.get("checkpoints_total_limit", None)

    if accelerator.is_main_process and checkpoints_total_limit is not None:
        ckpts = sorted(
            [d for d in output_dir.iterdir() if d.name.startswith("checkpoint")],
            key=lambda p: int(p.name.split("-")[1]),
        )
        if len(ckpts) >= checkpoints_total_limit:
            to_remove = ckpts[: len(ckpts) - checkpoints_total_limit + 1]
            logger.info(f"removing checkpoints: {', '.join(p.name for p in to_remove)}")
            for p in to_remove:
                shutil.rmtree(p, ignore_errors=True)

    save_base = output_dir / "ckpt"
    save_base.mkdir(exist_ok=True)

    accelerator.wait_for_everyone()
    if save_training_state_flag:
        save_training_state(model, config, name, lr_scheduler=lr_scheduler)
    accelerator.wait_for_everyone()

    model_to_save = accelerator.unwrap_model(model)
    _is_peft = hasattr(model_to_save, "peft_config")
    if _is_peft:
        # PEFT models: don't gather full base weights (huge) — only adapter
        # weights are trainable and are kept replicated on every rank.
        state_dict = None
    else:
        state_dict = accelerator.get_state_dict(model)

    if accelerator.is_main_process:
        save_dir = save_base / name
        tmp_dir = save_base / f"{name}.tmp"
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir, ignore_errors=True)
        tmp_dir.mkdir(parents=True, exist_ok=True)
        if _is_peft:
            # Save adapter only into <ckpt>/adapter/ — eval scripts that load
            # <ckpt>/ as a plain HF model won't see it; use train/lora_merge.py
            # if you need a merged checkpoint for eval.
            adapter_dir = tmp_dir / "adapter"
            adapter_dir.mkdir(parents=True, exist_ok=True)
            model_to_save.save_pretrained(
                str(adapter_dir),
                save_function=accelerator.save,
                safe_serialization=True,
            )
            # Persist the base config too, so resume / merge scripts can find it.
            base_to_save = getattr(model_to_save, "base_model", None)
            base_to_save = getattr(base_to_save, "model", base_to_save)
            if base_to_save is not None and hasattr(base_to_save, "config"):
                base_to_save.config.save_pretrained(str(tmp_dir))
        else:
            model_to_save.save_pretrained(
                tmp_dir,
                save_function=accelerator.save,
                state_dict=state_dict,
                safe_serialization=True,
            )
        tokenizer.save_pretrained(str(tmp_dir))

        def _copy_dynamic_modules(dst_dir, model_obj, tok_obj):
            copied = 0
            modules = set()
            for obj in [model_obj, getattr(model_obj, "config", None), tok_obj]:
                if obj is None:
                    continue
                modname = getattr(obj.__class__, "__module__", None)
                if modname:
                    modules.add(modname)

            for modname in modules:
                try:
                    mod = importlib.import_module(modname)
                    src_file = inspect.getsourcefile(mod)  # e.g. .../modeling_sdar.py
                    if not src_file or not os.path.exists(src_file):
                        continue
                    base_dir = os.path.dirname(src_file)

                    for pattern in ("modeling_*.py", "configuration_*.py", "tokenization_*.py", "processing_*.py"):
                        for fn in glob.glob(os.path.join(base_dir, pattern)):
                            dst = os.path.join(dst_dir, os.path.basename(fn))
                            if os.path.exists(dst):
                                continue
                            shutil.copy2(fn, dst)
                            copied += 1
                except Exception as e:
                    logger.warning(f"Skip copying from module {modname}: {e}")

            logger.info(f"Copied {copied} custom module files into {dst_dir}")

        _copy_dynamic_modules(str(tmp_dir), model_to_save, tokenizer)

        # Atomic swap: only delete old save_dir once tmp_dir is fully written.
        if save_dir.exists():
            old_dir = save_base / f"{name}.old"
            if old_dir.exists():
                shutil.rmtree(old_dir, ignore_errors=True)
            os.rename(save_dir, old_dir)
            os.rename(tmp_dir, save_dir)
            shutil.rmtree(old_dir, ignore_errors=True)
        else:
            os.rename(tmp_dir, save_dir)

        metadata = {
            "save_time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "current_epoch": config.experiment.current_epoch,
            "last_save_name": name,
        }
        with (save_base / "metadata.json").open("w") as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Saved model + tokenizer to {save_dir}")

def get_training_state_dir(config, name):
    return Path(config.experiment.project) / "training_state" / name


def load_training_state(model, config, name):
    training_state_dir = get_training_state_dir(config, name)
    if not training_state_dir.is_dir():
        logger.warning(
            f"No persisted training state found at {training_state_dir}. Starting with a fresh optimizer."
        )
        return False

    if not hasattr(model, "load_checkpoint"):
        logger.warning("Prepared model does not expose DeepSpeed checkpoint loading. Skipping optimizer restore.")
        return False

    load_path, _ = model.load_checkpoint(str(training_state_dir), tag=TRAINING_STATE_TAG)
    if load_path is None:
        logger.warning(
            f"DeepSpeed did not restore training state from {training_state_dir}. Starting with a fresh optimizer."
        )
        return False

    logger.info(f"Restored DeepSpeed training state from {load_path}")
    return True


def save_training_state(model, config, name, lr_scheduler=None):
    if not hasattr(model, "save_checkpoint"):
        logger.warning("Prepared model does not expose DeepSpeed checkpoint saving. Skipping optimizer persistence.")
        return

    training_state_dir = get_training_state_dir(config, name)
    parent_dir = training_state_dir.parent
    parent_dir.mkdir(parents=True, exist_ok=True)

    tmp_dir = parent_dir / f"{training_state_dir.name}.tmp"
    old_dir = parent_dir / f"{training_state_dir.name}.old"

    try:
        import torch.distributed as _dist
        is_main = (not _dist.is_initialized()) or (_dist.get_rank() == 0)
        _barrier = _dist.barrier if _dist.is_initialized() else (lambda: None)
    except Exception:
        is_main = True
        _barrier = lambda: None

    # Rank 0 clears any stale tmp dir and prepares a fresh one
    if is_main:
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir, ignore_errors=True)
        tmp_dir.mkdir(parents=True, exist_ok=True)
    _barrier()

    # All ranks write new state into tmp_dir; real dir stays intact until swap
    model.save_checkpoint(
        str(tmp_dir),
        tag=TRAINING_STATE_TAG,
        client_state={
            "save_time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "current_epoch": config.experiment.current_epoch,
        },
    )

    if is_main and lr_scheduler is not None:
        lr_path = tmp_dir / "lr_scheduler.pt"
        torch.save(lr_scheduler.state_dict(), lr_path)
        logger.info(f"Saved LR scheduler state to {lr_path}")

    _barrier()

    # Atomic swap on rank 0 — mirrors the model-save .old pattern
    if is_main:
        if old_dir.exists():
            shutil.rmtree(old_dir, ignore_errors=True)
        if training_state_dir.exists():
            os.rename(training_state_dir, old_dir)
        os.rename(tmp_dir, training_state_dir)
        shutil.rmtree(old_dir, ignore_errors=True)
        logger.info(f"Saved DeepSpeed training state to {training_state_dir}")
    _barrier()


def load_lr_scheduler_state(lr_scheduler, config, name):
    training_state_dir = get_training_state_dir(config, name)
    lr_path = training_state_dir / "lr_scheduler.pt"
    if not lr_path.is_file():
        logger.warning(f"No LR scheduler state at {lr_path}. Starting scheduler from step 0.")
        return False
    state_dict = torch.load(str(lr_path), map_location="cpu")
    lr_scheduler.load_state_dict(state_dict)
    logger.info(f"Restored LR scheduler state from {lr_path}")
    return True
















# ══════════════════════════════════════════════════════════════════════
# Importable API (for in-process use from rl.py under accelerate launch)
# ══════════════════════════════════════════════════════════════════════

def init_training(config):
    """One-time initialization of the training engine.

    Creates Accelerator, loads student + teacher models, optimizer, LR scheduler.
    Returns a state dict used by train_one_step().
    Must be called inside an `accelerate launch` distributed context.
    """
    wandb_enabled = bool(config.wandb.get("enabled", True))
    project_name = config.experiment.project
    _lora_enabled = getattr(getattr(config.training, "lora", None), "enabled", False)
    _lora_adapter_resume = None
    # On resume (current_epoch > 1 or start_from_scratch=False), load student
    # weights from the saved checkpoint rather than the original pretrained model.
    if config.experiment.current_epoch > 1 or not config.experiment.start_from_scratch:
        if _lora_enabled:
            # LoRA resume: keep base = original pretrained model; load adapter
            # from the previous save's `adapter/` subdir.
            pretrained_model = config.model.pretrained_model
            _lora_adapter_resume = os.path.join(
                project_name, "ckpt", config.model.optimized_name, "adapter"
            )
            if not os.path.isdir(_lora_adapter_resume):
                logger.warning(f"[LoRA] resume adapter not found at {_lora_adapter_resume}; "
                               f"starting fresh adapter on top of base.")
                _lora_adapter_resume = None
        else:
            pretrained_model = os.path.join(project_name, "ckpt", config.model.optimized_name)
            if not os.path.exists(pretrained_model):
                logger.warning(f"Resume requested but checkpoint not found at {pretrained_model}; falling back to {config.model.pretrained_model}")
                pretrained_model = config.model.pretrained_model
    else:
        pretrained_model = config.model.pretrained_model

    if config.training.enable_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

    config.experiment.logging_dir = str(Path(config.experiment.project) / "logs")
    _zero_stage = int(config.training.get("zero_stage", 3))
    _offload_opt = config.training.get("offload_optimizer_device", "cpu")
    _offload_param = config.training.get("offload_param_device", "cpu")
    _student_kwargs = dict(
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        gradient_clipping=config.training.max_grad_norm,
        zero_stage=_zero_stage,
        offload_optimizer_device=_offload_opt,
    )
    if _zero_stage == 3:
        _student_kwargs.update(
            offload_param_device=_offload_param,
            zero3_init_flag=True,
            zero3_save_16bit_model=True,
        )
    # Teacher is frozen (inference-only). Stage 2 requires an optimizer, so
    # force teacher to stage 3 regardless of student stage.
    _teacher_kwargs = dict(
        zero_stage=3,
        offload_param_device=_offload_param,
        zero3_init_flag=True,
    )
    deepspeed_plugins = {
        "student": DeepSpeedPlugin(**_student_kwargs),
        "teacher": DeepSpeedPlugin(**_teacher_kwargs),
    }
    accelerator = Accelerator(
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        mixed_precision=config.training.mixed_precision,
        log_with="wandb" if wandb_enabled else None,
        project_dir=config.experiment.logging_dir,
        split_batches=False,
        deepspeed_plugins=deepspeed_plugins,
    )

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        set_verbosity_info()
    else:
        set_verbosity_error()

    if accelerator.is_main_process and wandb_enabled:
        run_id = config.wandb.get("run_id", None) or os.getenv("WANDB_RUN_ID", None)
        if run_id is None:
            raise ValueError("WANDB_RUN_ID environment variable is not set.")

        wandb_init_kwargs = dict(id=run_id, resume="must")
        wandb_config = {k: v for k, v in flatten_omega_conf(config, resolve=True)}
        wandb_config.pop("experiment.resume_from_checkpoint", None)

        wandb_project = config.wandb.get("project") or config.experiment.project
        accelerator.init_trackers(
            wandb_project,
            config=wandb_config,
            init_kwargs={"wandb": wandb_init_kwargs},
        )
        wandb.define_metric("train/current_epoch")
        wandb.define_metric("*", step_metric="train/current_epoch")

    if accelerator.is_main_process:
        os.makedirs(config.experiment.project, exist_ok=True)
        config_path = Path(config.experiment.project) / "config.yaml"
        logging.info(f"Saving config to {config_path}")
        OmegaConf.save(config, config_path)

    # Prefer TRACERL_SEED (exported by rl.py as base_seed + current_epoch) so picks vary per RL step;
    # fall back to config.training.seed.
    _env_seed = os.environ.get("TRACERL_SEED")
    _step_seed = int(_env_seed) if _env_seed is not None else config.training.seed
    if _step_seed is not None:
        set_seed(_step_seed)

    logger.info("Loading models and optimizer")

    tokenizer = AutoTokenizer.from_pretrained(pretrained_model, trust_remote_code=True)
    uni_prompting = UniversalPrompting(tokenizer, max_prompt_len=config.training.max_prompt_len,
                                       max_gen_length=config.training.max_gen_length,
                                       ignore_id=-100)

    model_base = getattr(config.model, "model_base", "sdar")
    if model_base == "sdar":
        model = SDARForCausalLM.from_pretrained(pretrained_model, trust_remote_code=True, torch_dtype="auto")
    elif model_base == "bd3lm":
        import transformers as _tf
        from transformers.cache_utils import DynamicCache
        from transformers.modeling_outputs import BaseModelOutputWithPast
        from transformers.modeling_attn_mask_utils import _prepare_4d_attention_mask
        from torch import nn as _nn

        class _A2DQwen3Config(_tf.Qwen3Config):
            model_type = "a2d-qwen3"

        class _A2DQwen3Model(_tf.Qwen3Model):
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
                    _past = past_key_values.get_seq_length() if past_key_values is not None else 0
                    cache_position = torch.arange(_past, _past + inputs_embeds.shape[1], device=inputs_embeds.device)
                if position_ids is None:
                    position_ids = cache_position.unsqueeze(0)
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
                return BaseModelOutputWithPast(last_hidden_state=hidden_states,
                                               past_key_values=past_key_values if use_cache else None)

        class _A2DQwen3LMHeadModel(_tf.Qwen3ForCausalLM):
            config_class = _A2DQwen3Config
            def __init__(self, config):
                _tf.Qwen3PreTrainedModel.__init__(self, config)
                self.model = _A2DQwen3Model(config)
                self.vocab_size = config.vocab_size
                self.lm_head = _nn.Linear(config.hidden_size, config.vocab_size, bias=False)
                self.post_init()

        _tf.AutoConfig.register("a2d-qwen3", _A2DQwen3Config)
        _tf.AutoModel.register(_A2DQwen3Config, _A2DQwen3LMHeadModel)
        _tf.AutoModelForMaskedLM.register(_A2DQwen3Config, _A2DQwen3LMHeadModel)

        from transformers import AutoModelForMaskedLM
        model = AutoModelForMaskedLM.from_pretrained(pretrained_model, trust_remote_code=False, torch_dtype="auto")
    else:
        model = AutoModelForCausalLM.from_pretrained(pretrained_model, trust_remote_code=True, torch_dtype="auto")

    if hasattr(model, "config"):
        model.config.fuse_cross_entropy = False
        if getattr(config.training, "student_arm_shift", False):
            model.config.arm_shift = True

    # Optionally wrap student with LoRA (default off; only the adapter is
    # trainable when enabled).
    model = _maybe_wrap_lora(model, config, project_name, adapter_resume_dir=_lora_adapter_resume)

    if config.training.gradient_checkpointing_enable:
        model.gradient_checkpointing_enable()
        if hasattr(model, "config"):
            model.config.use_cache = False
    else:
        model = model.to(accelerator.device)

    mask_id = tokenizer.mask_token_id
    if mask_id is None:
        mask_id = tokenizer.pad_token_id
    pad_id = tokenizer.pad_token_id

    optimizer_config = config.optimizer.params
    no_decay = ["bias", "layer_norm.weight", "mlm_ln.weight", "embeddings.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if
                       p.requires_grad and not any(nd in n for nd in no_decay)],
            "weight_decay": optimizer_config.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if
                       p.requires_grad and any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]

    optimizer_type = config.optimizer.name
    if optimizer_type == "adamw":
        optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=optimizer_config.learning_rate,
            betas=(optimizer_config.beta1, optimizer_config.beta2),
            weight_decay=optimizer_config.weight_decay,
            eps=optimizer_config.epsilon,
        )
    else:
        raise ValueError(f"Optimizer {optimizer_type} not supported")

    # DeepSpeed requires train_micro_batch_size_per_gpu when no dataloader is
    # passed to prepare(). Set it explicitly so we can prepare model+optimizer
    # at init time (dataloader is created per step in train_one_step).
    accelerator.state.deepspeed_plugin.deepspeed_config["train_micro_batch_size_per_gpu"] = config.training.batch_size_lm

    # Prepare model + optimizer with accelerator
    model, optimizer = accelerator.prepare(model, optimizer)

    # Load training state if resuming
    should_resume = (config.experiment.current_epoch > 1 or not config.experiment.start_from_scratch)
    if should_resume:
        load_training_state(model, config, config.model.optimized_name)

    # Prepare teacher model
    _loss_type = getattr(config.training, "loss_type", "kl")
    if _loss_type in ("kl", "jsd"):
        accelerator.state.select_deepspeed_plugin("teacher")
        accelerator.state.deepspeed_plugin.deepspeed_config["train_micro_batch_size_per_gpu"] = 1
        teacher_model = AutoModelForCausalLM.from_pretrained(
            config.model.teacher_model, trust_remote_code=True, torch_dtype="auto"
        )
        teacher_model.requires_grad_(False)
        teacher_model = accelerator.prepare(teacher_model)
        teacher_model.eval()
        accelerator.state.select_deepspeed_plugin("student")
    else:
        teacher_model = None
        logger.info(f"{_loss_type} mode: skipping teacher model loading")

    _student_total_params = sum(getattr(p, 'ds_numel', p.numel()) for p in model.parameters())
    _teacher_total_params = sum(getattr(p, 'ds_numel', p.numel()) for p in teacher_model.parameters()) if teacher_model is not None else 0
    logger.info(f"Param counts: student={_student_total_params/1e6:.1f}M, teacher={_teacher_total_params/1e6:.1f}M")

    # Load cumulative loss token count
    _token_count_file = os.path.join(config.experiment.project, "temp_data", "cumulative_loss_tokens.txt")
    if os.path.exists(_token_count_file):
        with open(_token_count_file) as f:
            cumulative_loss_tokens = int(f.read().strip())
    else:
        cumulative_loss_tokens = 0

    # ── LR scheduler (created once, steps continuously across RL steps) ──
    # Estimate inner optimizer steps per RL step from config:
    #   num_training_rows ≈ num_task_per_step * rounds_per_prompt
    #     rounds_per_prompt = block_size (default TraceRL) or 1 (one_state_per_block=True)
    #   total_batch_size = batch_size_lm * num_processes * gradient_accumulation_steps
    #   steps_per_rl_step = ceil(num_training_rows / total_batch_size) * num_train_epochs
    _num_tasks = config.rollout.num_task_per_step
    _block_size = config.training.block_size
    _one_state_per_block = getattr(config.training, "one_state_per_block", False)
    _rounds_per_prompt = 1 if _one_state_per_block else _block_size
    _est_rows = _num_tasks * _rounds_per_prompt
    _total_batch_size = config.training.batch_size_lm * accelerator.num_processes * config.training.gradient_accumulation_steps
    _num_inner_epochs = config.training.num_train_epochs
    _est_steps_per_rl_step = math.ceil(_est_rows / _total_batch_size) * _num_inner_epochs

    total_rl_steps = config.experiment.total_step
    _scheduler_name = str(config.lr_scheduler.scheduler).lower()
    if total_rl_steps > 0:
        max_train_steps = _est_steps_per_rl_step * total_rl_steps + 1
    elif _scheduler_name in ("constant", "constant_with_warmup"):
        max_train_steps = None
    else:
        raise ValueError(
            f"lr_scheduler='{_scheduler_name}' needs a finite training horizon, "
            "but experiment.total_step<=0 and no epoch-mode override is active "
            "(set dataset.num_data_epochs>=1 or experiment.total_step>0, or use a constant scheduler)."
        )

    _decay_steps = getattr(config.lr_scheduler.params, "decay_steps", None)
    if _decay_steps is not None:
        _decay_steps = int(_decay_steps) * _est_steps_per_rl_step
    _warmup_rl_steps = config.lr_scheduler.params.warmup_steps
    _warmup_inner_steps = int(_warmup_rl_steps) * _est_steps_per_rl_step
    lr_scheduler = get_scheduler(
        config.lr_scheduler.scheduler,
        optimizer=optimizer,
        num_training_steps=max_train_steps,
        num_warmup_steps=_warmup_inner_steps,
        min_lr_scale=config.lr_scheduler.params.min_lr_scale,
        decay_steps=_decay_steps,
    )
    lr_scheduler = accelerator.prepare(lr_scheduler)
    logger.info(f"LR scheduler: {config.lr_scheduler.scheduler}, warmup={_warmup_inner_steps} inner steps, "
                f"total={max_train_steps} inner steps, est {_est_steps_per_rl_step} inner steps/RL step")

    if should_resume:
        load_lr_scheduler_state(lr_scheduler, config, config.model.optimized_name)

    return {
        "accelerator": accelerator,
        "model": model,
        "teacher_model": teacher_model,
        "tokenizer": tokenizer,
        "optimizer": optimizer,
        "lr_scheduler": lr_scheduler,
        "uni_prompting": uni_prompting,
        "mask_id": mask_id,
        "pad_id": pad_id,
        "model_base": model_base,
        "student_total_params": _student_total_params,
        "teacher_total_params": _teacher_total_params,
        "cumulative_loss_tokens": cumulative_loss_tokens,
        "wandb_enabled": wandb_enabled,
    }


def train_one_step(state, config):
    """Run one training step using persistent state from init_training().

    Reads data from {project_name}/temp_data/{optimization_data}.json,
    trains for num_train_epochs, saves checkpoint.
    """
    accelerator = state["accelerator"]
    model = state["model"]
    teacher_model = state["teacher_model"]
    tokenizer = state["tokenizer"]
    optimizer = state["optimizer"]
    lr_scheduler = state["lr_scheduler"]
    uni_prompting = state["uni_prompting"]
    # Update max_gen_length per step (supports max_token schedule)
    uni_prompting.max_gen_length = config.training.max_gen_length
    mask_id = state["mask_id"]
    pad_id = state["pad_id"]
    model_base = state["model_base"]
    _student_total_params = state["student_total_params"]
    _teacher_total_params = state["teacher_total_params"]
    cumulative_loss_tokens = state["cumulative_loss_tokens"]
    wandb_enabled = state["wandb_enabled"]

    project_name = config.experiment.project
    current_epoch = config.experiment.current_epoch

    # Read training data
    data_path = project_name + "/temp_data/" + config.dataset.optimization_data + ".json"
    with open(data_path, 'r') as f:
        dataset_load = json.load(f)

    if len(dataset_load) == 0:
        logger.warning("No training data after filtering. Skipping this training step.")
        return

    prompt_list = []
    response_list = []
    step_map_list = []
    reward_list = []
    source_list = []
    for x in dataset_load:
        prompt_list.append(x["prompt"])
        response_list.append(x["response"])
        reward_list.append(x["reward"])
        source_list.append(x.get("source", "student"))

    input_ids_lm, _, start_pos, drop_num = uni_prompting((prompt_list, response_list))

    _, L = input_ids_lm.shape
    L0 = start_pos
    L1 = L - L0
    post_num = config.training.post_num

    for x in dataset_load:
        if "step_map" not in x.keys() or len(x["step_map"]) == 0:
            step_map_list.append([j for j in range(L1)])
        else:
            step_map_i = x["step_map"]
            if len(step_map_i) > L1:
                step_map_i = step_map_i[:L1]
            else:
                step_map_i = step_map_i + [max(step_map_i) + 1] * (L1 - len(step_map_i))
            step_map_list.append(step_map_i)

    # ── Helper functions (closures over per-step state) ──

    def collapse_k_unique(lst, k):
        if k <= 0:
            raise ValueError("k must be > 0")
        uniq = sorted(set(lst))
        mapping = {}
        n = len(uniq)
        for idx, val in enumerate(uniq):
            group = idx // k
            end_idx = min((group + 1) * k - 1, n - 1)
            rep = uniq[end_idx]
            mapping[val] = rep
        return [mapping[x] for x in lst]

    def make_basic_block_attention(N, start_pos, block_size):
        B = 1
        L0 = start_pos
        L1 = (N - L0) // 2
        assert L0 + 2 * L1 == N
        bias = torch.full((B, 1, N, N), 0)
        rows = torch.arange(L0 + L1, L0 + 2 * L1)
        rows_token = torch.arange(L0, L0 + L1)
        for bi in range((L1 + block_size - 1) // block_size):
            left_end = L0 + min((bi) * block_size, L1)
            right_start = L0 + L1 + (left_end - L0)
            i_start = bi * block_size
            i_end = min((bi + 1) * block_size, L1)
            block_rows = rows[i_start:i_end]
            bias[:, :, block_rows.unsqueeze(-1), 0:left_end] = 1
            bias[:, :, block_rows.unsqueeze(-1), right_start:(right_start + block_size)] = 1
            block_rows = rows_token[i_start:i_end]
            left_end = L0 + min((bi + 1) * block_size, L1)
            bias[:, :, block_rows.unsqueeze(-1), 0:left_end] = 1
        if L0 > 0:
            num_blocks_pre = (L0 + block_size - 1) // block_size
            for bi in range(num_blocks_pre):
                row_end = max(L0 - bi * block_size, 0)
                row_start = max(L0 - (bi + 1) * block_size, 0)
                if row_end > row_start:
                    block_rows = torch.arange(row_start, row_end)
                    bias[:, :, block_rows.unsqueeze(-1), 0:row_end] = 1
        return bias

    basic_block_attention = make_basic_block_attention(L0 + 2 * L1, start_pos, config.training.block_size)
    basic_block_attention = basic_block_attention.cpu()

    def process_pad(attn, input_ids):
        N = L0 + 2 * L1
        device = input_ids.device
        cols = torch.arange(N, device=device)
        key_mask = (cols < start_pos).unsqueeze(0) & (input_ids == pad_id)
        attn.masked_fill_(key_mask[:, None, None, :], 0)
        A = attn[:, 0]
        bad = (A.sum(dim=-1) == 0) & (torch.arange(A.size(1), device=A.device).unsqueeze(0) < start_pos)
        b, r = bad.nonzero(as_tuple=True)
        A[b, r, :] = 0; A[b, r, r] = 1
        attn = attn.bool()
        return attn

    def one_round_vectorized(input_ids_b, step_map_b, L0, L1, block_size, mask_id):
        device = input_ids_b.device
        NB = (L1 + block_size - 1) // block_size
        step_pad = torch.full((NB * block_size,), -1, dtype=torch.long, device=device)
        step_pad[:L1] = step_map_b
        step_blk = step_pad.view(NB, block_size)
        valid = step_blk.ge(0)
        big = torch.iinfo(step_blk.dtype).max
        tmp = step_blk.masked_fill(~valid, big)
        min_vals, _ = tmp.min(dim=1, keepdim=True)
        pmask_blk = step_blk.eq(min_vals) & valid
        if not pmask_blk.any():
            return None, None, step_map_b, False
        ge_mask_blk = step_blk.ge(min_vals) & valid
        pmask_tail = pmask_blk.view(-1)[:L1]
        ge_mask_tail = ge_mask_blk.view(-1)[:L1]
        pmask_b = torch.zeros(L0 + L1, dtype=torch.bool, device=device)
        pmask_b[L0:] = pmask_tail
        tail = input_ids_b[L0:L0+L1].clone()
        tail[ge_mask_tail] = mask_id
        extended_input_ids_b = torch.empty(L0 + L1 + L1, dtype=input_ids_b.dtype, device=device)
        extended_input_ids_b[:L0+L1] = input_ids_b
        extended_input_ids_b[L0+L1:] = tail
        new_step_map_b = step_map_b.clone()
        new_step_map_b[pmask_tail] = -1
        return extended_input_ids_b, pmask_b, new_step_map_b, True

    # ── collect_training_data ──
    # (Same as in main(), uses closures over start_pos, L0, L1, etc.)
    def collect_training_data(input_ids, step_map_list, reward, source_list=None):
        B, L = input_ids.shape
        block_size = config.training.block_size
        lower = config.training.lower_p
        upper = config.training.upper_p
        has_mixed = source_list is not None and any(s == "teacher" for s in source_list)

        if has_mixed:
            extended_input_ids_list, pmask_list, reward_list = [], [], []
            for b in range(B):
                if source_list[b] == "teacher":
                    reward_list.append(reward[b])
                    extended_input_ids_b = input_ids[b]
                    pmask_b = torch.zeros(start_pos, dtype=torch.bool)
                    for j in range(int((L1 - 1) / block_size) + 1):
                        start = j * block_size
                        end = min(L1, (j + 1) * block_size)
                        pmask_b_j = torch.rand(end - start) <= torch.empty(end - start).uniform_(lower, upper)
                        pmask_b = torch.cat([pmask_b, pmask_b_j], dim=0)
                        noise_b_j = input_ids[b, (L0 + start):(L0 + end)].clone()
                        noise_b_j = noise_b_j.masked_fill_(pmask_b_j, mask_id)
                        extended_input_ids_b = torch.cat([extended_input_ids_b, noise_b_j], dim=0)
                    extended_input_ids_list.append(extended_input_ids_b)
                    pmask_list.append(pmask_b)
                else:
                    step_map_i = step_map_list[b]
                    for j in range(int((L1 - 1) / block_size) + 1):
                        s, e = j * block_size, min(L1, (j + 1) * block_size)
                        step_map_list[b][s:e] = collapse_k_unique(step_map_i[s:e], config.training.shrink)
                    step_b = torch.as_tensor(step_map_list[b], dtype=torch.long)
                    while True:
                        out = one_round_vectorized(input_ids[b], step_b, L0, L1, block_size, mask_id)
                        extended_b, pmask_b, step_b, has_any = out
                        if not has_any:
                            break
                        extended_input_ids_list.append(extended_b)
                        pmask_list.append(pmask_b)
                        reward_list.append(reward[b])

        elif config.training.method == "random_masking":
            extended_input_ids_list, pmask_list, reward_list = [], [], []
            for b in range(B):
                reward_list.append(reward[b])
                extended_input_ids_b = input_ids[b]
                pmask_b = torch.zeros(start_pos, dtype=torch.bool)
                for j in range(int((L1 - 1) / block_size) + 1):
                    start = j * block_size
                    end = min(L1, (j + 1) * block_size)
                    pmask_b_j = torch.rand(end - start) <= torch.empty(end - start).uniform_(lower, upper)
                    pmask_b = torch.cat([pmask_b, pmask_b_j], dim=0)
                    noise_b_j = input_ids[b, (L0 + start):(L0 + end)].clone()
                    noise_b_j = noise_b_j.masked_fill_(pmask_b_j, mask_id)
                    extended_input_ids_b = torch.cat([extended_input_ids_b, noise_b_j], dim=0)
                extended_input_ids_list.append(extended_input_ids_b)
                pmask_list.append(pmask_b)

        if config.training.method == "coupled" and not has_mixed:
            extended_input_ids_list, pmask_list, reward_list = [], [], []
            coupled_input_ids_list, coupled_pmask_list, coupled_reward_list = [], [], []
            for b in range(B):
                reward_list.append(reward[b])
                coupled_reward_list.append(reward[b])
                extended_input_ids_b = input_ids[b]
                pmask_b = torch.zeros(start_pos, dtype=torch.bool)
                coupled_input_ids_b = input_ids[b]
                coupled_pmask_b = torch.zeros(start_pos, dtype=torch.bool)
                for j in range(int((L1 - 1) / block_size) + 1):
                    start = j * block_size
                    end = min(L1, (j + 1) * block_size)
                    pmask_b_j = torch.rand(end - start) <= torch.empty(end - start).uniform_(lower, upper)
                    pmask_b = torch.cat([pmask_b, pmask_b_j], dim=0)
                    coupled_pmask_b = torch.cat([coupled_pmask_b, ~pmask_b_j], dim=0)
                    noise_b_j = input_ids[b, (L0 + start):(L0 + end)].clone()
                    noise_b_j = noise_b_j.masked_fill_(pmask_b_j, mask_id)
                    coupled_noise_b_j = input_ids[b, (L0 + start):(L0 + end)].clone()
                    coupled_noise_b_j = coupled_noise_b_j.masked_fill_(~pmask_b_j, mask_id)
                    extended_input_ids_b = torch.cat([extended_input_ids_b, noise_b_j], dim=0)
                    coupled_input_ids_b = torch.cat([coupled_input_ids_b, coupled_noise_b_j], dim=0)
                extended_input_ids_list.append(extended_input_ids_b)
                pmask_list.append(pmask_b)
                coupled_input_ids_list.append(coupled_input_ids_b)
                coupled_pmask_list.append(coupled_pmask_b)
            extended_input_ids_list += coupled_input_ids_list
            pmask_list += coupled_pmask_list
            reward_list += coupled_reward_list

        elif config.training.method == "TraceRL" and not has_mixed:
            for b in range(B):
                step_map_i = step_map_list[b]
                for j in range(int((L1 - 1) / block_size) + 1):
                    start = j * block_size
                    end = min(L1, (j + 1) * block_size)
                    step_map_list[b][start:end] = collapse_k_unique(step_map_i[start:end], config.training.shrink)
            step_map = torch.as_tensor(step_map_list, dtype=torch.long)
            assert step_map.shape[1] == L1
            extended_input_ids_list, pmask_list, reward_list = [], [], []
            one_state_per_block = getattr(config.training, "one_state_per_block", False)
            random_mask = getattr(config.rollout, "random_mask", False)
            for b in range(B):
                # rollout.random_mask: synthesize one row by random-masking the
                # clean response, ignoring the rollout trajectory.
                if random_mask:
                    ext_b, pm_b = _combine_rounds_random_mask(
                        input_ids[b], L0, L1, block_size, mask_id,
                    )
                    extended_input_ids_list.append(ext_b)
                    pmask_list.append(pm_b)
                    reward_list.append(reward[b])
                    continue
                step_b = step_map[b]
                per_prompt_ext, per_prompt_pm = [], []
                while True:
                    out = one_round_vectorized(
                        input_ids_b=input_ids[b], step_map_b=step_b,
                        L0=L0, L1=L1, block_size=block_size, mask_id=mask_id,
                    )
                    extended_b, pmask_b, step_b, has_any = out
                    if not has_any:
                        break
                    per_prompt_ext.append(extended_b)
                    per_prompt_pm.append(pmask_b)
                if not per_prompt_ext:
                    continue
                if one_state_per_block:
                    ext_b, pm_b = _combine_rounds_one_state_per_block(
                        per_prompt_ext, per_prompt_pm, input_ids[b], L0, L1, block_size,
                    )
                    extended_input_ids_list.append(ext_b)
                    pmask_list.append(pm_b)
                    reward_list.append(reward[b])
                else:
                    extended_input_ids_list.extend(per_prompt_ext)
                    pmask_list.extend(per_prompt_pm)
                    reward_list.extend([reward[b]] * len(per_prompt_ext))

        extended_input_ids = torch.stack(extended_input_ids_list, dim=0)
        p_mask = torch.stack(pmask_list, dim=0).to(torch.bool)
        pad_resp = (extended_input_ids[:, :L] == pad_id) & p_mask
        if post_num is not None:
            cum_pad = torch.cumsum(pad_resp.int(), dim=1)
            p_mask &= ~(pad_resp & (cum_pad > post_num))
        labels = extended_input_ids[:, :L].clone()
        idx = torch.arange(L).unsqueeze(0).expand(extended_input_ids.shape[0], -1)
        valid = (idx >= start_pos) | extended_input_ids[:, :L].ne(pad_id)
        tok_idx = valid.long().cumsum(dim=-1) - 1
        tok_idx = tok_idx.masked_fill(~valid, 1)
        tok_idx_resp = tok_idx[:, start_pos:]
        tok_idx_ext = torch.cat([tok_idx, tok_idx_resp], dim=1)
        keep = p_mask.view(p_mask.size(0), -1).any(dim=1)
        idx = keep.nonzero(as_tuple=True)[0]
        extended_input_ids = extended_input_ids[idx]
        p_mask = p_mask[idx]
        tok_idx_ext = tok_idx_ext[idx]
        labels = labels[idx]
        reward_list = [reward_list[i] for i in idx.tolist()]
        return extended_input_ids, p_mask, tok_idx_ext, labels, reward_list

    extended_input_ids, p_mask, tok_idx_ext, labels, rewards = collect_training_data(
        input_ids_lm, step_map_list, reward_list, source_list=source_list
    )

    def simple_collate(batch):
        idx, extended_input_ids, p_mask, tok_idx_ext, labels, reward = zip(*batch)
        return {
            "ids": torch.tensor(idx),
            "extended_input_ids": torch.stack(extended_input_ids),
            "p_mask": torch.stack(p_mask),
            "tok_idx_ext": torch.stack(tok_idx_ext),
            "labels": torch.stack(labels),
            "reward": reward,
        }

    dataset_lm = TrainDataset(extended_input_ids, p_mask, tok_idx_ext, labels, rewards)
    logger.info(f"  Num training rows (after expand+filter) = {len(dataset_lm)} (from {input_ids_lm.shape[0]} responses)")

    num_train_epochs = config.training.num_train_epochs

    train_dataloader_lm = DataLoader(
        dataset_lm,
        batch_size=config.training.batch_size_lm,
        sampler=None,
        shuffle=True,
        collate_fn=simple_collate,
        num_workers=0,
    )
    train_dataloader_lm = accelerator.prepare(train_dataloader_lm)

    if len(train_dataloader_lm) < accelerator.gradient_accumulation_steps:
        print(
            f"Number of batches ({len(train_dataloader_lm)}) is less than gradient accumulation steps "
            f"({accelerator.gradient_accumulation_steps}). Please reduce gradient accumulation steps "
            "or increase the number of training samples."
        )

    # ── Forward process functions (closures over model, teacher, config, etc.) ──
    import torch.nn.functional as F

    def forward_process(extended_input_ids, p_mask, tok_idx_ext, labels, adv, logp_old_tok):
        adv = torch.as_tensor(adv, device=extended_input_ids.device).detach()
        B, _L = p_mask.shape
        device = extended_input_ids.device
        attention_mask = basic_block_attention.clone()
        attention_mask = attention_mask.repeat_interleave(B, dim=0).to(device)
        attention_mask = process_pad(attention_mask, extended_input_ids)
        full_logits = model(input_ids=extended_input_ids, attention_mask=attention_mask, position_ids=tok_idx_ext).logits
        logits = torch.cat([full_logits[:, :L0, :], full_logits[:, L0 + L1:, :]], dim=1)
        if getattr(config.training, "student_arm_shift", False):
            logits = logits.roll(dims=1, shifts=1)
        log_probs = F.log_softmax(logits.float(), dim=-1)

        prompt_response = extended_input_ids[:, :L0 + L1]
        extended_input_ids_sampled = extended_input_ids.clone()
        masked_input_tokens = extended_input_ids_sampled == tokenizer.mask_token_id
        teacher_fill = getattr(config.training, "teacher_fill", "argmax_fill")
        if teacher_fill == "argmax_fill":
            masked_logits = full_logits[masked_input_tokens]
            sampled_response_tokens = masked_logits.argmax(dim=-1)
            extended_input_ids_sampled[masked_input_tokens] = sampled_response_tokens
        elif teacher_fill == "clean_fill":
            x0_repeated = torch.cat([extended_input_ids[:, :L0 + L1], extended_input_ids[:, L0:L0 + L1]], dim=1)
            extended_input_ids_sampled[masked_input_tokens] = x0_repeated[masked_input_tokens]

        causal_attention_mask_bool = attention_mask
        prompt_and_x0_mask = torch.ones(1, 1, L0 + L1, L0 + L1, dtype=torch.bool, device=device)
        prompt_and_x0_mask = torch.tril(prompt_and_x0_mask, diagonal=0).repeat(B, 1, 1, 1)
        xt_mask = torch.ones(1, 1, L1, L1, dtype=torch.bool, device=device)
        xt_mask = torch.tril(xt_mask, diagonal=0).repeat(B, 1, 1, 1)
        causal_attention_mask_bool[:, :, :L0 + L1, :L0 + L1] &= prompt_and_x0_mask
        causal_attention_mask_bool[:, :, L0 + L1:, L0 + L1:] &= xt_mask
        causal_attention_mask = torch.full(causal_attention_mask_bool.shape, float("-inf"), dtype=torch.bfloat16, device=device)
        causal_attention_mask[causal_attention_mask_bool] = 0.0

        with torch.no_grad():
            block_size = config.training.block_size
            logits_teacher_full = teacher_model(
                input_ids=extended_input_ids_sampled,
                attention_mask=causal_attention_mask.bfloat16(),
                position_ids=tok_idx_ext
            ).logits
            logits_teacher = torch.cat([logits_teacher_full[:, :L0, :], logits_teacher_full[:, L0 + L1:, :]], dim=1)
            replace_x0_indices = torch.arange(start=L0 + block_size - 1, end=L0 + L1, step=block_size)
            logits_teacher[:, replace_x0_indices] = logits_teacher_full[:, replace_x0_indices]
            logits_teacher = logits_teacher.roll(dims=1, shifts=1)
            teacher_logprobs = F.log_softmax(logits_teacher.float(), dim=-1)

        _divergence_type = getattr(config.training, "loss_type", "kl")
        if _divergence_type == "jsd":
            _jsd_alpha = getattr(config.training, "jsd_alpha", 0.5)
            teacher_probs = teacher_logprobs.exp()
            student_probs = log_probs.exp()
            M = _jsd_alpha * teacher_probs + (1.0 - _jsd_alpha) * student_probs
            log_M = M.log()
            kl_div = _jsd_alpha * (teacher_probs * (teacher_logprobs - log_M)).sum(dim=-1) \
                   + (1.0 - _jsd_alpha) * (student_probs * (log_probs - log_M)).sum(dim=-1)
        else:
            kl_div = (teacher_logprobs.exp() * (teacher_logprobs - log_probs)).sum(dim=-1)

        prompt = extended_input_ids[:, :L0]
        response_noised = extended_input_ids[:, L0 + L1:]
        prompt_mask = torch.cat([torch.zeros_like(prompt), torch.ones_like(response_noised)], dim=1).bool()

        im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
        is_im_end = prompt_response.eq(im_end_id)
        is_response_im_end = is_im_end & prompt_mask
        im_end_cumsum = is_response_im_end.cumsum(dim=1)
        im_end_shifted = F.pad(im_end_cumsum[:, :-1], (1, 0))
        im_end_mask = im_end_shifted.eq(0)
        if getattr(config.training, "exclude_im_end", False):
            im_end_mask = im_end_mask & ~is_response_im_end

        prompt_noised_response = torch.cat([prompt, response_noised], dim=1)
        pad_mask_t = prompt_noised_response.ne(tokenizer.pad_token_id)
        masked_token_mask = prompt_noised_response.eq(tokenizer.mask_token_id)
        response_mask = prompt_mask & pad_mask_t & im_end_mask
        loss_mask = response_mask & masked_token_mask
        kl_div_mask = kl_div * loss_mask
        loss_unreduced = kl_div_mask.sum(dim=1) / loss_mask.sum(dim=1).clamp_min(1.0)

        num_response_tokens = response_mask.sum(dim=1)
        num_masked_tokens = (response_mask & masked_token_mask).sum(dim=1)
        frac_masked = num_masked_tokens.float() / num_response_tokens.float().clamp(min=1)
        loss_by_masking_rate = {}
        for frac, loss_i in zip(frac_masked, loss_unreduced):
            metric_name = f"train/loss_masked_frac_{frac:.1f}"
            num_rate, total_loss = loss_by_masking_rate.get(metric_name, (0, 0))
            loss_by_masking_rate[metric_name] = (num_rate + 1, total_loss + loss_i.item())
        loss = loss_unreduced.mean()
        num_forward_tokens = prompt_response.ne(tokenizer.pad_token_id).sum().item()
        return loss, loss_by_masking_rate, num_forward_tokens

    def forward_process_causal(extended_input_ids, p_mask, tok_idx_ext, labels, adv, logp_old_tok):
        B, _L = labels.shape
        device = labels.device
        input_ids = labels
        pad_mask_t = input_ids.ne(pad_id)
        if model_base == "qwen":
            attn_mask = pad_mask_t
        else:
            attn_mask = torch.tril(torch.ones(_L, _L, dtype=torch.bool, device=device))
            attn_mask = attn_mask[None, None, :, :] & pad_mask_t[:, None, None, :]
        student_logits = model(input_ids=input_ids, attention_mask=attn_mask).logits
        student_logprobs = F.log_softmax(student_logits[:, :-1, :].float(), dim=-1)
        with torch.no_grad():
            teacher_logits = teacher_model(input_ids=input_ids, attention_mask=pad_mask_t).logits
            teacher_logprobs = F.log_softmax(teacher_logits[:, :-1, :].float(), dim=-1)
        _divergence_type = getattr(config.training, "loss_type", "kl")
        if _divergence_type == "jsd":
            _jsd_alpha = getattr(config.training, "jsd_alpha", 0.5)
            teacher_probs = teacher_logprobs.exp()
            student_probs = student_logprobs.exp()
            M = _jsd_alpha * teacher_probs + (1.0 - _jsd_alpha) * student_probs
            log_M = M.log()
            kl_div = _jsd_alpha * (teacher_probs * (teacher_logprobs - log_M)).sum(dim=-1) \
                   + (1.0 - _jsd_alpha) * (student_probs * (student_logprobs - log_M)).sum(dim=-1)
        else:
            kl_div = (teacher_logprobs.exp() * (teacher_logprobs - student_logprobs)).sum(dim=-1)
        response_start = max(L0 - 1, 0)
        loss_mask = torch.zeros(B, _L - 1, dtype=torch.bool, device=device)
        loss_mask[:, response_start:] = True
        loss_mask &= input_ids[:, 1:].ne(pad_id)
        im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
        is_im_end = input_ids[:, 1:].eq(im_end_id) & loss_mask
        im_end_cumsum = is_im_end.cumsum(dim=1)
        im_end_shifted = F.pad(im_end_cumsum[:, :-1], (1, 0))
        loss_mask &= im_end_shifted.eq(0)
        if getattr(config.training, "exclude_im_end", False):
            loss_mask &= ~is_im_end
        kl_masked = kl_div * loss_mask
        loss_unreduced = kl_masked.sum(dim=1) / loss_mask.sum(dim=1).clamp_min(1.0)
        loss_by_masking_rate = {}
        for n, loss_i in zip(loss_mask.sum(dim=1), loss_unreduced):
            metric_name = "train/loss_causal"
            num_rate, total_loss = loss_by_masking_rate.get(metric_name, (0, 0))
            loss_by_masking_rate[metric_name] = (num_rate + 1, total_loss + loss_i.item())
        loss = loss_unreduced.mean()
        num_forward_tokens = pad_mask_t.sum().item()
        return loss, loss_by_masking_rate, num_forward_tokens

    # ── Training loop ──

    first_epoch = 0
    data_time_m = AverageMeter()
    end = time.time()

    if wandb_enabled and len(rewards) > 0:
        reward_arr = np.asarray(rewards, dtype=np.float32)
        accelerator.log({
            "train/reward_mean": float(reward_arr.mean()),
            "train/reward_std": float(reward_arr.std()),
            "train/reward_min": float(reward_arr.min()),
            "train/reward_max": float(reward_arr.max()),
            "train/current_epoch": current_epoch,
        })

    from tqdm.auto import tqdm

    stepped = False
    for epoch in range(first_epoch, num_train_epochs):
        model.train()
        progress_bar = tqdm(
            train_dataloader_lm,
            desc=f"Epoch {epoch+1}/{num_train_epochs}",
            disable=not accelerator.is_local_main_process,
            dynamic_ncols=True, leave=True,
        )

        metrics = {}
        avg_loss = AverageMeter()
        for step, batch in enumerate(progress_bar):
            data_time_m.update(time.time() - end)
            extended_input_ids_b = batch["extended_input_ids"].to(accelerator.device)
            p_mask_b = batch["p_mask"].to(accelerator.device)
            tok_idx_ext_b = batch["tok_idx_ext"].to(accelerator.device)
            labels_b = batch["labels"].to(accelerator.device)
            reward_b = batch["reward"]
            old_lp = dataset_lm.logp_old_tok[batch["ids"].cpu()].to(accelerator.device)

            _loss_type = getattr(config.training, "loss_type", "kl")
            if _loss_type in ("kl", "jsd"):
                _fwd_fn = forward_process_causal if config.training.block_size == 1 else forward_process
            else:
                raise ValueError(f"Unknown loss_type: {_loss_type}")

            loss_lm, loss_metrics, batch_loss_tokens = _fwd_fn(
                extended_input_ids=extended_input_ids_b,
                p_mask=p_mask_b,
                tok_idx_ext=tok_idx_ext_b,
                labels=labels_b,
                adv=reward_b,
                logp_old_tok=old_lp,
            )
            _batch_loss_tokens_tensor = torch.tensor(batch_loss_tokens, device=accelerator.device)
            torch.distributed.all_reduce(_batch_loss_tokens_tensor, op=torch.distributed.ReduceOp.SUM)
            cumulative_loss_tokens += _batch_loss_tokens_tensor.item()
            for loss_metric_name, (loss_metric_ct, loss_metric_value) in loss_metrics.items():
                if loss_metric_name not in metrics:
                    metrics[loss_metric_name] = AverageMeter()
                loss_metric_value = loss_metric_value / loss_metric_ct
                metrics[loss_metric_name].update(loss_metric_value, n=loss_metric_ct)
            avg_loss.update(loss_lm.item(), n=len(extended_input_ids_b))

            with accelerator.accumulate(model):
                accelerator.backward(loss_lm)
                if accelerator.sync_gradients:
                    stepped = True
                    _clip_val = config.training.max_grad_norm if config.training.max_grad_norm is not None else float("inf")
                    grad_norm_before_clip = accelerator.clip_grad_norm_(model.parameters(), _clip_val)
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad(set_to_none=True)

                    float2tensor = lambda x: torch.tensor(x, device=accelerator.device)
                    all_sums = accelerator.gather(float2tensor(avg_loss.sum))
                    all_counts = accelerator.gather(float2tensor(avg_loss.count))
                    all_avg = all_sums.sum() / all_counts.sum().clamp_min(1)

                    all_possible_keys = sorted({
                        f"train/loss_masked_frac_{round(f, 1)}"
                        for f in [i / 10 for i in range(11)]
                    })
                    local_flags = torch.tensor(
                        [1.0 if k in metrics else 0.0 for k in all_possible_keys],
                        device=accelerator.device,
                    )
                    global_flags = accelerator.gather(local_flags)
                    if global_flags.ndim > 1:
                        global_flags = global_flags.sum(dim=0)
                    all_metrics_keys = [k for k, f in zip(all_possible_keys, global_flags) if f > 0]

                    all_metrics = {}
                    for key in all_metrics_keys:
                        all_sums = accelerator.gather(float2tensor(metrics[key].sum if key in metrics else 0.0))
                        all_counts = accelerator.gather(float2tensor(metrics[key].count if key in metrics else 0))
                        if all_counts.sum() > 0:
                            all_metrics[key] = all_sums.sum() / all_counts.sum()

                    logged_metrics = {
                        "train/loss": all_avg.item(),
                        "train/lr": float(lr_scheduler.get_last_lr()[0]),
                        "train/epoch": float(epoch + 1),
                        **all_metrics,
                    }
                    logged_metrics["train/grad_norm"] = grad_norm_before_clip.item() if hasattr(grad_norm_before_clip, 'item') else float(grad_norm_before_clip)
                    logged_metrics["train/num_training_samples"] = input_ids_lm.shape[0]
                    logged_metrics["train/cumulative_loss_tokens"] = cumulative_loss_tokens
                    logged_metrics["train/student_cumulative_tflops"] = 6 * _student_total_params * cumulative_loss_tokens / 1e18
                    logged_metrics["train/teacher_cumulative_tflops"] = 2 * _teacher_total_params * cumulative_loss_tokens / 1e18
                    _rollout_token_file = os.path.join(config.experiment.project, "temp_data", "cumulative_rollout_tokens.txt")
                    _cumulative_rollout_tokens = 0
                    if os.path.exists(_rollout_token_file):
                        with open(_rollout_token_file) as _f:
                            _cumulative_rollout_tokens = int(_f.read().strip())
                    logged_metrics["train/rollout_cumulative_tokens"] = _cumulative_rollout_tokens
                    logged_metrics["train/rollout_cumulative_tflops"] = 2 * _student_total_params * _cumulative_rollout_tokens / 1e18
                    logged_metrics["train/total_cumulative_tflops"] = (
                        logged_metrics["train/student_cumulative_tflops"]
                        + logged_metrics["train/teacher_cumulative_tflops"]
                    )
                    _gpu_hours_file = os.path.join(config.experiment.project, "temp_data", "cumulative_gpu_hours.txt")
                    if os.path.exists(_gpu_hours_file):
                        with open(_gpu_hours_file) as _ghf:
                            logged_metrics["train/cumulative_gpu_hours"] = float(_ghf.read().strip())
                    else:
                        logged_metrics["train/cumulative_gpu_hours"] = 0.0
                    logged_metrics["train/current_epoch"] = current_epoch
                    # if accelerator.is_main_process:
                    #     loss_val = logged_metrics.get("train/loss", 0)
                    #     grad_val = logged_metrics.get("train/grad_norm", 0)
                    #     lr_val = logged_metrics.get("train/lr", 0)
                    #     gpu_hrs = logged_metrics.get("train/cumulative_gpu_hours", 0)
                    #     print(f"Step {step+1} | loss={loss_val:.4f} grad={grad_val:.2f} lr={lr_val:.1e} gpu_hrs={gpu_hrs:.1f}")
                    if wandb_enabled:
                        accelerator.log(logged_metrics)
                        metrics = {}
                    avg_loss.reset()
                    torch.cuda.empty_cache()

            end = time.time()

    if not stepped:
        logger.warning("Training ended with no optimizer step taken.")

    # Persist cumulative loss tokens
    token_count_file = os.path.join(config.experiment.project, "temp_data", "cumulative_loss_tokens.txt")
    os.makedirs(os.path.dirname(token_count_file), exist_ok=True)
    with open(token_count_file, "w") as f:
        f.write(str(cumulative_loss_tokens))
    state["cumulative_loss_tokens"] = cumulative_loss_tokens

    accelerator.wait_for_everyone()

    # JetEngine reloads from ckpt/<optimized_name> every step (rl.py:reload_jetengine_weights),
    # so weights must be re-written every step or rollouts use stale weights. Training state
    # (DeepSpeed optimizer) is heavy → only persist it on save_every cadence.
    save_full_state = (config.experiment.current_epoch % config.experiment.save_every == 0)
    save_checkpoint(
        model, tokenizer, config, accelerator, config.model.optimized_name,
        save_training_state_flag=save_full_state,
        lr_scheduler=lr_scheduler if save_full_state else None,
    )
    if save_full_state:
        save_checkpoint(model, tokenizer, config, accelerator, f"epoch-{config.experiment.current_epoch}", save_training_state_flag=False)


if __name__ == "__main__":
    main()
