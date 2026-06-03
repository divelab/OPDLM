from collections import deque
import torch
from torch.nn import functional as F
import numpy as np

from jetengine_ext.config import Config
from jetengine_ext.engine.sequence import Sequence, SequenceStatus, RunType
from jetengine_ext.engine.block_manager import BlockManager
from jetengine_ext.layers.sampler import sample_with_temperature_topk_topp, top_k_logits, top_p_logits


def _compute_probs(logits, temperature=1.0, top_k=0, top_p=1.0):
    """Pure PyTorch replacement for FlashInfer LogitsPipe sampling pipeline.

    Returns ``(probs_sampling, probs_raw)``:

    * ``probs_sampling`` — distribution to draw the next token from
      (post temperature/top-k/top-p; one-hot when temperature==0 or top_k==1).
    * ``probs_raw`` — softmax of the unfiltered logits, used as the
      *confidence* signal for ``low_confidence_dynamic`` thresholding.

    The split exists because greedy decoding (temp=0 or top_k=1) collapses
    ``probs_sampling`` to one-hot, making the dynamic-threshold check vacuous
    (every position has confidence=1.0 and the whole block unmasks in one
    step). Mirrors Fast-dLLM v2's ``sample_with_top_p`` argmax branch
    (modeling.py:834), which returns the raw softmax alongside the argmax.
    """
    # Guard: upstream bf16 attention overflow can produce inf/nan in logits,
    # which makes softmax return nan and crashes torch.multinomial. Replace
    # non-finite entries with a large-negative so softmax sees 0 probability there.
    if not torch.isfinite(logits).all():
        num_bad = (~torch.isfinite(logits)).sum().item()
        import logging
        logging.getLogger(__name__).warning(
            f"[_compute_probs] non-finite logits: {num_bad} entries; clamping to finite"
        )
        logits = torch.nan_to_num(logits, nan=-1e4, posinf=1e4, neginf=-1e4)
    probs_raw = F.softmax(logits, dim=-1)
    sampling_logits = logits
    if temperature == 0.0:
        # Argmax: set top-1 logit to a large value, rest to -inf
        sampling_logits = torch.where(
            sampling_logits >= sampling_logits.max(dim=-1, keepdim=True).values,
            sampling_logits, torch.full_like(sampling_logits, float('-inf')),
        )
    elif temperature != 1.0:
        sampling_logits = sampling_logits / temperature
    if top_k > 0:
        sampling_logits = top_k_logits(sampling_logits, top_k)
    if top_p < 1.0:
        sampling_logits = top_p_logits(sampling_logits, top_p)
    return F.softmax(sampling_logits, dim=-1), probs_raw


class Scheduler:

    def __init__(self, config: Config):
        self.max_num_seqs = config.max_num_seqs
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.eos = config.eos
        self.mask_token_id = config.mask_token_id
        self.block_manager = BlockManager(config.num_kvcache_blocks, config.kvcache_block_size)
        self.running: list[Sequence] = []

    def add(self, seq: Sequence):
        self.running.append(seq)

    def is_finished(self):
        return not self.running

    def schedule(self) -> tuple[list[Sequence], RunType] | tuple[None, None]:
        # 1. Schedule new sequences for prefill
        prefill_candidates = [s for s in self.running if s.status == SequenceStatus.WAITING]
        if prefill_candidates:
            prefill_batch = []
            # Simple batching: take as many as fit
            for seq in prefill_candidates:
                # num_tokens for a waiting seq is its prefill length
                if len(prefill_batch) < self.max_num_seqs and self.block_manager.can_allocate(seq):
                    self.block_manager.allocate(seq)
                    seq.status = SequenceStatus.PREFILLING
                    prefill_batch.append(seq)
            if prefill_batch:
                return prefill_batch, RunType.PREFILL   
        # 2. If no prefilling, create a DENOISE batch.
        denoise_candidates = [s for s in self.running if s.status == SequenceStatus.DENOISING or s.status == SequenceStatus.SAVING]
        if denoise_candidates:
            denoise_batch = []
            for seq in denoise_candidates:
                num_new_blocks = seq.num_new_blocks_needed(self.block_manager.block_size)
                if len(denoise_batch) < self.max_num_seqs and self.block_manager.can_append_blocks(num_new_blocks):
                    self.block_manager.append_blocks(seq, num_new_blocks)
                    denoise_batch.append(seq)
            if denoise_batch:
                return denoise_batch, RunType.DENOISE

        return None, None     

    def postprocess(self, seqs: list[Sequence], logits: torch.Tensor, run_type: RunType):
        if run_type == RunType.PREFILL:
            for seq in seqs:
                seq.num_cached_tokens = seq.num_prefill_tokens
                seq.status = SequenceStatus.DENOISING
        
        elif run_type == RunType.DENOISE:
            start_idx = 0
            if self.consistent_sampling_params:
                probs, probs_raw = _compute_probs(logits, temperature=seqs[0].temperature, top_k=seqs[0].top_k, top_p=seqs[0].top_p)
            for seq in seqs:
                # Extract the part of the tensors relevant to this sequence
                if seq.status == SequenceStatus.DENOISING:
                    block_len = seq.block_length
                    if not self.consistent_sampling_params:
                        probs, probs_raw = _compute_probs(logits[start_idx : start_idx + block_len], temperature=seq.temperature, top_k=seq.top_k, top_p=seq.top_p)
                        seq_x0 = torch.multinomial(probs, num_samples=1).squeeze(-1)
                        seq_x0_p = torch.gather(probs_raw, -1, seq_x0.unsqueeze(-1)).squeeze(-1)
                    else:
                        seq_x0 = torch.multinomial(probs[start_idx : start_idx + block_len], num_samples=1).squeeze(-1)
                        seq_x0_p = torch.gather(probs_raw[start_idx : start_idx + block_len], -1, seq_x0.unsqueeze(-1)).squeeze(-1)
                    
                    current_block_tensor = torch.tensor(seq.intermediate_block_tokens, device=logits.device)
                    mask_index = (current_block_tensor == self.mask_token_id)
                    num_to_transfer = seq.num_transfer_tokens_per_step[seq.current_denoising_step]
                    
                    transfer_index = torch.zeros_like(seq_x0, dtype=torch.bool)
                    
                    if seq.remasking_strategy == 'sequential':
                        if mask_index.any():
                            first_mask_pos = mask_index.nonzero(as_tuple=True)[0].min().item()
                            end_pos = min(first_mask_pos + num_to_transfer, block_len)
                            transfer_index[first_mask_pos:end_pos] = True
                    
                    elif 'low_confidence_static' in seq.remasking_strategy:
                        confidence = torch.where(mask_index, seq_x0_p, -np.inf)
                        # For dynamic, add threshold logic here if desired
                        _, top_indices = torch.topk(confidence, num_to_transfer)
                        transfer_index[top_indices] = True
                    
                    elif 'low_confidence_dynamic' in seq.remasking_strategy:
                        confidence = torch.where(mask_index, seq_x0_p, -np.inf)
                        transfer_index = torch.where(confidence > seq.dynamic_threshold, True, False)
                        if sum(transfer_index) < num_to_transfer:
                            _, top_indices = torch.topk(confidence, num_to_transfer)
                            transfer_index[top_indices] = True
                        num_to_transfer = transfer_index.sum().item() if transfer_index.sum().item() > 0 else num_to_transfer
                    elif 'entropy_bounded' in seq.remasking_strategy:
                        block_probs = probs[start_idx : start_idx + block_len]
                        P = block_probs[mask_index]
                        eps = 1e-12
                        entropies = -(P.clamp_min(eps) * (P.clamp_min(eps)).log()).sum(dim=-1)
                        ent_sorted, order = torch.sort(entropies, dim=0, descending=False)
                        cumsum = torch.cumsum(ent_sorted, dim=0)
                        k = torch.searchsorted(cumsum, torch.tensor(seq.eb_threshold, device=P.device), right=False).item()
                        if k == 0:
                            k = 1
                        # print(k)
                        selected_token_indices = mask_index.nonzero(as_tuple=True)[0][order[:k]]
                        # print(selected_token_indices)
                        transfer_index[selected_token_indices] = True
                        num_to_transfer = k

                    # update
                    new_block_list = current_block_tensor.tolist()
                    accepted_tokens = seq_x0[transfer_index].tolist()
                    original_indices = transfer_index.nonzero(as_tuple=True)[0].tolist()





                    # newly added
                    if seq.block_first_unmask_steps is None or len(seq.block_first_unmask_steps) != block_len:
                        seq.block_first_unmask_steps = [0] * block_len
                    first_time_global = seq.global_denoising_step + 1
                    for idx in original_indices:
                        if seq.block_first_unmask_steps[idx] == 0:
                            seq.block_first_unmask_steps[idx] = first_time_global



                    

                    for idx, token in zip(original_indices, accepted_tokens):
                        new_block_list[idx] = token
                    seq.intermediate_block_tokens = new_block_list
                    
                    seq.current_denoising_step += 1
                    seq.global_denoising_step += 1
                    
                    # Check if block is fully denoised
                    is_fully_denoised = (self.mask_token_id not in seq.intermediate_block_tokens) or \
                                        (seq.current_denoising_step >= seq.denoising_steps)

                    if is_fully_denoised:
                        # Block is done, commit it and check if generation is finished
                        seq.status = SequenceStatus.FINISHED if seq.is_finished else SequenceStatus.SAVING
                    seq.num_to_transfer = num_to_transfer
                    
                elif seq.status == SequenceStatus.SAVING:
                    # If saving, commit the block and start a new one
                    seq.commit_block(seq.intermediate_block_tokens)
                    seq.num_to_transfer = 0
                    if not seq.is_finished:
                        seq.start_new_block()

                start_idx += seq.block_length
                
        # Filter out finished sequences from the running list
        finished_seqs = [seq for seq in self.running if seq.is_finished]
        self.running = [seq for seq in self.running if not seq.is_finished]
        for seq in finished_seqs:
            self.block_manager.deallocate(seq)