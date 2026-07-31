"""Batched self-speculative decoding for OPDLM using one HF model."""
from dataclasses import asdict, dataclass
from time import perf_counter
import torch

@dataclass
class SelfSpeculativeStats:
    accepted_draft_tokens: int = 0
    proposed_draft_tokens: int = 0
    sequence_iterations: int = 0
    blockwise_drafting_forwards: int = 0
    blockwise_prefill_forwards: int = 0
    blockwise_cache_update_forwards: int = 0
    causal_verification_forwards: int = 0
    causal_sequential_fallback_forwards: int = 0
    guarded_fallback_sequence_iterations: int = 0
    causal_prefill_forwards: int = 0
    causal_correction_forwards: int = 0
    generated_tokens: int = 0
    wall_clock_generation_seconds: float = 0.0
    def report(self):
        forwards = (self.blockwise_drafting_forwards + self.blockwise_prefill_forwards +
                    self.blockwise_cache_update_forwards + self.causal_verification_forwards +
                    self.causal_sequential_fallback_forwards +
                    self.causal_prefill_forwards + self.causal_correction_forwards)
        result = asdict(self)
        result.update(
            average_accepted_draft_tokens_per_iteration=self.accepted_draft_tokens / self.sequence_iterations if self.sequence_iterations else 0.0,
            draft_token_acceptance_rate=self.accepted_draft_tokens / self.proposed_draft_tokens if self.proposed_draft_tokens else 0.0,
            guarded_fallback_rate=self.guarded_fallback_sequence_iterations / self.sequence_iterations if self.sequence_iterations else 0.0,
            generated_tokens_per_total_forward_pass=self.generated_tokens / forwards if forwards else 0.0)
        return result

def _left_padded(sequences, pad_id, multiple):
    width = ((max(x.numel() for x in sequences) + multiple - 1) // multiple) * multiple
    batch = torch.full((len(sequences), width), pad_id, dtype=torch.long, device=sequences[0].device)
    valid = torch.zeros_like(batch, dtype=torch.bool)
    for row, seq in enumerate(sequences):
        batch[row, width-seq.numel():], valid[row, width-seq.numel():] = seq, True
    return batch, valid

def _position_ids(valid):
    positions = valid.long().cumsum(-1) - 1
    return torch.where(valid, positions, torch.zeros_like(positions))

def _blockwise_mask(valid, block_size):
    batch, length = valid.shape
    blocks = (torch.arange(length, device=valid.device) // block_size).view(1, length).expand(batch, -1)
    blocks = torch.where(valid, blocks, -1)
    query, key = blocks[:, None, :, None], blocks[:, None, None, :]
    return (query >= 0) & (key >= 0) & (key <= query)

def _causal_mask(valid):
    length = valid.shape[1]
    pos = torch.arange(length, device=valid.device)
    causal = pos.view(1, 1, 1, length) <= pos.view(1, 1, length, 1)
    return causal & valid[:, None, :, None] & valid[:, None, None, :]

@torch.no_grad()
def causal_greedy(model, sequences, max_new_tokens, eos_id, pad_id):
    """Reference greedy causal decoding used only for debug comparison."""
    outputs, prompt_lengths = [x.clone() for x in sequences], [x.numel() for x in sequences]
    done = [False] * len(outputs)
    while not all(done):
        active = [i for i, flag in enumerate(done) if not flag]
        batch, valid = _left_padded([outputs[i] for i in active], pad_id, 1)
        logits = model(batch, attention_mask=_causal_mask(valid), position_ids=_position_ids(valid), use_cache=False).logits
        tokens = logits[:, -1].argmax(-1)
        for row, original in enumerate(active):
            token = tokens[row:row+1]
            outputs[original] = torch.cat((outputs[original], token))
            generated = outputs[original].numel() - prompt_lengths[original]
            done[original] = (eos_id is not None and token.item() == eos_id) or generated >= max_new_tokens
    return outputs

@torch.no_grad()
def self_speculative_generate(model, sequences, max_new_tokens, draft_block_size, mask_id, eos_id, pad_id):
    """Generate a ragged batch with one drafting and one verification forward per iteration."""
    if draft_block_size <= 0:
        raise ValueError("draft_block_size must be positive")
    outputs, prompt_lengths = [x.clone() for x in sequences], [x.numel() for x in sequences]
    done, stats, started = [False] * len(outputs), SelfSpeculativeStats(), perf_counter()
    if outputs and outputs[0].is_cuda:
        torch.cuda.synchronize(outputs[0].device); started = perf_counter()
    while not all(done):
        active = [i for i, flag in enumerate(done) if not flag]
        prefixes, prefix_valid = _left_padded([outputs[i] for i in active], pad_id, draft_block_size)
        remaining = torch.tensor([max_new_tokens-(outputs[i].numel()-prompt_lengths[i]) for i in active], device=prefixes.device)
        proposed_lengths = remaining.clamp(max=draft_block_size)
        draft = torch.full((len(active), draft_block_size), mask_id, dtype=torch.long, device=prefixes.device)
        draft_valid = torch.arange(draft_block_size, device=prefixes.device)[None] < proposed_lengths[:, None]
        full_valid = torch.cat((prefix_valid, draft_valid), 1)
        position_ids, block_mask = _position_ids(full_valid), _blockwise_mask(full_valid, draft_block_size)
        draft_logits = model(
            torch.cat((prefixes, draft), 1),
            attention_mask=block_mask,
            position_ids=position_ids,
            use_cache=False,
        ).logits[:, -draft_block_size:]
        stats.blockwise_drafting_forwards += 1
        draft = torch.where(draft_valid, draft_logits.argmax(-1), draft)
        verify_logits = model(torch.cat((prefixes, draft), 1), attention_mask=_causal_mask(full_valid), position_ids=position_ids, use_cache=False).logits
        stats.causal_verification_forwards += 1
        prefix_width = prefixes.shape[1]
        # One-token shift: prediction for draft[i] is at prefix_width - 1 + i.
        causal_tokens = verify_logits[:, prefix_width-1:prefix_width-1+draft_block_size].argmax(-1)
        for row, original in enumerate(active):
            length = int(proposed_lengths[row]); candidate, verified = draft[row, :length], causal_tokens[row, :length]
            mismatches = (candidate != verified).nonzero(as_tuple=False)
            if mismatches.numel():
                first = int(mismatches[0, 0]); committed = torch.cat((candidate[:first], verified[first:first+1])); accepted = first
            else: committed, accepted = candidate, length
            if eos_id is not None:
                eos = (committed == eos_id).nonzero(as_tuple=False)
                if eos.numel(): committed, done[original] = committed[:int(eos[0, 0])+1], True
            outputs[original] = torch.cat((outputs[original], committed))
            stats.accepted_draft_tokens += accepted; stats.proposed_draft_tokens += length
            stats.sequence_iterations += 1; stats.generated_tokens += committed.numel()
            done[original] = done[original] or outputs[original].numel()-prompt_lengths[original] >= max_new_tokens
    if outputs and outputs[0].is_cuda: torch.cuda.synchronize(outputs[0].device)
    stats.wall_clock_generation_seconds = perf_counter() - started
    return outputs, stats

def _cached_query_mask(key_valid, query_valid, causal):
    """Build a 4D bool mask for queries appended to a padded DynamicCache."""
    batch, query_length = query_valid.shape
    prefix_length = key_valid.shape[1]
    prefix = key_valid[:, None, None, :].expand(batch, 1, query_length, prefix_length)
    prefix = prefix & query_valid[:, None, :, None]
    if causal:
        structure = torch.ones(query_length, query_length, dtype=torch.bool, device=query_valid.device).tril()
    else:
        structure = torch.ones(query_length, query_length, dtype=torch.bool, device=query_valid.device)
    current = structure[None, None] & query_valid[:, None, None, :] & query_valid[:, None, :, None]
    # Invalid padding queries still need one finite attention target.
    current = current | (
        torch.eye(query_length, dtype=torch.bool, device=query_valid.device)[None, None]
        & (~query_valid)[:, None, :, None]
    )
    return torch.cat((prefix, current), dim=-1)


def _truncate_dynamic_cache(cache, target_length):
    for layer in range(len(cache.key_cache)):
        cache.key_cache[layer] = cache.key_cache[layer][:, :, :target_length, :]
        cache.value_cache[layer] = cache.value_cache[layer][:, :, :target_length, :]


def _slice_dynamic_cache(cache, rows):
    for layer in range(len(cache.key_cache)):
        cache.key_cache[layer] = cache.key_cache[layer].index_select(0, rows)
        cache.value_cache[layer] = cache.value_cache[layer].index_select(0, rows)


def _copy_dynamic_cache_row(cache, row, target_length):
    """Copy one cache row through target_length for isolated q=1 verification."""
    from transformers.cache_utils import DynamicCache

    result = DynamicCache()
    result.key_cache = [
        keys[row:row + 1, :, :target_length, :].clone()
        for keys in cache.key_cache
    ]
    result.value_cache = [
        values[row:row + 1, :, :target_length, :].clone()
        for values in cache.value_cache
    ]
    return result


def _replace_dynamic_cache_row(cache, source, row, start, length):
    """Replace a candidate region with states produced by isolated q=1 calls."""
    for layer in range(len(cache.key_cache)):
        cache.key_cache[layer][row:row + 1, :, start:start + length, :] = (
            source.key_cache[layer][:, :, start:start + length, :]
        )
        cache.value_cache[layer][row:row + 1, :, start:start + length, :] = (
            source.value_cache[layer][:, :, start:start + length, :]
        )


def _top_two_margin(logits):
    top_two = logits.float().topk(2, dim=-1).values
    return top_two[..., 0] - top_two[..., 1]


@torch.no_grad()
def cached_self_speculative_generate(
    model, sequences, max_new_tokens, draft_block_size, mask_id, eos_id, pad_id,
    margin_threshold=0.05,
):
    """KV-cached self-speculation with guarded q=1 verification fallback."""
    from transformers.cache_utils import DynamicCache

    if draft_block_size <= 0:
        raise ValueError("draft_block_size must be positive")
    if margin_threshold < 0:
        raise ValueError("margin_threshold must be nonnegative")
    if not sequences:
        return [], SelfSpeculativeStats()

    device = sequences[0].device
    outputs = [sequence.clone() for sequence in sequences]
    prompt_lengths = [sequence.numel() for sequence in sequences]
    active_ids = list(range(len(outputs)))
    logical_lengths = torch.tensor(prompt_lengths, dtype=torch.long, device=device)
    prefix_ids, prefix_valid = _left_padded(sequences, pad_id, draft_block_size)
    prefix_positions = _position_ids(prefix_valid)
    physical_positions = torch.arange(prefix_ids.shape[1], device=device)
    block_cache = DynamicCache()
    causal_cache = DynamicCache()
    stats = SelfSpeculativeStats()

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = perf_counter()

    # Drafting keeps OPDLM's normal K-block prefix representation.
    model.model(
        input_ids=prefix_ids,
        attention_mask=_blockwise_mask(prefix_valid, draft_block_size),
        position_ids=prefix_positions,
        past_key_values=block_cache,
        use_cache=True,
        cache_position=physical_positions,
    )
    stats.blockwise_prefill_forwards += 1
    block_key_valid = prefix_valid

    # Verification has an independent causal representation of the same prefix.
    causal_prefix = model.model(
        input_ids=prefix_ids,
        attention_mask=_cached_query_mask(
            torch.zeros((len(sequences), 0), dtype=torch.bool, device=device),
            prefix_valid,
            causal=True,
        ),
        position_ids=prefix_positions,
        past_key_values=causal_cache,
        use_cache=True,
        cache_position=physical_positions,
    )
    next_logits = model.lm_head(causal_prefix.last_hidden_state[:, -1])
    next_tokens = next_logits.argmax(-1)
    stats.causal_prefill_forwards += 1
    causal_key_valid = prefix_valid.clone()

    while active_ids:
        batch = len(active_ids)
        block_offsets = torch.arange(draft_block_size, device=device)
        remaining = torch.tensor(
            [max_new_tokens - (outputs[index].numel() - prompt_lengths[index]) for index in active_ids],
            dtype=torch.long,
            device=device,
        )
        proposed_lengths = remaining.clamp(max=draft_block_size)
        query_valid = block_offsets[None] < proposed_lengths[:, None]
        position_ids = logical_lengths[:, None] + block_offsets[None]

        # One noncausal forward proposes all K tokens. Its mask-token cache
        # entries are temporary and removed immediately afterward.
        block_physical_length = block_key_valid.shape[1]
        block_cache_positions = torch.arange(
            block_physical_length,
            block_physical_length + draft_block_size,
            device=device,
        )
        draft_input = torch.full(
            (batch, draft_block_size), mask_id, dtype=torch.long, device=device,
        )
        draft_logits = model(
            input_ids=draft_input,
            attention_mask=_cached_query_mask(block_key_valid, query_valid, causal=False),
            position_ids=position_ids,
            past_key_values=block_cache,
            use_cache=True,
            cache_position=block_cache_positions,
        ).logits
        stats.blockwise_drafting_forwards += 1
        draft = draft_logits.argmax(-1)
        _truncate_dynamic_cache(block_cache, block_physical_length)

        # The causal cache receives the candidate block. Logits are shifted by
        # one position: y[0] comes from the cached prefix's final token.
        causal_physical_length = causal_key_valid.shape[1]
        causal_cache_positions = torch.arange(
            causal_physical_length,
            causal_physical_length + draft_block_size,
            device=device,
        )
        verify_logits = model(
            input_ids=draft,
            attention_mask=_cached_query_mask(causal_key_valid, query_valid, causal=True),
            position_ids=position_ids,
            past_key_values=causal_cache,
            use_cache=True,
            cache_position=causal_cache_positions,
        ).logits
        stats.causal_verification_forwards += 1
        verify_predictions = verify_logits.argmax(-1)
        causal_tokens = torch.empty_like(draft)
        causal_tokens[:, 0] = next_tokens
        causal_tokens[:, 1:] = verify_predictions[:, :-1]

        # q=K and q=1 attention kernels can disagree around nearly tied logits.
        # Reverify only ambiguous rows with the canonical token-by-token cached
        # path, then replace their q=K cache entries with the q=1 entries.
        if margin_threshold > 0:
            shifted_margins = torch.empty(
                (batch, draft_block_size), dtype=torch.float32, device=device,
            )
            shifted_margins[:, 0] = _top_two_margin(next_logits)
            shifted_margins[:, 1:] = _top_two_margin(verify_logits[:, :-1])
            relevant = torch.zeros_like(query_valid)
            for row in range(batch):
                length = int(proposed_lengths[row])
                mismatches = (
                    draft[row, :length] != causal_tokens[row, :length]
                ).nonzero(as_tuple=False)
                stop = int(mismatches[0, 0]) + 1 if mismatches.numel() else length
                relevant[row, :stop] = True
            ambiguous = (
                (shifted_margins <= margin_threshold) & relevant
            ).any(dim=1)
            for row in ambiguous.nonzero(as_tuple=False).flatten().tolist():
                length = int(proposed_lengths[row])
                sequential_cache = _copy_dynamic_cache_row(
                    causal_cache, row, causal_physical_length,
                )
                sequential_key_valid = causal_key_valid[row:row + 1].clone()
                sequential_logits = []
                for offset in range(length):
                    token_logits = model(
                        input_ids=draft[row:row + 1, offset:offset + 1],
                        attention_mask=_cached_query_mask(
                            sequential_key_valid,
                            torch.ones((1, 1), dtype=torch.bool, device=device),
                            causal=True,
                        ),
                        position_ids=position_ids[row:row + 1, offset:offset + 1],
                        past_key_values=sequential_cache,
                        use_cache=True,
                        cache_position=torch.tensor(
                            [causal_physical_length + offset], device=device,
                        ),
                    ).logits[:, 0]
                    sequential_logits.append(token_logits)
                    sequential_key_valid = torch.cat(
                        (
                            sequential_key_valid,
                            torch.ones((1, 1), dtype=torch.bool, device=device),
                        ),
                        dim=1,
                    )
                row_logits = torch.cat(sequential_logits, dim=0)
                verify_logits[row, :length] = row_logits
                verify_predictions[row, :length] = row_logits.argmax(-1)
                causal_tokens[row, 1:length] = verify_predictions[row, :length - 1]
                _replace_dynamic_cache_row(
                    causal_cache, sequential_cache, row,
                    causal_physical_length, length,
                )
                stats.causal_sequential_fallback_forwards += length
                stats.guarded_fallback_sequence_iterations += 1

        candidate_keep = torch.zeros(batch, dtype=torch.long, device=device)
        correction_valid = torch.zeros(batch, dtype=torch.bool, device=device)
        correction_tokens = torch.full((batch,), pad_id, dtype=torch.long, device=device)
        committed_by_row = []
        finished = []

        for row, original in enumerate(active_ids):
            length = int(proposed_lengths[row])
            candidate = draft[row, :length]
            verified = causal_tokens[row, :length]
            mismatches = (candidate != verified).nonzero(as_tuple=False)
            if mismatches.numel():
                first = int(mismatches[0, 0])
                committed = torch.cat((candidate[:first], verified[first:first + 1]))
                kept_drafts = first
                used_correction = True
            else:
                committed = candidate
                kept_drafts = length
                used_correction = False

            if eos_id is not None:
                eos = (committed == eos_id).nonzero(as_tuple=False)
                if eos.numel():
                    stop = int(eos[0, 0]) + 1
                    committed = committed[:stop]
                    if stop <= kept_drafts:
                        kept_drafts = stop
                        used_correction = False

            generated_before = outputs[original].numel() - prompt_lengths[original]
            committed = committed[:max_new_tokens - generated_before]
            kept_drafts = min(kept_drafts, committed.numel())
            used_correction = used_correction and committed.numel() > kept_drafts

            candidate_keep[row] = kept_drafts
            if used_correction:
                correction_valid[row] = True
                correction_tokens[row] = committed[-1]
            committed_by_row.append(committed)
            is_eos = bool(
                eos_id is not None and committed.numel() and committed[-1].item() == eos_id
            )
            finished.append(
                is_eos or generated_before + committed.numel() >= max_new_tokens
            )
            stats.accepted_draft_tokens += kept_drafts
            stats.proposed_draft_tokens += length
            stats.sequence_iterations += 1
            stats.generated_tokens += committed.numel()

        # Keep only the accepted candidate prefix in the causal cache.
        candidate_valid = block_offsets[None] < candidate_keep[:, None]
        causal_key_valid = torch.cat((causal_key_valid, candidate_valid), dim=1)

        # A mismatch correction replaces the rejected candidate at its logical
        # position and supplies the next-token logit for the following iteration.
        if correction_valid.any():
            correction_positions = torch.where(
                correction_valid,
                logical_lengths + candidate_keep,
                torch.zeros_like(logical_lengths),
            ).unsqueeze(1)
            correction_logits = model(
                input_ids=correction_tokens.unsqueeze(1),
                attention_mask=_cached_query_mask(
                    causal_key_valid, correction_valid.unsqueeze(1), causal=True,
                ),
                position_ids=correction_positions,
                past_key_values=causal_cache,
                use_cache=True,
                cache_position=torch.tensor([causal_key_valid.shape[1]], device=device),
            ).logits[:, 0]
            stats.causal_correction_forwards += 1
            next_logits = torch.where(
                correction_valid.unsqueeze(1), correction_logits, next_logits,
            )
            next_tokens = torch.where(
                correction_valid, correction_logits.argmax(-1), next_tokens,
            )
            causal_key_valid = torch.cat(
                (causal_key_valid, correction_valid.unsqueeze(1)), dim=1,
            )

        # Re-encode the committed chunk noncausally into the independent draft
        # cache. Padding slots let every batch row retain its own commit length.
        committed_input = torch.full(
            (batch, draft_block_size), pad_id, dtype=torch.long, device=device,
        )
        committed_valid = torch.zeros(
            (batch, draft_block_size), dtype=torch.bool, device=device,
        )
        for row, committed in enumerate(committed_by_row):
            committed_input[row, :committed.numel()] = committed
            committed_valid[row, :committed.numel()] = True
        block_update_positions = logical_lengths[:, None] + block_offsets[None]
        block_update_cache_positions = torch.arange(
            block_key_valid.shape[1],
            block_key_valid.shape[1] + draft_block_size,
            device=device,
        )
        model.model(
            input_ids=committed_input,
            attention_mask=_cached_query_mask(
                block_key_valid, committed_valid, causal=False,
            ),
            position_ids=block_update_positions,
            past_key_values=block_cache,
            use_cache=True,
            cache_position=block_update_cache_positions,
        )
        stats.blockwise_cache_update_forwards += 1
        block_key_valid = torch.cat((block_key_valid, committed_valid), dim=1)

        for row, original in enumerate(active_ids):
            committed = committed_by_row[row]
            outputs[original] = torch.cat((outputs[original], committed))
            if not correction_valid[row] and not finished[row]:
                next_logits[row] = verify_logits[row, int(candidate_keep[row]) - 1]
                next_tokens[row] = next_logits[row].argmax(-1)
        logical_lengths = logical_lengths + torch.tensor(
            [tokens.numel() for tokens in committed_by_row],
            dtype=torch.long,
            device=device,
        )

        keep = torch.tensor(
            [row for row, is_finished in enumerate(finished) if not is_finished],
            dtype=torch.long,
            device=device,
        )
        if keep.numel() == 0:
            break
        _slice_dynamic_cache(block_cache, keep)
        _slice_dynamic_cache(causal_cache, keep)
        block_key_valid = block_key_valid.index_select(0, keep)
        causal_key_valid = causal_key_valid.index_select(0, keep)
        logical_lengths = logical_lengths.index_select(0, keep)
        next_logits = next_logits.index_select(0, keep)
        next_tokens = next_tokens.index_select(0, keep)
        active_ids = [active_ids[row] for row in keep.tolist()]

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    stats.wall_clock_generation_seconds = perf_counter() - started
    return outputs, stats


def _sdar_verifier_mask(key_valid, valid):
    """Position-aligned SDAR verifier over [draft tokens | mask tokens]."""
    batch, block = valid.shape
    device = valid.device
    both = torch.cat((valid, valid), dim=1)
    prefix = key_valid[:, None, None, :].expand(batch, 1, 2 * block, -1)
    prefix = prefix & both[:, None, :, None]
    current = torch.zeros(
        (batch, 1, 2 * block, 2 * block), dtype=torch.bool, device=device,
    )
    pair_valid = valid[:, :, None] & valid[:, None, :]
    lower = torch.ones((block, block), dtype=torch.bool, device=device).tril()
    strict = torch.ones((block, block), dtype=torch.bool, device=device).tril(-1)
    eye = torch.eye(block, dtype=torch.bool, device=device)
    current[:, 0, :block, :block] = lower[None] & pair_valid
    current[:, 0, block:, :block] = strict[None] & pair_valid
    current[:, 0, block:, block:] = eye[None] & pair_valid
    current |= (
        torch.eye(2 * block, dtype=torch.bool, device=device)[None, None]
        & (~both)[:, None, :, None]
    )
    return torch.cat((prefix, current), dim=-1)


@torch.no_grad()
def cached_self_speculative_generate_sdar(
    model, sequences, max_new_tokens, draft_block_size, mask_id, eos_id, pad_id,
):
    """KV-cached greedy self-speculation for SDAR's position-aligned logits."""
    from transformers.cache_utils import DynamicCache

    if draft_block_size <= 0:
        raise ValueError("draft_block_size must be positive")
    if not sequences:
        return [], SelfSpeculativeStats()

    device = sequences[0].device
    outputs = [x.clone() for x in sequences]
    prompt_lengths = [x.numel() for x in sequences]
    active_ids = list(range(len(outputs)))
    logical_lengths = torch.tensor(prompt_lengths, dtype=torch.long, device=device)
    prefix_ids, prefix_valid = _left_padded(sequences, pad_id, draft_block_size)
    positions = _position_ids(prefix_valid)
    cache_positions = torch.arange(prefix_ids.shape[1], device=device)
    block_cache, causal_cache = DynamicCache(), DynamicCache()
    stats = SelfSpeculativeStats()

    torch.cuda.synchronize(device)
    started = perf_counter()
    model(
        input_ids=prefix_ids,
        attention_mask=_blockwise_mask(prefix_valid, draft_block_size),
        position_ids=positions,
        past_key_values=block_cache,
        use_cache=True,
        store_kv=True,
        cache_position=cache_positions,
    )
    stats.blockwise_prefill_forwards += 1
    model(
        input_ids=prefix_ids,
        attention_mask=_causal_mask(prefix_valid),
        position_ids=positions,
        past_key_values=causal_cache,
        use_cache=True,
        store_kv=True,
        cache_position=cache_positions,
    )
    stats.causal_prefill_forwards += 1
    block_valid, causal_valid = prefix_valid, prefix_valid.clone()
    offsets = torch.arange(draft_block_size, device=device)

    while active_ids:
        batch = len(active_ids)
        remaining = torch.tensor(
            [max_new_tokens - (outputs[i].numel() - prompt_lengths[i]) for i in active_ids],
            dtype=torch.long, device=device,
        )
        lengths = remaining.clamp(max=draft_block_size)
        valid = offsets[None] < lengths[:, None]
        positions = logical_lengths[:, None] + offsets[None]
        masks = torch.full(
            (batch, draft_block_size), mask_id, dtype=torch.long, device=device,
        )
        draft_logits = model(
            input_ids=masks,
            attention_mask=_cached_query_mask(block_valid, valid, causal=False),
            position_ids=positions,
            past_key_values=block_cache,
            use_cache=True,
            store_kv=False,
        ).logits
        stats.blockwise_drafting_forwards += 1
        draft = draft_logits.argmax(-1)

        verify_logits = model(
            input_ids=torch.cat((draft, masks), dim=1),
            attention_mask=_sdar_verifier_mask(causal_valid, valid),
            position_ids=torch.cat((positions, positions), dim=1),
            past_key_values=causal_cache,
            use_cache=True,
            store_kv=False,
        ).logits[:, draft_block_size:]
        stats.causal_verification_forwards += 1
        verified = verify_logits.argmax(-1)

        committed_rows, finished = [], []
        for row, original in enumerate(active_ids):
            length = int(lengths[row])
            mismatch = (
                draft[row, :length] != verified[row, :length]
            ).nonzero(as_tuple=False)
            if mismatch.numel():
                first = int(mismatch[0, 0])
                committed = torch.cat(
                    (draft[row, :first], verified[row, first:first + 1])
                )
                accepted = first
            else:
                committed, accepted = draft[row, :length], length
            if eos_id is not None:
                eos = (committed == eos_id).nonzero(as_tuple=False)
                if eos.numel():
                    committed = committed[:int(eos[0, 0]) + 1]
                    accepted = min(accepted, committed.numel())
            generated_before = outputs[original].numel() - prompt_lengths[original]
            committed = committed[:max_new_tokens - generated_before]
            accepted = min(accepted, committed.numel())
            outputs[original] = torch.cat((outputs[original], committed))
            committed_rows.append(committed)
            stats.accepted_draft_tokens += accepted
            stats.proposed_draft_tokens += length
            stats.sequence_iterations += 1
            stats.generated_tokens += committed.numel()
            finished.append(
                (eos_id is not None and committed.numel()
                 and committed[-1].item() == eos_id)
                or generated_before + committed.numel() >= max_new_tokens
            )

        update = torch.full(
            (batch, draft_block_size), pad_id, dtype=torch.long, device=device,
        )
        update_valid = torch.zeros(
            (batch, draft_block_size), dtype=torch.bool, device=device,
        )
        for row, committed in enumerate(committed_rows):
            update[row, :committed.numel()] = committed
            update_valid[row, :committed.numel()] = True
        update_positions = logical_lengths[:, None] + offsets[None]
        model(
            input_ids=update,
            attention_mask=_cached_query_mask(block_valid, update_valid, causal=False),
            position_ids=update_positions,
            past_key_values=block_cache,
            use_cache=True,
            store_kv=True,
        )
        stats.blockwise_cache_update_forwards += 1
        model(
            input_ids=update,
            attention_mask=_cached_query_mask(causal_valid, update_valid, causal=True),
            position_ids=update_positions,
            past_key_values=causal_cache,
            use_cache=True,
            store_kv=True,
        )
        stats.causal_correction_forwards += 1
        block_valid = torch.cat((block_valid, update_valid), dim=1)
        causal_valid = torch.cat((causal_valid, update_valid), dim=1)
        logical_lengths += torch.tensor(
            [x.numel() for x in committed_rows], dtype=torch.long, device=device,
        )

        keep = torch.tensor(
            [i for i, done in enumerate(finished) if not done],
            dtype=torch.long, device=device,
        )
        if keep.numel() == 0:
            break
        _slice_dynamic_cache(block_cache, keep)
        _slice_dynamic_cache(causal_cache, keep)
        block_valid = block_valid.index_select(0, keep)
        causal_valid = causal_valid.index_select(0, keep)
        logical_lengths = logical_lengths.index_select(0, keep)
        active_ids = [active_ids[i] for i in keep.tolist()]

    torch.cuda.synchronize(device)
    stats.wall_clock_generation_seconds = perf_counter() - started
    return outputs, stats
