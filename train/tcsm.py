"""Small, auditable helpers for Monte Carlo target concrete score matching."""

from dataclasses import dataclass
import math
import time
from typing import Callable, List, Optional

import torch
import torch.nn.functional as F
from transformers.cache_utils import DynamicCache


@dataclass
class TCSMSample:
    blocks: List[torch.Tensor]
    eligible_counts: torch.Tensor

@dataclass
class TCSMScores:
    row_indices: torch.Tensor
    token_indices: torch.Tensor
    candidate_indices: torch.Tensor
    opd_topk_logprobs: torch.Tensor
    tcs_topk_probs: torch.Tensor
    num_candidate_jobs: int
    elapsed_sec: float


def tcsm_lambda(training_step: int, lambda_max: float, ramp_steps: int, ramp_type: str) -> float:
    if ramp_steps == 0:
        return float(lambda_max)
    progress = min(max(float(training_step) / float(ramp_steps), 0.0), 1.0)
    if ramp_type == "linear":
        scale = progress
    elif ramp_type in ("cos", "cosine"):
        scale = 0.5 - 0.5 * math.cos(math.pi * progress)
    else:
        raise ValueError(f"Unknown tcsm.ramp_type: {ramp_type}")
    return float(lambda_max) * scale


def teacher_logprobs_from_logits(
    logits: torch.Tensor,
) -> torch.Tensor:
    """Normal full-vocabulary ARM log-probabilities in fp32."""
    return F.log_softmax(logits.float(), dim=-1)


def sample_disagreement_tcs_positions(
    arm_logits: torch.Tensor,
    dlm_logits: torch.Tensor,
    active_mask: torch.Tensor,
    generator: Optional[torch.Generator] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Route all argmax disagreements and 10% of agreements to block TCS."""
    disagree = arm_logits.argmax(dim=-1) != dlm_logits.argmax(dim=-1)
    pi = torch.where(
        disagree,
        torch.ones_like(disagree, dtype=torch.float32),
        torch.full_like(disagree, 0.1, dtype=torch.float32),
    )
    selected = torch.bernoulli(pi, generator=generator).bool() & active_mask
    return disagree & active_mask, selected, pi


def sample_tcsm_blocks_per_row(
    loss_mask: torch.Tensor,
    response_start: int,
    block_size: int,
    blocks_per_prompt: int,
    generator: Optional[torch.Generator] = None,
) -> TCSMSample:
    """Uniformly sample active blocks independently for every LM minibatch row."""
    if blocks_per_prompt <= 0:
        raise ValueError("tcsm.blocks_per_prompt must be positive")
    response_mask = loss_mask[:, response_start:]
    num_blocks = math.ceil(response_mask.size(1) / block_size)
    blocks = []
    counts = []
    for row_mask in response_mask:
        eligible = []
        for block_id in range(num_blocks):
            start = block_id * block_size
            end = min(start + block_size, row_mask.numel())
            if row_mask[start:end].any():
                eligible.append(block_id)
        eligible_tensor = torch.tensor(eligible, dtype=torch.long, device=loss_mask.device)
        counts.append(len(eligible))
        if len(eligible) > blocks_per_prompt:
            order = torch.randperm(len(eligible), generator=generator, device=loss_mask.device)
            eligible_tensor = eligible_tensor[order[:blocks_per_prompt]]
        blocks.append(eligible_tensor)
    return TCSMSample(
        blocks=blocks,
        eligible_counts=torch.tensor(counts, dtype=torch.long, device=loss_mask.device),
    )


def build_full_vocab_tcs_target(
    opd_probs: torch.Tensor,
    candidate_indices: torch.Tensor,
    tcs_topk_probs: torch.Tensor,
) -> torch.Tensor:
    """Reference target builder used by tests and diagnostics."""
    target = opd_probs.clone()
    opd_topk = opd_probs.gather(-1, candidate_indices)
    topk_mass = opd_topk.sum(dim=-1, keepdim=True)
    target.scatter_(-1, candidate_indices, topk_mass * tcs_topk_probs)
    return target


def full_target_forward_kl(
    student_logprobs: torch.Tensor,
    full_opd_logprobs: torch.Tensor,
    candidate_indices: torch.Tensor,
    tcs_topk_probs: torch.Tensor,
) -> torch.Tensor:
    """KL(q_hat || student), retaining the unchanged full-vocabulary OPD tail."""
    opd_probs = full_opd_logprobs.exp()
    finite_opd = torch.isfinite(full_opd_logprobs)
    full_opd_terms = torch.where(
        finite_opd,
        opd_probs * (full_opd_logprobs - student_logprobs),
        torch.zeros_like(opd_probs),
    )

    opd_topk_logprobs = full_opd_logprobs.gather(-1, candidate_indices)
    student_topk_logprobs = student_logprobs.gather(-1, candidate_indices)
    opd_topk_probs = opd_topk_logprobs.exp()
    topk_mass = opd_topk_probs.sum(dim=-1, keepdim=True)
    tcs_mass_probs = topk_mass * tcs_topk_probs
    tcs_mass_logprobs = topk_mass.log() + tcs_topk_probs.clamp_min(torch.finfo(torch.float32).tiny).log()
    finite_topk = torch.isfinite(opd_topk_logprobs)
    opd_topk_terms = torch.where(
        finite_topk,
        opd_topk_probs * (opd_topk_logprobs - student_topk_logprobs),
        torch.zeros_like(opd_topk_probs),
    )
    tcs_terms = tcs_mass_probs * (tcs_mass_logprobs - student_topk_logprobs)
    return (
        full_opd_terms.sum(dim=-1)
        - opd_topk_terms.sum(dim=-1)
        + tcs_terms.sum(dim=-1)
    )


def mc_tcsm_correction_per_row(
    token_deltas: torch.Tensor,
    row_indices: torch.Tensor,
    sample: TCSMSample,
    active_counts: torch.Tensor,
    batch_size: int,
) -> torch.Tensor:
    """N/m-reweight token sums, normalize per row, then average rows."""
    row_corrections = []
    for row in range(batch_size):
        n_blocks = int(sample.eligible_counts[row].item())
        m_blocks = int(sample.blocks[row].numel())
        if n_blocks == 0 or m_blocks == 0:
            row_corrections.append(token_deltas.new_zeros(()))
            continue
        row_sum = token_deltas[row_indices == row].sum()
        row_corrections.append(
            (float(n_blocks) / float(m_blocks)) * row_sum / active_counts[row].clamp_min(1)
        )
    return torch.stack(row_corrections).mean()


def token_tcsm_correction_per_row(
    token_deltas: torch.Tensor,
    row_indices: torch.Tensor,
    active_counts: torch.Tensor,
    batch_size: int,
) -> torch.Tensor:
    """Normalize token-routed TCS corrections per row, then average rows."""
    row_corrections = []
    for row in range(batch_size):
        row_sum = token_deltas[row_indices == row].sum()
        row_corrections.append(row_sum / active_counts[row].clamp_min(1))
    return torch.stack(row_corrections).mean()


def _select_cache_rows(cache, rows: torch.Tensor) -> DynamicCache:
    # DynamicCache.update concatenates into new tensors, so these selected prefix
    # tensors are not modified when the short candidate continuation is appended.
    return DynamicCache(
        (key.index_select(0, rows), value.index_select(0, rows))
        for key, value in zip(cache.key_cache, cache.value_cache)
    )


def _tcsm_position_masks(
    loss_mask: torch.Tensor,
    sequence_ends: torch.Tensor,
    sample: Optional[TCSMSample],
    response_start: int,
    block_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    selected = torch.zeros_like(loss_mask)
    counterfactual = torch.zeros_like(loss_mask)
    seq_len = loss_mask.size(1)
    if sample is None:
        block_ids_per_row = [
            torch.unique(
                loss_mask[row, response_start:].nonzero(as_tuple=True)[0] // block_size
            )
            for row in range(loss_mask.size(0))
        ]
    else:
        block_ids_per_row = sample.blocks
    for row, block_ids in enumerate(block_ids_per_row):
        for block_id in block_ids.tolist():
            start = response_start + block_id * block_size
            end = min(start + block_size, int(sequence_ends[row].item()), seq_len)
            selected[row, start:end] = loss_mask[row, start:end]
            if end > start:
                counterfactual[row, start:end] = loss_mask[row, start:end]
                counterfactual[row, end - 1] = False
    return selected, counterfactual


def count_tcsm_counterfactual_work(
    loss_mask: torch.Tensor,
    sequence_ends: torch.Tensor,
    sample: TCSMSample,
    response_start: int,
    block_size: int,
    top_k: int,
    counterfactual_batch_size: int,
) -> tuple[int, int]:
    """Return exact (candidate chunks, candidate jobs) for one LM minibatch."""
    _, counterfactual = _tcsm_position_masks(
        loss_mask, sequence_ends, sample, response_start, block_size
    )
    total_chunks = 0
    for pos in counterfactual.any(dim=0).nonzero(as_tuple=True)[0].tolist():
        rows = counterfactual[:, pos].nonzero(as_tuple=True)[0]
        branch_lengths = torch.minimum(
            sequence_ends[rows],
            torch.full_like(
                rows,
                response_start + ((pos - response_start) // block_size + 1) * block_size,
            ),
        ) - pos - 1
        for branch_length in branch_lengths.unique().tolist():
            total_chunks += math.ceil(
                int((branch_lengths == branch_length).sum().item()) * top_k
                / counterfactual_batch_size
            )
    return total_chunks, int(counterfactual.sum().item()) * top_k


@torch.no_grad()
def compute_block_tcs_topk_scores(
    teacher_model,
    clean_input_ids: torch.Tensor,
    position_ids: torch.Tensor,
    teacher_logprobs: torch.Tensor,
    loss_mask: torch.Tensor,
    sequence_ends: torch.Tensor,
    sample: Optional[TCSMSample],
    response_start: int,
    block_size: int,
    top_k: int,
    counterfactual_batch_size: int,
    logprob_fn: Callable[[torch.Tensor], torch.Tensor],
    pad_token_id: int,
    show_progress: bool,
    is_main_process: bool,
    progress_bar=None,
) -> TCSMScores:
    """Score Top-K candidate branches using one incrementally advanced prefix cache."""
    if top_k <= 0:
        raise ValueError("MC-TCS requires training.top_k_logits > 0")
    if counterfactual_batch_size <= 0:
        raise ValueError("tcsm.counterfactual_batch_size must be positive")
    if response_start <= 0:
        raise ValueError("MC-TCS requires a non-empty causal prompt prefix")

    device = clean_input_ids.device
    batch_size, seq_len = clean_input_ids.shape
    top_k = min(top_k, teacher_logprobs.size(-1))
    selected, counterfactual = _tcsm_position_masks(
        loss_mask, sequence_ends, sample, response_start, block_size
    )

    positions = selected.any(dim=0).nonzero(as_tuple=True)[0].tolist()
    counterfactual_positions = counterfactual.any(dim=0).nonzero(as_tuple=True)[0].tolist()
    candidate_jobs = int(counterfactual.sum().item()) * top_k
    empty_long = torch.empty(0, dtype=torch.long, device=device)
    empty_topk = torch.empty(0, top_k, dtype=torch.float32, device=device)
    if not positions:
        return TCSMScores(empty_long, empty_long, empty_long.view(0, 1).expand(0, top_k),
                          empty_topk, empty_topk, 0, 0.0)

    position_data = {}
    for pos in positions:
        rows = selected[:, pos].nonzero(as_tuple=True)[0]
        opd_topk_lp, candidates = teacher_logprobs[rows, pos].topk(top_k, dim=-1)
        position_data[pos] = {
            "rows": rows,
            "candidates": candidates,
            "opd_topk_logprobs": opd_topk_lp.float(),
            # At block end q_TCS_C is exactly q_OPD conditioned on C.
            "tcs_topk_probs": F.softmax(opd_topk_lp.float(), dim=-1),
        }

    if not counterfactual_positions:
        return TCSMScores(
            row_indices=torch.cat([position_data[pos]["rows"] for pos in positions]),
            token_indices=torch.cat([
                torch.full_like(position_data[pos]["rows"], pos) for pos in positions
            ]),
            candidate_indices=torch.cat([position_data[pos]["candidates"] for pos in positions]),
            opd_topk_logprobs=torch.cat([
                position_data[pos]["opd_topk_logprobs"] for pos in positions
            ]),
            tcs_topk_probs=torch.cat([
                position_data[pos]["tcs_topk_probs"] for pos in positions
            ]),
            num_candidate_jobs=0,
            elapsed_sec=0.0,
        )

    total_chunks = 0
    for pos in counterfactual_positions:
        rows = counterfactual[:, pos].nonzero(as_tuple=True)[0]
        branch_lengths = torch.minimum(
            sequence_ends[rows],
            torch.full_like(rows, response_start + ((pos - response_start) // block_size + 1) * block_size),
        ) - pos - 1
        for branch_length in branch_lengths.unique().tolist():
            total_chunks += math.ceil(
                int((branch_lengths == branch_length).sum().item()) * top_k
                / counterfactual_batch_size
            )
    progress = progress_bar
    owns_progress = False
    if progress is None and show_progress and is_main_process:
        from tqdm.auto import tqdm
        progress = tqdm(total=total_chunks, desc="MC-TCS counterfactuals", unit="batch",
                        dynamic_ncols=True, leave=False)
        owns_progress = True
    if owns_progress:
        active_blocks = int(sum(
            torch.unique((selected[row, response_start:].nonzero(as_tuple=True)[0]) // block_size).numel()
            for row in range(batch_size)
        ))
        progress.set_postfix(prompts=batch_size,
                             active_blocks=active_blocks,
                             candidate_jobs=candidate_jobs,
                             cf_batch=counterfactual_batch_size)

    pad_mask = clean_input_ids.ne(pad_token_id)
    first_pos = counterfactual_positions[0]
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    prefix_output = teacher_model(
        input_ids=clean_input_ids[:, :first_pos],
        attention_mask=pad_mask[:, :first_pos],
        position_ids=position_ids[:, :first_pos],
        use_cache=True,
        logits_to_keep=1,
    )
    prefix_cache = prefix_output.past_key_values

    try:
        for position_number, pos in enumerate(counterfactual_positions):
            rows = counterfactual[:, pos].nonzero(as_tuple=True)[0]
            all_rows = position_data[pos]["rows"]
            row_locations = torch.searchsorted(all_rows, rows)
            candidates = position_data[pos]["candidates"][row_locations]
            opd_topk_lp = position_data[pos]["opd_topk_logprobs"][row_locations]
            flat_rows = rows.repeat_interleave(top_k)
            flat_candidates = candidates.reshape(-1)
            flat_scores = opd_topk_lp.reshape(-1).float().clone()
            nominal_end = response_start + ((pos - response_start) // block_size + 1) * block_size
            row_branch_lengths = torch.minimum(
                sequence_ends[rows], torch.full_like(rows, nominal_end)
            ) - pos - 1
            flat_branch_lengths = row_branch_lengths.repeat_interleave(top_k)

            for branch_length in flat_branch_lengths.unique().tolist():
                bucket = (flat_branch_lengths == branch_length).nonzero(as_tuple=True)[0]
                for bucket_start in range(0, bucket.numel(), counterfactual_batch_size):
                    job_indices = bucket[bucket_start:bucket_start + counterfactual_batch_size]
                    chunk_rows = flat_rows[job_indices]
                    chunk_candidates = flat_candidates[job_indices]
                    suffix_targets = clean_input_ids[chunk_rows, pos + 1:pos + 1 + branch_length]
                    branch_inputs = torch.cat(
                        [chunk_candidates[:, None], suffix_targets[:, :-1]], dim=1
                    )
                    branch_cache = _select_cache_rows(prefix_cache, chunk_rows)
                    branch_attention = torch.cat(
                        [pad_mask[chunk_rows, :pos],
                         torch.ones(job_indices.numel(), branch_inputs.size(1),
                                    dtype=torch.bool, device=device)],
                        dim=1,
                    )
                    outputs = teacher_model(
                        input_ids=branch_inputs,
                        attention_mask=branch_attention,
                        position_ids=position_ids[chunk_rows, pos:pos + branch_inputs.size(1)],
                        past_key_values=branch_cache,
                        use_cache=False,
                    )
                    branch_lp = logprob_fn(outputs.logits).float()
                    flat_scores[job_indices] += branch_lp.gather(
                        -1, suffix_targets.unsqueeze(-1)
                    ).squeeze(-1).sum(dim=-1)
                    if progress is not None:
                        progress.update(1)

            position_data[pos]["tcs_topk_probs"][row_locations] = F.softmax(
                flat_scores.view(rows.numel(), top_k), dim=-1
            )

            if position_number + 1 < len(counterfactual_positions):
                next_pos = counterfactual_positions[position_number + 1]
                # Append x[pos:next_pos] once; the resulting cache is x_<next_pos.
                teacher_model(
                    input_ids=clean_input_ids[:, pos:next_pos],
                    attention_mask=pad_mask[:, :next_pos],
                    position_ids=position_ids[:, pos:next_pos],
                    past_key_values=prefix_cache,
                    use_cache=True,
                    logits_to_keep=1,
                )
    finally:
        if owns_progress:
            progress.close()

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return TCSMScores(
        row_indices=torch.cat([position_data[pos]["rows"] for pos in positions]),
        token_indices=torch.cat([
            torch.full_like(position_data[pos]["rows"], pos) for pos in positions
        ]),
        candidate_indices=torch.cat([position_data[pos]["candidates"] for pos in positions]),
        opd_topk_logprobs=torch.cat([
            position_data[pos]["opd_topk_logprobs"] for pos in positions
        ]),
        tcs_topk_probs=torch.cat([
            position_data[pos]["tcs_topk_probs"] for pos in positions
        ]),
        num_candidate_jobs=candidate_jobs,
        elapsed_sec=time.perf_counter() - started,
    )
