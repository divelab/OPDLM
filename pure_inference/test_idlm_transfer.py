# pure_inference/eval_opdlm_introspective.py

import argparse
import json
import os
import sys

import torch
from tqdm import tqdm
from transformers import AutoModelForMaskedLM, AutoTokenizer


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from eval_utils import DATASET_CONFIGS, build_evalplus_prompt
from domain_reward import extract_answer, check_answer
from sample.bd3lm_rl_rollout import _register_a2d_model_classes


DEFAULT_MODEL = (
    "/scratch/user/shubhamprshr_tamu.edu/OPDLM/"
    "pretrained_models/OPDLM-0.6B"
)


# ============================================================
# Batched causal forward with left padding
# ============================================================

def pad_left(seqs, pad_id):
    B = len(seqs)
    max_len = max(x.numel() for x in seqs)

    ids = torch.full(
        (B, max_len),
        pad_id,
        dtype=torch.long,
        device=seqs[0].device,
    )

    valid = torch.zeros(
        (B, max_len),
        dtype=torch.bool,
        device=seqs[0].device,
    )

    position_ids = torch.zeros(
        (B, max_len),
        dtype=torch.long,
        device=seqs[0].device,
    )

    for b, seq in enumerate(seqs):
        n = seq.numel()
        start = max_len - n

        ids[b, start:] = seq
        valid[b, start:] = True
        position_ids[b, start:] = torch.arange(
            n,
            device=seq.device,
        )

    return ids, valid, position_ids


def make_causal_mask(valid):
    """
    [B,T] -> [B,1,T,T]

    Strict causal mask plus padding masking.
    """
    B, T = valid.shape
    device = valid.device

    causal = torch.tril(
        torch.ones(T, T, dtype=torch.bool, device=device)
    )

    mask = causal[None, :, :].expand(B, -1, -1).clone()

    # Cannot attend to padded keys.
    mask &= valid[:, None, :]

    # Avoid completely empty padded query rows.
    for b in range(B):
        pad_q = ~valid[b]
        if pad_q.any():
            idx = pad_q.nonzero(as_tuple=False).squeeze(-1)
            mask[b, idx, idx] = True

    return mask[:, None, :, :]


@torch.no_grad()
def forward_causal_batch(model, seqs, pad_id):
    ids, valid, position_ids = pad_left(seqs, pad_id)

    logits = model(
        input_ids=ids,
        attention_mask=make_causal_mask(valid),
        position_ids=position_ids,
        use_cache=False,
    ).logits

    return logits


@torch.no_grad()
def forward_cached_tokens(
    model,
    input_ids,
    position_ids,
    past_key_values,
    cache_valid,
    active_rows,
):
    """Append a fixed-width token block to a shared batched KV cache.

    Cache tensors have one physical length for the whole batch. Tokens from
    rejected branches and mask proposals remain physically present but are
    excluded from all later attention by ``cache_valid``.
    """
    B, Q = input_ids.shape
    past_len = cache_valid.shape[1]
    device = input_ids.device

    attention_mask = torch.zeros(
        (B, 1, Q, past_len + Q),
        dtype=torch.bool,
        device=device,
    )
    if past_len:
        attention_mask[:, :, :, :past_len] = cache_valid[:, None, None, :]

    current_causal = torch.tril(
        torch.ones(Q, Q, dtype=torch.bool, device=device)
    )
    attention_mask[:, 0, :, past_len:] = (
        current_causal[None, :, :] & active_rows[:, None, None]
    )

    # Dummy rows still need a nonempty attention row to avoid softmax NaNs;
    # none of these dummy states are marked valid for future calls.
    inactive = (~active_rows).nonzero(as_tuple=False).flatten()
    if inactive.numel():
        diag = torch.arange(Q, device=device)
        attention_mask[inactive[:, None], 0, diag[None, :], past_len + diag[None, :]] = True

    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        use_cache=True,
    )
    return outputs.logits, outputs.past_key_values


@torch.no_grad()
def generate_isd_batch_cached(
    model,
    prefixes,
    mask_id,
    pad_id,
    eos_id,
    max_new_tokens,
    stride,
):
    """Batched IDLM decoding with a rollback-safe shared KV cache."""
    B = len(prefixes)
    device = prefixes[0].device
    committed = [x.clone() for x in prefixes]
    prompt_lens = [x.numel() for x in prefixes]

    forwards = [0] * B
    accepted_specs = [0] * B
    checked_specs = [0] * B
    rejections = [0] * B
    m1_agree_count = [0] * B
    m1_checked = [0] * B
    finished = [False] * B
    model_calls = 0

    # Prefill the left-padded prompts once. Padding KVs remain masked forever.
    prompt_ids, prompt_valid, prompt_positions = pad_left(prefixes, pad_id)
    prefill = model(
        input_ids=prompt_ids,
        attention_mask=make_causal_mask(prompt_valid),
        position_ids=prompt_positions,
        use_cache=True,
    )
    past_key_values = prefill.past_key_values
    cache_valid = prompt_valid.clone()
    exact = prefill.logits[:, -1].argmax(-1)
    model_calls += 1
    for b in range(B):
        forwards[b] += 1

    # Initial masked proposals. They may attend to each other within this call
    # but are not retained as valid cache entries afterward.
    masks = torch.full((B, stride), mask_id, dtype=torch.long, device=device)
    mask_positions = torch.stack([
        torch.arange(n, n + stride, device=device) for n in prompt_lens
    ])
    active_rows = torch.ones(B, dtype=torch.bool, device=device)
    mask_logits, past_key_values = forward_cached_tokens(
        model,
        masks,
        mask_positions,
        past_key_values,
        cache_valid,
        active_rows,
    )
    cache_valid = torch.cat([
        cache_valid,
        torch.zeros((B, stride), dtype=torch.bool, device=device),
    ], dim=1)
    mask_preds = mask_logits.argmax(-1)
    pending_by_row = {
        b: torch.cat([exact[b:b + 1], mask_preds[b, 1:]], dim=0)
        for b in range(B)
    }
    model_calls += 1
    for b in range(B):
        forwards[b] += 1
        m1_checked[b] += 1
        m1_agree_count[b] += int(exact[b].item() == mask_preds[b, 0].item())

    while not all(finished):
        active_rows = torch.tensor(
            [not value for value in finished],
            dtype=torch.bool,
            device=device,
        )
        main_ids = torch.full(
            (B, 2 * stride), pad_id, dtype=torch.long, device=device
        )
        main_positions = torch.zeros_like(main_ids)
        for b in range(B):
            if finished[b]:
                continue
            logical_start = committed[b].numel()
            main_ids[b, :stride] = pending_by_row[b]
            main_ids[b, stride:] = mask_id
            main_positions[b] = torch.arange(
                logical_start,
                logical_start + 2 * stride,
                device=device,
            )

        logits, past_key_values = forward_cached_tokens(
            model,
            main_ids,
            main_positions,
            past_key_values,
            cache_valid,
            active_rows,
        )
        model_calls += 1

        retain_main = torch.zeros(
            (B, 2 * stride), dtype=torch.bool, device=device
        )
        rejected_rows = []
        correction_tokens = {}

        for b in range(B):
            if finished[b]:
                continue
            forwards[b] += 1
            old_pending = pending_by_row[b]
            accepted = [old_pending[0].item()]
            matched_pending = 1
            rejected = False

            for i in range(1, stride):
                checked_specs[b] += 1
                causal_token = logits[b, i - 1].argmax(-1).item()
                draft_token = old_pending[i].item()
                if causal_token == draft_token:
                    accepted_specs[b] += 1
                    accepted.append(draft_token)
                    matched_pending += 1
                else:
                    accepted.append(causal_token)
                    rejections[b] += 1
                    rejected = True
                    break

            if eos_id is not None and eos_id in accepted:
                accepted = accepted[:accepted.index(eos_id) + 1]

            remaining = max_new_tokens - (
                committed[b].numel() - prompt_lens[b]
            )
            accepted = accepted[:remaining]
            retain_count = min(matched_pending, len(accepted))
            retain_main[b, :retain_count] = True

            if accepted:
                committed[b] = torch.cat([
                    committed[b],
                    torch.tensor(accepted, dtype=torch.long, device=device),
                ])

            generated = committed[b].numel() - prompt_lens[b]
            hit_eos = eos_id is not None and eos_id in accepted
            if generated >= max_new_tokens or hit_eos:
                finished[b] = True
                continue

            if rejected:
                rejected_rows.append(b)
                correction_tokens[b] = accepted[-1]
                continue

            bonus = logits[b, stride - 1].argmax(-1)
            fresh_mask_preds = logits[b, stride:].argmax(-1)
            m1_checked[b] += 1
            m1_agree_count[b] += int(
                bonus.item() == fresh_mask_preds[0].item()
            )
            pending_by_row[b] = torch.cat([
                bonus[None], fresh_mask_preds[1:]
            ])

        cache_valid = torch.cat([cache_valid, retain_main], dim=1)

        if rejected_rows:
            # Recompute only the causal correction token plus fresh masks.
            # Matching speculative-prefix KVs from the main pass were retained.
            Q = stride + 1
            retry_ids = torch.full(
                (B, Q), pad_id, dtype=torch.long, device=device
            )
            retry_positions = torch.zeros_like(retry_ids)
            retry_active = torch.zeros(B, dtype=torch.bool, device=device)
            for b in rejected_rows:
                retry_active[b] = True
                retry_ids[b, 0] = correction_tokens[b]
                retry_ids[b, 1:] = mask_id
                correction_pos = committed[b].numel() - 1
                retry_positions[b] = torch.arange(
                    correction_pos,
                    correction_pos + Q,
                    device=device,
                )

            retry_logits, past_key_values = forward_cached_tokens(
                model,
                retry_ids,
                retry_positions,
                past_key_values,
                cache_valid,
                retry_active,
            )
            model_calls += 1
            retain_retry = torch.zeros(
                (B, Q), dtype=torch.bool, device=device
            )
            retain_retry[rejected_rows, 0] = True
            cache_valid = torch.cat([cache_valid, retain_retry], dim=1)

            for b in rejected_rows:
                forwards[b] += 1
                next_exact = retry_logits[b, 0].argmax(-1)
                retry_mask_preds = retry_logits[b, 1:].argmax(-1)
                m1_checked[b] += 1
                m1_agree_count[b] += int(
                    next_exact.item() == retry_mask_preds[0].item()
                )
                pending_by_row[b] = torch.cat([
                    next_exact[None], retry_mask_preds[1:]
                ])

    outputs = []
    stats = []
    for b in range(B):
        out = committed[b][prompt_lens[b]:]
        outputs.append(out)
        n_tokens = out.numel()
        stats.append({
            "generated_tokens": n_tokens,
            "forward_passes": forwards[b],
            "tpf": n_tokens / max(forwards[b], 1),
            "accepted_specs": accepted_specs[b],
            "checked_specs": checked_specs[b],
            "spec_acceptance_rate": (
                accepted_specs[b] / max(checked_specs[b], 1)
            ),
            "rejections": rejections[b],
            "mask1_causal_agreement": (
                m1_agree_count[b] / max(m1_checked[b], 1)
            ),
        })

    return outputs, stats, model_calls


# ============================================================
# Bootstrap
# ============================================================

@torch.no_grad()
def bootstrap_batch(
    model,
    committed,
    mask_id,
    pad_id,
    stride,
):
    """
    For every sequence:

       [prefix][M1 M2 ... MB]

       last clean -> x1
       M1         -> x1
       M2         -> x2
       ...
       MB         -> xB
    """

    seqs = []

    for prefix in committed:
        masks = torch.full(
            (stride,),
            mask_id,
            dtype=torch.long,
            device=prefix.device,
        )

        seqs.append(torch.cat([prefix, masks]))

    logits = forward_causal_batch(
        model,
        seqs,
        pad_id,
    )

    # Because of LEFT padding, every sequence ends with B masks.
    T = logits.shape[1]

    exact = logits[:, T - stride - 1].argmax(-1)

    mask_preds = logits[
        :,
        T - stride:T,
    ].argmax(-1)

    m1_agree = exact == mask_preds[:, 0]

    pending = torch.cat(
        [
            exact[:, None],
            mask_preds[:, 1:],
        ],
        dim=1,
    )

    return pending, m1_agree


# ============================================================
# Batched introspective generation
# ============================================================

@torch.no_grad()
def generate_isd_batch(
    model,
    prefixes,
    mask_id,
    pad_id,
    eos_id,
    max_new_tokens,
    stride,
):
    B = len(prefixes)

    committed = [x.clone() for x in prefixes]
    prompt_lens = [x.numel() for x in prefixes]

    forwards = [0] * B
    accepted_specs = [0] * B
    checked_specs = [0] * B
    rejections = [0] * B

    m1_agree_count = [0] * B
    m1_checked = [0] * B

    finished = [False] * B

    # Actual batched model-call counter.
    model_calls = 0

    # --------------------------------------------------------
    # Initial bootstrap
    # --------------------------------------------------------

    pending, agree = bootstrap_batch(
        model,
        committed,
        mask_id,
        pad_id,
        stride,
    )

    model_calls += 1

    for b in range(B):
        forwards[b] += 1
        m1_checked[b] += 1
        m1_agree_count[b] += int(agree[b].item())

    pending_by_row = {
        b: pending[b].clone()
        for b in range(B)
    }

    # --------------------------------------------------------
    # Decode
    # --------------------------------------------------------

    while not all(finished):

        active = [
            b for b in range(B)
            if not finished[b]
        ]

        seqs = []

        for b in active:
            masks = torch.full(
                (stride,),
                mask_id,
                dtype=torch.long,
                device=committed[b].device,
            )

            seqs.append(
                torch.cat(
                    [
                        committed[b],
                        pending_by_row[b],
                        masks,
                    ]
                )
            )

        logits = forward_causal_batch(
            model,
            seqs,
            pad_id,
        )

        model_calls += 1

        T = logits.shape[1]

        # Layout at right edge:
        #
        # [pending B tokens][fresh B masks]
        #
        pending_start = T - 2 * stride
        fresh_start = T - stride

        rejected_rows = []

        for local_b, global_b in enumerate(active):
            forwards[global_b] += 1

            old_pending = pending_by_row[global_b]

            accepted = [old_pending[0].item()]
            rejected = False

            # Verify speculative positions 2..B.
            for i in range(1, stride):
                checked_specs[global_b] += 1

                anchor = pending_start + i - 1

                causal_token = (
                    logits[local_b, anchor]
                    .argmax(-1)
                    .item()
                )

                draft_token = old_pending[i].item()

                if causal_token == draft_token:
                    accepted_specs[global_b] += 1
                    accepted.append(draft_token)

                else:
                    accepted.append(causal_token)
                    rejections[global_b] += 1
                    rejected = True
                    break

            # Stop exactly at EOS if generated.
            if eos_id is not None and eos_id in accepted:
                eos_pos = accepted.index(eos_id)
                accepted = accepted[:eos_pos + 1]

            remaining = max_new_tokens - (
                committed[global_b].numel()
                - prompt_lens[global_b]
            )

            accepted = accepted[:remaining]

            if accepted:
                committed[global_b] = torch.cat(
                    [
                        committed[global_b],
                        torch.tensor(
                            accepted,
                            dtype=torch.long,
                            device=committed[global_b].device,
                        ),
                    ]
                )

            generated = (
                committed[global_b].numel()
                - prompt_lens[global_b]
            )

            if (
                generated >= max_new_tokens
                or (
                    eos_id is not None
                    and eos_id in accepted
                )
            ):
                finished[global_b] = True
                continue

            if rejected:
                rejected_rows.append(global_b)
                continue

            # ------------------------------------------------
            # Full acceptance:
            # reuse this same fused forward.
            # ------------------------------------------------

            bonus = (
                logits[
                    local_b,
                    pending_start + stride - 1,
                ]
                .argmax(-1)
            )

            fresh_mask_preds = (
                logits[
                    local_b,
                    fresh_start:fresh_start + stride,
                ]
                .argmax(-1)
            )

            m1_checked[global_b] += 1
            m1_agree_count[global_b] += int(
                bonus.item()
                == fresh_mask_preds[0].item()
            )

            pending_by_row[global_b] = torch.cat(
                [
                    bonus[None],
                    fresh_mask_preds[1:],
                ]
            )

        # ----------------------------------------------------
        # Rejected rows require a fresh bootstrap.
        # Batch all rejected rows together.
        # ----------------------------------------------------

        if rejected_rows:
            rejected_prefixes = [
                committed[b]
                for b in rejected_rows
            ]

            new_pending, agree = bootstrap_batch(
                model,
                rejected_prefixes,
                mask_id,
                pad_id,
                stride,
            )

            model_calls += 1

            for j, b in enumerate(rejected_rows):
                forwards[b] += 1

                m1_checked[b] += 1
                m1_agree_count[b] += int(
                    agree[j].item()
                )

                pending_by_row[b] = (
                    new_pending[j].clone()
                )

    # ========================================================
    # Final outputs/stats
    # ========================================================

    outputs = []
    stats = []

    for b in range(B):
        out = committed[b][prompt_lens[b]:]
        outputs.append(out)

        n_tokens = out.numel()

        stats.append({
            "generated_tokens": n_tokens,
            "forward_passes": forwards[b],

            # Sequence-normalized TPF.
            "tpf": (
                n_tokens / max(forwards[b], 1)
            ),

            "accepted_specs": accepted_specs[b],
            "checked_specs": checked_specs[b],

            "spec_acceptance_rate": (
                accepted_specs[b]
                / max(checked_specs[b], 1)
            ),

            "rejections": rejections[b],

            "mask1_causal_agreement": (
                m1_agree_count[b]
                / max(m1_checked[b], 1)
            ),
        })

    return outputs, stats, model_calls


# ============================================================
# Prompt
# ============================================================

def build_prompt(tokenizer, row, ds_cfg, prompt_style="idlm"):
    question = row["question"]

    if prompt_style == "idlm" and ds_cfg.get("chat_style") == "evalplus_prefill":
        return build_evalplus_prompt(
            question,
            tokenizer,
        )

    if prompt_style == "qwen" and ds_cfg.get("chat_style") == "evalplus_prefill":
        body = (
            "Please provide a self-contained Python script that solves the "
            "following problem in a markdown code block:\n```\n"
            f"{question.strip()}\n```"
        )
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": body}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )

    template = ds_cfg.get("prompt_template")

    if callable(template):
        body = template(row, question)
    elif template is not None:
        body = template.format(question=question)
    else:
        body = question

    messages = (
        body
        if isinstance(body, list)
        else [{"role": "user", "content": body}]
    )

    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
    )

    parser.add_argument(
        "--dataset",
        default="GSM8K",
    )

    parser.add_argument(
        "--max_token",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--stride",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--use_cache",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use rollback-safe KV caching (default: enabled).",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--scorer",
        default="math_verify",
        choices=["math_verify", "opencompass"],
    )

    parser.add_argument(
        "--out_dir",
        default="pure_inference/results",
    )

    parser.add_argument(
        "--prompt_style",
        choices=("idlm", "qwen"),
        default="idlm",
        help="IDLM uses EvalPlus assistant prefill; Qwen uses a normal user turn.",
    )

    args = parser.parse_args()

    ds_cfg = DATASET_CONFIGS[args.dataset]

    data_path = os.path.join(
        REPO_ROOT,
        "data",
        ds_cfg["path"],
    )

    with open(data_path) as f:
        data = json.load(f)

    if args.limit is not None:
        data = data[:args.limit]

    max_token = (
        args.max_token
        if args.max_token is not None
        else ds_cfg.get(
            "dllm_max_new_tokens",
            2048,
        )
    )

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        trust_remote_code=True,
    )

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    _register_a2d_model_classes()

    model = AutoModelForMaskedLM.from_pretrained(
        args.model,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    ).cuda().eval()

    mask_id = tokenizer.mask_token_id
    pad_id = tokenizer.pad_token_id

    print(f"Dataset: {args.dataset}")
    print(f"Examples: {len(data)}")
    print(f"Batch size: {args.batch_size}")
    print(f"Stride: {args.stride}")
    print(f"Max token: {max_token}")
    print(f"KV cache: {args.use_cache}")

    results = []

    total_correct = 0
    total_scored = 0

    total_tokens = 0
    total_forwards = 0
    total_accepted = 0
    total_checked = 0

    total_model_calls = 0

    deferred = ds_cfg.get("defer_scoring")

    # --------------------------------------------------------
    # Batched evaluation
    # --------------------------------------------------------

    for start in tqdm(
        range(0, len(data), args.batch_size),
        desc=args.dataset,
    ):
        batch_rows = data[
            start:start + args.batch_size
        ]

        prefixes = []

        for row in batch_rows:
            prompt = build_prompt(tokenizer, row, ds_cfg, args.prompt_style)

            ids = tokenizer(
                prompt,
                return_tensors="pt",
                add_special_tokens=False,
            ).input_ids[0].cuda()

            prefixes.append(ids)

        generate_fn = (
            generate_isd_batch_cached
            if args.use_cache
            else generate_isd_batch
        )
        outputs, batch_stats, model_calls = (
            generate_fn(
                model=model,
                prefixes=prefixes,
                mask_id=mask_id,
                pad_id=pad_id,
                eos_id=tokenizer.eos_token_id,
                max_new_tokens=max_token,
                stride=args.stride,
            )
        )

        total_model_calls += model_calls

        for row, output_ids, stats in zip(
            batch_rows,
            outputs,
            batch_stats,
        ):
            response = tokenizer.decode(
                output_ids,
                skip_special_tokens=True,
            )

            extracted = extract_answer(
                response,
                data_i=row,
                ds_cfg=ds_cfg,
                scorer=args.scorer,
            )

            if deferred:
                correct = None
            else:
                correct = check_answer(
                    extracted,
                    data_i=row,
                    ds_cfg=ds_cfg,
                    scorer=args.scorer,
                )

                total_correct += int(correct)
                total_scored += 1

            total_tokens += stats["generated_tokens"]
            total_forwards += stats["forward_passes"]

            total_accepted += stats["accepted_specs"]
            total_checked += stats["checked_specs"]

            results.append({
                **row,
                "response": response,
                "extracted_answer": extracted,
                "correct": correct,
                **stats,
            })

        msg = (
            f"TPF={total_tokens / max(total_forwards,1):.3f} "
            f"accept={total_accepted / max(total_checked,1):.3f}"
        )

        if total_scored:
            msg += (
                f" acc="
                f"{total_correct / total_scored:.3f}"
            )

        tqdm.write(msg)

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    model_name = os.path.basename(
        args.model.rstrip("/")
    )

    out_dir = os.path.join(
        REPO_ROOT,
        args.out_dir,
        f"{model_name}_{args.dataset}"
        f"_introspective_B{args.stride}",
    )

    os.makedirs(out_dir, exist_ok=True)

    out_path = os.path.join(
        out_dir,
        "outputs.json",
    )

    with open(out_path, "w") as f:
        json.dump(
            results,
            f,
            indent=2,
            ensure_ascii=False,
        )

    # HumanEval and MBPP require execution-based scoring. Run the repository's
    # canonical EvalPlus pipeline automatically after generation so direct
    # invocations of this script report code accuracy without a second job.
    evalplus_summary = None
    if deferred == "evalplus":
        from rl_execute import evaluate_evalplus_dataset

        for item in results:
            item["full_output"] = [item.get("response", "")]

        results = evaluate_evalplus_dataset(
            results,
            ds_cfg["evalplus_dataset"],
            work_dir=os.path.join(out_dir, "evalplus"),
        )
        evalplus_summary = (
            results[0].get("evalplus_pass_at_k", {})
            if results else {}
        )
        scored_path = os.path.join(out_dir, "outputs_scored.json")
        with open(scored_path, "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 70)
    print("FINAL")
    print("=" * 70)

    if total_scored:
        print(
            f"Accuracy: "
            f"{total_correct}/{total_scored} "
            f"= {total_correct / total_scored:.4f}"
        )
    elif evalplus_summary is not None:
        print(
            "EvalPlus base pass@1: "
            f"{evalplus_summary.get('base', 0.0):.4f}"
        )
        print(
            "EvalPlus plus pass@1: "
            f"{evalplus_summary.get('plus', 0.0):.4f}"
        )

    print(
        f"TPF: "
        f"{total_tokens / max(total_forwards,1):.4f}"
    )

    print(
        f"Spec acceptance: "
        f"{total_accepted / max(total_checked,1):.4f}"
    )

    print(f"Committed tokens: {total_tokens}")

    # Important distinction:
    print(
        f"Sequence-forwards: {total_forwards}"
    )
    print(
        f"Actual batched model calls: {total_model_calls}"
    )

    print(f"Saved: {out_path}")
    if evalplus_summary is not None:
        print(f"Saved scored outputs: {scored_path}")


if __name__ == "__main__":
    main()
