"""
Evaluation utilities for different dataset types.
Prompt formats and extraction match dllm's lm-evaluation-harness task configs exactly.
"""
import json
import re
import sys
import os

# Add reward/ to path for math_utils + domain_reward helpers used in DATASET_CONFIGS overrides.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "reward"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "reward"))

from domain_reward import (                          # noqa: E402
    extract_mmlu_pro_letter,                         # used by MMLU_Pro_sdar
    extract_humanevalx_python,                       # used by HumanEvalX_python_sdar
    extract_humaneval_sdar,                          # used by HumanEval_sdar
    extract_mc_oc,                                   # SDAR first_option_postprocess; used by ARC_C_sdar / MMLU_sdar
    # IFEval_sdar uses domain="ifeval_oc" → dispatched via CHECKERS["ifeval_oc"]
    # = check_ifeval_oc (SDAR vendored IFEvaluator). No direct import needed.
)


# ═══════════════════════════════════════════════════════════════════════
# Per-row prompt templates — move-from-rollouts pattern.
#
# `prompt_template` in DATASET_CONFIGS may be either:
#   - None → pass the question through as the chat-template body
#   - str  → standard `str.format(question=...)` (e.g. GSM8K, MATH500)
#   - callable(data_i, question) -> str  → per-row dispatch driven by
#     the row's own metadata (e.g. LMB-Hard mixes en+cn in one file;
#     MathBench mixes cloze/MC × cn/en).
#
# All three rollout scripts (sdar/bd3lm/qwen) detect callables and call
# them — so the prompt definitions and per-row logic live here, not in
# the rollouts.
# ═══════════════════════════════════════════════════════════════════════

# LiveMathBench (and LMB-Hard) — dispatched by language (subdivision suffix).
# These are SDAR's exact PROMPT_EN / PROMPT_CN from
# `evaluation/opencompass/opencompass/datasets/livemathbench/prompts.py`.
_LMB_PROMPT_EN = ("Here is a math question, please reasoning step by step, "
                  "and put your answer in \\boxed{{}}.\n{question}\n")
_LMB_PROMPT_CN = ("下面是一个数学问题，请逐步推理，并把最终答案放置于\\boxed{{}}中。"
                  "\n{question}\n")


def _lmb_template(data_i, question):
    """Pick PROMPT_EN/PROMPT_CN by row's `subdivision` suffix (`_en`/`_cn`)."""
    sub = data_i.get("subdivision", "")
    tmpl = _LMB_PROMPT_CN if sub.endswith("_cn") else _LMB_PROMPT_EN
    return tmpl.format(question=question)


# MMLU 5-shot — SDAR's `mmlu_gen_4d595a.py` builds the prompt as
#   <hint>\nQuestion: <q>\nA. {A}\nB. ...\nD. {D}\nAnswer: <gold>\n\n
#   ... (×5 dev examples from FixKRetriever[0:5]) ...
#   <hint>\nQuestion: <test_q>\nA. ...\nD. ...\nAnswer:
# The hint is per-subject. Dev examples live in `data/MMLU_sdar_5shot.json`
# (loaded once and cached). Each test row in MMLU_sdar.json carries
# `question, A, B, C, D, subject` so this builder runs at eval time.
_MMLU_HINT = ("There is a single choice question about {subject}. "
              "Answer the question by replying A, B, C or D.")
_MMLU_5SHOT_CACHE = None


def _get_mmlu_5shot():
    global _MMLU_5SHOT_CACHE
    if _MMLU_5SHOT_CACHE is None:
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "data", "MMLU_sdar_5shot.json")
        with open(path) as f:
            _MMLU_5SHOT_CACHE = json.load(f)
    return _MMLU_5SHOT_CACHE


# MBPP_sdar — SDAR's `sanitized_mbpp_mdblock_0shot_nocot_gen_a2e416.py` prompt.
# The test_list is INSIDE the prompt by design (gives the model 3 assert
# statements to satisfy). Per-row dispatch via this callable so the test
# assertions stay in the JSON `test_list` field, not baked into question.
_MBPP_SDAR_PROMPT = (
    "You are an expert Python programmer, and here is your task:\n"
    "{text}\n"
    "Your code should pass these tests:\n\n"
    "{test_list}\n"
    " You should submit your final solution in the following format: ```python\n\n```"
)


def _mbpp_sdar_template(data_i, question):
    test_list = "\n".join(data_i.get("test_list", []))
    return _MBPP_SDAR_PROMPT.format(text=question, test_list=test_list)


def _triviaqa_template(data_i, question):
    """SDAR triviaqa_wiki_1shot_gen_bc5f21 prompt as a multi-turn chat list.

    The 1-shot anchor (train[0]: "Where in England was Dame Judi Dench born?"
    / "York") is sent as a separate USER/ASSISTANT turn so the model treats
    the test question as the actual current question. Inlining both turns as
    text in a single user message caused the model to answer the example
    question instead — observed on BD3LM epoch-480 (acc 6.2%) and on
    SDAR-1.7B-Chat (37%). Multi-turn fixes both.
    """
    return [
        {"role": "user", "content": "Q: Where in England was Dame Judi Dench born?"},
        {"role": "assistant", "content": "A: York."},
        {"role": "user", "content": f"Q: {question}"},
    ]


def _mmlu_template(data_i, question):
    """SDAR mmlu_gen_4d595a 5-shot direct-letter prompt, per-subject hint.

    Returns a list of {role, content} chat messages so that OpenCompass-style
    in-context examples are sent as separate USER/ASSISTANT turns rather than
    inlined as text in a single user turn — this matches the way Qwen3 Chat
    saw few-shot data during instruction tuning. Earlier inline-text version
    underperformed by ~3pp; multi-turn closes most of that.

    The rollout's `_build_prompt_from_template` accepts either a string or a
    list of dicts and routes accordingly.
    """
    subj = data_i.get("subject", "")
    hint = _MMLU_HINT.format(subject=subj.replace("_", " "))
    messages = []
    for ex in _get_mmlu_5shot().get(subj, []):
        messages.append({
            "role": "user",
            "content": (
                f"{hint}\nQuestion: {ex['q']}\n"
                f"A. {ex['A']}\nB. {ex['B']}\nC. {ex['C']}\nD. {ex['D']}\nAnswer:"
            ),
        })
        messages.append({"role": "assistant", "content": ex["gold"]})
    messages.append({
        "role": "user",
        "content": (
            f"{hint}\nQuestion: {question}\n"
            f"A. {data_i.get('A','')}\nB. {data_i.get('B','')}\n"
            f"C. {data_i.get('C','')}\nD. {data_i.get('D','')}\nAnswer:"
        ),
    })
    return messages

# ═══════════════════════════════════════════════════════════════════════
# GSM8K: dllm/lm-evaluation-harness gsm8k-cot.yaml
# Prompt: "Q: {question}\n\nA:"
# Extraction: strict "The answer is X." then flexible numeric
# Normalization: remove commas, $, "#### " prefix, trailing period
# ═══════════════════════════════════════════════════════════════════════
_STRICT_RE = re.compile(r"The answer is (\-?[0-9\.\,]+)\.")
_FLEXIBLE_RE = re.compile(r"(-?[$0-9.,]{2,})|(-?[0-9]+)")
_IGNORE_PATTERNS = [re.compile(r","), re.compile(r"\$"), re.compile(r"(?s).*#### "), re.compile(r"\.$")]


def _normalize_gsm8k(s):
    for pat in _IGNORE_PATTERNS:
        s = pat.sub("", s)
    return s.lower().strip()


def extract_gsm8k_answer(text):
    # Filter 1: strict-match (first match)
    matches = _STRICT_RE.findall(text)
    if matches:
        return matches[0].strip()
    # Filter 2: flexible-extract (last match)
    matches = _FLEXIBLE_RE.findall(text)
    if matches:
        match = matches[-1]
        if isinstance(match, tuple):
            match = [m for m in match if m]
            if match:
                return match[0].strip()
        else:
            return match.strip()
    return "[invalid]"


def check_gsm8k(predicted, ground_truth):
    return _normalize_gsm8k(predicted) == _normalize_gsm8k(ground_truth)


# Note: math / MC extractors and checkers live in reward/domain_reward.py.
# eval_utils only retains dataset-specific helpers that DATASET_CONFIGS
# references directly (BBH, HellaSwag) below.


def check_mc(predicted, ground_truth):
    return predicted.strip().strip("()").upper() == ground_truth.strip().strip("()").upper()


# ═══════════════════════════════════════════════════════════════════════
# HellaSwag: dllm hellaswag_gen
# Prompt: "{ctx}\nQuestion: Which ending makes the most sense?\nA. ...\nAnswer:"
# Extraction: lowercase, regex [abcd]
# Note: our HellaSwag.json has pre-formatted "(A) ending" style, need to reformat
# ═══════════════════════════════════════════════════════════════════════
def extract_hellaswag_answer(text):
    """Match dllm's filter: lowercase, regex [abcd]."""
    text = text.strip().lower()
    m = re.search(r"[abcd]", text)
    if m:
        return m.group(0).upper()
    return "[invalid]"


# ═══════════════════════════════════════════════════════════════════════
# BBH: dllm bbh cot_fewshot
# Prompt: "Q: {input}\nA: Let's think step by step.\n" (with 3-shot CoT)
# Extraction: "So the answer is X."
# Note: 3-shot requires per-task few-shot examples, complex to reproduce
# ═══════════════════════════════════════════════════════════════════════
def extract_bbh_answer(text):
    m = re.search(r"[Ss]o the answer is[:\s]*(.+?)(?:\.|$)", text, re.IGNORECASE)
    if m:
        ans = m.group(1).strip().strip("()")
        return ans
    m = re.search(r"[Tt]he answer is[:\s]*(.+?)(?:\.|$)", text, re.IGNORECASE)
    if m:
        return m.group(1).strip().strip("()")
    lines = [l.strip() for l in text.strip().split("\n") if l.strip()]
    if lines:
        return lines[-1].strip().rstrip(".")
    return "[invalid]"


def check_bbh(predicted, ground_truth):
    p = predicted.strip().lower().strip("()")
    g = ground_truth.strip().lower().strip("()")
    return p == g


# ═══════════════════════════════════════════════════════════════════════
# LiveCodeBench prompt builder (v5 / v6 date-filtered windows)
#
# The `question` field in data/LCB_v{5,6}.json is the fully-rendered
# canonical LCB user prompt (Question + Format + Answer), baked by
# prepare_lcb_data.py. This builder only has to prepend the canonical
# SYSTEM_MESSAGE_GENERIC and run apply_chat_template.
#
# Compatible with BD3LM (block diffusion): the returned string is a
# normal chat-template string ending in `<|im_start|>assistant\n`, so
# the diffusion decoder generates into its output block as usual.
# ═══════════════════════════════════════════════════════════════════════
_LCB_SYSTEM_MESSAGE = (
    "You are an expert Python programmer. You will be given a question "
    "(problem specification) and will generate a correct Python program "
    "that matches the specification and passes all tests. You will NOT "
    "return anything except for the program."
)


def build_lcb_prompt(question_text, tokenizer, enable_thinking=False):
    """Return the canonical LCB chat prompt.

    `question_text` is the pre-baked user prompt (Question + Format +
    Answer). Chat template is applied with add_generation_prompt=True
    and enable_thinking=False (non-thinking mode, as used by Qwen3
    papers when reporting LCB numbers).
    """
    messages = [
        {"role": "system", "content": _LCB_SYSTEM_MESSAGE},
        {"role": "user", "content": question_text},
    ]
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
    except TypeError:
        # Older tokenizers without the enable_thinking kwarg.
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )


def extract_lcb_code(text, data_i=None):
    """LCB's canonical extractor: keep only the last ```...``` block."""
    # Lazy import so eval_utils stays importable without the lcb package.
    # `reward/` is not a python package (no __init__.py); callers add it
    # to sys.path as a top-level dir, so `lcb.xxx` is the portable form.
    _reward_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reward")
    if _reward_dir not in sys.path:
        sys.path.insert(0, _reward_dir)
    from lcb.extract_utils import extract_code_generation_v2
    return extract_code_generation_v2(text)


# ═══════════════════════════════════════════════════════════════════════
# Dataset configs
# ═══════════════════════════════════════════════════════════════════════
#
# Each entry carries a `domain` field ("math" / "mc" / "code") that drives
# the canonical extractor + checker in reward/domain_reward.py. Entries only
# need `extract` / `check` keys when they want to *override* the domain
# defaults (BBH short-answer string match, HellaSwag [abcd]); otherwise the
# domain dispatch handles everything.
#
# `prompt_template`:
#   - a format string with `{question}` → applied by the rollout scripts
#   - `None` → question is used as-is (prompt already encoded in the sample,
#     e.g. Hybrid_train where each question already has a domain-specific
#     prefix like "Solve the following science problem...").
DATASET_CONFIGS = {
    # ── math ──────────────────────────────────────────────────────────
    "GSM8K": {
        "path": "GSM8K.json",
        "domain": "math",
        "prompt_template": "{question}\nPlease reason step by step, and put your final answer within \\boxed{{}}.",
        "dllm_max_new_tokens": 256,
        "dllm_steps_per_block": 32,
    },
    "MATH500": {
        "path": "MATH500.json",
        "domain": "math",
        "prompt_template": "{question}\nPlease reason step by step, and put your final answer within \\boxed{{}}.",
        "dllm_max_new_tokens": 256,
        "dllm_steps_per_block": 32,
    },
    "AIME2024": {
        "path": "AIME2024.json",
        "domain": "math",
        "prompt_template": "{question}\nPlease reason step by step, and put your final answer within \\boxed{{}}.",
        "dllm_max_new_tokens": 256,
        "dllm_steps_per_block": 32,
    },
    "AIME2025": {
        "path": "AIME2025.json",
        "domain": "math",
        "prompt_template": "{question}\nPlease reason step by step, and put your final answer within \\boxed{{}}.",
        "dllm_max_new_tokens": 256,
        "dllm_steps_per_block": 32,
    },
    # LMB-Hard (SDAR's actual reported split). Two variants:
    #   LiveMathBench_Hard       — en + cn merged (45 rows; matches SDAR's
    #     pipeline: their loader does `product(['hard'],['cn','en'])` → one
    #     flat dataset → one np.mean accuracy. `_lmb_template` picks
    #     PROMPT_EN or PROMPT_CN per-row by subdivision suffix.
    #   LiveMathBench_Hard_en    — en subset only (21 rows; static PROMPT_EN).
    #     Useful when you want EN-only without CN's count weighting.
    # Both files come from `data/process_lmb_hard.py` (gated HF download).
    # NOTE: SDAR's published number averages over 32 stochastic runs — see
    # `data/readme.md` "LMB-Hard reproduction caveats" before reporting.
    "LiveMathBench_Hard": {
        "path": "LiveMathBench_Hard.json",
        "domain": "math",
        "prompt_template": _lmb_template,   # per-row dispatch by subdivision
        "dllm_max_new_tokens": 256,
        "dllm_steps_per_block": 32,
    },
    # "LiveMathBench_Hard_en": {
    #     "path": "LiveMathBench_Hard_en.json",
    #     "domain": "math",
    #     "prompt_template": _LMB_PROMPT_EN,  # static; en only
    #     "dllm_max_new_tokens": 256,
    #     "dllm_steps_per_block": 32,
    # },
    "MATH_hendrycks": {
        "path": "__debug_datasets__/hendrycks_math_test.json",
        "domain": "math",
        "prompt_template": "{question}\nPlease reason step by step, and put your final answer within \\boxed{{}}.",
        "dllm_max_new_tokens": 256,
        "dllm_steps_per_block": 32,
    },
    "GSM8K_train": {
        "path": "GSM8K_train.json",
        "domain": "math",
        "prompt_template": "{question}\nPlease reason step by step, and put your final answer within \\boxed{{}}.",
        "dllm_max_new_tokens": 256,
        "dllm_steps_per_block": 32,
    },
    "DAPO_Math_17k": {
        "path": "DAPO_Math_17k.json",
        "domain": "math",
        "prompt_template": "{question}\nPlease reason step by step, and put your final answer within \\boxed{{}}.",
        "dllm_max_new_tokens": 256,
        "dllm_steps_per_block": 32,
    },
    "DAPO_Math_17k_1k": {
        "path": "DAPO_Math_17k_1k.json",
        "domain": "math",
        "prompt_template": "{question}\nPlease reason step by step, and put your final answer within \\boxed{{}}.",
        "dllm_max_new_tokens": 256,
        "dllm_steps_per_block": 32,
    },
    "MATH_train": {
        "path": "MATH_train.json",
        "domain": "math",
        "prompt_template": "{question}\nPlease reason step by step, and put your final answer within \\boxed{{}}.",
        "dllm_max_new_tokens": 256,
        "dllm_steps_per_block": 32,
    },
    "MATH_train_traceRL": {
        "path": "MATH_train_traceRL.json",
        "domain": "math",
        "prompt_template": "{question}\nPlease reason step by step, and put your final answer within \\boxed{{}}.",
        "dllm_max_new_tokens": 256,
        "dllm_steps_per_block": 32,
    },
    # ── multiple choice ───────────────────────────────────────────────
    # "GPQA_Diamond": {
    #     "path": "GPQA_Diamond.json",
    #     "domain": "mc",
    #     "prompt_template": "{question}\nPlease show your choice in the answer field with only the choice letter, e.g., \"answer\": \"C\".",
    #     "reformat_choices": True,  # (A) -> A.
    #     "dllm_max_new_tokens": 3,
    #     "dllm_steps_per_block": 3,
    # },
    # "GPQA_Main": {
    #     "path": "GPQA_Main.json",
    #     "domain": "mc",
    #     "prompt_template": "{question}\nPlease show your choice in the answer field with only the choice letter, e.g., \"answer\": \"C\".",
    #     "reformat_choices": True,  # (A) -> A.
    #     "dllm_max_new_tokens": 3,
    #     "dllm_steps_per_block": 3,
    # },
    "GPQA_Diamond_shuffled": {
        "path": "GPQA_Diamond_shuffled.json",
        "domain": "mc",
        "prompt_template": "{question}\nPlease show your choice in the answer field with only the choice letter, e.g., \"answer\": \"C\".",
        "reformat_choices": True,
        "dllm_max_new_tokens": 3,
        "dllm_steps_per_block": 3,
    },
    "GPQA_Main_shuffled": {
        "path": "GPQA_Main_shuffled.json",
        "domain": "mc",
        "prompt_template": "{question}\nPlease show your choice in the answer field with only the choice letter, e.g., \"answer\": \"C\".",
        "reformat_choices": True,
        "dllm_max_new_tokens": 3,
        "dllm_steps_per_block": 3,
    },
    "BBH": {
        "path": "BBH.json",
        "domain": "mc",
        # BBH uses short free-form answers with custom string equality, so
        # override the domain defaults with the legacy helpers.
        "extract": extract_bbh_answer,
        "check": check_bbh,
        "prompt_template": "{question}",
        "dllm_max_new_tokens": 256,
        "dllm_steps_per_block": 32,
    },
    "BBH_3shot": {
        "path": "BBH.json",
        "domain": "mc",
        "extract": extract_bbh_answer,
        "check": check_bbh,
        "prompt_template": "{question}",
        "fewshot": 3,
        "fewshot_file": "bbh_fewshot_examples.json",
        "dllm_max_new_tokens": 256,
        "dllm_steps_per_block": 32,
    },
    "MMLU": {
        "path": "MMLU.json",
        "domain": "mc",
        # Our data has "(A) choice" format, need to reformat to "A. choice"
        "prompt_template": "{question}\nPlease show your choice in the answer field with only the choice letter, e.g., \"answer\": \"C\".",
        "reformat_choices": True,  # Signal to reformat (A) -> A.
        "dllm_max_new_tokens": 3,
        "dllm_steps_per_block": 3,
    },
    "MMLU_Redux": {
        "path": "MMLU_Redux.json",
        "domain": "mc",
        # MMLU_Redux: cleaned subset of MMLU. Our JSON already stores choices
        # as "A. choice" (not "(A) choice"), so reformat_choices=False.
        "prompt_template": "{question}\nPlease show your choice in the answer field with only the choice letter, e.g., \"answer\": \"C\".",
        "reformat_choices": False,
        "dllm_max_new_tokens": 3,
        "dllm_steps_per_block": 3,
    },
    # C-Eval (Chinese MMLU): 1,346 val rows across 52 subjects, 4-way MC.
    # Standard direct-letter MC; C-Eval paper's recommended protocol is greedy
    # 5-shot direct (no CoT). Our JSON stores choices already inlined as
    # "A. text\nB. text\n..." — reformat_choices=False. Direct-letter
    # answer; small max_tokens budget. Subject-level subdivision strings live
    # in `subdivision: "ceval_<subject>"` for any future per-subject reporting.
    "CEval": {
        "path": "CEval.json",
        "domain": "mc",
        "prompt_template": "{question}\nPlease show your choice in the answer field with only the choice letter, e.g., \"answer\": \"C\".",
        "reformat_choices": False,
        "dllm_max_new_tokens": 3,
        "dllm_steps_per_block": 3,
    },
    "MMLU_Pro": {
        "path": "MMLU_Pro.json",
        "domain": "mc",
        "prompt_template": "{question}\nPlease show your choice in the answer field with only the choice letter, e.g., \"answer\": \"C\".",
        "reformat_choices": True,
        "dllm_max_new_tokens": 3,
        "dllm_steps_per_block": 3,
    },
    "HellaSwag": {
        "path": "HellaSwag.json",
        "domain": "mc",
        # HellaSwag uses a custom [abcd] extractor (generations are 3 tokens
        # long with no "Answer:" preamble), so override the mc default.
        "extract": extract_hellaswag_answer,
        "check": check_mc,
        "prompt_template": "{question}",
        "reformat_choices": True,
        "dllm_max_new_tokens": 3,
        "dllm_steps_per_block": 3,
    },
    # ── code ──────────────────────────────────────────────────────────
    # HumanEval / MBPP use the EvalPlus framework's prompts and scoring:
    #   - Data files regenerated by prepare_evalplus_data.py
    #     (164 HumanEval+ tasks, 378 MBPP+ tasks with evalplus task_ids).
    #   - `question` field = evalplus `prompt` (signature+docstring for
    #     HumanEval, problem description for MBPP).
    #   - Prompt builder (`build_evalplus_prompt`) wraps the question in
    #     evalplus's instruction_prefix + response_prefix with an
    #     assistant-prefill ending in ```python, matching
    #     evalplus.provider.utility.make_raw_chat_prompt but with
    #     enable_thinking=False for Qwen3-derived chat templates.
    #   - Scoring is DEFERRED: reward/rl_execute.py detects
    #     defer_scoring=="evalplus" and shells out to `evalplus.evaluate`
    #     on the rollout's {task_id, solution} jsonl. That gives us the
    #     canonical HumanEval+/MBPP+ base and plus pass@1 numbers.
    "HumanEval": {
        "path": "HumanEval.json",
        "domain": "code",
        "chat_style": "evalplus_prefill",
        "defer_scoring": "evalplus",
        "evalplus_dataset": "humaneval",
        "dllm_max_new_tokens": 1024,
        "dllm_steps_per_block": 32,
    },
    "MBPP": {
        "path": "MBPP.json",
        "domain": "code",
        "chat_style": "evalplus_prefill",
        "defer_scoring": "evalplus",
        "evalplus_dataset": "mbpp",
        "dllm_max_new_tokens": 1024,
        "dllm_steps_per_block": 32,
    },
    # # ───── SDAR-spec code variants (separate entries; see data/readme.md) ─────
    # # HumanEval_sdar — same 164 task IDs as our HumanEval entry; SDAR's prompt
    # # is a plain instruction (no evalplus assistant-prefill). Reuses
    # # HumanEval.json. Scoring: defer_scoring="evalplus" gives us the canonical
    # # base pass@1 (which is what SDAR's HumanEvalEvaluator computes).
    # # Source: humaneval_openai_sample_evals_gen_dcae0e.py
    # "HumanEval_sdar": {
    #     "path": "HumanEval.json",
    #     "domain": "code",
    #     "prompt_template": (
    #         "Read the following function signature and docstring, and fully "
    #         "implement the function described. Your response should only "
    #         "contain the code for this function.\n{question}"
    #     ),
    #     # Custom extractor prepends the prompt (imports + signature + docstring)
    #     # so evalplus.sanitize sees a self-contained program. Without this,
    #     # SDAR-style chat responses often omit `from typing import List` and
    #     # fail at test-time with NameError. Mirrors what SDAR's
    #     # `human_eval.evaluate_functional_correctness` does internally.
    #     "extract": extract_humaneval_sdar,
    #     "defer_scoring": "evalplus",
    #     "evalplus_dataset": "humaneval",
    #     "dllm_max_new_tokens": 1024,
    #     "dllm_steps_per_block": 32,
    # },
    # # MBPP_sdar — sanitized MBPP (427 tasks; NOT MBPP+ 378). Different data,
    # # different prompt: SDAR includes 3 assert statements from `test_list`
    # # inside the prompt itself (gives the model the spec). Callable
    # # `_mbpp_sdar_template` assembles the test_list at eval time.
    # # Scoring uses the existing function-eval path (`evaluate_function_dataset`
    # # in reward/rl_execute.py): each row's `test_list` (3 asserts) is exec'd
    # # against the model's extracted code in a sandboxed subprocess.
    # # Source: sanitized_mbpp_mdblock_0shot_nocot_gen_a2e416.py
    # "MBPP_sdar": {
    #     "path": "MBPP_sdar.json",
    #     "domain": "code",
    #     "prompt_template": _mbpp_sdar_template,
    #     "dllm_max_new_tokens": 512,           # SDAR uses max_out_len=512
    #     "dllm_steps_per_block": 32,
    # },
    # # HumanEvalX_python_sdar — Python subset of CodeGeeX2/HumanEvalX (164 tasks,
    # # task_id "Python/0".."Python/163"; same problems as HumanEval but a
    # # different schema). SDAR's prompt is just `{question}` verbatim — pure
    # # signature+docstring with no instruction. Multi-language (cpp/go/java/js)
    # # is NOT included; SDAR scores those via a Docker code-eval server which
    # # we don't set up here.
    # # Custom extractor `extract_humanevalx_python`: prepends the signature
    # # to the model's body when needed (the model often emits only the body
    # # since the prompt is bare signature+docstring with no instruction).
    # # Scoring via the function-eval path: each row's `test_list` is
    # # `[<test_block>\ncheck(<entry_point>)]` (built by process_code_sdar.py).
    # # Source: humanevalx_gen_620cfa.py (python subset only)
    # "HumanEvalX_python_sdar": {
    #     "path": "HumanEvalX_python_sdar.json",
    #     "domain": "code",
    #     "prompt_template": "{question}",
    #     "extract": extract_humanevalx_python,
    #     "dllm_max_new_tokens": 1024,
    #     "dllm_steps_per_block": 32,
    # },
    # LiveCodeBench v5 / v6 (date-filtered windows; see prepare_lcb_data.py):
    #   - chat_style "lcb": system message (SYSTEM_MESSAGE_GENERIC) + pre-baked
    #     canonical user prompt. No assistant prefill; model generates the full
    #     ```python ...``` block. Uses enable_thinking=False to match Qwen3
    #     tech-report non-thinking LCB numbers.
    #   - defer_scoring "livecodebench": rl_execute dispatches to
    #     evaluate_lcb_dataset, which runs codegen_metrics over the pre-stored
    #     input_output tests (public + private) using reward.lcb.
    #   - extract override: extract_code_generation_v2 (keep last code block).
    "LCB_v5": {
        "path": "LCB_v5.json",
        "domain": "code",
        "chat_style": "lcb",
        "defer_scoring": "livecodebench",
        "extract": extract_lcb_code,
        "dllm_max_new_tokens": 4096,
        "dllm_steps_per_block": 32,
    },
    "LCB_v6": {
        "path": "LCB_v6.json",
        "domain": "code",
        "chat_style": "lcb",
        "defer_scoring": "livecodebench",
        "extract": extract_lcb_code,
        "dllm_max_new_tokens": 4096,
        "dllm_steps_per_block": 32,
    },
    # # LCB_v6_sdar — same data + same chat_style ("lcb" → SYSTEM_MESSAGE_GENERIC
    # # which is byte-identical to SDAR's), so it's effectively an alias of
    # # LCB_v6. The only thing that distinguishes "SDAR-style" is sampling:
    # # SDAR runs `n=6` (6 samples per question, computes pass@k). Set this in
    # # the runner script via `--num_response_per_task 6`, not here.
    # # Source: livecodebench_v6_academic.py
    # "LCB_v6_sdar": {
    #     "path": "LCB_v6.json",
    #     "domain": "code",
    #     "chat_style": "lcb",
    #     "defer_scoring": "livecodebench",
    #     "extract": extract_lcb_code,
    #     "dllm_max_new_tokens": 4096,
    #     "dllm_steps_per_block": 32,
    # },
    # PrimeIntellect: Gen-Verse competitive-programming problems, 100% stdio.
    # Prompt template matches dLLM-RL's trado stdio template verbatim
    # (sample/trado_rl_rollout.py:346), so the KL distillation target matches
    # the distribution the dLLM-RL teacher was trained under.
    # Scoring is skipped via dataset.skip_code_correctness=True in rl_bd3lm.yaml;
    # test_input/test_output fields are ignored at training time.
    "PrimeIntellect": {
        "path": "PrimeIntellect.json",
        "domain": "code",
        "prompt_template": (
            "This is the problem:\n{question}\n"
            "You should put your code in ```python ```. "
            "Use input() to read input and print() to produce output in your script. "
        ),
        "dllm_max_new_tokens": 2048,
        "dllm_steps_per_block": 32,
    },
    # Gen-Verse LiveCodeBench / LiveBench (stdio-only cuts, competitive
    # programming style). Schema: question / test_input / test_output /
    # test_time_limit / test_method. All samples are test_method="stdio".
    # Prompt template matches trado_rl_rollout.py:346 so evaluation matches
    # the distribution dLLM-RL reports numbers on.
    # NOTE: This is NOT the same dataset as LCB_v5/LCB_v6 (which are the
    # official livecodebench/code_generation_lite release with both stdio
    # and function-call problems, scored via `defer_scoring: "livecodebench"`).
    "LiveCodeBench": {
        "path": "LiveCodeBench.json",
        "domain": "code",
        "prompt_template": (
            "This is the problem:\n{question}\n"
            "You should put your code in ```python ```. "
            "Use input() to read input and print() to produce output in your script. "
        ),
        "dllm_max_new_tokens": 2048,
        "dllm_steps_per_block": 32,
    },
    # NOTE: the active "LiveBench" entry (domain="livebench", multi-category)
    # is defined ~300 lines below. An earlier code-domain stub was removed.
    # ── instruction following ─────────────────────────────────────────
    "IFEval": {
        "path": "IFEval.json",
        "domain": "ifeval",
        "prompt_template": None,          # raw prompt, no wrapping
        "dllm_max_new_tokens": 1280,
        "dllm_steps_per_block": 32,
    },
    # ───────────────────────────────────────────────────────────────────
    # SDAR-spec knowledge benchmarks (mirrors evaluation/opencompass/
    # configs/datasets/<bench>/<canonical_config>.py).
    # ───────────────────────────────────────────────────────────────────
    # ARC-c — two prompt variants, same data + same scoring (`extract_mc` +
    # letter-compare). Both read data/ARC_C.json (1,144 rows; question stem
    # with options inlined as "A. text" — `reformat_choices=False`).
    #
    # ARC_C_sdar  : SDAR's `ARC_c_cot_gen_926652.py` — 0-shot CoT,
    #               "ANSWER: $LETTER" output. Long generation budget.
    # ARC_C       : Qwen-style direct-letter prompt (matches the existing
    #               MMLU/MMLU_Redux entries). Short generation budget.
    # "ARC_C_sdar": {
    #     "path": "ARC_C.json",
    #     "domain": "mc",
    #     "extract": extract_mc_oc,            # SDAR's first_option_postprocess('ABCD')
    #     "prompt_template": (
    #         "Answer the following multiple choice question. The last line of "
    #         "your response should be of the following format: "
    #         "'ANSWER: $LETTER' (without quotes) where LETTER is one of ABCD. "
    #         "Think step by step before answering.\n\n{question}"
    #     ),
    #     "dllm_max_new_tokens": 2048,
    #     "dllm_steps_per_block": 32,
    # },
    "ARC_C": {
        "path": "ARC_C.json",
        "domain": "mc",
        "prompt_template": "{question}\nPlease show your choice in the answer field with only the choice letter, e.g., \"answer\": \"C\".",
        "dllm_max_new_tokens": 3,
        "dllm_steps_per_block": 3,
    },
    # # MMLU 5-shot — direct-letter (no CoT). JSON rows store the raw question +
    # # A,B,C,D + subject; `_mmlu_template` (above) loads the per-subject 5-shot
    # # dev examples from `data/MMLU_sdar_5shot.json` and assembles the SDAR
    # # prompt at eval time. Source: mmlu_gen_4d595a.py
    # "MMLU_sdar": {
    #     "path": "MMLU_sdar.json",
    #     "domain": "mc",
    #     "extract": extract_mc_oc,            # SDAR's first_option_postprocess('ABCD')
    #     "prompt_template": _mmlu_template,   # returns list[dict] → multi-turn 5-shot
    #     "dllm_max_new_tokens": 16,           # direct-letter; small budget is enough
    #     "dllm_steps_per_block": 4,
    # },
    # # MMLU-Pro 0-shot CoT — A-P (up to 16 options). Custom regex extractor.
    # # Source: mmlu_pro_0shot_cot_gen_08c1de.py
    # "MMLU_Pro_sdar": {
    #     "path": "MMLU_Pro_sdar.json",
    #     "domain": "mc",
    #     "extract": extract_mmlu_pro_letter,   # SDAR's match_answer_pattern A-P
    #     "prompt_template": (
    #         "Answer the following multiple choice question. The last line of "
    #         "your response should be of the following format: "
    #         "'ANSWER: $LETTER' (without quotes) where LETTER is one of "
    #         "Options(e.g. one of ABCDEFGHIJKLMNOP). Think step by step "
    #         "before answering.\n\nQuestion:\n{question}"
    #     ),
    #     "dllm_max_new_tokens": 2048,
    #     "dllm_steps_per_block": 32,
    # },
    # # GPQA-Diamond 0-shot — deterministic option-shuffle, "(A)..(D).." format.
    # # Source: gpqa_gen_4baadb.py
    # "GPQA_Diamond_sdar": {
    #     "path": "GPQA_Diamond_sdar.json",
    #     "domain": "mc",
    #     "prompt_template": (
    #         "What is the correct answer to this question: {question}\n"
    #         "Format your response as follows: "
    #         "\"The correct answer is (insert answer here)\""
    #     ),
    #     "dllm_max_new_tokens": 2048,
    #     "dllm_steps_per_block": 32,
    # },
    # # IFEval — verbatim prompt (instruction is embedded), IFEvaluator scoring.
    # # Source: IFEval_gen_353ae7.py.  (Same data as the existing IFEval entry;
    # # _sdar variant exists only so file naming is consistent with the others.)
    # "IFEval_sdar": {
    #     "path": "IFEval_sdar.json",
    #     # `ifeval_oc` is a separate domain → CHECKERS["ifeval_oc"] = check_ifeval_oc.
    #     # A/B-tested against `check_ifeval` (lm_eval) on saved 1.7B rollouts:
    #     # scores agreed within 1 row (187 vs 188 / 541). The swap is for
    #     # source-of-truth alignment (SDAR's vendored copy + drops the
    #     # lm_eval/langdetect dependency), not for the 9pp gap to SDAR's
    #     # reported 43.4 — that residual is inference-engine drift
    #     # (JetEngine vs HF), not scorer drift.
    #     "domain": "ifeval_oc",
    #     "prompt_template": None,
    #     "dllm_max_new_tokens": 1280,
    #     "dllm_steps_per_block": 32,
    # },
    # TriviaQA — 1-shot wiki Q→A, short answer. SDAR uses the wiki version
    # (TriviaQADatasetV2 reading triviaqa-{train,validation}.jsonl) with
    # FixKRetriever(fix_id_list=[0]) → the 1-shot anchor is always train[0]:
    #     Q: "Where in England was Dame Judi Dench born?" / A: "York"
    # Validation split (7993 rows) is the test set. Gold is a list of accepted
    # aliases. Custom domain triggers extract_triviaqa + check_triviaqa
    # (mirrors OpenCompass TriviaQAEvaluator: normalize + substring match).
    # Source: triviaqa_wiki_1shot_gen_bc5f21.py
    #
    # Two entries share data/TriviaQA.json:
    #   TriviaQA       — multi-turn callable; the 1-shot example is a separate
    #                    USER/ASSISTANT turn so the test question is unambiguous.
    #                    Use this for any new run.
    #   TriviaQA_sdar  — string template that inlines both Q/A pairs in one
    #                    user message. This matches the literal text SDAR's
    #                    OpenCompass would assemble before the chat template,
    #                    BUT chat-template-wrapping turns it into a single
    #                    user turn that confuses the model into answering the
    #                    example. Kept for backward-compat / SDAR-fidelity
    #                    comparison.
    "TriviaQA": {
        "path": "TriviaQA.json",
        "domain": "triviaqa",
        "prompt_template": _triviaqa_template,    # multi-turn list[dict]
        "dllm_max_new_tokens": 64,                # SDAR uses max_out_len=50
        "dllm_steps_per_block": 8,
    },
    # "TriviaQA_sdar": {
    #     "path": "TriviaQA.json",
    #     "domain": "triviaqa",
    #     "prompt_template": (
    #         "Q: Where in England was Dame Judi Dench born?\n"
    #         "A: York.\n"
    #         "Q: {question}\n"
    #         "A: "
    #     ),
    #     "dllm_max_new_tokens": 64,
    #     "dllm_steps_per_block": 8,
    # },
    # ── MLogiQA (multilingual LogiQA, 10 langs × 80 = 800 rows) ───────
    # Source: swiss-ai/mlogiqa. Cited by Qwen3 in Multilingual Tasks.
    # 4-way MC; questions are LogiQA-style logical-reasoning puzzles
    # (context + question + 4 options). Original answer is integer 0-3;
    # processor converts to A/B/C/D for the standard `mc` domain.
    # Per-language reporting via subdivision="mlogiqa_<lang>".
    "MLogiQA": {
        "path": "MLogiQA.json",
        "domain": "mc",
        "prompt_template": "{question}\nPlease show your choice in the answer field with only the choice letter, e.g., \"answer\": \"C\".",
        "reformat_choices": False,    # JSON already has "A. text\nB. ..." inlined
        # MC with logical reasoning — small budget for direct letter, larger
        # if thinking is enabled. Use 8000 to give thinking models room.
        "dllm_max_new_tokens": 8000,
        "dllm_steps_per_block": 32,
    },
    # ── INCLUDE-Lite (44 languages, native regional knowledge MC) ─────
    # Source: CohereLabs/include-lite-44. Paper: "INCLUDE: Evaluating
    # Multilingual Language Understanding with Regional Knowledge"
    # (Romanou et al., 2024). Cited by Qwen3 / Llama-3 / Aya / Gemini.
    #
    # Built from NATIVE regional exams in each country (university
    # entrance, professional licensing, civil service, driving) — not
    # MMLU translations. Tests cultural / civic / domain knowledge
    # specific to each language's region. ~250 rows × 44 langs ≈ 10.7k.
    #
    # 4-way MC, integer answer 0-3 in source → converted to letter A/B/C/D
    # by the processor. Routes through `mc` domain (no new scorer).
    # Per-language reporting via subdivision="include_<lang>".
    #
    # Note: Dutch-Flemish config in source has a stub but no parquet
    # (404); we use plain "Dutch" instead. German has only 89 rows
    # (upstream data issue, not ours).
    "INCLUDE_Lite": {
        "path": "INCLUDE_Lite.json",
        "domain": "mc",
        "prompt_template": "{question}\nPlease show your choice in the answer field with only the choice letter, e.g., \"answer\": \"C\".",
        "reformat_choices": False,    # JSON has options pre-inlined
        "dllm_max_new_tokens": 8000,
        "dllm_steps_per_block": 32,
    },
    # ── MMMLU-Lite (multilingual MMLU sample, 17 langs × 400 = 6800) ──
    # Source: CohereForAI/Global-MMLU-Lite. (No `openai/MMMLU-Lite` exists;
    # Cohere's Global-MMLU-Lite is the de-facto multilingual MMLU-Lite used
    # by Qwen3 / Llama-3 / Gemini cards.) Stratified sample of 400 MMLU
    # questions per language across 57 subjects (STEM / Humanities / Social
    # Sciences / Business / Medical / Other).
    #
    # 4-way MC, answer already letter A/B/C/D in the source. Routes through
    # the standard `mc` domain. Per-language reporting via
    # subdivision="mmmlu_<lang>".
    "MMMLU_Lite": {
        "path": "MMMLU_Lite.json",
        "domain": "mc",
        "prompt_template": "{question}\nPlease show your choice in the answer field with only the choice letter, e.g., \"answer\": \"C\".",
        "reformat_choices": False,    # JSON already has "A. text\nB. ..." inlined
        # MC factual recall — short budget for direct letter, larger if
        # thinking is enabled. 8000 matches MLogiQA / CEval.
        "dllm_max_new_tokens": 8000,
        "dllm_steps_per_block": 32,
    },
    # ── PolyMath (multilingual math, 18 langs × 4 levels × 125 = 9000) ─
    # Source: Qwen/PolyMath. Paper: "PolyMath: Evaluating Mathematical
    # Reasoning in Multilingual Contexts." All 18 languages share the SAME
    # 500 base questions translated (125 per level: low/medium/high/top).
    # Answers are LaTeX expressions (e.g. "$\\frac{\\pi}{3}$").
    #
    # Scoring uses our existing math/math_verify pipeline. The questions
    # don't ship with format instructions, so we wrap with the standard
    # boxed-answer prompt template.
    #
    # Per-(level, lang) reporting via subdivision="polymath_<level>_<lang>".
    "PolyMath": {
        "path": "PolyMath.json",
        "domain": "math",
        "prompt_template": "{question}\nPlease reason step by step, and put your final answer within \\boxed{{}}.",
        # Top-level problems can need extensive thinking; budget conservatively.
        "dllm_max_new_tokens": 16000,
        "dllm_steps_per_block": 32,
    },
    # ── MT-AIME2024 (multilingual AIME, 55 languages × 30 questions) ──
    # Source: Tobi2K/MT_AIME2024_transposed, the canonical set from
    # "Linguistic Generalizability of Test-Time Scaling in Mathematical
    # Reasoning". All 55 languages share the SAME 30 AIME-2024 problems
    # (parallel translations), so per-language acc measures cross-lingual
    # transfer of identical math reasoning. Total 1,650 rows.
    #
    # Scoring reuses the existing math/AIME pipeline (numeric 0-999 answer,
    # `\boxed{N}` extraction, exact-match). Per-language reporting via the
    # `subdivision="mt_aime_<lang>"` field.
    "MT_AIME2024": {
        "path": "MT_AIME2024.json",
        "domain": "math",
        # Standard AIME prompt: ask for boxed integer answer.
        "prompt_template": "{question}\nPlease reason step by step, and put your final answer within \\boxed{{}}.",
        # AIME problems often need 8-16K thinking budget for thinking models.
        "dllm_max_new_tokens": 16000,
        "dllm_steps_per_block": 32,
    },
    # ── AutoLogi (English split, 1575 logical-reasoning puzzles) ──────
    # qzhu/AutoLogi (en split). Each puzzle has constraints; the model must
    # output a Python dict satisfying them. Scoring runs the dataset-shipped
    # `Inputs_Check_code(inputs)` Python function in a sandboxed subprocess
    # against the model's parsed dict — programmatic verification, not
    # exact-match. Any valid solution scores 1.0 (puzzles often have many).
    #
    # Cited by Qwen3 in their "Math & Text Reasoning" task list. Strong
    # benchmark for thinking — multi-step constraint propagation required.
    "AutoLogi": {
        "path": "AutoLogi.json",
        "domain": "autologi",
        "prompt_template": None,        # prompt is self-contained in `question`
        # Puzzles need extensive thinking + a structured dict output;
        # 4-8K is comfortable for thinking-on, 2K for thinking-off.
        "dllm_max_new_tokens": 8192,
        "dllm_steps_per_block": 32,
    },
    # ── ZebraLogic (logic puzzles, 1000 grids, ZeroEval canonical set) ─
    # Each row's `question` already contains the ZeroEval JSON-output
    # instruction, so prompt_template=None (pass-through). Domain
    # "zebralogic" → CHECKERS["zebralogic"] = check_zebralogic which parses
    # the ```json fenced output and compares cell-by-cell against the
    # ground-truth grid (puzzle-level exact-match).
    #
    # Reference Qwen3-4B-Thinking score on this set: ~35% (puzzle accuracy).
    # Frontier (R1/o1-mini): 40-60%. Non-thinking small models: typically <10%.
    # Source: WildEval/ZebraLogic/grid_mode (1000 puzzles, balanced 25 sizes
    # 2x2..6x6, 40 each); paper Lin et al. 2025 (arxiv:2502.01100).
    "ZebraLogic": {
        "path": "ZebraLogic.json",
        "domain": "zebralogic",
        "prompt_template": None,
        # Larger grids (5x6, 6x6) need extensive thinking; budget conservatively.
        # Qwen3-4B-Thinking generates 4-8K tokens per puzzle on average.
        "dllm_max_new_tokens": 8192,
        "dllm_steps_per_block": 32,
    },
    # ── LiveBench (multi-domain, 6 categories, 1436 questions) ────────
    # Latest active set on HF (livebench/{math,reasoning,data_analysis,
    # coding,instruction_following,language}) is the 2024-11-25 release.
    # Each row's `question` is self-contained (the prompt already includes
    # the format instruction, e.g. "put your final answer in \boxed{}").
    # So prompt_template=None is correct — pass through verbatim.
    #
    # Domain "livebench" routes through CHECKERS["livebench"] = check_livebench
    # which dispatches per (livebench_category, livebench_task) to the right
    # scorer in livebench.process_results.<cat>.<task>.utils.
    #
    # Per-category breakdown is reported by rl_reward.py (math/reasoning/
    # data_analysis/coding/instruction_following/language).
    "LiveBench": {
        "path": "LiveBench.json",
        "domain": "livebench",
        "prompt_template": None,
        # Conservative budget: math olympiad rows can need up to ~3k tokens
        # of CoT; AMPS_Hard typically <500; coding completions <2k. 4096 is
        # enough for everything except long olympiad-style proofs.
        "dllm_max_new_tokens": 4096,
        "dllm_steps_per_block": 32,
    },
    # ── hybrid (per-sample domain mix) ────────────────────────────────
    # Hybrid_train samples store the BARE problem in `question` (math/MC/code
    # have been normalized by scripts/rewrite_hybrid_train.py). The actual
    # prompt wrapper is picked per-sample from `per_domain_template` below,
    # using `data_i["domain"]`. For code, the front-loaded instruction
    # (stdio vs `Implement a function named X`) is already baked into the
    # question by the rewrite — so code uses a pass-through template.
    "Hybrid_train": {
        "path": "Hybrid_train_new.json",
        "domain": None,                 # per-sample, resolved from data_i["domain"]
        "prompt_template": None,        # legacy global key — dispatch via per_domain_template
        "per_domain_template": {
            "math":    "{question}\nPlease reason step by step, and put your final answer within \\boxed{{}}.",
            "science": "{question}\nPlease show your choice in the answer field with only the choice letter, e.g., \"answer\": \"C\".",
            "code":    "{question}",    # instruction baked in (stdio or named-fn at front)
            "chat":    "{question}",    # free-form, no wrapper
        },
    },
    "Hybrid_train_new": {
        "path": "Hybrid_train_new.json",
        "domain": None,                 # per-sample, resolved from data_i["domain"]
        "prompt_template": None,        # legacy global key — dispatch via per_domain_template
        "per_domain_template": {
            "math":    "{question}\nPlease reason step by step, and put your final answer within \\boxed{{}}.",
            "science": "{question}\nPlease show your choice in the answer field with only the choice letter, e.g., \"answer\": \"C\".",
            "code":    "{question}",    # instruction baked in (stdio or named-fn at front)
            "chat":    "{question}",    # free-form, no wrapper
        },
    },
    "opdlm_train": {
        "path": "opdlm_train.json",
        "domain": None,                 # per-sample, resolved from data_i["domain"]
        "prompt_template": None,        # legacy global key — dispatch via per_domain_template
        "per_domain_template": {
            "math":    "{question}\nPlease reason step by step, and put your final answer within \\boxed{{}}.",
            "science": "{question}\nPlease show your choice in the answer field with only the choice letter, e.g., \"answer\": \"C\".",
            "code":    "{question}",    # instruction baked in (stdio or named-fn at front)
            "chat":    "{question}",    # free-form, no wrapper
        },
    },
    # ── legacy SFT mixture (no verifiable answers) ────────────────────
    # Plain passthrough, correctness check is skipped via
    # dataset.skip_correctness in the rl_bd3lm_sft_mixture.yaml config.
    "BD3LM_sft_mixture": {
        "path": "BD3LM_sft_mixture.json",
        "domain": "chat",
        "prompt_template": None,
    },
}


def reformat_choices(question_text):
    """Convert '(A) choice' format to 'A. choice' format to match dllm."""
    # Replace (A) -> A., (B) -> B., etc.
    text = re.sub(r"\(([A-J])\)\s*", r"\1. ", question_text)
    return text


# ═══════════════════════════════════════════════════════════════════════
# EvalPlus prompt builder
#
# Replicates `evalplus.provider.utility.make_raw_chat_prompt` (the
# instruction_prefix / response_prefix + assistant-prefill wrapper that
# evalplus.codegen uses for chat models) but passes enable_thinking=False
# so Qwen3-derived templates don't emit a <think> block.
#
# Unlike the other prompt builders in this file, this one REPLACES
# `_build_prompt_from_template` and produces a fully-rendered chat string
# ending inside a ```python fence — the model continues generation there.
# ═══════════════════════════════════════════════════════════════════════
_EVALPLUS_MAGIC = "-[[]]-this-is-really-our-highest-priority-[[]]-"
_EVALPLUS_INSTRUCTION_PREFIX = (
    "Please provide a self-contained Python script that solves the "
    "following problem in a markdown code block:"
)
_EVALPLUS_RESPONSE_PREFIX = (
    "Below is a Python script with a self-contained function that solves "
    "the problem and passes corresponding tests:"
)


def build_evalplus_prompt(question_text, tokenizer):
    """Return an evalplus-style chat prompt that ends inside a ```python
    fence, ready for the model to continue. Mirrors evalplus's codegen
    path so training-time eval produces the same prompt EvalPlus uses for
    its leaderboard numbers."""
    if tokenizer.chat_template is None:
        return question_text
    user_content = (
        f"{_EVALPLUS_INSTRUCTION_PREFIX}\n```\n{question_text.strip()}\n```\n"
    )
    response = f"{_EVALPLUS_RESPONSE_PREFIX}\n```python\n{_EVALPLUS_MAGIC}\n```\n"
    try:
        rendered = tokenizer.apply_chat_template(
            [
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": response},
            ],
            tokenize=False,
            enable_thinking=False,
        )
    except TypeError:
        # Older tokenizers without enable_thinking kwarg.
        rendered = tokenizer.apply_chat_template(
            [
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": response},
            ],
            tokenize=False,
        )
    return rendered.split(_EVALPLUS_MAGIC)[0]


def build_bbh_fewshot_prompt(question, subtask, fewshot_examples):
    """Build a BBH prompt with 3-shot CoT examples prepended.

    Format matches dllm: each example is "Q: {input}\nA: Let's think step by step.\n{target}\n\n"
    followed by the actual question "Q: {question}\nA: Let's think step by step.\n"
    """
    parts = []
    examples = fewshot_examples.get(subtask, [])
    for ex in examples[:3]:
        parts.append(f"Q: {ex['input']}\nA: Let's think step by step.\n{ex['target']}\n\n")
    parts.append(f"Q: {question}\nA: Let's think step by step.\n")
    return "".join(parts)
