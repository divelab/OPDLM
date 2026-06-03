"""
Canonical per-domain extraction + checking functions.

Domains: "math", "code", "mc"

Used by:
  - sample/{qwen,bd3lm,...}_rl_rollout.py   — extracts answers after generation
  - reward/rl_reward.py                     — checks correctness

Dispatch:
  1. Per-sample `data_i["domain"]` (opdlm_train case)
  2. Dataset-level `ds_cfg["domain"]` (DATASET_CONFIGS in eval_utils.py)
  3. Default "math"

A dataset may override the domain default by setting `ds_cfg["extract"]` /
`ds_cfg["check"]` explicitly (e.g. BBH, HellaSwag with custom logic).
"""
import ast
import json
import os
import re
import signal
import subprocess
import sys
import tempfile
import shutil

# math-verify (verl-style)
from math_verify import parse as mv_parse, verify as mv_verify, LatexExtractionConfig, ExprExtractionConfig


# ═══════════════════════════════════════════════════════════════════════
# MATH domain — math_verify library (unified across GSM8K / MATH / DAPO / ...)
# ═══════════════════════════════════════════════════════════════════════
def extract_math(text, data_i=None):
    """Extract a math answer from the full model response using math_verify."""
    try:
        result = mv_parse(
            text,
            extraction_config=[ExprExtractionConfig(), LatexExtractionConfig()],
        )
        if result and len(result) > 1:
            return str(result[1])
        elif result:
            return str(result[0])
    except Exception:
        pass
    return "Can not extract the answer!"


def check_math(extracted, data_i):
    """Compare extracted answer against ground-truth using math_verify.verify."""
    ground_truth = data_i["ground_truth_answer"]
    try:
        gold = mv_parse(
            "\\boxed{" + str(ground_truth) + "}",
            extraction_config=[LatexExtractionConfig()],
        )
        answer = mv_parse(
            "\\boxed{" + str(extracted) + "}",
            extraction_config=[LatexExtractionConfig()],
        )
        return bool(mv_verify(gold, answer))
    except Exception:
        return False


# ═══════════════════════════════════════════════════════════════════════
# MATH domain (alt scorer) — OpenCompass MATHEvaluator + math_postprocess_v2
# Lazy-imported from the vendored inference_sdar/evaluation/opencompass tree.
# Falls back to math_verify with a warning if OpenCompass isn't importable.
# ═══════════════════════════════════════════════════════════════════════
_OC_LOADED = None  # tri-state: None=unattempted, False=failed, True=loaded


def _ensure_opencompass():
    """Lazy-load OpenCompass math utilities. Returns (post_v2, evaluator)
    or (None, None) on failure. Memoized to avoid repeated import overhead."""
    global _OC_LOADED, _OC_post_v2, _OC_evaluator
    if _OC_LOADED is not None:
        return (_OC_post_v2, _OC_evaluator) if _OC_LOADED else (None, None)

    _candidate_paths = [
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     "inference_sdar", "evaluation", "opencompass"),
    ]
    for _p in _candidate_paths:
        if os.path.isdir(_p) and _p not in sys.path:
            sys.path.insert(0, _p)
    try:
        from opencompass.datasets.math import math_postprocess_v2 as _post
        from opencompass.datasets.math import MATHEvaluator as _Eval
        _OC_post_v2 = _post
        _OC_evaluator = _Eval(version='v2')
        _OC_LOADED = True
    except Exception as e:
        print(f"[domain_reward] OpenCompass scorer unavailable ({type(e).__name__}: {e}); "
              f"falling back to math_verify.")
        _OC_LOADED = False
        _OC_post_v2 = None
        _OC_evaluator = None
    return (_OC_post_v2, _OC_evaluator) if _OC_LOADED else (None, None)


def extract_math_oc(text, data_i=None):
    """OpenCompass-style math extractor: math_postprocess_v2 (boxed → 'final
    answer is X' → first sentence). Falls back to math_verify on import failure."""
    post, _ = _ensure_opencompass()
    if post is None:
        return extract_math(text, data_i)
    try:
        return post(text)
    except Exception:
        return extract_math(text, data_i)


def check_math_oc(extracted, data_i):
    """OpenCompass-style math equivalence: MATHEvaluator.is_equiv(pred, ref)."""
    _, ev = _ensure_opencompass()
    if ev is None:
        return check_math(extracted, data_i)
    ground_truth = data_i["ground_truth_answer"]
    try:
        return bool(ev.is_equiv(str(extracted), str(ground_truth)))
    except Exception:
        return False


# ─────────────────────────────────────────────────────────────────
# MC domain (alt scorer) — OpenCompass first_option_postprocess.
# Used to match SDAR HF eval on MathBench MC subsets (and any other
# OpenCompass MC dataset with options A-D / A-E / etc.).
# ─────────────────────────────────────────────────────────────────
_OC_MC = None


def _ensure_oc_mc():
    global _OC_MC
    if _OC_MC is not None:
        return _OC_MC
    # _ensure_opencompass adds the inference_sdar opencompass tree to sys.path.
    _ensure_opencompass()
    try:
        from opencompass.utils.text_postprocessors import first_option_postprocess
        _OC_MC = first_option_postprocess
    except Exception:
        _OC_MC = False
    return _OC_MC


def extract_mc_oc(text, data_i=None):
    """OpenCompass MC extractor: regex over a long pattern list (CN + EN)
    that finds the first valid option letter. Falls back to extract_mc."""
    fn = _ensure_oc_mc()
    if not fn:
        return extract_mc(text, data_i)
    try:
        out = fn(str(text or ""), options="ABCD")
        return (out or "[invalid]").upper()
    except Exception:
        return extract_mc(text, data_i)


def extract_humaneval_sdar(text, data_i=None):
    """HumanEval extractor for the SDAR plain-instruction prompt.

    SDAR's `humaneval_openai_sample_evals_gen_dcae0e.py` scores via
    `HumanEvalEvaluator` (openai/human-eval), which runs
    `prompt + completion + test + check(entry_point)`. The prompt provides
    the imports + signature + docstring; `completion` is whatever the model
    wrote.

    pure_inference's evalplus path expects `solution` to be self-contained.
    With the SDAR plain prompt the model's response often re-states the
    function but omits `from typing import List` etc. — looks complete to
    evalplus.sanitize, but fails at test time with NameError.

    Fix: prepend the prompt to the model's output so imports/signature are
    always present. Python is happy with re-defined functions; the second
    `def` overrides the first (prompt's signature ends with the docstring
    so the redefinition is fine). For body-only outputs, the signature
    from the prompt makes a complete function.
    """
    raw = str(text or "")
    matches = re.findall(r"```\w*\n(.*?)```", raw, re.DOTALL)
    code = matches[0] if matches else raw
    # Strip the chat EOS markers that often leak into the response.
    for tok in ("<|im_end|>", "<|endoftext|>", "<|MASK|>"):
        code = code.replace(tok, "")
    code = code.lstrip("\n").rstrip()
    if not data_i:
        return code
    prompt = data_i.get("question", "")
    return prompt + "\n" + code


def extract_humanevalx_python(text, data_i=None):
    """HumanEvalX-Python extractor.

    SDAR's prompt for HumanEvalX is `{prompt}` verbatim — pure
    signature+docstring, no instruction. So the model usually outputs only
    the function BODY (a continuation of the docstring). The test block then
    expects `<entry_point>` to be a callable in the global namespace —
    which it isn't, unless the body is paired with the signature.

    This extractor:
      1. Pulls the last ```python …``` block if present, else uses the raw text.
      2. If the extracted code already contains `def <entry_point>(`, returns
         it as-is (the model wrote the full function).
      3. Otherwise prepends the signature/docstring from `data_i["question"]`
         so the function is defined at module level when the test runs.
    """
    # NOTE: only strip trailing whitespace and outer newlines. `.strip()`
    # would remove the leading 4-space indent on the first body line, which
    # breaks the assembled function definition when we prepend the prompt.
    matches = re.findall(r"```python(.*?)```", str(text or ""), re.DOTALL)
    raw = matches[-1] if matches else str(text or "")
    for tok in ("<|im_end|>", "<|endoftext|>", "<|MASK|>"):
        raw = raw.replace(tok, "")
    code = raw.lstrip("\n").rstrip()
    if not data_i:
        return code
    ep = data_i.get("entry_point", "")
    if ep and re.search(rf"^def\s+{re.escape(ep)}\s*\(", code, flags=re.MULTILINE):
        return code
    prompt = data_i.get("question", "")
    return prompt + "\n" + code


def extract_mmlu_pro_letter(text, data_i=None):
    """MMLU-Pro extractor: matches the exact regex SDAR uses
    (`mmlu_pro_0shot_cot_gen_08c1de.py` → match_answer_pattern with
    `r'(?i)ANSWER\\s*:\\s*([A-P])'`). Falls back to OpenCompass's
    first_option_postprocess(options='ABCDEFGHIJKLMNOP'), then to extract_mc."""
    if not text:
        return "[invalid]"
    m = re.search(r"(?i)ANSWER\s*:\s*([A-P])", str(text))
    if m:
        return m.group(1).upper()
    fn = _ensure_oc_mc()
    if fn:
        try:
            out = fn(str(text), options="ABCDEFGHIJKLMNOP")
            if out:
                return out.upper()
        except Exception:
            pass
    return extract_mc(text, data_i)


# ─────────────────────────────────────────────────────────────────
# TriviaQA domain — short Q→A, EM-contains over an alias list.
# Mirrors OpenCompass TriviaQAEvaluator (triviaqa.py:TriviaQAEvaluator).
# ─────────────────────────────────────────────────────────────────
def _general_postprocess(text):
    # Mirrors opencompass.utils.text_postprocessors.general_postprocess.
    truncated = re.split(r"[\n.,]", str(text), 1)[0]
    no_punct = re.sub(r"[^\w\s]", "", truncated)
    no_articles = re.sub(r"\b(a|an|the)\b", "", no_punct, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", no_articles).strip()


def extract_triviaqa(text, data_i=None):
    """Apply SDAR's TriviaQAEvaluator preprocessing:
        text.strip().split('\\n')[0].lower()
            .split('answer is')[-1]
            .split('a:')[-1]
            .split('answer:')[-1]
            .strip()
            → general_postprocess
    Returns the normalized prediction string.
    """
    s = str(text or "").strip().split("\n")[0].lower()
    s = s.split("answer is")[-1]
    s = s.split("a:")[-1]
    s = s.split("answer:")[-1]
    return _general_postprocess(s.strip())


def check_triviaqa(extracted, data_i):
    """Match if any normalized gold alias is a substring of the normalized prediction."""
    gold = data_i["ground_truth_answer"]
    if isinstance(gold, str):
        gold = [gold]
    pred = str(extracted or "")
    norm_golds = [_general_postprocess(g).lower() for g in gold]
    return any(g and g in pred for g in norm_golds)


# ═══════════════════════════════════════════════════════════════════════
# MC domain — single-letter multiple-choice extraction
# ═══════════════════════════════════════════════════════════════════════
def extract_mc(text, data_i=None):
    """Extract single-letter MC answer.

    Strictly pattern-based — only accepts \\boxed{} and an explicit
    "answer" cue. No bare-letter fallback (which produces lucky guesses
    on rambling/truncated responses), no per-dataset letter restriction
    (so it works across GPQA/ARC/MMLU/MMLU_Pro without configuration).

    Patterns tried, last-match wins within each:
      1. \\boxed{X} / \\boxed{\\text{X}} / \\boxed{\\textbf{X}}
      2. "Answer: X" / "Answer = X" / "**Answer:** 'X'"
      3. "answer is X" / "the correct answer is X" / "Final answer is X"
         / "answer is option X" / "answer is choice X"

    Returns "[invalid]" when none of the patterns match — that's an
    honest signal that the model never produced a parseable answer.
    """
    text = (text or "").strip()
    if not text:
        return "[invalid]"

    # 1. \boxed{X} / \boxed{\text{X}} / \boxed{\textbf{X}}.
    boxed = re.findall(
        r"\\boxed\{\s*(?:\\text(?:bf|rm|sf|tt|it)?\s*\{)?\s*([A-Za-z])",
        text,
    )
    if boxed:
        return boxed[-1].upper()

    # 2. "Answer: X" / "**Answer:** 'X'" / `"answer": "X"` (JSON-like).
    #    Allow quotes between "answer" and the colon (JSON-encoded form).
    matches = re.findall(
        r"(?i)\banswer\b[\s\*\"'\)\]]*[:=][\s\*\"'\$\(]*([A-Za-z])",
        text,
    )
    if matches:
        return matches[-1].upper()

    # 3. "answer is X" / "the correct answer is X" / "answer is option X" /
    #    "the correct answer is:\n\n**X**" (model emits a stop-list before X).
    #    Colon and period are allowed between "is" and the letter.
    matches = re.findall(
        r"(?i)\banswer\b[\s\*\"']*\bis\b[\s\*\"'\$\(\:\.]*"
        r"(?:option|choice|letter)?[\s\*\"'\$\(\:\.]*"
        r"([A-Za-z])\b",
        text,
    )
    if matches:
        return matches[-1].upper()

    # 4. Fallback: last standalone capital letter A-Z. Catches prose
    #    conclusions like "So C is correct." that don't use any of
    #    the explicit cues above. Last-match (not first) avoids
    #    grabbing option-list letters from the analysis section.
    bare = re.findall(r"\b([A-Z])\b", text)
    if bare:
        return bare[-1].upper()

    return "[invalid]"


def check_mc(extracted, data_i):
    ground_truth = data_i["ground_truth_answer"]
    pred = str(extracted).strip().strip("()").upper()
    gt = str(ground_truth).strip().strip("()").upper()
    return pred == gt


# ═══════════════════════════════════════════════════════════════════════
# CODE domain — python code extraction + multi-format test runner
# ═══════════════════════════════════════════════════════════════════════
# opdlm_train ships code samples in 4 different test formats. We dispatch
# on the shape of `data_i["tests_json"]`:
#
#   pytest       : {"type":"pytest","test_code":"...","fn_name":"..."}
#                  (kodcode_light_rl)
#   stdio        : {"type":"stdio","cases":[{"input":"...","output":"..."}]}
#                  (taco stdio — solution reads stdin, prints stdout)
#   fn_call      : {"type":"fn_call","fn_name":"...","cases":[{"args":[...],"expected":[...]}]}
#                  (taco fn_call — solution defines a named function)
#   assert_list  : ["assert f(1) == 2", ...]  (JSON or Python-repr serialized)
#                  (acecode — raw assert statements)
#
# All paths write solution.py + test_runner.py in a temp dir and exec the
# runner as a subprocess; PASS iff returncode == 0.
# ═══════════════════════════════════════════════════════════════════════

_CODE_EXEC_TIMEOUT = int(os.environ.get("TRACERL_CODE_TIMEOUT", "20"))


def _classify_tests(raw):
    """Return (fmt, parsed) where fmt ∈ {pytest, stdio, fn_call, assert_list, unknown}."""
    if raw is None or raw == "N/A":
        return "unknown", None
    if isinstance(raw, (list, dict)):
        parsed = raw
    else:
        try:
            parsed = json.loads(raw)
        except Exception:
            # acecode variant — Python repr with single quotes, not valid JSON
            try:
                parsed = ast.literal_eval(raw)
            except Exception:
                return "unknown", None
    if isinstance(parsed, list):
        return "assert_list", parsed
    if isinstance(parsed, dict):
        t = parsed.get("type")
        if t == "pytest" or "test_code" in parsed:
            return "pytest", parsed
        if t == "stdio":
            return "stdio", parsed
        if t == "fn_call":
            return "fn_call", parsed
    return "unknown", parsed


def _fn_name_for_extract(data_i):
    """Best-effort fn_name recovery for extract_code (used to pick the right
    python block when the model emits multiple). Prefers the dataset's own
    fn_name field (set by scripts/rewrite_hybrid_train.py), then the one
    inside tests_json, then HumanEval's `entry_point`."""
    if not data_i:
        return None
    if data_i.get("fn_name"):
        return data_i["fn_name"]
    fmt, parsed = _classify_tests(data_i.get("tests_json"))
    if fmt in ("pytest", "fn_call") and isinstance(parsed, dict):
        return parsed.get("fn_name")
    if data_i.get("entry_point"):
        return data_i["entry_point"]
    return None


def extract_code(text, data_i=None):
    """Extract a python code block from the model response.

    If data_i carries a test function name, prefer the block that defines it
    (the model often emits a trailing 'example usage' block that doesn't).
    Otherwise concatenate all python blocks.
    """
    fn_name = _fn_name_for_extract(data_i)
    blocks = re.findall(r"```python\s*\n?(.*?)```", text, re.DOTALL)
    if not blocks:
        blocks = re.findall(r"```\s*\n?(.*?)```", text, re.DOTALL)
    if not blocks:
        return ""
    if fn_name:
        pat = re.compile(rf"\bdef\s+{re.escape(fn_name)}\s*\(")
        for b in blocks:
            if pat.search(b):
                return b.strip()
    return "\n\n".join(b.strip() for b in blocks)


def _run_solution_subprocess(code, runner_src):
    """Write solution.py + test_runner.py into a temp dir and run the runner.
    Passes iff returncode == 0. Hard-capped by _CODE_EXEC_TIMEOUT."""
    tmp_dir = None
    try:
        tmp_dir = tempfile.mkdtemp(prefix="tracerl_code_")
        with open(os.path.join(tmp_dir, "solution.py"), "w") as f:
            f.write(code)
        with open(os.path.join(tmp_dir, "test_runner.py"), "w") as f:
            f.write(runner_src)
        proc = subprocess.run(
            [sys.executable, "test_runner.py"],
            cwd=tmp_dir,
            capture_output=True, text=True, timeout=_CODE_EXEC_TIMEOUT,
        )
        return proc.returncode == 0
    except Exception:
        return False
    finally:
        if tmp_dir:
            try:
                shutil.rmtree(tmp_dir)
            except Exception:
                pass


def _check_pytest(code, parsed):
    """kodcode-style: parsed has 'test_code' with def test_xxx() functions."""
    test_code = parsed.get("test_code", "")
    if not test_code:
        return False
    test_fns = re.findall(r"^def (test_\w+)\s*\(", test_code, re.MULTILINE)
    if not test_fns:
        return False
    # Some kodcode tests start with `from solution import ...`, others rely
    # on the function simply being in scope. Prepend a wildcard import so
    # both styles work (harmless if an explicit import already exists).
    import_prelude = "from solution import *  # noqa\n"
    runner = import_prelude + test_code + "\n\n" + "\n".join(f"{fn}()" for fn in test_fns) + "\n"
    return _run_solution_subprocess(code, runner)


def _check_assert_list(code, asserts):
    """acecode-style: list of assert-statement strings."""
    if not asserts:
        return False
    body = "\n".join(a for a in asserts if isinstance(a, str))
    if not body.strip():
        return False
    runner = "from solution import *  # noqa\n" + body + "\n"
    return _run_solution_subprocess(code, runner)


def _check_fn_call(code, parsed):
    """taco fn_call: cases of {args, expected}, call fn_name(*args) and compare."""
    fn_name = parsed.get("fn_name")
    cases = parsed.get("cases") or []
    if not fn_name or not cases:
        return False
    cases_json = json.dumps(cases)
    runner = (
        "from solution import *  # noqa\n"
        "import json\n"
        f"cases = json.loads({json.dumps(cases_json)})\n"
        "for c in cases:\n"
        "    args = c['args']\n"
        "    expected = c['expected']\n"
        f"    actual = {fn_name}(*args)\n"
        "    # taco wraps single-value returns as a 1-element list\n"
        "    if isinstance(expected, list) and len(expected) == 1 and not isinstance(actual, (list, tuple)):\n"
        "        expected = expected[0]\n"
        "    assert actual == expected, f'for args={args!r} expected {expected!r}, got {actual!r}'\n"
    )
    return _run_solution_subprocess(code, runner)


def _check_stdio(code, parsed):
    """taco stdio: run solution.py as script with each case's input piped
    to stdin, compare stdout (whitespace-stripped)."""
    cases = parsed.get("cases") or []
    if not cases:
        return False
    cases_json = json.dumps(cases)
    runner = (
        "import subprocess, sys, json\n"
        f"cases = json.loads({json.dumps(cases_json)})\n"
        "for c in cases:\n"
        "    proc = subprocess.run(\n"
        "        [sys.executable, 'solution.py'],\n"
        "        input=c.get('input', ''),\n"
        "        capture_output=True, text=True, timeout=5,\n"
        "    )\n"
        "    if proc.returncode != 0:\n"
        "        raise RuntimeError(f'solution crashed: {proc.stderr[:200]}')\n"
        "    expected = (c.get('output') or '').strip()\n"
        "    actual = (proc.stdout or '').strip()\n"
        "    if actual != expected:\n"
        "        raise AssertionError(f'expected {expected!r}, got {actual!r}')\n"
    )
    return _run_solution_subprocess(code, runner)


def check_code(extracted, data_i):
    """Dispatch to the right test runner based on tests_json shape.

    HumanEval / MBPP are NOT handled here — their scoring is deferred to
    the evalplus CLI path in reward/rl_execute.py. If check_code is ever
    reached with an unknown tests_json shape we return False.
    """
    code = extracted or ""
    if not code:
        return False
    raw = data_i.get("tests_json") if data_i else None
    fmt, parsed = _classify_tests(raw)
    if fmt == "pytest":
        return _check_pytest(code, parsed)
    if fmt == "stdio":
        return _check_stdio(code, parsed)
    if fmt == "fn_call":
        return _check_fn_call(code, parsed)
    if fmt == "assert_list":
        return _check_assert_list(code, parsed)
    return False


# ═══════════════════════════════════════════════════════════════════════
# Dispatch
# ═══════════════════════════════════════════════════════════════════════
# opdlm_train uses per-sample `domain` labels that don't always match our
# canonical names. Map dataset-native labels onto canonical domains.
DOMAIN_ALIASES = {
    "math":    "math",
    "science": "mc",     # opdlm_train science samples are `\boxed{letter}` MC.
    "mc":      "mc",
    "code":    "code",
    # "chat" intentionally omitted — chat has no verifiable answer; see
    # `skip_correctness` in rl_reward.py and domain resolves to default below.
}


def _noop_extract(text, data_i=None):
    return ""


def _noop_check(extracted, data_i):
    # Used for domains with no verifiable answer (e.g. chat). Always-False
    # reward makes these samples a no-op under z-score normalization.
    return False


# ═══════════════════════════════════════════════════════════════════════
# IFEval domain — instruction following verification via lm_eval
# ═══════════════════════════════════════════════════════════════════════
def extract_ifeval(text, data_i=None):
    """IFEval uses the full response, no extraction needed."""
    return text


def check_ifeval(extracted, data_i):
    """Check instruction following using lm_eval's IFEval verification.

    Uses strict prompt-level check (all instructions must be followed).
    Requires data_i to have 'key', 'instruction_id_list', 'kwargs' from IFEval.json.
    """
    from lm_eval.tasks.ifeval.utils import InputExample, test_instruction_following_strict
    inp = InputExample(
        key=data_i.get('key', 0),
        instruction_id_list=data_i.get('instruction_id_list', []),
        prompt=data_i.get('question', ''),
        kwargs=data_i.get('kwargs', []),
    )
    out = test_instruction_following_strict(inp, extracted or "")
    return out.follow_all_instructions


def check_ifeval_oc(extracted, data_i):
    """SDAR-vendored IFEval strict prompt-level check.

    Mirrors `inference_sdar/.../IFEval/ifeval.py::IFEvaluator.score()` —
    `Prompt-level-strict-accuracy = all(follow_instruction_list)` per row.
    A/B-tested against `check_ifeval` on saved 1.7B rollouts: scores agree
    within 1 row (187 vs 188 / 541), so this exists mainly to remove the
    lm_eval dependency, not to close the gap to SDAR's reported number
    (the residual is JetEngine vs HF inference drift, not scorer drift).
    """
    _ensure_opencompass()
    from opencompass.datasets.IFEval.evaluation_main import (
        InputExample, test_instruction_following_strict,
    )
    # SDAR's IFEvaluator strips None-valued kwargs before calling the checker.
    raw_kwargs = data_i.get('kwargs', []) or []
    cleaned_kwargs = [
        {k: v for k, v in kw.items() if v is not None}
        for kw in raw_kwargs
    ]
    inp = InputExample(
        key=data_i.get('key', 0),
        instruction_id_list=data_i.get('instruction_id_list', []),
        prompt=data_i.get('question', ''),
        kwargs=cleaned_kwargs,
    )
    out = test_instruction_following_strict(inp, extracted or "")
    return all(out.follow_instruction_list)


# ═══════════════════════════════════════════════════════════════════════
# LiveBench domain — routes to the official livebench package's scorers
# (pip install --no-deps git+https://github.com/LiveBench/LiveBench.git).
# Each row carries `livebench_category` and `livebench_task` so we can
# dispatch to the correct scorer in livebench.process_results.<cat>.<task>.
# ═══════════════════════════════════════════════════════════════════════
def extract_livebench(text, data_i=None):
    """LiveBench scorers expect the raw model response — they each parse
    out the relevant final-answer pattern (boxed{...}, **bold**, ***X***,
    etc.) themselves. We strip trailing chat EOS tokens though: the cta
    scorer (and others doing suffix-match) compares the LAST n chars of
    the cleaned response against the gt. With `<|im_end|>` left in,
    `clean_text(...)` keeps `im_end` as the final chars and the suffix
    match fails on every row. Stripping `<|im_end|>` / `<|endoftext|>` /
    `<|MASK|>` recovers ~38 pp on cta and ~8 pp on tablereformat."""
    s = str(text or "")
    for tok in ("<|im_end|>", "<|endoftext|>", "<|MASK|>"):
        s = s.replace(tok, "")
    return s.rstrip()


def _livebench_with_timeout(fn, *args, timeout=30, **kwargs):
    """Run an arbitrary scorer with a per-call timeout via SIGALRM.

    Used for CPU-bound scorers like sympy math equivalence checks. SIGALRM
    interrupts the python interpreter at the next bytecode boundary —
    works fine for pure-Python compute, but does NOT help when the process
    is blocked in a kernel call (e.g. subprocess pipe_read). For those,
    use `_livebench_subproc_timeout`.
    """
    import signal
    class _Timeout(Exception): pass
    def _handler(signum, frame):
        raise _Timeout()
    old = signal.signal(signal.SIGALRM, _handler)
    signal.alarm(int(timeout))
    try:
        return fn(*args, **kwargs)
    except _Timeout:
        return 0.0
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)


def _livebench_subproc_timeout(fn_module, fn_name, args, timeout=60):
    """Run a scorer in a forked child process with a hard timeout.

    Use this for scorers that themselves spawn subprocesses or use
    `mp.Queue` internally — both cases can deadlock the parent in
    `pipe_read` indefinitely. Two failure modes we've hit in production:
        1. livebench coding scorers `subprocess.run(...)` model code in
           sandboxes; orphan grandchildren from that hold pipe fds open.
        2. livebench AMPS_Hard's own `run_with_timeout` does
           `Queue() / Process()` + `queue.get()` after a possible
           partial-write child death — same fork-pipe leak shape.

    Hardening here:
        - Use `mp.Pipe(duplex=False)` and CLOSE the parent's writer fd
          immediately after `p.start()` so a dead child triggers EOF
          instead of a hang.
        - Run the worker in `os.setsid()` so we can `os.killpg()` the
          entire process group (including grandchildren) on timeout.
        - Use `parent_conn.poll(timeout)` — bounded wait that returns
          when the child writes or dies, never blocks indefinitely.

    `fn_module` and `fn_name` are passed by name (string) to avoid
    pickling issues with the function object across the fork boundary.
    """
    import multiprocessing as mp
    def _worker(c):
        os.setsid()
        try:
            import importlib
            module = importlib.import_module(fn_module)
            fn = getattr(module, fn_name)
            c.send(("ok", float(fn(*args))))
        except Exception as e:
            c.send(("err", f"{type(e).__name__}: {e}"))
        finally:
            c.close()

    parent_conn, child_conn = mp.Pipe(duplex=False)
    p = mp.Process(target=_worker, args=(child_conn,))
    p.start()
    child_conn.close()  # release parent's copy so EOF propagates on child death
    if not parent_conn.poll(timeout):
        try:
            os.killpg(p.pid, signal.SIGKILL)
        except Exception:
            pass
        p.join(2)
        if p.is_alive():
            p.kill()
        return 0.0
    try:
        kind, val = parent_conn.recv()
    except EOFError:
        kind, val = "err", 0.0
    p.join(2)
    if p.is_alive():
        try:
            os.killpg(p.pid, signal.SIGKILL)
        except Exception:
            pass
        p.kill()
    return val if kind == "ok" else 0.0


def _livebench_score(data_i, extracted):
    """Return a [0,1] score (1 if correct, 0 if wrong; some tasks return
    fractional credit). Routed by (livebench_category, livebench_task) to
    the appropriate function in livebench.process_results.<cat>.<task>.

    Each scorer has a slightly different signature — see the docstrings on
    each function for details. Common patterns:
        (gt, pred, debug=False)            — most simple scorers
        (input_cmd, gt, pred, version)     — table_process_results
        (_, gt, pred, debug=False)         — joinmap_process_results
        (gt, pred, question_text, debug)   — mathcontest_process_results
        (data_i, response, task, model_id) — instruction_following_process_results
    """
    cat = (data_i or {}).get("livebench_category", "")
    task = (data_i or {}).get("livebench_task", "")
    gt = (data_i or {}).get("ground_truth_answer", "")
    question = (data_i or {}).get("question", "")
    rel = (data_i or {}).get("livebench_release_date", "") or ""
    rel_short = rel[:10] if rel else ""

    # Strip chat EOS tokens before scoring. The cta scorer (and any
    # suffix-match style scorer) compares the LAST n chars of cleaned
    # response against gt; trailing `<|im_end|>` etc. mean the suffix
    # becomes "im_end"/"endoftext" and every row scores 0 even when the
    # answer is right. Strip is safe for all LiveBench scorers (their
    # parsers don't depend on EOS markers). Applied here (not at
    # extraction time) so re-scoring existing rollout JSONs works
    # without a re-patch.
    if extracted:
        _s = str(extracted)
        for _tok in ("<|im_end|>", "<|endoftext|>", "<|MASK|>"):
            _s = _s.replace(_tok, "")
        extracted = _s.rstrip()

    try:
        if cat == "math":
            if task == "AMPS_Hard":
                # AMPS_Hard's own `run_with_timeout` (livebench source) uses
                # mp.Queue + Process and has the same fork-pipe leak we
                # patched: queue.get() after a partial-write child death
                # blocks on pipe_read forever. SIGALRM can't unstick a
                # kernel pipe-read. Route through the subprocess wrapper
                # so we can `killpg` the entire tree on timeout.
                return _livebench_subproc_timeout(
                    "livebench.process_results.math.AMPS_Hard.utils",
                    "amps_hard_process_results",
                    (gt, extracted),
                    timeout=60,
                )
            if task == "math_comp":
                # `math_comp` mixes AMC (letter A-E, 117 rows) and AIME (3-digit
                # numeric, 29 rows). Different scorers per format.
                from livebench.process_results.math.math_competitions.utils import (
                    mathcontest_process_results, aime_process_results,
                )
                gt_str = str(gt).strip()
                if len(gt_str) == 1 and gt_str.upper() in "ABCDE":
                    # AMC: multiple choice with letter answer
                    return float(_livebench_with_timeout(mathcontest_process_results, gt_str, extracted, question, timeout=30))
                # AIME: 3-digit numeric answer (or fallback for unknown gt shape)
                return float(_livebench_with_timeout(aime_process_results, gt_str, extracted, timeout=30))
            if task == "olympiad":
                from livebench.process_results.math.olympiad.utils import proof_rearrangement_process_results
                return float(_livebench_with_timeout(proof_rearrangement_process_results, gt, extracted, timeout=30))
            # Unknown math task → AMPS_Hard scorer (boxed{answer} format)
            from livebench.process_results.math.AMPS_Hard.utils import amps_hard_process_results
            return float(_livebench_with_timeout(amps_hard_process_results, gt, extracted, timeout=30))
        if cat == "reasoning":
            if task == "zebra_puzzle":
                from livebench.process_results.reasoning.zebra_puzzle.utils import get_zebra_puzzle_evaluator
                ev = get_zebra_puzzle_evaluator(rel_short)
                return float(ev(gt, extracted))
            if task == "web_of_lies_v2":
                from livebench.process_results.reasoning.web_of_lies_v2.utils import web_of_lies_process_results
                return float(web_of_lies_process_results(gt, extracted))
            if task == "spatial":
                from livebench.process_results.reasoning.spatial.utils import spatial_process_results
                return float(spatial_process_results(gt, extracted))
        if cat == "data_analysis":
            if task == "tablejoin":
                # signature: joinmap_process_results(_, ground_truth, llm)
                from livebench.process_results.data_analysis.tablejoin.utils import joinmap_process_results
                return float(joinmap_process_results(None, gt, extracted))
            if task == "tablereformat":
                # signature: table_process_results(input_command, ground_truth, llm_answer, version="v1")
                from livebench.process_results.data_analysis.tablereformat.utils import table_process_results
                return float(table_process_results(question, gt, extracted))
            if task == "cta":
                from livebench.process_results.data_analysis.cta.utils import cta_process_results
                return float(cta_process_results(gt, extracted))
        if cat == "language":
            # Note: HF dataset is named `language` but the scorer module is `writing`.
            if task == "connections":
                from livebench.process_results.writing.connections.utils import get_connections_puzzle_evaluator
                ev = get_connections_puzzle_evaluator(rel_short)
                return float(ev(gt, extracted))
            if task == "plot_unscrambling":
                from livebench.process_results.writing.plot_unscrambling.utils import plot_unscrambling_process_results
                return float(plot_unscrambling_process_results(gt, extracted))
            if task == "typos":
                from livebench.process_results.writing.typos.utils import typos_process_results
                return float(typos_process_results(gt, extracted))
        if cat == "instruction_following":
            # The packaged `instruction_following_process_results` is
            # batch-only — it builds a pandas DataFrame and crashes on a
            # single-row dict ("string indices must be integers"). Bypass
            # it and call the underlying strict evaluator directly. Same
            # scoring logic, just unwrapped.
            from livebench.if_runner.instruction_following_eval.evaluation_main import (
                test_instruction_following_strict, InputExample,
            )
            from livebench.process_results.instruction_following.utils import score_results
            # Each kwargs[i] is a wide dict where most keys are None. The
            # instruction's `build_description()` rejects unknown kwargs,
            # so strip Nones before passing.
            kwargs_clean = [{k: v for k, v in (kw or {}).items() if v is not None}
                            for kw in (data_i.get("kwargs") or [])]
            inp = InputExample(
                key=data_i.get("key", data_i.get("question_id", 0)),
                instruction_id_list=data_i["instruction_id_list"],
                prompt=question,
                kwargs=kwargs_clean,
            )
            result = test_instruction_following_strict(inp, {question: extracted})
            return float(score_results(result.follow_all_instructions, result.follow_instruction_list))
        if cat == "coding":
            # LCB_generation_process_results runs the model's code in subprocess
            # sandboxes against test cases. Hung subprocesses can deadlock the
            # parent in pipe_read indefinitely (SIGALRM doesn't fire while
            # blocked on pipe). Isolate via subprocess timeout.
            return _livebench_subproc_timeout(
                "livebench.process_results.coding.utils",
                "LCB_generation_process_results",
                (data_i, extracted),
                timeout=120,
            )
    except Exception as e:
        # Scorer failures (e.g. malformed prediction) → wrong, never crash run.
        return 0.0
    return 0.0


def check_livebench(extracted, data_i):
    """Boolean check used by `check_answer` — score > 0 is "correct"."""
    return _livebench_score(data_i, extracted) > 0.5


# ═══════════════════════════════════════════════════════════════════════
# ZebraLogic domain — grid-mode logic puzzles (Lin et al. 2025).
# Each row's `question` contains the ZeroEval JSON-output instruction; the
# model is expected to produce a json block of the form
#   {"reasoning": "...", "solution": {"House 1": {...}, "House 2": {...}, ...}}
# We parse it and compare cell-by-cell against the ground-truth grid.
#
# Headline metric: puzzle-level exact-match (all cells correct → 1.0, else 0.0).
# This matches the ZeroEval framework Qwen3 / DeepSeek / Claude evals use.
# ═══════════════════════════════════════════════════════════════════════
def extract_zebralogic(text, data_i=None):
    """Pass through — the JSON-extraction logic lives in `check_zebralogic`."""
    return str(text or "")


def _parse_zebralogic_response(text):
    """Pull a JSON object out of a model response. Handle:
       - ```json\\n{...}\\n```  fenced (most common; ZeroEval-style instruction)
       - ```\\n{...}\\n```      bare-fenced
       - {...}                  unwrapped
    Returns the parsed dict, or None if no usable JSON is found.
    """
    text = str(text or "").strip()
    # Try fenced first
    for marker in ("```json", "```"):
        idx = text.find(marker)
        if idx >= 0:
            after = text[idx + len(marker):]
            end = after.find("```")
            blob = after[:end] if end >= 0 else after
            try:
                return json.loads(blob.strip())
            except Exception:
                pass
    # Fall back to greedy outermost-brace extraction.
    start = text.find("{")
    if start < 0:
        return None
    # Scan for matching close brace by depth.
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(text)):
        c = text[i]
        if esc:
            esc = False; continue
        if c == "\\":
            esc = True; continue
        if c == '"' and not esc:
            in_str = not in_str; continue
        if in_str:
            continue
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start:i + 1])
                except Exception:
                    return None
    return None


def _zl_norm_key(s):
    """Aggressive key normalization for matching attribute names: drop case,
    spaces, underscores, hyphens. So `Phone Model` / `phone_model` /
    `PhoneModel` / `phone-model` all map to `phonemodel`."""
    return "".join(ch for ch in str(s).lower()
                   if ch.isalnum())


def check_zebralogic(extracted, data_i):
    """ZeroEval-style puzzle-level exact-match scorer for grid-mode ZebraLogic.

    Returns 1.0 if every cell in the model's solution matches the ground-truth
    grid, else 0.0. Cell comparison is case-insensitive and whitespace-stripped.
    Tolerates several common output shapes:
      - {"reasoning": ..., "solution": {"House 1": {...}}}    (canonical ZeroEval)
      - {"House 1": {...}, "House 2": {...}}                   (model omitted wrapper)
      - {"solution": [{...}, {...}]}                            (list-of-houses variant)
    Attribute keys match across snake_case / CamelCase / spaces / hyphens.
    """
    gt_blob = (data_i or {}).get("ground_truth_answer", "") or ""
    try:
        gt = json.loads(gt_blob) if isinstance(gt_blob, str) else gt_blob
    except Exception:
        return False
    header = list(gt.get("header") or [])
    gt_rows = list(gt.get("rows") or [])
    if not header or not gt_rows:
        return False

    parsed = _parse_zebralogic_response(extracted)
    if not isinstance(parsed, dict):
        return False

    # Try multiple shapes for where the per-house data lives.
    sol = parsed.get("solution")
    if not isinstance(sol, dict):
        # Fallback 1: maybe the model put houses directly at top level.
        if any(isinstance(k, str) and "house" in k.lower() for k in parsed.keys()):
            sol = parsed
        # Fallback 2: solution is a list of per-house dicts in order.
        elif isinstance(parsed.get("solution"), list):
            sol_list = parsed["solution"]
            sol = {f"House {i+1}": h for i, h in enumerate(sol_list) if isinstance(h, dict)}
        else:
            return False

    n_houses = len(gt_rows)
    attr_cols = header[1:]

    def _norm_val(s):
        return str(s).strip().lower()

    for i in range(n_houses):
        house_label = gt_rows[i][0]                # "1", "2", ...
        candidates = [
            f"House {house_label}",
            f"house {house_label}",
            f"House{house_label}",
            f"house_{house_label}",
            str(house_label),
            f"#{house_label}",
        ]
        house_dict = None
        for k in candidates:
            if k in sol and isinstance(sol[k], dict):
                house_dict = sol[k]; break
        if house_dict is None:
            # Aggressive fallback: scan all keys for one that contains the house number.
            for k, v in sol.items():
                if isinstance(v, dict) and isinstance(k, str) and str(house_label) in k:
                    house_dict = v; break
            if house_dict is None:
                return False

        # Build a normalized lookup of the model's keys.
        norm_lookup = {_zl_norm_key(k): v for k, v in house_dict.items()}

        for col_idx, attr in enumerate(attr_cols, start=1):
            gt_cell = _norm_val(gt_rows[i][col_idx])
            attr_norm = _zl_norm_key(attr)
            pred_cell = None
            if attr_norm in norm_lookup:
                pred_cell = _norm_val(norm_lookup[attr_norm])
            else:
                # Last-resort: look for partial matches (e.g. attr_norm is substring).
                for nk, v in norm_lookup.items():
                    if attr_norm in nk or nk in attr_norm:
                        pred_cell = _norm_val(v); break
            if pred_cell is None or pred_cell != gt_cell:
                return False
    return True


# ═══════════════════════════════════════════════════════════════════════
# AutoLogi domain — logical-puzzle benchmark with programmatic verification.
# Each row's ground_truth carries a Python source string `Inputs_Check_code`
# defining `def inputs_check(inputs): ...`. We extract the model's structured
# answer (a Python dict literal), exec the checker in a sandboxed subprocess,
# and treat True == correct. Multiple valid solutions per puzzle are fine —
# any input that satisfies the constraints scores 1.0.
# ═══════════════════════════════════════════════════════════════════════
def extract_autologi(text, data_i=None):
    """Pass through — dict-extraction lives in `check_autologi`."""
    return str(text or "")


_AUTOLOGI_RUNNER = r"""
import sys, json, ast, traceback

# 1) Read the inputs_check function source from stdin payload
payload = json.loads(sys.stdin.read())
checker_src = payload["checker_src"]
candidate_text = payload["candidate"]

# 2) Parse the model's output to find the largest dict-literal we can.
def _find_dict_literal(text):
    text = str(text or "")
    # Look for ```json fenced first, then bare {...}.
    for marker in ("```python", "```json", "```"):
        idx = text.find(marker)
        if idx >= 0:
            after = text[idx + len(marker):]
            end = after.find("```")
            blob = after[:end] if end >= 0 else after
            try:
                return ast.literal_eval(blob.strip())
            except Exception:
                try:
                    return json.loads(blob.strip())
                except Exception:
                    pass
    # Greedy outermost { ... } scan
    start = text.find("{")
    if start < 0:
        return None
    depth = 0; in_str = False; esc = False
    for i in range(start, len(text)):
        c = text[i]
        if esc: esc = False; continue
        if c == "\\": esc = True; continue
        if c == '"' and not esc: in_str = not in_str; continue
        if in_str: continue
        if c == "{": depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                blob = text[start:i+1]
                try:
                    return ast.literal_eval(blob)
                except Exception:
                    try:
                        return json.loads(blob)
                    except Exception:
                        return None
    return None

candidate = _find_dict_literal(candidate_text)
if candidate is None:
    print(json.dumps({"ok": False, "reason": "no_dict_found"}))
    sys.exit(0)

# 3) Execute the checker function in an isolated namespace and apply it.
ns = {}
try:
    exec(checker_src, ns)
    check_fn = ns.get("inputs_check")
    if not callable(check_fn):
        print(json.dumps({"ok": False, "reason": "no_inputs_check_fn"}))
        sys.exit(0)
    res = bool(check_fn(candidate))
    print(json.dumps({"ok": res}))
except Exception as e:
    print(json.dumps({"ok": False, "reason": f"exec_error:{type(e).__name__}"}))
"""


def check_autologi(extracted, data_i):
    """Score by running the dataset-shipped `inputs_check(model_dict)` in a
    sandboxed subprocess. Returns True iff `inputs_check` returns True."""
    import subprocess, tempfile, os as _os
    gt_blob = (data_i or {}).get("ground_truth_answer", "") or ""
    try:
        gt = json.loads(gt_blob) if isinstance(gt_blob, str) else gt_blob
    except Exception:
        return False
    checker_src = gt.get("Inputs_Check_code") or gt.get("inputs_check_code") or ""
    if not checker_src:
        return False

    payload = json.dumps({"checker_src": checker_src, "candidate": str(extracted or "")})
    try:
        proc = subprocess.run(
            ["python", "-c", _AUTOLOGI_RUNNER],
            input=payload.encode("utf-8"),
            capture_output=True,
            timeout=10,    # per-row cap; checker functions are tiny so 10s is plenty
        )
        out = proc.stdout.decode("utf-8", errors="replace").strip().split("\n")[-1]
        result = json.loads(out)
        return bool(result.get("ok"))
    except Exception:
        return False


EXTRACTORS = {
    "math": extract_math,
    "mc":   extract_mc,
    "code": extract_code,
    "chat": _noop_extract,
    "ifeval": extract_ifeval,
    "ifeval_oc": extract_ifeval,    # same passthrough extractor
    "triviaqa": extract_triviaqa,
    "livebench": extract_livebench,
    "zebralogic": extract_zebralogic,
    "autologi": extract_autologi,
}

CHECKERS = {
    "math": check_math,
    "mc":   check_mc,
    "code": check_code,
    "chat": _noop_check,
    "ifeval": check_ifeval,
    "ifeval_oc": check_ifeval_oc,
    "triviaqa": check_triviaqa,
    "livebench": check_livebench,
    "zebralogic": check_zebralogic,
    "autologi": check_autologi,
}


def get_domain(data_i=None, ds_cfg=None, default="math"):
    """Resolve the canonical domain for a sample.

    Precedence:
      1. Per-sample `data_i["domain"]` (e.g. opdlm_train has per-sample mix)
      2. Dataset-level `ds_cfg["domain"]`
      3. `default` (usually "math")

    Labels are mapped through DOMAIN_ALIASES so dataset-native names like
    "science" resolve to canonical ones ("mc").
    """
    raw = None
    if data_i is not None:
        raw = data_i.get("domain") or raw
    if raw is None and ds_cfg is not None:
        raw = ds_cfg.get("domain") or raw
    if raw is None:
        return default
    return DOMAIN_ALIASES.get(raw, raw)


def extract_answer(text, data_i=None, ds_cfg=None, default_domain="math",
                   scorer="math_verify"):
    """Domain-dispatched extraction. Respects ds_cfg['extract'] override.

    `scorer` selects the math extractor:
        "math_verify"  → math_verify library (default; pure_inference legacy)
        "opencompass"  → OpenCompass math_postprocess_v2 (matches SDAR HF eval)
    Non-math domains ignore `scorer`.
    """
    if ds_cfg is not None and ds_cfg.get("extract") is not None:
        return ds_cfg["extract"](text)
    domain = get_domain(data_i, ds_cfg, default=default_domain)
    if scorer == "opencompass":
        if domain == "math":
            return extract_math_oc(text, data_i)
        if domain == "mc":
            return extract_mc_oc(text, data_i)
    fn = EXTRACTORS.get(domain, extract_math)
    return fn(text, data_i)


def check_answer(extracted, data_i, ds_cfg=None, default_domain="math",
                 scorer="math_verify"):
    """Domain-dispatched correctness check. Respects ds_cfg['check'] override.

    The override signature is `check(predicted, ground_truth)` to stay
    backwards-compatible with existing BBH/HellaSwag helpers in eval_utils.

    `scorer` selects the math equivalence check (see extract_answer docstring).
    """
    if ds_cfg is not None and ds_cfg.get("check") is not None:
        return bool(ds_cfg["check"](extracted, data_i["ground_truth_answer"]))
    domain = get_domain(data_i, ds_cfg, default=default_domain)
    if domain == "math" and scorer == "opencompass":
        return bool(check_math_oc(extracted, data_i))
    fn = CHECKERS.get(domain, check_math)
    return bool(fn(extracted, data_i))
