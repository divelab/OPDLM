"""
Minimal conversion from mix_60k_4096_v2.parquet → opdlm_train.json.

Only strips instruction preambles from each domain. No prompt rewriting,
no test_method/fn_name extraction. Just rename columns and remove templates.

  math     : strip "Please reason step by step..." / "Solve the following..."
  science  : strip "Please show your choice..."
  code     : strip "You are an expert python programmer..." preamble
  chat     : untouched
"""
import json
import os
import re
import pandas as pd
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_IN = os.path.join(ROOT, "data", "mix_60k_4096_v1", "mix_60k_4096_v2.parquet")
DATA_OUT = os.path.join(ROOT, "data", "opdlm_train.json")

# ── Prefixes to strip ────────────────────────────────────────────
MATH_PREFIX = "Please reason step by step, and put your final answer within \\boxed{}."
MATH_SECONDARY = (
    "Solve the following math problem. Make sure to put the answer "
    "(and only answer) inside \\boxed{}."
)
MC_PREFIX = 'Please show your choice in the answer field with only the choice letter, e.g., "answer": "C".'


def strip_leading_line(q: str, line: str) -> str:
    if q.startswith(line):
        return q[len(line):].lstrip("\n")
    return q


def strip_code_preamble(q: str) -> str:
    """Strip 'You are an expert python programmer...' preamble."""
    marker = "# Your code here\n"
    idx = q.find(marker)
    if idx != -1:
        rest = q[idx + len(marker):]
        m = re.match(r"```?\s*\n+", rest)
        if m:
            return rest[m.end():].lstrip("\n")
        return rest.lstrip("\n")
    # Fallback: strip first paragraph
    parts = q.split("\n\n", 1)
    if len(parts) == 2:
        return parts[1].lstrip("\n")
    return q


def strip_instruction(domain: str, question: str) -> str:
    if domain == "math":
        question = strip_leading_line(question, MATH_PREFIX)
        question = strip_leading_line(question, MATH_SECONDARY)
    elif domain == "science":
        question = strip_leading_line(question, MC_PREFIX)
    elif domain == "code":
        question = strip_code_preamble(question)
    return question.strip()


def main():
    print(f"Loading {DATA_IN}...")
    df = pd.read_parquet(DATA_IN)
    print(f"  {len(df)} samples")

    out = []
    domain_counts = Counter()
    for _, row in df.iterrows():
        domain = row["domain"]
        question = strip_instruction(domain, row["input"])
        d = {
            "question": question,
            "ground_truth_answer": row["answer"],
            "domain": domain,
            "source": row["source"],
        }
        if row.get("tests") is not None and str(row["tests"]).strip():
            d["tests_json"] = row["tests"]
        domain_counts[domain] += 1
        out.append(d)

    print(f"\nDomain counts:")
    for dom, n in sorted(domain_counts.items()):
        print(f"  {dom}: {n}")

    print(f"\nWriting {DATA_OUT}...")
    with open(DATA_OUT, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"Done. {len(out)} samples written.")


if __name__ == "__main__":
    main()
