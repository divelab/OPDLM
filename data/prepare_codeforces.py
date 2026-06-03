"""Preprocess open-r1/codeforces (verifiable subset) into Codeforces.json.

Output schema matches PrimeIntellect.json / LiveCodeBench.json so the existing
stdio scoring path in `reward/rl_execute.py` (`evaluate_stdio_dataset`) can
score rollouts unchanged:

    {
        "question":         str,            # full problem statement
        "test_input":       List[str],      # stdin per test case
        "test_output":      List[str],      # expected stdout per test case
        "test_time_limit":  int,            # seconds (per test case)
        "test_method":      "stdio",
        # extras for analysis (ignored by the runner / ok to have):
        "task_id":          str,            # CF "<contest_id>/<index>"
        "rating":           int | None,
        "tags":             List[str],
        "platform":         "codeforces",
    }

Filtering: keep only standard stdio problems with `official_tests` available —
drop interactive problems and problems with a `generated_checker` (these need a
piston-mediated checker we don't have). Of 422 test-split rows, 377 survive.

The codeforces verifiable train split (8,338 rows) is also written out for
optional RL training. Test split is what we'll use for benchmarking.

Usage (from repo root or `data/`):
    python data/prepare_codeforces.py
    python data/prepare_codeforces.py --split test          # only test
    python data/prepare_codeforces.py --max 50              # smoke test
"""
import argparse
import json
import math
import os
import sys

from datasets import load_dataset


HF_REPO = "open-r1/codeforces"
HF_CONFIG = "verifiable"

# Codeforces sets per-language time limits. CF's own Python runners use a 3x
# multiplier vs. the C++ baseline. Our stdio runner exec()s the snippet in a
# subprocess.Process; on top of the CF python multiplier we add a small slack
# for spawn overhead. The runner's deadline = test_time_limit (clamped int).
PYTHON_TIME_MULTIPLIER = 3
TIME_LIMIT_SLACK = 1


def _normalize_text(s: str) -> str:
    """Strip Codeforces' triple-dollar tex markers down to single-dollar.

    `$$$x$$$` is CF's Markdown convention; standard LaTeX uses `$x$` for
    inline math. Most LLMs are trained on the single-dollar form, so we
    rewrite to make the prompt closer to the training distribution.
    """
    return s.replace("$$$", "$") if s else s


def _normalize_io(s: str) -> str:
    """Strip CRLF in tests so stdin lines match what `input()` returns."""
    return s.replace("\r\n", "\n").replace("\r", "\n") if s else s


def build_question(row: dict) -> str:
    """Assemble a CF problem statement from its component fields.

    Mirrors the structure CF uses on the contest page: title -> statement ->
    Input -> Output -> Examples -> Note. Examples come from `examples` (not
    `official_tests`) — the former are the small public ones meant for the
    prompt, the latter include all hidden cases.
    """
    parts = []
    title = row.get("title")
    if title:
        parts.append(title.strip())
        parts.append("")  # blank line after title

    parts.append(_normalize_text(row["description"].strip()))

    if row.get("input_format"):
        parts.append("")
        parts.append("Input")
        parts.append(_normalize_text(row["input_format"].strip()))

    if row.get("output_format"):
        parts.append("")
        parts.append("Output")
        parts.append(_normalize_text(row["output_format"].strip()))

    examples = row.get("examples") or []
    if examples:
        parts.append("")
        parts.append("Examples")
        for i, ex in enumerate(examples, 1):
            parts.append("")
            parts.append(f"Input")
            parts.append(ex["input"].rstrip())
            parts.append(f"Output")
            parts.append(ex["output"].rstrip())

    note = row.get("note")
    if note:
        parts.append("")
        parts.append("Note")
        parts.append(_normalize_text(note.strip()))

    return "\n".join(parts).strip()


def keep_row(row: dict) -> bool:
    """Filter to standard stdio problems we can score with our runner.

    Drops:
      - non-stdio (file IO) problems
      - interactive problems (need a separate judge process)
      - problems with a `generated_checker` (special judge — partial credit
        / multiple-correct-answers cases that need a checker script)
      - problems flagged not-executable
      - problems missing official_tests
    """
    if row.get("input_mode") != "stdio":
        return False
    if not row.get("executable"):
        return False
    if row.get("interaction_format"):
        return False
    if row.get("generated_checker"):
        return False
    tests = row.get("official_tests") or []
    if not tests:
        return False
    return True


def convert(row: dict) -> dict:
    tests = row["official_tests"]
    test_input = [_normalize_io(t["input"]) for t in tests]
    test_output = [_normalize_io(t["output"]) for t in tests]

    # Convert CF's per-test float seconds to an integer deadline in our runner,
    # multiplied for Python and with a small spawn-overhead slack.
    raw_tl = float(row.get("time_limit") or 1.0)
    test_time_limit = max(1, int(math.ceil(raw_tl * PYTHON_TIME_MULTIPLIER)) + TIME_LIMIT_SLACK)

    return {
        "task_id": row["id"],
        "question": build_question(row),
        "test_input": test_input,
        "test_output": test_output,
        "test_time_limit": test_time_limit,
        "test_method": "stdio",
        "rating": row.get("rating"),
        "tags": list(row.get("tags") or []),
        "platform": "codeforces",
    }


def process_split(split: str, max_n: int | None) -> list[dict]:
    print(f"[codeforces] loading {HF_REPO}:{HF_CONFIG} split={split} ...")
    ds = load_dataset(HF_REPO, HF_CONFIG, split=split)
    n_total = len(ds)
    out = []
    for row in ds:
        if not keep_row(row):
            continue
        out.append(convert(row))
        if max_n and len(out) >= max_n:
            break
    print(f"[codeforces] {split}: kept {len(out)} / {n_total} rows")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["test", "train", "both"], default="both")
    ap.add_argument("--max", type=int, default=None,
                    help="Cap number of kept rows per split (smoke testing).")
    ap.add_argument("--data-dir", default=os.path.dirname(os.path.abspath(__file__)),
                    help="Output directory (defaults to this script's parent — i.e. data/).")
    args = ap.parse_args()

    os.makedirs(args.data_dir, exist_ok=True)
    splits = ["test", "train"] if args.split == "both" else [args.split]
    for split in splits:
        rows = process_split(split, args.max)
        out_name = "Codeforces.json" if split == "test" else "Codeforces_train.json"
        out_path = os.path.join(args.data_dir, out_name)
        with open(out_path, "w") as f:
            json.dump(rows, f, indent=2, ensure_ascii=False)
        print(f"[codeforces] wrote {len(rows)} -> {out_path}")


if __name__ == "__main__":
    sys.exit(main())
