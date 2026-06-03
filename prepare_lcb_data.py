"""
Prepare LiveCodeBench v5 / v6 data files (date-filtered windows).

- v5 window: 2024-10-01 → 2025-02-28  (the canonical "LCB v5" eval window
  used in the Qwen3 technical report; matches 21.3% for Qwen3-4B non-thinking)
- v6 window: 2024-08-01 → 2025-05-31  (the canonical "LCB v6" eval window)

Output: data/LCB_v5.json, data/LCB_v6.json
Each entry: {task_id, question, ground_truth_answer, input_output, platform,
             difficulty, contest_date, starter_code}
- `question` is the fully-rendered canonical LCB user prompt
  (Question + Format + Answer section).
- `input_output` is a JSON string with {"inputs","outputs","fn_name"}.

Usage:
    python prepare_lcb_data.py
"""
import base64
import json
import os
import pickle
import sys
import zlib

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "reward"))
from lcb.prompts import CodeGenerationPromptConstants

HF_FILES = ["test.jsonl", "test2.jsonl", "test3.jsonl",
            "test4.jsonl", "test5.jsonl", "test6.jsonl"]

# Date-filter windows (inclusive, YYYY-MM).
WINDOWS = {
    "LCB_v5": ("2024-10", "2025-02"),
    "LCB_v6": ("2024-08", "2025-05"),
}

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


def _load_raw():
    from huggingface_hub import hf_hub_download
    rows = []
    for fn in HF_FILES:
        path = hf_hub_download(
            repo_id="livecodebench/code_generation_lite",
            repo_type="dataset",
            filename=fn,
        )
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    return rows


def _decode_private(raw):
    try:
        return json.loads(raw)
    except Exception:
        return json.loads(pickle.loads(zlib.decompress(
            base64.b64decode(raw.encode("utf-8")))))


def _build_user_prompt(item):
    if item["starter_code"]:
        fmt = (f"### Format: {CodeGenerationPromptConstants.FORMATTING_MESSAGE_WITH_STARTER_CODE}\n"
               f"```python\n{item['starter_code']}\n```\n\n")
    else:
        fmt = (f"### Format: {CodeGenerationPromptConstants.FORMATTING_WITHOUT_STARTER_CODE}\n"
               f"```python\n# YOUR CODE HERE\n```\n\n")
    return (f"### Question:\n{item['question_content']}\n\n"
            f"{fmt}### Answer: (use the provided format with backticks)\n\n")


def _to_record(item):
    public_tc = json.loads(item["public_test_cases"])
    private_tc = _decode_private(item["private_test_cases"])
    metadata = json.loads(item["metadata"])
    inputs = [t["input"] for t in public_tc + private_tc]
    outputs = [t["output"] for t in public_tc + private_tc]
    fn_name = metadata.get("func_name", None)

    return {
        "task_id": item["question_id"],
        "question": _build_user_prompt(item),
        # LCB scoring uses input_output, not a single reference answer;
        # `ground_truth_answer` is kept for schema parity with other datasets.
        "ground_truth_answer": "",
        "input_output": json.dumps({
            "inputs": inputs, "outputs": outputs, "fn_name": fn_name,
        }),
        "platform": item.get("platform"),
        "difficulty": item.get("difficulty"),
        "contest_date": item.get("contest_date"),
        "starter_code": item.get("starter_code") or "",
    }


def main():
    print(f"[lcb] downloading release_v6 (cumulative, 6 shards)...")
    raw = _load_raw()
    print(f"[lcb] loaded {len(raw)} raw problems")

    records = [_to_record(it) for it in raw]
    os.makedirs(OUT_DIR, exist_ok=True)

    for name, (lo, hi) in WINDOWS.items():
        subset = [r for r in records if lo <= (r.get("contest_date") or "")[:7] <= hi]
        out_path = os.path.join(OUT_DIR, f"{name}.json")
        with open(out_path, "w") as f:
            json.dump(subset, f, indent=2, ensure_ascii=False)
        print(f"[lcb] {name}: {len(subset)} problems  window={lo}..{hi}  -> {out_path}")


if __name__ == "__main__":
    main()
