#!/usr/bin/env python3
"""Compare cached and uncached IDLM generation outputs."""

import argparse
import json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("cached")
    parser.add_argument("uncached")
    parser.add_argument("--allow_drift", action="store_true")
    parser.add_argument("--max_accuracy_gap", type=float, default=0.10)
    args = parser.parse_args()

    with open(args.cached) as f:
        cached = json.load(f)
    with open(args.uncached) as f:
        uncached = json.load(f)

    if len(cached) != len(uncached):
        raise AssertionError(f"Row-count mismatch: {len(cached)} != {len(uncached)}")

    mismatches = []
    for index, (cached_row, uncached_row) in enumerate(zip(cached, uncached)):
        if cached_row["response"] != uncached_row["response"]:
            mismatches.append(index)

    cached_correct = sum(bool(row.get("correct")) for row in cached)
    uncached_correct = sum(bool(row.get("correct")) for row in uncached)
    cached_accuracy = cached_correct / max(len(cached), 1)
    uncached_accuracy = uncached_correct / max(len(uncached), 1)
    answer_matches = sum(
        cached_row.get("extracted_answer") == uncached_row.get("extracted_answer")
        for cached_row, uncached_row in zip(cached, uncached)
    )

    if mismatches and not args.allow_drift:
        raise AssertionError(f"Cached/uncached response mismatches: {mismatches}")
    if abs(cached_accuracy - uncached_accuracy) > args.max_accuracy_gap:
        raise AssertionError(
            f"Accuracy gap too large: cached={cached_accuracy:.4f}, "
            f"uncached={uncached_accuracy:.4f}"
        )

    cached_calls = sum(row["forward_passes"] for row in cached)
    uncached_calls = sum(row["forward_passes"] for row in uncached)
    print(
        f"Exact response matches: {len(cached) - len(mismatches)}/{len(cached)}"
    )
    print(f"Extracted-answer matches: {answer_matches}/{len(cached)}")
    print(
        f"Accuracy: cached={cached_accuracy:.4f}, "
        f"uncached={uncached_accuracy:.4f}"
    )
    print(f"Sequence forwards: cached={cached_calls}, uncached={uncached_calls}")


if __name__ == "__main__":
    main()
