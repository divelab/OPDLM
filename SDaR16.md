# SDAR-4B-Chat-b16 K=16 evaluation results

Consolidated on 2026-07-31 from the completed 10-GPU evaluation suite.

## Evaluation setting

- Model: `models/SDAR-4B-Chat-b16`
- Decoding block/draft size: K=16
- Sampling: greedy, one response per problem
- Sharding: 10 independent GPU shards per run
- Modes:
  1. Self-speculative K=16
  2. Native blockwise K=16, `low_confidence_static`
  3. Native blockwise K=16, `low_confidence_dynamic`, threshold 0.9
- Datasets: AIME 2024, AIME 2025, MATH-500, GSM8K, GPQA-Diamond, HumanEval, MBPP, and LCB-v6

All 24 dataset/mode runs completed. Every run has 10 shard outputs, one merged output with the expected number of records, an evaluator result, and a merge/score log. A scan of the suite logs found no tracebacks or launcher failures.

## Accuracy summary

For HumanEval and MBPP, each cell reports `base / plus` from the EvalPlus result JSON. Other cells report the benchmark primary accuracy. LCB-v6 uses the independently verified per-shard scores documented below.

| Dataset | N | Max new tokens | Self-speculative K=16 | Native static K=16 | Native dynamic K=16 |
|---|---:|---:|---:|---:|---:|
| AIME 2024 | 30 | 8,000 | 3.33% (1/30) | 3.33% (1/30) | 6.67% (2/30) |
| AIME 2025 | 30 | 8,000 | 16.67% (5/30) | 3.33% (1/30) | 6.67% (2/30) |
| MATH-500 | 500 | 4,000 | 63.80% (319/500) | 63.20% (316/500) | 62.20% (311/500) |
| GSM8K | 1,319 | 2,000 | 85.97% (1,134/1,319) | 86.81% (1,145/1,319) | 85.90% (1,133/1,319) |
| GPQA-Diamond | 198 | 4,000 | 32.32% (64/198) | 19.70% (39/198) | 20.71% (41/198) |
| HumanEval base / plus | 164 | 8,000 | 75.00% (123/164) / 68.90% (113/164) | 70.12% (115/164) / 65.24% (107/164) | 70.12% (115/164) / 63.41% (104/164) |
| MBPP base / plus | 378 | 1,000 | 75.93% (287/378) / 63.49% (240/378) | 72.75% (275/378) / 62.43% (236/378) | 74.07% (280/378) / 64.02% (242/378) |
| LCB-v6 | 454 | 8,000 | 11.67% (53/454) | 11.01% (50/454) | 10.79% (49/454) |

Across the eight benchmarks, self-speculation is best on AIME 2025, MATH-500, GPQA-Diamond, HumanEval base/plus, MBPP base, and LCB-v6. Static is best on GSM8K. Dynamic is best on AIME 2024 and MBPP+.

## Self-speculative aggregate decoding metrics

All counts below are sums across the 10 shards. Parallel wall time is the slowest shard. `Tokens/conceptual forward` follows the established experiment convention:

```text
generated_tokens / (2 * sequence_iterations)
```

The two conceptual calls are one draft and one verification call per sequence iteration. This metric is intentionally separate from the implementation-level batched-forward count.

| Dataset | Accepted / proposed | Acceptance | Iterations | Generated tokens | Accepted/iteration | Generated/iteration | Tokens/conceptual forward | Batched draft / verify calls | Parallel wall time |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| AIME 2024 | 65,774 / 92,648 | 70.99% | 5,794 | 68,632 | 11.35 | 11.85 | 5.92 | 4,023 / 4,023 | 2m 16s |
| AIME 2025 | 56,803 / 87,104 | 65.21% | 5,447 | 59,917 | 10.43 | 11.00 | 5.50 | 3,179 / 3,179 | 2m 45s |
| MATH-500 | 297,445 / 506,919 | 58.68% | 31,695 | 319,191 | 9.38 | 10.07 | 5.04 | 10,972 / 10,972 | 4m 10s |
| GSM8K | 213,565 / 458,421 | 46.59% | 28,655 | 236,927 | 7.45 | 8.27 | 4.13 | 6,806 / 6,806 | 2m 16s |
| GPQA-Diamond | 6,695 / 28,336 | 23.63% | 1,771 | 8,223 | 3.78 | 4.64 | 2.32 | 922 / 922 | 31s |
| HumanEval | 96,103 / 147,931 | 64.96% | 9,249 | 100,814 | 10.39 | 10.90 | 5.45 | 4,684 / 4,684 | 3m 12s |
| MBPP | 55,312 / 172,141 | 32.13% | 10,767 | 64,732 | 5.14 | 6.01 | 3.01 | 3,168 / 3,168 | 1m 05s |
| LCB-v6 | 103,315 / 184,750 | 55.92% | 11,551 | 110,613 | 8.94 | 9.58 | 4.79 | 6,103 / 6,103 | 4m 47s |

## Native blockwise decoding metrics

`Tokens/step` is the evaluator's merged `avg tok/step` value. Parallel wall time is the slowest of the 10 shard progress-bar elapsed times.

| Dataset | Static tokens/step | Static wall time | Dynamic tokens/step | Dynamic wall time |
|---|---:|---:|---:|---:|
| AIME 2024 | 1.000 | 4m 01s | 2.941 | 2m 33s |
| AIME 2025 | 1.000 | 3m 57s | 2.769 | 35s |
| MATH-500 | 1.000 | 4m 20s | 3.433 | 58s |
| GSM8K | 1.000 | 3m 15s | 2.799 | 1m 02s |
| GPQA-Diamond | 1.000 | 4s | 1.255 | 3s |
| HumanEval | 1.000 | 4m 10s | 2.445 | 17s |
| MBPP | 1.000 | 45s | 1.570 | 20s |
| LCB-v6 | 1.000 | 8m 51s | 2.160 | 1m 05s |

The native and self-speculative efficiency columns are not interchangeable: native reports tokens per blockwise decoding step, while self-speculative reports generated tokens per two conceptual model calls.

## Generation-limit hits

Counts are responses whose recorded response length reached or exceeded that run's configured maximum.

| Dataset | Self-speculative | Native static | Native dynamic |
|---|---:|---:|---:|
| AIME 2024 | 7/30 | 5/30 | 3/30 |
| AIME 2025 | 5/30 | 2/30 | 1/30 |
| MATH-500 | 23/500 | 23/500 | 19/500 |
| GSM8K | 6/1,319 | 25/1,319 | 24/1,319 |
| GPQA-Diamond | 0/198 | 0/198 | 0/198 |
| HumanEval | 8/164 | 1/164 | 0/164 |
| MBPP | 18/378 | 1/378 | 2/378 |
| LCB-v6 | 7/454 | 18/454 | 12/454 |

The self-speculative MBPP result deserves caution because 18/378 responses reached the 1,000-token cap. AIME also has non-trivial cap rates in every mode.

## Evaluator caveats

### EvalPlus summary parsing

EvalPlus completed and wrote full result JSON files for all HumanEval and MBPP runs. The wrapper's console summary printed `base pass@1=None` and `plus pass@1=None` because it looked for a summary layout not present in the generated JSON. The base/plus counts in this report were calculated directly from each sample's `base_status` and `plus_status` in `samples-sanitized_eval_results.json`.

### LCB-v6 independent per-shard verification

The monolithic LCB evaluator reported an invalid 0/454 for all three modes. Each raw shard was therefore rescored independently with the repository private-test evaluator using its established six-second per-test limit and four evaluation processes. The 10 shard sizes sum to 454 problems: shards 0 through 8 contain 46 problems each, and shard 9 contains 40.

| Shard | Problems | Self-speculative passed | Native static passed | Native dynamic passed |
|---:|---:|---:|---:|---:|
| 0 | 46 | 6 | 5 | 6 |
| 1 | 46 | 7 | 6 | 5 |
| 2 | 46 | 6 | 7 | 7 |
| 3 | 46 | 6 | 4 | 4 |
| 4 | 46 | 3 | 2 | 2 |
| 5 | 46 | 5 | 5 | 4 |
| 6 | 46 | 8 | 9 | 9 |
| 7 | 46 | 5 | 6 | 6 |
| 8 | 46 | 3 | 3 | 2 |
| 9 | 40 | 4 | 3 | 4 |
| **Total** | **454** | **53 (11.67%)** | **50 (11.01%)** | **49 (10.79%)** |

Platform cross-check:

| Mode | AtCoder | LeetCode | Combined |
|---|---:|---:|---:|
| Self-speculative | 30/287 (10.45%) | 23/167 (13.77%) | 53/454 (11.67%) |
| Native static | 29/287 (10.10%) | 21/167 (12.57%) | 50/454 (11.01%) |
| Native dynamic | 30/287 (10.45%) | 19/167 (11.38%) | 49/454 (10.79%) |

These independently summed results replace the invalid monolithic 0% values. Self-speculation leads static by 3 solved problems and 0.66 percentage points, and leads dynamic by 4 problems and 0.88 points. The margins are small and should not be overstated.

## Artifact locations

- Launcher: `pure_inference/run_sdar4b_b16_selfspec_k16_all.sh`
- Master log: `pure_inference/results/sdar4b_b16_k16_all_modes_logs/master.log`
- Per-run logs: `pure_inference/results/sdar4b_b16_k16_all_modes_logs/{self_speculative,static,dynamic}/<dataset>/`
- Result directories: `pure_inference/results/SDAR-4B-Chat-b16_<dataset>_<mode-tag>_all_suite/`

The retained tmux pane reports that the suite process exited and all evaluations completed; the session remains open only for inspection.
