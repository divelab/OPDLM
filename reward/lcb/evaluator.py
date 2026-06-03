# Minimal LCB code-generation evaluator, adapted from SDAR's opencompass copy.
# Evaluates extracted code predictions against LCB input/output test samples.

import json
import multiprocessing
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
from tqdm import tqdm

from .testing_util import run_test
from .pass_k_utils import compute_metrics_from_results


def _codegen_check_correctness(sample, generation, timeout, debug=False):
    def _temp_run(sample, generation, debug, result, metadata_list, timeout):
        res, metadata = run_test(sample, test=generation, debug=debug, timeout=timeout)
        result.append(res)
        metadata_list.append(metadata)

    manager = multiprocessing.Manager()
    result = manager.list()
    metadata_list = manager.list()
    p = multiprocessing.Process(
        target=_temp_run,
        args=(sample, generation, debug, result, metadata_list, timeout),
    )
    p.start()
    p.join(timeout=(timeout + 1) *
           len(json.loads(sample['input_output'])['inputs']) + 5)
    if p.is_alive():
        p.kill()
    if not result:
        in_outs = json.loads(sample['input_output'])
        result = [[-1 for _ in range(len(in_outs['inputs']))]]
    return result[0], (metadata_list[0] if metadata_list else {})


def _evaluate_generations_by_problem(problem_generations, sample, debug, timeout):
    res, metadata = [], []
    for o in problem_generations:
        try:
            curr_res, curr_metadata = _codegen_check_correctness(sample, o, timeout=timeout, debug=debug)
            fixed = []
            for e in curr_res:
                if isinstance(e, np.ndarray):
                    e = e.item(0)
                if isinstance(e, np.bool_):
                    e = bool(e)
                fixed.append(e)
            curr_res = fixed
        except Exception as e:  # noqa
            curr_res = [-2]
            curr_metadata = {'error': repr(e)}
        res.append(curr_res)
        metadata.append(curr_metadata)
    return res, metadata


def codegen_metrics(samples_list, generations_list, k_list=[1],
                    num_process_evaluate=4, timeout=6, debug=False):
    samples_linear, generations_linear, remap_index = [], [], []
    results = defaultdict(list)
    metadatas = defaultdict(list)
    for idx, (sample, gen_list) in enumerate(zip(samples_list, generations_list)):
        for gen in gen_list:
            samples_linear.append(sample)
            generations_linear.append([gen])
            remap_index.append(idx)

    results_linear, metadatas_linear = {}, {}
    with tqdm(total=len(samples_linear), desc="evaluating") as pbar:
        with ProcessPoolExecutor(max_workers=1 if debug else num_process_evaluate) as ex:
            futures = {
                ex.submit(_evaluate_generations_by_problem, [generations_linear[i][0]],
                          samples_linear[i], debug, timeout): i
                for i in range(len(samples_linear))
            }
            for fut in as_completed(futures):
                i = futures[fut]
                r, m = fut.result()
                results_linear[i] = r
                metadatas_linear[i] = m
                pbar.update(1)

    for i in sorted(results_linear.keys()):
        results[remap_index[i]].append(results_linear[i][0])
        metadatas[remap_index[i]].append(metadatas_linear[i][0] if metadatas_linear[i] else {})

    metrics = compute_metrics_from_results(results, k_list=k_list)
    return metrics, dict(results), dict(metadatas)
