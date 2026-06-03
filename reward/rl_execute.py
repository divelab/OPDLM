import argparse, ast, io, json, os, sys, time, textwrap, multiprocessing as mp
import concurrent.futures as cf
import subprocess
import tempfile
from pathlib import Path
import math
import numpy as np
from termcolor import cprint
from tqdm import tqdm
from omegaconf import DictConfig, ListConfig, OmegaConf


# Datasets whose scoring is delegated to the evalplus CLI (HumanEval+/MBPP+).
# eval_utils.DATASET_CONFIGS marks these with defer_scoring="evalplus"; we
# re-use the same names here to keep the logic in one place.
# `HumanEval_sdar` shares HumanEval.json (same 164 task IDs as evalplus's set)
# but uses SDAR's plain instruction prompt instead of evalplus prefill — the
# scoring path is the same evalplus base pass@1.
_EVALPLUS_DATASETS = {
    "HumanEval": "humaneval",
    "HumanEval_sdar": "humaneval",
    "MBPP": "mbpp",
}

# LiveCodeBench date-filtered windows (see prepare_lcb_data.py).
# Scoring goes through reward.lcb.codegen_metrics (pyext RuntimeModule +
# per-test-case timeout) on the input_output stored on each item.
# `LCB_v6_sdar` shares LCB_v6.json with `LCB_v6`; both score the same way.
_LCB_DATASETS = {"LCB_v5", "LCB_v6", "LCB_v6_sdar"}


def evaluate_lcb_dataset(data: list[dict], timeout: int = 6,
                        num_process_evaluate: int = 4) -> list[dict]:
    """Score LCB v5/v6 rollouts via reward.lcb.codegen_metrics.

    Each item must carry `task_id`, `input_output` (JSON string with
    inputs/outputs/fn_name), and `extracted_output` (list of extracted code
    candidates per k_sample). We populate `correctness[i]` = [bool(all tests
    pass)] so downstream rl_code_reward aggregation works unchanged.
    """
    # Lazy import so rl_execute stays importable without the lcb package
    # (e.g. on nodes that only run evalplus or function/stdio scoring).
    # `reward/` is not a python package; we live inside it, so `lcb.xxx`
    # resolves as a top-level subpackage via sys.path.
    import sys as _sys, os as _os
    _here = _os.path.dirname(_os.path.abspath(__file__))
    if _here not in _sys.path:
        _sys.path.insert(0, _here)
    from lcb.evaluator import codegen_metrics

    # Build (samples_list, generations_list) in the shape codegen_metrics wants.
    # Each item has k candidates; we flatten into 1 sample per candidate so we
    # can fill correctness[i] in order.
    samples_list, generations_list, flat_index = [], [], []
    for i, item in enumerate(data):
        io = item.get("input_output")
        if io is None:
            continue
        cands = item.get("extracted_output") or []
        for k_i, cand in enumerate(cands):
            samples_list.append({"input_output": io})
            generations_list.append([cand])
            flat_index.append((i, k_i))

    if not samples_list:
        # No code to score — mark everything fail so nothing silently passes.
        for item in data:
            k = len(item.get("extracted_output") or [])
            item["correctness"] = [[False] for _ in range(k)]
            item["execution_result"] = [["missing"] for _ in range(k)]
            item.setdefault("step_map", [])
        return data

    cprint(f"[lcb] scoring {len(samples_list)} submissions "
           f"({len(data)} problems) with {num_process_evaluate} procs...",
           "cyan")
    metrics, results, _meta = codegen_metrics(
        samples_list, generations_list,
        k_list=[1],
        num_process_evaluate=num_process_evaluate,
        timeout=timeout,
    )
    pass1 = metrics.get("pass@1") if isinstance(metrics, dict) else None
    cprint(f"[lcb] pass@1 (micro across submissions) = {pass1}", "green")

    # Initialize per-item correctness buckets with the right width.
    for item in data:
        k = len(item.get("extracted_output") or [])
        item["correctness"] = [[False] for _ in range(k)]
        item["execution_result"] = [["missing"] for _ in range(k)]
        item.setdefault("step_map", [])

    # results is {flat_problem_index: [[t1,t2,...]]}; fill per-candidate pass.
    for flat_i, (data_i, k_i) in enumerate(flat_index):
        cand_res = results.get(flat_i, [[]])
        tests = cand_res[0] if cand_res else []
        passed = bool(tests) and all(x is True for x in tests)
        data[data_i]["correctness"][k_i] = [passed]
        data[data_i]["execution_result"][k_i] = ["pass" if passed else "fail"]

    # Per-platform breakdown (k_i=0 only, as in the debug script).
    from collections import defaultdict
    per_plat = defaultdict(list)
    for item in data:
        plat = item.get("platform") or "unknown"
        corr = item.get("correctness") or [[False]]
        per_plat[plat].append(bool(corr[0][0]))
    cprint("[lcb] by platform:", "cyan")
    for plat in sorted(per_plat):
        lst = per_plat[plat]
        acc = 100 * sum(lst) / len(lst) if lst else 0.0
        cprint(f"  {plat:12s}: {sum(lst):4d}/{len(lst):4d}  ({acc:.2f}%)",
               "cyan")

    # Stash rollup on every item so downstream can log without re-reading.
    for item in data:
        item["lcb_pass_at_1"] = pass1
    return data


def evaluate_evalplus_dataset(data: list[dict], evalplus_name: str,
                              work_dir: str) -> list[dict]:
    """Score HumanEval/MBPP rollouts via `evalplus.evaluate`.

    - Writes a {task_id, solution} jsonl where `solution` = each candidate's
      raw `full_output` (evalplus.sanitize extracts the function itself).
    - Calls `evalplus.evaluate --i_just_wanna_run` on that jsonl.
    - Parses the resulting `.eval_results.json` per-task `base_status` /
      `plus_status` and writes them into each item's `correctness` /
      `execution_result` so downstream rl_code_reward.py works unchanged.
      `correctness[i]` = [base_pass] so existing `all(x)` aggregation
      reports base pass@1. We also stash the plus pass result on each item
      under `evalplus_per_task` for optional logging.
    """
    os.makedirs(work_dir, exist_ok=True)
    samples_path = os.path.join(work_dir, "samples.jsonl")

    # Build samples.jsonl. evalplus expects one row per (task_id, solution);
    # if k_sample > 1 we write k rows per task and the CLI's pass@k math
    # uses all of them (we only care about k=1 greedy for eval).
    written = 0
    with open(samples_path, "w") as f:
        for item in data:
            task_id = item.get("task_id")
            if task_id is None:
                continue
            for cand in item.get("full_output", []):
                f.write(json.dumps({"task_id": task_id, "solution": cand}) + "\n")
                written += 1
    cprint(f"[evalplus] wrote {written} samples -> {samples_path}", "cyan")

    # Step 1: sanitize. Our rollouts wrap code in markdown fences and may
    # include chatter before/after the ```python block; evalplus.sanitize
    # extracts just the function body and writes a sibling
    # `samples-sanitized.jsonl`. We must run this before evaluate, otherwise
    # the `--i_just_wanna_run` path will try to exec markdown-wrapped text
    # and every task reports base_status=fail with empty fail_tests.
    log_path = os.path.join(work_dir, "evalplus.log")
    sanitized_path = samples_path.replace(".jsonl", "-sanitized.jsonl")
    sanitize_cmd = [
        "evalplus.sanitize",
        "--samples", samples_path,
    ]
    cprint(f"[evalplus] running: {' '.join(sanitize_cmd)}", "cyan")
    # Stream subprocess output straight to the log file (not via PIPE) so a
    # stuck child can never deadlock the parent on pipe drain.
    with open(log_path, "w") as logf:
        rc = subprocess.call(sanitize_cmd, stdout=logf, stderr=subprocess.STDOUT)
    if rc != 0 or not os.path.isfile(sanitized_path):
        # The CLI processes the whole jsonl in one shot — if any single row's
        # output blows up the AST walker (RecursionError on deeply nested
        # generated code, common with thinking-on + large max_tokens), the
        # entire subprocess crashes and all 378 rows are lost.
        #
        # Fallback: replicate evalplus.sanitize's per-row script() in-process
        # with sys.setrecursionlimit raised AND a try/except per row. Bad
        # rows get an empty `solution` (counts as failure, doesn't kill the
        # run). Good rows are sanitized normally.
        cprint(
            f"[evalplus] CLI sanitize failed (rc={rc}); falling back to "
            f"in-process per-row sanitize with try/except. Log: {log_path}",
            "yellow",
        )
        import sys
        from evalplus.sanitize import sanitize as _sanitize_one
        from evalplus.data.utils import load_solutions
        from evalplus.data import get_human_eval_plus, get_mbpp_plus
        sys.setrecursionlimit(20000)

        entry_point = {}
        dataset = {**get_human_eval_plus(), **get_mbpp_plus()}
        for tid, problem in dataset.items():
            entry_point[tid] = problem["entry_point"]

        new_solutions = []
        n_total, n_failed = 0, 0
        for solution in load_solutions(samples_path):
            tid = solution["task_id"]
            if tid not in dataset:
                continue
            n_total += 1
            fn_name = entry_point.get(tid)
            old_code = solution.get("solution") or (
                dataset[tid]["prompt"] + "\n" + solution.get("completion", "")
            )
            try:
                new_code = _sanitize_one(code=old_code, entrypoint=fn_name)
            except (RecursionError, Exception) as e:
                n_failed += 1
                new_code = ""    # empty → guaranteed test failure, but doesn't crash
            new_solutions.append({"task_id": tid, "solution": new_code})

        with open(sanitized_path, "w") as f:
            for s in new_solutions:
                f.write(json.dumps(s) + "\n")
        cprint(
            f"[evalplus] in-process fallback wrote {len(new_solutions)} rows "
            f"({n_failed} parse errors → empty solution). "
            f"Path: {sanitized_path}",
            "yellow",
        )

    # Step 2: evaluate the sanitized samples. Writes a sibling
    # `samples-sanitized.eval_results.json`.
    cmd = [
        "evalplus.evaluate",
        "--dataset", evalplus_name,
        "--samples", sanitized_path,
        "--i_just_wanna_run",
    ]
    # evalplus.evaluate refuses to overwrite and prompts [Y/N] on stdin if
    # a stale eval_results.json exists — if we run the same samples twice,
    # that prompt silently deadlocks the parent (pipe buffer never drains).
    # Delete the stale file and feed /dev/null to stdin so any future prompt
    # errors out immediately instead of hanging.
    stale_eval_json = sanitized_path.replace(".jsonl", ".eval_results.json")
    if os.path.exists(stale_eval_json):
        os.remove(stale_eval_json)
    cprint(f"[evalplus] running: {' '.join(cmd)}", "cyan")
    with open(log_path, "a") as logf, open(os.devnull, "rb") as devnull:
        rc = subprocess.call(cmd, stdout=logf, stderr=subprocess.STDOUT,
                             stdin=devnull)
    if rc != 0:
        raise RuntimeError(
            f"evalplus.evaluate failed (rc={rc}); log: {log_path}"
        )

    eval_json = sanitized_path.replace(".jsonl", ".eval_results.json")
    if not os.path.isfile(eval_json):
        raise RuntimeError(
            f"evalplus did not produce {eval_json}. Log: {log_path}"
        )
    with open(eval_json) as f:
        eval_data = json.load(f)

    # Per-task results: eval_data["eval"][task_id] is a list (one row per
    # submitted sample, in order) of dicts with base_status / plus_status.
    per_task = eval_data.get("eval", {})

    # Map base/plus rollup for logging.
    pak = eval_data.get("pass_at_k", {}) or {}
    base_p1 = pak.get("base", {}).get("pass@1")
    plus_p1 = pak.get("plus", {}).get("pass@1")
    cprint(
        f"[evalplus] {evalplus_name}: base pass@1={base_p1}  plus pass@1={plus_p1}",
        "green",
    )

    for item in data:
        task_id = item.get("task_id")
        m_code = len(item.get("full_output", []))
        if task_id is None or task_id not in per_task:
            # Missing task → mark everything as fail so it doesn't silently
            # count as correct downstream.
            item["correctness"] = [[False] for _ in range(m_code)]
            item["execution_result"] = [["missing"] for _ in range(m_code)]
            item.setdefault("step_map", [])
            item["evalplus_per_task"] = [{"base": False, "plus": False}] * m_code
            continue
        rows = per_task[task_id]
        # Align candidates by index. evalplus preserves submission order.
        correctness = []
        execution_result = []
        per_cand = []
        for i in range(m_code):
            if i < len(rows):
                r = rows[i]
                base_pass = (r.get("base_status") == "pass")
                plus_pass = (r.get("plus_status") == "pass")
            else:
                base_pass = plus_pass = False
            correctness.append([base_pass])
            execution_result.append([
                "pass" if base_pass else "fail"
            ])
            per_cand.append({"base": base_pass, "plus": plus_pass})
        item["correctness"] = correctness
        item["execution_result"] = execution_result
        item.setdefault("step_map", [])
        item["evalplus_per_task"] = per_cand

    # Stash top-level summary on every item so rl_code_reward can pick it
    # up without re-reading the JSON.
    for item in data:
        item["evalplus_pass_at_k"] = {
            "base": base_p1,
            "plus": plus_p1,
        }

    return data


def get_config():
    cli_conf   = OmegaConf.from_cli()
    yaml_conf  = OmegaConf.load(cli_conf.config)
    return OmegaConf.merge(yaml_conf, cli_conf)


from concurrent.futures import as_completed

import textwrap

def _run_many_pipe(snippet: str, tests: list[str], conn):
    import textwrap
    results = []
    try:
        ns = {}
        exec(textwrap.dedent(snippet), ns, ns)
        for stmt in tests:
            try:
                exec(stmt, ns, ns)
                results.append(True)
            except SystemExit:
                results.append(True)
            except Exception:
                results.append(False)
        conn.send(results)
    except SystemExit:
        conn.send([True] * len(tests))
    except Exception:
        conn.send([False] * len(tests))
    finally:
        try: conn.close()
        except Exception: pass


def _check_snippet_many(snippet: str, tests: list[str], t_limit: int,
                        spawn_slack: float = 2.0) -> list[bool]:
    import time, multiprocessing as mp
    ctx = mp.get_context("spawn") 
    parent_conn, child_conn = ctx.Pipe(duplex=False)
    p = ctx.Process(target=_run_many_pipe, args=(snippet, tests, child_conn), daemon=True)
    p.start()
    child_conn.close()

    deadline = time.monotonic() + t_limit + spawn_slack
    res = None
    try:
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            wait = remaining if remaining < 0.05 else 0.05
            if parent_conn.poll(wait):
                try:
                    res = parent_conn.recv()
                except EOFError:
                    res = None
                break
            if not p.is_alive():
                if parent_conn.poll(0.05):
                    try:
                        res = parent_conn.recv()
                    except EOFError:
                        res = None
                break

        if res is None and parent_conn.poll(0.05):
            try:
                res = parent_conn.recv()
            except EOFError:
                res = None

        if res is None:
            if p.is_alive():
                p.terminate()
            res = [False] * len(tests)
    finally:
        try: p.join(timeout=0.5)
        except Exception: pass
        try: parent_conn.close()
        except Exception: pass

    return [bool(x) for x in res]

from concurrent.futures import ThreadPoolExecutor, as_completed

def evaluate_function_dataset(data: list[dict], n_workers: int | None = None):
    import os
    n_cpu = os.cpu_count() or 4
    n_workers = max(1, int(n_workers)) if n_workers is not None else n_cpu

    for item in data:
        m_code = len(item["extracted_output"])
        m_test = len(item["test_list"])
        item["execution_result"] = [[None]  * m_test for _ in range(m_code)]
        item["correctness"]      = [[False] * m_test for _ in range(m_code)]
        item.setdefault("step_map", [])

    tasks = []
    for idx, item in enumerate(data):
        t_limit = item.get("test_time_limit", 1)
        tests   = item["test_list"]
        for i, snippet in enumerate(item["extracted_output"]):
            tasks.append((idx, i, snippet, tests, t_limit))

    futures = {}
    from tqdm.auto import tqdm
    with ThreadPoolExecutor(max_workers=n_workers) as pool, \
        tqdm(total=len(tasks)*len(data[0]["test_list"]), desc=f"Function tests ({n_workers} threads)",
            dynamic_ncols=True, mininterval=0.1, miniters=1) as pbar:

        for idx, i, snippet, tests, t_limit in tasks:
            fut = pool.submit(_check_snippet_many, snippet, tests, t_limit)
            futures[fut] = (idx, i)

        for fut in as_completed(futures):
            idx, i = futures[fut]
            try:
                ok_list = fut.result()
            except Exception:
                ok_list = [False] * len(data[idx]["test_list"])

            for j, ok in enumerate(ok_list):
                data[idx]["execution_result"][i][j] = bool(ok)
                data[idx]["correctness"][i][j]      = bool(ok)
                pbar.update(1)

    return data




def worker_stdio(script, input_val, output_queue):
    # Create an iterator over the input lines.
    input_lines = iter(input_val.splitlines())

    # Override the input() function in the exec context.
    def fake_input(prompt=""):
        try:
            return next(input_lines)
        except StopIteration:
            raise EOFError("No more input")
    
    # Redirect sys.stdout to capture printed output.
    stdout_capture = io.StringIO()
    original_stdout = sys.stdout
    original_stdin = sys.stdin  # Save original stdin
    sys.stdout = stdout_capture
    sys.stdin = io.StringIO(input_val)  # Simulate stdin with input_val

    context = {
        "__name__": "__main__",   # Ensures that `if __name__ == "__main__": ...` will fire
        "input": fake_input
    }

    try:
        exec(script, context)
        printed_output = stdout_capture.getvalue()
        output_queue.put(printed_output)

    except SystemExit:
        printed_output = stdout_capture.getvalue()
        output_queue.put(printed_output)

    except Exception as e:
        output_queue.put(f"error: {e}")

    finally:
        sys.stdout = original_stdout
        sys.stdin = original_stdin



def run_scripts_with_timeout(scripts, inputs, time_limits, worker):
    results = [None] * len(scripts)
    processes = []
    queues = []
    deadlines = []

    for i in range(len(scripts)):
        q = mp.Queue()
        p = mp.Process(target=worker, args=(scripts[i], inputs[i], q))
        processes.append(p)
        queues.append(q)
        p.start()
        deadlines.append(time.time() + time_limits[i])

    while any(p.is_alive() for p in processes):
        now = time.time()
        for i, p in enumerate(processes):
            if p.is_alive() and now >= deadlines[i]:
                p.terminate()
                results[i] = "Timeout Error"
        time.sleep(0.001)

    for i, p in enumerate(processes):
        if results[i] is None:
            try:
                results[i] = queues[i].get_nowait()
            except Exception as e:
                results[i] = f"Execution Error: {e}"

    return results

def test_if_eq(x, y):  
    return " ".join(x.split()) == " ".join(y.split())

def get_chunk_indices(n, num_chunks):
    size, rem = divmod(n, num_chunks)
    idx, start = [], 0
    for i in range(num_chunks):
        extra = 1 if i < rem else 0
        end   = start + size + extra
        idx.append((start, end)); start = end
    return idx







from tqdm import tqdm 

def run_scripts_with_chunk(code_list, test_input_list, time_limit_list,
                           worker, num_chunks):
    chunks = get_chunk_indices(len(code_list), num_chunks)

    exe_results = []
    pbar = tqdm(total=len(code_list), desc=f"STDIO tests ({num_chunks} ch)")

    for start, end in chunks:
        sub_code_list       = code_list[start:end]
        sub_test_input_list = test_input_list[start:end]
        sub_time_limit_list = time_limit_list[start:end]

        sub_exe_results = run_scripts_with_timeout(
            sub_code_list,
            sub_test_input_list,
            sub_time_limit_list,
            worker
        )
        exe_results.extend(sub_exe_results)
        pbar.update(end - start)   

    pbar.close()             
    return exe_results


def evaluate_stdio_dataset(data: list[dict], num_chunks: int):
    
    idx_code, idx_case = [], []
    code_list, inp_list, tl_list = [], [], []

    for idx, item in enumerate(data):
        tl = item.get("test_time_limit", 1)
        m_code = len(item["extracted_output"])
        m_case = len(item["test_input"])

        data[idx]["execution_result"] = [[] for _ in range(m_code)]
        data[idx]["correctness"] = [[] for _ in range(m_code)]
        item.setdefault("step_map",           [])

        for c_idx, code in enumerate(item["extracted_output"]):
            for k in range(m_case):
                idx_code.append((idx, c_idx))  
                idx_case.append(k)      
                code_list.append(code)
                inp_list.append(item["test_input"][k])
                tl_list.append(tl)


    exe_results = run_scripts_with_chunk(
        code_list, inp_list, tl_list, worker_stdio, num_chunks
    )

    for i, res in enumerate(exe_results):
        idx, c_idx = idx_code[i]
        k          = idx_case[i]
        item       = data[idx]


        while len(item["execution_result"][c_idx]) < k + 1:
            item["execution_result"][c_idx].append("")
            item["correctness"][c_idx].append(False)
        item["execution_result"][c_idx][k] = res
        exp_out = item["test_output"][k]
        item["correctness"][c_idx][k]      = test_if_eq(res, exp_out)

    return data





def main():
    config          = get_config()
    project_name = config.experiment.project
    num_node = config.experiment.num_node
    node_index = config.experiment.node_index

    if config.experiment.current_epoch == 1:
        pretrained_model = config.model.pretrained_model
    else:
        pretrained_model = "../" + project_name + "/ckpt/" + config.model.optimized_name

    if config.experiment.function == "train":
        dataset = config.dataset.train_dataset
        outputs_name = "rl-" + pretrained_model.replace("/", ".") + "-" + dataset
        
    elif config.experiment.function == "evaluation":
        dataset = config.evaluation.eval_dataset
        outputs_name = "eval-" + pretrained_model.replace("/", ".") + "-" + dataset

    if num_node > 1:
        file_name    = f"../{project_name}/temp_data/outputs-{node_index}-{outputs_name}.json"
    else:
        file_name    = f"../{project_name}/temp_data/outputs-{outputs_name}.json"

    with open(file_name, 'r') as f:
        data = json.load(f)

    # --- 0) evalplus-delegated datasets (HumanEval+/MBPP+) ---
    # Detect by dataset name. These bypass the in-process function/stdio
    # runners entirely and defer to `evalplus.evaluate` via subprocess.
    evalplus_name = _EVALPLUS_DATASETS.get(dataset)
    if evalplus_name is not None:
        work_dir = os.path.join(
            f"../{project_name}/temp_data",
            f"evalplus_{evalplus_name}_{config.experiment.current_epoch}",
        )
        data = evaluate_evalplus_dataset(data, evalplus_name, work_dir=work_dir)
    elif dataset in _LCB_DATASETS:
        # LiveCodeBench v5/v6: in-process scoring via reward.lcb.
        # Keep worker count modest — LCB uses pyext.RuntimeModule + a SIGALRM
        # timeout, and with too many concurrent workers (e.g. 128) the alarms
        # collide and most tests report -1 (timeout) instantly. 4 matches the
        # SDAR / LCB-official default and reproduces paper numbers exactly.
        timeout = int(OmegaConf.select(config, "execute.lcb_timeout", default=6))
        nproc = int(OmegaConf.select(config, "execute.lcb_num_process", default=4))
        data = evaluate_lcb_dataset(data, timeout=timeout,
                                    num_process_evaluate=nproc)
    else:
        func_items  = [itm for itm in data if itm.get("test_method","function") == "function"]
        stdio_items = [itm for itm in data if itm.get("test_method") == "stdio"]

        # --- 1) function ---
        if func_items:
            updated_func = evaluate_function_dataset(func_items, n_workers=config.execute.num_chunk)
            func_iter = iter(updated_func)
            for i,it in enumerate(data):
                if it.get("test_method","function") == "function":
                    data[i] = next(func_iter)


        # --- 2) stdio ---
        if stdio_items:
            total_scripts = sum(len(it["extracted_output"]) for it in stdio_items)
            num_chunks    = max(1, math.ceil(total_scripts / config.execute.num_chunk))
            updated_stdio = evaluate_stdio_dataset(stdio_items, num_chunks=num_chunks)
            it_stdio = iter(updated_stdio)
            for i, it in enumerate(data):
                if it.get("test_method") == "stdio":
                    data[i] = next(it_stdio)

    os.makedirs(os.path.dirname(file_name), exist_ok=True)
    with open(file_name, "w", encoding="utf-8", errors="surrogatepass") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    

    

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
