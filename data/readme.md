# Data Setup

The OPDLM datasets are split across two Hugging Face datasets in the
[`divelab/opdlm`](https://huggingface.co/collections/divelab/opdlm) collection:

```bash
# Evaluation — 19 of the 20 paper benchmarks
huggingface-cli download divelab/opdlm_eval_data --local-dir data/ --repo-type dataset

# Training — opdlm_train.json (61,816-row mix of math/code/STEM/chat)
huggingface-cli download divelab/opdlm_train_data --local-dir data/ --repo-type dataset
```

## Datasets not in the OPDLM collection

### Codeforces (paper benchmark)

Built locally from [`open-r1/codeforces`](https://huggingface.co/datasets/open-r1/codeforces):

```bash
python data/prepare_codeforces.py
```

Writes `data/Codeforces.json` (377 problems, schema matches
`LiveCodeBench.json` so the stdio scoring path in `reward/rl_execute.py`
runs unchanged).

### DAPO_Math_17k (used by `scripts/post_train_dapo/*`)

```bash
huggingface-cli download BytedTsinghua-SIA/DAPO-Math-17k \
    --local-dir data/ --repo-type dataset
```

## Where the JSONs go

`pure_inference/eval.py` and `rl.py` both read datasets via
`data/<DATASET_NAME>.json`. Dataset names are registered in
[`eval_utils.py`](../eval_utils.py) under `DATASET_CONFIGS` — that's the
authoritative list of what's wired up.
