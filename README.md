<div align="center">

# OPDLM

### Data-Efficient Autoregressive-to-Diffusion Language Models via On-Policy Distillation

[![arXiv](https://img.shields.io/badge/arXiv-XXXX.XXXXX-b31b1b.svg)](https://arxiv.org/abs/XXXX.XXXXX)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Collection-yellow)](https://huggingface.co/collections/divelab/opdlm)

</div>

Implementation for **"Data-Efficient Autoregressive-to-Diffusion Language Models via On-Policy Distillation"**.

OPDLM is an efficient, on-policy method for converting a pre-trained autoregressive LM into a block-diffusion language model.

All data and models for this release live in the
[`divelab/opdlm`](https://huggingface.co/collections/divelab/opdlm)
Hugging Face collection.

---

## 1. Environment

The pipeline was developed against Python 3.10 / CUDA 12.4-12.8 / PyTorch
2.6.0+cu124. flash-attn must be installed **after** torch with
`--no-build-isolation`, otherwise it pulls its own torch and breaks the env.

```bash
conda create -n opdlm python=3.10.19 -y
conda activate opdlm

# torch first
pip install torch==2.6.0+cu124 --index-url https://download.pytorch.org/whl/cu124

# everything else
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu124

# flash-attn last
pip install flash-attn==2.7.4.post1 --no-build-isolation
```

If `nvcc` and the torch CUDA version disagree (e.g., driver CUDA 12.8 but torch
built for 12.4), DeepSpeed will refuse to JIT-compile its CPU Adam op. Set
`export DS_SKIP_CUDA_CHECK=1` to bypass the check — torch is forward-compat
across cu12.x minor versions.

---

## 2. Data

The OPDLM datasets are split across two Hugging Face datasets in the
[`divelab/opdlm`](https://huggingface.co/collections/divelab/opdlm) collection:

```bash
# Evaluation — 19 of the 20 paper benchmarks
huggingface-cli download divelab/opdlm_eval_data --local-dir data/ --repo-type dataset

# Training — opdlm_train.json, 61,816 rows (math/code/STEM/chat mix)
huggingface-cli download divelab/opdlm_train_data --local-dir data/ --repo-type dataset
```

`opdlm_train.json` is the OPDLM training corpus — a 61,816-row mix of
code (TACO / KodCode-Light-RL / AceCode), math (DAPO, Nemotron-v2-Math), STEM
(Nemotron-v2-STEM) and chat (Nemotron-v2-Chat).

One paper dataset is **not** in the OPDLM collection and needs a separate step:

```bash
# Codeforces (paper eval) — built from open-r1/codeforces (verifiable subset)
python data/prepare_codeforces.py
```

The math post-training data (`MATH_train_traceRL.json`, Hendrycks MATH
level 3-5, ~8K hard tasks, following the traceRL setup) is bundled in
the `divelab/opdlm_eval_data` repo, and is already downloaded by the
step above.

See [`data/readme.md`](data/readme.md) for per-dataset details.

---

## 3. Models

OPDLM trains a BD3LM-architecture student initialised from a Qwen3 ARM whose
attention has been switched from causal to bidirectional. Two artefacts are
needed:

| Role | Hugging Face repo |
|------|-------------------|
| Teacher (ARM)                      | [`Qwen/Qwen3-4B`](https://huggingface.co/Qwen/Qwen3-4B), [`Qwen/Qwen3-8B`](https://huggingface.co/Qwen/Qwen3-8B) (and `Qwen3-0.6B` / `Qwen3-1.7B` for the Table 6 smaller scales) |
| Student init (A2D-converted Qwen3) | [`divelab/Qwen3-4B-a2d-init`](https://huggingface.co/divelab/Qwen3-4B-a2d-init), [`divelab/Qwen3-8B-a2d-init`](https://huggingface.co/divelab/Qwen3-8B-a2d-init) — both in the [`divelab/opdlm`](https://huggingface.co/collections/divelab/opdlm) collection |

For the smaller-scale init models (`Qwen3-{0.6B,1.7B}-a2d-init`), or if you
want to rebuild any init from scratch, regenerate locally:

```bash
python convert_qwen_to_bd3lm.py    # edit SRC_MODEL / OUTPUT_DIR at the top
```

---

## 4. Training

Training runs through `rl.py` with the BD3LM config:

```bash
python rl.py config=configs/rl_bd3lm.yaml \
    model.pretrained_model=$HF_HOME/<a2d-init> \
    model.teacher_model=$HF_HOME/<Qwen3-teacher>           \
    dataset.train_dataset=opdlm_train
```

All training runs reported in the paper use 1 node × 8 NVIDIA H200 GPUs.

Reference launchers mirror the exact hyperparameters from Table 10 of the
paper: block_size=4, denoising_steps=4, forward KL, one-state-per-block,
LR 1e-5→1e-6 cosine, batch=8, tasks/rollout=128, max_rollout 100→4000 over
the first 100 steps. The KL is computed over the **full vocabulary** at
the 0.6B / 1.7B scales and restricted to the teacher's **top-16 tokens**
(Nemotron-style sparse KL, `training.top_k_logits=16`) at the 4B / 8B scales.

| Stage | Launcher |
|-------|----------|
| OPDLM 0.6B / 1.7B (full-vocab KL, opdlm_train)        | `scripts/general_pre_train/BD3LM_{06B,17B}.sh` |
| OPDLM 4B / 8B (top-16 sparse KL, opdlm_train)         | `scripts/general_pre_train/BD3LM_{4B,8B}.sh` |
| OPDLM-MATH 4B / 8B, non-thinking (MATH_train_traceRL) | `scripts/post_train_math/BD3LM_MATH_{4B,8B}.sh` |
| OPDLM-MATH 4B / 8B, thinking-on (MATH_train_traceRL)  | `scripts/post_train_math/BD3LM_MATH_{4B,8B}_thinking.sh` |

Dynamic-threshold remasking is an **inference-time** choice (see Section 5);
The launchers above all train with `dynamic_threshold_schedule.enabled=False`.

Each launcher hardcodes its author's `$HF_HOME` path — edit `DATA_PATH`,
`STUDENT`, `TEACHER`, and the SBATCH header to match your cluster before
submitting.

The relevant accelerate configs (single-node, 1/2/4/8 GPU, ZeRO-3) live in
`accelerate_configs/`.

---

## 5. Evaluation

`pure_inference/eval.py` is the canonical evaluation entry point. It supports
both BD3LM (diffusion) and Qwen (autoregressive) backbones, with static or
dynamic-threshold remasking.

```bash
python pure_inference/eval.py \
    --models    <path-to-your-trained-opdlm-ckpt> \
    --model_bases bd3lm \
    --datasets  HumanEval MBPP MATH500 GSM8K AIME2024 \
    --max_token 2048 \
    --remasking_strategy low_confidence_static \
    --dynamic_threshold 0.9 \
    --temperature 0.0 \
    --block_size 4 --denoising_steps_per_block 4 \
    --out_dir pure_inference/results
```

The trained OPDLM paper checkpoints will land in the
[`divelab/opdlm`](https://huggingface.co/collections/divelab/opdlm) collection
when released. Until then, train your own via §4 and point `--models` at
`experiments/<run>/ckpt/optimized`.

Convenience wrappers for each model family are in `pure_inference/`:

| Wrapper | Purpose |
|---------|---------|
| `run_eval_greedy_4B_base.sh`, `run_eval_greedy_8B_base.sh` | OPDLM at 4B / 8B (paper Table 1) |
| `run_eval_greedy_06B_base.sh`, `run_eval_greedy_17B_base.sh` | Smaller-scale ablation (paper Table 6) |
| `run_eval_greedy_4B_qwen.sh` | Qwen3-4B autoregressive baseline |
| `run_eval_greedy_4B_base_dynamic.sh`, `..._fix_thres.sh` | Dynamic-threshold sweeps |
| `run_eval_greedy_math.sh` | Math-only quick eval |

Each wrapper edits `MODELS`, `MODEL_BASES`, `DATASETS`, and `TAG` near the top
— set those to point at your downloaded checkpoints and HF cache, then run:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash pure_inference/run_eval_greedy_4B_base.sh
# multi-GPU data-parallel: pass GPUS=0,1,...,7 — each shard runs on its own GPU
GPUS=0,1,2,3,4,5,6,7 bash pure_inference/run_eval_greedy_4B_base.sh
```

Results land in `pure_inference/results/<model>_<dataset>_<tag>/`. For
LiveCodeBench v6 at 16k tokens (Table 1, 8B), pass `--num_chunks 4` to
shard the generation across GPUs.

---

## 6. Reproducing paper tables

| Paper table | Stage | Entry point |
|-------------|-------|-------------|
| T1 — Main (4B/8B)              | Train + eval | `general_pre_train/BD3LM_{4B,8B}.sh` → `pure_inference/run_eval_greedy_{4B,8B}_base.sh` |
| T2 — Zero-shot think           | Eval         | `run_eval_greedy_{4B,8B}_base.sh --enable_thinking` |
| T3 — Multilingual              | Eval         | `run_eval_greedy_{4B,8B}_base.sh` on MMMLU-lite / INCLUDE-lite / MT-AIME2024 / MLogiQA |
| T5 — OPDLM-MATH vs TraDo       | Train + eval | `post_train_math/BD3LM_MATH_{4B,8B}.sh` (non-thinking) and `BD3LM_MATH_{4B,8B}_thinking.sh` (thinking-on) |
| T6 — Smaller scales (0.6B/1.7B) | Train + eval | `general_pre_train/BD3LM_{06B,17B}.sh` → `run_eval_greedy_{06B,17B}_base.sh` |
| Figures 3-5 — decoding sweeps   | Eval        | `run_eval_greedy_4B_base_{dynamic,fix_thres}.sh` |

---

## 7. Citation

If you use this code or the OPDLM models, please cite the preprint:

```bibtex
@misc{su2026opdlm,
  title  = {Data-Efficient Autoregressive-to-Diffusion Language Models via On-Policy Distillation},
  author = {Su, Xingyu and Helwig, Jacob and Parashar, Shubham and Chagi, Atharv and
            Jotsna, Lakshmi and Caverlee, James and Kalathil, Dileep and Ji, Shuiwang},
  year   = {2026},
  note   = {Preprint}
}
```

---

## 8. Acknowledgements

This codebase builds on two prior open-source releases:

- **SDAR** ([JetAstra/SDAR](https://github.com/JetAstra/SDAR)) — block-diffusion
  language models from pre-trained autoregressive models. 
- **TraceRL** ([Gen-Verse/dLLM-RL](https://github.com/Gen-Verse/dLLM-RL),
  [paper](https://arxiv.org/abs/2509.06949)) — RL training framework for
  diffusion LMs. 
