# Configs

`rl_bd3lm.yaml` is the only config in the release — the **OPDLM training
config.** Block-diffusion student, Qwen3 teacher, forward KL, block_size=4,
denoising_steps=4. Matches Table 10 of the paper.

It is consumed by every training launcher under `scripts/`
(`general_pre_train/BD3LM_{06B,17B,4B,8B}.sh`,
`post_train_dapo/**/BD3LM_DAPO_*.sh`).

Edit dataset name, model paths, GPU count, and sweep knobs at the top of the
launcher (or via `python rl.py config=configs/rl_bd3lm.yaml KEY=VALUE ...`
overrides) before running.

## Entry points

```bash
# Training
python rl.py config=configs/rl_bd3lm.yaml \
    model.pretrained_model=$HF_HOME/<a2d-init> \
    model.teacher_model=$HF_HOME/<Qwen3-teacher>

# Evaluation (CLI-driven)
python pure_inference/eval.py --models <ckpt> --model_bases bd3lm \
    --datasets MATH500 GSM8K HumanEval MBPP
```
