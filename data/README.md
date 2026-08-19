---
task_categories:
- text-generation
---

This repository contains the evaluation data for **OPDLM (On-Policy Diffusion Language Model)**, as presented in the paper [Data-Efficient Autoregressive-to-Diffusion Language Models via On-Policy Distillation](https://huggingface.co/papers/2606.06712).

- **Project Page:** [https://opdlm.vercel.app/](https://opdlm.vercel.app/)
- **GitHub Repository:** [https://github.com/divelab/OPDLM](https://github.com/divelab/OPDLM)

### Dataset Summary

OPDLM is an efficient, on-policy method for converting pre-trained autoregressive language models (ARLMs) into block-diffusion language models (DLMs). The datasets provided include:

<!-- - **Training Data (`opdlm_train.json`):** A 61,816-row corpus consisting of a mix of math (DAPO, Nemotron-v2-Math), code (TACO, KodCode-Light-RL, AceCode), STEM (Nemotron-v2-STEM), and chat data (Nemotron-v2-Chat). -->
- **Evaluation Data:** Includes all 20 evaluation benchmarks used in the paper, such as HumanEval, MBPP, Codeforces, MATH500, GSM8K, and AIME2024.

### Citation

```bibtex
@misc{su2026opdlm,
      title={Data-Efficient Autoregressive-to-Diffusion Language Models via On-Policy Distillation},
      author={Xingyu Su and Jacob Helwig and Shubham Parashar and Atharv Chagi and Lakshmi Jotsna and Degui Zhi and James Caverlee and Dileep Kalathil and Shuiwang Ji},
      year={2026},
      eprint={2606.06712},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2606.06712},
}
```