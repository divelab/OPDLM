"""
Convert Qwen3-0.6B (ARM) → untrained BD3LM (A2D architecture, same weights).

Only changes:
  - config.model_type: "qwen3" → "a2d-qwen3"
  - Attention: causal → bidirectional (handled by A2D model class)
  - vocab_size: resized to 151936 if needed (GPU-aligned padding)

Weights are preserved exactly from the original Qwen3 model.

Usage:
    python convert_qwen_to_bd3lm.py
"""

import os
import json
import shutil
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

# ── Paths ──
SRC_MODEL = "pretrained_models/Qwen/Qwen3-0.6B"
REF_BD3LM = "pretrained_models/BD3LM/Qwen3-0.6B-diffusion-bd3lm-v0.1"
OUTPUT_DIR = "pretrained_models/BD3LM/Qwen3-0.6B-a2d-init"

# Target vocab_size (must match reference BD3LM for compatibility)
TARGET_VOCAB_SIZE = 151936


def main():
    print(f"Loading source model: {SRC_MODEL}")
    src_model = AutoModelForCausalLM.from_pretrained(SRC_MODEL, torch_dtype=torch.bfloat16)
    src_tokenizer = AutoTokenizer.from_pretrained(SRC_MODEL)

    src_vocab = src_model.config.vocab_size
    print(f"Source vocab_size: {src_vocab}, target: {TARGET_VOCAB_SIZE}")

    # Resize embeddings if needed (zero-pads new rows)
    if src_vocab != TARGET_VOCAB_SIZE:
        src_model.resize_token_embeddings(TARGET_VOCAB_SIZE)
        print(f"Resized embeddings: {src_vocab} → {TARGET_VOCAB_SIZE}")

    # Save weights
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    src_model.save_pretrained(OUTPUT_DIR)
    print(f"Saved model weights to {OUTPUT_DIR}")

    # ── Build BD3LM-compatible config.json ──
    # Load reference BD3LM config as the template
    with open(os.path.join(REF_BD3LM, "config.json")) as f:
        ref_config = json.load(f)

    # Load source config and override structural fields from it
    src_config = src_model.config.to_dict()

    # Start from reference config (has correct model_type, auto_map, architectures)
    # but update all model architecture fields from source
    arch_fields = [
        "attention_bias", "attention_dropout", "bos_token_id", "eos_token_id",
        "head_dim", "hidden_act", "hidden_size", "initializer_range",
        "intermediate_size", "max_position_embeddings", "max_window_layers",
        "num_attention_heads", "num_hidden_layers", "num_key_value_heads",
        "rms_norm_eps", "rope_scaling", "rope_theta", "sliding_window",
        "tie_word_embeddings", "use_cache", "use_sliding_window",
    ]
    for field in arch_fields:
        if field in src_config:
            ref_config[field] = src_config[field]
    if "layer_types" in src_config:
        ref_config["layer_types"] = src_config["layer_types"]

    # Ensure BD3LM-specific fields
    ref_config["model_type"] = "a2d-qwen3"
    ref_config["vocab_size"] = TARGET_VOCAB_SIZE
    ref_config["dtype"] = "bfloat16"
    ref_config["pad_token_id"] = ref_config.get("pad_token_id", ref_config["bos_token_id"])

    with open(os.path.join(OUTPUT_DIR, "config.json"), "w") as f:
        json.dump(ref_config, f, indent=2)
    print("Wrote config.json")

    # ── Copy modeling_qwen3.py from reference BD3LM (needed for trust_remote_code) ──
    modeling_src = os.path.join(REF_BD3LM, "modeling_qwen3.py")
    if os.path.exists(modeling_src):
        shutil.copy2(modeling_src, os.path.join(OUTPUT_DIR, "modeling_qwen3.py"))
        print("Copied modeling_qwen3.py from reference BD3LM")

    # ── Save tokenizer ──
    src_tokenizer.save_pretrained(OUTPUT_DIR)
    print("Saved tokenizer")

    print(f"\nDone! Untrained BD3LM saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
