import os
import torch
from dataclasses import dataclass
from transformers import AutoConfig


@dataclass
class Config:
    model: str
    max_num_batched_tokens: int = 16384
    max_num_seqs: int = 512
    max_model_len: int = 4096
    gpu_memory_utilization: float = 0.5
    tensor_parallel_size: int = 1
    enforce_eager: bool = False
    hf_config: AutoConfig | None = None
    eos: int = -1
    kvcache_block_size: int = 256
    num_kvcache_blocks: int = -1
    mask_token_id: int = -1
    block_length: int = 4
    arm_shift: bool = False

    def __post_init__(self):
        assert os.path.isdir(self.model)
        assert self.kvcache_block_size % 256 == 0
        assert 1 <= self.tensor_parallel_size <= 8
        try:
            self.hf_config = AutoConfig.from_pretrained(self.model, trust_remote_code=True)
        except (ImportError, OSError):
            self.hf_config = AutoConfig.from_pretrained(self.model, trust_remote_code=False)
        if self.hf_config.torch_dtype is None:
            self.hf_config.torch_dtype = torch.bfloat16
        elif isinstance(self.hf_config.torch_dtype, str):
            self.hf_config.torch_dtype = getattr(torch, self.hf_config.torch_dtype)
        self.max_model_len = min(self.max_model_len, self.hf_config.max_position_embeddings)
        assert self.max_num_batched_tokens >= self.max_model_len
        assert self.mask_token_id != -1, "Mask token ID must be set"
