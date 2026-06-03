from dataclasses import dataclass, field
from typing import List
import torch

from jetengine_ext.engine.sequence import RunType

@dataclass
class Context:
    run_type: RunType | None = None
    cu_seqlens_q: torch.Tensor | None = None
    cu_seqlens_k: torch.Tensor | None = None
    max_seqlen_q: int = 0
    max_seqlen_k: int = 0
    slot_mapping: torch.Tensor | None = None
    context_lens: torch.Tensor | None = None
    block_tables: torch.Tensor | None = None
    is_last_denoise_step: List[bool] = field(default_factory=lambda: [False])
    block_length: int = 4

_CONTEXT = Context()

# JetEngine tensor-parallel dimensions. Set by ModelRunner before model
# construction so that model code uses the JetEngine tp size (not the
# potentially-larger training world size from accelerate/DeepSpeed).
_JE_TP_SIZE: int = 1
_JE_TP_RANK: int = 0

def get_je_tp() -> tuple[int, int]:
    """Return (tp_size, tp_rank) for JetEngine tensor parallelism."""
    return _JE_TP_SIZE, _JE_TP_RANK

def set_je_tp(tp_size: int, tp_rank: int):
    global _JE_TP_SIZE, _JE_TP_RANK
    _JE_TP_SIZE = tp_size
    _JE_TP_RANK = tp_rank

def get_context():
    return _CONTEXT

def set_context(run_type, cu_seqlens_q=None, cu_seqlens_k=None, max_seqlen_q=0, max_seqlen_k=0, slot_mapping=None, context_lens=None, block_tables=None, is_last_denoise_step=[False], block_length=4):
    global _CONTEXT
    _CONTEXT = Context(run_type, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, slot_mapping, context_lens, block_tables, is_last_denoise_step, block_length)

def reset_context():
    global _CONTEXT
    _CONTEXT = Context()
