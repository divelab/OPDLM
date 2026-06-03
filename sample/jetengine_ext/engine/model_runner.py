import pickle
import os
import socket
import torch
import torch.distributed as dist
from multiprocessing.synchronize import Event
from multiprocessing.shared_memory import SharedMemory

from jetengine_ext.config import Config
from jetengine_ext.engine.sequence import Sequence, RunType, SequenceStatus
from jetengine_ext.models.sdar import SDARForCausalLM
from jetengine_ext.models.sdar_moe import SDARMoeForCausalLM
from jetengine_ext.utils.context import set_context, get_context, reset_context, set_je_tp
from jetengine_ext.utils.loader import load_model


def _find_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


class ModelRunner:

    def __init__(self, config: Config, rank: int, event: Event | list[Event]):
        self.config = config
        hf_config = config.hf_config
        self.block_size = config.kvcache_block_size
        self.enforce_eager = config.enforce_eager
        self.world_size = config.tensor_parallel_size
        self.rank = rank
        self.event = event

        if dist.is_initialized():
            # Running inside an existing distributed context (e.g. accelerate launch).
            # Reuse the existing process group; do NOT init or destroy it.
            # Don't call torch.cuda.set_device — the parent context already set it.
            self._owns_process_group = False
        else:
            jetengine_port = int(os.environ.get("JETENGINE_PORT", 0)) or _find_free_port()
            dist.init_process_group("nccl", f"tcp://localhost:{jetengine_port}", world_size=self.world_size, rank=rank)
            self._owns_process_group = True
            torch.cuda.set_device(rank)
        set_je_tp(self.world_size, rank)
        default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(hf_config.torch_dtype)
        torch.set_default_device("cuda")
        if "sdar" in hf_config.model_type and "moe" in hf_config.model_type:
            self.model = SDARMoeForCausalLM(hf_config)
        elif "sdar" in hf_config.model_type or hf_config.model_type in ("qwen3", "qwen2", "a2d-qwen3"):
            self.model = SDARForCausalLM(hf_config)
        else:
            raise ValueError(f"Unsupported model type: {hf_config.model_type}")
        # When running inside an existing distributed context (e.g. accelerate
        # launch with world_size > tp), pass explicit tp dimensions so the
        # loader doesn't mistake the training world_size for tensor-parallel.
        _tp_kwargs = {}
        if not self._owns_process_group:
            _tp_kwargs = dict(tp_size=self.world_size, tp_rank=rank)
        load_model(self.model, config.model, **_tp_kwargs)
        # Sampler is removed from here
        self.warmup_model()
        self.allocate_kv_cache()
        # CUDA graph capture for block diffusion is complex and omitted for this example
        if not self.enforce_eager:
            self.capture_cudagraph()
        torch.set_default_device("cpu")
        torch.set_default_dtype(default_dtype)

        if self.world_size > 1:
            je_port = os.environ.get("JETENGINE_PORT", "0")
            shm_name = f"jetengineshm_{je_port}"
            if rank == 0:
                # Create (and clean up stale) shared memory
                try:
                    self.shm = SharedMemory(name=shm_name, create=True, size=2**22)
                except FileExistsError:
                    try:
                        stale = SharedMemory(name=shm_name)
                        stale.close()
                        stale.unlink()
                    except FileNotFoundError:
                        pass
                    self.shm = SharedMemory(name=shm_name, create=True, size=2**22)
                dist.barrier()
            else:
                dist.barrier()
                self.shm = SharedMemory(name=shm_name)
                self.loop()

    def exit(self):
        if self.world_size > 1:
            self.shm.close()
            if self._owns_process_group:
                dist.barrier()
            if self.rank == 0:
                try:
                    self.shm.unlink()
                except FileNotFoundError:
                    pass
        if not self.enforce_eager:
            del self.graphs, self.graph_pool
        torch.cuda.synchronize()
        if self._owns_process_group:
            dist.destroy_process_group()

    def loop(self):
        while True:
            method_name, args = self.read_shm()
            self.call(method_name, *args)
            if method_name == "exit":
                break

    def read_shm(self):
        assert self.world_size > 1 and self.rank
        self.event.wait()
        n = int.from_bytes(self.shm.buf[0:4], "little")
        method_name, *args = pickle.loads(self.shm.buf[4:n+4])
        self.event.clear()
        return method_name, args

    def write_shm(self, method_name, *args):
        assert self.world_size > 1 and not self.rank
        data = pickle.dumps([method_name, *args])
        n = len(data)
        self.shm.buf[0:4] = n.to_bytes(4, "little")
        self.shm.buf[4:n+4] = data
        for event in self.event:
            event.set()

    def call(self, method_name, *args):
        if self.world_size > 1 and self.rank == 0:
            self.write_shm(method_name, *args)
        method = getattr(self, method_name, None)
        return method(*args)

    def warmup_model(self):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        max_num_batched_tokens, max_model_len = self.config.max_num_batched_tokens, self.config.max_model_len
        num_seqs = min(max_num_batched_tokens // max_model_len, self.config.max_num_seqs)
        seqs = [Sequence([0] * max_model_len, self.config.mask_token_id) for _ in range(num_seqs)]
        self.run(seqs, RunType.PREFILL)
        torch.cuda.empty_cache()
        # Warmup/compile can transiently spike peak memory and overly shrink KV budget.
        torch.cuda.reset_peak_memory_stats()

    def allocate_kv_cache(self):
        config = self.config
        hf_config = config.hf_config
        free, total = torch.cuda.mem_get_info()
        used = total - free
        peak = torch.cuda.memory_stats()["allocated_bytes.all.peak"]
        current = torch.cuda.memory_stats()["allocated_bytes.all.current"]
        num_kv_heads = hf_config.num_key_value_heads // self.world_size
        block_bytes = 2 * hf_config.num_hidden_layers * self.block_size * num_kv_heads * hf_config.head_dim * hf_config.torch_dtype.itemsize

        target_bytes = int(total * config.gpu_memory_utilization - used - peak + current)
        if target_bytes <= 0:
            # Fallback: if total-based budget is exhausted (often due fragmentation or background usage),
            # try reserving part of currently free memory so we can still make progress.
            fallback_bytes = int(free * min(0.95, max(0.1, config.gpu_memory_utilization)) * 0.9)
            target_bytes = fallback_bytes
            print(
                "[kv-cache] total-based budget <= 0; using free-memory fallback: "
                f"free={free}, total={total}, used={used}, peak={peak}, current={current}, "
                f"fallback_bytes={fallback_bytes}",
                flush=True,
            )

        init_blocks = target_bytes // block_bytes
        if init_blocks <= 0:
            if free < block_bytes:
                raise RuntimeError(
                    "Insufficient GPU memory for KV cache allocation: "
                    f"free={free}, block_bytes={block_bytes}, "
                    f"gpu_memory_utilization={config.gpu_memory_utilization}"
                )
            init_blocks = max(1, int((free * 0.9) // block_bytes))

        blocks = init_blocks
        last_oom = None
        while blocks > 0:
            try:
                self.kv_cache = torch.zeros(
                    2,
                    hf_config.num_hidden_layers,
                    blocks,
                    self.block_size,
                    num_kv_heads,
                    hf_config.head_dim,
                )
                config.num_kvcache_blocks = blocks
                break
            except RuntimeError as e:
                if "out of memory" not in str(e).lower():
                    raise
                last_oom = e
                torch.cuda.empty_cache()
                blocks //= 2
        else:
            raise RuntimeError(
                "KV cache allocation failed even after backoff. "
                f"init_blocks={init_blocks}, block_bytes={block_bytes}, free={free}, total={total}"
            ) from last_oom

        if config.num_kvcache_blocks < init_blocks:
            print(
                f"[kv-cache] reduced blocks after OOM backoff: {init_blocks} -> {config.num_kvcache_blocks}",
                flush=True,
            )

        layer_id = 0
        for module in self.model.modules():
            if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
                module.k_cache = self.kv_cache[0, layer_id]
                module.v_cache = self.kv_cache[1, layer_id]
                layer_id += 1

    def prepare_block_tables(self, seqs: list[Sequence]):
        max_len = max(len(seq.block_table) for seq in seqs)
        if max_len == 0: return None
        block_tables = [seq.block_table + [-1] * (max_len - len(seq.block_table)) for seq in seqs]
        return torch.tensor(block_tables, dtype=torch.int32).cuda()

    def prepare_prefill(self, seqs: list[Sequence]):
        input_ids, positions, cu_seqlens_q, slot_mapping, is_last_step = [], [], [0], [], []
        max_seqlen_q = 0
        for seq in seqs:
            seqlen = len(seq)
            input_ids.extend(seq.token_ids)
            positions.extend(range(seqlen))
            cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen)
            max_seqlen_q = max(max_seqlen_q, seqlen)
            is_last_step.append(False)
            # Slot mapping for prefill
            if not seq.block_table:
                continue
            # Slot mapping for prefill
            if not seq.block_table:
                continue
            for i in range(seqlen):
                block_idx = i // self.block_size 
                block_offset = i % self.block_size 
                physical_block_id = seq.block_table[block_idx]
                slot = physical_block_id * self.block_size + block_offset
                slot_mapping.append(slot)

        input_ids = torch.tensor(input_ids, dtype=torch.int64).cuda()
        positions = torch.tensor(positions, dtype=torch.int64).cuda()
        cu_seqlens_q = torch.tensor(cu_seqlens_q, dtype=torch.int32).cuda()
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32).cuda()
        set_context(
            run_type=RunType.PREFILL,
            cu_seqlens_q=cu_seqlens_q, 
            cu_seqlens_k=cu_seqlens_q, 
            max_seqlen_q=max_seqlen_q, 
            max_seqlen_k=max_seqlen_q, 
            slot_mapping=slot_mapping, 
            is_last_denoise_step=is_last_step, # <-- Pass the new flag
            block_length=self.config.block_length
        )
        return input_ids, positions

    def prepare_denoise(self, seqs: list[Sequence]):
        input_ids, positions = [], []
        cached_lens = []
        
        for seq in seqs:
            # The query is the current intermediate block
            q_tokens = seq.intermediate_block_tokens
            q_len = len(q_tokens)
            
            # The context (key/value) is the confirmed part of the sequence
            k_len = len(seq)
            
            input_ids.extend(q_tokens)
            # Positions are global
            positions.extend(range(k_len, k_len + q_len))
            cached_lens.append(k_len)

        input_ids = torch.tensor(input_ids, dtype=torch.int64).cuda()
        positions = torch.tensor(positions, dtype=torch.int64).cuda()
        cached_lens = torch.tensor(cached_lens, dtype=torch.int32).cuda()
        block_tables = self.prepare_block_tables(seqs)
        
        set_context(
            run_type=RunType.DENOISE,
            context_lens=cached_lens,
            block_tables=block_tables,
            block_length=self.config.block_length
        )
        
        return input_ids, positions

    @torch.inference_mode()
    def run_model(self, input_ids: torch.Tensor, positions: torch.Tensor):
        return self.model.compute_logits(self.model(input_ids, positions))

    def run(self, seqs: list[Sequence], run_type: RunType) -> torch.Tensor:
        if run_type == RunType.PREFILL:
            input_ids, positions = self.prepare_prefill(seqs)
        elif run_type == RunType.DENOISE:
            input_ids, positions = self.prepare_denoise(seqs)
        else:
            return None

        logits = self.run_model(input_ids, positions)

        # ARM next-token → current-token alignment (only during denoise, not prefill)
        if run_type == RunType.DENOISE and self.config.arm_shift and logits is not None:
            bl = self.config.block_length
            T, V = logits.shape
            if T % bl == 0:
                logits = logits.view(-1, bl, V).roll(dims=1, shifts=1).reshape(T, V)

        reset_context()
        return logits if self.rank == 0 else None

    @torch.inference_mode()
    def capture_cudagraph(self):
        config = self.config
        hf_config = config.hf_config
        max_bs = min(self.config.max_num_seqs, 256)
        max_global_bs = max_bs * self.config.block_length
        max_num_blocks = (config.max_model_len + self.block_size - 1) // self.block_size
        input_ids = torch.zeros(max_global_bs, dtype=torch.int64)
        positions = torch.zeros(max_global_bs, dtype=torch.int64)
        context_lens = torch.zeros(max_bs, dtype=torch.int32)
        block_tables = torch.zeros(max_bs, max_num_blocks, dtype=torch.int32)
        outputs = torch.zeros(max_global_bs, hf_config.hidden_size)
        self.graph_bs = [1, 2, 4, 8] + list(range(16, max_bs + 1, 16))
        self.graphs = {}
        self.graph_pool = None

        for bs in reversed(self.graph_bs):
            graph = torch.cuda.CUDAGraph()
            set_context(run_type=RunType.DENOISE, context_lens=context_lens[:bs], block_tables=block_tables[:bs], block_length=self.config.block_length)
            global_bs = bs * self.config.block_length
            outputs[:global_bs] = self.model(input_ids[:global_bs], positions[:global_bs])    # warmup
            with torch.cuda.graph(graph, self.graph_pool):
                outputs[:global_bs] = self.model(input_ids[:global_bs], positions[:global_bs])    # capture
            if self.graph_pool is None:
                self.graph_pool = graph.pool()
            self.graphs[bs] = graph
            torch.cuda.synchronize()
            reset_context()

        self.graph_vars = dict(
            input_ids=input_ids,
            positions=positions,
            context_lens=context_lens,
            block_tables=block_tables,
            outputs=outputs,
        )
