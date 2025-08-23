import sys
from pathlib import Path
import torch
import torch.nn as nn
import nvtx
import math
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).parent))
from paged_attention.paged_attention import PagedAttention


class CUDAPagedKVCache:
    def __init__(
            self,
            num_blocks: int,
            block_size: int,
            num_heads: int,
            head_dim: int,
            max_blocks_per_seq: int = 256,
            dtype: torch.dtype = torch.float16,
            device: str = "cuda"
    ):
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.max_blocks_per_seq = max_blocks_per_seq
        self.dtype = dtype
        self.device = device

        with nvtx.annotate("cuda_paged_cache_init", color="purple"):
            # 物理KV缓存块：[num_blocks, block_size, num_heads, head_dim]
            self.key_cache = torch.zeros(
                (num_blocks, block_size, num_heads, head_dim),
                dtype=dtype, device=device
            )
            self.value_cache = torch.zeros(
                (num_blocks, block_size, num_heads, head_dim),
                dtype=dtype, device=device
            )

            # 空闲块池
            self.free_blocks = list(range(num_blocks))
            # 序列到物理块的映射表（PyTorch层面的映射）
            self.sequence_blocks: Dict[int, List[int]] = {}
            self.sequence_lengths: Dict[int, int] = {}

    @nvtx.annotate("allocate_sequence", color="orange")
    def allocate_sequence(self, seq_id: int, initial_length: int, max_length: int = None) -> torch.Tensor:
        """为序列分配物理块并返回block_table"""
        # 如果没有指定最大长度，预留一些额外空间用于decode阶段
        if max_length is None:
            max_length = initial_length * 2  # 预留双倍空间用于生成

        required_blocks = math.ceil(max_length / self.block_size)

        if len(self.free_blocks) < required_blocks:
            raise RuntimeError(f"Not enough free blocks! Need {required_blocks}, have {len(self.free_blocks)}")

        # 分配物理块
        allocated_blocks = []
        for _ in range(required_blocks):
            block_id = self.free_blocks.pop(0)
            allocated_blocks.append(block_id)

        self.sequence_blocks[seq_id] = allocated_blocks
        self.sequence_lengths[seq_id] = initial_length  # 当前实际使用长度

        # 创建block_table tensor
        block_table = torch.full((self.max_blocks_per_seq,), -1, dtype=torch.int32, device=self.device)
        block_table[:len(allocated_blocks)] = torch.tensor(allocated_blocks, dtype=torch.int32, device=self.device)

        return block_table

    @nvtx.annotate("write_kv", color="red")
    def write_kv(
            self,
            seq_id: int,
            start_pos: int,
            keys: torch.Tensor,
            values: torch.Tensor
    ):
        """写入KV数据到物理块"""
        if seq_id not in self.sequence_blocks:
            raise ValueError(f"Sequence {seq_id} not allocated")

        seq_len = keys.shape[0]
        allocated_blocks = self.sequence_blocks[seq_id]

        # 检查是否需要扩展分配的块
        max_pos = start_pos + seq_len
        required_blocks = math.ceil(max_pos / self.block_size)

        if required_blocks > len(allocated_blocks):
            # 需要分配更多块
            additional_blocks_needed = required_blocks - len(allocated_blocks)

            if len(self.free_blocks) < additional_blocks_needed:
                raise RuntimeError(
                    f"Not enough free blocks for sequence {seq_id}! "
                    f"Need {additional_blocks_needed}, have {len(self.free_blocks)}"
                )

            # 分配额外的块
            for _ in range(additional_blocks_needed):
                new_block_id = self.free_blocks.pop(0)
                allocated_blocks.append(new_block_id)

            self.sequence_blocks[seq_id] = allocated_blocks
            print(f"Extended sequence {seq_id} to {len(allocated_blocks)} blocks")

        # 写入每个token的KV
        for i in range(seq_len):
            pos = start_pos + i
            block_idx = pos // self.block_size
            in_block_pos = pos % self.block_size

            if block_idx >= len(allocated_blocks):
                raise ValueError(f"Position {pos} exceeds allocated blocks for sequence {seq_id}")

            physical_block = allocated_blocks[block_idx]

            # 写入K和V: [num_heads, head_dim] -> [1, num_heads, head_dim]
            self.key_cache[physical_block, in_block_pos] = keys[i]
            self.value_cache[physical_block, in_block_pos] = values[i]

    @nvtx.annotate("deallocate_sequence", color="gray")
    def deallocate_sequence(self, seq_id: int):
        """释放序列占用的物理块"""
        if seq_id in self.sequence_blocks:
            # 回收物理块
            for block_id in self.sequence_blocks[seq_id]:
                self.free_blocks.append(block_id)

            del self.sequence_blocks[seq_id]
            del self.sequence_lengths[seq_id]

    def get_block_table(self, seq_id: int) -> torch.Tensor:
        """获取序列的block_table"""
        if seq_id not in self.sequence_blocks:
            raise ValueError(f"Sequence {seq_id} not allocated")

        block_table = torch.full((self.max_blocks_per_seq,), -1, dtype=torch.int32, device=self.device)
        allocated_blocks = self.sequence_blocks[seq_id]
        block_table[:len(allocated_blocks)] = torch.tensor(allocated_blocks, dtype=torch.int32, device=self.device)

        return block_table


class CUDAPagedMultiHeadAttention(nn.Module):
    def __init__(
            self,
            d_model: int,
            num_heads: int,
            block_size: int = 16,
            dtype: torch.dtype = torch.float16,
            device: str = "cuda"
    ):
        super().__init__()

        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.block_size = block_size
        self.dtype = dtype
        self.device = device
        self.scale = 1.0 / math.sqrt(self.head_dim)

        self.W_q = nn.Linear(d_model, d_model, bias=False, dtype=dtype, device=device)
        self.W_k = nn.Linear(d_model, d_model, bias=False, dtype=dtype, device=device)
        self.W_v = nn.Linear(d_model, d_model, bias=False, dtype=dtype, device=device)
        self.W_o = nn.Linear(d_model, d_model, bias=False, dtype=dtype, device=device)

        self.paged_attention = PagedAttention(
            num_heads=num_heads,
            head_dim=self.head_dim,
            block_size=block_size
        )

        self._init_weights()

    def _init_weights(self):
        """初始化权重"""
        for module in [self.W_q, self.W_k, self.W_v, self.W_o]:
            nn.init.xavier_uniform_(module.weight)

    @nvtx.annotate("cuda_mha_prefill", color="lightgreen")
    def forward_prefill(
            self,
            input_embeds: torch.Tensor,  # [batch_size, seq_len, d_model]
            kv_cache: CUDAPagedKVCache,
            seq_ids: List[int]
    ) -> torch.Tensor:
        """
        Prefill阶段：处理完整的prompt序列
        """
        batch_size, seq_len, _ = input_embeds.shape

        with nvtx.annotate("compute_qkv_prefill", color="yellow"):
            Q = self.W_q(input_embeds)  # [batch_size, seq_len, d_model]
            K = self.W_k(input_embeds)
            V = self.W_v(input_embeds)

            Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim)
            K = K.view(batch_size, seq_len, self.num_heads, self.head_dim)
            V = V.view(batch_size, seq_len, self.num_heads, self.head_dim)

        with nvtx.annotate("write_kv_to_cache", color="orange"):
            for batch_idx, seq_id in enumerate(seq_ids):
                # 分配序列空间（如果还未分配）
                # 为decode阶段预留额外空间
                if seq_id not in kv_cache.sequence_blocks:
                    max_expected_length = seq_len + 256  # 预留256个token用于生成
                    kv_cache.allocate_sequence(seq_id, seq_len, max_expected_length)

                # 写入KV数据
                kv_cache.write_kv(
                    seq_id=seq_id,
                    start_pos=0,
                    keys=K[batch_idx],  # [seq_len, num_heads, head_dim]
                    values=V[batch_idx]
                )

        with nvtx.annotate("prepare_paged_attention_inputs", color="cyan"):
            block_tables = []
            context_lens = []

            for seq_id in seq_ids:
                block_table = kv_cache.get_block_table(seq_id)
                block_tables.append(block_table)
                context_lens.append(kv_cache.sequence_lengths[seq_id])

            block_tables = torch.stack(block_tables, dim=0)  # [batch_size, max_blocks_per_seq]
            context_lens = torch.tensor(context_lens, dtype=torch.int32, device=self.device)

        with nvtx.annotate("cuda_paged_attention_prefill", color="red"):
            # 调用CUDA kernel进行prefill
            # 如果Q的维度是4维，会自动触发prefill模式
            attn_output = self.paged_attention(
                query=Q,  # [batch_size, seq_len, num_heads, head_dim]
                key_cache=kv_cache.key_cache,
                value_cache=kv_cache.value_cache,
                block_tables=block_tables,
                context_lens=context_lens,
                max_context_len=seq_len
            )

        with nvtx.annotate("output_projection", color="green"):
            attn_output = attn_output.view(batch_size, seq_len, self.d_model)
            output = self.W_o(attn_output)

        return output

    @nvtx.annotate("cuda_mha_decode", color="lightblue")
    def forward_decode(
            self,
            input_embeds: torch.Tensor,  # [batch_size, 1, d_model] - 单个新token
            kv_cache: CUDAPagedKVCache,
            seq_ids: List[int],
            positions: List[int]  # 每个序列当前的位置
    ) -> torch.Tensor:
        """
        Decode阶段：处理单个新生成的token
        """
        batch_size, _, _ = input_embeds.shape
        assert input_embeds.shape[1] == 1, "Decode phase should only process 1 token at a time"

        with nvtx.annotate("compute_qkv_decode", color="yellow"):
            input_flat = input_embeds.squeeze(1)  # [batch_size, d_model]

            Q = self.W_q(input_flat)  # [batch_size, d_model]
            K = self.W_k(input_flat)
            V = self.W_v(input_flat)

            Q = Q.view(batch_size, self.num_heads, self.head_dim)  # [batch_size, num_heads, head_dim]
            K = K.view(batch_size, self.num_heads, self.head_dim)
            V = V.view(batch_size, self.num_heads, self.head_dim)

        with nvtx.annotate("append_kv_to_cache", color="orange"):
            for batch_idx, (seq_id, pos) in enumerate(zip(seq_ids, positions)):
                # 写入新token的KV
                kv_cache.write_kv(
                    seq_id=seq_id,
                    start_pos=pos,
                    keys=K[batch_idx].unsqueeze(0),  # [1, num_heads, head_dim]
                    values=V[batch_idx].unsqueeze(0)
                )
                # 更新序列长度
                kv_cache.sequence_lengths[seq_id] = pos + 1

        with nvtx.annotate("prepare_decode_inputs", color="cyan"):
            block_tables = []
            context_lens = []

            for seq_id in seq_ids:
                block_table = kv_cache.get_block_table(seq_id)
                block_tables.append(block_table)
                context_lens.append(kv_cache.sequence_lengths[seq_id])

            block_tables = torch.stack(block_tables, dim=0)
            context_lens = torch.tensor(context_lens, dtype=torch.int32, device=self.device)

        with nvtx.annotate("cuda_paged_attention_decode", color="red"):
            # 如果Q的维度是3维，会自动触发decode模式
            attn_output = self.paged_attention(
                query=Q,  # [batch_size, num_heads, head_dim] - 3维触发decode
                key_cache=kv_cache.key_cache,
                value_cache=kv_cache.value_cache,
                block_tables=block_tables,
                context_lens=context_lens
            )

        with nvtx.annotate("output_projection_decode", color="green"):
            attn_output = attn_output.view(batch_size, self.d_model)
            output = self.W_o(attn_output)
            return output.unsqueeze(1)  # [batch_size, 1, d_model]


@nvtx.annotate("test_cuda_paged_prefill", color="gold")
def test_cuda_paged_prefill():
    """测试CUDA PagedAttention的Prefill阶段"""
    print("\n   Testing CUDA Paged Attention Prefill")

    config = {
        "d_model": 1024,
        "num_heads": 16,
        "head_dim": 64,
        "block_size": 32,
        "batch_size": 32,
        "seq_len": 512,
        "num_blocks": 1024,
    }

    kv_cache = CUDAPagedKVCache(
        num_blocks=config["num_blocks"],
        block_size=config["block_size"],
        num_heads=config["num_heads"],
        head_dim=config["head_dim"]
    )

    mha = CUDAPagedMultiHeadAttention(
        d_model=config["d_model"],
        num_heads=config["num_heads"],
        block_size=config["block_size"]
    )

    input_embeds = torch.randn(
        config["batch_size"], config["seq_len"], config["d_model"],
        dtype=torch.float16, device="cuda"
    )
    seq_ids = list(range(config["batch_size"]))

    print(f"Input shape: {input_embeds.shape}")
    torch.cuda.synchronize()
    output = mha.forward_prefill(input_embeds, kv_cache, seq_ids)
    torch.cuda.synchronize()
    print(f"Output shape: {output.shape}")

    return mha, kv_cache, seq_ids


@nvtx.annotate("test_cuda_paged_decode", color="darkblue")
def test_cuda_paged_decode(mha, kv_cache, seq_ids):
    """测试CUDA PagedAttention的Decode阶段"""
    print("\n   Testing CUDA Paged Attention Decode")

    batch_size = len(seq_ids)
    max_new_tokens = 64
    current_positions = [kv_cache.sequence_lengths[seq_id] for seq_id in seq_ids]
    print(f"Starting positions: {current_positions}")

    # 自回归生成
    for step in range(max_new_tokens):
        # 生成新token的embedding（在实际应用中，这来自于上一步的logits采样）
        new_token_embeds = torch.randn(
            batch_size, 1, mha.d_model,
            dtype=torch.float16, device="cuda"
        )

        torch.cuda.synchronize()

        with nvtx.annotate(f"decode_step_{step}", color="blue"):
            output = mha.forward_decode(
                new_token_embeds, kv_cache, seq_ids, current_positions
            )

        torch.cuda.synchronize()
        # 更新位置
        current_positions = [pos + 1 for pos in current_positions]



@nvtx.annotate("test_memory_efficiency", color="darkgreen")
def test_memory_efficiency():
    """测试内存效率"""
    print("\n   Testing Memory Efficiency")

    configs = [
        {"name": "Small", "d_model": 512, "num_heads": 8, "seq_len": 256},
        {"name": "Medium", "d_model": 1024, "num_heads": 16, "seq_len": 512},
        {"name": "Large", "d_model": 2048, "num_heads": 32, "seq_len": 1024},
    ]

    for config in configs:
        kv_cache = CUDAPagedKVCache(
            num_blocks=2048,
            block_size=16,
            num_heads=config["num_heads"],
            head_dim=config["d_model"] // config["num_heads"]
        )

        seq_lengths = [128, 256, 64, 512, 32, 384, 96, 448]

        total_tokens = 0
        for i, seq_len in enumerate(seq_lengths):
            kv_cache.allocate_sequence(i, seq_len)
            total_tokens += seq_len


@nvtx.annotate("comprehensive_cuda_paged_test", color="gold")
def comprehensive_cuda_paged_test():
    print(f"Using GPU: {torch.cuda.get_device_name()}")

    # 测试 prefill
    mha, kv_cache, seq_ids = test_cuda_paged_prefill()
    # 测试 decode
    test_cuda_paged_decode(mha, kv_cache, seq_ids)
    # 测试显存利用率
    test_memory_efficiency()


if __name__ == "__main__":
    comprehensive_cuda_paged_test()
    print("All Paged attention tests passed successfully!")