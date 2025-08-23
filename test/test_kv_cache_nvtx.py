import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import nvtx
from typing import Tuple
import math

from kv_cache import KVCache


class SimpleMultiHeadAttention(nn.Module):
    """简化版多头注意力，用于KV缓存性能测试"""
    def __init__(self, d_model: int, n_heads: int, device: str = "cuda"):
        super().__init__()
        assert d_model % n_heads == 0

        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.device = device

        self.W_q = nn.Linear(d_model, d_model, bias=False, device=device)
        self.W_k = nn.Linear(d_model, d_model, bias=False, device=device)
        self.W_v = nn.Linear(d_model, d_model, bias=False, device=device)
        self.W_o = nn.Linear(d_model, d_model, bias=False, device=device)

        self.scale = 1.0 / math.sqrt(self.head_dim)

        self._init_weights()

    def _init_weights(self):
        for module in [self.W_q, self.W_k, self.W_v, self.W_o]:
            nn.init.xavier_uniform_(module.weight)

    @nvtx.annotate("mha_forward", color="lightgreen")
    def forward(
            self,
            query: torch.Tensor,  # [seq_len, d_model]
            kv_cache: KVCache,
            batch_idx: int,
            start_pos: int,
            use_cache: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        seq_len = query.shape[0]

        with nvtx.annotate("compute_qkv", color="yellow"):
            Q = self.W_q(query)  # [seq_len, d_model]
            K = self.W_k(query)  # [seq_len, d_model]
            V = self.W_v(query)  # [seq_len, d_model]

            Q = Q.view(seq_len, self.n_heads, self.head_dim)  # [seq_len, n_heads, head_dim]
            K = K.view(seq_len, self.n_heads, self.head_dim)  # [seq_len, n_heads, head_dim]
            V = V.view(seq_len, self.n_heads, self.head_dim)  # [seq_len, n_heads, head_dim]

        if use_cache:
            with nvtx.annotate("update_kv_cache", color="orange"):
                cached_K, cached_V = kv_cache.update(batch_idx, start_pos, K, V)
        else:
            cached_K, cached_V = K, V

        with nvtx.annotate("compute_attention", color="red"):
            Q_t = Q.transpose(0, 1)  # [n_heads, seq_len, head_dim]
            cached_K_t = cached_K.transpose(0, 1)  # [n_heads, cached_seq_len, head_dim]
            cached_V_t = cached_V.transpose(0, 1)  # [n_heads, cached_seq_len, head_dim]
            with nvtx.annotate("attention_scores", color="pink"):
                scores = torch.matmul(Q_t, cached_K_t.transpose(-2, -1)) * self.scale
            with nvtx.annotate("causal_mask", color="purple"):
                cached_seq_len = cached_K.shape[0]
                if cached_seq_len > seq_len:
                    mask = torch.triu(torch.ones(seq_len, cached_seq_len, device=self.device), diagonal=start_pos + 1)
                    scores.masked_fill_(mask.bool().unsqueeze(0), float('-inf'))
            with nvtx.annotate("attention_softmax", color="cyan"):
                attn_weights = F.softmax(scores, dim=-1)
            with nvtx.annotate("attention_output", color="blue"):
                attn_output = torch.matmul(attn_weights, cached_V_t)
                attn_output = attn_output.transpose(0, 1).contiguous()  # [seq_len, n_heads, head_dim]
                attn_output = attn_output.view(seq_len, self.d_model)  # [seq_len, d_model]

        with nvtx.annotate("output_projection", color="green"):
            output = self.W_o(attn_output)

        return output, K, V


@nvtx.annotate("realistic_autoregressive_test", color="gold")
def test_realistic_autoregressive_generation(cache: KVCache, config: dict):
    """使用真实多头注意力的自回归生成测试"""
    mha = SimpleMultiHeadAttention(
        d_model=config["d_model"],
        n_heads=config["n_heads"],
        device="cuda"
    )

    batch_idx = 0
    initial_seq_len = 128
    max_new_tokens = min(256, config["max_seq_len"] - initial_seq_len)
    cache.clear(batch_idx)

    with nvtx.annotate("process_initial_prompt", color="lightblue"):
        initial_input = torch.randn(initial_seq_len, config["d_model"], device="cuda")
        prompt_output, initial_k, initial_v = mha(
            query=initial_input,
            kv_cache=cache,
            batch_idx=batch_idx,
            start_pos=0,
            use_cache=True
        )
        cached_k, cached_v = cache.get(batch_idx)
        assert cached_k.shape[0] == initial_seq_len
        print(f"    Initial cache verification passed: {cached_k.shape}")

    print(f"    Starting autoregressive generation ({max_new_tokens} tokens)...")
    current_pos = initial_seq_len
    last_token_output = prompt_output[-1:, :]  # [1, d_model]
    for token_idx in range(max_new_tokens):
        if current_pos >= config["max_seq_len"]:
            break

        with nvtx.annotate(f"generate_token_{token_idx}_pos_{current_pos}", color="orange"):
            token_output, new_k, new_v = mha(
                query=last_token_output,  # [1, d_model]
                kv_cache=cache,
                batch_idx=batch_idx,
                start_pos=current_pos,
                use_cache=True
            )
            cached_k, cached_v = cache.get(batch_idx)
            expected_len = current_pos + 1
            if cached_k.shape[0] != expected_len:
                break

            # 模拟下一个token的输入（在实际应用中，这里会是词表的采样结果再经过embedding）
            # 为了测试，我们添加一些随机性但保持合理的连续性
            noise = torch.randn_like(token_output) * 0.1
            last_token_output = token_output + noise
            current_pos += 1
    torch.cuda.synchronize()


@nvtx.annotate("batch_parallel_generation_test", color="darkgreen")
def test_batch_parallel_generation(cache: KVCache, config: dict):
    """测试批量并行生成"""
    mha = SimpleMultiHeadAttention(
        d_model=config["d_model"],
        n_heads=config["n_heads"],
        device="cuda"
    )

    batch_size = min(config["max_batch_size"], 16)
    initial_seq_len = 64
    max_new_tokens = 256
    chat_rounds = 5

    cache.clear(None)

    with nvtx.annotate("batch_prompt_processing", color="lightcyan"):
        for _ in range(chat_rounds):
            for batch_idx in range(batch_size):
                with nvtx.annotate(f"batch_{batch_idx}_prompt", color="lightgreen"):
                    initial_input = torch.randn(initial_seq_len, config["d_model"], device="cuda")

                    prompt_output, _, _ = mha(
                        query=initial_input,
                        kv_cache=cache,
                        batch_idx=batch_idx,
                        start_pos=0,
                        use_cache=True
                    )

    last_outputs = {}
    for batch_idx in range(batch_size):
        cached_k, cached_v = cache.get(batch_idx)
        last_outputs[batch_idx] = torch.randn(1, config["d_model"], device="cuda")

    for step in range(max_new_tokens):
        with nvtx.annotate(f"parallel_generation_step_{step}", color="darkblue"):
            for batch_idx in range(batch_size):
                current_pos = initial_seq_len + step

                if current_pos >= config["max_seq_len"]:
                    continue

                with nvtx.annotate(f"batch_{batch_idx}_step_{step}", color="blue"):
                    token_output, _, _ = mha(
                        query=last_outputs[batch_idx],
                        kv_cache=cache,
                        batch_idx=batch_idx,
                        start_pos=current_pos,
                        use_cache=True
                    )

                    noise = torch.randn_like(token_output) * 0.05
                    last_outputs[batch_idx] = token_output + noise
    torch.cuda.synchronize()


@nvtx.annotate("attention_memory_analysis", color="darkred")
def test_attention_memory_patterns(cache: KVCache, config: dict):
    """分析注意力计算的内存访问模式"""
    mha = SimpleMultiHeadAttention(
        d_model=config["d_model"],
        n_heads=config["n_heads"],
        device="cuda"
    )

    batch_idx = 0
    cache.clear(batch_idx)

    sequence_lengths = [64, 256, 1024, 4096]

    for seq_len in sequence_lengths:
        if seq_len > config["max_seq_len"]:
            continue

        cache.clear(batch_idx)
        torch.cuda.empty_cache()

        with nvtx.annotate(f"memory_test_seqlen_{seq_len}", color="red"):
            chunk_size = 64
            for start_pos in range(0, seq_len, chunk_size):
                actual_chunk = min(chunk_size, seq_len - start_pos)

                input_chunk = torch.randn(actual_chunk, config["d_model"], device="cuda")

                with nvtx.annotate(f"chunk_{start_pos}_{start_pos + actual_chunk}", color="pink"):
                    output, _, _ = mha(
                        query=input_chunk,
                        kv_cache=cache,
                        batch_idx=batch_idx,
                        start_pos=start_pos,
                        use_cache=True
                    )

        torch.cuda.synchronize()

@nvtx.annotate("comprehensive_mha_kv_test", color="gold")
def comprehensive_mha_kv_cache_test():
    """带多头注意力的全面KV缓存测试"""
    configs = [
        # 小模型测试
        {"max_batch_size": 16, "max_seq_len": 1024, "d_model": 768, "n_heads": 12, "name": "small"},
        # 中等模型测试
        {"max_batch_size": 64, "max_seq_len": 2048, "d_model": 1536, "n_heads": 24, "name": "medium"},
        # 大模型测试
        {"max_batch_size": 128, "max_seq_len": 4096, "d_model": 4096, "n_heads": 32, "name": "large"},
    ]

    for config in configs:
        print(f"Current test for {config['name']} model...")
        with nvtx.annotate(f"init_{config['name']}_cache", color="green"):
            cache = KVCache(
                max_batch_size=config["max_batch_size"],
                max_seq_len=config["max_seq_len"],
                d_model=config["d_model"],
                n_heads=config["n_heads"],
                device="cuda"
            )

        # 模拟真实自回归生成
        test_realistic_autoregressive_generation(cache, config)
        # 模拟批量并行生成
        test_batch_parallel_generation(cache, config)
        # 模拟内存访问模式分析
        test_attention_memory_patterns(cache, config)

if __name__ == '__main__':
    print(f"Using CUDA device: {torch.cuda.current_device()}")
    print(f"GPU: {torch.cuda.get_device_name()}")

    comprehensive_mha_kv_cache_test()

    print("All NVTX KV Cache tests completed!")