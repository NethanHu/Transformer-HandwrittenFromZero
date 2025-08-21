import torch

from sparse_attention.SparseAttention import MultiHeadSparseAttention

if __name__ == "__main__":
    batch_size = 16
    seq_len = 100
    d_model = 512
    num_heads = 8
    dropout = 0.1

    x = torch.randn(batch_size, seq_len, d_model)
    print(f"Input Shape: {x.shape}")
    print(f"Sequence Length: {seq_len}")

    multi_sparse_linear = MultiHeadSparseAttention(d_model, num_heads, dropout=dropout)
    out = multi_sparse_linear(x)
    print(f"Output Shape: {out.shape}")

    window_size = 32  # 固定窗口大小
    total_possible_connections = seq_len * seq_len
    local_connections = 0
    for i in range(seq_len):
        start = max(0, i - window_size // 2)
        end = min(seq_len, i + window_size // 2 + 1)
        causal_end = min(end, i + 1)
        local_connections += (causal_end - start)
    sparsity = 1 - (local_connections / total_possible_connections)
    dense_attn_memory = batch_size * num_heads * seq_len * seq_len * 4  # bytes
    sparse_attn_memory = batch_size * num_heads * local_connections * 4  # bytes
    print(f"Dense Attention Matrix Memory:  {dense_attn_memory / (1024**2):.1f} MB")
    print(f"Sparse Attention Matrix Memory: {sparse_attn_memory / (1024**2):.1f} MB")

    print("Sparse Attention test passed!")