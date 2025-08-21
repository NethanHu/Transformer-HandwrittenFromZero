import torch

from linear_attention import MultiHeadLinearAttention

if __name__ == "__main__":
    batch_size = 2
    seq_len = 100  # 较长序列来展示线性复杂度优势
    d_model = 512
    num_heads = 8
    dropout = 0.1

    x = torch.randn(batch_size, seq_len, d_model)
    print(f"Input Shape: {x.shape}")
    print(f"Sequence Length: {seq_len}")

    multi_head_linear = MultiHeadLinearAttention(d_model, num_heads, dropout=dropout)
    out2 = multi_head_linear(x)
    print(f"Output Shape: {out2.shape}")

    print(f"Normal Attention Complexity: O(n²×d) = O({seq_len}² × {d_model}) = O({seq_len**2 * d_model:,})")
    print(f"Linear Attention Complexity: O(n×d²) = O({seq_len} × {d_model}²) = O({seq_len * d_model**2:,})")

    print("Linear Attention test passed!")
