import torch
import torch.nn.functional as F
import torch.nn as nn


class SparseAttention(nn.Module):
    """
    稀疏注意力：使用固定的局部窗口模式
    每个位置只关注前后各16个位置（窗口大小32）
    """
    def __init__(self, d_model: int, head_size: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.head_size = head_size
        self.dropout = dropout
        self.window_size = 32  # 固定窗口大小
        self.scale = torch.sqrt(torch.tensor(self.head_size, dtype=torch.float32))

        self.Wq = nn.Linear(self.d_model, self.head_size, bias=False)
        self.Wk = nn.Linear(self.d_model, self.head_size, bias=False)
        self.Wv = nn.Linear(self.d_model, self.head_size, bias=False)

        self.dropout_layer = nn.Dropout(self.dropout)

    def _create_local_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """创建局部窗口掩码"""
        mask = torch.zeros(seq_len, seq_len, device=device, dtype=torch.bool)

        for i in range(seq_len):
            start = max(0, i - self.window_size // 2)
            end = min(seq_len, i + self.window_size // 2 + 1)
            mask[i, start:end] = True

        return mask

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape

        q = self.Wq(x)  # [B, T, head_size]
        k = self.Wk(x)  # [B, T, head_size]
        v = self.Wv(x)  # [B, T, head_size]

        scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale  # [B, T, T]
        sparse_mask = self._create_local_mask(T, x.device)  # [T, T]
        causal_mask = torch.tril(torch.ones(T, T, device=x.device, dtype=torch.bool))
        # 合并掩码：既要满足稀疏性，又要满足因果性
        final_mask = sparse_mask & causal_mask
        scores = scores.masked_fill(~final_mask.unsqueeze(0), float('-inf'))
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout_layer(attn_weights)
        out = torch.matmul(attn_weights, v)  # [B, T, head_size]

        return out


class MultiHeadSparseAttention(nn.Module):
    """
    多头稀疏注意力
    """
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_size = d_model // num_heads
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.heads = nn.ModuleList([
            SparseAttention(d_model, self.head_size, dropout)
            for _ in range(num_heads)
        ])

        self.Wo = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        head_outputs = [head(x) for head in self.heads]
        out = torch.cat(head_outputs, dim=-1)  # [B, T, d_model]
        out = self.Wo(out)
        out = self.dropout(out)

        return out


class TransformerBlockWithSparseAttention(nn.Module):
    """
    使用稀疏注意力的Transformer块
    """
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.dropout = dropout

        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

        self.sparse_attention = MultiHeadSparseAttention(d_model, num_heads, dropout=dropout)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x + self.sparse_attention(self.ln1(x))
        out = out + self.ffn(self.ln2(out))
        return out
