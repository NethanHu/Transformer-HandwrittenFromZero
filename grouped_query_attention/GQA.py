import math
import torch
import torch.nn.functional as F
import torch.nn as nn


class GroupedQueryAttention(nn.Module):
    def __init__(self, d_model: int, num_q_heads: int, num_kv_heads: int, head_size: int, context_len: int, dropout: float):
        super().__init__()
        self.d_model = d_model
        self.num_q_heads = num_q_heads
        self.num_kv_heads = num_kv_heads
        self.head_size = head_size
        self.context_len = context_len
        self.dropout = dropout
        self.scale = math.sqrt(self.head_size)

        assert num_q_heads % num_kv_heads == 0, "num_q_heads must be divisible by num_kv_heads"
        self.group_size = num_q_heads // num_kv_heads

        self.Wq = nn.Linear(self.d_model, self.num_q_heads * self.head_size, bias=False)
        self.Wk = nn.Linear(self.d_model, self.num_kv_heads * self.head_size, bias=False)
        self.Wv = nn.Linear(self.d_model, self.num_kv_heads * self.head_size, bias=False)

        self.register_buffer('mask', torch.tril(torch.ones(self.context_len, self.context_len)))
        self.dropout_layer = nn.Dropout(self.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape  # [batch_size, context_len, d_model]

        # 计算Q, K, V
        q = self.Wq(x)  # [B, T, num_q_heads * head_size]
        k = self.Wk(x)  # [B, T, num_kv_heads * head_size]
        v = self.Wv(x)  # [B, T, num_kv_heads * head_size]

        # 重塑为多头格式
        q = q.view(B, T, self.num_q_heads, self.head_size)  # [B, T, num_q_heads, head_size]
        k = k.view(B, T, self.num_kv_heads, self.head_size)  # [B, T, num_kv_heads, head_size]
        v = v.view(B, T, self.num_kv_heads, self.head_size)  # [B, T, num_kv_heads, head_size]

        # 转置以便进行注意力计算: [B, num_heads, T, head_size]
        q = q.transpose(1, 2)  # [B, num_q_heads, T, head_size]
        k = k.transpose(1, 2)  # [B, num_kv_heads, T, head_size]
        v = v.transpose(1, 2)  # [B, num_kv_heads, T, head_size]

        # 扩展K和V以匹配Q的头数
        # 每个kv头需要被复制group_size次来匹配对应的q头
        k = k.repeat_interleave(self.group_size, dim=1)  # [B, num_q_heads, T, head_size]
        v = v.repeat_interleave(self.group_size, dim=1)  # [B, num_q_heads, T, head_size]

        scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale  # [B, num_q_heads, T, T]
        scores = scores.masked_fill(self.mask[:T, :T] == 0, float('-inf'))
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout_layer(attn_weights)

        out = torch.matmul(attn_weights, v)  # [B, num_q_heads, T, head_size]

        out = out.transpose(1, 2).contiguous().view(B, T, self.num_q_heads * self.head_size)

        return out


class MultiHeadGroupedQueryAttention(nn.Module):
    def __init__(self, d_model: int, num_q_heads: int, num_kv_heads: int, context_len: int, dropout: float):
        super().__init__()
        self.d_model = d_model
        self.num_q_heads = num_q_heads
        self.num_kv_heads = num_kv_heads
        self.head_size = d_model // num_q_heads

        self.gqa = GroupedQueryAttention(
            d_model=d_model,
            num_q_heads=num_q_heads,
            num_kv_heads=num_kv_heads,
            head_size=self.head_size,
            context_len=context_len,
            dropout=dropout
        )

        self.Wo = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.gqa(x)
        out = self.Wo(out)
        out = self.dropout(out)
        return out


class TransformerBlockWithGQA(nn.Module):
    def __init__(self, d_model: int, num_q_heads: int, num_kv_heads: int, context_len: int, dropout: float):
        super().__init__()
        self.d_model = d_model
        self.num_q_heads = num_q_heads
        self.num_kv_heads = num_kv_heads
        self.context_len = context_len
        self.dropout = dropout

        self.ln1 = nn.LayerNorm(self.d_model)
        self.ln2 = nn.LayerNorm(self.d_model)

        # 使用Grouped-Query Attention替代普通的多头注意力
        self.gqa = MultiHeadGroupedQueryAttention(
            d_model=self.d_model,
            num_q_heads=self.num_q_heads,
            num_kv_heads=self.num_kv_heads,
            context_len=self.context_len,
            dropout=self.dropout
        )

        self.ffn = nn.Sequential(
            nn.Linear(self.d_model, self.d_model * 4),
            nn.ReLU(),
            nn.Linear(self.d_model * 4, self.d_model),
            nn.Dropout(self.dropout)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x + self.gqa(self.ln1(x))
        out = out + self.ffn(self.ln2(out))
        return out