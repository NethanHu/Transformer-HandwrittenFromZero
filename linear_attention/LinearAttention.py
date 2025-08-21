import torch
import torch.nn.functional as F
import torch.nn as nn


class LinearAttention(nn.Module):
    """
    Linear Attention: 将注意力计算复杂度从O(n²)降低到O(n)
    使用特征映射 φ(x) 来近似softmax操作
    """
    def __init__(self, d_model: int, head_size: int, feature_dim: int = None, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.head_size = head_size
        self.feature_dim = feature_dim if feature_dim is not None else head_size
        self.dropout = dropout

        self.Wq = nn.Linear(self.d_model, self.head_size, bias=False)
        self.Wk = nn.Linear(self.d_model, self.head_size, bias=False)
        self.Wv = nn.Linear(self.d_model, self.head_size, bias=False)

        # 特征映射的投影层：正是有这样的投影层才能让Linear Attention近似原生Attention
        self.phi_q = nn.Linear(self.head_size, self.feature_dim, bias=False)
        self.phi_k = nn.Linear(self.head_size, self.feature_dim, bias=False)

        self.dropout_layer = nn.Dropout(self.dropout)

    def feature_map(self, x: torch.Tensor) -> torch.Tensor:
        """
        特征映射函数 φ(x) = [sin(x), cos(x), x, x²/2, ...]
        也可以使用其他映射如ReLU特征映射: φ(x) = ReLU(x) + ε
        """
        # ELU特征映射
        return F.elu(x) + 1.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q = self.Wq(x)  # [B, T, head_size]
        k = self.Wk(x)  # [B, T, head_size]
        v = self.Wv(x)  # [B, T, head_size]

        phi_q = self.feature_map(q)  # [B, T, feature_dim]
        phi_k = self.feature_map(k)  # [B, T, feature_dim]
        output = self._causal_linear_attention(phi_q, phi_k, v)
        return self.dropout_layer(output)

    def _causal_linear_attention(self, phi_q: torch.Tensor, phi_k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """
        因果线性注意力：O(n*d²)复杂度
        使用累积求和实现因果掩码
        """
        B, T, D = phi_q.shape
        _, _, V_dim = v.shape

        kv_state = torch.zeros(B, D, V_dim, device=phi_q.device, dtype=phi_q.dtype)
        k_state = torch.zeros(B, D, device=phi_q.device, dtype=phi_q.dtype)

        outputs = []

        for t in range(T):
            # 当前时间步的查询、键、值
            q_t = phi_q[:, t:t+1, :]  # [B, 1, D]
            k_t = phi_k[:, t:t+1, :]  # [B, 1, D]
            v_t = v[:, t:t+1, :]      # [B, 1, V_dim]

            # 更新累积状态
            kv_state = kv_state + torch.bmm(k_t.transpose(-1, -2), v_t)  # [B, D, V_dim]
            k_state = k_state + k_t.sum(dim=1)  # [B, D]

            # 计算注意力输出
            numerator = torch.bmm(q_t, kv_state)  # [B, 1, V_dim]
            denominator = torch.bmm(q_t, k_state.unsqueeze(-1))  # [B, 1, 1]
            out_t = numerator / denominator

            outputs.append(out_t)

        return torch.cat(outputs, dim=1)  # [B, T, V_dim]


class MultiHeadLinearAttention(nn.Module):
    """
    多头线性注意力
    """
    def __init__(self, d_model: int, num_heads: int, feature_dim: int = None, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_size = d_model // num_heads
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.heads = nn.ModuleList([
            LinearAttention(d_model, self.head_size, feature_dim, dropout)
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


class TransformerBlockWithLinearAttention(nn.Module):
    """
    使用线性注意力的Transformer块
    """
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.dropout = dropout

        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

        self.linear_attention = MultiHeadLinearAttention(d_model, num_heads, dropout=dropout)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x + self.linear_attention(self.ln1(x))
        out = out + self.ffn(self.ln2(out))
        return out

