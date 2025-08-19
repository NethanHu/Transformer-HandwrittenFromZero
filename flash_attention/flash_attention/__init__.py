import torch
import torch.nn as nn
import math
from . import flash_attention_cuda

class FlashAttention(nn.Module):
    """
    Flash Attention v1 implementation

    Args:
        embed_dim: Embedding dimension
        num_heads: Number of attention heads
        dropout: Dropout rate (not implemented in this simple version)
        causal: Whether to use causal masking
    """
    def __init__(self, embed_dim, num_heads, dropout=0.0, causal=True):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)
        self.causal = causal

        # Linear projections
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(self, x):
        """
        Forward pass of Flash Attention

        Args:
            x: Input tensor of shape [batch_size, seq_len, embed_dim]

        Returns:
            Output tensor of shape [batch_size, seq_len, embed_dim]
        """
        batch_size, seq_len, embed_dim = x.shape

        # Project to Q, K, V
        Q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Apply Flash Attention
        attn_output = flash_attention_cuda.forward(Q, K, V, self.scale, self.causal)

        # Reshape and project output
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, embed_dim)
        output = self.out_proj(attn_output)

        return output

# Functional interface
def flash_attention(Q, K, V, scale=None, causal=True):
    """
    Functional Flash Attention interface

    Args:
        Q: Query tensor [batch_size, num_heads, seq_len, head_dim]
        K: Key tensor [batch_size, num_heads, seq_len, head_dim]
        V: Value tensor [batch_size, num_heads, seq_len, head_dim]
        scale: Scaling factor (default: 1/sqrt(head_dim))
        causal: Whether to use causal masking

    Returns:
        Attention output [batch_size, num_heads, seq_len, head_dim]
    """
    if scale is None:
        scale = 1.0 / math.sqrt(Q.shape[-1])

    return flash_attention_cuda.forward(Q, K, V, scale, causal)