import torch
import torch.nn as nn
import math
import os
from pathlib import Path
from torch.utils.cpp_extension import load

current_dir = Path(__file__).parent  # flash_attention/flash_attention/
parent_dir = current_dir.parent       # flash_attention/


try:
    flash_attention_cuda = load(
        name='flash_attention_cuda',
        sources=[
            str(current_dir / 'flash_attention.cpp'),
            str(parent_dir / 'flash_attention_kernel.cu'),
        ],
        extra_cuda_cflags=['-O3', '--use_fast_math'],
        extra_cflags=['-O3'],
        verbose=True
    )
    print("Flash Attention CUDA module loaded successfully!")
except Exception as e:
    print(f"Warning: Could not compile Flash Attention CUDA extension: {e}")
    print("Falling back to PyTorch implementation")
    flash_attention_cuda = None

class FlashAttention(nn.Module):
    """Flash Attention v1 implementation"""
    def __init__(self, embed_dim, num_heads, dropout=0.0, causal=True):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)
        self.causal = causal
        self.dropout = dropout
        
        # Linear projections
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        
    def forward(self, x):
        batch_size, seq_len, embed_dim = x.shape
        
        # Project to Q, K, V
        Q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Apply Flash Attention (with fallback)
        if flash_attention_cuda is not None:
            attn_output = flash_attention_cuda.forward(Q, K, V, self.scale, self.causal)
        else:
            # Fallback to standard attention
            attn_output = standard_attention(Q, K, V, self.scale, self.causal)
        
        # Reshape and project output
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, embed_dim)
        output = self.out_proj(attn_output)
        
        return output

def standard_attention(Q, K, V, scale, causal=True):
    """Standard PyTorch attention as fallback"""
    scores = torch.matmul(Q, K.transpose(-2, -1)) * scale
    
    if causal:
        seq_len = Q.size(-2)
        mask = torch.triu(torch.ones(seq_len, seq_len, device=Q.device), diagonal=1).bool()
        scores.masked_fill_(mask, float('-inf'))
    
    attn_weights = torch.softmax(scores, dim=-1)
    output = torch.matmul(attn_weights, V)
    return output

def flash_attention(Q, K, V, scale=None, causal=True):
    """Functional Flash Attention interface"""
    if scale is None:
        scale = 1.0 / math.sqrt(Q.shape[-1])
    
    if flash_attention_cuda is not None:
        return flash_attention_cuda.forward(Q, K, V, scale, causal)
    else:
        return standard_attention(Q, K, V, scale, causal)
