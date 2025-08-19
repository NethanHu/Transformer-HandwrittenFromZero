import torch.nn as nn
import math
from pathlib import Path
from torch.utils.cpp_extension import load

current_dir = Path(__file__).parent  # flash_attention/flash_attention/
parent_dir = current_dir.parent       # flash_attention/

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
print("Has already compiled the source file. Loading Flash-Attention CUDA module...")

class FlashAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)
        self.dropout = dropout
        
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        
    def forward(self, x):
        batch_size, seq_len, embed_dim = x.shape
        
        Q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # This function will call the cuda implementation
        attn_out = flash_attention_cuda.forward(Q, K, V, self.scale)
        
        attn_out = attn_out.transpose(1, 2).contiguous().view(batch_size, seq_len, embed_dim)
        out = self.out_proj(attn_out)
        
        return out
