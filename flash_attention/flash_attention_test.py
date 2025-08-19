import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import torch.nn as nn
from flash_attention import FlashAttention

class TransformerBlockWithFlash(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads!"
        self.d_model = d_model
        self.n_heads = n_heads
        self.dropout = dropout
        self.d_ff = d_model * 4

        self.norm1 = nn.LayerNorm(self.d_model)
        self.norm2 = nn.LayerNorm(self.d_model)
        self.Dropout = nn.Dropout(self.dropout)
        self.attention = FlashAttention(d_model, n_heads, causal=True)
        self.ffn = nn.Sequential(
            nn.Linear(self.d_model, self.d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.d_ff, self.d_model),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        attn_out = self.attention(self.norm1(x))
        x = x + self.Dropout(attn_out)
        ffn_out = self.ffn(self.norm2(x))
        x = x + ffn_out
        return x

if __name__ == "__main__":
    d_model: int = 512
    model = TransformerBlockWithFlash(d_model=d_model, n_heads=16, dropout=0.1).cuda()
    x = torch.randn(2, 1024, d_model).cuda()
    output = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
