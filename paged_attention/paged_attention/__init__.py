from pathlib import Path

import torch
import torch.nn as nn

from torch.utils.cpp_extension import load

current_dir = Path(__file__).parent  # flash_attention/flash_attention/
parent_dir = current_dir.parent       # flash_attention/

paged_attention_cuda = load(
    name='paged_attention_cuda',
    sources=[
        str(current_dir / 'paged_attention.cpp'),
        str(parent_dir / 'paged_attention_kernel.cu'),
    ],
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    extra_cflags=['-O3'],
    verbose=True
)
print("Has already compiled the source file. Loading Paged-Attention CUDA module...")

class PagedAttention(nn.Module):
    def __init__(
            self,
            num_heads: int,
            head_dim: int,
            scale: float | None = None,
            block_size: int = 16,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale or (head_dim ** -0.5)
        self.block_size = block_size

    def forward(
            self,
            query: torch.Tensor,
            key_cache: torch.Tensor,
            value_cache: torch.Tensor,
            block_tables: torch.Tensor,
            context_lens: torch.Tensor,
            max_context_len: int | None = None,
    ) -> torch.Tensor:
        if query.dim() == 3:
            batch_size, num_heads, head_dim = query.shape
            is_decode = True
        else:
            batch_size, seq_len, num_heads, head_dim = query.shape
            is_decode = False

        if max_context_len is None:
            max_context_len = int(context_lens.max().item())
        if self.scale is None:
            self.scale = head_dim ** -0.5

        output = torch.empty_like(query)

        if is_decode:
            paged_attention_cuda.forward_decode(
                output,
                query,
                key_cache,
                value_cache,
                block_tables,
                context_lens,
                self.block_size,
                max_context_len,
                self.scale
            )
        else:
            paged_attention_cuda.forward_prefill(
                output,
                query,
                key_cache,
                value_cache,
                block_tables,
                context_lens,
                self.block_size,
                max_context_len,
                self.scale
            )

        return output