import sys
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).parent))

from paged_attention.paged_attention import PagedAttention


def test_module_interface():
    """Test nn.Module interface"""
    print("Testing Module Interface...")

    batch_size = 4
    num_heads = 32
    head_dim = 128
    block_size = 16

    paged_attn = PagedAttention(
        num_heads=num_heads,
        head_dim=head_dim,
        block_size=block_size
    )

    query = torch.randn(batch_size, num_heads, head_dim).cuda().half()
    print(f"Input query shape: {query.shape}")

    # We try to use FP16
    key_cache = torch.randn(64, block_size, num_heads, head_dim).cuda().half()
    value_cache = torch.randn(64, block_size, num_heads, head_dim).cuda().half()
    block_tables = torch.arange(8).repeat(batch_size, 1).int().cuda()
    context_lens = torch.full((batch_size,), 128, dtype=torch.int32).cuda()

    output = paged_attn(query, key_cache, value_cache, block_tables, context_lens)

    print(f"Module output shape: {output.shape}")
    assert output.shape == query.shape
    print("Module interface test passed!")


if __name__ == "__main__":
    test_module_interface()
    print("Paged attention test passed successfully!")
