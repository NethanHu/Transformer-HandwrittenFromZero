import torch

from grouped_query_attention import TransformerBlockWithGQA

if __name__ == "__main__":
    batch_size = 16
    seq_len = 128
    d_model = 512
    num_q_heads = 8
    num_kv_heads = 2
    context_len = 512
    dropout = 0.1

    x: torch.Tensor = torch.randn(batch_size, seq_len, d_model)
    print(f"Input shape: {x.shape}")

    gqa_block = TransformerBlockWithGQA(
        d_model=d_model,
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        context_len=context_len,
        dropout=dropout
    )

    output = gqa_block(x)
    print(f"Output shape: {output.shape}")

    total_params = sum(p.numel() for p in gqa_block.parameters())
    gqa_params = sum(p.numel() for p in gqa_block.gqa.parameters())

    print(f"Parameter # in GQA module: {gqa_params:,}")
    print(f"Parameter # in total Block: {total_params:,}")

    print("Grouped Query Attention test passed!")