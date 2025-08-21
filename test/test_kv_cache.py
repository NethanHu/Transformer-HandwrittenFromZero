import torch

from kv_cache import KVCache

def test_kv_cache():
    cache: KVCache = KVCache(
        max_batch_size=2,
        max_seq_len=100,
        d_model=512,
        n_heads=8,
        device="cuda"
    )

    seq_len: int = 50
    batch_idx: int = 0
    keys: torch.Tensor = torch.randn(seq_len, cache.n_heads, cache.head_dim, device=cache.device)
    values: torch.Tensor = torch.randn(seq_len, cache.n_heads, cache.head_dim, device=cache.device)

    cached_k, cached_v = cache.update(batch_idx, 0, keys, values)

    assert cached_k.shape == (seq_len, cache.n_heads, cache.head_dim)
    assert cached_v.shape == (seq_len, cache.n_heads, cache.head_dim)

    assert torch.allclose(cached_k, keys)
    assert torch.allclose(cached_v, values)

    print(f"K Cache shape: {cached_k.shape}")
    print(f"V Cache shape: {cached_v.shape}")
    print("KV Cache test passed!")

if __name__ == '__main__':
    test_kv_cache()