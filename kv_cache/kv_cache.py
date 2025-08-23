from typing import Tuple

import torch


class KVCache:
    def __init__(
            self,
            max_batch_size: int,
            max_seq_len: int,
            d_model: int,
            n_heads: int,
            dtype: torch.dtype=torch.float32,  # FP32 impl
            device: str="cuda" if torch.cuda.is_available() else "cpu"
    ):
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        head_dim: int = d_model // n_heads
        self.max_batch_size: int = max_batch_size
        self.max_seq_len: int = max_seq_len
        self.d_model: int = d_model
        self.n_heads: int = n_heads
        self.head_dim: int = d_model // n_heads
        self.dtype: torch.dtype = dtype
        self.device: str = device
        # initialize the K, V cache tensor
        self.K_cache: torch.Tensor = torch.zeros(
           (max_batch_size, n_heads, max_seq_len, head_dim),
           dtype=dtype, device=device
        )
        self.V_cache: torch.Tensor = torch.zeros(
           (max_batch_size, n_heads, max_seq_len, head_dim),
           dtype=dtype, device=device
        )
        # 用来记录当前批次里，每一个句子已经生成到第几个字，
        # 这样，在下一步更新缓存时，我们就知道应该把新的K、V向量写到哪个位置。
        self.seq_lens: torch.Tensor = torch.zeros(
            max_batch_size, dtype=torch.long, device=device  # 应该用整数类型
        )

    def update(
            self,
            batch_idx: int,
            start_pos: int,
            keys: torch.Tensor,
            values: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        seq_len: int = keys.size(0)
        end_pos: int = start_pos + seq_len
        self.K_cache[batch_idx, :, start_pos:end_pos, :] = keys.transpose(0, 1)
        self.V_cache[batch_idx, :, start_pos:end_pos, :] = values.transpose(0, 1)
        self.seq_lens[batch_idx] = end_pos
        return (
            self.K_cache[batch_idx, :, :end_pos, :].transpose(0, 1),
            self.V_cache[batch_idx, :, :end_pos, :].transpose(0, 1)
        )

    def get(
            self,
            batch_idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        seq_len = self.seq_lens[batch_idx].item()
        return (
            self.K_cache[batch_idx, :, :seq_len, :].transpose(0, 1),
            self.V_cache[batch_idx, :, :seq_len, :].transpose(0, 1)
        )

    def clear(
            self,
            batch_idx: int | None
    ):
        if batch_idx is not None:
            self.K_cache[batch_idx].zero_()
            self.V_cache[batch_idx].zero_()
            self.seq_lens[batch_idx] = 0
        else:
            self.K_cache.zero_()
            self.V_cache.zero_()
            self.seq_lens.zero_()