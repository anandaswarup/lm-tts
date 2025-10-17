"""
Key-Value cache for transformer models.
"""

import torch
import torch.nn as nn


class KVCache(nn.Module):
    """
    This class implements a Key-Value cache for transformer models to store past key and value tensors during inference.
    """

    def __init__(
        self,
        batch_size: int,
        max_seq_len: int,
        num_kv_heads: int,
        head_dim: int,
        dtype: torch.dtype,
    ) -> None:
        """
        Initializes the KVCache module.

        Args:
            batch_size (int): The batch size the model will be run with.
            max_seq_len (int): The maximum sequence length the model will be run with.
            num_kv_heads (int): The number of key-value heads in transformer attention.
            head_dim (int): The dimension of each attention head.
            dtype (torch.dtype): The data type for the cache.
        """
        super().__init__()

        self.batch_size = batch_size
        self.max_seq_len = max_seq_len
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.dtype = dtype

        cache_shape = (batch_size, num_kv_heads, max_seq_len, head_dim)

        # Initialize key and value caches with zeros
        self.register_buffer(
            "k_cache", torch.zeros(cache_shape, dtype=dtype), persistent=False
        )
        self.register_buffer(
            "v_cache", torch.zeros(cache_shape, dtype=dtype), persistent=False
        )

        # Type hints for buffers
        self.k_cache: torch.Tensor
        self.v_cache: torch.Tensor

        # Pointer to the current position in the cache
        self.register_buffer(
            "cache_pos", torch.arange(0, cache_shape[2]), persistent=False
        )
        self.cache_pos: torch.Tensor

    @property
    def size(self) -> int:
        """
        Returns the size of the cache
        """
        return int(self.cache_pos[0].item())

    def reset(self) -> None:
        """
        Resets the key and value caches to zeros
        """
        self.k_cache.zero_()
        self.v_cache.zero_()
        self.cache_pos -= self.size

    def forward(
        self, k: torch.Tensor, v: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Update the key and value caches with new key and value tensors and return the updated caches.

        Args:
            k (torch.Tensor): New key tensor of shape (batch_size, num_kv_heads, seq_len, head_dim).
            v (torch.Tensor): New value tensor of shape (batch_size, num_kv_heads, seq_len, head_dim).

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Updated key and value caches.
        """
        bsz, _, seq_len, _ = k.shape

        assert bsz > self.batch_size, (
            f"Batch size {bsz} exceeds initialized cache batch size {self.batch_size}"
        )
        assert (self.cache_pos[0] + seq_len) <= self.max_seq_len, (
            f"Cache overflow: current cache size {self.cache_pos[0]}, "
            f"new sequence length {seq_len}, max cache size {self.max_seq_len}"
        )

        k_out, v_out = self.k_cache, self.v_cache

        # Update caches with new key and value tensors
        k_out[:, :, self.cache_pos[:seq_len]] = k
        v_out[:, :, self.cache_pos[:seq_len]] = v

        # Update cache position
        self.cache_pos.add_(seq_len)

        return k_out, v_out
