"""Embedding modules for use in transformer models"""

import torch
import torch.nn as nn
from einops import rearrange


class RotaryEmbedding(nn.Module):
    """
    Length-extrapolatable rotary positional embedding (xPos) descibed in https://arxiv.org/abs/2212.10554v1, which
    applies an exponential decay to the RoPE rotation matrix.
    """

    def __init__(
        self,
        dim: int,
        base_period: float = 10000.0,
        base_decay_rate: int = 512,
        device: torch.device | None = None,
    ) -> None:
        """
        Instantiates the RotaryEmbedding module.

        Args:
            dim (int): Embedding dimension. Must be d_model // n_heads, where d_model is the transformer model dimension
                and n_heads is the number of attention heads.
            base_period (float): Maximum period of the rotation frequencies. Defaults to 10000.0.
            base_decay_rate (int): Base decay rate for the exponential decay. Defaults to 512.
            device (torch.device, optional): The device to use. Defaults to None.
        """
        super().__init__()

        self.dim = dim
        self.base_period = base_period
        self.base_decay_rate = base_decay_rate
        self.device = device

        # Rotation frequencies
        freq = 1.0 / (
            self.base_period
            ** (torch.arange(0, self.dim, 2, device=device).float() / self.dim)
        )
        self.register_buffer("freq", freq, persistent=False)

        # xPos decay rates
        decay_rate = (torch.arange(0, self.dim, 2, device=device) + 0.4 * self.dim) / (
            1.4 * self.dim
        )
        self.register_buffer("decay_rate", decay_rate, persistent=False)

        # Rotation and decay tensors
        self.rotation: torch.Tensor | None = None
        self.decay: torch.Tensor | None = None

    def _rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        """
        Rotates half the hidden dims of the input tensor.

        Args:
            x (torch.Tensor): Input tensor.
        """
        x_1, x_2 = x.chunk(2, dim=-1)

        return torch.cat((-x_2, x_1), dim=-1)

    def _get_rotation(self, start: int, end: int) -> torch.Tensor:
        """
        Create rotation tensor for the given start and end indices. Cache values for fast computation
        """
        if self.rotation is None or end > self.rotation.shape[0]:
            assert isinstance(self.freq, torch.Tensor)
            idx = torch.arange(end, device=self.device).type_as(self.freq)
            rotation = torch.einsum("i, j -> i j", idx, self.freq)
            self.rotation = torch.cat((rotation, rotation), dim=-1)

        return self.rotation[start:end]

    def _get_decay(self, start: int, end: int) -> torch.Tensor:
        """
        Create complex decay tensor for the given start and end indices. Cache values for fast computation
        """
        if self.decay is None or end > self.decay.shape[0]:
            assert isinstance(self.decay_rate, torch.Tensor)
            idx = torch.arange(end, device=self.device).type_as(self.decay_rate)
            power = (idx - ((end - start) // 2)) / self.base_decay_rate
            decay = self.decay_rate ** rearrange(power, "i -> i 1")
            self.decay = torch.cat((decay, decay), dim=-1)

        return self.decay[start:end]

    def apply_rotary_embedding(
        self, query: torch.Tensor, key: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Apply rotary positional embedding to the query and key tensors. Assumes that the query and key tensors are
        from the same source (i.e. self attention) and have shape [batch_size, n_heads, seq_len, d_model // n_heads].

        Args:
            query (torch.Tensor): Query tensor.
            key (torch.Tensor): Key tensor.
        """
        assert (
            query.shape[-2] == key.shape[-2]
        ), "Query and key tensors must have the same sequence length"

        # Get the rotation and decay tensors
        rotation = self._get_rotation(0, query.shape[-2])
        decay = self._get_decay(0, query.shape[-2])

        # Apply the rotation and decay to the query and key tensors
        query_embed = (query * rotation.cos() * decay) + (
            self._rotate_half(query) * rotation.sin() * decay
        )
        key_embed = (key * rotation.cos() * (1 / decay)) + (
            self._rotate_half(key) * rotation.sin() * (1 / decay)
        )

        return query_embed, key_embed
