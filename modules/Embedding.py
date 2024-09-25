"""Embedding modules"""

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
        base: float = 10000.0,
        scale_base: float = 2048,
    ) -> None:
        """
        Instantiate the RotaryEmbedding module.

        Args:
            dim (int): The embedding dimension.
            max_embedding_positions (int): The maximum number of positions to embed. Defaults to 2048.
            base (float): The base period of the rotary embedding. Defaults to 10000.0.
            scale_base (float): The base decay rate in terms of scaling time. Defaults to 2048.
        """
        super().__init__()

        self.dim = dim
        self.base = base
        self.scale_base = scale_base

        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

        decay_rate = (torch.arange(0, dim, 2) + 0.4 * dim) / (1.4 * dim)
        self.register_buffer("decay_rate", decay_rate)

        self.rotation: torch.Tensor | None = None
        self.decay: torch.Tensor | None = None

    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        """
        Rotates half the hidden dims of the input tensor.

        Args:
            x (torch.Tensor): The input tensor to rotate.
        """
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]

        return torch.cat((-x2, x1), dim=-1)

    def _get_rotation(
        self, seq_len: int, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        """
        Get the rotation matrix for the given sequence length.

        Args:
            seq_len (int): The length of the input tensor.
            device (torch.device): The device to place the rotation matrix on.
            dtype (torch.dtype): The data type of the rotation matrix.
        """
        if self.rotation is None or seq_len > self.rotation.shape[0]:
            idx = torch.arange(seq_len, device=device).type_as(self.inv_freq)
            rotation = torch.einsum("i, j -> i j", idx, self.inv_freq)
            self.rotation = torch.cat((rotation, rotation), dim=-1).to(dtype=dtype)

        return self.rotation[:seq_len]

    def _get_decay(
        self, seq_len: int, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        """
        Get the decay matrix for the given sequence length.

        Args:
            seq_len (int): The length of the input tensor.
            device (torch.device): The device to place the decay matrix on.
            dtype (torch.dtype): The data type of the decay matrix.
        """
        if self.decay is None or seq_len > self.decay.shape[0]:
            idx = torch.arange(seq_len, device=device).type_as(self.inv_freq)
            power = (idx - (seq_len // 2)) / self.scale_base
            decay = self.decay_rate ** rearrange(power, "i -> i 1")
            self.decay = torch.cat((decay, decay), dim=-1).to(dtype=dtype)

        return self.decay[:seq_len]

    def forward(self, x: torch.Tensor, invert_decay: False) -> torch.Tensor:
        """
        Forward pass of the RotaryEmbedding module.

        Args:
            x (torch.Tensor): The input tensor to embed of shape [bsz, num_heads, seq_len, d_model // num_heads]
            invert_decay (bool): Whether to invert the decay matrix. Defaults to False.
        """
        seq_len = x.shape[-2]

        # Get the rotation and decay matrices
        rotation = self._get_rotation(seq_len=seq_len, device=x.device, dtype=x.dtype)
        decay = self._get_decay(seq_len=seq_len, device=x.device, dtype=x.dtype)

        if invert_decay:
            decay = 1 / decay

        x_embed = (x * rotation.cos() * decay) + (
            self._rotate_half(x) * rotation.sin() * decay
        )

        return x_embed
