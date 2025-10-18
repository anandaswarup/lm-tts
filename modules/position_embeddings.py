"""
Positional embeddings for transformer models.
"""

import torch
import torch.nn as nn
from einops import rearrange


class RotaryPositionalEmbeddings(nn.Module):
    """
    This class implements rotary positional embeddings for transformer models, as described in
    https://arxiv.org/pdf/2104.09864
    """

    def __init__(
        self, dim: int, max_seq_len: int = 4096, rope_base: int = 10000
    ) -> None:
        """
        Initializes the RotaryPositionalEmbeddings module.

        Args:
            dim (int): Dimension of the input tensor to be embedded. Must be equal to the dimension of each
                attention head.
            max_seq_len (int): Maximum sequence length for which to precompute embeddings.
                Default is 4096.
            rope_base (int): The base for the geometric progression used to compute the rotation angles.
                Default is 10000.
        """
        super().__init__()

        self.cache: torch.Tensor

        self.dim = dim
        self.max_seq_len = max_seq_len
        self.rope_base = rope_base

        # Precompute the inverse frequencies
        theta = 1.0 / (
            self.rope_base
            ** (torch.arange(0, self.dim, 2)[: (self.dim // 2)].float() / self.dim)
        )

        # Create position indices
        position_idxs = torch.arange(
            max_seq_len, dtype=theta.dtype, device=theta.device
        )

        # Outer product of theta and position index. Shape: [max_seq_len, dim // 2]
        idx_theta = torch.einsum("i, j -> ij", position_idxs, theta).float()

        # Compute the frequency embeddings and cache them. Shape: [max_seq_len, dim // 2, 2]
        cache = torch.stack([torch.cos(idx_theta), torch.sin(idx_theta)], dim=-1)
        self.register_buffer("cache", cache, persistent=False)

    def forward(
        self, x: torch.Tensor, input_pos: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        Forward pass of the RotaryPositionalEmbeddings module.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_heads, seq_len, head_dim), where head_dim is
                the dimension of each attention head.
            input_pos (torch.Tensor | None): Optional tensor of shape (..., seq_len) containing position indices.
                If None, positions are assumed to be [0, 1, ..., seq_len - 1].

        Returns:
            torch.Tensor: Tensor of shape (batch_size, num_heads, seq_len, head_dim) with rotary positional embeddings
                applied.
        """
        # Get seq length of input tensor
        seq_len = x.shape[2]

        # Get the relevant positional embeddings from the cache based on input_pos if provided or seq_len
        freqs = self.cache[input_pos] if input_pos is not None else self.cache[:seq_len]

        # Reshape input tensor to separate even and odd dimensions
        x = rearrange(x, "... (d k) -> ... d k", d=self.dim // 2)

        # Apply rotary embeddings
        x_rope = torch.stack(
            [
                x[..., 0] * freqs[..., 0] - x[..., 1] * freqs[..., 1],
                x[..., 1] * freqs[..., 0] + x[..., 0] * freqs[..., 1],
            ],
            -1,
        )

        # Reshape back to original dimensions
        x_rope = rearrange(x_rope, "... d k -> ... (d k)")

        return x_rope
