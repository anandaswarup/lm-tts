"""
Position embeddings for transformer models.
"""

import torch
import torch.nn as nn
from einops import rearrange


class RotaryPositionalEmbeddings(nn.Module):
    """
    Rotary Positional Embeddings as described in "RoFormer: Enhanced Transformer with Rotary Position Embedding"
    (https://arxiv.org/pdf/2104.09864). This implementation caches the embeddings upto max_seq_len for efficiency.
    """

    def __init__(self, dim: int, max_seq_len: int = 8192, base: int = 10000) -> None:
        """
        Initializes the RotaryPositionalEmbeddings module.

        Args:
            dim (int): Embedding dimension. Must equal embed_dim // num_heads where embed_dim is the transformer model
                embedding dimension and num_heads is the number of attention heads.
            max_seq_len (int): Maximum sequence length to cache the embeddings for.
            base (int): Base for the progression of frequencies used to compute the rotation angles.

        """
        super().__init__()

        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base

        # Compute the inverse frequencies
        theta = 1.0 / (base ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))

        # Compute the position indices [0, 1, ..., max_seq_len - 1]
        position_idxs = torch.arange(
            max_seq_len, dtype=theta.dtype, device=theta.device
        )

        # Compute outer product of position_idxs and theta to get the angles. Shape: (max_seq_len, dim // 2)
        angles = torch.einsum("i,j->ij", position_idxs, theta)

        # Compute the embeddings by concatenating cosines and sines of the angles and cache them.
        # Shape: (max_seq_len, dim, 2)
        cache = torch.stack([torch.cos(angles), torch.sin(angles)], dim=-1)
        self.register_buffer("cache", cache, persistent=False)
        self.cache: torch.Tensor

    def forward(
        self, x: torch.Tensor, input_pos: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        Forward pass of the RotaryPositionalEmbeddings module.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_heads, seq_len, dim).
            input_pos (torch.Tensor | None): Optional input tensor of shape (batch_size, seq_len) indicating the positions
                to use for each element in the batch. If None, positions are assumed to be [0, 1, ..., seq_len - 1].

        Returns:
            torch.Tensor: Tensor with rotary positional embeddings applied, of shape
                (batch_size, num_heads, seq_len, dim).
        """
        seq_len = x.size(-2)

        # Retrieve the cached embeddings for the required sequence length
        emb = self.cache[:seq_len] if input_pos is None else self.cache[input_pos]

        # Split the last dimension of the input tensor into two halves for applying rotations
        x = rearrange(x.float(), " ... (d x) -> ... d x", x=2)

        # Apply the rotary positional embeddings
        x_rope = torch.stack(
            [
                x[..., 0] * emb[..., 0] - x[..., 1] * emb[..., 1],
                x[..., 1] * emb[..., 0] + x[..., 0] * emb[..., 1],
            ],
            -1,
        )

        # Merge the last two dimensions back to the original shape
        x_rope = rearrange(x_rope, " ... d x -> ... (d x)")

        # Cast back to the original dtype
        x_rope = x_rope.type_as(x)

        return x_rope
