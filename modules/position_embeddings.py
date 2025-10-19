"""
Position embeddings for transformer models.
"""

import torch
import torch.nn as nn
from einops import rearrange


class RotaryPositionalEmbeddings(nn.Module):
    """
    Rotary Positional Embeddings (RoPE) implementation as described in https://arxiv.org/abs/2104.09864.
    """

    def __init__(
        self, head_dim: int, max_seq_len: int = 4096, base: int = 10000
    ) -> None:
        """
        Initialize RotaryPositionalEmbeddings.

        Args:
            head_dim (int): Dimension of each attention head.
            max_seq_len (int): Maximum sequence length supported by the model.
            base (int): Base for the geometric progression used to compute the rotation angles.
        """
        super().__init__()

        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.base = base

        # Initialize the frequencies for computing the rotation angles
        theta = 1.0 / (
            base ** (torch.arange(0, head_dim, 2)[: (head_dim // 2)].float() / head_dim)
        )

        # Create position indices [0, 1, 2, ..., max_seq_len - 1]
        seq_idx = torch.arange(max_seq_len, dtype=theta.dtype, device=theta.device)

        # Compute the rotation angles. Shape: (max_seq_len, head_dim // 2)
        angles = torch.einsum("i , j -> i j", seq_idx, theta)

        # Create embedding matrix by stacking cosine and sine of the angles
        emb = torch.stack([torch.cos(angles), torch.sin(angles)], dim=-1)
        self.register_buffer("emb", emb, persistent=False)
        self.emb: torch.Tensor

    def forward(
        self, x: torch.Tensor, input_pos: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        Apply rotary positional embeddings to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_heads, seq_len, head_dim).
            input_pos (torch.Tensor | None): Optional tensor specifying positions for each element in the sequence.
                If none, assume the index of the token is its position id. Default is None

        Returns:
            torch.Tensor: Tensor with rotary positional embeddings applied.
        """
        seq_len = x.size(-2)

        # Get the embeddings for the current sequence length. Shape: (seq_len, head_dim // 2, 2)
        emb = self.emb[input_pos] if input_pos is not None else self.emb[:seq_len]

        # Split the input tensor into two parts across the last dimension for rotation
        x = rearrange(x, "b n s (d x) -> b n s d x", x=2)

        # Apply the rotary embeddings
        x_rope = torch.stack(
            [
                x[..., 0] * emb[..., 0] - x[..., 1] * emb[..., 1],
                x[..., 1] * emb[..., 0] + x[..., 0] * emb[..., 1],
            ],
            -1,
        )

        # Rearrange back to the original shape of the input tensor
        x_rope = rearrange(x_rope, "b n s d x -> b n s (d x)")

        return x_rope
