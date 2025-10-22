"""
Feed-Forward network for transformer models.
"""

import torch
import torch.nn as nn


class LLaMAFeedForward(nn.Module):
    """
    Implementation of Feed-Forward network for LLaMA models.
    """

    def __init__(self, dim: int) -> None:
        """
        Initializes the LLaMAFeedForward module.

        Args:
            dim (int): The input and output dimension of the feed-forward network.
        """
        super().__init__()

        self.dim = dim

        # Compute the hidden dimension as per LLaMA architecture
        # hidden_dim = 4 * ((2/3) * dim) rounded up to the nearest multiple of 64
        hidden_dim = 4 * int(2 * dim / 3)
        hidden_dim = 64 * ((hidden_dim + 63) // 64)
        self.hidden_dim = hidden_dim

        # Linear projections
        self.up_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.gate_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=False)

        # Activation
        self.activation = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the LLaMAFeedForward module.

        Args:
            x (torch.Tensor): Input tensor of shape (..., dim).

        Returns:
            torch.Tensor: Output tensor of shape (..., dim).
        """
        hidden = self.up_proj(x) * self.activation(self.gate_proj(x))
        output = self.down_proj(hidden)

        return output
