"""
Linear layers for transformer models.
"""

import math

import torch
import torch.nn as nn
from einops import einsum
from jaxtyping import Float


class Linear(nn.Module):
    """
    Implementation of a linear layer initialized with truncated normal fan-in fan-out distribution, and no bias.
    """

    def __init__(self, d_in: int, d_out: int) -> None:
        """
        Initialize the Linear layer.

        Args:
            d_in (int): The number of input features.
            d_out (int): The number of output features.
        """
        super().__init__()

        # Calculate standard deviation for truncated normal distribution based on fan-in and fan-out
        std = math.sqrt(2 / (d_in + d_out))

        # Initialize layer weights with truncated normal distribution
        self.weight: Float[torch.Tensor, " d_out d_in"] = nn.Parameter(
            nn.init.trunc_normal_(
                torch.empty(d_out, d_in), std=std, a=-3 * std, b=3 * std
            ),
            requires_grad=True,
        )

    def forward(
        self, x: Float[torch.Tensor, " ... d_in"]
    ) -> Float[torch.Tensor, " ... d_out"]:
        """
        Forward pass of the Linear layer.

        Args:
            x (Float[torch.Tensor, " ... d_in"]): Input tensor with shape (..., d_in).

        Returns:
            Float[torch.Tensor, " ... d_out"]: Output tensor with shape (..., d_out).
        """
        return einsum(x, self.weight, "... d_in, d_out d_in -> ... d_out")

    def extra_repr(self) -> str:
        """
        Return a string representation of the Linear layer.
        """
        return f"d_in={self.weight.shape[1]}, d_out={self.weight.shape[0]}, bias=False"
