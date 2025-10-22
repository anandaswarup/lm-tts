"""
Layer normalization for transformer models.
"""

import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    """
    Root mean square layer normalization in fp32.
    """

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        """
        Initializes the RMSNorm module.

        Args:
            dim (int): The dimension of the input tensor.
            eps (float): A small value to avoid division by zero.
        """
        super().__init__()

        self.dim = dim
        self.eps = eps

        # Learnable scale parameter
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the RMSNorm module.

        Args:
            x (torch.Tensor): Input tensor of shape (..., dim).

        Returns:
            torch.Tensor: Normalized and scaled tensor of shape (..., dim).
        """
        # Computation in fp32 for numerical stability
        x_fp32 = x.float()

        # Compute RMS normalization and apply to input tensor; and cast back to original dtype
        x_normed = (
            x_fp32 * torch.rsqrt(x_fp32.pow(2).mean(-1, keepdim=True) + self.eps)
        ).type_as(x)

        # Scale the normalized tensor by the learnable scale parameter
        output = x_normed * self.scale

        return output
