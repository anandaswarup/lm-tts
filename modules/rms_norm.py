"""
Root mean square layer normalization module.
"""

import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    """
    This class implements the Root Mean Square Layer Normalization (RMSNorm) in fp32 precision.
    """

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        """
        Initializes the RMSNorm module.

        Args:
            dim (int): Dimension of the input tensor to be normalized.
            eps (float): A small value to avoid division by zero. Default is 1e-6.
        """
        super().__init__()

        self.normalized_shape = (dim,)
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the RMSNorm module.

        Args:
            x (torch.Tensor): Input tensor of shape (..., dim), where dim is the dimension to be normalized.

        Returns:
            torch.Tensor: Normalized tensor of the same shape as input.
        """
        # Convert to fp32
        x_fp32 = x.float()

        # Normalize input tensor
        x_normed = (
            x_fp32 * torch.rsqrt(x_fp32.pow(2).mean(-1, keepdim=True) + self.eps)
        ).type_as(x)

        # Scale normalized tensor
        output = x_normed * self.scale

        return output
