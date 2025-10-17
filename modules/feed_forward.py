"""
Feed-Forward network module.
"""

import torch
import torch.nn as nn


class FeedForward(nn.Module):
    """
    This class implements the Feed-Forward network for LLaMA.
    """

    def __init__(
        self,
        gate_proj: nn.Module,
        up_proj: nn.Module,
        down_proj: nn.Module,
        activation: nn.Module = nn.SiLU(),
    ) -> None:
        """
        Initializes the Feed-Forward network.
        """
        super().__init__()

        self.gate_proj = gate_proj
        self.up_proj = up_proj
        self.down_proj = down_proj
        self.activation = activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the Feed-Forward network.

        Args:
            x (torch.Tensor): Input tensor of shape (..., in_dim), where in_dim is the input dimension of both
                gate_proj and up_proj.

        Returns:
            torch.Tensor: Output tensor of shape (..., out_dim), where out_dim is the output dimension of down_proj.
        """
        hidden = self.activation(self.gate_proj(x)) * self.up_proj(x)
        output = self.down_proj(hidden)

        return output
