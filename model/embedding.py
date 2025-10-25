"""
Embedding layers for transformer models.
"""

import torch
import torch.nn as nn
from jaxtyping import Float, Int


class Embedding(nn.Module):
    """
    Implementation of an embedding layer initialized with a truncated normal distribution.
    """

    def __init__(self, vocab_size: int, d_model: int) -> None:
        """
        Initialize the Embedding layer.

        Args:
            vocab_size (int): The size of the vocabulary.
            d_model (int): The dimensionality of the embeddings.
        """
        super().__init__()

        # Standard deviation for truncated normal distribution
        std = 1.0

        # Initialize embedding weights with truncated normal distribution
        self.weight: Float[torch.Tensor, "vocab_size d_model"] = nn.Parameter(
            nn.init.trunc_normal_(
                torch.empty(vocab_size, d_model), std=std, a=-3 * std, b=3 * std
            ),
            requires_grad=True,
        )

    def forward(
        self, token_ids: Int[torch.Tensor, "..."]
    ) -> Float[torch.Tensor, "... d_model"]:
        """
        Forward pass of the Embedding layer.

        Args:
            token_ids (Int[torch.Tensor, "..."]): Input tensor containing token IDs.

        Returns:
            Float[torch.Tensor, "... d_model"]: Output tensor containing embeddings.
        """
        return self.weight[token_ids, :]
    
    def extra_repr(self) -> str:
        """
        Return a string representation of the Embedding layer.
        """
        return f"vocab_size={self.weight.shape[0]}, d_model={self.weight.shape[1]}"
