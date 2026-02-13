"""
Multiple Instance Learning (MIL) aggregators.

This module provides various aggregation strategies for combining
features from multiple instances (e.g., multiple pathology patches,
multiple regions of interest).
"""

from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F


class MeanPoolingAggregator(nn.Module):
    """
    Simple mean pooling aggregator.

    Averages features across all instances.

    Args:
        input_dim: Input feature dimension

    Example:
        >>> aggregator = MeanPoolingAggregator(input_dim=512)
        >>> features = torch.randn(4, 10, 512)  # [B, N, D]
        >>> aggregated = aggregator(features)    # [4, 512]
    """

    def __init__(self, input_dim: int):
        super().__init__()
        self.input_dim = input_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Instance features [B, N, input_dim]

        Returns:
            Aggregated features [B, input_dim]
        """
        return x.mean(dim=1)


class MaxPoolingAggregator(nn.Module):
    """
    Max pooling aggregator.

    Takes the maximum value across all instances for each feature dimension.

    Args:
        input_dim: Input feature dimension

    Example:
        >>> aggregator = MaxPoolingAggregator(input_dim=512)
        >>> features = torch.randn(4, 10, 512)
        >>> aggregated = aggregator(features)  # [4, 512]
    """

    def __init__(self, input_dim: int):
        super().__init__()
        self.input_dim = input_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Instance features [B, N, input_dim]

        Returns:
            Aggregated features [B, input_dim]
        """
        return x.max(dim=1)[0]


class AttentionAggregator(nn.Module):
    """
    Attention-based MIL aggregator.

    Uses attention mechanism to weight instances based on their importance.

    Args:
        input_dim: Input feature dimension
        attention_dim: Attention hidden dimension
        dropout: Dropout rate

    Example:
        >>> aggregator = AttentionAggregator(input_dim=512, attention_dim=128)
        >>> features = torch.randn(4, 10, 512)
        >>> aggregated, weights = aggregator(features, return_attention=True)
    """

    def __init__(
        self,
        input_dim: int,
        attention_dim: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.attention_dim = attention_dim

        # Attention network
        self.attention = nn.Sequential(
            nn.Linear(input_dim, attention_dim),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(attention_dim, 1),
        )

    def forward(
        self, x: torch.Tensor, return_attention: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            x: Instance features [B, N, input_dim]
            return_attention: Return attention weights

        Returns:
            Aggregated features [B, input_dim]
            If return_attention=True, also returns attention weights [B, N, 1]
        """
        # Compute attention weights
        attention_logits = self.attention(x)  # [B, N, 1]
        attention_weights = F.softmax(attention_logits, dim=1)  # [B, N, 1]

        # Weighted sum
        aggregated = (x * attention_weights).sum(dim=1)  # [B, input_dim]

        if return_attention:
            return aggregated, attention_weights
        return aggregated


class GatedAttentionAggregator(nn.Module):
    """
    Gated attention MIL aggregator.

    Uses gating mechanism in addition to attention for more expressive aggregation.

    Args:
        input_dim: Input feature dimension
        attention_dim: Attention hidden dimension
        dropout: Dropout rate

    Example:
        >>> aggregator = GatedAttentionAggregator(input_dim=512)
        >>> features = torch.randn(4, 10, 512)
        >>> aggregated = aggregator(features)  # [4, 512]
    """

    def __init__(
        self,
        input_dim: int,
        attention_dim: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.attention_dim = attention_dim

        # Attention branch
        self.attention_V = nn.Sequential(
            nn.Linear(input_dim, attention_dim),
            nn.Tanh(),
            nn.Dropout(dropout),
        )

        # Gate branch
        self.attention_U = nn.Sequential(
            nn.Linear(input_dim, attention_dim),
            nn.Sigmoid(),
            nn.Dropout(dropout),
        )

        # Attention weights
        self.attention_w = nn.Linear(attention_dim, 1)

    def forward(
        self, x: torch.Tensor, return_attention: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            x: Instance features [B, N, input_dim]
            return_attention: Return attention weights

        Returns:
            Aggregated features [B, input_dim]
            If return_attention=True, also returns attention weights
        """
        # Gated attention
        A_V = self.attention_V(x)  # [B, N, attention_dim]
        A_U = self.attention_U(x)  # [B, N, attention_dim]
        A = A_V * A_U  # Element-wise multiplication (gating)

        # Attention weights
        attention_logits = self.attention_w(A)  # [B, N, 1]
        attention_weights = F.softmax(attention_logits, dim=1)  # [B, N, 1]

        # Weighted sum
        aggregated = (x * attention_weights).sum(dim=1)  # [B, input_dim]

        if return_attention:
            return aggregated, attention_weights
        return aggregated


class DeepSetsAggregator(nn.Module):
    """
    Deep Sets aggregator.

    Implements the Deep Sets architecture for permutation-invariant aggregation.

    Args:
        input_dim: Input feature dimension
        hidden_dim: Hidden dimension
        output_dim: Output dimension
        dropout: Dropout rate

    Example:
        >>> aggregator = DeepSetsAggregator(input_dim=512, output_dim=256)
        >>> features = torch.randn(4, 10, 512)
        >>> aggregated = aggregator(features)  # [4, 256]
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        output_dim: int | None = None,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim or input_dim

        # Phi network (applied to each instance)
        self.phi = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

        # Rho network (applied to aggregated features)
        self.rho = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, self.output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Instance features [B, N, input_dim]

        Returns:
            Aggregated features [B, output_dim]
        """
        # Apply phi to each instance
        phi_x = self.phi(x)  # [B, N, hidden_dim]

        # Sum pooling (permutation invariant)
        pooled = phi_x.sum(dim=1)  # [B, hidden_dim]

        # Apply rho
        output = self.rho(pooled)  # [B, output_dim]

        return output


class TransformerAggregator(nn.Module):
    """
    Transformer-based aggregator.

    Uses transformer encoder to aggregate instance features.

    Args:
        input_dim: Input feature dimension
        num_heads: Number of attention heads
        num_layers: Number of transformer layers
        dropout: Dropout rate

    Example:
        >>> aggregator = TransformerAggregator(input_dim=512, num_heads=8)
        >>> features = torch.randn(4, 10, 512)
        >>> aggregated = aggregator(features)  # [4, 512]
    """

    def __init__(
        self,
        input_dim: int,
        num_heads: int = 8,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.num_heads = num_heads
        self.num_layers = num_layers

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim,
            nhead=num_heads,
            dim_feedforward=input_dim * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        # Learnable query token for aggregation
        self.query_token = nn.Parameter(torch.randn(1, 1, input_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Instance features [B, N, input_dim]

        Returns:
            Aggregated features [B, input_dim]
        """
        batch_size = x.size(0)

        # Expand query token
        query = self.query_token.expand(batch_size, -1, -1)  # [B, 1, input_dim]

        # Concatenate query with instances
        x_with_query = torch.cat([query, x], dim=1)  # [B, N+1, input_dim]

        # Apply transformer
        transformed = self.transformer(x_with_query)  # [B, N+1, input_dim]

        # Extract query token (aggregated representation)
        aggregated = transformed[:, 0, :]  # [B, input_dim]

        return aggregated


class MILAggregator(nn.Module):
    """
    Unified MIL aggregator with multiple strategies.

    Args:
        input_dim: Input feature dimension
        strategy: Aggregation strategy
        attention_dim: Attention hidden dimension (for attention-based methods)
        hidden_dim: Hidden dimension (for deep sets)
        output_dim: Output dimension (if None, same as input_dim)
        num_heads: Number of attention heads (for transformer)
        num_layers: Number of layers (for transformer)
        dropout: Dropout rate

    Example:
        >>> aggregator = MILAggregator(input_dim=512, strategy='attention')
        >>> features = torch.randn(4, 10, 512)
        >>> aggregated = aggregator(features)  # [4, 512]
    """

    def __init__(
        self,
        input_dim: int,
        strategy: Literal['mean', 'max', 'attention', 'gated', 'deepsets', 'transformer'] = 'attention',
        attention_dim: int = 128,
        hidden_dim: int = 256,
        output_dim: int | None = None,
        num_heads: int = 8,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.strategy = strategy
        self.output_dim = output_dim or input_dim

        # Create aggregator based on strategy
        if strategy == 'mean':
            self.aggregator = MeanPoolingAggregator(input_dim)
        elif strategy == 'max':
            self.aggregator = MaxPoolingAggregator(input_dim)
        elif strategy == 'attention':
            self.aggregator = AttentionAggregator(input_dim, attention_dim, dropout)
        elif strategy == 'gated':
            self.aggregator = GatedAttentionAggregator(input_dim, attention_dim, dropout)
        elif strategy == 'deepsets':
            self.aggregator = DeepSetsAggregator(input_dim, hidden_dim, self.output_dim, dropout)
        elif strategy == 'transformer':
            self.aggregator = TransformerAggregator(input_dim, num_heads, num_layers, dropout)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        # Output projection if needed
        if self.output_dim != input_dim and strategy not in ['deepsets']:
            self.output_proj = nn.Linear(input_dim, self.output_dim)
        else:
            self.output_proj = None

    def forward(
        self, x: torch.Tensor, return_attention: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            x: Instance features [B, N, input_dim]
            return_attention: Return attention weights (only for attention-based methods)

        Returns:
            Aggregated features [B, output_dim]
            If return_attention=True and strategy supports it, also returns attention weights
        """
        # Aggregate
        if return_attention and self.strategy in ['attention', 'gated']:
            aggregated, attention_weights = self.aggregator(x, return_attention=True)
        else:
            aggregated = self.aggregator(x)
            attention_weights = None

        # Project if needed
        if self.output_proj is not None:
            aggregated = self.output_proj(aggregated)

        if return_attention and attention_weights is not None:
            return aggregated, attention_weights
        return aggregated
