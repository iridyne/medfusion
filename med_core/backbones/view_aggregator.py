"""
Multi-view aggregation modules.

Provides various strategies for aggregating features from multiple views
into a single representation.

Available aggregators:
- MaxPoolAggregator: Max pooling across views
- MeanPoolAggregator: Mean pooling with optional masking
- AttentionAggregator: Learnable attention weights
- CrossViewAttentionAggregator: Cross-view interaction with self-attention
- LearnedWeightAggregator: Fixed learnable weights per view
"""

from abc import ABC, abstractmethod
from typing import Any, Literal

import torch
from torch import nn

from med_core.datasets.multiview_types import ViewTensor


class BaseViewAggregator(ABC, nn.Module):
    """
    Abstract base class for view aggregation modules.

    All aggregators must implement the forward method that takes
    features from multiple views and produces a single aggregated representation.
    """

    def __init__(self, feature_dim: int):
        """
        Initialize aggregator.

        Args:
            feature_dim: Dimension of input/output features
        """
        super().__init__()
        self.feature_dim = feature_dim

    @abstractmethod
    def forward(
        self,
        view_features: ViewTensor,
        view_mask: dict[str, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        Aggregate features from multiple views.

        Args:
            view_features: Dict of view_name -> features (B, feature_dim)
            view_mask: Optional dict of view_name -> mask (B,) indicating valid views

        Returns:
            Tuple of:
                - aggregated_features: (B, feature_dim)
                - aux_outputs: Dict with auxiliary outputs (e.g., attention weights)
        """

    def _stack_views(
        self,
        view_features: ViewTensor,
        view_mask: dict[str, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, list[str], torch.Tensor | None]:
        """
        Helper to stack view features into a tensor.

        Args:
            view_features: Dict of view features
            view_mask: Optional view masks

        Returns:
            Tuple of:
                - stacked: (B, num_views, feature_dim)
                - view_names: List of view names in order
                - masks: (B, num_views) or None
        """
        view_names = sorted(view_features.keys())
        features_list = [view_features[name] for name in view_names]
        stacked = torch.stack(features_list, dim=1)  # (B, num_views, D)

        if view_mask is not None:
            mask_list = []
            for name in view_names:
                if name in view_mask:
                    mask_list.append(view_mask[name])
                else:
                    # Default to all valid
                    mask_list.append(
                        torch.ones(
                            stacked.size(0), device=stacked.device, dtype=torch.bool,
                        ),
                    )
            masks = torch.stack(mask_list, dim=1)  # (B, num_views)
        else:
            masks = None

        return stacked, view_names, masks


class MaxPoolAggregator(BaseViewAggregator):
    """
    Max pooling aggregation across views.

    Takes the maximum value across all views for each feature dimension.
    Simple but effective for capturing the most prominent features.
    """

    def forward(
        self, view_features: ViewTensor, view_mask: dict[str, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        stacked, view_names, masks = self._stack_views(view_features, view_mask)

        # Apply mask if provided
        if masks is not None:
            # Set masked positions to -inf so they don't affect max
            stacked = stacked.masked_fill(~masks.unsqueeze(-1), float("-inf"))

        # Max pooling across views
        aggregated, max_indices = torch.max(stacked, dim=1)  # (B, D)

        return aggregated, {"max_indices": max_indices}


class MeanPoolAggregator(BaseViewAggregator):
    """
    Mean pooling aggregation with optional masking.

    Computes the average across all views, properly handling missing views
    through masking.
    """

    def forward(
        self, view_features: ViewTensor, view_mask: dict[str, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        stacked, view_names, masks = self._stack_views(view_features, view_mask)

        if masks is not None:
            # Masked mean
            masks_expanded = masks.unsqueeze(-1).float()  # (B, num_views, 1)
            masked_features = stacked * masks_expanded
            aggregated = masked_features.sum(dim=1) / masks_expanded.sum(dim=1).clamp(
                min=1,
            )
        else:
            # Simple mean
            aggregated = stacked.mean(dim=1)  # (B, D)

        return aggregated, {}


class AttentionAggregator(BaseViewAggregator):
    """
    Attention-based aggregation.

    Uses multi-head attention to learn importance weights for each view.
    A learnable query vector attends to all view features.
    """

    def __init__(self, feature_dim: int, num_heads: int = 4, dropout: float = 0.1):
        """
        Initialize attention aggregator.

        Args:
            feature_dim: Feature dimension
            num_heads: Number of attention heads
            dropout: Dropout rate
        """
        super().__init__(feature_dim)
        self.num_heads = num_heads

        # Multi-head attention
        self.attention = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        # Learnable query vector
        self.query = nn.Parameter(torch.randn(1, 1, feature_dim))

        # Layer norm
        self.norm = nn.LayerNorm(feature_dim)

    def forward(
        self, view_features: ViewTensor, view_mask: dict[str, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        batch_size = next(iter(view_features.values())).size(0)
        stacked, view_names, masks = self._stack_views(view_features, view_mask)

        # Expand query for batch
        query = self.query.expand(batch_size, -1, -1)  # (B, 1, D)

        # Attention
        aggregated, attn_weights = self.attention(
            query,
            stacked,
            stacked,
            key_padding_mask=~masks if masks is not None else None,
            need_weights=True,
        )  # aggregated: (B, 1, D), attn_weights: (B, 1, num_views)

        aggregated = aggregated.squeeze(1)  # (B, D)
        aggregated = self.norm(aggregated)

        # Build attention weight dict
        attn_dict = {}
        for i, view_name in enumerate(view_names):
            attn_dict[view_name] = attn_weights[:, 0, i]  # (B,)

        return aggregated, {"attention_weights": attn_dict}


class CrossViewAttentionAggregator(BaseViewAggregator):
    """
    Cross-view attention aggregation.

    Allows views to interact with each other through self-attention
    before aggregation. Uses a CLS token for final aggregation.
    """

    def __init__(self, feature_dim: int, num_heads: int = 4, dropout: float = 0.1):
        """
        Initialize cross-view attention aggregator.

        Args:
            feature_dim: Feature dimension
            num_heads: Number of attention heads
            dropout: Dropout rate
        """
        super().__init__(feature_dim)

        # Self-attention for cross-view interaction
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        # Layer norms
        self.norm1 = nn.LayerNorm(feature_dim)
        self.norm2 = nn.LayerNorm(feature_dim)

        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(feature_dim, feature_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feature_dim * 4, feature_dim),
            nn.Dropout(dropout),
        )

        # CLS token for aggregation
        self.cls_token = nn.Parameter(torch.randn(1, 1, feature_dim))

    def forward(
        self, view_features: ViewTensor, view_mask: dict[str, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        batch_size = next(iter(view_features.values())).size(0)
        stacked, view_names, masks = self._stack_views(view_features, view_mask)

        # Add CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # (B, 1, D)
        tokens = torch.cat([cls_tokens, stacked], dim=1)  # (B, num_views+1, D)

        # Extend mask for CLS token (always valid)
        if masks is not None:
            cls_mask = torch.ones(batch_size, 1, device=masks.device, dtype=torch.bool)
            extended_mask = torch.cat([cls_mask, masks], dim=1)
        else:
            extended_mask = None

        # Self-attention across views
        attended, _ = self.cross_attention(
            tokens,
            tokens,
            tokens,
            key_padding_mask=~extended_mask if extended_mask is not None else None,
        )
        tokens = self.norm1(tokens + attended)

        # Feed-forward
        tokens = self.norm2(tokens + self.ffn(tokens))

        # Extract CLS token as aggregated representation
        aggregated = tokens[:, 0, :]  # (B, D)

        return aggregated, {}


class LearnedWeightAggregator(BaseViewAggregator):
    """
    Learnable weight aggregation.

    Assigns a learnable scalar weight to each view and computes
    a weighted sum. Weights are normalized with softmax.
    """

    def __init__(self, feature_dim: int, view_names: list[str]):
        """
        Initialize learned weight aggregator.

        Args:
            feature_dim: Feature dimension
            view_names: List of view names (order matters)
        """
        super().__init__(feature_dim)
        self.view_names = sorted(view_names)  # Ensure consistent ordering
        self.num_views = len(self.view_names)

        # Learnable weights (one per view)
        self.view_weights = nn.Parameter(torch.ones(self.num_views) / self.num_views)

    def forward(
        self, view_features: ViewTensor, view_mask: dict[str, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        stacked, view_names, masks = self._stack_views(view_features, view_mask)

        # Ensure view order matches initialization
        if view_names != self.view_names:
            raise ValueError(
                f"View names mismatch. Expected {self.view_names}, got {view_names}",
            )

        # Softmax weights
        weights = torch.softmax(self.view_weights, dim=0)  # (num_views,)
        weights = weights.view(1, -1, 1)  # (1, num_views, 1)

        # Apply mask to weights if provided
        if masks is not None:
            weights = weights * masks.unsqueeze(-1).float()
            # Renormalize
            weights = weights / weights.sum(dim=1, keepdim=True).clamp(min=1e-8)

        # Weighted sum
        aggregated = (stacked * weights).sum(dim=1)  # (B, D)

        # Build weight dict
        weight_dict = {}
        for i, view_name in enumerate(view_names):
            weight_dict[view_name] = weights[0, i, 0].item()

        return aggregated, {"view_weights": weight_dict}


# Registry for factory function
AGGREGATOR_REGISTRY = {
    "max": MaxPoolAggregator,
    "mean": MeanPoolAggregator,
    "attention": AttentionAggregator,
    "cross_attention": CrossViewAttentionAggregator,
    "learned_weight": LearnedWeightAggregator,
}


def create_view_aggregator(
    aggregator_type: Literal[
        "max", "mean", "attention", "cross_attention", "learned_weight",
    ],
    feature_dim: int,
    **kwargs: Any,
) -> BaseViewAggregator:
    """
    Factory function to create view aggregators.

    Args:
        aggregator_type: Type of aggregator
        feature_dim: Feature dimension
        **kwargs: Additional arguments for specific aggregators
            - num_heads: For attention-based aggregators
            - dropout: For attention-based aggregators
            - view_names: For learned_weight aggregator

    Returns:
        View aggregator instance

    Example:
        >>> aggregator = create_view_aggregator(
        ...     "attention",
        ...     feature_dim=128,
        ...     num_heads=4,
        ... )
    """
    if aggregator_type not in AGGREGATOR_REGISTRY:
        available = list(AGGREGATOR_REGISTRY.keys())
        raise ValueError(
            f"Unknown aggregator type: {aggregator_type}. Available: {available}",
        )

    aggregator_cls = AGGREGATOR_REGISTRY[aggregator_type]

    # Filter kwargs based on aggregator type
    filtered_kwargs = {}
    if aggregator_type in ["attention", "cross_attention"]:
        # Attention-based aggregators accept num_heads and dropout
        for key in ("num_heads", "dropout"):
            if key in kwargs:
                filtered_kwargs[key] = kwargs[key]
    elif aggregator_type == "learned_weight":
        # LearnedWeightAggregator accepts view_names
        if "view_names" in kwargs:
            filtered_kwargs["view_names"] = kwargs["view_names"]
    # max and mean aggregators don't accept any additional kwargs

    return aggregator_cls(feature_dim=feature_dim, **filtered_kwargs)


def list_available_aggregators() -> list[str]:
    """Return list of available aggregator names."""
    return list(AGGREGATOR_REGISTRY.keys())
