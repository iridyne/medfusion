"""
Fusion strategy implementations for multimodal learning.

Provides various strategies for combining vision and tabular features:
- ConcatenateFusion: Simple concatenation
- GatedFusion: Learnable gating mechanism
- AttentionFusion: Self-attention based fusion
- CrossAttentionFusion: Cross-modal attention
- BilinearFusion: Bilinear pooling
"""

from typing import Literal

import torch
import torch.nn as nn

from med_core.fusion.base import BaseFusion


class ConcatenateFusion(BaseFusion):
    """
    Simple concatenation fusion.

    Concatenates features from all modalities and projects
    to the output dimension.
    """

    def __init__(
        self,
        vision_dim: int,
        tabular_dim: int,
        output_dim: int = 96,
        dropout: float = 0.3,
    ):
        """
        Initialize concatenation fusion.

        Args:
            vision_dim: Dimension of vision features
            tabular_dim: Dimension of tabular features
            output_dim: Output dimension after fusion
            dropout: Dropout rate
        """
        super().__init__(output_dim=output_dim)

        self.vision_dim = vision_dim
        self.tabular_dim = tabular_dim
        concat_dim = vision_dim + tabular_dim

        self.projection = nn.Sequential(
            nn.Linear(concat_dim, output_dim),
            nn.ReLU(),
            nn.LayerNorm(output_dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        vision_features: torch.Tensor,
        tabular_features: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor] | None]:
        """Concatenate and project features."""
        concat = torch.cat([vision_features, tabular_features], dim=1)
        fused = self.projection(concat)
        return fused, None


class GatedFusion(BaseFusion):
    """
    Gated fusion with learnable modality weights.

    Uses a gating mechanism to learn the optimal balance
    between vision and tabular modalities.

    Features:
    - Learnable global modality weights (alpha, beta)
    - Instance-specific gating via sigmoid
    - Smooth interpolation between modalities
    """

    def __init__(
        self,
        vision_dim: int,
        tabular_dim: int,
        output_dim: int = 96,
        initial_vision_weight: float = 0.3,
        initial_tabular_weight: float = 0.7,
        learnable_weights: bool = True,
        dropout: float = 0.3,
    ):
        """
        Initialize gated fusion.

        Args:
            vision_dim: Dimension of vision features
            tabular_dim: Dimension of tabular features
            output_dim: Output dimension after fusion
            initial_vision_weight: Initial weight for vision modality
            initial_tabular_weight: Initial weight for tabular modality
            learnable_weights: Whether global weights are learnable
            dropout: Dropout rate
        """
        super().__init__(output_dim=output_dim)

        self.vision_dim = vision_dim
        self.tabular_dim = tabular_dim

        # Project modalities to same dimension
        self.vision_proj = nn.Linear(vision_dim, output_dim)
        self.tabular_proj = nn.Linear(tabular_dim, output_dim)

        # Gate network
        self.gate = nn.Linear(vision_dim + tabular_dim, output_dim)

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # Learnable global weights
        if learnable_weights:
            self.alpha_vision = nn.Parameter(torch.tensor(initial_vision_weight))
            self.alpha_tabular = nn.Parameter(torch.tensor(initial_tabular_weight))
        else:
            self.register_buffer("alpha_vision", torch.tensor(initial_vision_weight))
            self.register_buffer("alpha_tabular", torch.tensor(initial_tabular_weight))

        self._last_gate_values: torch.Tensor | None = None

    def forward(
        self,
        vision_features: torch.Tensor,
        tabular_features: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor] | None]:
        """Apply gated fusion."""
        # Project to same dimension with tanh activation
        h_vision = torch.tanh(self.vision_proj(vision_features))
        h_tabular = torch.tanh(self.tabular_proj(tabular_features))

        # Compute instance-specific gate
        gate_input = torch.cat([vision_features, tabular_features], dim=1)
        z = torch.sigmoid(self.gate(gate_input))

        # Compute global weights (sigmoid to ensure [0, 1])
        w_vision = torch.sigmoid(self.alpha_vision)
        w_tabular = torch.sigmoid(self.alpha_tabular)

        # Gated combination
        fused = w_vision * z * h_vision + w_tabular * (1 - z) * h_tabular

        # Apply dropout
        fused = self.dropout(fused)

        # Store gate values for analysis (only in eval mode to prevent memory leak)
        if not self.training:
            self._last_gate_values = z.detach()

        aux_outputs = {
            "gate_values": z,
            "vision_weight": w_vision,
            "tabular_weight": w_tabular,
        }

        return fused, aux_outputs

    def get_modality_weights(self) -> dict[str, float]:
        """Get current modality weight values."""
        return {
            "vision_weight": torch.sigmoid(self.alpha_vision).item(),
            "tabular_weight": torch.sigmoid(self.alpha_tabular).item(),
        }


class AttentionFusion(BaseFusion):
    """
    Self-attention based fusion.

    Treats each modality as a token and applies self-attention
    to learn cross-modal interactions.
    """

    def __init__(
        self,
        vision_dim: int,
        tabular_dim: int,
        output_dim: int = 96,
        num_heads: int = 4,
        dropout: float = 0.3,
    ):
        """
        Initialize attention fusion.

        Args:
            vision_dim: Dimension of vision features
            tabular_dim: Dimension of tabular features
            output_dim: Output dimension after fusion
            num_heads: Number of attention heads
            dropout: Dropout rate
        """
        super().__init__(output_dim=output_dim)

        self.vision_dim = vision_dim
        self.tabular_dim = tabular_dim

        # Project to common dimension
        self.vision_proj = nn.Linear(vision_dim, output_dim)
        self.tabular_proj = nn.Linear(tabular_dim, output_dim)

        # Multi-head self-attention
        self.self_attention = nn.MultiheadAttention(
            embed_dim=output_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        # Layer norm and feed-forward
        self.norm1 = nn.LayerNorm(output_dim)
        self.norm2 = nn.LayerNorm(output_dim)

        self.ffn = nn.Sequential(
            nn.Linear(output_dim, output_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim * 4, output_dim),
            nn.Dropout(dropout),
        )

        # Learnable [CLS] token for aggregation
        self.cls_token = nn.Parameter(torch.randn(1, 1, output_dim))

        self._last_attention_weights: torch.Tensor | None = None

    def forward(
        self,
        vision_features: torch.Tensor,
        tabular_features: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor] | None]:
        """Apply attention-based fusion."""
        batch_size = vision_features.size(0)

        # Project features
        h_vision = self.vision_proj(vision_features).unsqueeze(1)  # (B, 1, D)
        h_tabular = self.tabular_proj(tabular_features).unsqueeze(1)  # (B, 1, D)

        # Expand CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # (B, 1, D)

        # Concatenate: [CLS, vision, tabular]
        tokens = torch.cat([cls_tokens, h_vision, h_tabular], dim=1)  # (B, 3, D)

        # Self-attention
        attended, attn_weights = self.self_attention(
            tokens, tokens, tokens, need_weights=True
        )

        # Residual + LayerNorm
        tokens = self.norm1(tokens + attended)

        # FFN with residual
        tokens = self.norm2(tokens + self.ffn(tokens))

        # Extract CLS token as fused representation
        fused = tokens[:, 0, :]  # (B, D)

        # Store attention weights (only in eval mode to prevent memory leak)
        if not self.training:
            self._last_attention_weights = attn_weights.detach()

        aux_outputs = {"attention_weights": attn_weights}

        return fused, aux_outputs

    def get_attention_weights(self) -> torch.Tensor | None:
        """Get last computed attention weights."""
        return self._last_attention_weights


class CrossAttentionFusion(BaseFusion):
    """
    Cross-attention fusion between modalities.

    Vision attends to tabular and vice versa,
    capturing cross-modal dependencies.
    """

    def __init__(
        self,
        vision_dim: int,
        tabular_dim: int,
        output_dim: int = 96,
        num_heads: int = 4,
        dropout: float = 0.3,
    ):
        """
        Initialize cross-attention fusion.

        Args:
            vision_dim: Dimension of vision features
            tabular_dim: Dimension of tabular features
            output_dim: Output dimension after fusion
            num_heads: Number of attention heads
            dropout: Dropout rate
        """
        super().__init__(output_dim=output_dim)

        self.vision_dim = vision_dim
        self.tabular_dim = tabular_dim

        # Project to common dimension
        self.vision_proj = nn.Linear(vision_dim, output_dim)
        self.tabular_proj = nn.Linear(tabular_dim, output_dim)

        # Cross-attention: vision attends to tabular
        self.vision_to_tabular = nn.MultiheadAttention(
            embed_dim=output_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        # Cross-attention: tabular attends to vision
        self.tabular_to_vision = nn.MultiheadAttention(
            embed_dim=output_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        # Layer norms
        self.norm_v2t = nn.LayerNorm(output_dim)
        self.norm_t2v = nn.LayerNorm(output_dim)

        # Final projection
        self.output_proj = nn.Sequential(
            nn.Linear(output_dim * 2, output_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self._v2t_attn: torch.Tensor | None = None
        self._t2v_attn: torch.Tensor | None = None

    def forward(
        self,
        vision_features: torch.Tensor,
        tabular_features: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor] | None]:
        """Apply cross-attention fusion."""
        # Project features
        h_vision = self.vision_proj(vision_features).unsqueeze(1)  # (B, 1, D)
        h_tabular = self.tabular_proj(tabular_features).unsqueeze(1)  # (B, 1, D)

        # Vision attends to tabular
        v2t, v2t_attn = self.vision_to_tabular(
            h_vision, h_tabular, h_tabular, need_weights=True
        )
        v2t = self.norm_v2t(h_vision + v2t)

        # Tabular attends to vision
        t2v, t2v_attn = self.tabular_to_vision(
            h_tabular, h_vision, h_vision, need_weights=True
        )
        t2v = self.norm_t2v(h_tabular + t2v)

        # Squeeze and concatenate
        v2t = v2t.squeeze(1)  # (B, D)
        t2v = t2v.squeeze(1)  # (B, D)

        concat = torch.cat([v2t, t2v], dim=1)  # (B, 2D)
        fused = self.output_proj(concat)  # (B, D)

        # Store attention weights (only in eval mode to prevent memory leak)
        if not self.training:
            self._v2t_attn = v2t_attn.detach()
            self._t2v_attn = t2v_attn.detach()

        aux_outputs = {
            "vision_to_tabular_attn": v2t_attn,
            "tabular_to_vision_attn": t2v_attn,
        }

        return fused, aux_outputs


class BilinearFusion(BaseFusion):
    """
    Bilinear pooling fusion.

    Captures multiplicative interactions between modalities
    through outer product operations.

    Uses low-rank approximation for efficiency.
    """

    def __init__(
        self,
        vision_dim: int,
        tabular_dim: int,
        output_dim: int = 96,
        rank: int = 16,
        dropout: float = 0.3,
    ):
        """
        Initialize bilinear fusion.

        Args:
            vision_dim: Dimension of vision features
            tabular_dim: Dimension of tabular features
            output_dim: Output dimension after fusion
            rank: Rank for low-rank approximation
            dropout: Dropout rate
        """
        super().__init__(output_dim=output_dim)

        self.vision_dim = vision_dim
        self.tabular_dim = tabular_dim
        self.rank = rank

        # Low-rank factorization matrices
        self.U = nn.Linear(vision_dim, rank, bias=False)
        self.V = nn.Linear(tabular_dim, rank, bias=False)
        self.P = nn.Linear(rank, output_dim)

        # Residual paths
        self.vision_skip = nn.Linear(vision_dim, output_dim)
        self.tabular_skip = nn.Linear(tabular_dim, output_dim)

        # Output layers
        self.norm = nn.BatchNorm1d(output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        vision_features: torch.Tensor,
        tabular_features: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor] | None]:
        """Apply bilinear fusion with low-rank approximation."""
        # Low-rank bilinear: P(U*v ⊙ V*t)
        u = self.U(vision_features)  # (B, rank)
        v = self.V(tabular_features)  # (B, rank)

        bilinear = u * v  # Element-wise product (B, rank)
        bilinear = self.P(bilinear)  # (B, output_dim)

        # Add skip connections
        skip_v = self.vision_skip(vision_features)
        skip_t = self.tabular_skip(tabular_features)

        fused = bilinear + skip_v + skip_t
        fused = self.norm(fused)
        fused = self.dropout(fused)

        return fused, None


# Factory function for creating fusion modules
FUSION_REGISTRY = {
    "concatenate": ConcatenateFusion,
    "gated": GatedFusion,
    "attention": AttentionFusion,
    "cross_attention": CrossAttentionFusion,
    "bilinear": BilinearFusion,
}


def create_fusion_module(
    fusion_type: Literal[
        "concatenate", "gated", "attention", "cross_attention", "bilinear"
    ],
    vision_dim: int,
    tabular_dim: int,
    output_dim: int = 96,
    **kwargs,
) -> BaseFusion:
    """
    Factory function to create fusion modules.

    Args:
        fusion_type: Type of fusion strategy
        vision_dim: Dimension of vision features
        tabular_dim: Dimension of tabular features
        output_dim: Output dimension after fusion
        **kwargs: Additional arguments passed to fusion constructor

    Returns:
        Fusion module instance

    Example:
        >>> fusion = create_fusion_module(
        ...     "gated",
        ...     vision_dim=128,
        ...     tabular_dim=32,
        ...     output_dim=96,
        ...     initial_vision_weight=0.3,
        ... )
    """
    if fusion_type not in FUSION_REGISTRY:
        available = list(FUSION_REGISTRY.keys())
        raise ValueError(f"Unknown fusion type: {fusion_type}. Available: {available}")

    fusion_cls = FUSION_REGISTRY[fusion_type]

    return fusion_cls(
        vision_dim=vision_dim,
        tabular_dim=tabular_dim,
        output_dim=output_dim,
        **kwargs,
    )


def list_available_fusions() -> list[str]:
    """Return list of available fusion strategy names."""
    return list(FUSION_REGISTRY.keys())
