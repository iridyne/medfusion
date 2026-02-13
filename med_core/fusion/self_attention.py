"""
Self-Attention Fusion for multimodal feature fusion.

This module implements self-attention based fusion strategies that
allow modalities to attend to each other without explicit cross-modal
attention mechanisms.
"""


import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttentionFusion(nn.Module):
    """
    Self-attention fusion that concatenates features and applies self-attention.

    This is a simpler alternative to cross-modal attention that treats
    all modality features as a sequence and applies self-attention.

    Args:
        dim1: Dimension of first modality features
        dim2: Dimension of second modality features
        output_dim: Output dimension after fusion
        num_heads: Number of attention heads
        dropout: Dropout rate

    Example:
        >>> fusion = SelfAttentionFusion(dim1=512, dim2=512, output_dim=256)
        >>> x1 = torch.randn(4, 512)  # Modality 1
        >>> x2 = torch.randn(4, 512)  # Modality 2
        >>> fused = fusion(x1, x2)  # [4, 256]
    """

    def __init__(
        self,
        dim1: int,
        dim2: int,
        output_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.dim1 = dim1
        self.dim2 = dim2
        self.output_dim = output_dim
        self.num_heads = num_heads

        # Project both modalities to same dimension
        hidden_dim = max(dim1, dim2)
        self.proj1 = nn.Linear(dim1, hidden_dim)
        self.proj2 = nn.Linear(dim2, hidden_dim)

        # Self-attention layer
        self.self_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        # Layer norm
        self.norm = nn.LayerNorm(hidden_dim)

        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim * 2, output_dim),  # *2 because we have 2 tokens
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

    def forward(
        self, x1: torch.Tensor, x2: torch.Tensor, return_attention: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            x1: First modality features [B, dim1]
            x2: Second modality features [B, dim2]
            return_attention: Return attention weights

        Returns:
            Fused features [B, output_dim]
            If return_attention=True, also returns attention weights
        """
        batch_size = x1.size(0)

        # Project to same dimension
        x1_proj = self.proj1(x1)  # [B, hidden_dim]
        x2_proj = self.proj2(x2)  # [B, hidden_dim]

        # Stack as sequence [B, 2, hidden_dim]
        x_seq = torch.stack([x1_proj, x2_proj], dim=1)

        # Self-attention
        attn_out, attn_weights = self.self_attention(
            x_seq, x_seq, x_seq, need_weights=return_attention
        )  # [B, 2, hidden_dim]

        # Residual connection and norm
        x_seq = self.norm(x_seq + attn_out)

        # Flatten and project
        x_flat = x_seq.view(batch_size, -1)  # [B, hidden_dim * 2]
        fused = self.output_proj(x_flat)  # [B, output_dim]

        if return_attention:
            return fused, attn_weights
        return fused


class AdditiveAttentionFusion(nn.Module):
    """
    Additive attention fusion (Bahdanau-style attention).

    Computes attention weights using additive scoring and fuses features
    using weighted combination.

    Args:
        dim1: Dimension of first modality features
        dim2: Dimension of second modality features
        output_dim: Output dimension after fusion
        attention_dim: Attention hidden dimension
        dropout: Dropout rate

    Example:
        >>> fusion = AdditiveAttentionFusion(dim1=512, dim2=512, output_dim=256)
        >>> x1 = torch.randn(4, 512)
        >>> x2 = torch.randn(4, 512)
        >>> fused = fusion(x1, x2)  # [4, 256]
    """

    def __init__(
        self,
        dim1: int,
        dim2: int,
        output_dim: int,
        attention_dim: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.dim1 = dim1
        self.dim2 = dim2
        self.output_dim = output_dim

        # Attention mechanism
        self.W1 = nn.Linear(dim1, attention_dim, bias=False)
        self.W2 = nn.Linear(dim2, attention_dim, bias=False)
        self.v = nn.Linear(attention_dim, 1, bias=False)

        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(dim1 + dim2, output_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

    def forward(
        self, x1: torch.Tensor, x2: torch.Tensor, return_attention: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            x1: First modality features [B, dim1]
            x2: Second modality features [B, dim2]
            return_attention: Return attention weights

        Returns:
            Fused features [B, output_dim]
            If return_attention=True, also returns attention weights
        """
        # Compute attention scores
        # score = v^T * tanh(W1*x1 + W2*x2)
        score1 = self.v(torch.tanh(self.W1(x1)))  # [B, 1]
        score2 = self.v(torch.tanh(self.W2(x2)))  # [B, 1]

        # Attention weights
        scores = torch.cat([score1, score2], dim=1)  # [B, 2]
        attention_weights = F.softmax(scores, dim=1)  # [B, 2]

        # Weighted features
        alpha1 = attention_weights[:, 0:1]  # [B, 1]
        alpha2 = attention_weights[:, 1:2]  # [B, 1]

        weighted_x1 = alpha1 * x1  # [B, dim1]
        weighted_x2 = alpha2 * x2  # [B, dim2]

        # Concatenate and project
        fused = torch.cat([weighted_x1, weighted_x2], dim=1)  # [B, dim1 + dim2]
        fused = self.output_proj(fused)  # [B, output_dim]

        if return_attention:
            return fused, attention_weights
        return fused


class BilinearAttentionFusion(nn.Module):
    """
    Bilinear attention fusion using bilinear pooling.

    Computes attention using bilinear interaction between modalities.

    Args:
        dim1: Dimension of first modality features
        dim2: Dimension of second modality features
        output_dim: Output dimension after fusion
        dropout: Dropout rate

    Example:
        >>> fusion = BilinearAttentionFusion(dim1=512, dim2=512, output_dim=256)
        >>> x1 = torch.randn(4, 512)
        >>> x2 = torch.randn(4, 512)
        >>> fused = fusion(x1, x2)  # [4, 256]
    """

    def __init__(
        self,
        dim1: int,
        dim2: int,
        output_dim: int,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.dim1 = dim1
        self.dim2 = dim2
        self.output_dim = output_dim

        # Bilinear weight matrix
        self.bilinear = nn.Bilinear(dim1, dim2, 1)

        # Feature projections
        self.proj1 = nn.Linear(dim1, output_dim)
        self.proj2 = nn.Linear(dim2, output_dim)

        # Output layers
        self.output = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

    def forward(
        self, x1: torch.Tensor, x2: torch.Tensor, return_attention: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            x1: First modality features [B, dim1]
            x2: Second modality features [B, dim2]
            return_attention: Return attention score

        Returns:
            Fused features [B, output_dim]
            If return_attention=True, also returns attention score
        """
        # Compute bilinear attention score
        attention_score = torch.sigmoid(self.bilinear(x1, x2))  # [B, 1]

        # Project features
        x1_proj = self.proj1(x1)  # [B, output_dim]
        x2_proj = self.proj2(x2)  # [B, output_dim]

        # Weighted combination
        fused = attention_score * x1_proj + (1 - attention_score) * x2_proj

        # Apply activation and dropout
        fused = self.output(fused)

        if return_attention:
            return fused, attention_score
        return fused


class GatedAttentionFusion(nn.Module):
    """
    Gated attention fusion with learnable gates.

    Uses gating mechanism to control information flow from each modality.

    Args:
        dim1: Dimension of first modality features
        dim2: Dimension of second modality features
        output_dim: Output dimension after fusion
        dropout: Dropout rate

    Example:
        >>> fusion = GatedAttentionFusion(dim1=512, dim2=512, output_dim=256)
        >>> x1 = torch.randn(4, 512)
        >>> x2 = torch.randn(4, 512)
        >>> fused = fusion(x1, x2)  # [4, 256]
    """

    def __init__(
        self,
        dim1: int,
        dim2: int,
        output_dim: int,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.dim1 = dim1
        self.dim2 = dim2
        self.output_dim = output_dim

        # Gate networks
        self.gate1 = nn.Sequential(
            nn.Linear(dim1 + dim2, dim1),
            nn.Sigmoid(),
        )
        self.gate2 = nn.Sequential(
            nn.Linear(dim1 + dim2, dim2),
            nn.Sigmoid(),
        )

        # Feature projections
        self.proj1 = nn.Linear(dim1, output_dim)
        self.proj2 = nn.Linear(dim2, output_dim)

        # Output layers
        self.output = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

    def forward(
        self, x1: torch.Tensor, x2: torch.Tensor, return_gates: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass.

        Args:
            x1: First modality features [B, dim1]
            x2: Second modality features [B, dim2]
            return_gates: Return gate values

        Returns:
            Fused features [B, output_dim]
            If return_gates=True, also returns (gate1, gate2)
        """
        # Concatenate for gate computation
        x_concat = torch.cat([x1, x2], dim=1)  # [B, dim1 + dim2]

        # Compute gates
        gate1 = self.gate1(x_concat)  # [B, dim1]
        gate2 = self.gate2(x_concat)  # [B, dim2]

        # Apply gates
        gated_x1 = gate1 * x1  # [B, dim1]
        gated_x2 = gate2 * x2  # [B, dim2]

        # Project and combine
        x1_proj = self.proj1(gated_x1)  # [B, output_dim]
        x2_proj = self.proj2(gated_x2)  # [B, output_dim]

        fused = x1_proj + x2_proj  # [B, output_dim]
        fused = self.output(fused)

        if return_gates:
            return fused, (gate1, gate2)
        return fused


class MultimodalSelfAttentionFusion(nn.Module):
    """
    Self-attention fusion for multiple modalities (3+).

    Extends SelfAttentionFusion to handle arbitrary number of modalities.

    Args:
        modality_dims: List of feature dimensions for each modality
        output_dim: Output dimension after fusion
        num_heads: Number of attention heads
        dropout: Dropout rate

    Example:
        >>> fusion = MultimodalSelfAttentionFusion(
        ...     modality_dims=[512, 512, 256],
        ...     output_dim=256
        ... )
        >>> features = [torch.randn(4, 512), torch.randn(4, 512), torch.randn(4, 256)]
        >>> fused = fusion(features)  # [4, 256]
    """

    def __init__(
        self,
        modality_dims: list[int],
        output_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.modality_dims = modality_dims
        self.output_dim = output_dim
        self.num_modalities = len(modality_dims)

        # Project all modalities to same dimension
        hidden_dim = max(modality_dims)
        self.projections = nn.ModuleList([
            nn.Linear(dim, hidden_dim) for dim in modality_dims
        ])

        # Self-attention
        self.self_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        # Layer norm
        self.norm = nn.LayerNorm(hidden_dim)

        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim * self.num_modalities, output_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

    def forward(
        self, features: list[torch.Tensor], return_attention: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            features: List of feature tensors, each [B, dim_i]
            return_attention: Return attention weights

        Returns:
            Fused features [B, output_dim]
            If return_attention=True, also returns attention weights
        """
        if len(features) != self.num_modalities:
            raise ValueError(
                f"Expected {self.num_modalities} modalities, got {len(features)}"
            )

        batch_size = features[0].size(0)

        # Project all modalities
        projected = [
            proj(feat) for proj, feat in zip(self.projections, features)
        ]  # Each [B, hidden_dim]

        # Stack as sequence [B, num_modalities, hidden_dim]
        x_seq = torch.stack(projected, dim=1)

        # Self-attention
        attn_out, attn_weights = self.self_attention(
            x_seq, x_seq, x_seq, need_weights=return_attention
        )

        # Residual and norm
        x_seq = self.norm(x_seq + attn_out)

        # Flatten and project
        x_flat = x_seq.view(batch_size, -1)
        fused = self.output_proj(x_flat)

        if return_attention:
            return fused, attn_weights
        return fused
