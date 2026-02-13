"""
Fused Attention Fusion for multimodal feature fusion.

This module implements the Fused Attention mechanism from SMuRF, which
combines cross-modal attention with Kronecker product fusion to capture
both first-order and second-order interactions between modalities.

Reference:
    - SMuRF: "Multimodal Fusion for Survival Prediction in Lung Cancer"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FusedAttentionFusion(nn.Module):
    """
    Fused Attention Fusion combining cross-attention and Kronecker product.

    This fusion mechanism:
    1. Applies cross-modal attention to capture first-order interactions
    2. Computes Kronecker product for second-order interactions
    3. Combines both through a gating mechanism

    Args:
        dim1: Dimension of first modality features
        dim2: Dimension of second modality features
        output_dim: Output dimension after fusion
        num_heads: Number of attention heads
        dropout: Dropout rate
        use_kronecker: Whether to use Kronecker product (if False, only attention)

    Example:
        >>> fusion = FusedAttentionFusion(dim1=512, dim2=512, output_dim=256)
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
        use_kronecker: bool = True,
    ):
        super().__init__()

        self.dim1 = dim1
        self.dim2 = dim2
        self.output_dim = output_dim
        self.num_heads = num_heads
        self.use_kronecker = use_kronecker

        # Cross-modal attention
        self.cross_attention = CrossModalAttention(
            dim1=dim1,
            dim2=dim2,
            num_heads=num_heads,
            dropout=dropout,
        )

        # Kronecker product fusion (if enabled)
        if use_kronecker:
            # Compact Kronecker to save memory
            self.kronecker_dim = min(dim1 * dim2, 2048)
            self.sketch_matrix1 = nn.Parameter(
                self._generate_sketch_matrix(dim1, self.kronecker_dim),
                requires_grad=False,
            )
            self.sketch_matrix2 = nn.Parameter(
                self._generate_sketch_matrix(dim2, self.kronecker_dim),
                requires_grad=False,
            )

            # Projection for Kronecker features
            self.kronecker_proj = nn.Sequential(
                nn.Linear(self.kronecker_dim, output_dim),
                nn.LayerNorm(output_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            )

        # Projection for attention features
        self.attention_proj = nn.Sequential(
            nn.Linear(dim1 + dim2, output_dim),
            nn.LayerNorm(output_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

        # Gating mechanism to combine attention and Kronecker
        if use_kronecker:
            self.gate = nn.Sequential(
                nn.Linear(output_dim * 2, output_dim),
                nn.Sigmoid(),
            )

            self.final_proj = nn.Sequential(
                nn.Linear(output_dim * 2, output_dim),
                nn.LayerNorm(output_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            )

    def _generate_sketch_matrix(self, input_dim: int, sketch_dim: int) -> torch.Tensor:
        """Generate random sketch matrix for compact Kronecker product."""
        indices = torch.randint(0, sketch_dim, (input_dim,))
        signs = torch.randint(0, 2, (input_dim,)).float() * 2 - 1
        sketch_matrix = torch.zeros(input_dim, sketch_dim)
        sketch_matrix[torch.arange(input_dim), indices] = signs
        return sketch_matrix

    def forward(
        self, x1: torch.Tensor, x2: torch.Tensor, return_attention: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        Fuse two modality features using fused attention.

        Args:
            x1: First modality features [B, dim1]
            x2: Second modality features [B, dim2]
            return_attention: Return attention weights

        Returns:
            Fused features [B, output_dim]
            If return_attention=True, also returns attention weights
        """
        batch_size = x1.size(0)

        # Ensure inputs are 2D
        if x1.dim() > 2:
            x1 = x1.view(batch_size, -1)
        if x2.dim() > 2:
            x2 = x2.view(batch_size, -1)

        # 1. Cross-modal attention
        attended_x1, attended_x2, attn_weights = self.cross_attention(x1, x2)

        # Concatenate attended features
        attention_features = torch.cat([attended_x1, attended_x2], dim=1)
        attention_output = self.attention_proj(attention_features)

        if not self.use_kronecker:
            if return_attention:
                return attention_output, attn_weights
            return attention_output

        # 2. Kronecker product (second-order interactions)
        sketch1 = torch.matmul(x1, self.sketch_matrix1)
        sketch2 = torch.matmul(x2, self.sketch_matrix2)
        kronecker_features = sketch1 * sketch2
        kronecker_output = self.kronecker_proj(kronecker_features)

        # 3. Gating mechanism
        combined = torch.cat([attention_output, kronecker_output], dim=1)
        gate = self.gate(combined)

        # Weighted combination
        gated_attention = gate * attention_output
        gated_kronecker = (1 - gate) * kronecker_output

        # Final fusion
        fused = torch.cat([gated_attention, gated_kronecker], dim=1)
        output = self.final_proj(fused)

        if return_attention:
            return output, attn_weights
        return output


class CrossModalAttention(nn.Module):
    """
    Cross-modal attention between two modalities.

    Computes attention from modality 1 to modality 2 and vice versa.

    Args:
        dim1: Dimension of first modality
        dim2: Dimension of second modality
        num_heads: Number of attention heads
        dropout: Dropout rate
    """

    def __init__(
        self,
        dim1: int,
        dim2: int,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.dim1 = dim1
        self.dim2 = dim2
        self.num_heads = num_heads
        self.head_dim = max(dim1, dim2) // num_heads

        # Query, Key, Value projections for modality 1
        self.q1 = nn.Linear(dim1, num_heads * self.head_dim)
        self.k1 = nn.Linear(dim1, num_heads * self.head_dim)
        self.v1 = nn.Linear(dim1, num_heads * self.head_dim)

        # Query, Key, Value projections for modality 2
        self.q2 = nn.Linear(dim2, num_heads * self.head_dim)
        self.k2 = nn.Linear(dim2, num_heads * self.head_dim)
        self.v2 = nn.Linear(dim2, num_heads * self.head_dim)

        # Output projections
        self.out1 = nn.Linear(num_heads * self.head_dim, dim1)
        self.out2 = nn.Linear(num_heads * self.head_dim, dim2)

        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5

    def forward(
        self, x1: torch.Tensor, x2: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Apply cross-modal attention.

        Args:
            x1: First modality features [B, dim1]
            x2: Second modality features [B, dim2]

        Returns:
            attended_x1: Attended features for modality 1 [B, dim1]
            attended_x2: Attended features for modality 2 [B, dim2]
            attn_weights: Attention weights [B, num_heads, 1, 1]
        """
        batch_size = x1.size(0)

        # Add sequence dimension: [B, dim] -> [B, 1, dim]
        x1 = x1.unsqueeze(1)
        x2 = x2.unsqueeze(1)

        # Project to Q, K, V
        q1 = self.q1(x1).view(batch_size, 1, self.num_heads, self.head_dim).transpose(1, 2)
        k1 = self.k1(x1).view(batch_size, 1, self.num_heads, self.head_dim).transpose(1, 2)
        v1 = self.v1(x1).view(batch_size, 1, self.num_heads, self.head_dim).transpose(1, 2)

        q2 = self.q2(x2).view(batch_size, 1, self.num_heads, self.head_dim).transpose(1, 2)
        k2 = self.k2(x2).view(batch_size, 1, self.num_heads, self.head_dim).transpose(1, 2)
        v2 = self.v2(x2).view(batch_size, 1, self.num_heads, self.head_dim).transpose(1, 2)

        # Cross-attention: modality 1 attends to modality 2
        attn_12 = (q1 @ k2.transpose(-2, -1)) * self.scale
        attn_12 = F.softmax(attn_12, dim=-1)
        attn_12 = self.dropout(attn_12)
        out_12 = (attn_12 @ v2).transpose(1, 2).contiguous().view(batch_size, 1, -1)
        attended_x1 = self.out1(out_12).squeeze(1)

        # Cross-attention: modality 2 attends to modality 1
        attn_21 = (q2 @ k1.transpose(-2, -1)) * self.scale
        attn_21 = F.softmax(attn_21, dim=-1)
        attn_21 = self.dropout(attn_21)
        out_21 = (attn_21 @ v1).transpose(1, 2).contiguous().view(batch_size, 1, -1)
        attended_x2 = self.out2(out_21).squeeze(1)

        # Average attention weights for visualization
        attn_weights = (attn_12 + attn_21) / 2

        return attended_x1, attended_x2, attn_weights


class MultimodalFusedAttention(nn.Module):
    """
    Fused Attention Fusion for multiple modalities (>2).

    Extends FusedAttentionFusion to handle N modalities by computing
    pairwise fused attention and aggregating results.

    Args:
        modality_dims: List of feature dimensions for each modality
        output_dim: Output dimension after fusion
        num_heads: Number of attention heads
        fusion_strategy: How to combine multiple modalities
            - "sequential": Fuse modalities sequentially
            - "pairwise": Fuse all pairs and aggregate
            - "star": Fuse each modality with the first one
        dropout: Dropout rate

    Example:
        >>> fusion = MultimodalFusedAttention(
        ...     modality_dims=[512, 512, 256],
        ...     output_dim=256,
        ...     fusion_strategy="sequential"
        ... )
        >>> features = [
        ...     torch.randn(4, 512),
        ...     torch.randn(4, 512),
        ...     torch.randn(4, 256),
        ... ]
        >>> fused = fusion(features)  # [4, 256]
    """

    def __init__(
        self,
        modality_dims: list[int],
        output_dim: int,
        num_heads: int = 8,
        fusion_strategy: str = "sequential",
        dropout: float = 0.1,
    ):
        super().__init__()

        self.modality_dims = modality_dims
        self.output_dim = output_dim
        self.fusion_strategy = fusion_strategy
        self.num_modalities = len(modality_dims)

        if self.num_modalities < 2:
            raise ValueError("Need at least 2 modalities for fusion")

        # Create fusion modules based on strategy
        if fusion_strategy == "sequential":
            self.fusion_modules = nn.ModuleList()
            current_dim = modality_dims[0]

            for i in range(1, self.num_modalities):
                intermediate_dim = output_dim if i == self.num_modalities - 1 else current_dim

                fusion = FusedAttentionFusion(
                    dim1=current_dim,
                    dim2=modality_dims[i],
                    output_dim=intermediate_dim,
                    num_heads=num_heads,
                    dropout=dropout,
                )
                self.fusion_modules.append(fusion)
                current_dim = intermediate_dim

        elif fusion_strategy == "pairwise":
            self.fusion_modules = nn.ModuleList()

            for i in range(self.num_modalities):
                for j in range(i + 1, self.num_modalities):
                    fusion = FusedAttentionFusion(
                        dim1=modality_dims[i],
                        dim2=modality_dims[j],
                        output_dim=output_dim // (self.num_modalities - 1),
                        num_heads=num_heads,
                        dropout=dropout,
                    )
                    self.fusion_modules.append(fusion)

            num_pairs = len(self.fusion_modules)
            self.final_projection = nn.Linear(
                num_pairs * (output_dim // (self.num_modalities - 1)),
                output_dim,
            )

        elif fusion_strategy == "star":
            self.fusion_modules = nn.ModuleList()

            for i in range(1, self.num_modalities):
                fusion = FusedAttentionFusion(
                    dim1=modality_dims[0],
                    dim2=modality_dims[i],
                    output_dim=output_dim // (self.num_modalities - 1),
                    num_heads=num_heads,
                    dropout=dropout,
                )
                self.fusion_modules.append(fusion)

            self.final_projection = nn.Linear(
                (self.num_modalities - 1) * (output_dim // (self.num_modalities - 1)),
                output_dim,
            )

        else:
            raise ValueError(f"Unknown fusion strategy: {fusion_strategy}")

    def forward(self, features: list[torch.Tensor]) -> torch.Tensor:
        """
        Fuse multiple modality features.

        Args:
            features: List of feature tensors, one per modality

        Returns:
            Fused features [B, output_dim]
        """
        if len(features) != self.num_modalities:
            raise ValueError(
                f"Expected {self.num_modalities} modalities, got {len(features)}"
            )

        if self.fusion_strategy == "sequential":
            fused = features[0]
            for i, fusion_module in enumerate(self.fusion_modules):
                fused = fusion_module(fused, features[i + 1])
            return fused

        elif self.fusion_strategy == "pairwise":
            pair_features = []
            idx = 0
            for i in range(self.num_modalities):
                for j in range(i + 1, self.num_modalities):
                    pair_fused = self.fusion_modules[idx](features[i], features[j])
                    pair_features.append(pair_fused)
                    idx += 1

            concatenated = torch.cat(pair_features, dim=1)
            return self.final_projection(concatenated)

        elif self.fusion_strategy == "star":
            star_features = []
            for i, fusion_module in enumerate(self.fusion_modules):
                star_fused = fusion_module(features[0], features[i + 1])
                star_features.append(star_fused)

            concatenated = torch.cat(star_features, dim=1)
            return self.final_projection(concatenated)
