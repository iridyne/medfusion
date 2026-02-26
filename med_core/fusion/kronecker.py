"""
Kronecker Product Fusion for multimodal feature fusion.

This module implements Kronecker product-based fusion, which captures
second-order interactions between modalities through outer products.

Reference:
    - "Multimodal Compact Bilinear Pooling for Visual Question Answering"
    - SMuRF: "Multimodal Fusion for Survival Prediction"
"""

import torch
from torch import nn


class KroneckerFusion(nn.Module):
    """
    Kronecker Product Fusion for multimodal features.

    Computes the outer product between two feature vectors and projects
    to a lower-dimensional space. This captures second-order interactions
    between modalities.

    Args:
        dim1: Dimension of first modality features
        dim2: Dimension of second modality features
        output_dim: Output dimension after fusion
        dropout: Dropout rate
        use_bilinear: Use bilinear projection instead of linear

    Example:
        >>> fusion = KroneckerFusion(dim1=512, dim2=512, output_dim=256)
        >>> x1 = torch.randn(4, 512)  # Modality 1
        >>> x2 = torch.randn(4, 512)  # Modality 2
        >>> fused = fusion(x1, x2)  # [4, 256]
    """

    def __init__(
        self,
        dim1: int,
        dim2: int,
        output_dim: int,
        dropout: float = 0.1,
        use_bilinear: bool = False,
    ):
        super().__init__()

        self.dim1 = dim1
        self.dim2 = dim2
        self.output_dim = output_dim
        self.use_bilinear = use_bilinear

        # Kronecker product dimension
        kronecker_dim = dim1 * dim2

        if use_bilinear:
            # Bilinear projection: more expressive but more parameters
            self.projection = nn.Bilinear(dim1, dim2, output_dim)
        else:
            # Linear projection from Kronecker product
            self.projection = nn.Sequential(
                nn.Linear(kronecker_dim, output_dim),
                nn.LayerNorm(output_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """
        Fuse two modality features using Kronecker product.

        Args:
            x1: First modality features [B, dim1]
            x2: Second modality features [B, dim2]

        Returns:
            Fused features [B, output_dim]
        """
        batch_size = x1.size(0)

        # Ensure inputs are 2D
        if x1.dim() > 2:
            x1 = x1.view(batch_size, -1)
        if x2.dim() > 2:
            x2 = x2.view(batch_size, -1)

        if self.use_bilinear:
            # Bilinear projection
            fused = self.projection(x1, x2)
        else:
            # Compute Kronecker product (outer product)
            # [B, dim1, 1] x [B, 1, dim2] -> [B, dim1, dim2]
            kronecker = torch.bmm(
                x1.unsqueeze(2),  # [B, dim1, 1]
                x2.unsqueeze(1),  # [B, 1, dim2]
            )

            # Flatten: [B, dim1, dim2] -> [B, dim1 * dim2]
            kronecker_flat = kronecker.view(batch_size, -1)

            # Project to output dimension
            fused = self.projection(kronecker_flat)

        return fused


class CompactKroneckerFusion(nn.Module):
    """
    Compact Kronecker Product Fusion using low-rank approximation.

    Uses Count Sketch or Random Maclaurin to approximate the full
    Kronecker product with much lower memory footprint.

    Args:
        dim1: Dimension of first modality features
        dim2: Dimension of second modality features
        output_dim: Output dimension after fusion
        sketch_dim: Sketch dimension (intermediate dimension)
        dropout: Dropout rate

    Example:
        >>> fusion = CompactKroneckerFusion(
        ...     dim1=512, dim2=512, output_dim=256, sketch_dim=2048
        ... )
        >>> x1 = torch.randn(4, 512)
        >>> x2 = torch.randn(4, 512)
        >>> fused = fusion(x1, x2)  # [4, 256]
    """

    def __init__(
        self,
        dim1: int,
        dim2: int,
        output_dim: int,
        sketch_dim: int = 2048,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.dim1 = dim1
        self.dim2 = dim2
        self.output_dim = output_dim
        self.sketch_dim = sketch_dim

        # Count Sketch parameters (fixed random projections)
        self.register_buffer(
            "sketch_matrix1",
            self._generate_sketch_matrix(dim1, sketch_dim),
        )
        self.register_buffer(
            "sketch_matrix2",
            self._generate_sketch_matrix(dim2, sketch_dim),
        )

        # Projection to output dimension
        self.projection = nn.Sequential(
            nn.Linear(sketch_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

    def _generate_sketch_matrix(self, input_dim: int, sketch_dim: int) -> torch.Tensor:
        """Generate random sketch matrix for Count Sketch."""
        # Random hash: each input dimension maps to a sketch dimension
        indices = torch.randint(0, sketch_dim, (input_dim,))

        # Random sign: +1 or -1
        signs = torch.randint(0, 2, (input_dim,)).float() * 2 - 1

        # Create sparse sketch matrix
        sketch_matrix = torch.zeros(input_dim, sketch_dim)
        sketch_matrix[torch.arange(input_dim), indices] = signs

        return sketch_matrix

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """
        Fuse two modality features using compact Kronecker product.

        Args:
            x1: First modality features [B, dim1]
            x2: Second modality features [B, dim2]

        Returns:
            Fused features [B, output_dim]
        """
        batch_size = x1.size(0)

        # Ensure inputs are 2D
        if x1.dim() > 2:
            x1 = x1.view(batch_size, -1)
        if x2.dim() > 2:
            x2 = x2.view(batch_size, -1)

        # Sketch both features
        sketch1 = torch.matmul(x1, self.sketch_matrix1)  # [B, sketch_dim]
        sketch2 = torch.matmul(x2, self.sketch_matrix2)  # [B, sketch_dim]

        # Element-wise product (approximates Kronecker product)
        compact_kronecker = sketch1 * sketch2  # [B, sketch_dim]

        # Project to output dimension
        fused = self.projection(compact_kronecker)

        return fused


class MultimodalKroneckerFusion(nn.Module):
    """
    Kronecker Fusion for multiple modalities (>2).

    Supports fusion of N modalities by computing pairwise Kronecker
    products and aggregating them.

    Args:
        modality_dims: List of feature dimensions for each modality
        output_dim: Output dimension after fusion
        fusion_strategy: How to combine multiple modalities
            - "sequential": Fuse modalities sequentially (1+2, then +3, etc.)
            - "pairwise": Fuse all pairs and concatenate
            - "star": Fuse each modality with the first one
        dropout: Dropout rate

    Example:
        >>> fusion = MultimodalKroneckerFusion(
        ...     modality_dims=[512, 512, 256],
        ...     output_dim=256,
        ...     fusion_strategy="sequential"
        ... )
        >>> features = [
        ...     torch.randn(4, 512),  # Modality 1
        ...     torch.randn(4, 512),  # Modality 2
        ...     torch.randn(4, 256),  # Modality 3
        ... ]
        >>> fused = fusion(features)  # [4, 256]
    """

    def __init__(
        self,
        modality_dims: list[int],
        output_dim: int,
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
                # Intermediate dimension for sequential fusion
                intermediate_dim = (
                    output_dim if i == self.num_modalities - 1 else current_dim
                )

                fusion = KroneckerFusion(
                    dim1=current_dim,
                    dim2=modality_dims[i],
                    output_dim=intermediate_dim,
                    dropout=dropout,
                )
                self.fusion_modules.append(fusion)
                current_dim = intermediate_dim

        elif fusion_strategy == "pairwise":
            self.fusion_modules = nn.ModuleList()

            # Create fusion for all pairs
            for i in range(self.num_modalities):
                for j in range(i + 1, self.num_modalities):
                    fusion = KroneckerFusion(
                        dim1=modality_dims[i],
                        dim2=modality_dims[j],
                        output_dim=output_dim // (self.num_modalities - 1),
                        dropout=dropout,
                    )
                    self.fusion_modules.append(fusion)

            # Final projection
            num_pairs = len(self.fusion_modules)
            self.final_projection = nn.Linear(
                num_pairs * (output_dim // (self.num_modalities - 1)),
                output_dim,
            )

        elif fusion_strategy == "star":
            self.fusion_modules = nn.ModuleList()

            # Fuse each modality with the first one
            for i in range(1, self.num_modalities):
                fusion = KroneckerFusion(
                    dim1=modality_dims[0],
                    dim2=modality_dims[i],
                    output_dim=output_dim // (self.num_modalities - 1),
                    dropout=dropout,
                )
                self.fusion_modules.append(fusion)

            # Final projection
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
                Each tensor has shape [B, modality_dim]

        Returns:
            Fused features [B, output_dim]
        """
        if len(features) != self.num_modalities:
            raise ValueError(
                f"Expected {self.num_modalities} modalities, got {len(features)}",
            )

        if self.fusion_strategy == "sequential":
            # Fuse sequentially: (f1 ⊗ f2) ⊗ f3 ⊗ ...
            fused = features[0]
            for i, fusion_module in enumerate(self.fusion_modules):
                fused = fusion_module(fused, features[i + 1])
            return fused

        if self.fusion_strategy == "pairwise":
            # Fuse all pairs and concatenate
            pair_features = []
            idx = 0
            for i in range(self.num_modalities):
                for j in range(i + 1, self.num_modalities):
                    pair_fused = self.fusion_modules[idx](features[i], features[j])
                    pair_features.append(pair_fused)
                    idx += 1

            # Concatenate and project
            concatenated = torch.cat(pair_features, dim=1)
            return self.final_projection(concatenated)

        if self.fusion_strategy == "star":
            # Fuse each with the first modality
            star_features = []
            for i, fusion_module in enumerate(self.fusion_modules):
                star_fused = fusion_module(features[0], features[i + 1])
                star_features.append(star_fused)

            # Concatenate and project
            concatenated = torch.cat(star_features, dim=1)
            return self.final_projection(concatenated)
