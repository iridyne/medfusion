"""HIPT-style embedding backbone.

This backbone is designed for offline/precomputed pathology embeddings, e.g.
embeddings exported by a HIPT preprocessing pipeline.

Input shapes:
- [B, D]   : one embedding vector per sample
- [B, N, D]: a bag/sequence of embedding vectors per sample (for MIL)

Output shapes:
- [B, F]
- [B, N, F]
"""

from __future__ import annotations

import torch
from torch import nn

from med_core.backbones.base import BaseBackbone


class HIPTEmbeddingBackbone(BaseBackbone):
    """Lightweight projection backbone for precomputed HIPT embeddings."""

    def __init__(
        self,
        input_dim: int,
        feature_dim: int = 512,
        dropout: float = 0.1,
        use_layer_norm: bool = True,
    ) -> None:
        super().__init__()

        if input_dim <= 0:
            raise ValueError(f"input_dim must be positive, got {input_dim}")
        if feature_dim <= 0:
            raise ValueError(f"feature_dim must be positive, got {feature_dim}")

        self.input_dim = int(input_dim)
        self._output_dim = int(feature_dim)

        layers: list[nn.Module] = []
        if use_layer_norm:
            layers.append(nn.LayerNorm(self.input_dim))
        layers.extend(
            [
                nn.Linear(self.input_dim, self._output_dim),
                nn.GELU(),
            ]
        )
        if dropout > 0:
            layers.append(nn.Dropout(float(dropout)))

        self._projection = nn.Sequential(*layers)

    def _validate_last_dim(self, x: torch.Tensor) -> None:
        if x.shape[-1] != self.input_dim:
            raise ValueError(
                "HIPTEmbeddingBackbone input dimension mismatch: "
                f"expected last dim={self.input_dim}, got {x.shape[-1]}",
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project embedding vectors to model feature space."""
        if x.ndim == 2:
            self._validate_last_dim(x)
            return self._projection(x.float())

        if x.ndim == 3:
            self._validate_last_dim(x)
            bsz, num_instances, _ = x.shape
            flattened = x.reshape(bsz * num_instances, self.input_dim).float()
            projected = self._projection(flattened)
            return projected.reshape(bsz, num_instances, self.output_dim)

        raise ValueError(
            "HIPTEmbeddingBackbone expects input shape [B, D] or [B, N, D], "
            f"got {tuple(x.shape)}",
        )
