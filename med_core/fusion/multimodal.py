"""Generic multimodal fusion strategies for more than two modalities."""

import torch
from torch import nn


class MultimodalConcatenateFusion(nn.Module):
    """Concatenate an arbitrary number of modality features and project."""

    def __init__(
        self,
        modality_dims: list[int],
        output_dim: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.modality_dims = modality_dims
        self.num_modalities = len(modality_dims)
        self.output_dim = output_dim

        self.projection = nn.Sequential(
            nn.Linear(sum(modality_dims), output_dim),
            nn.ReLU(inplace=True),
            nn.LayerNorm(output_dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        features: list[torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor] | None]:
        if len(features) != self.num_modalities:
            raise ValueError(
                f"Expected {self.num_modalities} modalities, got {len(features)}",
            )

        fused = self.projection(torch.cat(features, dim=1))
        return fused, None


class MultimodalGatedFusion(nn.Module):
    """Fuse multiple modalities with learned instance-wise gates."""

    def __init__(
        self,
        modality_dims: list[int],
        output_dim: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.modality_dims = modality_dims
        self.num_modalities = len(modality_dims)
        self.output_dim = output_dim

        self.projections = nn.ModuleList(
            [nn.Linear(dim, output_dim) for dim in modality_dims],
        )
        self.gate_network = nn.Sequential(
            nn.Linear(sum(modality_dims), output_dim),
            nn.ReLU(inplace=True),
            nn.Linear(output_dim, self.num_modalities),
        )
        self.dropout = nn.Dropout(dropout)
        self._last_gate_values: torch.Tensor | None = None

    def forward(
        self,
        features: list[torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor] | None]:
        if len(features) != self.num_modalities:
            raise ValueError(
                f"Expected {self.num_modalities} modalities, got {len(features)}",
            )

        gate_values = torch.softmax(
            self.gate_network(torch.cat(features, dim=1)),
            dim=1,
        )
        projected = torch.stack(
            [torch.tanh(proj(feat)) for proj, feat in zip(self.projections, features)],
            dim=1,
        )
        fused = (projected * gate_values.unsqueeze(-1)).sum(dim=1)

        if not self.training:
            self._last_gate_values = gate_values.detach()

        return self.dropout(fused), {"gate_values": gate_values}

    def get_gate_values(self) -> torch.Tensor | None:
        """Return the most recent gate values."""
        return self._last_gate_values
