"""Core three-phase CT + tabular fusion model."""

from __future__ import annotations

import torch
from torch import nn


class _PhaseEncoder(nn.Module):
    def __init__(self, output_dim: int) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv3d(1, 8, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2),
            nn.Conv3d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool3d((1, 1, 1)),
        )
        self.projection = nn.Linear(16, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.features(x)
        return self.projection(features.flatten(1))


class ThreePhaseCTFusionModel(nn.Module):
    """Three-phase CT + clinical fusion classifier with optional risk head."""

    def __init__(
        self,
        phase_feature_dim: int,
        clinical_input_dim: int,
        clinical_hidden_dim: int,
        fusion_hidden_dim: int,
        phase_fusion_type: str = "concatenate",
        share_phase_encoder: bool = False,
        freeze_phase_encoder: bool = False,
        use_risk_head: bool = False,
    ) -> None:
        super().__init__()
        self.phase_fusion_type = phase_fusion_type
        if share_phase_encoder:
            shared_encoder = _PhaseEncoder(phase_feature_dim)
            self.arterial_encoder = shared_encoder
            self.portal_encoder = shared_encoder
            self.noncontrast_encoder = shared_encoder
        else:
            self.arterial_encoder = _PhaseEncoder(phase_feature_dim)
            self.portal_encoder = _PhaseEncoder(phase_feature_dim)
            self.noncontrast_encoder = _PhaseEncoder(phase_feature_dim)

        self.clinical_encoder = nn.Sequential(
            nn.Linear(clinical_input_dim, clinical_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
        )
        phase_input_dim = phase_feature_dim * 3
        if self.phase_fusion_type == "mean":
            phase_input_dim = phase_feature_dim
        self.phase_fusion = nn.Sequential(
            nn.Linear(phase_input_dim, fusion_hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.multimodal_fusion = nn.Sequential(
            nn.Linear(fusion_hidden_dim + clinical_hidden_dim, fusion_hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Linear(fusion_hidden_dim, 2)
        self.risk_head = nn.Linear(fusion_hidden_dim, 1) if use_risk_head else None

        if freeze_phase_encoder:
            for encoder in (
                self.arterial_encoder,
                self.portal_encoder,
                self.noncontrast_encoder,
            ):
                for parameter in encoder.parameters():
                    parameter.requires_grad = False

    def forward(
        self,
        arterial: torch.Tensor,
        portal: torch.Tensor,
        noncontrast: torch.Tensor,
        clinical: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        arterial_features = self.arterial_encoder(arterial)
        portal_features = self.portal_encoder(portal)
        noncontrast_features = self.noncontrast_encoder(noncontrast)
        clinical_features = self.clinical_encoder(clinical)

        phase_stack = torch.stack(
            [arterial_features, portal_features, noncontrast_features],
            dim=1,
        )
        if self.phase_fusion_type == "mean":
            fused_phase_input = phase_stack.mean(dim=1)
        else:
            fused_phase_input = phase_stack.flatten(1)
        phase_features = self.phase_fusion(fused_phase_input)
        fused_features = self.multimodal_fusion(
            torch.cat([phase_features, clinical_features], dim=1)
        )
        logits = self.classifier(fused_features)
        probability = torch.softmax(logits, dim=1)[:, 1]
        risk_score = (
            self.risk_head(fused_features).squeeze(-1)
            if self.risk_head is not None
            else probability
        )

        return {
            "logits": logits,
            "probability": probability,
            "risk_score": risk_score,
            "fused_features": fused_features,
            "phase_features": phase_features,
        }
