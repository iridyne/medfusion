"""Core three-phase CT + tabular fusion model."""

from __future__ import annotations

import torch
from torch import nn
from med_core.models.doctor_interest import (
    DoctorInterestMapHead,
    InterestGuidedPooling3D,
    TopKLocalFocus3D,
)


def _build_norm(num_channels: int, norm_type: str) -> nn.Module:
    if norm_type == "instance":
        return nn.InstanceNorm3d(num_channels)
    if norm_type == "group":
        group_count = min(4, num_channels)
        while num_channels % group_count != 0 and group_count > 1:
            group_count -= 1
        return nn.GroupNorm(group_count, num_channels)
    return nn.BatchNorm3d(num_channels)


class _PhaseEncoder3D(nn.Module):
    def __init__(
        self,
        output_dim: int,
        *,
        base_channels: int = 16,
        num_blocks: int = 3,
        dropout: float = 0.1,
        norm_type: str = "batch",
    ) -> None:
        super().__init__()
        channels: list[int] = [
            base_channels * (2**index) for index in range(max(num_blocks, 1))
        ]
        layers: list[nn.Module] = []
        in_channels = 1
        for index, out_channels in enumerate(channels):
            layers.extend(
                [
                    nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
                    _build_norm(out_channels, norm_type),
                    nn.ReLU(inplace=True),
                ]
            )
            if dropout > 0:
                layers.append(nn.Dropout3d(dropout))
            if index < len(channels) - 1:
                layers.append(nn.MaxPool3d(kernel_size=2))
            in_channels = out_channels

        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.projection = nn.Linear(channels[-1], output_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feature_map = self.features(x)
        pooled = self.pool(feature_map).flatten(1)
        return self.projection(pooled), feature_map


class ThreePhaseCTFusionModel(nn.Module):
    """Three-phase CT + clinical fusion classifier with optional risk head."""

    def __init__(
        self,
        phase_feature_dim: int,
        clinical_input_dim: int,
        clinical_hidden_dim: int,
        fusion_hidden_dim: int,
        num_classes: int = 2,
        phase_fusion_type: str = "concatenate",
        share_phase_encoder: bool = False,
        freeze_phase_encoder: bool = False,
        use_risk_head: bool = False,
        clinical_mask_dim: int = 0,
        phase_encoder_base_channels: int = 16,
        phase_encoder_num_blocks: int = 3,
        phase_encoder_dropout: float = 0.1,
        phase_encoder_norm_type: str = "batch",
        doctor_interest_enabled: bool = False,
        doctor_interest_hidden_channels: int = 8,
        doctor_interest_temperature: float = 6.0,
        topk_focus_enabled: bool = False,
        topk_focus_k: int = 3,
        topk_focus_patch_size: tuple[int, int, int] = (4, 4, 4),
        topk_focus_projection_dim: int = 16,
    ) -> None:
        if topk_focus_enabled and not doctor_interest_enabled:
            raise ValueError(
                "topk_focus_enabled requires doctor_interest_enabled=True"
            )
        super().__init__()
        self.phase_fusion_type = phase_fusion_type
        self.clinical_mask_dim = clinical_mask_dim
        self.doctor_interest_enabled = doctor_interest_enabled
        self.topk_focus_enabled = topk_focus_enabled
        self._phase_encoder_channels = phase_encoder_base_channels * (2 ** max(phase_encoder_num_blocks - 1, 0))
        encoder_kwargs = {
            "base_channels": phase_encoder_base_channels,
            "num_blocks": phase_encoder_num_blocks,
            "dropout": phase_encoder_dropout,
            "norm_type": phase_encoder_norm_type,
        }
        if share_phase_encoder:
            shared_encoder = _PhaseEncoder3D(phase_feature_dim, **encoder_kwargs)
            self.arterial_encoder = shared_encoder
            self.portal_encoder = shared_encoder
            self.noncontrast_encoder = shared_encoder
        else:
            self.arterial_encoder = _PhaseEncoder3D(phase_feature_dim, **encoder_kwargs)
            self.portal_encoder = _PhaseEncoder3D(phase_feature_dim, **encoder_kwargs)
            self.noncontrast_encoder = _PhaseEncoder3D(
                phase_feature_dim,
                **encoder_kwargs,
            )

        self.clinical_encoder = nn.Sequential(
            nn.Linear(clinical_input_dim + clinical_mask_dim, clinical_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
        )
        phase_input_dim = phase_feature_dim * 3
        self.phase_gate = None
        if self.phase_fusion_type in {"mean", "gated"}:
            phase_input_dim = phase_feature_dim
        if self.phase_fusion_type == "gated":
            self.phase_gate = nn.Linear(phase_feature_dim * 3, 3)
        self.phase_fusion = nn.Sequential(
            nn.Linear(phase_input_dim, fusion_hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.multimodal_fusion = nn.Sequential(
            nn.Linear(fusion_hidden_dim + clinical_hidden_dim, fusion_hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Linear(fusion_hidden_dim, num_classes)
        self.risk_head = nn.Linear(fusion_hidden_dim, 1) if use_risk_head else None

        if self.doctor_interest_enabled:
            self.doctor_interest_heads = nn.ModuleDict(
                {
                    "arterial": DoctorInterestMapHead(
                        self._phase_encoder_channels,
                        hidden_channels=doctor_interest_hidden_channels,
                        temperature=doctor_interest_temperature,
                    ),
                    "portal": DoctorInterestMapHead(
                        self._phase_encoder_channels,
                        hidden_channels=doctor_interest_hidden_channels,
                        temperature=doctor_interest_temperature,
                    ),
                    "noncontrast": DoctorInterestMapHead(
                        self._phase_encoder_channels,
                        hidden_channels=doctor_interest_hidden_channels,
                        temperature=doctor_interest_temperature,
                    ),
                }
            )
            self.interest_pool = InterestGuidedPooling3D()
            self.interest_projection = nn.ModuleDict(
                {
                    "arterial": nn.Linear(self._phase_encoder_channels, phase_feature_dim),
                    "portal": nn.Linear(self._phase_encoder_channels, phase_feature_dim),
                    "noncontrast": nn.Linear(self._phase_encoder_channels, phase_feature_dim),
                }
            )
            self.topk_focus_modules = (
                nn.ModuleDict(
                    {
                        "arterial": TopKLocalFocus3D(
                            k=topk_focus_k,
                            patch_size=topk_focus_patch_size,
                            projection_dim=topk_focus_projection_dim,
                        ),
                        "portal": TopKLocalFocus3D(
                            k=topk_focus_k,
                            patch_size=topk_focus_patch_size,
                            projection_dim=topk_focus_projection_dim,
                        ),
                        "noncontrast": TopKLocalFocus3D(
                            k=topk_focus_k,
                            patch_size=topk_focus_patch_size,
                            projection_dim=topk_focus_projection_dim,
                        ),
                    }
                )
                if topk_focus_enabled
                else None
            )
        else:
            self.doctor_interest_heads = None
            self.interest_pool = None
            self.interest_projection = None
            self.topk_focus_modules = None

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
        clinical_missing_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        arterial_features, arterial_map = self.arterial_encoder(arterial)
        portal_features, portal_map = self.portal_encoder(portal)
        noncontrast_features, noncontrast_map = self.noncontrast_encoder(noncontrast)

        doctor_interest_maps: dict[str, dict[str, torch.Tensor]] = {}
        topk_focus_centers: dict[str, torch.Tensor] = {}
        topk_focus_scores: dict[str, torch.Tensor] = {}

        phase_feature_lookup = {
            "arterial": (arterial_features, arterial_map),
            "portal": (portal_features, portal_map),
            "noncontrast": (noncontrast_features, noncontrast_map),
        }

        if self.doctor_interest_enabled and self.doctor_interest_heads is not None:
            interest_features: list[torch.Tensor] = []
            for phase_name, (phase_features, phase_map) in phase_feature_lookup.items():
                interest_head = self.doctor_interest_heads[phase_name]
                interest_outputs = interest_head(phase_map)
                doctor_interest_maps[phase_name] = interest_outputs
                pooled_features = self.interest_pool(
                    phase_map, interest_outputs["prob_map"]
                )
                interest_features.append(
                    self.interest_projection[phase_name](pooled_features)
                )
                if self.topk_focus_enabled and self.topk_focus_modules is not None:
                    focus_outputs = self.topk_focus_modules[phase_name](
                        phase_map, interest_outputs["score_map"]
                    )
                    topk_focus_centers[phase_name] = focus_outputs["centers"]
                    topk_focus_scores[phase_name] = focus_outputs["scores"]

            arterial_features = arterial_features + interest_features[0]
            portal_features = portal_features + interest_features[1]
            noncontrast_features = noncontrast_features + interest_features[2]
        else:
            doctor_interest_maps = {}
            topk_focus_centers = {}
            topk_focus_scores = {}

        if self.clinical_mask_dim > 0:
            if clinical_missing_mask is None:
                clinical_missing_mask = torch.zeros(
                    clinical.shape[0],
                    self.clinical_mask_dim,
                    dtype=clinical.dtype,
                    device=clinical.device,
                )
            clinical_inputs = torch.cat([clinical, clinical_missing_mask], dim=1)
        else:
            clinical_inputs = clinical
        clinical_features = self.clinical_encoder(clinical_inputs)

        phase_stack = torch.stack(
            [arterial_features, portal_features, noncontrast_features],
            dim=1,
        )
        if self.phase_fusion_type == "mean":
            fused_phase_input = phase_stack.mean(dim=1)
            phase_importance = torch.full(
                (phase_stack.shape[0], 3),
                1.0 / 3.0,
                dtype=phase_stack.dtype,
                device=phase_stack.device,
            )
        elif self.phase_fusion_type == "gated":
            gate_logits = self.phase_gate(phase_stack.flatten(1))
            phase_importance = torch.softmax(gate_logits, dim=1)
            fused_phase_input = (phase_stack * phase_importance.unsqueeze(-1)).sum(dim=1)
        else:
            fused_phase_input = phase_stack.flatten(1)
            phase_importance = torch.full(
                (phase_stack.shape[0], 3),
                1.0 / 3.0,
                dtype=phase_stack.dtype,
                device=phase_stack.device,
            )
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
            "clinical_features": clinical_features,
            "phase_importance": phase_importance,
            "feature_maps": {
                "arterial": arterial_map,
                "portal": portal_map,
                "noncontrast": noncontrast_map,
            },
            "doctor_interest_maps": doctor_interest_maps,
            "topk_focus_centers": topk_focus_centers,
            "topk_focus_scores": topk_focus_scores,
        }
