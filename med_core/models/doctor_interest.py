from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


def _spatial_softmax3d(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    batch_size = logits.size(0)
    scaled = logits.view(batch_size, -1) * temperature
    weights = torch.softmax(scaled, dim=1)
    return weights.view_as(logits)


class DoctorInterestMapHead(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int = 8, temperature: float = 6.0) -> None:
        super().__init__()
        self.temperature = temperature
        self.net = nn.Sequential(
            nn.Conv3d(in_channels, hidden_channels, kernel_size=1),
            nn.BatchNorm3d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(hidden_channels, 1, kernel_size=1),
        )

    def forward(self, feature_map: torch.Tensor) -> dict[str, torch.Tensor]:
        logits = self.net(feature_map)
        return {
            "logits": logits,
            "score_map": torch.sigmoid(logits),
            "prob_map": _spatial_softmax3d(logits, temperature=self.temperature),
        }


class InterestGuidedPooling3D(nn.Module):
    def forward(self, feature_map: torch.Tensor, prob_map: torch.Tensor) -> torch.Tensor:
        weighted = feature_map * prob_map
        return weighted.sum(dim=(2, 3, 4))


class TopKLocalFocus3D(nn.Module):
    def __init__(self, k: int, patch_size: tuple[int, int, int], projection_dim: int) -> None:
        super().__init__()
        self.k = k
        self.patch_size = patch_size
        self.projection = nn.LazyLinear(projection_dim)

    def forward(self, feature_map: torch.Tensor, score_map: torch.Tensor) -> dict[str, torch.Tensor]:
        if score_map.dim() != 5:
            raise ValueError("score_map must be a 5D tensor shaped (batch, channels, depth, height, width)")
        if score_map.size(1) != 1:
            raise ValueError(
                f"TopKLocalFocus3D expects a single-channel score_map, got {score_map.size(1)} channels"
            )
        batch, channels, depth, height, width = feature_map.shape
        flat_scores = score_map.view(batch, -1)
        spatial_positions = flat_scores.size(1)
        topk_count = min(self.k, spatial_positions)
        topk_scores, topk_indices = torch.topk(flat_scores, k=topk_count, dim=1)
        z = topk_indices // (height * width)
        y = (topk_indices % (height * width)) // width
        x = topk_indices % width
        centers = torch.stack([z, y, x], dim=-1)
        patches = []
        dz, dy, dx = self.patch_size
        for batch_index in range(batch):
            patch_features = []
            for center_index in range(self.k):
                if center_index >= centers.size(1):
                    break
                cz, cy, cx = [int(v) for v in centers[batch_index, center_index]]
                z0 = max(cz - dz // 2, 0)
                y0 = max(cy - dy // 2, 0)
                x0 = max(cx - dx // 2, 0)
                patch = feature_map[
                    batch_index : batch_index + 1,
                    :,
                    z0 : min(z0 + dz, depth),
                    y0 : min(y0 + dy, height),
                    x0 : min(x0 + dx, width),
                ]
                patch_features.append(F.adaptive_avg_pool3d(patch, (1, 1, 1)).flatten(1))
            if len(patch_features) < self.k:
                pad_slots = self.k - len(patch_features)
                pad_feature = torch.zeros(
                    (1, channels * pad_slots),
                    device=feature_map.device,
                    dtype=feature_map.dtype,
                )
                if patch_features:
                    patch_features.append(pad_feature)
                else:
                    patch_features = [pad_feature]
            patches.append(torch.cat(patch_features, dim=1))
        patch_tensor = torch.cat(patches, dim=0)
        return {
            "centers": centers,
            "scores": topk_scores,
            "feature": self.projection(patch_tensor),
        }


def build_body_mask_prior(volume: torch.Tensor, threshold_hu: float, border_ratio: float) -> torch.Tensor:
    body = (volume > threshold_hu).float()
    _, _, _, height, width = body.shape
    border_h = int(round(height * border_ratio)) if border_ratio > 0 else 0
    border_w = int(round(width * border_ratio)) if border_ratio > 0 else 0
    border_h = min(max(border_h, 0), height // 2)
    border_w = min(max(border_w, 0), width // 2)
    prior = body.clone()
    if border_h > 0:
        prior[:, :, :, :border_h, :] *= 0.5
        prior[:, :, :, height - border_h :, :] *= 0.5
    if border_w > 0:
        prior[:, :, :, :, :border_w] *= 0.5
        prior[:, :, :, :, width - border_w :] *= 0.5
    return prior


def compute_doctor_interest_losses(
    *,
    prob_map: torch.Tensor,
    teacher_map: torch.Tensor,
    augmented_prob_map: torch.Tensor | None,
    topk_centers: torch.Tensor,
    body_prior: torch.Tensor | None,
    cam_align_weight: float,
    consistency_weight: float,
    sparse_weight: float,
    diverse_weight: float,
    body_prior_weight: float,
) -> dict[str, object]:
    components: dict[str, torch.Tensor] = {}
    teacher_norm = teacher_map / (teacher_map.sum(dim=(2, 3, 4), keepdim=True) + 1e-8)
    components["cam_align"] = F.mse_loss(prob_map, teacher_norm)
    components["consistency"] = (
        F.mse_loss(prob_map, augmented_prob_map)
        if augmented_prob_map is not None
        else torch.zeros((), device=prob_map.device)
    )
    components["sparse"] = -(prob_map * torch.log(prob_map + 1e-8)).sum(dim=(2, 3, 4)).mean()
    num_centers = int(topk_centers.size(1)) if topk_centers.dim() >= 2 else 0
    if num_centers < 2:
        components["diverse"] = torch.zeros((), device=prob_map.device)
    else:
        pairwise = topk_centers.unsqueeze(2) - topk_centers.unsqueeze(1)
        distances = pairwise.float().pow(2).sum(dim=-1).sqrt()
        pairwise_mask = torch.triu(
            torch.ones(num_centers, num_centers, device=prob_map.device, dtype=torch.bool),
            diagonal=1,
        )
        valid_distances = distances[:, pairwise_mask]
        if valid_distances.numel() == 0:
            components["diverse"] = torch.zeros((), device=prob_map.device)
        else:
            components["diverse"] = torch.relu(2.0 - valid_distances).mean()
    if body_prior is not None:
        components["body_prior"] = ((1.0 - body_prior) * prob_map).mean()
    else:
        components["body_prior"] = torch.zeros((), device=prob_map.device)
    total = (
        cam_align_weight * components["cam_align"]
        + consistency_weight * components["consistency"]
        + sparse_weight * components["sparse"]
        + diverse_weight * components["diverse"]
        + body_prior_weight * components["body_prior"]
    )
    return {"total": total, "components": components}
