"""Shared helpers for volumetric heatmap generation and slice selection."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F


def compute_gradcam_volume(
    feature_map: torch.Tensor,
    gradients: torch.Tensor,
    *,
    output_shape: Sequence[int],
) -> np.ndarray:
    """Project 3D feature activations and gradients into a normalized CAM volume."""
    if feature_map.ndim == 5:
        feature_map = feature_map.squeeze(0)
    if gradients.ndim == 5:
        gradients = gradients.squeeze(0)
    if feature_map.ndim != 4 or gradients.ndim != 4:
        raise ValueError("Grad-CAM expects feature maps shaped as [C, D, H, W].")
    if tuple(feature_map.shape) != tuple(gradients.shape):
        raise ValueError("Feature-map and gradient shapes must match for Grad-CAM.")

    weights = gradients.mean(dim=(1, 2, 3), keepdim=True)
    cam = torch.relu((weights * feature_map).sum(dim=0, keepdim=True).unsqueeze(0))
    if tuple(cam.shape[-3:]) != tuple(output_shape):
        cam = F.interpolate(
            cam,
            size=tuple(int(value) for value in output_shape),
            mode="trilinear",
            align_corners=False,
        )

    cam_np = cam.squeeze().detach().cpu().numpy().astype(np.float32)
    cam_np = np.nan_to_num(cam_np, nan=0.0, posinf=0.0, neginf=0.0)
    cam_min = float(cam_np.min())
    cam_max = float(cam_np.max())
    if cam_max > cam_min:
        cam_np = (cam_np - cam_min) / (cam_max - cam_min)
    else:
        cam_np = np.zeros_like(cam_np, dtype=np.float32)
    return cam_np


def select_representative_slice(heatmap_volume: np.ndarray) -> int:
    """Pick the depth slice with the strongest average heatmap activation."""
    if heatmap_volume.ndim != 3:
        raise ValueError("Representative-slice selection expects a [D, H, W] volume.")
    if heatmap_volume.shape[0] == 0:
        return 0
    slice_scores = heatmap_volume.mean(axis=(1, 2))
    if not np.isfinite(slice_scores).any() or float(slice_scores.max()) <= 0.0:
        return int(heatmap_volume.shape[0] // 2)
    return int(np.nanargmax(slice_scores))


def prepare_overlay_image(image_slice: np.ndarray) -> np.ndarray:
    """Normalize a grayscale CT slice and expand it into an RGB image."""
    if image_slice.ndim != 2:
        raise ValueError("Overlay image preparation expects a 2D slice.")
    slice_np = np.asarray(image_slice, dtype=np.float32)
    slice_np = np.nan_to_num(slice_np, nan=0.0, posinf=0.0, neginf=0.0)
    min_value = float(slice_np.min())
    max_value = float(slice_np.max())
    if max_value > min_value:
        slice_np = (slice_np - min_value) / (max_value - min_value)
    else:
        slice_np = np.zeros_like(slice_np, dtype=np.float32)
    return np.repeat(slice_np[..., None], 3, axis=-1)
