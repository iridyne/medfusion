"""Shared helpers for volumetric heatmap generation and slice selection."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt

from med_core.visualization.attention_viz import visualize_attention_overlay


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


def map_slice_index_between_depths(
    *,
    source_index: int,
    source_depth: int,
    target_depth: int,
) -> int:
    """Map a slice index between depth axes using endpoint-preserving scaling."""
    if source_depth <= 0 or target_depth <= 0:
        raise ValueError("Source and target depths must be positive.")
    if target_depth == 1 or source_depth == 1:
        return 0
    clamped_index = min(max(int(source_index), 0), source_depth - 1)
    mapped = round(clamped_index * (target_depth - 1) / (source_depth - 1))
    return int(min(max(mapped, 0), target_depth - 1))


def resize_heatmap_slice(
    attention_slice: np.ndarray,
    *,
    target_shape: Sequence[int],
) -> np.ndarray:
    """Resize a 2D heatmap slice into a target image shape."""
    attention_np = np.asarray(attention_slice, dtype=np.float32)
    if attention_np.ndim != 2:
        raise ValueError("Heatmap resizing expects a 2D slice.")
    if len(tuple(target_shape)) != 2:
        raise ValueError("Heatmap resizing expects a 2D target shape.")

    resized = F.interpolate(
        torch.from_numpy(attention_np).unsqueeze(0).unsqueeze(0),
        size=tuple(int(value) for value in target_shape),
        mode="bilinear",
        align_corners=False,
    )
    resized_np = resized.squeeze(0).squeeze(0).detach().cpu().numpy().astype(np.float32)
    resized_np = np.nan_to_num(resized_np, nan=0.0, posinf=0.0, neginf=0.0)
    min_value = float(resized_np.min())
    max_value = float(resized_np.max())
    if max_value > min_value:
        resized_np = (resized_np - min_value) / (max_value - min_value)
    else:
        resized_np = np.zeros_like(resized_np, dtype=np.float32)
    return resized_np


def build_rendering_metadata(
    *,
    space: str,
    kind: str,
    image_path: str | Path,
    slice_index: int,
    image_shape: Sequence[int],
) -> dict[str, object]:
    """Build a JSON-friendly artifact record for a rendered heatmap view."""
    return {
        "space": str(space),
        "kind": str(kind),
        "image_path": str(image_path),
        "slice_index": int(slice_index),
        "image_shape": [int(value) for value in image_shape],
    }


def render_overlay_artifact(
    *,
    image_slice: np.ndarray,
    attention_slice: np.ndarray,
    save_path: str | Path,
    space: str,
    kind: str,
    slice_index: int,
    title: str | None = None,
    alpha: float = 0.45,
    cmap: str = "jet",
) -> dict[str, object]:
    """Render an overlay artifact and return metadata for manifest assembly."""
    image_np = np.asarray(image_slice, dtype=np.float32)
    if image_np.ndim == 2:
        overlay_image = prepare_overlay_image(image_np)
        image_shape = image_np.shape
    elif image_np.ndim == 3:
        overlay_image = image_np
        image_shape = image_np.shape[:2]
    else:
        raise ValueError("Overlay rendering expects a 2D or 3D image slice.")

    resized_attention = resize_heatmap_slice(attention_slice, target_shape=image_shape)
    output_path = Path(save_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure = visualize_attention_overlay(
        image=overlay_image,
        attention=resized_attention,
        alpha=alpha,
        cmap=cmap,
        title=title,
        save_path=output_path,
    )
    plt.close(figure)
    return build_rendering_metadata(
        space=space,
        kind=kind,
        image_path=output_path,
        slice_index=slice_index,
        image_shape=image_shape,
    )
