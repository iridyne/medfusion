"""
Interpretability tools for medical models.

Provides Grad-CAM and attention visualization to explain model decisions.
"""

import logging
from typing import Any

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class GradCAM:
    """
    Grad-CAM implementation for visual explanations.

    Generates heatmaps showing which regions influenced the prediction.
    Supports both CNNs and hybrid architectures.
    """

    def __init__(self, model: nn.Module, target_layer: str | nn.Module | None = None):
        """
        Initialize Grad-CAM.

        Args:
            model: The model to interpret
            target_layer: The target layer to visualize (name or module).
                          If None, tries to find the last convolutional layer.
        """
        self.model = model
        self.model.eval()

        self.gradients: torch.Tensor | None = None
        self.activations: torch.Tensor | None = None

        # Find target layer if not provided
        if target_layer is None:
            self.target_layer = self._find_target_layer()
        elif isinstance(target_layer, str):
            self.target_layer = self._get_layer_by_name(target_layer)
        else:
            self.target_layer = target_layer

        if self.target_layer is None:
            raise ValueError("Could not find a valid target layer for Grad-CAM.")

        self._register_hooks()

    def _find_target_layer(self) -> nn.Module | None:
        """Heuristics to find the last convolutional layer."""
        # Check for specific architectures first
        if hasattr(self.model, "vision_backbone"):
            # It's a MultiModalFusionModel
            backbone = self.model.vision_backbone
            if hasattr(backbone, "_backbone"):
                # Inside BaseVisionBackbone
                inner = backbone._backbone

                # ResNet
                if hasattr(inner, "layer4"):
                    return inner.layer4[-1]

                # MobileNet/EfficientNet often use 'features'
                if hasattr(inner, "features"):
                    return inner.features[-1]

                # Iterating to find last Conv2d
                for module in reversed(list(inner.modules())):
                    if isinstance(module, nn.Conv2d):
                        return module

        # Generic search through the whole model
        for module in reversed(list(self.model.modules())):
            if isinstance(module, nn.Conv2d):
                return module

        return None

    def _get_layer_by_name(self, layer_name: str) -> nn.Module:
        """Get layer by dot-separated name."""
        parts = layer_name.split(".")
        current = self.model
        for part in parts:
            if part.isdigit():
                current = current[int(part)]  # type: ignore
            else:
                current = getattr(current, part)
        return current

    def _register_hooks(self) -> None:
        """Register forward and backward hooks."""

        def forward_hook(module: Any, input: Any, output: Any) -> None:
            self.activations = output.detach()

        def backward_hook(module: Any, grad_input: Any, grad_output: Any) -> None:
            self.gradients = grad_output[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate(
        self,
        input_tensor: torch.Tensor,
        target_class: int | None = None,
        return_logits: bool = False,
        additional_inputs: dict[str, Any] | None = None,
    ) -> np.ndarray | tuple[np.ndarray, torch.Tensor]:
        """
        Generate Grad-CAM heatmap.

        Args:
            input_tensor: Input image tensor (1, C, H, W)
            target_class: Target class index. If None, uses predicted class.
            return_logits: Whether to return logits along with heatmap
            additional_inputs: Optional dictionary of other inputs (e.g., 'tabular')

        Returns:
            Heatmap array (H, W) or tuple (Heatmap, Logits)
        """
        self.model.zero_grad()

        # Handle inputs
        if additional_inputs and "tabular" in additional_inputs:
            tabular_input = additional_inputs["tabular"]
        else:
            # Create dummy tabular input if not provided
            tabular_input = torch.zeros(
                input_tensor.size(0), 1, device=input_tensor.device
            )

        # Forward pass
        output = self.model(input_tensor, tabular_input)

        # Handle different output formats
        if isinstance(output, dict):
            logits = output["logits"]
        else:
            logits = output

        if target_class is None:
            target_class = int(logits.argmax(dim=1).item())

        # Backward pass
        self.model.zero_grad()
        one_hot = torch.zeros_like(logits)
        one_hot[0, target_class] = 1
        logits.backward(gradient=one_hot, retain_graph=True)

        if self.gradients is None or self.activations is None:
            raise RuntimeError(
                "Gradients or activations not captured. Check target layer."
            )

        # Compute Grad-CAM
        # Global average pooling of gradients
        weights = torch.mean(self.gradients, dim=[2, 3])  # [B, C]

        # Weighted combination of activation maps
        # activations: [B, C, H, W]
        activations = self.activations.detach()
        b, c, h, w = activations.size()

        heatmap = torch.zeros((b, h, w), device=activations.device)
        for i in range(c):
            heatmap += weights[:, i].view(b, 1, 1) * activations[:, i, :, :]

        # ReLU
        heatmap = F.relu(heatmap)

        # Normalize
        if torch.max(heatmap) > 0:
            heatmap /= torch.max(heatmap)

        # Convert to numpy and resize
        heatmap = heatmap.squeeze().cpu().numpy()

        # Resize to input image size
        input_h, input_w = input_tensor.shape[2], input_tensor.shape[3]
        heatmap = cv2.resize(heatmap, (input_w, input_h))

        if return_logits:
            return heatmap, logits
        return heatmap


def visualize_gradcam(
    image: np.ndarray | Any,  # PIL Image or numpy
    heatmap: np.ndarray,
    alpha: float = 0.4,
    colormap: int = cv2.COLORMAP_JET,
    save_path: str | None = None,
    show: bool = True,
    title: str = "Grad-CAM",
    ax: plt.Axes | None = None,
) -> np.ndarray:
    """
    Overlay Grad-CAM heatmap on image.

    Args:
        image: Original image (PIL or numpy)
        heatmap: Grad-CAM heatmap (0-1)
        alpha: Opacity of heatmap
        colormap: OpenCV colormap
        save_path: Path to save visualization
        show: Whether to display
        title: Plot title
        ax: Axes to plot on

    Returns:
        Visualization image as numpy array
    """
    # Convert image to numpy
    if not isinstance(image, np.ndarray):
        image = np.array(image)

    # Ensure RGB
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    # Resize heatmap to match image
    heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))

    # Apply colormap
    heatmap_uint8 = (heatmap * 255).astype(np.uint8)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, colormap)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

    # Overlay
    superimposed = (heatmap_color * alpha + image * (1 - alpha)).astype(np.uint8)

    # Plotting
    if show or save_path or ax:
        if ax is None:
            fig, ax = plt.subplots()
        else:
            ax.get_figure()

        ax.imshow(superimposed)
        ax.set_title(title)
        ax.axis("off")

        if save_path:
            plt.savefig(save_path, bbox_inches="tight")
            logger.info(f"Grad-CAM saved to {save_path}")

        if show and ax is None:  # Only show if we created the figure
            plt.show()

    return superimposed


def visualize_attention_weights(
    attention_weights: torch.Tensor | np.ndarray,
    title: str = "Attention Weights",
    labels: list[str] | None = None,
    save_path: str | None = None,
    show: bool = True,
    cmap: str = "viridis",
    figsize: tuple[float, float] = (8, 6),
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """
    Visualize attention weights as a heatmap.

    Args:
        attention_weights: Attention matrix (square or rectangular)
        title: Plot title
        labels: Labels for axis (e.g., feature names)
        save_path: Path to save figure
        show: Whether to display
        cmap: Colormap name
        figsize: Figure size
        ax: Axes to plot on

    Returns:
        Matplotlib axes
    """
    if isinstance(attention_weights, torch.Tensor):
        attention_weights = attention_weights.detach().cpu().numpy()

    # Handle batch dimension if present (take first sample)
    if attention_weights.ndim == 3:
        attention_weights = attention_weights[0]

    # Create figure
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        ax.get_figure()

    # Plot heatmap
    im = ax.imshow(attention_weights, cmap=cmap)

    # Add colorbar
    plt.colorbar(im, ax=ax)

    # Set title
    ax.set_title(title)

    # Set labels if provided
    if labels:
        if len(labels) == attention_weights.shape[1]:
            ax.set_xticks(np.arange(len(labels)))
            ax.set_xticklabels(labels, rotation=45, ha="right")
        if len(labels) == attention_weights.shape[0]:
            ax.set_yticks(np.arange(len(labels)))
            ax.set_yticklabels(labels)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
        logger.info(f"Attention weights saved to {save_path}")

    if show:
        plt.show()

    return ax
