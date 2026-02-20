"""
Abstract base classes for backbone modules.

These classes define the interface that all backbones must implement,
ensuring consistency and pluggability across the framework.
"""

from abc import ABC, abstractmethod
from typing import Any

import torch
import torch.nn as nn


class BaseBackbone(ABC, nn.Module):
    """
    Abstract base class for all backbone modules.

    All backbones must implement:
    - forward(): Process input and return features
    - output_dim: Property returning the output feature dimension
    """

    def __init__(self):
        super().__init__()
        self._output_dim: int = 0

    @property
    def output_dim(self) -> int:
        """Return the output feature dimension."""
        return self._output_dim

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the backbone.

        Args:
            x: Input tensor

        Returns:
            Feature tensor
        """
        pass

    def get_config(self) -> dict[str, Any]:
        """Return backbone configuration for serialization."""
        return {"output_dim": self.output_dim}


class BaseVisionBackbone(BaseBackbone):
    """
    Abstract base class for vision backbones.

    Vision backbones process image inputs and extract visual features.
    They support:
    - Pretrained weight loading
    - Partial/full freezing strategies
    - Optional attention mechanisms
    - Optional attention supervision (returns intermediate features)
    - Gradient checkpointing for memory efficiency
    """

    def __init__(
        self,
        pretrained: bool = True,
        freeze: bool = False,
        feature_dim: int = 128,
    ):
        super().__init__()
        self.pretrained = pretrained
        self.freeze = freeze
        self.feature_dim = feature_dim
        self.enable_attention_supervision = False  # Set by subclasses
        self._gradient_checkpointing_enabled = False
        self._backbone: nn.Module | None = None
        self._attention: nn.Module | None = None
        self._projection: nn.Module | None = None
        self._pool: nn.Module | None = None

    @property
    @abstractmethod
    def backbone_output_dim(self) -> int:
        """Return the raw backbone output dimension before projection."""
        pass

    def freeze_backbone(self, strategy: str = "full", unfreeze_last_n: int = 2) -> None:
        """
        Freeze backbone parameters.

        Args:
            strategy: Freezing strategy
                - "full": Freeze all backbone parameters
                - "partial": Freeze all except last n layers
                - "none": Don't freeze any parameters
            unfreeze_last_n: Number of last layers to keep trainable (for partial)
        """
        if self._backbone is None:
            return

        if strategy == "none":
            return

        # Freeze all parameters first
        for param in self._backbone.parameters():
            param.requires_grad = False

        if strategy == "partial":
            # Unfreeze last n layers
            layers = list(self._backbone.children())
            for layer in layers[-unfreeze_last_n:]:
                for param in layer.parameters():
                    param.requires_grad = True

    def unfreeze_backbone(self) -> None:
        """Unfreeze all backbone parameters."""
        if self._backbone is None:
            return
        for param in self._backbone.parameters():
            param.requires_grad = True

    def set_attention(self, attention_module: nn.Module | None) -> None:
        """Set the attention module."""
        self._attention = attention_module

    def enable_gradient_checkpointing(self, segments: int | None = None) -> None:
        """
        Enable gradient checkpointing for memory-efficient training.

        Gradient checkpointing trades compute for memory by recomputing
        intermediate activations during backward pass instead of storing them.

        Args:
            segments: Number of segments to split the backbone into.
                     None means automatic (typically one checkpoint per major block).

        Example:
            >>> backbone = ResNetBackbone("resnet50")
            >>> backbone.enable_gradient_checkpointing()
            >>> # Training will use less memory but take slightly longer
        """
        self._gradient_checkpointing_enabled = True
        self._checkpoint_segments = segments
        # Subclasses should implement the actual checkpointing logic

    def disable_gradient_checkpointing(self) -> None:
        """Disable gradient checkpointing."""
        self._gradient_checkpointing_enabled = False

    def is_gradient_checkpointing_enabled(self) -> bool:
        """Check if gradient checkpointing is enabled."""
        return self._gradient_checkpointing_enabled

    @abstractmethod
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract raw features from the backbone.

        Args:
            x: Input image tensor (B, C, H, W)

        Returns:
            Feature map tensor
        """
        pass

    def forward(self, x: torch.Tensor, return_intermediates: bool = False) -> torch.Tensor | dict[str, torch.Tensor]:
        """
        Forward pass through vision backbone.

        Args:
            x: Input image tensor (B, C, H, W)
            return_intermediates: If True, return intermediate features for attention supervision

        Returns:
            If return_intermediates=False:
                Feature vector (B, feature_dim)
            If return_intermediates=True:
                Dictionary containing:
                    - "features": Output features (B, feature_dim)
                    - "feature_maps": Feature maps before pooling (B, C, H, W)
                    - "attention_weights": Spatial attention weights (B, 1, H, W) or None
        """
        # Extract features from backbone
        feature_maps = self.extract_features(x)

        # Apply attention if available
        attention_weights = None
        if self._attention is not None:
            if self.enable_attention_supervision:
                # CBAM returns (features, weights_dict) when configured
                feature_maps, weights_dict = self._attention(feature_maps)
                attention_weights = weights_dict.get("spatial_weights")
            else:
                # Normal mode
                feature_maps = self._attention(feature_maps)

        # Global average pooling
        if feature_maps.dim() == 4:
            if self._pool is not None:
                pooled = self._pool(feature_maps)
                pooled = pooled.view(pooled.size(0), -1)
            else:
                pooled = torch.mean(feature_maps, dim=[2, 3])
        else:
            pooled = feature_maps

        # Project to target dimension
        if self._projection is not None:
            features = self._projection(pooled)
        else:
            features = pooled

        # Return based on request
        if return_intermediates:
            return {
                "features": features,
                "feature_maps": feature_maps,
                "attention_weights": attention_weights,
            }
        else:
            return features


class BaseTabularBackbone(BaseBackbone):
    """
    Abstract base class for tabular (structured data) backbones.

    Tabular backbones process structured data (numerical + categorical features)
    and extract meaningful representations.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int = 32,
        hidden_dims: list[int] | None = None,
    ):
        super().__init__()
        self.input_dim = input_dim
        self._output_dim = output_dim
        self.hidden_dims = hidden_dims or [64, 64]

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through tabular backbone.

        Args:
            x: Input tabular tensor (B, input_dim)

        Returns:
            Feature vector (B, output_dim)
        """
        pass

    @classmethod
    def auto_infer_input_dim(
        cls,
        numerical_features: list[str],
        categorical_features: list[str],
        categorical_cardinalities: dict[str, int] | None = None,
    ) -> int:
        """
        Automatically infer input dimension from feature specifications.

        Args:
            numerical_features: List of numerical feature names
            categorical_features: List of categorical feature names
            categorical_cardinalities: Cardinality of each categorical feature

        Returns:
            Total input dimension
        """
        # Numerical features contribute 1 dimension each
        dim = len(numerical_features)

        # Categorical features: use one-hot encoding by default
        if categorical_cardinalities:
            for feat in categorical_features:
                dim += categorical_cardinalities.get(feat, 2)  # Default binary
        else:
            # Assume binary categorical
            dim += len(categorical_features)

        return dim
