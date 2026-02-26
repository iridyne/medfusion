"""
Generic Multi-Modal Model Builder

Provides a unified interface for building multi-modal models from components.
"""

from typing import Any, Literal

import torch
from torch import nn

from med_core.aggregators import MILAggregator
from med_core.backbones import (
    create_tabular_backbone,
    create_vision_backbone,
)
from med_core.backbones.swin_2d import SwinTransformer2DBackbone
from med_core.backbones.swin_3d import SwinTransformer3DBackbone
from med_core.fusion import create_fusion_module
from med_core.heads import ClassificationHead
from med_core.heads.survival import (
    CoxSurvivalHead,
    DeepSurvivalHead,
    DiscreteTimeSurvivalHead,
)


class GenericMultiModalModel(nn.Module):
    """
    Generic multi-modal model that combines arbitrary modalities.

    This model supports:
    - Arbitrary number of modalities (2+)
    - Any backbone for each modality
    - Any fusion strategy
    - Any task head
    - Optional MIL aggregation per modality

    Args:
        modality_backbones: Dict mapping modality names to backbone modules
        fusion_module: Fusion module to combine modality features
        head: Task-specific head (classification, survival, etc.)
        mil_aggregators: Optional dict of MIL aggregators per modality
        modality_names: List of modality names in order

    Example:
        >>> backbones = {
        ...     'ct': SwinTransformer3DBackbone(...),
        ...     'pathology': SwinTransformer2DBackbone(...),
        ... }
        >>> fusion = FusedAttentionFusion(...)
        >>> head = ClassificationHead(...)
        >>> model = GenericMultiModalModel(backbones, fusion, head)
        >>> outputs = model({'ct': ct_data, 'pathology': path_data})
    """

    def __init__(
        self,
        modality_backbones: dict[str, nn.Module],
        fusion_module: nn.Module,
        head: nn.Module,
        mil_aggregators: dict[str, nn.Module] | None = None,
        modality_names: list[str] | None = None,
    ):
        super().__init__()

        self.modality_names = modality_names or sorted(modality_backbones.keys())
        self.modality_backbones = nn.ModuleDict(modality_backbones)
        self.fusion_module = fusion_module
        self.head = head
        self.mil_aggregators = nn.ModuleDict(mil_aggregators or {})

        # Validate
        if len(self.modality_backbones) < 2:
            raise ValueError("At least 2 modalities are required for multi-modal model")

    def forward(
        self,
        inputs: dict[str, torch.Tensor],
        return_features: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, Any]]:
        """
        Forward pass through the multi-modal model.

        Args:
            inputs: Dict mapping modality names to input tensors
            return_features: Whether to return intermediate features

        Returns:
            Model output (logits, survival predictions, etc.)
            If return_features=True, also returns dict of intermediate features
        """
        # Extract features from each modality
        modality_features = {}
        mil_attention_weights = {}

        for modality_name in self.modality_names:
            if modality_name not in inputs:
                raise ValueError(f"Missing input for modality: {modality_name}")

            x = inputs[modality_name]
            backbone = self.modality_backbones[modality_name]

            # Extract features
            # Handle MIL input: [B, N, C, H, W] -> [B*N, C, H, W] -> backbone -> [B, N, feature_dim]
            if modality_name in self.mil_aggregators and x.ndim == 5:
                B, N, C, H, W = x.shape
                # Reshape to [B*N, C, H, W] for backbone processing
                x_reshaped = x.view(B * N, C, H, W)
                features = backbone(x_reshaped)
                # Reshape back to [B, N, feature_dim]
                features = features.view(B, N, -1)
            elif modality_name in self.mil_aggregators and x.ndim == 6:
                # 3D MIL input: [B, N, C, D, H, W]
                B, N, C, D, H, W = x.shape
                x_reshaped = x.view(B * N, C, D, H, W)
                features = backbone(x_reshaped)
                features = features.view(B, N, -1)
            else:
                features = backbone(x)

            # Apply MIL aggregation if configured
            if modality_name in self.mil_aggregators:
                # Save pre-aggregation features
                modality_features[f"{modality_name}_patches"] = features

                aggregator = self.mil_aggregators[modality_name]
                if (
                    hasattr(aggregator, "forward")
                    and "return_attention" in aggregator.forward.__code__.co_varnames
                ):
                    aggregated_features, attention = aggregator(
                        features, return_attention=True,
                    )
                    mil_attention_weights[modality_name] = attention
                else:
                    aggregated_features = aggregator(features)

                # Save post-aggregation features
                modality_features[f"{modality_name}_aggregated"] = aggregated_features
                modality_features[modality_name] = aggregated_features
            else:
                modality_features[modality_name] = features

        # Fuse features
        # Handle different fusion module interfaces
        if len(self.modality_names) == 2:
            # Binary fusion (most common case)
            feat1 = modality_features[self.modality_names[0]]
            feat2 = modality_features[self.modality_names[1]]

            # Check fusion module signature
            if hasattr(self.fusion_module, "forward"):
                fusion_result = self.fusion_module(feat1, feat2)
                if isinstance(fusion_result, tuple):
                    fused_features, fusion_aux = fusion_result
                else:
                    fused_features = fusion_result
                    fusion_aux = None
            else:
                fused_features = self.fusion_module(feat1, feat2)
                fusion_aux = None
        else:
            # Multi-modal fusion (>2 modalities)
            # Stack features and use attention-based pooling
            feature_list = [modality_features[name] for name in self.modality_names]
            feature_tensor = torch.stack(
                feature_list, dim=1,
            )  # [B, num_modalities, feature_dim]

            # Use learnable attention weights for weighted pooling
            if not hasattr(self, 'multimodal_attention'):
                # Initialize attention weights on first use
                feature_dim = feature_tensor.size(-1)
                self.multimodal_attention = nn.Sequential(
                    nn.Linear(feature_dim, feature_dim // 4),
                    nn.ReLU(),
                    nn.Linear(feature_dim // 4, 1),
                ).to(feature_tensor.device)

            # Compute attention scores: [B, num_modalities, 1]
            attention_scores = self.multimodal_attention(feature_tensor)
            attention_weights = torch.softmax(attention_scores, dim=1)

            # Weighted sum: [B, feature_dim]
            fused_features = (feature_tensor * attention_weights).sum(dim=1)
            fusion_aux = {"multimodal_attention_weights": attention_weights.squeeze(-1)}

        # Apply task head
        output = self.head(fused_features)

        if return_features:
            features_dict = {
                "modality_features": modality_features,
                "fused_features": fused_features,
            }
            if mil_attention_weights:
                features_dict["mil_attention_weights"] = mil_attention_weights
            if fusion_aux is not None:
                features_dict["fusion_aux"] = fusion_aux

            return output, features_dict

        return output

    def get_modality_contribution(self) -> dict[str, float]:
        """
        Get the contribution of each modality to the final prediction.

        Returns:
            Dict mapping modality names to contribution scores
        """
        if hasattr(self.fusion_module, "get_attention_weights"):
            attention = self.fusion_module.get_attention_weights()
            if attention is not None:
                # Compute contribution from attention weights
                contributions = {}
                for i, name in enumerate(self.modality_names):
                    contributions[name] = attention[..., i].mean().item()
                return contributions

        # Default: equal contribution
        equal_contrib = 1.0 / len(self.modality_names)
        return dict.fromkeys(self.modality_names, equal_contrib)


class MultiModalModelBuilder:
    """
    Builder for constructing multi-modal models.

    Provides a fluent API for configuring and building multi-modal models
    from individual components.

    Example:
        >>> builder = MultiModalModelBuilder()
        >>> model = (builder
        ...     .add_modality('ct', backbone='swin3d_small', in_channels=1)
        ...     .add_modality('pathology', backbone='swin2d_small', in_channels=3)
        ...     .set_fusion('fused_attention', num_heads=8)
        ...     .set_head('classification', num_classes=4)
        ...     .build())
    """

    def __init__(self) -> None:
        self._modalities: dict[str, dict[str, Any]] = {}
        self._fusion_config: dict[str, Any] | None = None
        self._head_config: dict[str, Any] | None = None
        self._mil_configs: dict[str, dict[str, Any]] = {}

    def add_modality(
        self,
        name: str,
        backbone: str | nn.Module,
        modality_type: Literal["vision", "vision3d", "tabular", "custom"] = "vision",
        feature_dim: int | None = None,
        **kwargs: Any,
    ) -> "MultiModalModelBuilder":
        """
        Add a modality to the model.

        Args:
            name: Modality name (e.g., 'ct', 'pathology', 'clinical')
            backbone: Backbone name (e.g., 'resnet18', 'swin3d_small') or module
            modality_type: Type of modality ('vision', 'vision3d', 'tabular', 'custom')
            feature_dim: Output feature dimension (if None, use backbone default)
            **kwargs: Additional arguments for backbone creation

        Returns:
            Self for chaining
        """
        self._modalities[name] = {
            "backbone": backbone,
            "modality_type": modality_type,
            "feature_dim": feature_dim,
            "kwargs": kwargs,
        }
        return self

    def add_mil_aggregation(
        self,
        modality_name: str,
        strategy: Literal[
            "mean", "max", "attention", "gated", "deepsets", "transformer",
        ] = "attention",
        **kwargs: Any,
    ) -> "MultiModalModelBuilder":
        """
        Add MIL aggregation for a modality.

        Args:
            modality_name: Name of the modality to apply MIL to
            strategy: Aggregation strategy
            **kwargs: Additional arguments for aggregator

        Returns:
            Self for chaining
        """
        if modality_name not in self._modalities:
            raise ValueError(
                f"Modality '{modality_name}' not found. Add it first with add_modality()",
            )

        self._mil_configs[modality_name] = {
            "strategy": strategy,
            "kwargs": kwargs,
        }
        return self

    def set_fusion(
        self,
        strategy: Literal[
            "concat",
            "gated",
            "attention",
            "cross_attention",
            "bilinear",
            "kronecker",
            "fused_attention",
        ],
        **kwargs: Any,
    ) -> "MultiModalModelBuilder":
        """
        Set the fusion strategy.

        Args:
            strategy: Fusion strategy name
            **kwargs: Additional arguments for fusion module

        Returns:
            Self for chaining
        """
        self._fusion_config = {
            "strategy": strategy,
            "kwargs": kwargs,
        }
        return self

    def set_head(
        self,
        task_type: Literal[
            "classification", "survival_cox", "survival_deep", "survival_discrete",
        ],
        **kwargs: Any,
    ) -> "MultiModalModelBuilder":
        """
        Set the task head.

        Args:
            task_type: Type of task head
            **kwargs: Additional arguments for head (e.g., num_classes, hidden_dims)

        Returns:
            Self for chaining
        """
        self._head_config = {
            "task_type": task_type,
            "kwargs": kwargs,
        }
        return self

    def build(self) -> GenericMultiModalModel:
        """
        Build the multi-modal model.

        Returns:
            Configured GenericMultiModalModel

        Raises:
            ValueError: If configuration is incomplete or invalid
        """
        # Validate configuration
        if len(self._modalities) < 2:
            raise ValueError("At least 2 modalities are required")
        if self._fusion_config is None:
            raise ValueError("Fusion strategy not set. Call set_fusion() first")
        if self._head_config is None:
            raise ValueError("Task head not set. Call set_head() first")

        # Build backbones
        modality_backbones = {}
        modality_dims = {}

        for name, config in self._modalities.items():
            backbone = config["backbone"]
            modality_type = config["modality_type"]
            feature_dim = config["feature_dim"]
            kwargs = config["kwargs"]

            # Create backbone
            if isinstance(backbone, nn.Module):
                # Use provided module
                modality_backbones[name] = backbone
                if hasattr(backbone, "output_dim"):
                    modality_dims[name] = backbone.output_dim
                elif feature_dim is not None:
                    modality_dims[name] = feature_dim
                else:
                    raise ValueError(
                        f"Cannot determine output dimension for modality '{name}'",
                    )
            # Create from string
            elif modality_type == "vision":
                # Handle Swin2D separately as it needs in_channels
                if "swin2d" in backbone.lower() or "swin_2d" in backbone.lower():
                    variant = backbone.replace("swin2d_", "").replace(
                        "swin_2d_", "",
                    )
                    bb = SwinTransformer2DBackbone(
                        variant=variant,
                        feature_dim=feature_dim or 512,
                        **kwargs,
                    )
                    modality_backbones[name] = bb
                    modality_dims[name] = bb.output_dim
                else:
                    # Remove in_channels from kwargs for standard vision backbones
                    vision_kwargs = {
                        k: v for k, v in kwargs.items() if k != "in_channels"
                    }
                    bb = create_vision_backbone(
                        backbone_name=backbone,
                        feature_dim=feature_dim or 512,
                        **vision_kwargs,
                    )
                    modality_backbones[name] = bb
                    modality_dims[name] = bb.output_dim
            elif modality_type == "vision3d":
                # Handle 3D backbones
                if "swin3d" in backbone.lower():
                    variant = backbone.replace("swin3d_", "").replace(
                        "swin_3d_", "",
                    )
                    bb = SwinTransformer3DBackbone(
                        variant=variant,
                        feature_dim=feature_dim or 512,
                        **kwargs,
                    )
                    modality_backbones[name] = bb
                    modality_dims[name] = bb.output_dim
                else:
                    raise ValueError(f"Unsupported 3D backbone: {backbone}")
            elif modality_type == "tabular":
                input_dim = kwargs.pop("input_dim", None)
                if input_dim is None:
                    raise ValueError(
                        f"input_dim required for tabular modality '{name}'",
                    )
                bb = create_tabular_backbone(
                    input_dim=input_dim,
                    output_dim=feature_dim or 64,
                    **kwargs,
                )
                modality_backbones[name] = bb
                modality_dims[name] = bb.output_dim
            else:
                raise ValueError(f"Unsupported modality type: {modality_type}")

        # Build MIL aggregators
        mil_aggregators = {}
        for modality_name, mil_config in self._mil_configs.items():
            input_dim = modality_dims[modality_name]
            aggregator = MILAggregator(
                input_dim=input_dim,
                strategy=mil_config["strategy"],
                **mil_config["kwargs"],
            )
            mil_aggregators[modality_name] = aggregator

            # Update dimension if aggregator changes it
            if hasattr(aggregator, "output_dim"):
                modality_dims[modality_name] = aggregator.output_dim

        # Build fusion module
        modality_names = sorted(self._modalities.keys())
        if len(modality_names) == 2:
            # Binary fusion
            dim1 = modality_dims[modality_names[0]]
            dim2 = modality_dims[modality_names[1]]

            fusion_strategy = self._fusion_config["strategy"]
            fusion_kwargs = self._fusion_config["kwargs"].copy()

            # Map fusion strategy aliases to canonical names
            fusion_alias_map = {
                "concat": "concatenate",
                "concatenate": "concatenate",
                "gated": "gated",
                "attention": "attention",
                "cross_attention": "cross_attention",
                "bilinear": "bilinear",
            }

            # Handle special fusion types
            if fusion_strategy in ["kronecker", "fused_attention"]:
                # These are not in the standard fusion registry
                # For now, fall back to concatenate and clear incompatible kwargs
                fusion_strategy = "concatenate"
                # Only keep output_dim if present
                fusion_kwargs = {
                    k: v for k, v in fusion_kwargs.items() if k == "output_dim"
                }
            elif fusion_strategy in fusion_alias_map:
                fusion_strategy = fusion_alias_map[fusion_strategy]
            else:
                raise ValueError(
                    f"Unknown fusion strategy: {fusion_strategy}. "
                    f"Available: {list(fusion_alias_map.keys()) + ['kronecker', 'fused_attention']}",
                )

            # Determine output dimension
            if "output_dim" not in fusion_kwargs:
                if fusion_strategy == "concatenate":
                    fusion_kwargs["output_dim"] = dim1 + dim2
                else:
                    fusion_kwargs["output_dim"] = min(dim1, dim2)

            fusion_module = create_fusion_module(
                fusion_type=fusion_strategy,
                vision_dim=dim1,
                tabular_dim=dim2,
                **fusion_kwargs,
            )
            fusion_dim = fusion_kwargs["output_dim"]
        else:
            # Multi-modal fusion (>2 modalities)
            # For now, use simple mean pooling
            # All modalities should have same dimension
            dims = list(modality_dims.values())
            if len(set(dims)) > 1:
                raise ValueError(
                    f"For >2 modalities, all feature dimensions must match. Got: {modality_dims}",
                )
            fusion_dim = dims[0]
            fusion_module = nn.Identity()  # Placeholder

        # Build task head
        head_type = self._head_config["task_type"]
        head_kwargs = self._head_config["kwargs"].copy()

        if head_type == "classification":
            head = ClassificationHead(
                input_dim=fusion_dim,
                **head_kwargs,
            )
        elif head_type == "survival_cox":
            head = CoxSurvivalHead(
                input_dim=fusion_dim,
                **head_kwargs,
            )
        elif head_type == "survival_deep":
            head = DeepSurvivalHead(
                input_dim=fusion_dim,
                **head_kwargs,
            )
        elif head_type == "survival_discrete":
            head = DiscreteTimeSurvivalHead(
                input_dim=fusion_dim,
                **head_kwargs,
            )
        else:
            raise ValueError(f"Unsupported head type: {head_type}")

        # Build final model
        model = GenericMultiModalModel(
            modality_backbones=modality_backbones,
            fusion_module=fusion_module,
            head=head,
            mil_aggregators=mil_aggregators or None,
            modality_names=modality_names,
        )

        return model

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "MultiModalModelBuilder":
        """
        Create builder from configuration dict.

        Args:
            config: Configuration dictionary with keys:
                - modalities: Dict of modality configurations
                - fusion: Fusion configuration
                - head: Head configuration
                - mil: Optional MIL configurations

        Returns:
            Configured builder

        Example:
            >>> config = {
            ...     'modalities': {
            ...         'ct': {'backbone': 'swin3d_small', 'modality_type': 'vision3d'},
            ...         'pathology': {'backbone': 'swin2d_small', 'modality_type': 'vision'},
            ...     },
            ...     'fusion': {'strategy': 'fused_attention', 'num_heads': 8},
            ...     'head': {'task_type': 'classification', 'num_classes': 4},
            ... }
            >>> builder = MultiModalModelBuilder.from_config(config)
            >>> model = builder.build()
        """
        builder = cls()

        # Add modalities
        for name, mod_config in config["modalities"].items():
            mod_config = mod_config.copy()
            backbone = mod_config.pop("backbone")
            modality_type = mod_config.pop("modality_type", "vision")
            feature_dim = mod_config.pop("feature_dim", None)
            builder.add_modality(
                name, backbone, modality_type, feature_dim, **mod_config,
            )

        # Add MIL if configured
        if "mil" in config:
            for modality_name, mil_config in config["mil"].items():
                mil_config = mil_config.copy()
                strategy = mil_config.pop("strategy", "attention")
                builder.add_mil_aggregation(modality_name, strategy, **mil_config)

        # Set fusion
        fusion_config = config["fusion"].copy()
        strategy = fusion_config.pop("strategy")
        builder.set_fusion(strategy, **fusion_config)

        # Set head
        head_config = config["head"].copy()
        task_type = head_config.pop("task_type")
        builder.set_head(task_type, **head_config)

        return builder


def build_model_from_config(
    config: str | dict[str, Any],
) -> GenericMultiModalModel:
    """
    Build a multi-modal model from configuration.

    Args:
        config: Configuration dict or path to YAML file

    Returns:
        Configured GenericMultiModalModel

    Example:
        >>> config = {
        ...     'modalities': {...},
        ...     'fusion': {...},
        ...     'head': {...},
        ... }
        >>> model = build_model_from_config(config)
    """
    if isinstance(config, str):
        # Load from YAML file
        import yaml

        with open(config) as f:
            config = yaml.safe_load(f)

    # Extract model config if nested
    if "model" in config:
        config = config["model"]

    builder = MultiModalModelBuilder.from_config(config)
    return builder.build()
