"""
Configuration validation module.

Provides comprehensive validation for experiment configurations with
clear error messages and suggestions.
"""

from dataclasses import dataclass
from typing import Any

from med_core.configs.base_config import ExperimentConfig


@dataclass
class ValidationError:
    """Represents a configuration validation error."""

    path: str
    message: str
    suggestion: str | None = None
    error_code: str | None = None


class ConfigValidator:
    """Validates experiment configurations."""

    # Valid options for various config fields
    VALID_BACKBONES = [
        "resnet18", "resnet34", "resnet50", "resnet101", "resnet152",
        "mobilenet_v2", "mobilenet_v3_small", "mobilenet_v3_large",
        "efficientnet_b0", "efficientnet_b1", "efficientnet_b2",
        "efficientnet_b3", "efficientnet_b4", "efficientnet_b5",
        "efficientnet_b6", "efficientnet_b7",
        "efficientnet_v2_s", "efficientnet_v2_m", "efficientnet_v2_l",
        "convnext_tiny", "convnext_small", "convnext_base", "convnext_large",
        "regnet_y_400mf", "regnet_y_800mf", "regnet_y_1_6gf",
        "regnet_y_3_2gf", "regnet_y_8gf", "regnet_y_16gf", "regnet_y_32gf",
        "maxvit_t",
        "vit_b_16", "vit_b_32", "vit_l_16", "vit_l_32",
        "swin_t", "swin_s", "swin_b",
    ]

    VALID_FUSION_TYPES = [
        "concatenate", "gated", "attention", "cross_attention", "bilinear"
    ]

    VALID_ATTENTION_TYPES = ["cbam", "se", "eca", "none"]

    VALID_OPTIMIZERS = ["adam", "adamw", "sgd", "rmsprop"]

    VALID_SCHEDULERS = ["none", "cosine", "step", "plateau", "exponential"]

    VALID_AGGREGATORS = [
        "max", "mean", "attention", "cross_view_attention", "learned_weight"
    ]

    VALID_ATTENTION_SUPERVISION_METHODS = ["mask_guided", "cam_based", "consistency"]

    def __init__(self):
        self.errors: list[ValidationError] = []

    def validate(self, config: ExperimentConfig) -> list[ValidationError]:
        """
        Validate the entire configuration.

        Args:
            config: Configuration to validate

        Returns:
            List of validation errors (empty if valid)
        """
        self.errors = []

        # Validate each section
        self._validate_model_config(config)
        self._validate_data_config(config)
        self._validate_training_config(config)
        self._validate_logging_config(config)
        self._validate_cross_dependencies(config)

        return self.errors

    def _validate_model_config(self, config: ExperimentConfig) -> None:
        """Validate model configuration."""
        model = config.model

        # Validate num_classes
        if model.num_classes < 2:
            self.errors.append(ValidationError(
                path="model.num_classes",
                message=f"num_classes must be >= 2, got {model.num_classes}",
                suggestion="Set num_classes to at least 2 for binary classification",
                error_code="E001"
            ))

        # Validate vision config
        if model.vision.backbone not in self.VALID_BACKBONES:
            self.errors.append(ValidationError(
                path="model.vision.backbone",
                message=f"Invalid backbone '{model.vision.backbone}'",
                suggestion=f"Choose from: {', '.join(self.VALID_BACKBONES[:5])}...",
                error_code="E002"
            ))

        if model.vision.attention_type not in self.VALID_ATTENTION_TYPES:
            self.errors.append(ValidationError(
                path="model.vision.attention_type",
                message=f"Invalid attention type '{model.vision.attention_type}'",
                suggestion=f"Choose from: {', '.join(self.VALID_ATTENTION_TYPES)}",
                error_code="E003"
            ))

        if model.vision.feature_dim <= 0:
            self.errors.append(ValidationError(
                path="model.vision.feature_dim",
                message=f"feature_dim must be positive, got {model.vision.feature_dim}",
                suggestion="Set feature_dim to a positive integer (e.g., 128, 256, 512)",
                error_code="E004"
            ))

        if not 0 <= model.vision.dropout < 1:
            self.errors.append(ValidationError(
                path="model.vision.dropout",
                message=f"dropout must be in [0, 1), got {model.vision.dropout}",
                suggestion="Set dropout to a value between 0 and 1 (e.g., 0.3)",
                error_code="E005"
            ))

        # Validate tabular config
        if not model.tabular.hidden_dims:
            self.errors.append(ValidationError(
                path="model.tabular.hidden_dims",
                message="hidden_dims cannot be empty",
                suggestion="Provide at least one hidden dimension (e.g., [64])",
                error_code="E006"
            ))

        if any(dim <= 0 for dim in model.tabular.hidden_dims):
            self.errors.append(ValidationError(
                path="model.tabular.hidden_dims",
                message="All hidden dimensions must be positive",
                suggestion="Use positive integers for hidden dimensions",
                error_code="E007"
            ))

        if model.tabular.output_dim <= 0:
            self.errors.append(ValidationError(
                path="model.tabular.output_dim",
                message=f"output_dim must be positive, got {model.tabular.output_dim}",
                suggestion="Set output_dim to a positive integer (e.g., 32, 64)",
                error_code="E008"
            ))

        # Validate fusion config
        if model.fusion.fusion_type not in self.VALID_FUSION_TYPES:
            self.errors.append(ValidationError(
                path="model.fusion.fusion_type",
                message=f"Invalid fusion type '{model.fusion.fusion_type}'",
                suggestion=f"Choose from: {', '.join(self.VALID_FUSION_TYPES)}",
                error_code="E009"
            ))

        if model.fusion.hidden_dim <= 0:
            self.errors.append(ValidationError(
                path="model.fusion.hidden_dim",
                message=f"hidden_dim must be positive, got {model.fusion.hidden_dim}",
                suggestion="Set hidden_dim to a positive integer (e.g., 96, 128)",
                error_code="E010"
            ))

    def _validate_data_config(self, config: ExperimentConfig) -> None:
        """Validate data configuration."""
        data = config.data

        # Validate splits
        total_ratio = data.train_ratio + data.val_ratio + data.test_ratio
        if not (0.99 <= total_ratio <= 1.01):  # Allow small floating point errors
            self.errors.append(ValidationError(
                path="data.train_ratio/val_ratio/test_ratio",
                message=f"Split ratios must sum to 1.0, got {total_ratio:.4f}",
                suggestion="Adjust ratios so they sum to 1.0 (e.g., 0.7, 0.15, 0.15)",
                error_code="E011"
            ))

        if data.batch_size <= 0:
            self.errors.append(ValidationError(
                path="data.batch_size",
                message=f"batch_size must be positive, got {data.batch_size}",
                suggestion="Set batch_size to a positive integer (e.g., 16, 32, 64)",
                error_code="E012"
            ))

        if data.image_size <= 0:
            self.errors.append(ValidationError(
                path="data.image_size",
                message=f"image_size must be positive, got {data.image_size}",
                suggestion="Set image_size to a positive integer (e.g., 224, 256, 512)",
                error_code="E013"
            ))

        if data.num_workers < 0:
            self.errors.append(ValidationError(
                path="data.num_workers",
                message=f"num_workers must be non-negative, got {data.num_workers}",
                suggestion="Set num_workers to 0 or a positive integer",
                error_code="E014"
            ))

        # Validate multiview config if enabled (only for MultiViewDataConfig)
        if hasattr(data, 'enable_multiview') and data.enable_multiview:
            if hasattr(data, 'view_names') and not data.view_names:
                self.errors.append(ValidationError(
                    path="data.view_names",
                    message="view_names cannot be empty when enable_multiview=True",
                    suggestion="Provide view names (e.g., ['axial', 'coronal', 'sagittal'])",
                    error_code="E015"
                ))

            if hasattr(data, 'aggregator_type') and data.aggregator_type not in self.VALID_AGGREGATORS:
                self.errors.append(ValidationError(
                    path="data.aggregator_type",
                    message=f"Invalid aggregator type '{data.aggregator_type}'",
                    suggestion=f"Choose from: {', '.join(self.VALID_AGGREGATORS)}",
                    error_code="E016"
                ))

    def _validate_training_config(self, config: ExperimentConfig) -> None:
        """Validate training configuration."""
        training = config.training

        if training.num_epochs <= 0:
            self.errors.append(ValidationError(
                path="training.num_epochs",
                message=f"num_epochs must be positive, got {training.num_epochs}",
                suggestion="Set num_epochs to a positive integer (e.g., 50, 100)",
                error_code="E017"
            ))

        if training.gradient_clip is not None and training.gradient_clip <= 0:
            self.errors.append(ValidationError(
                path="training.gradient_clip",
                message=f"gradient_clip must be positive, got {training.gradient_clip}",
                suggestion="Set gradient_clip to a positive value (e.g., 1.0) or None",
                error_code="E018"
            ))

        if not 0 <= training.label_smoothing < 1:
            self.errors.append(ValidationError(
                path="training.label_smoothing",
                message=f"label_smoothing must be in [0, 1), got {training.label_smoothing}",
                suggestion="Set label_smoothing to a value between 0 and 1 (e.g., 0.1)",
                error_code="E019"
            ))

        # Validate progressive training
        if training.use_progressive_training:
            total_epochs = (training.stage1_epochs + training.stage2_epochs +
                          training.stage3_epochs)
            if total_epochs != training.num_epochs:
                self.errors.append(ValidationError(
                    path="training.stage*_epochs",
                    message=f"Stage epochs sum ({total_epochs}) != num_epochs ({training.num_epochs})",
                    suggestion=f"Adjust stage epochs to sum to {training.num_epochs}",
                    error_code="E020"
                ))

        # Validate optimizer
        if training.optimizer.optimizer not in self.VALID_OPTIMIZERS:
            self.errors.append(ValidationError(
                path="training.optimizer.optimizer",
                message=f"Invalid optimizer '{training.optimizer.optimizer}'",
                suggestion=f"Choose from: {', '.join(self.VALID_OPTIMIZERS)}",
                error_code="E021"
            ))

        if training.optimizer.learning_rate <= 0:
            self.errors.append(ValidationError(
                path="training.optimizer.learning_rate",
                message=f"learning_rate must be positive, got {training.optimizer.learning_rate}",
                suggestion="Set learning_rate to a positive value (e.g., 1e-4, 1e-3)",
                error_code="E022"
            ))

        # Validate scheduler
        if training.scheduler.scheduler not in self.VALID_SCHEDULERS:
            self.errors.append(ValidationError(
                path="training.scheduler.scheduler",
                message=f"Invalid scheduler '{training.scheduler.scheduler}'",
                suggestion=f"Choose from: {', '.join(self.VALID_SCHEDULERS)}",
                error_code="E023"
            ))

        # Validate attention supervision
        if training.use_attention_supervision:
            # Check if attention_supervision_method exists and is valid
            if hasattr(training, 'attention_supervision_method'):
                method = training.attention_supervision_method
                # Handle both old format ("mask", "cam") and new format ("mask_guided", "cam_based")
                valid_methods = self.VALID_ATTENTION_SUPERVISION_METHODS + ["mask", "cam", "none"]
                if method not in valid_methods:
                    self.errors.append(ValidationError(
                        path="training.attention_supervision_method",
                        message=f"Invalid method '{method}'",
                        suggestion=f"Choose from: {', '.join(self.VALID_ATTENTION_SUPERVISION_METHODS)}",
                        error_code="E024"
                    ))

            if hasattr(training, 'attention_loss_weight'):
                if not 0 <= training.attention_loss_weight <= 1:
                    self.errors.append(ValidationError(
                        path="training.attention_loss_weight",
                        message=f"attention_loss_weight must be in [0, 1], got {training.attention_loss_weight}",
                        suggestion="Set attention_loss_weight between 0 and 1 (e.g., 0.1)",
                        error_code="E025"
                    ))

    def _validate_logging_config(self, config: ExperimentConfig) -> None:
        """Validate logging configuration."""
        logging = config.logging

        if logging.log_every_n_steps <= 0:
            self.errors.append(ValidationError(
                path="logging.log_every_n_steps",
                message=f"log_every_n_steps must be positive, got {logging.log_every_n_steps}",
                suggestion="Set log_every_n_steps to a positive integer (e.g., 10, 50)",
                error_code="E026"
            ))

        if logging.use_wandb and not logging.wandb_project:
            self.errors.append(ValidationError(
                path="logging.wandb_project",
                message="wandb_project must be set when use_wandb=True",
                suggestion="Provide a wandb project name (e.g., 'medical-ai')",
                error_code="E027"
            ))

    def _validate_cross_dependencies(self, config: ExperimentConfig) -> None:
        """Validate cross-field dependencies."""
        # Attention supervision requires CBAM
        if config.training.use_attention_supervision:
            if config.model.vision.attention_type != "cbam":
                self.errors.append(ValidationError(
                    path="training.use_attention_supervision",
                    message="Attention supervision requires CBAM attention mechanism",
                    suggestion="Set model.vision.attention_type='cbam' or disable attention supervision",
                    error_code="E028"
                ))

            if not config.model.vision.enable_attention_supervision:
                self.errors.append(ValidationError(
                    path="model.vision.enable_attention_supervision",
                    message="Must enable attention supervision in vision config",
                    suggestion="Set model.vision.enable_attention_supervision=True",
                    error_code="E029"
                ))

        # Multiview requires consistent configuration (only for MultiView configs)
        if hasattr(config.data, 'enable_multiview') and config.data.enable_multiview:
            if hasattr(config.model.vision, 'enable_multiview'):
                if not config.model.vision.enable_multiview:
                    self.errors.append(ValidationError(
                        path="model.vision.enable_multiview",
                        message="Model must enable multiview when data has multiview enabled",
                        suggestion="Set model.vision.enable_multiview=True",
                        error_code="E030"
                    ))


def validate_config(config: ExperimentConfig) -> list[ValidationError]:
    """
    Validate an experiment configuration.

    Args:
        config: Configuration to validate

    Returns:
        List of validation errors (empty if valid)

    Example:
        >>> config = ExperimentConfig.from_yaml("config.yaml")
        >>> errors = validate_config(config)
        >>> if errors:
        ...     for error in errors:
        ...         print(f"❌ {error.path}: {error.message}")
        ...         if error.suggestion:
        ...             print(f"   💡 {error.suggestion}")
    """
    validator = ConfigValidator()
    return validator.validate(config)


def validate_config_or_exit(config: ExperimentConfig) -> None:
    """
    Validate configuration and exit if errors found.

    Args:
        config: Configuration to validate

    Raises:
        SystemExit: If validation errors are found
    """
    import sys

    errors = validate_config(config)
    if errors:
        print("❌ Configuration validation failed:\n")
        for error in errors:
            print(f"  [{error.error_code}] {error.path}")
            print(f"    ❌ {error.message}")
            if error.suggestion:
                print(f"    💡 Suggestion: {error.suggestion}")
            print()
        sys.exit(1)
    else:
        print("✅ Configuration validation passed")
