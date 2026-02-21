"""
Configuration validation module.

Provides basic validation for experiment configurations with
clear error messages.
"""

import logging

from med_core.configs.base_config import ExperimentConfig

logger = logging.getLogger(__name__)


class ConfigValidator:
    """Validates experiment configurations with basic checks."""

    def __init__(self):
        self.errors: list[str] = []

    def validate(self, config: ExperimentConfig) -> list[str]:
        """
        Validate the entire configuration.

        Args:
            config: Configuration to validate

        Returns:
            List of validation error messages (empty if valid)
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
            self.errors.append(
                f"model.num_classes must be >= 2, got {model.num_classes}"
            )

        # Validate vision config
        if model.vision.feature_dim <= 0:
            self.errors.append(
                f"model.vision.feature_dim must be positive, got {model.vision.feature_dim}"
            )

        if not 0 <= model.vision.dropout < 1:
            self.errors.append(
                f"model.vision.dropout must be in [0, 1), got {model.vision.dropout}"
            )

        # Validate tabular config
        if not model.tabular.hidden_dims:
            self.errors.append(
                "model.tabular.hidden_dims cannot be empty"
            )

        if any(dim <= 0 for dim in model.tabular.hidden_dims):
            self.errors.append(
                "model.tabular.hidden_dims: all dimensions must be positive"
            )

        if model.tabular.output_dim <= 0:
            self.errors.append(
                f"model.tabular.output_dim must be positive, got {model.tabular.output_dim}"
            )

        # Validate fusion config
        if model.fusion.hidden_dim <= 0:
            self.errors.append(
                f"model.fusion.hidden_dim must be positive, got {model.fusion.hidden_dim}"
            )

    def _validate_data_config(self, config: ExperimentConfig) -> None:
        """Validate data configuration."""
        data = config.data

        # Validate splits
        total_ratio = data.train_ratio + data.val_ratio + data.test_ratio
        if not (0.99 <= total_ratio <= 1.01):  # Allow small floating point errors
            self.errors.append(
                f"data split ratios must sum to 1.0, got {total_ratio:.4f}"
            )

        if data.batch_size <= 0:
            self.errors.append(
                f"data.batch_size must be positive, got {data.batch_size}"
            )

        if data.image_size <= 0:
            self.errors.append(
                f"data.image_size must be positive, got {data.image_size}"
            )

        if data.num_workers < 0:
            self.errors.append(
                f"data.num_workers must be non-negative, got {data.num_workers}"
            )

        # Validate multiview config if enabled
        if hasattr(data, 'enable_multiview') and data.enable_multiview:
            if hasattr(data, 'view_names') and not data.view_names:
                self.errors.append(
                    "data.view_names cannot be empty when enable_multiview=True"
                )

    def _validate_training_config(self, config: ExperimentConfig) -> None:
        """Validate training configuration."""
        training = config.training

        if training.num_epochs <= 0:
            self.errors.append(
                f"training.num_epochs must be positive, got {training.num_epochs}"
            )

        if training.gradient_clip is not None and training.gradient_clip <= 0:
            self.errors.append(
                f"training.gradient_clip must be positive or None, got {training.gradient_clip}"
            )

        if not 0 <= training.label_smoothing < 1:
            self.errors.append(
                f"training.label_smoothing must be in [0, 1), got {training.label_smoothing}"
            )

        # Validate progressive training
        if training.use_progressive_training:
            total_epochs = (training.stage1_epochs + training.stage2_epochs +
                          training.stage3_epochs)
            if total_epochs != training.num_epochs:
                self.errors.append(
                    f"training stage epochs sum ({total_epochs}) must equal num_epochs ({training.num_epochs})"
                )

        # Validate optimizer
        if training.optimizer.learning_rate <= 0:
            self.errors.append(
                f"training.optimizer.learning_rate must be positive, got {training.optimizer.learning_rate}"
            )

        # Validate attention supervision
        if training.use_attention_supervision:
            if hasattr(training, 'attention_loss_weight'):
                if not 0 <= training.attention_loss_weight <= 1:
                    self.errors.append(
                        f"training.attention_loss_weight must be in [0, 1], got {training.attention_loss_weight}"
                    )

    def _validate_logging_config(self, config: ExperimentConfig) -> None:
        """Validate logging configuration."""
        logging = config.logging

        if logging.log_every_n_steps <= 0:
            self.errors.append(
                f"logging.log_every_n_steps must be positive, got {logging.log_every_n_steps}"
            )

        if logging.save_every_n_epochs <= 0:
            self.errors.append(
                f"logging.save_every_n_epochs must be positive, got {logging.save_every_n_epochs}"
            )

    def _validate_cross_dependencies(self, config: ExperimentConfig) -> None:
        """Validate cross-dependencies between config sections."""
        # Attention supervision requires attention to be enabled
        if config.training.use_attention_supervision:
            if not config.model.vision.enable_attention_supervision:
                self.errors.append(
                    "training.use_attention_supervision=True requires model.vision.enable_attention_supervision=True"
                )

        # Multiview requires consistent configuration
        if hasattr(config.data, 'enable_multiview') and config.data.enable_multiview:
            if hasattr(config.model.vision, 'enable_multiview'):
                if not config.model.vision.enable_multiview:
                    self.errors.append(
                        "data.enable_multiview=True requires model.vision.enable_multiview=True"
                    )


def validate_config(config: ExperimentConfig) -> list[str]:
    """
    Validate an experiment configuration.

    Args:
        config: Configuration to validate

    Returns:
        List of validation error messages (empty if valid)

    Example:
        >>> config = ExperimentConfig.from_yaml("config.yaml")
        >>> errors = validate_config(config)
        >>> if errors:
        ...     for error in errors:
        ...         print(f"❌ {error}")
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
        logger.error("Configuration validation failed:\n")
        for i, error in enumerate(errors, 1):
            logger.error(f"  {i}. {error}")
        sys.exit(1)
    else:
        logger.info("✅ Configuration validation passed")
