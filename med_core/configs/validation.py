"""
Configuration validation module.

Provides basic validation for experiment configurations with
clear error messages.
"""

import logging
from dataclasses import dataclass

from med_core.backbones.vision import list_available_backbones
from med_core.configs.base_config import ExperimentConfig
from med_core.fusion.strategies import list_available_fusions

logger = logging.getLogger(__name__)


VALID_BACKBONES = frozenset(list_available_backbones())
VALID_FUSION_TYPES = frozenset(list_available_fusions())
VALID_THREE_PHASE_FUSION_TYPES = frozenset({"concatenate", "mean"})


@dataclass
class ValidationError:
    """Structured validation error with path, code, and suggestion."""

    path: str
    message: str
    error_code: str
    suggestion: str | None = None


class ConfigValidator:
    """Validates experiment configurations with basic checks."""

    def __init__(self) -> None:
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
            self.errors.append(
                ValidationError(
                    path="model.num_classes",
                    message=f"num_classes must be >= 2, got {model.num_classes}",
                    error_code="E001",
                    suggestion="Set num_classes to at least 2 for binary classification",
                )
            )

        # Validate backbone
        if model.vision.backbone not in VALID_BACKBONES:
            self.errors.append(
                ValidationError(
                    path="model.vision.backbone",
                    message=f"Invalid backbone: {model.vision.backbone}",
                    error_code="E002",
                    suggestion=f"Choose from: {', '.join(sorted(VALID_BACKBONES))}",
                )
            )

        # Validate vision config
        if model.vision.feature_dim <= 0:
            self.errors.append(
                ValidationError(
                    path="model.vision.feature_dim",
                    message=f"feature_dim must be positive, got {model.vision.feature_dim}",
                    error_code="E003",
                    suggestion="Set feature_dim to a positive integer (e.g., 128, 256, 512)",
                )
            )

        if not 0 <= model.vision.dropout < 1:
            self.errors.append(
                ValidationError(
                    path="model.vision.dropout",
                    message=f"dropout must be in [0, 1), got {model.vision.dropout}",
                    error_code="E005",
                    suggestion="Set dropout to a value between 0 and 1 (e.g., 0.3, 0.5)",
                )
            )

        # Validate fusion type
        if model.fusion.fusion_type not in VALID_FUSION_TYPES:
            self.errors.append(
                ValidationError(
                    path="model.fusion.fusion_type",
                    message=f"Invalid fusion_type: {model.fusion.fusion_type}",
                    error_code="E009",
                    suggestion=f"Choose from: {', '.join(sorted(VALID_FUSION_TYPES))}",
                )
            )

        # Validate tabular config
        if not model.tabular.hidden_dims:
            self.errors.append(
                ValidationError(
                    path="model.tabular.hidden_dims",
                    message="hidden_dims cannot be empty",
                    error_code="E006",
                    suggestion="Provide at least one hidden dimension (e.g., [64, 32])",
                )
            )

        if any(dim <= 0 for dim in model.tabular.hidden_dims):
            self.errors.append(
                ValidationError(
                    path="model.tabular.hidden_dims",
                    message="all dimensions must be positive",
                    error_code="E007",
                    suggestion="Ensure all hidden dimensions are positive integers",
                )
            )

        if model.tabular.output_dim <= 0:
            self.errors.append(
                ValidationError(
                    path="model.tabular.output_dim",
                    message=f"output_dim must be positive, got {model.tabular.output_dim}",
                    error_code="E008",
                    suggestion="Set output_dim to a positive integer",
                )
            )

        # Validate fusion config
        if model.fusion.hidden_dim <= 0:
            self.errors.append(
                ValidationError(
                    path="model.fusion.hidden_dim",
                    message=f"hidden_dim must be positive, got {model.fusion.hidden_dim}",
                    error_code="E010",
                    suggestion="Set hidden_dim to a positive integer",
                )
            )

        if model.model_type == "three_phase_ct_fusion":
            if model.phase_feature_dim <= 0:
                self.errors.append(
                    ValidationError(
                        path="model.phase_feature_dim",
                        message=(
                            "phase_feature_dim must be positive for three_phase_ct_fusion, "
                            f"got {model.phase_feature_dim}"
                        ),
                        error_code="E029",
                        suggestion="Set model.phase_feature_dim to a positive integer",
                    )
                )

            if model.phase_fusion_type not in VALID_THREE_PHASE_FUSION_TYPES:
                self.errors.append(
                    ValidationError(
                        path="model.phase_fusion_type",
                        message=f"Invalid phase_fusion_type: {model.phase_fusion_type}",
                        error_code="E030",
                        suggestion=(
                            "Choose from: "
                            + ", ".join(sorted(VALID_THREE_PHASE_FUSION_TYPES))
                        ),
                    )
                )

    def _validate_data_config(self, config: ExperimentConfig) -> None:
        """Validate data configuration."""
        data = config.data

        # Validate splits
        total_ratio = data.train_ratio + data.val_ratio + data.test_ratio
        if not (0.99 <= total_ratio <= 1.01):  # Allow small floating point errors
            self.errors.append(
                ValidationError(
                    path="data.train_ratio",  # Changed to match test expectation
                    message=f"data split ratios must sum to 1.0, got {total_ratio:.4f}",
                    error_code="E011",
                    suggestion="Adjust train_ratio, val_ratio, and test_ratio to sum to 1.0",
                )
            )

        if data.batch_size <= 0:
            self.errors.append(
                ValidationError(
                    path="data.batch_size",
                    message=f"batch_size must be positive, got {data.batch_size}",
                    error_code="E012",
                    suggestion="Set batch_size to a positive integer (e.g., 16, 32, 64)",
                )
            )

        if data.image_size <= 0:
            self.errors.append(
                ValidationError(
                    path="data.image_size",
                    message=f"image_size must be positive, got {data.image_size}",
                    error_code="E013",
                    suggestion="Set image_size to a positive integer (e.g., 224, 256, 512)",
                )
            )

        if data.num_workers < 0:
            self.errors.append(
                ValidationError(
                    path="data.num_workers",
                    message=f"num_workers must be non-negative, got {data.num_workers}",
                    error_code="E014",
                    suggestion="Set num_workers to 0 or a positive integer (typically 4-8)",
                )
            )

        if data.dataset_type == "three_phase_ct_tabular":
            missing_phase_keys = [
                phase_name
                for phase_name in ("arterial", "portal", "noncontrast")
                if phase_name not in data.phase_dir_columns
            ]
            if missing_phase_keys:
                self.errors.append(
                    ValidationError(
                        path="data.phase_dir_columns",
                        message=(
                            "three_phase_ct_tabular requires phase_dir_columns to define "
                            + ", ".join(missing_phase_keys)
                        ),
                        error_code="E031",
                        suggestion=(
                            "Set data.phase_dir_columns with arterial / portal / "
                            "noncontrast column names"
                        ),
                    )
                )

            if not data.patient_id_column:
                self.errors.append(
                    ValidationError(
                        path="data.patient_id_column",
                        message=(
                            "three_phase_ct_tabular requires patient_id_column to identify cases"
                        ),
                        error_code="E032",
                        suggestion="Set data.patient_id_column to the case identifier column",
                    )
                )

            if not data.clinical_feature_columns:
                self.errors.append(
                    ValidationError(
                        path="data.clinical_feature_columns",
                        message=(
                            "three_phase_ct_tabular currently requires at least one clinical "
                            "feature column"
                        ),
                        error_code="E033",
                        suggestion=(
                            "Provide one or more numeric columns in "
                            "data.clinical_feature_columns"
                        ),
                    )
                )

            if (
                data.target_shape is None
                or len(data.target_shape) != 3
                or any(int(dimension) <= 0 for dimension in data.target_shape)
            ):
                self.errors.append(
                    ValidationError(
                        path="data.target_shape",
                        message=(
                            "three_phase_ct_tabular requires target_shape as three positive "
                            f"integers, got {data.target_shape}"
                        ),
                        error_code="E034",
                        suggestion="Set data.target_shape like [16, 64, 64]",
                    )
                )

            if not data.window_preset:
                self.errors.append(
                    ValidationError(
                        path="data.window_preset",
                        message="three_phase_ct_tabular requires a non-empty window_preset",
                        error_code="E035",
                        suggestion="Set data.window_preset, for example 'liver'",
                    )
                )

    def _validate_training_config(self, config: ExperimentConfig) -> None:
        """Validate training configuration."""
        training = config.training

        if training.num_epochs <= 0:
            self.errors.append(
                ValidationError(
                    path="training.num_epochs",
                    message=f"num_epochs must be positive, got {training.num_epochs}",
                    error_code="E015",
                    suggestion="Set num_epochs to a positive integer (e.g., 50, 100, 200)",
                )
            )

        if training.gradient_clip is not None and training.gradient_clip <= 0:
            self.errors.append(
                ValidationError(
                    path="training.gradient_clip",
                    message=f"gradient_clip must be positive or None, got {training.gradient_clip}",
                    error_code="E016",
                    suggestion="Set gradient_clip to a positive value (e.g., 1.0, 5.0) or None",
                )
            )

        if not 0 <= training.label_smoothing < 1:
            self.errors.append(
                ValidationError(
                    path="training.label_smoothing",
                    message=f"label_smoothing must be in [0, 1), got {training.label_smoothing}",
                    error_code="E017",
                    suggestion="Set label_smoothing to a value between 0 and 1 (e.g., 0.1, 0.2)",
                )
            )

        # Validate progressive training
        if training.use_progressive_training:
            total_epochs = (
                training.stage1_epochs + training.stage2_epochs + training.stage3_epochs
            )
            if total_epochs != training.num_epochs:
                self.errors.append(
                    ValidationError(
                        path="training.stage1_epochs",  # Changed to match test expectation
                        message=f"training stage epochs sum ({total_epochs}) must equal num_epochs ({training.num_epochs})",
                        error_code="E020",  # Changed from E018 to E020
                        suggestion=f"Adjust stage epochs to sum to {training.num_epochs}",
                    )
                )

        # Validate optimizer
        if training.optimizer.learning_rate <= 0:
            self.errors.append(
                ValidationError(
                    path="training.optimizer.learning_rate",
                    message=f"learning_rate must be positive, got {training.optimizer.learning_rate}",
                    error_code="E018",  # Changed from E019 to E018
                    suggestion="Set learning_rate to a positive value (e.g., 0.001, 0.0001)",
                )
            )

        # Validate attention supervision
        if (
            training.use_attention_supervision
            and hasattr(training, "attention_loss_weight")
            and not 0 <= training.attention_loss_weight <= 1
        ):
            self.errors.append(
                ValidationError(
                    path="training.attention_loss_weight",
                    message=f"attention_loss_weight must be in [0, 1], got {training.attention_loss_weight}",
                    error_code="E019",  # Changed from E020 to E019
                    suggestion="Set attention_loss_weight to a value between 0 and 1 (e.g., 0.1, 0.5)",
                )
            )

    def _validate_logging_config(self, config: ExperimentConfig) -> None:
        """Validate logging configuration."""
        logging = config.logging

        if logging.log_every_n_steps <= 0:
            self.errors.append(
                ValidationError(
                    path="logging.log_every_n_steps",
                    message=f"log_every_n_steps must be positive, got {logging.log_every_n_steps}",
                    error_code="E021",
                    suggestion="Set log_every_n_steps to a positive integer (e.g., 10, 50, 100)",
                )
            )

        # Validate wandb configuration
        if logging.use_wandb and not logging.wandb_project:
            self.errors.append(
                ValidationError(
                    path="logging.wandb_project",
                    message="wandb_project must be specified when use_wandb=True",
                    error_code="E027",  # Changed from E022 to E027
                    suggestion="Set logging.wandb_project to your W&B project name or disable use_wandb",
                )
            )

    def _validate_cross_dependencies(self, config: ExperimentConfig) -> None:
        """Validate cross-dependencies between config sections."""
        # Attention supervision requires attention to be enabled
        if (
            config.training.use_attention_supervision
            and not config.model.vision.enable_attention_supervision
        ):
            self.errors.append(
                ValidationError(
                    path="training.use_attention_supervision",
                    message="use_attention_supervision=True requires model.vision.enable_attention_supervision=True",
                    error_code="E028",
                    suggestion="Set model.vision.enable_attention_supervision=True or disable training.use_attention_supervision",
                )
            )

        if (
            config.data.dataset_type == "three_phase_ct_tabular"
            and config.model.model_type != "three_phase_ct_fusion"
        ):
            self.errors.append(
                ValidationError(
                    path="model.model_type",
                    message=(
                        "three_phase_ct_tabular must be paired with "
                        "model.model_type=three_phase_ct_fusion"
                    ),
                    error_code="E036",
                    suggestion="Set model.model_type to three_phase_ct_fusion",
                )
            )

        if (
            config.model.model_type == "three_phase_ct_fusion"
            and config.data.dataset_type != "three_phase_ct_tabular"
        ):
            self.errors.append(
                ValidationError(
                    path="data.dataset_type",
                    message=(
                        "three_phase_ct_fusion requires "
                        "data.dataset_type=three_phase_ct_tabular"
                    ),
                    error_code="E037",
                    suggestion="Set data.dataset_type to three_phase_ct_tabular",
                )
            )


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
        ...         print(f"❌ [{error.error_code}] {error.path}: {error.message}")
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
            logger.error(f"  {i}. [{error.error_code}] {error.path}: {error.message}")
            if error.suggestion:
                logger.error(f"      💡 {error.suggestion}")
        sys.exit(1)
    else:
        logger.info("✅ Configuration validation passed")
