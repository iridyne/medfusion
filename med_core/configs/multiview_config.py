"""
Multi-view configuration extensions.

Extends base configuration classes to support multi-view imaging scenarios.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from med_core.configs.base_config import (
    BaseConfig,
    DataConfig,
    FusionConfig,
    LoggingConfig,
    TabularConfig,
    TrainingConfig,
    VisionConfig,
)


@dataclass
class MultiViewDataConfig(DataConfig):
    """
    Data configuration for multi-view datasets.

    Extends DataConfig to support multiple images per patient.
    """

    # Multi-view settings
    enable_multiview: bool = False
    view_names: list[str] = field(
        default_factory=list,
    )  # e.g., ["axial", "coronal", "sagittal"]

    # View path columns in CSV
    # Option 1: Separate columns for each view
    view_path_columns: dict[str, str] = field(default_factory=dict)
    # Example: {"axial": "axial_path", "coronal": "coronal_path", "sagittal": "sagittal_path"}

    # Option 2: Single column with JSON/dict format
    multiview_path_column: str | None = None

    # Missing view handling
    missing_view_strategy: Literal["skip", "zero", "duplicate"] = "zero"
    # - skip: Skip samples with missing views
    # - zero: Fill with zero tensor
    # - duplicate: Duplicate an available view

    # View-specific augmentation
    use_view_specific_augmentation: bool = False
    view_augmentation_configs: dict[str, dict] = field(default_factory=dict)

    # Validation
    require_all_views: bool = False  # If True, skip samples missing any view
    min_views_required: int = 1  # Minimum number of views required per sample


@dataclass
class MultiViewVisionConfig(VisionConfig):
    """
    Vision backbone configuration for multi-view processing.

    Extends VisionConfig to support view aggregation strategies.
    """

    # Multi-view settings
    enable_multiview: bool = False

    # View aggregation
    aggregator_type: Literal[
        "max", "mean", "attention", "cross_view_attention", "learned_weight",
    ] = "attention"

    # Aggregator-specific settings
    aggregator_num_heads: int = 4  # For attention-based aggregators
    aggregator_dropout: float = 0.1

    # Weight sharing
    share_backbone_weights: bool = True  # If False, use separate backbone per view

    # View-specific settings
    view_specific_preprocessing: dict[str, dict] = field(default_factory=dict)
    # Example: {"axial": {"normalize": True}, "coronal": {"normalize": False}}

    # Progressive view training
    use_progressive_view_training: bool = False
    initial_views: list[str] = field(default_factory=list)  # Start with subset of views
    add_views_every_n_epochs: int = 10


@dataclass
class MultiViewModelConfig(BaseConfig):
    """
    Complete model configuration for multi-view multimodal learning.

    Combines multi-view vision, tabular, and fusion configurations.
    """

    num_classes: int = 2

    # Sub-configs
    vision: MultiViewVisionConfig = field(default_factory=MultiViewVisionConfig)
    tabular: TabularConfig = field(default_factory=TabularConfig)
    fusion: FusionConfig = field(default_factory=FusionConfig)

    # Multi-task auxiliary heads
    use_auxiliary_heads: bool = True
    use_view_specific_heads: bool = False  # Per-view classification heads


@dataclass
class MultiViewExperimentConfig(BaseConfig):
    """
    Complete experiment configuration for multi-view experiments.

    Extends ExperimentConfig with multi-view specific settings.
    """

    # Experiment metadata
    project_name: str = "medical-multimodal-multiview"
    experiment_name: str = "default_multiview"
    description: str = ""
    tags: list[str] = field(default_factory=list)

    # Random seed
    seed: int = 42
    deterministic: bool = True

    # Device
    device: str = "auto"

    # Sub-configs
    data: MultiViewDataConfig = field(default_factory=MultiViewDataConfig)
    model: MultiViewModelConfig = field(default_factory=MultiViewModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    def __post_init__(self) -> None:
        """Post-initialization validation and setup."""
        from pathlib import Path

        # Validate multi-view consistency
        if self.data.enable_multiview:
            if not self.data.view_names:
                raise ValueError(
                    "view_names must be specified when enable_multiview=True",
                )

            if not self.data.view_path_columns and not self.data.multiview_path_column:
                raise ValueError(
                    "Either view_path_columns or multiview_path_column must be specified "
                    "when enable_multiview=True",
                )

            # Enable multi-view in model config
            self.model.vision.enable_multiview = True

        # Create output directories
        output_path = Path(self.logging.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Set device
        if self.device == "auto":
            self.device = self._detect_device()

    @staticmethod
    def _detect_device() -> str:
        """Detect available device."""
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    @property
    def device_obj(self) -> Any:
        """Get torch device object."""
        import torch

        return torch.device(self.device)

    @property
    def checkpoint_dir(self) -> Path:
        """Get checkpoint directory path."""
        from pathlib import Path

        path = Path(self.logging.output_dir) / "checkpoints"
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def log_dir(self) -> Path:
        """Get log directory path."""
        from pathlib import Path

        path = Path(self.logging.output_dir) / "logs"
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def results_dir(self) -> Path:
        """Get results directory path."""
        from pathlib import Path

        path = Path(self.logging.output_dir) / "results"
        path.mkdir(parents=True, exist_ok=True)
        return path


# Example configuration presets


def create_ct_multiview_config(
    view_names: list[str] = None,
    aggregator_type: str = "attention",
    backbone: str = "resnet18",
) -> MultiViewExperimentConfig:
    """
    Create a preset configuration for multi-view CT imaging.

    Args:
        view_names: List of view names (default: ["axial", "coronal", "sagittal"])
        aggregator_type: View aggregation strategy
        backbone: Vision backbone name

    Returns:
        Configured MultiViewExperimentConfig

    Example:
        >>> config = create_ct_multiview_config(
        ...     view_names=["axial", "coronal", "sagittal"],
        ...     aggregator_type="attention",
        ...     backbone="resnet18",
        ... )
    """
    if view_names is None:
        view_names = ["axial", "coronal", "sagittal"]

    config = MultiViewExperimentConfig(
        project_name="ct-multiview",
        experiment_name=f"{backbone}_{aggregator_type}",
        tags=["ct", "multiview", aggregator_type],
    )

    # Data config
    config.data.enable_multiview = True
    config.data.view_names = view_names
    config.data.view_path_columns = {view: f"{view}_path" for view in view_names}
    config.data.missing_view_strategy = "zero"
    config.data.require_all_views = False

    # Model config
    config.model.vision.enable_multiview = True
    config.model.vision.backbone = backbone
    config.model.vision.aggregator_type = aggregator_type
    config.model.vision.share_backbone_weights = True

    return config


def create_temporal_multiview_config(
    num_timepoints: int = 2,
    aggregator_type: str = "attention",
    backbone: str = "resnet18",
) -> MultiViewExperimentConfig:
    """
    Create a preset configuration for temporal multi-view imaging.

    Useful for pre/post treatment comparison or disease progression tracking.

    Args:
        num_timepoints: Number of time points
        aggregator_type: View aggregation strategy
        backbone: Vision backbone name

    Returns:
        Configured MultiViewExperimentConfig

    Example:
        >>> config = create_temporal_multiview_config(
        ...     num_timepoints=2,  # Pre and post treatment
        ...     aggregator_type="attention",
        ... )
    """
    view_names = [f"t{i}" for i in range(num_timepoints)]

    config = MultiViewExperimentConfig(
        project_name="temporal-multiview",
        experiment_name=f"{backbone}_{aggregator_type}_t{num_timepoints}",
        tags=["temporal", "multiview", aggregator_type],
    )

    # Data config
    config.data.enable_multiview = True
    config.data.view_names = view_names
    config.data.view_path_columns = {view: f"image_path_{view}" for view in view_names}
    config.data.missing_view_strategy = "skip"  # Skip incomplete temporal sequences
    config.data.require_all_views = True

    # Model config
    config.model.vision.enable_multiview = True
    config.model.vision.backbone = backbone
    config.model.vision.aggregator_type = aggregator_type
    config.model.vision.share_backbone_weights = True

    return config
