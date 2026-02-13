"""Configuration module for Med-Core framework."""

from med_core.configs.base_config import (
    BaseConfig,
    DataConfig,
    ExperimentConfig,
    FusionConfig,
    LoggingConfig,
    ModelConfig,
    OptimizerConfig,
    SchedulerConfig,
    TabularConfig,
    TrainingConfig,
    VisionConfig,
)
from med_core.configs.config_loader import load_config, save_config
from med_core.configs.multiview_config import (
    MultiViewDataConfig,
    MultiViewExperimentConfig,
    MultiViewModelConfig,
    MultiViewVisionConfig,
    create_ct_multiview_config,
    create_temporal_multiview_config,
)

__all__ = [
    # Base configs
    "BaseConfig",
    "DataConfig",
    "ExperimentConfig",
    "FusionConfig",
    "LoggingConfig",
    "ModelConfig",
    "OptimizerConfig",
    "SchedulerConfig",
    "TabularConfig",
    "TrainingConfig",
    "VisionConfig",
    # Multi-view configs
    "MultiViewDataConfig",
    "MultiViewVisionConfig",
    "MultiViewModelConfig",
    "MultiViewExperimentConfig",
    "create_ct_multiview_config",
    "create_temporal_multiview_config",
    # Config utilities
    "load_config",
    "save_config",
]
