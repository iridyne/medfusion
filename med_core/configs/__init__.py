"""Configuration module for Med-Core framework."""

from med_core.configs.base_config import (
    BaseConfig,
    ClinicalPreprocessingConfig,
    DataConfig,
    ExperimentConfig,
    ExplainabilityConfig,
    FusionConfig,
    LoggingConfig,
    ModelConfig,
    OptimizerConfig,
    PhaseEncoderConfig,
    PhaseFusionConfig,
    SchedulerConfig,
    TabularConfig,
    TrainingConfig,
    VisionConfig,
)
from med_core.configs.config_loader import (
    create_default_config,
    load_config,
    save_config,
)
from med_core.configs.doctor import (
    ConfigDoctor,
    ConfigDoctorReport,
    DoctorIssue,
    analyze_config,
)
from med_core.configs.multiview_config import (
    MultiViewDataConfig,
    MultiViewExperimentConfig,
    MultiViewModelConfig,
    MultiViewVisionConfig,
    create_ct_multiview_config,
    create_temporal_multiview_config,
)
from med_core.configs.validation import (
    ConfigValidator,
    validate_config,
    validate_config_or_exit,
)

__all__ = [
    # Base configs
    "BaseConfig",
    "ClinicalPreprocessingConfig",
    "DataConfig",
    "ExperimentConfig",
    "ExplainabilityConfig",
    "FusionConfig",
    "LoggingConfig",
    "ModelConfig",
    "OptimizerConfig",
    "PhaseEncoderConfig",
    "PhaseFusionConfig",
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
    "create_default_config",
    "load_config",
    "save_config",
    # Validation
    "ConfigValidator",
    "ConfigDoctor",
    "ConfigDoctorReport",
    "DoctorIssue",
    "analyze_config",
    "validate_config",
    "validate_config_or_exit",
]
