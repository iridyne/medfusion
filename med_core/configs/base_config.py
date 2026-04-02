"""
Base configuration classes using dataclass pattern.

Supports loading from YAML files for easy project switching.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import torch

from med_core.output_layout import RunOutputLayout, resolve_oss_path


@dataclass
class BaseConfig:
    """Base configuration with common utilities."""

    def to_dict(self) -> dict[str, Any]:
        """Convert config to dictionary."""
        result = {}
        for k, v in self.__dict__.items():
            if isinstance(v, BaseConfig):
                result[k] = v.to_dict()
            elif isinstance(v, Path):
                result[k] = str(v)
            else:
                result[k] = v
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BaseConfig":
        """Create config from dictionary."""
        return cls(**data)


@dataclass
class ClinicalPreprocessingConfig(BaseConfig):
    """Clinical preprocessing options for tabular case features."""

    normalize: bool = False
    strategy: Literal["none", "zero_with_mask"] = "none"


@dataclass
class DataConfig(BaseConfig):
    """Data-related configuration."""

    dataset_type: Literal["image_tabular", "three_phase_ct_tabular"] = "image_tabular"

    # Paths
    data_root: str = "data"
    csv_path: str = "data/dataset.csv"
    image_dir: str = "data/images"

    # Data split
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    random_seed: int = 42

    # Image settings
    image_size: int = 224
    image_channels: int = 3
    image_view: str = "default"  # e.g., "coronal", "axial", "sagittal"

    # Tabular features
    categorical_features: list[str] = field(default_factory=list)
    numerical_features: list[str] = field(default_factory=list)
    target_column: str = "label"
    survival_time_column: str | None = None
    survival_event_column: str | None = None
    patient_id_column: str | None = None
    image_path_column: str = "image_path"
    clinical_feature_columns: list[str] = field(default_factory=list)
    clinical_preprocessing: ClinicalPreprocessingConfig = field(
        default_factory=ClinicalPreprocessingConfig
    )
    phase_dir_columns: dict[str, str] = field(default_factory=dict)
    target_shape: list[int] | None = None
    window_preset: str = "soft_tissue"

    # Optional offline pathology embeddings (e.g., HIPT exports)
    hipt_embeddings_dir: str | None = None

    # Dataloader settings
    batch_size: int = 16
    num_workers: int = 4
    pin_memory: bool = True

    # Augmentation
    use_augmentation: bool = True
    augmentation_strength: Literal["light", "medium", "heavy"] = "medium"

    def __post_init__(self) -> None:
        if isinstance(self.clinical_preprocessing, dict):
            self.clinical_preprocessing = ClinicalPreprocessingConfig(
                **self.clinical_preprocessing
            )

    @property
    def resolved_data_root(self) -> Path:
        """Resolve data_root against the OSS repository root."""
        return resolve_oss_path(self.data_root)

    @property
    def resolved_csv_path(self) -> Path:
        """Resolve csv_path against the OSS repository root."""
        return resolve_oss_path(self.csv_path)

    @property
    def resolved_image_dir(self) -> Path:
        """Resolve image_dir against the OSS repository root."""
        return resolve_oss_path(self.image_dir)

    @property
    def resolved_hipt_embeddings_dir(self) -> Path | None:
        """Resolve optional HIPT embeddings directory against the OSS repository root."""
        if self.hipt_embeddings_dir is None:
            return None
        return resolve_oss_path(self.hipt_embeddings_dir)


@dataclass
class VisionConfig(BaseConfig):
    """Vision backbone configuration."""

    backbone: Literal[
        "resnet18",
        "resnet34",
        "resnet50",
        "resnet101",
        "mobilenetv2",
        "mobilenetv3_small",
        "mobilenetv3_large",
        "efficientnet_b0",
        "efficientnet_b1",
        "efficientnet_b2",
        "efficientnet_v2_s",
        "efficientnet_v2_m",
        "efficientnet_v2_l",
        "convnext_tiny",
        "convnext_small",
        "convnext_base",
        "convnext_large",
        "maxvit_t",
        "regnet_y_400mf",
        "regnet_y_800mf",
        "regnet_y_1_6gf",
        "regnet_y_3_2gf",
        "regnet_y_8gf",
        "regnet_y_16gf",
        "regnet_y_32gf",
        "vit_b_16",
        "vit_b_32",
        "swin_t",
        "swin_s",
    ] = "resnet18"
    pretrained: bool = True
    freeze_backbone: bool = True
    freeze_strategy: Literal["full", "partial", "progressive", "none"] = "progressive"
    unfreeze_last_n_layers: int = 2

    # Feature dimensions
    feature_dim: int = 128
    dropout: float = 0.3

    # Attention
    use_attention: bool = True
    attention_type: Literal["cbam", "se", "eca", "none"] = "cbam"

    # Attention supervision (only works with CBAM)
    enable_attention_supervision: bool = False


@dataclass
class TabularConfig(BaseConfig):
    """Tabular stream configuration."""

    hidden_dims: list[int] = field(default_factory=lambda: [64, 64])
    output_dim: int = 32
    dropout: float = 0.2
    use_batch_norm: bool = True
    activation: Literal["relu", "gelu", "silu"] = "relu"


@dataclass
class FusionConfig(BaseConfig):
    """Fusion module configuration."""

    fusion_type: Literal[
        "concatenate", "gated", "attention", "cross_attention", "bilinear",
    ] = "gated"
    hidden_dim: int = 96
    dropout: float = 0.4
    num_heads: int = 4  # For attention-based fusion

    # Modality weights (for weighted fusion)
    initial_image_weight: float = 0.3
    initial_tabular_weight: float = 0.7
    learnable_weights: bool = True


@dataclass
class PhaseEncoderConfig(BaseConfig):
    """Three-phase CT encoder structure config."""

    base_channels: int = 16
    num_blocks: int = 3
    dropout: float = 0.1
    norm: Literal["batch", "instance", "group"] = "batch"


@dataclass
class PhaseFusionConfig(BaseConfig):
    """Three-phase feature fusion config."""

    mode: Literal["concatenate", "mean", "gated"] = "concatenate"
    hidden_dim: int = 64


@dataclass
class ModelConfig(BaseConfig):
    """Complete model configuration."""

    model_type: Literal["multimodal_fusion", "three_phase_ct_fusion"] = (
        "multimodal_fusion"
    )
    num_classes: int = 2

    # Sub-configs
    vision: VisionConfig = field(default_factory=VisionConfig)
    tabular: TabularConfig = field(default_factory=TabularConfig)
    fusion: FusionConfig = field(default_factory=FusionConfig)

    # Three-phase CT fusion settings
    phase_feature_dim: int = 64
    share_phase_encoder: bool = False
    phase_fusion_type: Literal["concatenate", "mean", "gated"] = "concatenate"
    phase_encoder: PhaseEncoderConfig = field(default_factory=PhaseEncoderConfig)
    phase_fusion: PhaseFusionConfig = field(default_factory=PhaseFusionConfig)
    use_risk_head: bool = False

    # Pathology encoder selection
    pathology_encoder: Literal["patch_mil", "hipt"] = "patch_mil"
    hipt_embedding_dim: int = 192

    # Multi-task auxiliary heads
    use_auxiliary_heads: bool = True

    def __post_init__(self) -> None:
        if isinstance(self.vision, dict):
            self.vision = VisionConfig(**self.vision)
        if isinstance(self.tabular, dict):
            self.tabular = TabularConfig(**self.tabular)
        if isinstance(self.fusion, dict):
            self.fusion = FusionConfig(**self.fusion)
        if isinstance(self.phase_encoder, dict):
            self.phase_encoder = PhaseEncoderConfig(**self.phase_encoder)
        if isinstance(self.phase_fusion, dict):
            self.phase_fusion = PhaseFusionConfig(**self.phase_fusion)

        default_phase_fusion_mode = PhaseFusionConfig().mode
        default_phase_fusion_type = "concatenate"
        if (
            self.phase_fusion.mode == default_phase_fusion_mode
            and self.phase_fusion_type != default_phase_fusion_type
        ):
            self.phase_fusion.mode = self.phase_fusion_type
        elif (
            self.phase_fusion.mode != default_phase_fusion_mode
            and self.phase_fusion_type == default_phase_fusion_type
        ):
            self.phase_fusion_type = self.phase_fusion.mode
        elif self.phase_fusion.mode != self.phase_fusion_type:
            self.phase_fusion_type = self.phase_fusion.mode


@dataclass
class OptimizerConfig(BaseConfig):
    """Optimizer configuration."""

    optimizer: Literal["adam", "adamw", "sgd"] = "adamw"
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    momentum: float = 0.9  # For SGD

    # Differential learning rates
    use_differential_lr: bool = True
    lr_backbone: float = 1e-5
    lr_tabular: float = 1e-4
    lr_fusion: float = 5e-5
    lr_classifier: float = 1e-4


@dataclass
class SchedulerConfig(BaseConfig):
    """Learning rate scheduler configuration."""

    scheduler: Literal["cosine", "step", "plateau", "onecycle", "none"] = "cosine"
    warmup_epochs: int = 5
    min_lr: float = 1e-7

    # For StepLR
    step_size: int = 10
    gamma: float = 0.1

    # For ReduceLROnPlateau
    patience: int = 5
    factor: float = 0.5


@dataclass
class TrainingConfig(BaseConfig):
    """Training-related configuration."""

    # Training params
    num_epochs: int = 100
    gradient_clip: float | None = 1.0
    accumulation_steps: int = 1
    mixed_precision: bool = True

    # Loss settings
    label_smoothing: float = 0.1
    class_weights: list[float] | None = None

    # Attention supervision (requires VisionConfig.enable_attention_supervision=True)
    use_attention_supervision: bool = False
    attention_loss_weight: float = 0.1
    attention_supervision_method: Literal["mask", "cam", "none"] = "none"
    # For mask-based supervision: expects masks in dataset
    # For CAM-based supervision: automatically generates CAM from features

    # Progressive training stages
    use_progressive_training: bool = True
    stage1_epochs: int = 15  # Train image stream
    stage2_epochs: int = 20  # Full model fine-tuning
    stage3_epochs: int = 15  # Fusion layer only

    # Early stopping
    early_stopping: bool = True
    patience: int = 20
    min_delta: float = 0.001
    monitor: str = "accuracy"
    mode: Literal["min", "max"] = "max"

    # Checkpointing
    save_top_k: int = 3
    save_last: bool = True

    # Sub-configs
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)


@dataclass
class LoggingConfig(BaseConfig):
    """Logging and experiment tracking configuration."""

    output_dir: str = "outputs"
    experiment_name: str = "experiment"

    # Logging backends
    use_tensorboard: bool = True
    use_wandb: bool = False
    wandb_project: str = "med-core"
    wandb_entity: str | None = None

    # Logging frequency
    log_every_n_steps: int = 10
    val_check_interval: float = 1.0  # Check validation every epoch

    # Visualization
    save_visualizations: bool = True
    gradcam_samples: int = 10


@dataclass
class ExplainabilityConfig(BaseConfig):
    """Structured result artifact export toggles."""

    export_phase_importance: bool = False
    export_case_explanations: bool = False
    heatmap_ready: bool = False
    build_results_split: Literal["train", "val", "test", "all"] = "test"
    min_global_importance_samples: int = 8


@dataclass
class ExperimentConfig(BaseConfig):
    """Complete experiment configuration combining all sub-configs."""

    # Experiment metadata
    project_name: str = "medical-multimodal"
    experiment_name: str = "default"
    description: str = ""
    tags: list[str] = field(default_factory=list)

    # Random seed
    seed: int = 42
    deterministic: bool = True

    # Device
    device: str = "auto"  # "auto", "cuda", "cpu", "mps"

    # Sub-configs
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    explainability: ExplainabilityConfig = field(default_factory=ExplainabilityConfig)

    def __post_init__(self) -> None:
        """Post-initialization setup."""
        if isinstance(self.data, dict):
            self.data = DataConfig(**self.data)
        if isinstance(self.model, dict):
            self.model = ModelConfig(**self.model)
        if isinstance(self.training, dict):
            self.training = TrainingConfig(**self.training)
        if isinstance(self.logging, dict):
            self.logging = LoggingConfig(**self.logging)
        if isinstance(self.explainability, dict):
            self.explainability = ExplainabilityConfig(**self.explainability)

        # Create the run root and its structured subdirectories.
        self.output_layout.ensure_exists()

        # Set device
        if self.device == "auto":
            self.device = self._detect_device()

    @staticmethod
    def _detect_device() -> str:
        """Detect available device."""
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    @property
    def device_obj(self) -> torch.device:
        """Get torch device object."""
        return torch.device(self.device)

    @property
    def output_layout(self) -> RunOutputLayout:
        """Get the canonical output layout for this run."""
        return RunOutputLayout(self.logging.output_dir)

    @property
    def checkpoint_dir(self) -> Path:
        """Get checkpoint directory path."""
        return self.output_layout.ensure_exists().checkpoints_dir

    @property
    def log_dir(self) -> Path:
        """Get log directory path."""
        return self.output_layout.ensure_exists().logs_dir

    @property
    def reports_dir(self) -> Path:
        """Get report directory path."""
        return self.output_layout.ensure_exists().reports_dir

    @property
    def metrics_dir(self) -> Path:
        """Get metrics directory path."""
        return self.output_layout.ensure_exists().metrics_dir

    @property
    def artifacts_dir(self) -> Path:
        """Get artifact directory path."""
        return self.output_layout.ensure_exists().artifacts_dir

    @property
    def history_path(self) -> Path:
        """Get training history artifact path."""
        return self.output_layout.ensure_exists().history_path

    @property
    def results_dir(self) -> Path:
        """Backward-compatible alias for the report directory."""
        return self.reports_dir
