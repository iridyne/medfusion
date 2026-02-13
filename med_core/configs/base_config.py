"""
Base configuration classes using dataclass pattern.

Supports loading from YAML files for easy project switching.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import torch


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
class DataConfig(BaseConfig):
    """Data-related configuration."""

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
    patient_id_column: str | None = None
    image_path_column: str = "image_path"

    # Dataloader settings
    batch_size: int = 16
    num_workers: int = 4
    pin_memory: bool = True

    # Augmentation
    use_augmentation: bool = True
    augmentation_strength: Literal["light", "medium", "heavy"] = "medium"


@dataclass
class VisionConfig(BaseConfig):
    """Vision backbone configuration."""

    backbone: Literal[
        "resnet18", "resnet34", "resnet50", "resnet101",
        "mobilenetv2", "mobilenetv3_small", "mobilenetv3_large",
        "efficientnet_b0", "efficientnet_b1", "efficientnet_b2",
        "vit_b_16", "vit_b_32", "swin_t", "swin_s"
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
        "concatenate", "gated", "attention", "cross_attention", "bilinear"
    ] = "gated"
    hidden_dim: int = 96
    dropout: float = 0.4
    num_heads: int = 4  # For attention-based fusion

    # Modality weights (for weighted fusion)
    initial_image_weight: float = 0.3
    initial_tabular_weight: float = 0.7
    learnable_weights: bool = True


@dataclass
class ModelConfig(BaseConfig):
    """Complete model configuration."""

    num_classes: int = 2

    # Sub-configs
    vision: VisionConfig = field(default_factory=VisionConfig)
    tabular: TabularConfig = field(default_factory=TabularConfig)
    fusion: FusionConfig = field(default_factory=FusionConfig)

    # Multi-task auxiliary heads
    use_auxiliary_heads: bool = True


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

    scheduler: Literal[
        "cosine", "step", "plateau", "onecycle", "none"
    ] = "cosine"
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
    monitor: str = "val_auc"
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

    def __post_init__(self):
        """Post-initialization setup."""
        # Create output directories
        output_path = Path(self.logging.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Set device
        if self.device == "auto":
            self.device = self._detect_device()

    @staticmethod
    def _detect_device() -> str:
        """Detect available device."""
        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"

    @property
    def device_obj(self) -> torch.device:
        """Get torch device object."""
        return torch.device(self.device)

    @property
    def checkpoint_dir(self) -> Path:
        """Get checkpoint directory path."""
        path = Path(self.logging.output_dir) / "checkpoints"
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def log_dir(self) -> Path:
        """Get log directory path."""
        path = Path(self.logging.output_dir) / "logs"
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def results_dir(self) -> Path:
        """Get results directory path."""
        path = Path(self.logging.output_dir) / "results"
        path.mkdir(parents=True, exist_ok=True)
        return path
