"""
Configuration loader utilities.

Supports loading from YAML files and merging with command-line overrides.
"""

from pathlib import Path
from typing import Any, TypeVar

import yaml

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

T = TypeVar("T", bound=BaseConfig)


def _merge_dicts(base: dict, override: dict) -> dict:
    """Recursively merge override dict into base dict."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _merge_dicts(result[key], value)
        else:
            result[key] = value
    return result


def _dict_to_config(data: dict, config_class: type[T]) -> T:
    """Convert nested dictionary to config dataclass."""
    if not isinstance(data, dict):
        raise TypeError(f"Expected dict, got {type(data).__name__}")

    if config_class == ExperimentConfig:
        if "data" in data:
            if not isinstance(data["data"], dict):
                raise TypeError(
                    f"Expected dict for 'data', got {type(data['data']).__name__}",
                )
            data["data"] = DataConfig(**data["data"])

        if "model" in data:
            if not isinstance(data["model"], dict):
                raise TypeError(
                    f"Expected dict for 'model', got {type(data['model']).__name__}",
                )
            model_data = data["model"]

            if "vision" in model_data:
                if not isinstance(model_data["vision"], dict):
                    raise TypeError(
                        f"Expected dict for 'model.vision', got {type(model_data['vision']).__name__}",
                    )
                model_data["vision"] = VisionConfig(**model_data["vision"])

            if "tabular" in model_data:
                if not isinstance(model_data["tabular"], dict):
                    raise TypeError(
                        f"Expected dict for 'model.tabular', got {type(model_data['tabular']).__name__}",
                    )
                model_data["tabular"] = TabularConfig(**model_data["tabular"])

            if "fusion" in model_data:
                if not isinstance(model_data["fusion"], dict):
                    raise TypeError(
                        f"Expected dict for 'model.fusion', got {type(model_data['fusion']).__name__}",
                    )
                model_data["fusion"] = FusionConfig(**model_data["fusion"])

            data["model"] = ModelConfig(**model_data)

        if "training" in data:
            if not isinstance(data["training"], dict):
                raise TypeError(
                    f"Expected dict for 'training', got {type(data['training']).__name__}",
                )
            training_data = data["training"]

            if "optimizer" in training_data:
                if not isinstance(training_data["optimizer"], dict):
                    raise TypeError(
                        f"Expected dict for 'training.optimizer', got {type(training_data['optimizer']).__name__}",
                    )
                training_data["optimizer"] = OptimizerConfig(
                    **training_data["optimizer"],
                )

            if "scheduler" in training_data:
                if not isinstance(training_data["scheduler"], dict):
                    raise TypeError(
                        f"Expected dict for 'training.scheduler', got {type(training_data['scheduler']).__name__}",
                    )
                training_data["scheduler"] = SchedulerConfig(
                    **training_data["scheduler"],
                )

            data["training"] = TrainingConfig(**training_data)

        if "logging" in data:
            if not isinstance(data["logging"], dict):
                raise TypeError(
                    f"Expected dict for 'logging', got {type(data['logging']).__name__}",
                )
            data["logging"] = LoggingConfig(**data["logging"])

    return config_class(**data)


def load_config(
    config_path: str | Path,
    config_class: type[T] = ExperimentConfig,
    overrides: dict[str, Any] | None = None,
) -> T:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to YAML configuration file
        config_class: Configuration class to instantiate
        overrides: Optional dictionary of values to override

    Returns:
        Instantiated configuration object

    Example:
        >>> config = load_config("configs/dermoscopy.yaml")
        >>> config = load_config("configs/base.yaml", overrides={"training.num_epochs": 50})
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if data is None:
        data = {}

    # Apply overrides
    if overrides:
        data = _merge_dicts(data, overrides)

    return _dict_to_config(data, config_class)


def save_config(config: BaseConfig, output_path: str | Path) -> None:
    """
    Save configuration to YAML file.

    Args:
        config: Configuration object to save
        output_path: Path to output YAML file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    data = config.to_dict()

    with open(output_path, "w", encoding="utf-8") as f:
        yaml.dump(
            data, f, default_flow_style=False, allow_unicode=True, sort_keys=False,
        )


def create_default_config() -> ExperimentConfig:
    """Create a default experiment configuration."""
    return ExperimentConfig()


def list_config_templates() -> list[str]:
    """List available configuration templates."""
    templates_dir = Path(__file__).parent.parent.parent / "configs"
    if not templates_dir.exists():
        return []
    return [f.stem for f in templates_dir.glob("*.yaml")]
