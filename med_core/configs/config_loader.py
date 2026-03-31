"""
Configuration loader utilities.

Supports loading from YAML files and merging with command-line overrides.
"""

from pathlib import Path
from typing import Any, TypeVar

import yaml

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

T = TypeVar("T", bound=BaseConfig)


class UnsupportedConfigSchemaError(ValueError):
    """Raised when a config file uses a different schema than the requested loader."""


def _merge_dicts(base: dict, override: dict) -> dict:
    """Recursively merge override dict into base dict."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _merge_dicts(result[key], value)
        else:
            result[key] = value
    return result


def _looks_like_builder_model_config(model_data: dict[str, Any]) -> bool:
    """Detect builder-style model configs that cannot be loaded as ExperimentConfig."""
    if "modalities" in model_data or "head" in model_data:
        return True

    fusion_config = model_data.get("fusion")
    if isinstance(fusion_config, dict) and "strategy" in fusion_config:
        return True

    return False


def _raise_for_unsupported_schema(
    data: dict[str, Any],
    config_path: Path,
    config_class: type[T],
) -> None:
    """Fail fast with a clear error when the YAML uses a different config schema."""
    if config_class is not ExperimentConfig:
        return

    model_data = data.get("model")
    if not isinstance(model_data, dict):
        return

    if not _looks_like_builder_model_config(model_data):
        return

    raise UnsupportedConfigSchemaError(
        "检测到 builder 风格配置，不能直接作为 CLI/Web 训练主链的 ExperimentConfig 读取。"
        f"\n  config: {config_path}"
        "\n  当前文件更像 `configs/builder/*` 结构，请使用 "
        "`med_core.models.build_model_from_config()` 或 `MultiModalModelBuilder`。"
        "\n  如果你想直接运行 `medfusion validate-config / train / build-results`，"
        "请改用 `configs/starter/`、`configs/public_datasets/` 或 `configs/testing/` 下的 train schema。"
    )


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
            data_config = data["data"]
            if "clinical_preprocessing" in data_config:
                if not isinstance(data_config["clinical_preprocessing"], dict):
                    raise TypeError(
                        "Expected dict for 'data.clinical_preprocessing', "
                        f"got {type(data_config['clinical_preprocessing']).__name__}",
                    )
                data_config["clinical_preprocessing"] = ClinicalPreprocessingConfig(
                    **data_config["clinical_preprocessing"]
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

            if "phase_encoder" in model_data:
                if not isinstance(model_data["phase_encoder"], dict):
                    raise TypeError(
                        "Expected dict for 'model.phase_encoder', got "
                        f"{type(model_data['phase_encoder']).__name__}",
                    )
                model_data["phase_encoder"] = PhaseEncoderConfig(
                    **model_data["phase_encoder"]
                )

            if "phase_fusion" in model_data:
                if not isinstance(model_data["phase_fusion"], dict):
                    raise TypeError(
                        "Expected dict for 'model.phase_fusion', got "
                        f"{type(model_data['phase_fusion']).__name__}",
                    )
                model_data["phase_fusion"] = PhaseFusionConfig(
                    **model_data["phase_fusion"]
                )

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

        if "explainability" in data:
            if not isinstance(data["explainability"], dict):
                raise TypeError(
                    "Expected dict for 'explainability', "
                    f"got {type(data['explainability']).__name__}",
                )
            data["explainability"] = ExplainabilityConfig(**data["explainability"])

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
        >>> config = load_config("configs/starter/quickstart.yaml")
        >>> config = load_config("configs/starter/default.yaml", overrides={"training": {"num_epochs": 50}})
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

    _raise_for_unsupported_schema(data, config_path, config_class)

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
