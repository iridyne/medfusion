# Configs Module

<!-- Parent: ../AGENTS.md -->

**Last Updated:** 2025-02-27

## Purpose

Configuration system for MedFusion framework with structured validation, error reporting, and multi-view support.

## Key Components

### Base Configuration (`base_config.py`)
- **BaseConfig**: Base configuration with common fields
- **DataConfig**: Dataset paths and data loading settings
- **VisionConfig**: Vision backbone configuration
- **TabularConfig**: Tabular backbone configuration
- **FusionConfig**: Fusion strategy configuration
- **ModelConfig**: Complete model architecture
- **OptimizerConfig**: Optimizer settings
- **SchedulerConfig**: Learning rate scheduler
- **TrainingConfig**: Training hyperparameters
- **LoggingConfig**: TensorBoard/WandB logging
- **ExperimentConfig**: Complete experiment configuration

### Multi-View Configuration (`multiview_config.py`)
- **MultiViewDataConfig**: Multi-view dataset configuration
- **MultiViewVisionConfig**: Multi-view vision backbone
- **MultiViewModelConfig**: Multi-view model architecture
- **MultiViewExperimentConfig**: Complete multi-view experiment

Factory functions:
- `create_ct_multiview_config()`: CT multi-angle configuration
- `create_temporal_multiview_config()`: Temporal sequence configuration

### Configuration Loading (`config_loader.py`)
- `load_config(path)`: Load YAML config file
- `save_config(config, path)`: Save config to YAML
- `create_default_config()`: Generate default configuration

### Validation (`validation.py`)
- **ConfigValidator**: Validates configuration with structured error reporting
- `validate_config(config)`: Validate and return errors
- `validate_config_or_exit(config)`: Validate or exit with error messages

Error codes: E001-E028 for different validation failures

## Architecture

```
YAML File → load_config() → Config Object → validate_config() → Validated Config
```

## Usage Patterns

### Load and Validate Config
```python
from med_core.configs import load_config, validate_config_or_exit

config = load_config('configs/default.yaml')
validate_config_or_exit(config)  # Exits if invalid
```

### Create Default Config
```python
from med_core.configs import create_default_config, save_config

config = create_default_config()
config.training.epochs = 200
config.model.vision.backbone = 'resnet50'
save_config(config, 'configs/my_config.yaml')
```

### Multi-View Config
```python
from med_core.configs import create_ct_multiview_config

config = create_ct_multiview_config(
    num_views=4,
    aggregator_type='attention',
    backbone='resnet50'
)
```

### Manual Validation
```python
from med_core.configs import ConfigValidator

validator = ConfigValidator()
errors = validator.validate(config)

if errors:
    for error in errors:
        print(f"{error.code}: {error.message}")
        print(f"  Path: {error.path}")
        print(f"  Suggestion: {error.suggestion}")
```

## Key Files

- `base_config.py`: Base configuration classes
- `multiview_config.py`: Multi-view configuration
- `config_loader.py`: Load/save utilities
- `validation.py`: Configuration validation with error codes

## Configuration Schema

### Minimal Config
```yaml
data:
  train_csv: data/train.csv
  image_col: image_path
  label_col: label

model:
  vision:
    backbone: resnet50
  num_classes: 2

training:
  epochs: 100
  batch_size: 32
```

### Full Config
```yaml
data:
  train_csv: data/train.csv
  val_csv: data/val.csv
  test_csv: data/test.csv
  image_col: image_path
  tabular_cols: [age, sex, bmi]
  label_col: label
  batch_size: 32
  num_workers: 4

model:
  vision:
    backbone: resnet50
    pretrained: true
    freeze_backbone: false
  tabular:
    hidden_dims: [256, 128, 64]
    dropout: 0.3
  fusion:
    type: attention
    hidden_dim: 256
  num_classes: 2

training:
  epochs: 100
  learning_rate: 0.001
  weight_decay: 0.0001
  mixed_precision: true
  early_stopping_patience: 10

logging:
  use_tensorboard: true
  use_wandb: false
  log_dir: outputs/logs
```

## Dependencies

- dataclasses (configuration classes)
- PyYAML (YAML parsing)
- Used by: All modules for configuration

## Testing

Run tests with:
```bash
uv run pytest tests/test_configs.py -v
```

## Common Issues

1. **Missing required fields**: Validator provides specific error codes and suggestions
2. **Invalid backbone name**: Check supported backbones in `backbones/vision.py`
3. **Incompatible fusion type**: Ensure fusion type matches model architecture
4. **Path not found**: Validate CSV paths exist before training

## Validation Error Codes

- **E001-E005**: Data configuration errors
- **E006-E010**: Model configuration errors
- **E011-E015**: Training configuration errors
- **E016-E020**: Fusion configuration errors
- **E021-E025**: Multi-view configuration errors
- **E026-E028**: General validation errors

## Related Modules

- `cli/`: Uses configs for CLI commands
- `trainers/`: Reads training configuration
- `models/`: Builds models from config
- `datasets/`: Loads data based on config
