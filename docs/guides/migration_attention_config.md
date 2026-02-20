# Migration Guide: attention_config.py Removal

## Overview

The `med_core.configs.attention_config` module has been **removed** as of version 0.2.0.
All attention supervision functionality is now integrated into the main configuration system.

## Why Was It Removed?

1. **Redundancy**: The functionality was duplicated in `ExperimentConfig`
2. **Confusion**: Having two configuration systems caused confusion
3. **Maintenance**: Maintaining two parallel systems was error-prone
4. **Simplicity**: A single unified configuration is easier to use

## Migration Steps

### Step 1: Update Imports

**Before (Deprecated):**
```python
from med_core.configs.attention_config import (
    AttentionSupervisionConfig,
    ExperimentConfigWithAttention,
    DataConfigWithMask,
    TrainingConfigWithAttention,
)
```

**After (Current):**
```python
from med_core.configs import ExperimentConfig
```

### Step 2: Update Configuration Creation

#### Example 1: Basic Attention Supervision

**Before:**
```python
from med_core.configs.attention_config import (
    ExperimentConfigWithAttention,
    AttentionSupervisionConfig,
)

config = ExperimentConfigWithAttention(
    experiment_name="my_experiment",
    training=TrainingConfigWithAttention(
        attention_supervision=AttentionSupervisionConfig(
            enabled=True,
            method="mask",
            loss_weight=0.1,
            loss_type="kl",
        )
    )
)
```

**After:**
```python
from med_core.configs import ExperimentConfig

config = ExperimentConfig()
config.experiment_name = "my_experiment"
config.training.use_attention_supervision = True
config.training.attention_supervision_method = "mask"
config.training.attention_loss_weight = 0.1
config.training.attention_loss_type = "kl"
```

#### Example 2: Mask Supervision

**Before:**
```python
from med_core.configs.attention_config import create_mask_supervised_config

attention_config = create_mask_supervised_config(
    loss_weight=0.1,
    loss_type="kl"
)
```

**After:**
```python
from med_core.configs import ExperimentConfig

config = ExperimentConfig()
config.training.use_attention_supervision = True
config.training.attention_supervision_method = "mask"
config.training.attention_loss_weight = 0.1
config.training.attention_loss_type = "kl"
config.training.attention_temperature = 10.0
```

#### Example 3: CAM Supervision

**Before:**
```python
from med_core.configs.attention_config import create_cam_supervised_config

attention_config = create_cam_supervised_config(
    loss_weight=0.1,
    consistency_method="entropy"
)
```

**After:**
```python
from med_core.configs import ExperimentConfig

config = ExperimentConfig()
config.training.use_attention_supervision = True
config.training.attention_supervision_method = "cam"
config.training.attention_loss_weight = 0.1
config.training.cam_consistency_method = "entropy"
config.training.cam_consistency_weight = 1.0
```

#### Example 4: MIL Configuration

**Before:**
```python
from med_core.configs.attention_config import create_mil_config

attention_config = create_mil_config(
    loss_weight=0.1,
    patch_size=16
)
```

**After:**
```python
from med_core.configs import ExperimentConfig

config = ExperimentConfig()
config.training.use_attention_supervision = True
config.training.attention_supervision_method = "mil"
config.training.attention_loss_weight = 0.1
config.training.mil_patch_size = 16
config.training.mil_attention_dim = 128
```

#### Example 5: BBox Supervision

**Before:**
```python
from med_core.configs.attention_config import create_bbox_supervised_config

attention_config = create_bbox_supervised_config(
    loss_weight=0.1,
    bbox_format="xyxy"
)
```

**After:**
```python
from med_core.configs import ExperimentConfig

config = ExperimentConfig()
config.training.use_attention_supervision = True
config.training.attention_supervision_method = "bbox"
config.training.attention_loss_weight = 0.1
config.training.bbox_format = "xyxy"
```

### Step 3: Update Data Configuration

**Before:**
```python
from med_core.configs.attention_config import DataConfigWithMask

data_config = DataConfigWithMask(
    data_dir="data/",
    csv_file="annotations.csv",
    mask_dir="masks/",
    return_mask=True,
    return_bbox=False,
)
```

**After:**
```python
from med_core.configs import ExperimentConfig

config = ExperimentConfig()
config.data.data_dir = "data/"
config.data.csv_file = "annotations.csv"
config.data.mask_dir = "masks/"
config.data.return_mask = True
config.data.return_bbox = False
```

## Configuration Mapping

### AttentionSupervisionConfig → ExperimentConfig.training

| Old (attention_config) | New (ExperimentConfig.training) |
|------------------------|----------------------------------|
| `enabled` | `use_attention_supervision` |
| `method` | `attention_supervision_method` |
| `loss_weight` | `attention_loss_weight` |
| `loss_type` | `attention_loss_type` |
| `temperature` | `attention_temperature` |
| `add_smooth_loss` | `attention_add_smooth_loss` |
| `smooth_weight` | `attention_smooth_weight` |
| `consistency_method` | `cam_consistency_method` |
| `consistency_weight` | `cam_consistency_weight` |
| `alignment_weight` | `cam_alignment_weight` |
| `cam_threshold` | `cam_threshold` |
| `patch_size` | `mil_patch_size` |
| `attention_dim` | `mil_attention_dim` |
| `diversity_weight` | `mil_diversity_weight` |
| `bbox_format` | `bbox_format` |
| `gaussian_sigma` | `keypoint_gaussian_sigma` |

### DataConfigWithMask → ExperimentConfig.data

| Old (DataConfigWithMask) | New (ExperimentConfig.data) |
|--------------------------|------------------------------|
| `data_dir` | `data_dir` |
| `csv_file` | `csv_file` |
| `image_dir` | `image_dir` |
| `mask_dir` | `mask_dir` |
| `image_col` | `image_col` |
| `mask_col` | `mask_col` |
| `label_col` | `label_col` |
| `tabular_cols` | `tabular_cols` |
| `return_mask` | `return_mask` |
| `return_bbox` | `return_bbox` |
| `return_keypoint` | `return_keypoint` |

## Complete Migration Example

### Before (Using attention_config)

```python
from med_core.configs.attention_config import (
    ExperimentConfigWithAttention,
    DataConfigWithMask,
    TrainingConfigWithAttention,
    AttentionSupervisionConfig,
)

config = ExperimentConfigWithAttention(
    experiment_name="pneumonia_detection",
    output_dir="outputs/pneumonia",
    data=DataConfigWithMask(
        data_dir="data/pneumonia",
        csv_file="annotations.csv",
        image_dir="images",
        mask_dir="lesion_masks",
        return_mask=True,
    ),
    training=TrainingConfigWithAttention(
        num_epochs=100,
        batch_size=32,
        learning_rate=1e-4,
        attention_supervision=AttentionSupervisionConfig(
            enabled=True,
            method="mask",
            loss_weight=0.1,
            loss_type="kl",
            temperature=10.0,
            add_smooth_loss=True,
            smooth_weight=0.01,
        ),
        log_attention_every=100,
        save_attention_maps=True,
    ),
    device="cuda",
    num_workers=4,
)
```

### After (Using ExperimentConfig)

```python
from med_core.configs import ExperimentConfig

config = ExperimentConfig()

# Experiment settings
config.experiment_name = "pneumonia_detection"
config.output_dir = "outputs/pneumonia"

# Data settings
config.data.data_dir = "data/pneumonia"
config.data.csv_file = "annotations.csv"
config.data.image_dir = "images"
config.data.mask_dir = "lesion_masks"
config.data.return_mask = True

# Training settings
config.training.num_epochs = 100
config.training.batch_size = 32
config.training.learning_rate = 1e-4

# Attention supervision settings
config.training.use_attention_supervision = True
config.training.attention_supervision_method = "mask"
config.training.attention_loss_weight = 0.1
config.training.attention_loss_type = "kl"
config.training.attention_temperature = 10.0
config.training.attention_add_smooth_loss = True
config.training.attention_smooth_weight = 0.01
config.training.log_attention_every = 100
config.training.save_attention_maps = True

# Hardware settings
config.device = "cuda"
config.num_workers = 4
```

## YAML Configuration

You can also use YAML configuration files:

```yaml
# config.yaml
experiment_name: pneumonia_detection
output_dir: outputs/pneumonia

data:
  data_dir: data/pneumonia
  csv_file: annotations.csv
  image_dir: images
  mask_dir: lesion_masks
  return_mask: true

training:
  num_epochs: 100
  batch_size: 32
  learning_rate: 0.0001
  
  # Attention supervision
  use_attention_supervision: true
  attention_supervision_method: mask
  attention_loss_weight: 0.1
  attention_loss_type: kl
  attention_temperature: 10.0
  attention_add_smooth_loss: true
  attention_smooth_weight: 0.01
  log_attention_every: 100
  save_attention_maps: true

device: cuda
num_workers: 4
```

Load with:
```python
from med_core.configs import load_config

config = load_config("config.yaml")
```

## Troubleshooting

### Import Error

**Error:**
```
ImportError: cannot import name 'AttentionSupervisionConfig' from 'med_core.configs.attention_config'
```

**Solution:**
Update your imports to use `ExperimentConfig` instead.

### Missing Attributes

**Error:**
```
AttributeError: 'ExperimentConfig' object has no attribute 'attention_supervision'
```

**Solution:**
Use the new attribute names. For example:
- `config.training.attention_supervision.enabled` → `config.training.use_attention_supervision`
- `config.training.attention_supervision.method` → `config.training.attention_supervision_method`

### Configuration Validation

If you're unsure about your configuration, use the validation system:

```python
from med_core.configs import ExperimentConfig
from med_core.configs.validation import validate_config

config = ExperimentConfig()
# ... set your configuration ...

errors = validate_config(config)
if errors:
    for error in errors:
        print(error)
```

## Benefits of Migration

1. **Simpler API**: Single configuration class instead of multiple
2. **Better Validation**: Integrated validation system
3. **YAML Support**: Easy configuration via YAML files
4. **Type Safety**: Better type hints and IDE support
5. **Consistency**: Unified configuration across all features
6. **Maintainability**: Easier to maintain and extend

## Need Help?

- Check the [FAQ](faq_troubleshooting.md)
- See [Configuration Validation Guide](../reference/framework_error_codes.md)
- Review [Quick Reference](quick_reference.md)

## Timeline

- **v0.1.x**: `attention_config` deprecated with warnings
- **v0.2.0**: `attention_config` removed (current)
- **Future**: All configuration through `ExperimentConfig`

---

**Last Updated**: 2026-02-20  
**Applies to**: MedFusion v0.2.0+
