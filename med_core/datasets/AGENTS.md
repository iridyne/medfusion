# Datasets Module

<!-- Parent: ../AGENTS.md -->

**Last Updated:** 2025-02-27

## Purpose

Dataset loaders for medical multimodal learning, supporting single-view and multi-view scenarios with flexible data preprocessing and augmentation.

## Key Components

### Base Classes

1. **BaseMultimodalDataset** (`base.py`)
   - Abstract base for single-view multimodal datasets
   - Combines medical images with tabular clinical data

2. **BaseMultiViewDataset** (`multiview_base.py`)
   - Abstract base for multi-view datasets
   - Handles multiple images per patient (multi-angle CT, time-series, etc.)

### Concrete Implementations

1. **MedicalMultimodalDataset** (`medical.py`)
   - Single-view medical image + tabular data
   - Supports train/val/test splitting
   - Automatic caching for faster loading

2. **MedicalMultiViewDataset** (`medical_multiview.py`)
   - Multi-view medical images + tabular data
   - Supports 5 scenarios: multi-angle CT, time-series, multi-modal, multi-slice, multi-region

### Utilities

1. **Data Cleaning** (`data_cleaner.py`)
   - `DataCleaner`: Handles missing values, outliers, normalization

2. **Transforms** (`transforms.py`)
   - `get_train_transforms()`: Training augmentation pipeline
   - `get_val_transforms()`: Validation transforms (no augmentation)
   - `get_medical_augmentation()`: Medical-specific augmentation

3. **Multi-View Types** (`multiview_types.py`)
   - `MultiViewConfig`: Configuration for multi-view scenarios
   - `ViewDict`: Type alias for view dictionaries
   - `ViewTensor`: Type alias for view tensors
   - `convert_to_multiview_paths()`: Convert paths to multi-view format
   - `create_single_view_dict()`: Create single-view dictionary

4. **Data Loading** (`medical.py`)
   - `create_dataloaders()`: Create train/val/test dataloaders
   - `split_dataset()`: Split dataset into train/val/test

## Architecture

```
CSV File → Dataset → DataLoader → Batch
         ↓
    Image + Tabular + Label
```

### Single-View Flow
```
Image Path → Load Image → Transform → Tensor [C, H, W]
Tabular Data → Normalize → Tensor [F]
```

### Multi-View Flow
```
Image Paths → Load Images → Transform → Tensor [V, C, H, W]
Tabular Data → Normalize → Tensor [F]
```

## Usage Patterns

### Single-View Dataset
```python
from med_core.datasets import MedicalMultimodalDataset, create_dataloaders

dataset = MedicalMultimodalDataset(
    csv_path='data/train.csv',
    image_col='image_path',
    tabular_cols=['age', 'sex', 'bmi'],
    label_col='label',
    transform=get_train_transforms()
)

train_loader, val_loader, test_loader = create_dataloaders(
    dataset,
    batch_size=32,
    num_workers=4
)
```

### Multi-View Dataset
```python
from med_core.datasets import MedicalMultiViewDataset, MultiViewConfig

config = MultiViewConfig(
    num_views=4,
    view_aggregation='attention',
    scenario='multi_angle_ct'
)

dataset = MedicalMultiViewDataset(
    csv_path='data/train.csv',
    image_col='image_path',  # Will be expanded to multiple views
    tabular_cols=['age', 'sex'],
    label_col='label',
    multiview_config=config,
    transform=get_train_transforms()
)
```

### Data Cleaning
```python
from med_core.datasets import DataCleaner

cleaner = DataCleaner()
cleaned_df = cleaner.clean(
    df,
    numeric_cols=['age', 'bmi'],
    categorical_cols=['sex'],
    handle_missing='mean',
    remove_outliers=True
)
```

## Key Files

- `base.py`: Single-view base class
- `multiview_base.py`: Multi-view base class
- `medical.py`: Single-view implementation + utilities
- `medical_multiview.py`: Multi-view implementation
- `data_cleaner.py`: Data cleaning utilities
- `transforms.py`: Image augmentation pipelines
- `multiview_types.py`: Multi-view type definitions

## Dependencies

- PyTorch (Dataset, DataLoader)
- pandas (CSV loading)
- PIL/OpenCV (image loading)
- torchvision.transforms (augmentation)
- Used by: `med_core.trainers`, `med_core.cli`

## Testing

Run tests with:
```bash
uv run pytest tests/test_datasets.py -v
```

## Common Issues

1. **Missing images**: Check image paths in CSV are correct and files exist
2. **Inconsistent tabular columns**: Ensure all rows have the same columns
3. **Memory usage**: Use `num_workers > 0` and `pin_memory=True` for faster loading
4. **Multi-view path format**: Use `convert_to_multiview_paths()` to format paths correctly
5. **Transform errors**: Ensure transforms match image format (PIL vs tensor)

## Data Format

### CSV Format (Single-View)
```csv
image_path,age,sex,bmi,label
data/images/001.png,45,M,25.3,0
data/images/002.png,52,F,28.1,1
```

### CSV Format (Multi-View)
```csv
image_path,age,sex,label
data/images/001_view1.png|data/images/001_view2.png,45,M,0
data/images/002_view1.png|data/images/002_view2.png,52,F,1
```

Or use separate columns:
```csv
view1,view2,view3,view4,age,sex,label
data/001_v1.png,data/001_v2.png,data/001_v3.png,data/001_v4.png,45,M,0
```

## Related Modules

- `backbones/`: Processes images from datasets
- `trainers/`: Uses datasets for training
- `configs/`: Configures dataset parameters
- `preprocessing/`: Preprocesses images before loading
