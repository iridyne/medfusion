# preprocessing/ - Data Preprocessing Module

<!-- Parent: ../AGENTS.md -->

**Last Updated:** 2025-02-27

## Purpose

Provides specialized preprocessing pipelines for medical imaging data. Handles intensity normalization, contrast enhancement, artifact detection, quality assessment, and ROI extraction for CT, MRI, pathology, and X-ray images.

## Key Components

### Core Classes

- **ImagePreprocessor**: Main preprocessing pipeline for medical images
- **QualityMetrics**: Image quality assessment metrics

### Preprocessing Functions

- `normalize_intensity()`: Intensity normalization (z-score, min-max, percentile)
- `apply_clahe()`: Contrast Limited Adaptive Histogram Equalization
- `crop_center()`: Center cropping with configurable size
- `assess_image_quality()`: Compute quality metrics (SNR, contrast, sharpness)
- `detect_artifacts()`: Detect common imaging artifacts

## Architecture

```
ImagePreprocessor:
  ├── normalize: bool
  ├── clahe: bool
  ├── crop_size: Optional[Tuple[int, int]]
  └── __call__(image) -> np.ndarray

QualityMetrics:
  ├── snr: float (Signal-to-Noise Ratio)
  ├── contrast: float
  ├── sharpness: float
  └── artifact_score: float
```

## Usage Patterns

### Basic Preprocessing

```python
from med_core.preprocessing import ImagePreprocessor

# Create preprocessor
preprocessor = ImagePreprocessor(
    normalize=True,
    clahe=True,
    crop_size=(224, 224)
)

# Process image
processed_image = preprocessor(raw_image)
```

### Individual Operations

```python
from med_core.preprocessing import normalize_intensity, apply_clahe, crop_center

# Normalize intensity
normalized = normalize_intensity(image, method='zscore')

# Apply CLAHE for contrast enhancement
enhanced = apply_clahe(image, clip_limit=2.0, tile_grid_size=(8, 8))

# Center crop
cropped = crop_center(image, output_size=(512, 512))
```

### Quality Assessment

```python
from med_core.preprocessing import assess_image_quality, detect_artifacts

# Assess quality
quality = assess_image_quality(image)
print(f"SNR: {quality.snr:.2f}, Contrast: {quality.contrast:.2f}")

# Detect artifacts
artifacts = detect_artifacts(image)
if artifacts['motion_blur'] > 0.5:
    print("Warning: Motion blur detected")
```

### Pipeline Integration

```python
from med_core.preprocessing import ImagePreprocessor
from med_core.datasets import MedicalDataset

class PreprocessedDataset(MedicalDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.preprocessor = ImagePreprocessor(
            normalize=True,
            clahe=True,
            crop_size=(224, 224)
        )

    def __getitem__(self, idx):
        image, label = super().__getitem__(idx)
        image = self.preprocessor(image)
        return image, label
```

## Key Features

1. **Medical-Specific**: Optimized for medical imaging characteristics
2. **Modality-Agnostic**: Works with CT, MRI, pathology, X-ray
3. **Quality-Aware**: Built-in quality assessment and artifact detection
4. **Configurable**: Flexible pipeline with optional steps
5. **Efficient**: Optimized NumPy operations for speed
6. **DICOM Support**: Integration with DICOM loading utilities

## Preprocessing Methods

### Intensity Normalization

- **Z-score**: `(x - mean) / std`
- **Min-Max**: `(x - min) / (max - min)`
- **Percentile**: Clip to [p1, p99] then normalize
- **Window-Level**: CT windowing (HU units)

### Contrast Enhancement

- **CLAHE**: Adaptive histogram equalization
- **Gamma Correction**: Power-law transformation
- **Histogram Matching**: Match reference histogram

### Spatial Operations

- **Center Crop**: Extract center region
- **Resize**: Bilinear/bicubic interpolation
- **Padding**: Zero/reflect/edge padding
- **ROI Extraction**: Bounding box extraction

### Quality Metrics

- **SNR**: Signal-to-Noise Ratio
- **CNR**: Contrast-to-Noise Ratio
- **Sharpness**: Laplacian variance
- **Entropy**: Image information content

### Artifact Detection

- **Motion Blur**: Frequency domain analysis
- **Noise**: Local variance estimation
- **Aliasing**: High-frequency artifacts
- **Truncation**: Edge artifacts

## Integration Points

### Upstream Dependencies

- `numpy`: Array operations
- `opencv-python`: Image processing
- `scikit-image`: Advanced operations
- `med_core.shared.data_utils`: DICOM loading

### Downstream Consumers

- `med_core.datasets`: Dataset preprocessing pipelines
- `med_core.trainers`: Data augmentation integration
- `med_core.evaluation`: Quality-aware evaluation
- `scripts/preprocess_data.py`: Batch preprocessing

## File Structure

```
preprocessing/
├── __init__.py          # Public API exports
├── image.py            # ImagePreprocessor + basic operations
└── quality.py          # QualityMetrics + artifact detection
```

## Testing

```bash
# Test preprocessing operations
uv run pytest tests/test_preprocessing.py::test_normalize_intensity
uv run pytest tests/test_preprocessing.py::test_clahe
uv run pytest tests/test_preprocessing.py::test_quality_assessment

# Test full pipeline
uv run pytest tests/test_preprocessing.py::test_image_preprocessor
```

## Common Tasks

### Add New Normalization Method

```python
# In image.py
def normalize_intensity(image, method='zscore', **kwargs):
    if method == 'zscore':
        return (image - image.mean()) / (image.std() + 1e-8)
    elif method == 'new_method':
        # Implement new normalization
        return normalized_image
```

### Add New Quality Metric

```python
# In quality.py
def assess_image_quality(image):
    metrics = QualityMetrics()
    metrics.snr = compute_snr(image)
    metrics.new_metric = compute_new_metric(image)
    return metrics
```

### Batch Preprocessing Script

```python
from pathlib import Path
from med_core.preprocessing import ImagePreprocessor
import numpy as np

preprocessor = ImagePreprocessor(normalize=True, clahe=True)

input_dir = Path('data/raw')
output_dir = Path('data/processed')
output_dir.mkdir(exist_ok=True)

for img_path in input_dir.glob('*.npy'):
    image = np.load(img_path)
    processed = preprocessor(image)
    np.save(output_dir / img_path.name, processed)
```

## Performance Notes

- **Speed**: CLAHE is slowest operation (~50ms for 512x512)
- **Memory**: In-place operations where possible
- **Parallelization**: Use multiprocessing for batch preprocessing
- **Caching**: Cache preprocessed images for faster training

## Best Practices

1. **Normalize First**: Always normalize before other operations
2. **Quality Check**: Assess quality before training
3. **Artifact Filtering**: Remove low-quality images
4. **Consistent Pipeline**: Use same preprocessing for train/val/test
5. **Document Settings**: Save preprocessing config with model

## Related Documentation

- Dataset integration: `med_core/datasets/AGENTS.md`
- DICOM utilities: `med_core/shared/AGENTS.md`
- Training pipelines: `med_core/trainers/AGENTS.md`
- CLI preprocessing: `CLAUDE.md` → Development Commands
