# Extractors Module

<!-- Parent: ../AGENTS.md -->

**Last Updated:** 2025-02-27

## Purpose

Feature extractors for medical imaging, specializing in multi-region and multi-scale feature extraction from medical images.

## Key Components

### Multi-Region Extractors (`multi_region.py`)

1. **MultiRegionExtractor**
   - Extracts features from multiple predefined regions of interest (ROIs)
   - Useful for organ-specific analysis

2. **AdaptiveRegionExtractor**
   - Automatically identifies and extracts features from salient regions
   - Uses attention mechanisms to locate important areas

3. **HierarchicalRegionExtractor**
   - Extracts features at multiple hierarchical levels
   - Captures both local and global context

4. **MultiScaleRegionExtractor**
   - Extracts features at multiple scales
   - Combines fine-grained and coarse-grained information

## Architecture

```
Input Image → Region Detection → Feature Extraction → Multi-Region Features
[B, C, H, W] → [B, N_regions, D]
```

## Usage Patterns

### Multi-Region Extraction
```python
from med_core.extractors import MultiRegionExtractor

extractor = MultiRegionExtractor(
    backbone='resnet50',
    num_regions=4,
    region_size=(64, 64)
)
features = extractor(images)  # [B, 4, 2048]
```

### Adaptive Region Extraction
```python
from med_core.extractors import AdaptiveRegionExtractor

extractor = AdaptiveRegionExtractor(
    backbone='resnet50',
    num_regions=5,
    attention_type='spatial'
)
features, attention_maps = extractor(images)
```

### Hierarchical Extraction
```python
from med_core.extractors import HierarchicalRegionExtractor

extractor = HierarchicalRegionExtractor(
    backbone='resnet50',
    num_levels=3
)
features = extractor(images)  # List of features at different levels
```

## Key Files

- `multi_region.py`: All multi-region extractor implementations
- `__init__.py`: Public API exports

## Dependencies

- PyTorch (nn.Module)
- `med_core.backbones`: Uses vision backbones for feature extraction
- Used by: `med_core.models`, `med_core.trainers`

## Testing

Run tests with:
```bash
uv run pytest tests/test_extractors.py -v
```

## Common Issues

1. **Region size**: Ensure region size is compatible with backbone input requirements
2. **Number of regions**: Too many regions can cause memory issues
3. **Attention maps**: Some extractors return attention maps as auxiliary output
4. **Hierarchical features**: May need custom aggregation for downstream tasks

## Related Modules

- `backbones/`: Provides base feature extraction
- `aggregators/`: Aggregates multi-region features
- `attention_supervision/`: Can supervise region selection
