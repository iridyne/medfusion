# models/ - Model Architecture Module

<!-- Parent: ../AGENTS.md -->

**Last Updated:** 2025-02-27

## Purpose

Provides unified model building infrastructure for multi-modal medical deep learning. Implements the Generic Multi-Modal Model architecture that combines arbitrary modalities (vision, tabular, 3D imaging) with flexible fusion strategies and task-specific heads.

## Key Components

### Core Classes

- **GenericMultiModalModel**: Main model class supporting 2+ modalities with arbitrary backbones, fusion strategies, and task heads
- **MultiModalModelBuilder**: Fluent API builder for constructing models from components
- **SMuRFModel**: Specialized radiology-pathology fusion model
- **SMuRFWithMIL**: SMuRF variant with Multi-Instance Learning aggregation

### Factory Functions

- `build_model_from_config()`: Build models from YAML configuration
- `smurf_small()`, `smurf_base()`: Pre-configured SMuRF model variants
- `smurf_with_mil_small()`: SMuRF with MIL aggregation

## Architecture

```
Model = Backbones + Fusion + Head + (Optional) MIL Aggregators

GenericMultiModalModel:
  ├── modality_backbones: Dict[str, nn.Module]
  ├── fusion_module: nn.Module
  ├── head: nn.Module (classification/survival)
  └── mil_aggregators: Optional[Dict[str, nn.Module]]
```

## Usage Patterns

### Pattern 1: Builder API (Recommended)

```python
from med_core.models import MultiModalModelBuilder

builder = MultiModalModelBuilder()
model = (builder
    .add_modality('ct', backbone='swin3d_tiny', modality_type='vision3d', in_channels=1)
    .add_modality('pathology', backbone='resnet50', modality_type='vision', pretrained=True)
    .set_fusion('attention', hidden_dim=256)
    .set_head('classification', num_classes=4)
    .build())
```

### Pattern 2: Config-Driven

```python
from med_core.models import build_model_from_config

config = {
    'modalities': {
        'ct': {'backbone': 'swin3d_small', 'modality_type': 'vision3d'},
        'clinical': {'backbone': 'mlp', 'modality_type': 'tabular', 'input_dim': 20}
    },
    'fusion': {'strategy': 'gated', 'output_dim': 512},
    'head': {'task_type': 'classification', 'num_classes': 2}
}
model = build_model_from_config(config)
```

### Pattern 3: Direct Construction

```python
from med_core.models import GenericMultiModalModel
from med_core.backbones import create_vision_backbone
from med_core.fusion import create_fusion_module
from med_core.heads import ClassificationHead

backbones = {
    'ct': create_vision_backbone('swin3d_tiny', in_channels=1),
    'pathology': create_vision_backbone('resnet50', pretrained=True)
}
fusion = create_fusion_module('attention', vision_dim=768, tabular_dim=2048, output_dim=512)
head = ClassificationHead(input_dim=512, num_classes=2)
model = GenericMultiModalModel(backbones, fusion, head)
```

## Supported Configurations

### Modality Types

- **vision**: 2D medical imaging (CT slices, pathology, X-ray)
- **vision3d**: 3D volumetric imaging (CT volumes, MRI)
- **tabular**: Clinical data (demographics, lab results)
- **custom**: User-provided backbone modules

### Fusion Strategies

- `concatenate`: Simple feature concatenation
- `gated`: Gated fusion with learned gates
- `attention`: Attention-based fusion
- `cross_attention`: Cross-modal attention
- `bilinear`: Bilinear pooling
- `kronecker`: Kronecker product fusion
- `fused_attention`: Multi-head fused attention

### Task Heads

- `classification`: Multi-class classification
- `survival_cox`: Cox proportional hazards model
- `survival_deep`: Deep survival analysis
- `survival_discrete`: Discrete-time survival

### MIL Aggregation

- `mean`: Average pooling
- `max`: Max pooling
- `attention`: Attention-based aggregation
- `gated`: Gated attention aggregation
- `deepsets`: DeepSets aggregation
- `transformer`: Transformer-based aggregation

## Key Features

1. **Modality Flexibility**: Support 2+ modalities with different input types
2. **Component Decoupling**: Backbones, fusion, and heads are completely independent
3. **MIL Support**: Optional multi-instance learning for patch-based inputs
4. **Feature Extraction**: `return_features=True` returns intermediate representations
5. **Modality Contribution**: `get_modality_contribution()` analyzes modality importance
6. **Config-Driven**: All architectures definable via YAML without code changes

## Integration Points

### Upstream Dependencies

- `med_core.backbones`: Vision and tabular backbone networks
- `med_core.fusion`: Multi-modal fusion strategies
- `med_core.heads`: Task-specific output layers
- `med_core.aggregators`: MIL aggregation modules

### Downstream Consumers

- `med_core.trainers`: Training loops use these models
- `med_core.web`: Web UI for model management
- `configs/*.yaml`: Configuration files define model architectures

## File Structure

```
models/
├── __init__.py          # Public API exports
├── builder.py           # GenericMultiModalModel + Builder
└── smurf.py            # SMuRF specialized models
```

## Testing

```bash
# Test model building
uv run pytest tests/test_models.py::test_model_builder

# Test config-driven construction
uv run pytest tests/test_models.py::test_build_from_config

# Test SMuRF models
uv run pytest tests/test_models.py::test_smurf_models
```

## Common Tasks

### Add New Modality Type

1. Implement backbone in `med_core/backbones/`
2. Register with backbone factory
3. Add modality type to `MultiModalModelBuilder.add_modality()`
4. Update type hints in `builder.py`

### Add New Fusion Strategy

1. Implement in `med_core/fusion/strategies.py`
2. Register with fusion factory
3. Add to fusion strategy type hints
4. Update documentation

### Add New Task Head

1. Implement in `med_core/heads/`
2. Add to `MultiModalModelBuilder.set_head()`
3. Update task type hints
4. Add tests

## Performance Notes

- **Memory**: MIL aggregation processes N instances in batch (B*N reshape)
- **Computation**: Fusion complexity varies by strategy (attention > concat)
- **Optimization**: Use gradient checkpointing for large 3D backbones

## Related Documentation

- Architecture overview: `CLAUDE.md` → Architecture section
- Backbone registry: `med_core/backbones/AGENTS.md`
- Fusion strategies: `med_core/fusion/AGENTS.md`
- Training: `med_core/trainers/AGENTS.md`
