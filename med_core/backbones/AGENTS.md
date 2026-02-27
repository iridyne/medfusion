# Backbones Module

<!-- Parent: ../AGENTS.md -->

**Last Updated:** 2025-02-27

## Purpose

Pluggable backbone networks for feature extraction from medical images and tabular data. Supports 29+ vision architectures and flexible tabular MLPs.

## Key Components

### Base Classes (`base.py`)
- **BaseBackbone**: Abstract base for all backbones
- **BaseVisionBackbone**: Base for vision backbones
- **BaseTabularBackbone**: Base for tabular backbones

### Vision Backbones (`vision.py`)
- **ResNetBackbone**: ResNet-18/34/50/101/152
- **MobileNetBackbone**: MobileNetV2/V3
- **EfficientNetBackbone**: EfficientNet-B0 to B7
- **EfficientNetV2Backbone**: EfficientNetV2-S/M/L
- **ConvNeXtBackbone**: ConvNeXt-Tiny/Small/Base/Large
- **MaxViTBackbone**: MaxViT-Tiny/Small/Base
- **RegNetBackbone**: RegNetY/X variants
- **ViTBackbone**: Vision Transformer (ViT-B/L/H)
- **SwinBackbone**: Swin Transformer (Tiny/Small/Base/Large)

Factory: `create_vision_backbone(name, **kwargs)`

### Tabular Backbones (`tabular.py`)
- **AdaptiveMLP**: Flexible MLP with batch norm and dropout

Factory: `create_tabular_backbone(input_dim, hidden_dims, **kwargs)`

### Multi-View Support (`multiview_vision.py`)
- **MultiViewVisionBackbone**: Processes multiple views per sample
- Supports view aggregation strategies (mean, max, attention)

Factory: `create_multiview_vision_backbone(backbone_name, aggregator_type, **kwargs)`

### View Aggregators (`view_aggregator.py`)
- **MeanPoolAggregator**: Average pooling across views
- **MaxPoolAggregator**: Max pooling across views
- **AttentionAggregator**: Attention-weighted aggregation
- **CrossViewAttentionAggregator**: Cross-view attention
- **LearnedWeightAggregator**: Learnable view weights

Factory: `create_view_aggregator(aggregator_type, **kwargs)`

### Attention Modules (`attention.py`)
- **SEBlock**: Squeeze-and-Excitation
- **ECABlock**: Efficient Channel Attention
- **CBAM**: Convolutional Block Attention Module

Factory: `create_attention_module(module_type, **kwargs)`

## Architecture

```
Input → Backbone → Features
Image [B, C, H, W] → Vision Backbone → [B, D]
Tabular [B, F] → Tabular Backbone → [B, D]
Multi-View [B, V, C, H, W] → MultiView Backbone → [B, D]
```

## Usage Patterns

### Single Vision Backbone
```python
from med_core.backbones import create_vision_backbone

backbone = create_vision_backbone(
    'resnet50',
    pretrained=True,
    in_channels=3,
    num_classes=None  # Feature extraction mode
)
features = backbone(images)  # [B, 2048]
```

### Tabular Backbone
```python
from med_core.backbones import create_tabular_backbone

backbone = create_tabular_backbone(
    input_dim=50,
    hidden_dims=[256, 128, 64],
    dropout=0.3
)
features = backbone(tabular_data)  # [B, 64]
```

### Multi-View Backbone
```python
from med_core.backbones import create_multiview_vision_backbone

backbone = create_multiview_vision_backbone(
    backbone_name='resnet50',
    aggregator_type='attention',
    num_views=4,
    pretrained=True
)
features = backbone(multi_view_images)  # [B, 2048]
```

## Key Files

- `base.py`: Abstract base classes
- `vision.py`: All vision backbone implementations
- `tabular.py`: Tabular MLP backbones
- `multiview_vision.py`: Multi-view vision support
- `view_aggregator.py`: View aggregation strategies
- `attention.py`: Attention mechanisms (SE, ECA, CBAM)
- `swin_components.py`: Swin Transformer components

## Dependencies

- PyTorch (nn.Module)
- torchvision.models (pretrained weights)
- timm (additional pretrained models)
- Used by: `med_core.models`, `med_core.fusion`

## Testing

Run tests with:
```bash
uv run pytest tests/test_backbones.py -v
```

## Common Issues

1. **Pretrained weights**: Not all backbones support pretrained weights for medical imaging (1-channel input)
2. **Feature dimension**: Check `output_dim` property for fusion compatibility
3. **Memory usage**: Large backbones (ViT-L, Swin-L) require significant GPU memory
4. **Input size**: Some backbones (ViT, Swin) require specific input sizes (224x224, 384x384)

## Related Modules

- `fusion/`: Combines features from multiple backbones
- `models/`: Builds complete models with backbones
- `aggregators/`: Aggregates multi-instance features
