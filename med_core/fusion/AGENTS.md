# Fusion Module

<!-- Parent: ../AGENTS.md -->

**Last Updated:** 2025-02-27

## Purpose

Multimodal fusion strategies for combining features from different modalities (vision, tabular, multi-view) in medical deep learning.

## Key Components

### Base Classes (`base.py`)
- **BaseFusion**: Abstract base for all fusion modules
- **MultiModalFusionModel**: Complete multimodal model with fusion
- **create_fusion_model()**: Factory function for fusion models

### Basic Fusion Strategies (`strategies.py`)

1. **ConcatenateFusion**: Simple concatenation of features
2. **GatedFusion**: Gated mechanism for modality weighting
3. **AttentionFusion**: Attention-based fusion
4. **CrossAttentionFusion**: Cross-modal attention
5. **BilinearFusion**: Bilinear pooling for feature interaction

Factory: `create_fusion_module(fusion_type, **kwargs)`

### Advanced Fusion Strategies

1. **Kronecker Fusion** (`kronecker.py`)
   - **KroneckerFusion**: Kronecker product fusion
   - **CompactKroneckerFusion**: Memory-efficient variant
   - **MultimodalKroneckerFusion**: Multi-modality Kronecker fusion

2. **Fused Attention** (`fused_attention.py`)
   - **FusedAttentionFusion**: Fused attention mechanism
   - **CrossModalAttention**: Cross-modal attention layers
   - **MultimodalFusedAttention**: Multi-modality fused attention

3. **Self-Attention Fusion** (`self_attention.py`)
   - **SelfAttentionFusion**: Self-attention based fusion
   - **AdditiveAttentionFusion**: Additive attention mechanism
   - **BilinearAttentionFusion**: Bilinear attention
   - **GatedAttentionFusion**: Gated attention fusion
   - **MultimodalSelfAttentionFusion**: Multi-modality self-attention

### Multi-View Support (`multiview_model.py`)
- **MultiViewMultiModalFusionModel**: Fusion for multi-view scenarios
- **create_multiview_fusion_model()**: Factory for multi-view fusion

## Architecture

```
Modality 1 Features →
Modality 2 Features → Fusion Module → Fused Features
Modality N Features →
```

## Usage Patterns

### Basic Fusion
```python
from med_core.fusion import create_fusion_module

fusion = create_fusion_module(
    fusion_type='attention',
    input_dims=[2048, 128],  # Vision + Tabular
    output_dim=512
)
fused = fusion([vision_features, tabular_features])
```

### Complete Fusion Model
```python
from med_core.fusion import create_fusion_model

model = create_fusion_model(
    vision_backbone='resnet50',
    tabular_input_dim=50,
    fusion_type='attention',
    num_classes=2
)
output = model(images, tabular_data)
```

### Kronecker Fusion
```python
from med_core.fusion import KroneckerFusion

fusion = KroneckerFusion(
    input_dims=[2048, 128],
    output_dim=512,
    rank=64  # Low-rank approximation
)
fused = fusion([vision_features, tabular_features])
```

### Multi-View Fusion
```python
from med_core.fusion import create_multiview_fusion_model

model = create_multiview_fusion_model(
    vision_backbone='resnet50',
    num_views=4,
    view_aggregation='attention',
    tabular_input_dim=50,
    fusion_type='cross_attention',
    num_classes=2
)
output = model(multi_view_images, tabular_data)
```

## Key Files

- `base.py`: Base classes and model builder
- `strategies.py`: Basic fusion strategies
- `kronecker.py`: Kronecker product fusion
- `fused_attention.py`: Fused attention mechanisms
- `self_attention.py`: Self-attention fusion
- `multiview_model.py`: Multi-view fusion support

## Fusion Strategy Comparison

| Strategy | Complexity | Parameters | Best For |
|----------|-----------|------------|----------|
| Concatenate | Low | None | Baseline, simple fusion |
| Gated | Medium | O(D) | Modality weighting |
| Attention | Medium | O(D²) | Selective fusion |
| Cross-Attention | High | O(D²) | Inter-modality interaction |
| Bilinear | High | O(D₁×D₂) | Feature interaction |
| Kronecker | Very High | O(D₁×D₂) | Rich interaction |
| Self-Attention | High | O(D²) | Multi-modality fusion |

## Dependencies

- PyTorch (nn.Module, attention mechanisms)
- `med_core.backbones`: Provides features to fuse
- Used by: `med_core.models`, `med_core.trainers`

## Testing

Run tests with:
```bash
uv run pytest tests/test_fusion.py -v
```

## Common Issues

1. **Dimension mismatch**: Ensure input_dims match backbone output dimensions
2. **Memory usage**: Bilinear and Kronecker fusion can be memory-intensive
3. **Gradient flow**: Some fusion strategies may have gradient flow issues
4. **Overfitting**: Complex fusion strategies may overfit on small datasets

## Related Modules

- `backbones/`: Provides features to fuse
- `models/`: Builds complete models with fusion
- `aggregators/`: May be used within fusion modules
