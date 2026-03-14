# Backbones

Feature extraction backbones for medical imaging.

## Vision Backbones

Supports 29+ vision backbones including:
- ResNet, EfficientNet, ViT, Swin Transformer (2D/3D)
- DenseNet, ConvNeXt, MaxViT, RegNet, MobileNet

## Tabular Backbones

MLP-based backbones for processing tabular/clinical features.

## Attention Mechanisms

Built-in attention mechanisms for feature refinement and interpretability.

## Usage Example

```python
from med_core.backbones import create_backbone

# Create vision backbone
vision_backbone = create_backbone(
    'resnet50',
    pretrained=True,
    num_classes=0  # Feature extraction only
)

# Create tabular backbone
tabular_backbone = create_backbone(
    'mlp',
    input_dim=20,
    hidden_dims=[128, 64]
)
```
