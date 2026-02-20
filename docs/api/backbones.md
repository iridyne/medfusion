# Backbones

Feature extraction backbones for medical imaging.

```{eval-rst}
.. automodule:: med_core.backbones
   :members:
   :undoc-members:
   :show-inheritance:
```

## Vision Backbones

```{eval-rst}
.. automodule:: med_core.backbones.vision
   :members:
   :undoc-members:
   :show-inheritance:
```

## Tabular Backbones

```{eval-rst}
.. automodule:: med_core.backbones.tabular
   :members:
   :undoc-members:
   :show-inheritance:
```

## Attention Mechanisms

```{eval-rst}
.. automodule:: med_core.backbones.attention
   :members:
   :undoc-members:
   :show-inheritance:
```

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
