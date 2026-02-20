# Attention Supervision

Attention supervision mechanisms for interpretable medical imaging models.

```{eval-rst}
.. automodule:: med_core.attention_supervision
   :members:
   :undoc-members:
   :show-inheritance:
```

## Core Components

```{eval-rst}
.. automodule:: med_core.attention_supervision.core
   :members:
   :undoc-members:
   :show-inheritance:
```

## Supervision Strategies

```{eval-rst}
.. automodule:: med_core.attention_supervision.supervision
   :members:
   :undoc-members:
   :show-inheritance:
```

## Usage Example

```python
from med_core.attention_supervision import AttentionSupervision

# Create attention supervision
supervision = AttentionSupervision(
    supervision_type='region',
    loss_weight=0.1
)

# Apply supervision
attention_maps = model.get_attention_maps()
supervision_loss = supervision(attention_maps, region_masks)
```
