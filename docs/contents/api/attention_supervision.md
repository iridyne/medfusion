# Attention Supervision

Attention supervision mechanisms for interpretable medical imaging models.

## Core Components

The attention supervision module provides mechanisms to guide model attention using supervision signals, improving both interpretability and performance.

## Supervision Strategies

Various supervision strategies are available including region-based, point-based, and scribble-based supervision for different annotation types.

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
