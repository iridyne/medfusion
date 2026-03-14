# Aggregators

Multiple Instance Learning (MIL) aggregators for combining features from multiple instances.

## Overview

The aggregators module provides various strategies for aggregating instance-level features
into bag-level representations, commonly used in medical imaging tasks where multiple
patches or regions need to be combined.

## Available Aggregators

### Simple Aggregators

```{eval-rst}
.. autoclass:: med_core.aggregators.mil.MeanPoolingAggregator
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: med_core.aggregators.mil.MaxPoolingAggregator
   :members:
   :undoc-members:
   :show-inheritance:
```

### Attention-Based Aggregators

```{eval-rst}
.. autoclass:: med_core.aggregators.mil.AttentionAggregator
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: med_core.aggregators.mil.GatedAttentionAggregator
   :members:
   :undoc-members:
   :show-inheritance:
```

### Advanced Aggregators

```{eval-rst}
.. autoclass:: med_core.aggregators.mil.DeepSetsAggregator
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: med_core.aggregators.mil.TransformerAggregator
   :members:
   :undoc-members:
   :show-inheritance:
```

### Unified Interface

```{eval-rst}
.. autoclass:: med_core.aggregators.mil.MILAggregator
   :members:
   :undoc-members:
   :show-inheritance:
```

## Usage Examples

### Basic Usage

```python
from med_core.aggregators.mil import MILAggregator

# Create aggregator
aggregator = MILAggregator(
    input_dim=512,
    strategy='attention',
    attention_dim=128
)

# Aggregate features
features = torch.randn(4, 10, 512)  # [batch, instances, features]
aggregated = aggregator(features)    # [batch, features]
```

### With Attention Weights

```python
# Get attention weights for interpretability
aggregated, attention = aggregator(features, return_attention=True)
print(attention.shape)  # [batch, instances, 1]
```

### Different Strategies

```python
# Mean pooling
mean_agg = MILAggregator(input_dim=512, strategy='mean')

# Max pooling
max_agg = MILAggregator(input_dim=512, strategy='max')

# Gated attention
gated_agg = MILAggregator(input_dim=512, strategy='gated')

# Deep Sets
deepsets_agg = MILAggregator(
    input_dim=512,
    strategy='deepsets',
    hidden_dim=256,
    output_dim=256
)

# Transformer
transformer_agg = MILAggregator(
    input_dim=512,
    strategy='transformer',
    num_heads=8,
    num_layers=2
)
```
