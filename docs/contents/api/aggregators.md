# Aggregators

Multiple Instance Learning (MIL) aggregators for combining features from multiple instances.

## Overview

The aggregators module provides various strategies for aggregating instance-level features
into bag-level representations, commonly used in medical imaging tasks where multiple
patches or regions need to be combined.

## Available Aggregators

### Simple Aggregators

- **MeanPoolingAggregator** - Averages features across instances
- **MaxPoolingAggregator** - Takes maximum values across instances

### Attention-Based Aggregators

- **AttentionAggregator** - Uses attention weights to aggregate instances
- **GatedAttentionAggregator** - Gated attention mechanism for instance aggregation

### Advanced Aggregators

- **DeepSetsAggregator** - Deep Sets architecture for permutation-invariant aggregation
- **TransformerAggregator** - Transformer-based aggregation with multi-head attention

### Unified Interface

The **MILAggregator** class provides a unified interface to all aggregation strategies.

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
