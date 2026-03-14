# Classification Heads

Classification heads for various medical imaging tasks.

## Overview

The heads module provides different classification head implementations for
multi-class, multi-label, ordinal, and attention-based classification tasks.

## Available Heads

### Standard Classification

```{eval-rst}
.. autoclass:: med_core.heads.classification.ClassificationHead
   :members:
   :undoc-members:
   :show-inheritance:
```

### Multi-Label Classification

```{eval-rst}
.. autoclass:: med_core.heads.classification.MultiLabelClassificationHead
   :members:
   :undoc-members:
   :show-inheritance:
```

### Ordinal Classification

```{eval-rst}
.. autoclass:: med_core.heads.classification.OrdinalClassificationHead
   :members:
   :undoc-members:
   :show-inheritance:
```

### Attention-Based Classification

```{eval-rst}
.. autoclass:: med_core.heads.classification.AttentionClassificationHead
   :members:
   :undoc-members:
   :show-inheritance:
```

### Ensemble Classification

```{eval-rst}
.. autoclass:: med_core.heads.classification.EnsembleClassificationHead
   :members:
   :undoc-members:
   :show-inheritance:
```

## Survival Analysis

```{eval-rst}
.. automodule:: med_core.heads.survival
   :members:
   :undoc-members:
   :show-inheritance:
```

## Usage Examples

### Basic Classification

```python
from med_core.heads.classification import ClassificationHead

# Create head
head = ClassificationHead(
    input_dim=512,
    num_classes=4,
    hidden_dims=[256, 128],
    dropout=0.5
)

# Forward pass
features = torch.randn(8, 512)
logits = head(features)  # [8, 4]
```

### Multi-Label Classification

```python
from med_core.heads.classification import MultiLabelClassificationHead

# Create head with independent classifiers
head = MultiLabelClassificationHead(
    input_dim=512,
    num_labels=5,
    use_independent_classifiers=True
)

# Forward pass
logits = head(features)  # [8, 5]
```

### Ordinal Classification

```python
from med_core.heads.classification import OrdinalClassificationHead

# Create head for tumor grading (4 grades)
head = OrdinalClassificationHead(
    input_dim=512,
    num_classes=4
)

# Get class probabilities
probs = head.predict_probabilities(features)  # [8, 4]
```

### Attention-Based Classification

```python
from med_core.heads.classification import AttentionClassificationHead

# Create head
head = AttentionClassificationHead(
    input_dim=512,
    num_classes=4,
    attention_dim=128
)

# Get predictions with attention weights
logits, attention = head(features, return_attention=True)
```

### Ensemble Classification

```python
from med_core.heads.classification import EnsembleClassificationHead

# Create ensemble of 3 heads
head = EnsembleClassificationHead(
    input_dim=512,
    num_classes=4,
    num_heads=3,
    aggregation='mean'
)

# Get aggregated predictions
logits = head(features)

# Get individual predictions
logits, individual = head(features, return_individual=True)
```
