# Classification Heads

Classification heads for various medical imaging tasks.

## Overview

The heads module provides different classification head implementations for
multi-class, multi-label, ordinal, and attention-based classification tasks.

## Available Heads

### Standard Classification

**ClassificationHead** - Standard multi-class classification head with configurable hidden layers and dropout.

### Multi-Label Classification

**MultiLabelClassificationHead** - Multi-label classification with independent classifiers for each label.

### Ordinal Classification

**OrdinalClassificationHead** - Ordinal classification for ordered categories (e.g., tumor grading).

### Attention-Based Classification

**AttentionClassificationHead** - Classification with attention weights for interpretability.

### Ensemble Classification

**EnsembleClassificationHead** - Ensemble of multiple classification heads with aggregation.

## Survival Analysis

The survival module provides heads for survival analysis tasks including Cox regression, DeepSurvival, and discrete-time survival models.

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
