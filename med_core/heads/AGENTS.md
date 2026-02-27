# Heads Module

<!-- Parent: ../AGENTS.md -->

**Last Updated:** 2025-02-27

## Purpose

Task-specific output heads for medical imaging models, supporting classification and survival analysis tasks.

## Key Components

### Classification Heads (`classification.py`)

1. **ClassificationHead**
   - Standard classification head with linear layer
   - Supports binary and multi-class classification

2. **MultiLabelClassificationHead**
   - Multi-label classification (multiple diseases)
   - Uses sigmoid activation for independent predictions

3. **OrdinalClassificationHead**
   - Ordinal classification (ordered categories)
   - Preserves ordinal relationships (e.g., disease severity)

4. **AttentionClassificationHead**
   - Classification with attention mechanism
   - Returns attention weights for interpretability

5. **EnsembleClassificationHead**
   - Ensemble of multiple classifiers
   - Combines predictions from multiple heads

### Survival Analysis Heads (`survival.py`)

1. **CoxSurvivalHead**
   - Cox proportional hazards model
   - Predicts hazard ratios

2. **DiscreteTimeSurvivalHead**
   - Discrete-time survival analysis
   - Predicts survival probabilities at discrete time points

3. **DeepSurvivalHead**
   - Deep learning-based survival analysis
   - Flexible neural network for survival prediction

4. **MultiTaskSurvivalHead**
   - Multi-task learning for survival
   - Jointly predicts survival and auxiliary tasks

5. **RankingSurvivalHead**
   - Ranking-based survival analysis
   - Uses ranking loss for survival prediction

## Architecture

```
Features → Head → Task-Specific Output
[B, D] → [B, num_classes] (Classification)
[B, D] → [B, num_time_points] (Survival)
```

## Usage Patterns

### Classification
```python
from med_core.heads import ClassificationHead

head = ClassificationHead(
    input_dim=512,
    num_classes=2,
    dropout=0.5
)
logits = head(features)  # [B, 2]
```

### Multi-Label Classification
```python
from med_core.heads import MultiLabelClassificationHead

head = MultiLabelClassificationHead(
    input_dim=512,
    num_classes=5,  # 5 diseases
    dropout=0.5
)
probabilities = head(features)  # [B, 5], each in [0, 1]
```

### Ordinal Classification
```python
from med_core.heads import OrdinalClassificationHead

head = OrdinalClassificationHead(
    input_dim=512,
    num_classes=4,  # 4 severity levels
    dropout=0.5
)
logits = head(features)  # [B, 4]
```

### Cox Survival
```python
from med_core.heads import CoxSurvivalHead

head = CoxSurvivalHead(
    input_dim=512,
    hidden_dims=[256, 128]
)
hazard_ratios = head(features)  # [B, 1]
```

### Discrete-Time Survival
```python
from med_core.heads import DiscreteTimeSurvivalHead

head = DiscreteTimeSurvivalHead(
    input_dim=512,
    num_time_points=10,
    hidden_dims=[256, 128]
)
survival_probs = head(features)  # [B, 10]
```

## Key Files

- `classification.py`: All classification head implementations
- `survival.py`: All survival analysis head implementations
- `__init__.py`: Public API exports

## Dependencies

- PyTorch (nn.Module, nn.Linear)
- Used by: `med_core.models`, `med_core.trainers`

## Testing

Run tests with:
```bash
uv run pytest tests/test_heads.py -v
```

## Common Issues

1. **Input dimension mismatch**: Ensure input_dim matches fusion output dimension
2. **Number of classes**: Binary classification uses num_classes=2, not 1
3. **Dropout rate**: Too high dropout (>0.7) may hurt performance
4. **Survival time points**: Choose appropriate discretization for survival tasks

## Loss Functions

### Classification
- Binary: `nn.BCEWithLogitsLoss` or `nn.CrossEntropyLoss`
- Multi-class: `nn.CrossEntropyLoss`
- Multi-label: `nn.BCEWithLogitsLoss`
- Ordinal: Custom ordinal loss

### Survival
- Cox: Negative log partial likelihood
- Discrete-time: `nn.BCEWithLogitsLoss` per time point
- Ranking: Ranking loss (e.g., margin ranking loss)

## Related Modules

- `models/`: Combines heads with backbones and fusion
- `trainers/`: Uses heads for training
- `evaluation/`: Evaluates head predictions
