# trainers/ - Training Module

<!-- Parent: ../AGENTS.md -->

**Last Updated:** 2025-02-27

## Purpose

Provides training infrastructure for medical deep learning models. Implements flexible training loops with support for multi-modal fusion, multi-view aggregation, mixed precision, progressive training, and comprehensive logging.

## Key Components

### Core Classes

- **BaseTrainer**: Abstract base class with common training logic
- **MultimodalTrainer**: Trainer for vision + tabular fusion models
- **MultiViewMultimodalTrainer**: Trainer for multi-view/multi-instance scenarios

### Factory Functions

- `create_trainer()`: Create MultimodalTrainer from config
- `create_multiview_trainer()`: Create MultiViewMultimodalTrainer from config

## Architecture

```
BaseTrainer (Abstract):
  ├── model: nn.Module
  ├── optimizer: torch.optim.Optimizer
  ├── scheduler: Optional[LRScheduler]
  ├── train_loader: DataLoader
  ├── val_loader: DataLoader
  ├── training_step() -> loss
  ├── validation_step() -> metrics
  └── train() -> None

MultimodalTrainer(BaseTrainer):
  ├── Handles vision + tabular inputs
  ├── Mixed precision training (AMP)
  ├── Progressive training support
  └── TensorBoard/WandB logging

MultiViewMultimodalTrainer(MultimodalTrainer):
  ├── Multi-view/multi-instance support
  ├── MIL aggregation
  └── Attention supervision
```

## Usage Patterns

### Basic Training

```python
from med_core.trainers import MultimodalTrainer
from med_core.models import build_model_from_config

# Build model
model = build_model_from_config('configs/smurf_config.yaml')

# Create trainer
trainer = MultimodalTrainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    optimizer=optimizer,
    device='cuda',
    num_epochs=100
)

# Train
trainer.train()
```

### Config-Driven Training

```python
from med_core.trainers import create_trainer
import yaml

# Load config
with open('configs/smurf_config.yaml') as f:
    config = yaml.safe_load(f)

# Create trainer from config
trainer = create_trainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    config=config
)

# Train
trainer.train()
```

### Multi-View Training

```python
from med_core.trainers import create_multiview_trainer

# Create multi-view trainer
trainer = create_multiview_trainer(
    model=model,
    train_loader=multiview_train_loader,
    val_loader=multiview_val_loader,
    config=config
)

# Train with multi-view data
trainer.train()
```

### Progressive Training

```python
from med_core.trainers import MultimodalTrainer

trainer = MultimodalTrainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    optimizer=optimizer,
    progressive_training=True,
    progressive_stages=[
        {'epochs': 10, 'freeze': ['backbone']},
        {'epochs': 20, 'freeze': []},
    ]
)

trainer.train()
```

## Key Features

1. **Mixed Precision**: Automatic mixed precision (AMP) for faster training
2. **Progressive Training**: Stage-wise unfreezing of model components
3. **Differential LR**: Different learning rates per component
4. **Early Stopping**: Automatic early stopping with patience
5. **Checkpointing**: Save best and latest checkpoints
6. **Logging**: TensorBoard and Weights & Biases integration
7. **Gradient Clipping**: Prevent gradient explosion
8. **Learning Rate Scheduling**: Cosine, step, plateau schedulers

## Training Configuration

### Config Schema

```yaml
training:
  epochs: 100
  batch_size: 32
  learning_rate: 0.001
  optimizer: adam
  scheduler: cosine
  mixed_precision: true
  gradient_clip: 1.0
  early_stopping:
    patience: 10
    min_delta: 0.001
  progressive_training:
    enabled: true
    stages:
      - epochs: 10
        freeze: [backbone]
      - epochs: 20
        freeze: []
  logging:
    tensorboard: true
    wandb: false
    log_interval: 10
```

## Integration Points

### Upstream Dependencies

- `med_core.models`: Model architectures
- `med_core.datasets`: Data loaders
- `torch.optim`: Optimizers
- `torch.cuda.amp`: Mixed precision
- `tensorboard`: Logging

### Downstream Consumers

- `med_core.cli`: CLI training commands
- `med_core.web`: Web UI training jobs
- Training scripts in `examples/`

## File Structure

```
trainers/
├── __init__.py              # Public API exports
├── base.py                 # BaseTrainer abstract class
├── multimodal.py           # MultimodalTrainer
└── multiview_trainer.py    # MultiViewMultimodalTrainer
```

## Testing

```bash
# Test base trainer
uv run pytest tests/test_trainers.py::test_base_trainer

# Test multimodal trainer
uv run pytest tests/test_trainers.py::test_multimodal_trainer

# Test multi-view trainer
uv run pytest tests/test_trainers.py::test_multiview_trainer

# Test progressive training
uv run pytest tests/test_trainers.py::test_progressive_training
```

## Common Tasks

### Custom Training Step

```python
from med_core.trainers import BaseTrainer

class CustomTrainer(BaseTrainer):
    def training_step(self, batch):
        inputs, targets = batch
        outputs = self.model(inputs)
        loss = self.criterion(outputs, targets)

        # Add custom loss terms
        custom_loss = compute_custom_loss(outputs)
        total_loss = loss + 0.1 * custom_loss

        return total_loss
```

### Custom Validation Metrics

```python
class CustomTrainer(MultimodalTrainer):
    def validation_step(self, batch):
        metrics = super().validation_step(batch)

        # Add custom metrics
        metrics['custom_metric'] = compute_custom_metric(outputs, targets)

        return metrics
```

### Custom Callbacks

```python
class CustomTrainer(MultimodalTrainer):
    def on_epoch_end(self, epoch, train_loss, val_metrics):
        super().on_epoch_end(epoch, train_loss, val_metrics)

        # Custom callback logic
        if val_metrics['accuracy'] > 0.95:
            print("High accuracy achieved!")
```

## Performance Optimization

### Mixed Precision Training

```python
trainer = MultimodalTrainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    optimizer=optimizer,
    mixed_precision=True  # Enable AMP
)
```

### Gradient Accumulation

```python
trainer = MultimodalTrainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    optimizer=optimizer,
    gradient_accumulation_steps=4  # Accumulate over 4 batches
)
```

### DataLoader Optimization

```python
train_loader = DataLoader(
    dataset,
    batch_size=32,
    num_workers=4,  # Parallel data loading
    pin_memory=True,  # Faster GPU transfer
    persistent_workers=True  # Keep workers alive
)
```

## Best Practices

1. **Start Small**: Test with small dataset first
2. **Monitor Metrics**: Watch for overfitting
3. **Save Checkpoints**: Regular checkpointing
4. **Log Everything**: Comprehensive logging
5. **Validate Config**: Check config before training
6. **Use Mixed Precision**: Faster training on modern GPUs
7. **Progressive Training**: Stabilize training for complex models

## Related Documentation

- Model building: `med_core/models/AGENTS.md`
- Dataset loading: `med_core/datasets/AGENTS.md`
- Configuration: `med_core/configs/AGENTS.md`
- CLI training: `CLAUDE.md` → Development Commands
