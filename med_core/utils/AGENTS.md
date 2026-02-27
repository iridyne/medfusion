# utils/ - Utilities Module

<!-- Parent: ../AGENTS.md -->

**Last Updated:** 2025-02-27

## Purpose

Provides common utility functions and helper modules used across the MedFusion framework. Contains essential tools for device management, checkpointing, logging, seeding, gradient checkpointing, and performance optimization.

## Key Components

### Core Utilities

- **Seed**: Reproducibility via `set_seed()`
- **Device**: GPU/CPU management and device info
- **Logging**: Structured logging setup
- **Checkpoint**: Model checkpoint save/load/management
- **Gradient Checkpointing**: Memory optimization for large models
- **AMP**: Automatic mixed precision utilities
- **Distributed**: Multi-GPU training support
- **Export**: Model export (ONNX, TorchScript)
- **Compression**: Model quantization and pruning
- **Benchmark**: Performance profiling

## Architecture

```
utils/
├── seed.py                    # set_seed()
├── device.py                  # get_device(), move_to_device()
├── logging.py                 # setup_logging(), get_logger()
├── checkpoint.py              # save/load/find/cleanup checkpoints
├── gradient_checkpointing.py  # Memory optimization
├── amp.py                     # Mixed precision utilities
├── distributed.py             # Multi-GPU support
├── export.py                  # Model export
├── compression.py             # Quantization/pruning
├── benchmark.py               # Performance profiling
└── tuning.py                  # Hyperparameter tuning
```

## Usage Patterns

### Reproducibility

```python
from med_core.utils import set_seed

# Set seed for reproducibility
set_seed(42)
```

### Device Management

```python
from med_core.utils import get_device, move_to_device

# Get best available device
device = get_device()  # Returns 'cuda' if available, else 'cpu'

# Move model and data to device
model = move_to_device(model, device)
data = move_to_device(data, device)

# Get device info
from med_core.utils import get_device_info
info = get_device_info()
print(f"GPU: {info['name']}, Memory: {info['memory_total']} GB")
```

### Logging

```python
from med_core.utils import setup_logging, get_logger

# Setup logging
setup_logging(level='INFO', log_file='training.log')

# Get logger
logger = get_logger(__name__)
logger.info("Training started")
logger.warning("Low GPU memory")
logger.error("Training failed")
```

### Checkpointing

```python
from med_core.utils import save_checkpoint, load_checkpoint, find_best_checkpoint

# Save checkpoint
save_checkpoint(
    model=model,
    optimizer=optimizer,
    epoch=10,
    metrics={'val_loss': 0.5, 'val_acc': 0.9},
    save_path='checkpoints/epoch_10.pth'
)

# Load checkpoint
checkpoint = load_checkpoint('checkpoints/epoch_10.pth', model, optimizer)
start_epoch = checkpoint['epoch']
metrics = checkpoint['metrics']

# Find best checkpoint
best_path = find_best_checkpoint('checkpoints/', metric='val_acc', mode='max')

# Cleanup old checkpoints
from med_core.utils import cleanup_checkpoints
cleanup_checkpoints('checkpoints/', keep_best=5, metric='val_acc')
```

### Gradient Checkpointing

```python
from med_core.utils import apply_gradient_checkpointing, estimate_memory_savings

# Apply gradient checkpointing to model
apply_gradient_checkpointing(model, checkpoint_ratio=0.5)

# Estimate memory savings
savings = estimate_memory_savings(model, checkpoint_ratio=0.5)
print(f"Estimated memory savings: {savings['memory_saved_gb']:.2f} GB")
```

### Mixed Precision

```python
from med_core.utils.amp import create_scaler, autocast

# Create gradient scaler
scaler = create_scaler()

# Training loop with AMP
for batch in train_loader:
    with autocast():
        outputs = model(inputs)
        loss = criterion(outputs, targets)

    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

### Model Export

```python
from med_core.utils.export import export_to_onnx, export_to_torchscript

# Export to ONNX
export_to_onnx(
    model=model,
    dummy_input=dummy_input,
    output_path='model.onnx',
    opset_version=14
)

# Export to TorchScript
export_to_torchscript(
    model=model,
    example_input=example_input,
    output_path='model.pt'
)
```

## Key Features

1. **Reproducibility**: Seed management for deterministic training
2. **Device Abstraction**: Unified GPU/CPU interface
3. **Checkpoint Management**: Automatic save/load/cleanup
4. **Memory Optimization**: Gradient checkpointing for large models
5. **Performance**: Mixed precision and distributed training
6. **Export**: ONNX and TorchScript export
7. **Profiling**: Built-in performance benchmarking

## Integration Points

### Upstream Dependencies

- `torch`: Core PyTorch utilities
- `numpy`: Random seed management
- `logging`: Python logging
- `pathlib`: Path handling

### Downstream Consumers

- `med_core.trainers`: Training loops use all utilities
- `med_core.models`: Model building uses device management
- `med_core.datasets`: Data loading uses device utilities
- `med_core.web`: Web UI uses logging and checkpointing
- All modules use logging and device management

## File Structure

```
utils/
├── __init__.py                    # Public API exports
├── seed.py                       # Reproducibility
├── device.py                     # Device management
├── logging.py                    # Logging setup
├── checkpoint.py                 # Checkpoint utilities
├── gradient_checkpointing.py     # Memory optimization
├── amp.py                        # Mixed precision
├── distributed.py                # Multi-GPU support
├── export.py                     # Model export
├── compression.py                # Quantization/pruning
├── benchmark.py                  # Performance profiling
└── tuning.py                     # Hyperparameter tuning
```

## Testing

```bash
# Test seed utilities
uv run pytest tests/test_utils.py::test_set_seed

# Test device utilities
uv run pytest tests/test_utils.py::test_device_management

# Test checkpoint utilities
uv run pytest tests/test_utils.py::test_checkpoint_save_load

# Test gradient checkpointing
uv run pytest tests/test_utils.py::test_gradient_checkpointing
```

## Common Tasks

### Custom Checkpoint Format

```python
from med_core.utils import save_checkpoint

# Save with custom metadata
save_checkpoint(
    model=model,
    optimizer=optimizer,
    epoch=10,
    metrics={'val_loss': 0.5},
    save_path='checkpoint.pth',
    custom_data={
        'config': config,
        'dataset_info': dataset_info,
        'git_commit': git_commit_hash
    }
)
```

### Distributed Training Setup

```python
from med_core.utils.distributed import setup_distributed, cleanup_distributed

# Setup distributed training
rank, world_size = setup_distributed()

# Training code
model = DistributedDataParallel(model)
# ...

# Cleanup
cleanup_distributed()
```

### Performance Profiling

```python
from med_core.utils.benchmark import profile_model

# Profile model
stats = profile_model(
    model=model,
    input_shape=(1, 3, 224, 224),
    num_iterations=100
)

print(f"Average latency: {stats['avg_latency_ms']:.2f} ms")
print(f"Throughput: {stats['throughput_samples_per_sec']:.2f} samples/s")
```

## Performance Notes

- **Gradient Checkpointing**: Reduces memory by ~50% with ~20% slowdown
- **Mixed Precision**: 2-3x speedup on modern GPUs (V100, A100)
- **Distributed Training**: Near-linear scaling up to 8 GPUs
- **Checkpoint I/O**: Use SSD for faster checkpoint save/load

## Best Practices

1. **Always Set Seed**: Use `set_seed()` at start of training
2. **Device Abstraction**: Use `get_device()` instead of hardcoding 'cuda'
3. **Structured Logging**: Use `get_logger()` instead of print()
4. **Regular Checkpoints**: Save checkpoints every N epochs
5. **Cleanup Old Checkpoints**: Use `cleanup_checkpoints()` to save disk space
6. **Profile First**: Use benchmark utilities before optimization
7. **Export for Production**: Export to ONNX/TorchScript for deployment

## Related Documentation

- Training: `med_core/trainers/AGENTS.md`
- Model building: `med_core/models/AGENTS.md`
- Configuration: `med_core/configs/AGENTS.md`
- Deployment: `med_core/serving/AGENTS.md`
