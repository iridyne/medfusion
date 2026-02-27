# CLI Module

<!-- Parent: ../AGENTS.md -->

**Last Updated:** 2025-02-27

## Purpose

Command-line interfaces for training, evaluation, and preprocessing in the MedFusion framework.

## Key Components

### Entry Points

1. **Training** (`train.py`)
   - Command: `med-train` or `medfusion-train`
   - Trains multimodal models from config files
   - Supports multi-GPU training, mixed precision, progressive training

2. **Evaluation** (`evaluate.py`)
   - Command: `med-evaluate` or `medfusion-evaluate`
   - Evaluates trained models on test/validation sets
   - Generates comprehensive evaluation reports

3. **Preprocessing** (`preprocess.py`)
   - Command: `med-preprocess` or `medfusion-preprocess`
   - Preprocesses medical images (normalization, resizing, augmentation)
   - Batch processing with parallel workers

## Architecture

```
CLI Command → Argument Parser → Config Loader → Main Function → Output
```

## Usage Patterns

### Training
```bash
# Basic training
uv run med-train --config configs/default.yaml

# With custom output directory
uv run med-train --config configs/smurf_config.yaml --output-dir outputs/experiment1

# Resume from checkpoint
uv run med-train --config configs/default.yaml --resume outputs/checkpoints/last.pth

# Multi-GPU training
uv run med-train --config configs/default.yaml --gpus 0,1,2,3
```

### Evaluation
```bash
# Evaluate on test set
uv run med-evaluate --config configs/default.yaml \
    --checkpoint outputs/checkpoints/best.pth \
    --split test

# Generate detailed report
uv run med-evaluate --config configs/default.yaml \
    --checkpoint outputs/checkpoints/best.pth \
    --split test \
    --output-dir outputs/evaluation
```

### Preprocessing
```bash
# Preprocess images
uv run med-preprocess \
    --input-dir data/raw_images \
    --output-dir data/processed_images \
    --size 224 \
    --normalize

# With parallel workers
uv run med-preprocess \
    --input-dir data/raw_images \
    --output-dir data/processed_images \
    --workers 8
```

## Key Files

- `train.py`: Training CLI implementation
- `evaluate.py`: Evaluation CLI implementation
- `preprocess.py`: Preprocessing CLI implementation
- `__init__.py`: Public API exports

## Dependencies

- argparse (CLI argument parsing)
- `med_core.configs`: Configuration loading
- `med_core.trainers`: Training logic
- `med_core.evaluation`: Evaluation metrics
- `med_core.preprocessing`: Image preprocessing

## Testing

Run tests with:
```bash
uv run pytest tests/test_cli.py -v
```

## Common Issues

1. **Config file not found**: Ensure config path is correct and file exists
2. **Checkpoint loading**: Verify checkpoint path and model architecture match
3. **GPU availability**: Check CUDA availability with `torch.cuda.is_available()`
4. **Memory errors**: Reduce batch size in config or use gradient accumulation

## Configuration

All CLI commands use YAML configuration files. Key sections:

```yaml
data:
  train_csv: data/train.csv
  val_csv: data/val.csv
  test_csv: data/test.csv

model:
  vision:
    backbone: resnet50
    pretrained: true
  fusion:
    type: attention

training:
  epochs: 100
  batch_size: 32
  learning_rate: 0.001
```

## Related Modules

- `configs/`: Configuration system
- `trainers/`: Training logic
- `evaluation/`: Evaluation metrics and reports
- `preprocessing/`: Image preprocessing utilities
