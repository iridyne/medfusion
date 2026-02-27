# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

MedFusion is a highly modular medical multimodal deep learning research framework. It supports 29+ vision backbones and 5+ fusion strategies for combining medical imaging (CT, pathology, etc.) with tabular clinical data.

**Key Design Principles:**
- **Component Decoupling**: Backbones, fusion strategies, and aggregators are completely independent and swappable
- **Configuration-Driven**: All model architectures can be defined via YAML configs without code changes
- **Multi-View Support**: Handles multi-angle CT, time-series, multi-slice, and other complex medical data scenarios

## Development Commands

### Environment Setup
```bash
# Install dependencies (preferred)
uv sync

# Install with optional dependencies
uv sync --extra dev --extra web

# Alternative: pip install
pip install -e ".[dev,web]"
```

### Testing
```bash
# Run all tests
uv run pytest

# Run specific test file
uv run pytest tests/test_models.py

# Run specific test function
uv run pytest tests/test_models.py::test_model_builder

# Run with coverage
uv run pytest --cov=med_core --cov-report=html

# Run tests matching pattern
uv run pytest -k "fusion"
```

### Code Quality
```bash
# Lint and format check
ruff check med_core/

# Auto-fix issues
ruff check med_core/ --fix

# Type checking
mypy med_core/

# Format code
ruff format med_core/
```

### Training
```bash
# Train with default config
uv run medfusion-train --config configs/default.yaml

# Train with custom config
uv run medfusion-train --config configs/smurf_config.yaml

# Use CLI commands directly
uv run med-train --config configs/default.yaml
uv run med-evaluate --checkpoint path/to/checkpoint.pth
uv run med-preprocess --data-dir data/raw
```

### Web UI
```bash
# Start web server
./start-webui.sh

# Or manually
uv run python -m med_core.web.cli web

# Access at http://localhost:8000
```

## Architecture

### Core Component Hierarchy

```
Model = Backbones + Fusion + Head + (Optional) MIL Aggregators
```

**1. Backbones** (`med_core/backbones/`)
- Extract features from raw inputs
- Vision: ResNet, EfficientNet, ViT, Swin Transformer (2D/3D), DenseNet, etc.
- Tabular: MLP-based networks with batch norm and dropout
- Factory functions: `create_vision_backbone()`, `create_tabular_backbone()`

**2. Fusion Modules** (`med_core/fusion/`)
- Combine features from multiple modalities
- Strategies: Concatenate, Gated, Attention, Cross-Attention, Bilinear, Kronecker, Fused-Attention, Self-Attention
- Factory function: `create_fusion_module(fusion_type, ...)`

**3. Heads** (`med_core/heads/`)
- Task-specific output layers
- Classification: `ClassificationHead`
- Survival: `CoxSurvivalHead`, `DeepSurvivalHead`, `DiscreteTimeSurvivalHead`

**4. MIL Aggregators** (`med_core/aggregators/`)
- Aggregate multiple instances (e.g., WSI patches, multi-view images)
- Types: Mean, Max, Attention-based, Gated Attention

**5. Attention Supervision** (`med_core/attention_supervision/`)
- Specialized supervision for attention mechanisms
- CAM-based, Mask-based, MIL-based supervision

### Model Building Patterns

**Pattern 1: Using MultiModalModelBuilder (Recommended)**
```python
from med_core.models import MultiModalModelBuilder

builder = MultiModalModelBuilder(num_classes=2)
builder.add_modality("ct", backbone="swin3d_tiny", input_channels=1)
builder.add_modality("pathology", backbone="resnet50", pretrained=True)
builder.set_fusion("attention", hidden_dim=256)
builder.set_head("classification")
model = builder.build()
```

**Pattern 2: Using Factory Functions**
```python
from med_core.models import build_model_from_config

config = {...}  # Load from YAML
model = build_model_from_config(config)
```

**Pattern 3: Direct Construction**
```python
from med_core.models import GenericMultiModalModel
from med_core.backbones import create_vision_backbone
from med_core.fusion import create_fusion_module
from med_core.heads import ClassificationHead

backbones = {
    'ct': create_vision_backbone('swin3d_tiny', in_channels=1),
    'pathology': create_vision_backbone('resnet50', pretrained=True)
}
fusion = create_fusion_module('attention', input_dims=[768, 2048], output_dim=512)
head = ClassificationHead(input_dim=512, num_classes=2)
model = GenericMultiModalModel(backbones, fusion, head)
```

### Configuration System

Configs are in `configs/` directory. Key sections:

- **data**: Dataset paths, features, augmentation, data loaders
- **model**: Architecture (vision/tabular backbones, fusion, heads)
- **training**: Epochs, optimizer, scheduler, mixed precision, progressive training
- **logging**: TensorBoard, Weights & Biases, output directories

**Important**: The config system uses `med_core/configs/base_config.py` for validation. All configs must conform to the schema defined there.

**Validation System**: The `ConfigValidator` in `med_core/configs/validation.py` provides structured error reporting with:
- Error codes (E001-E028) for different validation failures
- Detailed error messages with context paths
- Actionable suggestions for fixing issues
- Use `validate_config_or_exit()` to validate configs before training

### Dataset System

**Base Classes:**
- `MedicalDataset`: Single-modality medical imaging
- `MultiViewDataset`: Multi-view/multi-instance data (e.g., multiple CT slices)
- `AttentionSupervisedDataset`: Datasets with attention supervision signals

**Key Features:**
- Automatic caching with `DatasetCache` for faster loading
- Flexible column mapping via config
- Built-in train/val/test splitting
- Comprehensive augmentation pipeline

### Training System

**Trainers** (`med_core/trainers/`):
- `BaseTrainer`: Foundation class with common training logic
- `MultimodalTrainer`: For vision + tabular fusion models
- `MultiViewTrainer`: For multi-view/multi-instance scenarios

**Key Features:**
- Mixed precision training (AMP)
- Progressive training (stage-wise unfreezing)
- Differential learning rates per component
- Early stopping and checkpointing
- TensorBoard/WandB logging

### Web UI System

**Architecture** (`med_core/web/`):
- FastAPI backend with WebSocket support
- RESTful APIs for training, evaluation, model management
- Real-time training progress via WebSocket
- Workflow editor for complex pipelines
- React frontend (in `web/frontend/`)

**Key Endpoints:**
- `/api/training/*`: Training job management
- `/api/models/*`: Model registry and checkpoints
- `/api/datasets/*`: Dataset management
- `/api/system/*`: System info (GPU, disk, etc.)
- `/api/experiments/*`: Experiment tracking

## Important Implementation Notes

### Type Annotations
- This project uses **modern Python type hints** (PEP 585/604)
- Use `dict`/`list` instead of `Dict`/`List`
- Use `X | None` instead of `Optional[X]`
- Use `X | Y` instead of `Union[X, Y]`
- All functions must have complete type annotations (enforced by mypy)

### Code Quality Standards
- Line length: 88 characters (Black-compatible)
- Target Python: 3.11+
- Linting: ruff with E, W, F, I, B, C4, UP rules (currently 0 errors)
- Type checking: mypy with strict settings (see pyproject.toml)
- Current type coverage: ~488 type errors remaining (down from 678)

### Common Patterns

**Registry Pattern**: Many components use factory registries
```python
from med_core.backbones import BACKBONE_REGISTRY

@BACKBONE_REGISTRY.register("my_backbone")
def create_my_backbone(**kwargs):
    return MyBackbone(**kwargs)
```

**Config-Driven Construction**: Always support building from config
```python
def create_from_config(config: dict) -> nn.Module:
    backbone_type = config["backbone"]
    return BACKBONE_REGISTRY.get(backbone_type)(**config)
```

**Auxiliary Outputs**: Models often return `(main_output, aux_dict)`
```python
logits, aux = model(inputs)
# aux may contain: attention_maps, intermediate_features, etc.
```

### Testing Guidelines

- Use fixtures from `tests/conftest.py` for common test data
- Test files follow `test_*.py` naming convention
- Integration tests are in `test_*_integration.py`
- End-to-end tests are in `test_end_to_end.py`
- Mock external dependencies (file I/O, network calls)

**Test Coverage:**
- Total: 699 tests collected
- Core modules: 251+ tests passing
- Known issues: `test_export.py` requires `onnxscript` dependency
- Run with: `uv run pytest -q` for minimal output

### Performance Optimization

**Quick Tips:**
- Use `torch.cuda.amp` for mixed precision training
- Enable `pin_memory=True` for data loaders on GPU
- Use `num_workers > 0` for parallel data loading
- Cache preprocessed data with `DatasetCache`
- Profile with `torch.profiler` for bottlenecks

**When facing performance issues, follow this priority order:**

**1. Algorithm-Level Optimization (Highest Priority)**
- Mixed precision training (already supported via `training.mixed_precision: true`)
- Gradient accumulation to reduce memory usage
- Model pruning and quantization
- More efficient data augmentation strategies
- Optimize model architecture (reduce parameters, use efficient blocks)

**2. Engineering-Level Optimization**
- Data caching (use `DatasetCache` for preprocessed data)
- Precompute and save features when possible
- Use faster data formats (HDF5, LMDB instead of individual files)
- Optimize DataLoader `num_workers` (typically 4-8 per GPU)
- Enable persistent workers: `persistent_workers=True`
- Use faster image loading libraries (e.g., `pillow-simd`, `opencv`)

**3. Infrastructure Optimization**
- Use better GPUs (A100 > V100 > RTX 3090)
- Distributed training (multi-GPU via DDP, multi-node)
- Use NVMe SSDs for faster I/O
- Increase system RAM to cache more data
- Use faster network for distributed training

**4. Model Deployment Optimization**
- TorchScript compilation: `torch.jit.script(model)`
- ONNX export for cross-platform inference
- TensorRT for NVIDIA GPU inference acceleration
- Model quantization (INT8/FP16) for faster inference

**5. Custom Kernel Optimization (Last Resort)**
- Write custom CUDA kernels with Triton
- Use PyTorch C++ extensions for critical paths
- Consider Rust only if:
  - Profiling shows a specific Python function is the bottleneck (>20% time)
  - The function is pure CPU computation (not PyTorch ops)
  - No existing C++ library can replace it
  - You have Rust expertise in the team

**Important Notes:**
- **Do NOT migrate to Rust prematurely**: PyTorch core is already C++/CUDA optimized
- **Profile first**: Use `torch.profiler` or `cProfile` to identify actual bottlenecks
- **Most bottlenecks are I/O or GPU utilization**, not Python overhead
- **Rust migration has high cost**: Maintenance burden, ecosystem immaturity, limited benefit

**Common Bottlenecks and Solutions:**
- **Slow data loading**: Increase `num_workers`, use data caching, faster storage
- **Low GPU utilization**: Increase batch size, optimize DataLoader, check CPU preprocessing
- **Out of memory**: Use gradient accumulation, mixed precision, smaller batch size
- **Long training time**: Distributed training, better GPU, model architecture optimization

## Project-Specific Conventions

### Naming Conventions
- Backbones: `{architecture}_{variant}` (e.g., `swin3d_tiny`, `resnet50`)
- Fusion: `{strategy}` (e.g., `attention`, `gated`, `bilinear`)
- Configs: `{model}_{variant}_config.yaml` (e.g., `smurf_mil_config.yaml`)

### Module Organization
- Each major component has `__init__.py` with `__all__` exports
- Factory functions are named `create_*` (e.g., `create_fusion_module`)
- Base classes are in `base.py` files
- Strategies/implementations are in separate files

### Error Handling
- Use custom exceptions from `med_core/exceptions.py`
- Provide detailed error messages with context
- Validate configs early (in `__post_init__` or constructors)

### Documentation
- All public APIs have docstrings with Args/Returns/Example sections
- Complex algorithms have inline comments explaining the logic
- Config options are documented in example YAML files

## Common Tasks

### Adding a New Backbone
1. Implement in `med_core/backbones/`
2. Register with `BACKBONE_REGISTRY`
3. Add factory function if needed
4. Update `__init__.py` exports
5. Add tests in `tests/test_backbones.py`

### Adding a New Fusion Strategy
1. Implement in `med_core/fusion/strategies.py` or new file
2. Register with fusion factory
3. Update `create_fusion_module()` dispatcher
4. Add tests in `tests/test_fusion.py`

### Adding a New Dataset
1. Inherit from `MedicalDataset` or `MultiViewDataset`
2. Implement `__getitem__` and `__len__`
3. Add to `med_core/datasets/__init__.py`
4. Add tests in `tests/test_datasets.py`

### Modifying Training Logic
1. Override methods in `BaseTrainer` subclass
2. Key methods: `training_step`, `validation_step`, `configure_optimizers`
3. Update config schema if adding new options
4. Add tests in `tests/test_trainers.py`
