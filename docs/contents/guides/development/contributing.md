# Contributing Guide

Thank you for your interest in contributing to MedFusion! This guide will help you get started.

## Development Environment Setup

### Prerequisites

- Python 3.11 or higher
- Git
- CUDA-capable GPU (optional, for training)

### Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/iridyne/medfusion.git
   cd medfusion
   ```

2. **Install uv** (if not already installed):
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

3. **Install dependencies**:
   ```bash
   uv sync --extra dev --extra web
   ```

4. **Verify installation**:
   ```bash
   bash test/smoke.sh
   ```

### Frontend Setup (Optional)

If you're working on the Web UI:

```bash
cd web/frontend
npm install
npm run dev
```

## Development Workflow

### 1. Create a Branch

```bash
git checkout -b feature/your-feature-name
```

Branch naming conventions:
- `feature/` - New features
- `fix/` - Bug fixes
- `docs/` - Documentation changes
- `refactor/` - Code refactoring
- `test/` - Test additions or modifications

### 2. Make Changes

Follow the project's coding standards (see below).

### 3. Local Preflight

```bash
# Fast local preflight
bash scripts/full_regression.sh --quick

# Local smoke / CI handoff
bash scripts/full_regression.sh --ci

# Broader local non-pytest validation
bash scripts/full_regression.sh --full
```

`pytest` runs in GitHub Actions CI. When CI fails, inspect the Actions logs or run:

```bash
bash scripts/inspect_ci_failure.sh
```

### 4. Format and Lint

```bash
# Format code
ruff format .

# Lint and auto-fix
ruff check . --fix

# Type check
mypy med_core/
```

### 5. Commit Changes

Follow conventional commit format:

```bash
git commit -m "feat: add new fusion strategy"
git commit -m "fix: resolve memory leak in data loader"
git commit -m "docs: update API documentation"
```

Commit message format:
- `feat:` - New feature
- `fix:` - Bug fix
- `docs:` - Documentation changes
- `style:` - Code style changes (formatting, etc.)
- `refactor:` - Code refactoring
- `test:` - Test additions or modifications
- `chore:` - Maintenance tasks

### 6. Push and Create PR

```bash
git push -u origin feature/your-feature-name
```

Then create a pull request on GitHub.

## Coding Standards

### Python Style

- **Line length**: 88 characters (Black-compatible)
- **Imports**: Sorted with `isort` (handled by ruff)
- **Type hints**: Required for all functions
- **Docstrings**: Google-style format (see [Documentation Standards](documentation-standards.md))

### Code Quality Rules

1. **Use modern type hints** (Python 3.11+):
   ```python
   # Good
   def process(items: list[str]) -> dict[str, int]:
       ...

   # Bad
   from typing import List, Dict
   def process(items: List[str]) -> Dict[str, int]:
       ...
   ```

2. **Keep functions focused**:
   - One function, one responsibility
   - Target: < 50 lines per function
   - Hard limit: 100 lines per function

3. **Keep files manageable**:
   - Target: < 300 lines per file
   - Hard limit: 500 lines per file
   - Split large files into logical modules

4. **Avoid over-engineering**:
   - Don't add features not requested
   - Don't refactor unrelated code
   - Keep solutions simple and focused

5. **Write secure code**:
   - Validate user inputs
   - Avoid SQL injection, XSS, command injection
   - Don't commit secrets or credentials

### Testing Guidelines

1. **Write tests for new features**:
   ```python
   def test_attention_fusion():
       fusion = create_fusion_module('attention', [512, 768], 256)
       inputs = [torch.randn(32, 512), torch.randn(32, 768)]
       output = fusion(inputs)
       assert output.shape == (32, 256)
   ```

2. **Use fixtures** from `tests/conftest.py`:
   ```python
   def test_model_training(sample_config, sample_dataloader):
       model = build_model_from_config(sample_config)
       trainer = MultimodalTrainer(model, sample_dataloader)
       trainer.train(epochs=1)
   ```

3. **Mark test types**:
   ```python
   @pytest.mark.unit
   def test_fusion_module():
       ...

   @pytest.mark.integration
   def test_end_to_end_training():
       ...
   ```

4. **Mock external dependencies**:
   ```python
   @patch('med_core.utils.io.load_image')
   def test_dataset_loading(mock_load):
       mock_load.return_value = np.zeros((224, 224, 3))
       dataset = MedicalDataset('data.csv')
       assert len(dataset) > 0
   ```

## Project Structure

Understanding the project structure helps you know where to add new code:

```
med_core/
├── models/              # Model architectures
│   ├── backbones/       # Vision and tabular backbones
│   ├── fusion/          # Fusion strategies
│   ├── heads/           # Task-specific heads
│   └── builder.py       # Model builder
├── datasets/            # Data loaders
├── trainers/            # Training logic
├── configs/             # Configuration system
├── preprocessing/       # Data preprocessing
├── evaluation/          # Evaluation and metrics
├── aggregators/         # Multi-instance aggregators
├── attention_supervision/  # Attention supervision
├── web/                 # Web API (FastAPI)
└── utils/               # Utilities
```

## Adding New Components

### Adding a New Backbone

1. **Implement the backbone** in `med_core/backbones/`:
   ```python
   # med_core/backbones/my_backbone.py
   import torch.nn as nn

   class MyBackbone(nn.Module):
       """My custom backbone."""

       def __init__(self, in_channels: int = 3, **kwargs):
           super().__init__()
           # Implementation

       def forward(self, x):
           # Forward pass
           return features
   ```

2. **Register the backbone**:
   ```python
   from med_core.backbones import BACKBONE_REGISTRY

   @BACKBONE_REGISTRY.register("my_backbone")
   def create_my_backbone(**kwargs):
       return MyBackbone(**kwargs)
   ```

3. **Add tests**:
   ```python
   # tests/test_backbones.py
   def test_my_backbone():
       backbone = create_vision_backbone('my_backbone', in_channels=3)
       x = torch.randn(2, 3, 224, 224)
       out = backbone(x)
       assert out.shape[0] == 2
   ```

4. **Update documentation** in `docs/contents/api/backbones.md`

### Adding a New Fusion Strategy

1. **Implement the fusion module** in `med_core/fusion/strategies.py`
2. **Update the factory** in `med_core/fusion/__init__.py`
3. **Add tests** in `tests/test_fusion.py`
4. **Update documentation** in `docs/contents/api/fusion.md`

### Adding a New Dataset

1. **Inherit from base class** in `med_core/datasets/`
2. **Implement `__getitem__` and `__len__`**
3. **Add to exports** in `med_core/datasets/__init__.py`
4. **Add tests** in `tests/test_datasets.py`

## Pull Request Guidelines

### PR Title

Keep it short and descriptive (< 70 characters):

```
✅ Good: Add Swin Transformer 3D backbone
❌ Bad: This PR adds a new backbone which is Swin Transformer 3D for processing 3D medical images
```

### PR Description

Use this template:

```markdown
## Summary
- Brief description of changes
- Why these changes are needed

## Changes
- List of specific changes made

## Testing
- [ ] Unit tests added/updated
- [ ] Integration tests pass
- [ ] Manual testing completed

## Documentation
- [ ] Docstrings updated
- [ ] VitePress docs updated (if needed)
- [ ] README/docs updated (if needed)
```

### Review Process

1. **Automated checks** must pass:
   - Tests (pytest)
   - Linting (ruff)
   - Type checking (mypy)

2. **Code review** by maintainers:
   - Code quality and style
   - Test coverage
   - Documentation completeness

3. **Address feedback**:
   - Make requested changes
   - Push updates to the same branch
   - Respond to comments

## Getting Help

- **Documentation**: Check the [VitePress docs](https://vitepress.dev/)
- **Issues**: Search existing [GitHub issues](https://github.com/iridyne/medfusion/issues)
- **Discussions**: Start a [GitHub discussion](https://github.com/iridyne/medfusion/discussions)

## Code of Conduct

- Be respectful and inclusive
- Provide constructive feedback
- Focus on the code, not the person
- Help others learn and grow

## License

By contributing, you agree that your contributions will be licensed under the same license as the project.
