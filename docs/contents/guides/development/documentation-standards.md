# Documentation Standards

This guide defines the documentation standards for the MedFusion project to ensure consistency, clarity, and maintainability.

## Language Policy

### Code Documentation (MANDATORY English)

All code-level documentation MUST be written in English:

- **Docstrings**: Function, class, and module docstrings
- **Inline comments**: Code comments explaining logic
- **Type hints**: Variable and parameter annotations
- **API documentation**: Auto-generated API references
- **Error messages**: Exception messages and logging

**Rationale**: English code documentation ensures:
- International collaboration and contribution
- Compatibility with documentation generation tools
- Consistency with the broader Python/PyTorch ecosystem
- Easier code review and maintenance

### User Documentation (Flexible)

User-facing documentation can be written in Chinese or English:

- **Tutorials**: Step-by-step guides
- **How-to guides**: Task-oriented instructions
- **Case studies**: Example applications
- **README files**: Project overviews

**Rationale**: User documentation in the user's native language improves accessibility and learning outcomes.

## Docstring Format

MedFusion uses **Google-style docstrings** for all Python code.

### Function Docstrings

```python
def create_fusion_module(
    fusion_type: str,
    input_dims: list[int],
    output_dim: int,
    **kwargs
) -> nn.Module:
    """Create a fusion module for combining multimodal features.

    Args:
        fusion_type: Type of fusion strategy ('attention', 'gated', 'bilinear')
        input_dims: List of input feature dimensions for each modality
        output_dim: Dimension of the fused output features
        **kwargs: Additional arguments passed to the fusion module

    Returns:
        Initialized fusion module

    Raises:
        ValueError: If fusion_type is not supported
        ValueError: If input_dims is empty

    Example:
        >>> fusion = create_fusion_module('attention', [512, 768], 256)
        >>> output = fusion([vision_features, tabular_features])
    """
```

### Class Docstrings

```python
class AttentionFusion(nn.Module):
    """Attention-based fusion for multimodal features.

    This module uses cross-attention to dynamically weight and combine
    features from different modalities based on their relevance.

    Attributes:
        input_dims: List of input feature dimensions
        output_dim: Dimension of fused output
        attention: Multi-head attention module
        projection: Linear projection layer

    Example:
        >>> fusion = AttentionFusion([512, 768], 256, num_heads=8)
        >>> fused = fusion([vision_feat, tabular_feat])
        >>> fused.shape
        torch.Size([32, 256])
    """
```

### Module Docstrings

```python
"""Fusion strategies for multimodal learning.

This module provides various fusion strategies for combining features
from multiple modalities (vision, tabular, text, etc.):

- Concatenation: Simple feature concatenation
- Gated: Learnable gating mechanism
- Attention: Cross-attention based fusion
- Bilinear: Bilinear pooling
- Kronecker: Kronecker product fusion

Example:
    >>> from med_core.fusion import create_fusion_module
    >>> fusion = create_fusion_module('attention', [512, 768], 256)
"""
```

## Comment Guidelines

### When to Write Comments

Write comments to explain **WHY**, not **WHAT**:

```python
# Good: Explains reasoning
# Use gradient accumulation to fit larger effective batch sizes on limited GPU memory
if step % accumulation_steps == 0:
    optimizer.step()

# Bad: Restates the code
# Step the optimizer
optimizer.step()
```

### When NOT to Write Comments

Avoid comments for self-explanatory code:

```python
# Bad: Obvious from code
# Create a list of losses
losses = []

# Good: No comment needed
losses = []
```

### Comment Maintenance

- **Update comments** when code changes
- **Remove outdated comments** immediately
- **Keep comments minimal** - less is more

## Type Annotations

All functions must have complete type annotations:

```python
# Good: Complete type hints
def train_model(
    model: nn.Module,
    dataloader: DataLoader,
    epochs: int,
    device: str = "cuda"
) -> dict[str, float]:
    """Train the model."""
    ...

# Bad: Missing type hints
def train_model(model, dataloader, epochs, device="cuda"):
    """Train the model."""
    ...
```

### Modern Type Hints (Python 3.11+)

Use modern type hint syntax:

```python
# Good: Modern syntax
def process_data(items: list[str]) -> dict[str, int]:
    ...

def get_config(path: str | None = None) -> dict:
    ...

# Bad: Old syntax
from typing import List, Dict, Optional

def process_data(items: List[str]) -> Dict[str, int]:
    ...

def get_config(path: Optional[str] = None) -> Dict:
    ...
```

## Documentation Structure

### VitePress Documentation

The project uses VitePress for comprehensive documentation:

```
docs/
├── .vitepress/
│   └── config.mts          # VitePress configuration
├── contents/
│   ├── tutorials/          # Step-by-step tutorials
│   ├── guides/             # Feature guides
│   ├── api/                # API references
│   ├── architecture/       # Architecture docs
│   └── case-studies/       # Example applications
└── index.md                # Landing page
```

Published-site boundary:

- `docs/` is the published documentation source root
- reader-facing tutorials, guides, references, and architecture pages belong here
- internal plans, implementation notes, temporary specs, and working research memos do **not** belong here
- for this workspace, keep non-published working docs in the companion Vault project:
  `C:\Users\Administrator\Vault\Projects\MedML\仓库文档`

Rationale:

- VitePress treats markdown under the project source root as source content
- keeping internal working docs outside `docs/` avoids accidental publication
- separating published docs from working notes also reduces duplicate maintenance

### Adding New Documentation

1. **Create the markdown file** in the appropriate directory
2. **Update VitePress config** (`docs/.vitepress/config.mts`) to add the page to the sidebar
3. **Test locally**: `cd docs && npm run dev`
4. **Verify links** and navigation work correctly

Before adding a new markdown file, first decide:

- Is this intended for external readers and safe to publish? Put it under `docs/`
- Is this a working note, implementation plan, or internal draft? Put it in the Vault-backed internal docs area, not in the repo

## Migration Plan

### New Code (Immediate)

All new code MUST follow these standards:

- English docstrings and comments
- Google-style docstring format
- Complete type annotations
- Minimal, purposeful comments

### Existing Code (Gradual)

Migrate existing code opportunistically:

- **When modifying a function**: Update its docstring to English
- **When refactoring a module**: Update all docstrings in that module
- **During code review**: Suggest English docstrings for new contributions
- **No dedicated migration sprints**: Migrate as you work

### Priority Order

1. **Public APIs**: Functions and classes used by external code
2. **Core modules**: Frequently modified or critical code paths
3. **Utilities**: Helper functions and common utilities
4. **Legacy code**: Rarely touched code (lowest priority)

## Tools and Validation

### Linting

```bash
# Check docstring format
ruff check med_core/ --select D

# Auto-fix docstring issues
ruff check med_core/ --select D --fix
```

### Type Checking

```bash
# Run mypy type checker
mypy med_core/

# Check specific file
mypy med_core/models/builder.py
```

### Documentation Generation

```bash
# Build VitePress docs
cd docs && npm run build

# Preview built docs
cd docs && npm run preview
```

## Best Practices

### Be Concise

```python
# Good: Concise and clear
def load_checkpoint(path: str) -> dict:
    """Load model checkpoint from disk."""
    return torch.load(path)

# Bad: Overly verbose
def load_checkpoint(path: str) -> dict:
    """Load a model checkpoint from the specified path on disk.

    This function takes a file path as input and loads the checkpoint
    file from that location, returning the checkpoint data as a dictionary.
    """
    return torch.load(path)
```

### Use Examples

Include examples for complex APIs:

```python
def build_model_from_config(config: dict) -> nn.Module:
    """Build a model from configuration dictionary.

    Args:
        config: Model configuration with 'backbones', 'fusion', 'head' keys

    Returns:
        Initialized model

    Example:
        >>> config = {
        ...     'backbones': {'ct': {'type': 'resnet50'}},
        ...     'fusion': {'type': 'attention'},
        ...     'head': {'type': 'classification', 'num_classes': 2}
        ... }
        >>> model = build_model_from_config(config)
    """
```

### Document Assumptions

Make implicit assumptions explicit:

```python
def normalize_image(image: np.ndarray) -> np.ndarray:
    """Normalize image to [0, 1] range.

    Assumes input image is in [0, 255] range with dtype uint8.

    Args:
        image: Input image array (H, W, C) in [0, 255]

    Returns:
        Normalized image in [0, 1] range with dtype float32
    """
```

## References

- [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings)
- [PEP 257 - Docstring Conventions](https://peps.python.org/pep-0257/)
- [PEP 484 - Type Hints](https://peps.python.org/pep-0484/)
- [VitePress Documentation](https://vitepress.dev/)
