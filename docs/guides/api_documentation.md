# API Documentation Guide

## Overview

MedFusion uses Sphinx for automatic API documentation generation. The documentation
is built from docstrings in the source code and provides comprehensive API reference.

## Building Documentation

### Quick Build

```bash
# Build HTML documentation
./scripts/build_docs.sh
```

### Manual Build

```bash
# Navigate to docs directory
cd docs

# Build HTML
make html

# Build PDF (requires LaTeX)
make latexpdf

# Clean build
make clean
```

### View Documentation

After building, open the documentation in your browser:

```bash
# Option 1: Direct file access
open docs/_build/html/index.html

# Option 2: Local server
cd docs/_build/html
python -m http.server 8000
# Visit http://localhost:8000
```

## Documentation Structure

```
docs/
├── conf.py                 # Sphinx configuration
├── index.md               # Main documentation page
├── Makefile               # Build automation
├── api/                   # API reference
│   ├── med_core.md       # Core package
│   ├── aggregators.md    # MIL aggregators
│   ├── attention_supervision.md
│   ├── backbones.md      # Feature extractors
│   ├── datasets.md       # Dataset implementations
│   ├── evaluation.md     # Evaluation metrics
│   ├── fusion.md         # Fusion strategies
│   ├── heads.md          # Classification heads
│   ├── models.md         # Complete models
│   ├── preprocessing.md  # Data preprocessing
│   ├── trainers.md       # Training loops
│   └── utils.md          # Utilities
├── guides/               # User guides
│   ├── quick_reference.md
│   ├── faq_troubleshooting.md
│   ├── docker_deployment.md
│   └── ci_cd.md
└── reference/            # Reference materials
    └── framework_error_codes.md
```

## Sphinx Configuration

### Extensions

The documentation uses the following Sphinx extensions:

- **sphinx.ext.autodoc**: Automatic API documentation from docstrings
- **sphinx.ext.autosummary**: Generate summary tables
- **sphinx.ext.napoleon**: Support for Google and NumPy style docstrings
- **sphinx.ext.viewcode**: Add links to source code
- **sphinx.ext.intersphinx**: Link to other project documentation
- **myst_parser**: Markdown support

### Theme

We use the **Read the Docs** theme (`sphinx_rtd_theme`) for a clean,
professional appearance.

## Writing Documentation

### Docstring Format

Use Google-style docstrings:

```python
def my_function(param1: int, param2: str) -> bool:
    """
    Brief description of the function.

    Longer description with more details about what the function does,
    its behavior, and any important notes.

    Args:
        param1: Description of param1
        param2: Description of param2

    Returns:
        Description of return value

    Raises:
        ValueError: When param1 is negative
        TypeError: When param2 is not a string

    Example:
        >>> result = my_function(42, "hello")
        >>> print(result)
        True

    Note:
        Additional notes or warnings about the function.
    """
    pass
```

### Class Documentation

```python
class MyClass:
    """
    Brief description of the class.

    Longer description with details about the class purpose,
    behavior, and usage patterns.

    Args:
        param1: Description of initialization parameter
        param2: Description of another parameter

    Attributes:
        attr1: Description of attribute
        attr2: Description of another attribute

    Example:
        >>> obj = MyClass(param1=10, param2="test")
        >>> obj.method()
    """

    def __init__(self, param1: int, param2: str):
        self.attr1 = param1
        self.attr2 = param2
```

### Module Documentation

Add module-level docstrings at the top of each file:

```python
"""
Module name and brief description.

This module provides functionality for X, Y, and Z.
It includes the following main components:

- Component1: Description
- Component2: Description
- Component3: Description

Example:
    >>> from med_core.module import Component1
    >>> comp = Component1()
"""
```

## Adding New Documentation

### 1. Add API Reference

Create a new markdown file in `docs/api/`:

```markdown
# Module Name

Brief description.

\`\`\`{eval-rst}
.. automodule:: med_core.module_name
   :members:
   :undoc-members:
   :show-inheritance:
\`\`\`

## Usage Example

\`\`\`python
from med_core.module_name import MyClass

# Example usage
obj = MyClass()
\`\`\`
```

### 2. Update Index

Add the new page to `docs/index.md`:

```markdown
\`\`\`{toctree}
:maxdepth: 2
:caption: API Reference

api/existing_module
api/new_module  # Add this line
\`\`\`
```

### 3. Rebuild Documentation

```bash
./scripts/build_docs.sh
```

## CI/CD Integration

Documentation is automatically built and deployed via GitHub Actions:

```yaml
# .github/workflows/docs.yml
name: Documentation

on:
  push:
    branches: [main]

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Build docs
        run: ./scripts/build_docs.sh
      - name: Deploy to GitHub Pages
        uses: peaceiris/actions-gh-pages@v3
        with:
          github_token: ${{ secrets.GITHUB_TOKEN }}
          publish_dir: ./docs/_build/html
```

## Hosting Options

### GitHub Pages

1. Enable GitHub Pages in repository settings
2. Set source to `gh-pages` branch
3. Documentation will be available at `https://username.github.io/medfusion`

### Read the Docs

1. Import project on readthedocs.org
2. Connect to GitHub repository
3. Documentation builds automatically on push

### Self-Hosted

```bash
# Build documentation
./scripts/build_docs.sh

# Serve with nginx or any web server
cp -r docs/_build/html /var/www/medfusion-docs
```

## Troubleshooting

### Missing Dependencies

```bash
# Install all documentation dependencies
pip install sphinx sphinx-rtd-theme myst-parser
```

### Build Warnings

```bash
# Build with verbose output
cd docs
sphinx-build -b html . _build/html -v
```

### Import Errors

Make sure the package is installed:

```bash
pip install -e .
```

### Theme Not Found

```bash
pip install sphinx-rtd-theme
```

## Best Practices

1. **Keep docstrings up to date**: Update documentation when changing code
2. **Include examples**: Add usage examples in docstrings
3. **Document parameters**: Describe all parameters and return values
4. **Add type hints**: Use type annotations for better documentation
5. **Write clear descriptions**: Be concise but complete
6. **Test examples**: Ensure code examples actually work
7. **Build regularly**: Check documentation builds without errors

## Resources

- [Sphinx Documentation](https://www.sphinx-doc.org/)
- [Read the Docs Theme](https://sphinx-rtd-theme.readthedocs.io/)
- [MyST Parser](https://myst-parser.readthedocs.io/)
- [Google Style Docstrings](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings)

## Summary

The API documentation system provides:

- ✅ Automatic generation from source code
- ✅ Professional Read the Docs theme
- ✅ Markdown and reStructuredText support
- ✅ Code examples and usage patterns
- ✅ Cross-references and search functionality
- ✅ Easy to build and deploy
- ✅ CI/CD integration ready

Build the documentation with `./scripts/build_docs.sh` and view it in your browser!
