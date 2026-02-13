"""
CLI entry points for Med-Core framework.

This module has been refactored into a package structure.
The functions are now imported from med_core.cli submodules.

Provides command-line interfaces for:
- med-train: Train multimodal models
- med-evaluate: Evaluate trained models
- med-preprocess: Preprocess medical images
"""

# Import from new modular structure
from med_core.cli.evaluate import evaluate
from med_core.cli.preprocess import preprocess
from med_core.cli.train import train

__all__ = ["train", "evaluate", "preprocess"]
