"""
CLI entry points for Med-Core framework.

Provides command-line interfaces for:
- med-train: Train multimodal models
- med-evaluate: Evaluate trained models
- med-preprocess: Preprocess medical images
"""

from med_core.cli.evaluate import evaluate
from med_core.cli.preprocess import preprocess
from med_core.cli.train import train

__all__ = ["evaluate", "preprocess", "train"]
