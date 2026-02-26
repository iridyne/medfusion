"""
Trainer modules for medical model training.

Provides:
- BaseTrainer: Abstract base class for training loops
- MultimodalTrainer: Specialized trainer for multimodal medical models
- MultiViewMultimodalTrainer: Trainer for multi-view multimodal models
"""

from med_core.trainers.base import BaseTrainer
from med_core.trainers.multimodal import MultimodalTrainer, create_trainer
from med_core.trainers.multiview_trainer import (
    MultiViewMultimodalTrainer,
    create_multiview_trainer,
)

__all__ = [
    "BaseTrainer",
    "MultiViewMultimodalTrainer",
    "MultimodalTrainer",
    "create_multiview_trainer",
    "create_trainer",
]
