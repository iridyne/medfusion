"""Worker layer entrypoints for MedFusion Web."""

from .local_training_worker import (
    LocalTrainingWorkerRegistry,
    local_training_worker_registry,
)

__all__ = ["LocalTrainingWorkerRegistry", "local_training_worker_registry"]
