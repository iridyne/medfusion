"""Web service compatibility shims and shared helpers."""

from .training_service import RemovedTrainingServiceError, TrainingService

__all__ = ["RemovedTrainingServiceError", "TrainingService"]
