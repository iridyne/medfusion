"""
Custom exception classes for Med-Core framework.

Provides specific exception types for better error handling and debugging.
"""


class MedCoreError(Exception):
    """Base exception class for all Med-Core errors."""
    pass


class ConfigurationError(MedCoreError):
    """Raised when there's an error in configuration."""
    pass


class DatasetError(MedCoreError):
    """Raised when there's an error loading or processing datasets."""
    pass


class ModelError(MedCoreError):
    """Raised when there's an error in model construction or loading."""
    pass


class BackboneError(ModelError):
    """Raised when there's an error with backbone modules."""
    pass


class FusionError(ModelError):
    """Raised when there's an error with fusion modules."""
    pass


class TrainingError(MedCoreError):
    """Raised when there's an error during training."""
    pass


class PreprocessingError(MedCoreError):
    """Raised when there's an error during image preprocessing."""
    pass


class EvaluationError(MedCoreError):
    """Raised when there's an error during model evaluation."""
    pass


class CheckpointError(MedCoreError):
    """Raised when there's an error loading or saving checkpoints."""
    pass


class InvalidInputError(MedCoreError):
    """Raised when input data is invalid or malformed."""
    pass


class DimensionMismatchError(InvalidInputError):
    """Raised when tensor dimensions don't match expected shapes."""

    def __init__(self, expected: tuple, actual: tuple, context: str = ""):
        self.expected = expected
        self.actual = actual
        message = f"Dimension mismatch: expected {expected}, got {actual}"
        if context:
            message = f"{context}: {message}"
        super().__init__(message)


class MissingDependencyError(MedCoreError):
    """Raised when a required dependency is not installed."""

    def __init__(self, package: str, install_cmd: str = ""):
        self.package = package
        message = f"Missing required package: {package}"
        if install_cmd:
            message += f"\nInstall with: {install_cmd}"
        super().__init__(message)
