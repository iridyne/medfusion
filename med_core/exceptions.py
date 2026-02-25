"""
Custom exception classes for Med-Core framework.

Provides specific exception types for better error handling and debugging.
All exceptions include error codes, context information, and helpful suggestions.
"""

from typing import Any


class MedCoreError(Exception):
    """
    Base exception class for all Med-Core errors.

    All Med-Core exceptions include:
    - error_code: Unique identifier for the error type
    - context: Additional context information
    - suggestion: Helpful suggestion for fixing the error
    """

    def __init__(
        self,
        message: str,
        error_code: str | None = None,
        context: dict[str, Any] | None = None,
        suggestion: str | None = None,
    ) -> None:
        self.error_code = error_code or "E000"
        self.context = context or {}
        self.suggestion = suggestion

        # Build comprehensive error message
        full_message = f"[{self.error_code}] {message}"

        if context:
            context_str = ", ".join(f"{k}={v}" for k, v in context.items())
            full_message += f"\n  Context: {context_str}"

        if suggestion:
            full_message += f"\n  💡 Suggestion: {suggestion}"

        super().__init__(full_message)
        self.base_message = message


class ConfigurationError(MedCoreError):
    """Raised when there's an error in configuration."""

    def __init__(
        self,
        message: str,
        config_path: str | None = None,
        invalid_value: Any = None,
        suggestion: str | None = None,
    ) -> None:
        context = {}
        if config_path:
            context["config_path"] = config_path
        if invalid_value is not None:
            context["invalid_value"] = invalid_value

        super().__init__(
            message=message,
            error_code="E100",
            context=context,
            suggestion=suggestion,
        )


class DatasetError(MedCoreError):
    """Raised when there's an error loading or processing datasets."""

    def __init__(
        self,
        message: str,
        dataset_path: str | None = None,
        sample_id: str | None = None,
        suggestion: str | None = None,
    ) -> None:
        context = {}
        if dataset_path:
            context["dataset_path"] = dataset_path
        if sample_id:
            context["sample_id"] = sample_id

        super().__init__(
            message=message,
            error_code="E200",
            context=context,
            suggestion=suggestion,
        )


class DatasetNotFoundError(DatasetError):
    """Raised when dataset file or directory is not found."""

    def __init__(self, path: str) -> None:
        super().__init__(
            message=f"Dataset not found: {path}",
            dataset_path=path,
            suggestion="Check that the path exists and is accessible",
        )
        self.error_code = "E201"


class MissingColumnError(DatasetError):
    """Raised when required column is missing from dataset."""

    def __init__(self, column: str, available_columns: list[str]) -> None:
        super().__init__(
            message=f"Required column '{column}' not found in dataset",
            suggestion=f"Available columns: {', '.join(available_columns)}",
        )
        self.error_code = "E202"
        self.context["column"] = column
        self.context["available_columns"] = available_columns


class ModelError(MedCoreError):
    """Raised when there's an error in model construction or loading."""

    def __init__(
        self,
        message: str,
        model_name: str | None = None,
        suggestion: str | None = None,
    ) -> None:
        context = {}
        if model_name:
            context["model_name"] = model_name

        super().__init__(
            message=message,
            error_code="E300",
            context=context,
            suggestion=suggestion,
        )


class BackboneError(ModelError):
    """Raised when there's an error with backbone modules."""

    def __init__(
        self,
        message: str,
        backbone_name: str | None = None,
        suggestion: str | None = None,
    ) -> None:
        super().__init__(
            message=message, model_name=backbone_name, suggestion=suggestion
        )
        self.error_code = "E310"


class BackboneNotFoundError(BackboneError):
    """Raised when requested backbone is not available."""

    def __init__(self, backbone_name: str, available_backbones: list[str]) -> None:
        super().__init__(
            message=f"Backbone '{backbone_name}' not found",
            backbone_name=backbone_name,
            suggestion=f"Available backbones: {', '.join(available_backbones[:5])}...",
        )
        self.error_code = "E311"
        self.context["available_backbones"] = available_backbones


class FusionError(ModelError):
    """Raised when there's an error with fusion modules."""

    def __init__(
        self,
        message: str,
        fusion_type: str | None = None,
        suggestion: str | None = None,
    ) -> None:
        super().__init__(message=message, model_name=fusion_type, suggestion=suggestion)
        self.error_code = "E320"


class FusionNotFoundError(FusionError):
    """Raised when requested fusion strategy is not available."""

    def __init__(self, fusion_type: str, available_fusions: list[str]) -> None:
        super().__init__(
            message=f"Fusion type '{fusion_type}' not found",
            fusion_type=fusion_type,
            suggestion=f"Available fusion types: {', '.join(available_fusions)}",
        )
        self.error_code = "E321"
        self.context["available_fusions"] = available_fusions


class TrainingError(MedCoreError):
    """Raised when there's an error during training."""

    def __init__(
        self,
        message: str,
        epoch: int | None = None,
        step: int | None = None,
        suggestion: str | None = None,
    ) -> None:
        context = {}
        if epoch is not None:
            context["epoch"] = epoch
        if step is not None:
            context["step"] = step

        super().__init__(
            message=message,
            error_code="E400",
            context=context,
            suggestion=suggestion,
        )


class CheckpointError(MedCoreError):
    """Raised when there's an error loading or saving checkpoints."""

    def __init__(
        self,
        message: str,
        checkpoint_path: str | None = None,
        suggestion: str | None = None,
    ) -> None:
        context = {}
        if checkpoint_path:
            context["checkpoint_path"] = checkpoint_path

        super().__init__(
            message=message,
            error_code="E410",
            context=context,
            suggestion=suggestion,
        )


class CheckpointNotFoundError(CheckpointError):
    """Raised when checkpoint file is not found."""

    def __init__(self, checkpoint_path: str) -> None:
        super().__init__(
            message=f"Checkpoint not found: {checkpoint_path}",
            checkpoint_path=checkpoint_path,
            suggestion="Check that the checkpoint path is correct",
        )
        self.error_code = "E411"


class PreprocessingError(MedCoreError):
    """Raised when there's an error during image preprocessing."""

    def __init__(
        self,
        message: str,
        image_path: str | None = None,
        suggestion: str | None = None,
    ) -> None:
        context = {}
        if image_path:
            context["image_path"] = image_path

        super().__init__(
            message=message,
            error_code="E500",
            context=context,
            suggestion=suggestion,
        )


class EvaluationError(MedCoreError):
    """Raised when there's an error during model evaluation."""

    def __init__(
        self,
        message: str,
        metric_name: str | None = None,
        suggestion: str | None = None,
    ) -> None:
        context = {}
        if metric_name:
            context["metric_name"] = metric_name

        super().__init__(
            message=message,
            error_code="E600",
            context=context,
            suggestion=suggestion,
        )


class InvalidInputError(MedCoreError):
    """Raised when input data is invalid or malformed."""

    def __init__(
        self,
        message: str,
        input_name: str | None = None,
        expected_type: str | None = None,
        actual_type: str | None = None,
        suggestion: str | None = None,
    ) -> None:
        context = {}
        if input_name:
            context["input_name"] = input_name
        if expected_type:
            context["expected_type"] = expected_type
        if actual_type:
            context["actual_type"] = actual_type

        super().__init__(
            message=message,
            error_code="E700",
            context=context,
            suggestion=suggestion,
        )


class DimensionMismatchError(InvalidInputError):
    """Raised when tensor dimensions don't match expected shapes."""

    def __init__(
        self,
        expected: tuple,
        actual: tuple,
        tensor_name: str = "",
        suggestion: str | None = None,
    ) -> None:
        self.expected = expected
        self.actual = actual

        message = f"Dimension mismatch: expected {expected}, got {actual}"
        if tensor_name:
            message = f"{tensor_name}: {message}"

        if not suggestion:
            suggestion = f"Reshape input to match expected dimensions {expected}"

        super().__init__(
            message=message,
            input_name=tensor_name,
            expected_type=str(expected),
            actual_type=str(actual),
            suggestion=suggestion,
        )
        self.error_code = "E701"


class MissingDependencyError(MedCoreError):
    """Raised when a required dependency is not installed."""

    def __init__(self, package: str, install_cmd: str = "") -> None:
        self.package = package

        if not install_cmd:
            install_cmd = f"pip install {package}"

        super().__init__(
            message=f"Missing required package: {package}",
            error_code="E800",
            context={"package": package},
            suggestion=f"Install with: {install_cmd}",
        )


class AttentionSupervisionError(MedCoreError):
    """Raised when there's an error with attention supervision."""

    def __init__(
        self,
        message: str,
        attention_type: str | None = None,
        suggestion: str | None = None,
    ) -> None:
        context = {}
        if attention_type:
            context["attention_type"] = attention_type

        super().__init__(
            message=message,
            error_code="E900",
            context=context,
            suggestion=suggestion,
        )


class MultiViewError(MedCoreError):
    """Raised when there's an error with multi-view processing."""

    def __init__(
        self,
        message: str,
        view_name: str | None = None,
        num_views: int | None = None,
        suggestion: str | None = None,
    ) -> None:
        context: dict[str, str | int] = {}
        if view_name:
            context["view_name"] = view_name
        if num_views is not None:
            context["num_views"] = num_views

        super().__init__(
            message=message,
            error_code="E1000",
            context=context,
            suggestion=suggestion,
        )


class IncompatibleConfigError(ConfigurationError):
    """Raised when configuration options are incompatible."""

    def __init__(
        self,
        message: str,
        conflicting_options: list[str] | None = None,
        suggestion: str | None = None,
    ) -> None:
        super().__init__(message=message, suggestion=suggestion)
        self.error_code = "E101"
        if conflicting_options:
            self.context["conflicting_options"] = conflicting_options


# Convenience function for error reporting
def format_error_report(error: Exception) -> str:
    """
    Format an error into a user-friendly report.

    Args:
        error: Exception to format

    Returns:
        Formatted error report string
    """
    if isinstance(error, MedCoreError):
        report = f"❌ Error [{error.error_code}]: {error.base_message}\n"

        if error.context:
            report += "\n📋 Context:\n"
            for key, value in error.context.items():
                report += f"  • {key}: {value}\n"

        if error.suggestion:
            report += f"\n💡 Suggestion: {error.suggestion}\n"

        return report
    else:
        return f"❌ Error: {str(error)}\n"
