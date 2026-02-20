"""Tests for enhanced exception system."""

import pytest

from med_core.exceptions import (
    AttentionSupervisionError,
    BackboneNotFoundError,
    CheckpointNotFoundError,
    ConfigurationError,
    DatasetNotFoundError,
    DimensionMismatchError,
    FusionNotFoundError,
    IncompatibleConfigError,
    MedCoreError,
    MissingColumnError,
    MissingDependencyError,
    MultiViewError,
    TrainingError,
    format_error_report,
)


class TestMedCoreError:
    """Test base MedCoreError class."""

    def test_basic_error(self):
        """Test basic error creation."""
        error = MedCoreError("Something went wrong")
        assert error.error_code == "E000"
        assert error.base_message == "Something went wrong"
        assert "[E000]" in str(error)

    def test_error_with_code(self):
        """Test error with custom code."""
        error = MedCoreError("Test error", error_code="E999")
        assert error.error_code == "E999"
        assert "[E999]" in str(error)

    def test_error_with_context(self):
        """Test error with context."""
        error = MedCoreError(
            "Test error",
            context={"key1": "value1", "key2": 42}
        )
        assert error.context == {"key1": "value1", "key2": 42}
        assert "Context:" in str(error)
        assert "key1=value1" in str(error)

    def test_error_with_suggestion(self):
        """Test error with suggestion."""
        error = MedCoreError(
            "Test error",
            suggestion="Try doing this instead"
        )
        assert error.suggestion == "Try doing this instead"
        assert "💡 Suggestion:" in str(error)


class TestBackboneNotFoundError:
    """Test BackboneNotFoundError."""

    def test_backbone_not_found(self):
        """Test backbone not found error."""
        available = ["resnet18", "resnet50", "efficientnet_b0"]
        error = BackboneNotFoundError("invalid_backbone", available)
        
        assert error.error_code == "E311"
        assert "invalid_backbone" in error.base_message
        assert "resnet18" in str(error)
        assert error.context["available_backbones"] == available


class TestFusionNotFoundError:
    """Test FusionNotFoundError."""

    def test_fusion_not_found(self):
        """Test fusion not found error."""
        available = ["concatenate", "gated", "attention"]
        error = FusionNotFoundError("invalid_fusion", available)
        
        assert error.error_code == "E321"
        assert "invalid_fusion" in error.base_message
        assert "concatenate" in str(error)


class TestDatasetNotFoundError:
    """Test DatasetNotFoundError."""

    def test_dataset_not_found(self):
        """Test dataset not found error."""
        error = DatasetNotFoundError("/path/to/dataset.csv")
        
        assert error.error_code == "E201"
        assert "/path/to/dataset.csv" in error.base_message
        assert "Check that the path exists" in str(error)


class TestMissingColumnError:
    """Test MissingColumnError."""

    def test_missing_column(self):
        """Test missing column error."""
        available = ["col1", "col2", "col3"]
        error = MissingColumnError("missing_col", available)
        
        assert error.error_code == "E202"
        assert "missing_col" in error.base_message
        assert "col1, col2, col3" in str(error)


class TestDimensionMismatchError:
    """Test DimensionMismatchError."""

    def test_dimension_mismatch(self):
        """Test dimension mismatch error."""
        error = DimensionMismatchError(
            expected=(32, 3, 224, 224),
            actual=(32, 3, 256, 256),
            tensor_name="input_image"
        )
        
        assert error.error_code == "E701"
        assert error.expected == (32, 3, 224, 224)
        assert error.actual == (32, 3, 256, 256)
        assert "input_image" in error.base_message
        assert "Reshape input" in str(error)


class TestMissingDependencyError:
    """Test MissingDependencyError."""

    def test_missing_dependency_default(self):
        """Test missing dependency with default install command."""
        error = MissingDependencyError("torch")
        
        assert error.error_code == "E800"
        assert error.package == "torch"
        assert "pip install torch" in str(error)

    def test_missing_dependency_custom_install(self):
        """Test missing dependency with custom install command."""
        error = MissingDependencyError("torch", "conda install pytorch")
        
        assert "conda install pytorch" in str(error)


class TestTrainingError:
    """Test TrainingError."""

    def test_training_error_with_epoch(self):
        """Test training error with epoch info."""
        error = TrainingError(
            "Loss exploded",
            epoch=10,
            step=500,
            suggestion="Reduce learning rate"
        )
        
        assert error.error_code == "E400"
        assert error.context["epoch"] == 10
        assert error.context["step"] == 500
        assert "Reduce learning rate" in str(error)


class TestCheckpointNotFoundError:
    """Test CheckpointNotFoundError."""

    def test_checkpoint_not_found(self):
        """Test checkpoint not found error."""
        error = CheckpointNotFoundError("/path/to/checkpoint.pth")
        
        assert error.error_code == "E411"
        assert "/path/to/checkpoint.pth" in error.base_message


class TestAttentionSupervisionError:
    """Test AttentionSupervisionError."""

    def test_attention_supervision_error(self):
        """Test attention supervision error."""
        error = AttentionSupervisionError(
            "Attention supervision requires CBAM",
            attention_type="se",
            suggestion="Use CBAM attention instead"
        )
        
        assert error.error_code == "E900"
        assert error.context["attention_type"] == "se"
        assert "Use CBAM" in str(error)


class TestMultiViewError:
    """Test MultiViewError."""

    def test_multiview_error(self):
        """Test multi-view error."""
        error = MultiViewError(
            "Missing view",
            view_name="coronal",
            num_views=2,
            suggestion="Provide all required views"
        )
        
        assert error.error_code == "E1000"
        assert error.context["view_name"] == "coronal"
        assert error.context["num_views"] == 2


class TestIncompatibleConfigError:
    """Test IncompatibleConfigError."""

    def test_incompatible_config(self):
        """Test incompatible config error."""
        error = IncompatibleConfigError(
            "Cannot use attention supervision with SE attention",
            conflicting_options=["use_attention_supervision", "attention_type=se"],
            suggestion="Use CBAM attention for supervision"
        )
        
        assert error.error_code == "E101"
        assert "use_attention_supervision" in error.context["conflicting_options"]


class TestFormatErrorReport:
    """Test error report formatting."""

    def test_format_medcore_error(self):
        """Test formatting MedCore error."""
        error = BackboneNotFoundError(
            "resnet999",
            ["resnet18", "resnet50"]
        )
        
        report = format_error_report(error)
        
        assert "❌ Error [E311]" in report
        assert "📋 Context:" in report
        assert "💡 Suggestion:" in report
        assert "resnet999" in report

    def test_format_standard_error(self):
        """Test formatting standard Python error."""
        error = ValueError("Invalid value")
        
        report = format_error_report(error)
        
        assert "❌ Error:" in report
        assert "Invalid value" in report

    def test_format_error_without_suggestion(self):
        """Test formatting error without suggestion."""
        error = MedCoreError("Test error", error_code="E999")
        
        report = format_error_report(error)
        
        assert "❌ Error [E999]" in report
        assert "💡 Suggestion:" not in report


class TestErrorHierarchy:
    """Test exception hierarchy."""

    def test_all_inherit_from_medcore_error(self):
        """Test that all custom exceptions inherit from MedCoreError."""
        errors = [
            ConfigurationError("test"),
            DatasetNotFoundError("/path"),
            BackboneNotFoundError("test", []),
            TrainingError("test"),
            CheckpointNotFoundError("/path"),
        ]
        
        for error in errors:
            assert isinstance(error, MedCoreError)
            assert isinstance(error, Exception)

    def test_specific_inheritance(self):
        """Test specific inheritance relationships."""
        error = BackboneNotFoundError("test", [])
        
        assert isinstance(error, BackboneNotFoundError)
        assert isinstance(error, MedCoreError)
        assert isinstance(error, Exception)


class TestErrorCodes:
    """Test error code uniqueness and consistency."""

    def test_error_codes_are_unique(self):
        """Test that error codes are unique."""
        errors = [
            MedCoreError("test", error_code="E000"),
            ConfigurationError("test"),  # E100
            IncompatibleConfigError("test"),  # E101
            DatasetNotFoundError("/path"),  # E201
            MissingColumnError("col", []),  # E202
            BackboneNotFoundError("test", []),  # E311
            FusionNotFoundError("test", []),  # E321
            TrainingError("test"),  # E400
            CheckpointNotFoundError("/path"),  # E411
            DimensionMismatchError((1,), (2,)),  # E701
            MissingDependencyError("pkg"),  # E800
            AttentionSupervisionError("test"),  # E900
            MultiViewError("test"),  # E1000
        ]
        
        codes = [e.error_code for e in errors]
        assert len(codes) == len(set(codes)), "Error codes must be unique"

    def test_error_codes_format(self):
        """Test that error codes follow format."""
        error = BackboneNotFoundError("test", [])
        
        assert error.error_code.startswith("E")
        assert error.error_code[1:].isdigit()
