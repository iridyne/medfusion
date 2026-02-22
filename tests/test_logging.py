"""Tests for enhanced logging system."""

import json
import logging

import pytest

from med_core.utils.logging import (
    LogContext,
    MetricsLogger,
    PerformanceLogger,
    get_logger,
    log_function_call,
    setup_logging,
)


class TestSetupLogging:
    """Test logging setup."""

    def test_basic_setup(self):
        """Test basic logging setup."""
        logger = setup_logging(level="INFO")
        assert logger.level == logging.INFO
        assert len(logger.handlers) > 0

    def test_setup_with_file(self, tmp_path):
        """Test logging setup with file output."""
        log_file = tmp_path / "test.log"
        logger = setup_logging(level="DEBUG", log_file=log_file)

        assert logger.level == logging.DEBUG
        assert log_file.exists()

    def test_setup_with_json(self, tmp_path):
        """Test logging setup with JSON formatting."""
        log_file = tmp_path / "test.json"
        setup_logging(level="INFO", log_file=log_file, use_json=True)

        test_logger = get_logger("test")
        test_logger.info("Test message")

        # Read and verify JSON format
        with open(log_file) as f:
            line = f.readline()
            data = json.loads(line)
            assert "timestamp" in data
            assert "level" in data
            assert "message" in data
            assert data["message"] == "Test message"


class TestLogContext:
    """Test LogContext context manager."""

    def test_log_context(self, tmp_path):
        """Test that context is added to logs."""
        log_file = tmp_path / "test.json"
        setup_logging(level="INFO", log_file=log_file, use_json=True)

        logger = get_logger("test")

        with LogContext(experiment="exp1", epoch=5):
            logger.info("Training started")

        # Verify context in log
        with open(log_file) as f:
            line = f.readline()
            data = json.loads(line)
            assert "context" in data
            assert data["context"]["experiment"] == "exp1"
            assert data["context"]["epoch"] == 5

    def test_nested_context(self, tmp_path):
        """Test nested log contexts."""
        log_file = tmp_path / "test.json"
        setup_logging(level="INFO", log_file=log_file, use_json=True)

        logger = get_logger("test")

        with LogContext(experiment="exp1"):
            with LogContext(epoch=5):
                logger.info("Nested context")

        # Verify both contexts are present
        with open(log_file) as f:
            line = f.readline()
            data = json.loads(line)
            assert data["context"]["experiment"] == "exp1"
            assert data["context"]["epoch"] == 5

    def test_context_cleanup(self, tmp_path):
        """Test that context is cleaned up after exit."""
        log_file = tmp_path / "test.json"
        setup_logging(level="INFO", log_file=log_file, use_json=True)

        logger = get_logger("test")

        with LogContext(experiment="exp1"):
            logger.info("Inside context")

        logger.info("Outside context")

        # Verify context is removed
        with open(log_file) as f:
            lines = f.readlines()
            assert len(lines) == 2

            # First log should have context
            data1 = json.loads(lines[0])
            assert "context" in data1

            # Second log should not have context (or empty context)
            data2 = json.loads(lines[1])
            assert "context" not in data2 or data2["context"] == {}


class TestPerformanceLogger:
    """Test PerformanceLogger."""

    def test_performance_logging(self, caplog):
        """Test performance logging."""
        with caplog.at_level(logging.INFO):
            with PerformanceLogger("test_operation"):
                pass  # Simulate work

        # Check that performance was logged
        assert any(
            "test_operation completed" in record.message for record in caplog.records
        )
        assert any(hasattr(record, "elapsed_time") for record in caplog.records)

    def test_performance_logging_with_error(self, caplog):
        """Test performance logging when error occurs."""
        with caplog.at_level(logging.ERROR):
            try:
                with PerformanceLogger("failing_operation"):
                    raise ValueError("Test error")
            except ValueError:
                pass

        # Check that error was logged with timing
        assert any(
            "failing_operation failed" in record.message for record in caplog.records
        )


class TestLogFunctionCall:
    """Test log_function_call decorator."""

    def test_function_call_logging(self, caplog):
        """Test function call logging."""

        @log_function_call(level=logging.INFO)
        def test_function(x, y):
            return x + y

        with caplog.at_level(logging.INFO):
            result = test_function(2, 3)

        assert result == 5
        assert any(
            "Calling" in record.message and "test_function" in record.message
            for record in caplog.records
        )
        assert any("completed" in record.message for record in caplog.records)

    def test_function_call_logging_with_error(self, caplog):
        """Test function call logging when error occurs."""

        @log_function_call(level=logging.DEBUG)
        def failing_function():
            raise ValueError("Test error")

        with caplog.at_level(logging.ERROR):
            with pytest.raises(ValueError):
                failing_function()

        assert any("failed" in record.message for record in caplog.records)


class TestMetricsLogger:
    """Test MetricsLogger."""

    def test_metrics_logging(self, caplog):
        """Test metrics logging."""
        metrics = MetricsLogger("training")

        with caplog.at_level(logging.INFO):
            metrics.log("loss", 0.5, step=100)
            metrics.log("accuracy", 0.85, step=100)

        # Check logs
        assert any(
            "training.loss" in record.message and "0.5" in record.message
            for record in caplog.records
        )
        assert any(
            "training.accuracy" in record.message and "0.85" in record.message
            for record in caplog.records
        )

    def test_metrics_retrieval(self):
        """Test retrieving logged metrics."""
        metrics = MetricsLogger("training")

        metrics.log("loss", 0.5, step=1)
        metrics.log("loss", 0.4, step=2)
        metrics.log("loss", 0.3, step=3)

        loss_metrics = metrics.get_metrics("loss")
        assert len(loss_metrics) == 3
        assert loss_metrics[0]["value"] == 0.5
        assert loss_metrics[1]["value"] == 0.4
        assert loss_metrics[2]["value"] == 0.3

    def test_metrics_summary(self):
        """Test metrics summary statistics."""
        metrics = MetricsLogger("training")

        metrics.log("loss", 0.5, step=1)
        metrics.log("loss", 0.4, step=2)
        metrics.log("loss", 0.3, step=3)
        metrics.log("accuracy", 0.8, step=1)
        metrics.log("accuracy", 0.85, step=2)

        summary = metrics.summary()

        assert "loss" in summary
        assert summary["loss"]["min"] == 0.3
        assert summary["loss"]["max"] == 0.5
        assert abs(summary["loss"]["mean"] - 0.4) < 0.01  # Allow floating point error
        assert summary["loss"]["count"] == 3

        assert "accuracy" in summary
        assert summary["accuracy"]["min"] == 0.8
        assert summary["accuracy"]["max"] == 0.85

    def test_metrics_with_context(self, tmp_path):
        """Test metrics logging with context."""
        log_file = tmp_path / "metrics.json"
        setup_logging(level="INFO", log_file=log_file, use_json=True)

        metrics = MetricsLogger("training")

        with LogContext(experiment="exp1"):
            metrics.log("loss", 0.5, step=100)

        # Verify context in log
        with open(log_file) as f:
            line = f.readline()
            data = json.loads(line)
            assert data["context"]["experiment"] == "exp1"
            assert data["metric_name"] == "loss"
            assert data["value"] == 0.5


class TestIntegration:
    """Test integration of logging features."""

    def test_combined_features(self, tmp_path):
        """Test using multiple logging features together."""
        log_file = tmp_path / "combined.json"
        setup_logging(level="INFO", log_file=log_file, use_json=True)

        logger = get_logger("test")
        metrics = MetricsLogger("training")

        with LogContext(experiment="exp1", model="resnet50"):
            logger.info("Starting training")

            with PerformanceLogger("epoch_1"):
                metrics.log("loss", 0.5, step=1)
                metrics.log("accuracy", 0.8, step=1)

        # Verify all logs are present
        with open(log_file) as f:
            lines = f.readlines()
            assert len(lines) >= 3  # At least 3 log entries

            # All should have context
            for line in lines:
                data = json.loads(line)
                if "context" in data:
                    assert data["context"]["experiment"] == "exp1"

    def test_error_with_context(self, tmp_path):
        """Test error logging with context."""
        log_file = tmp_path / "error.json"
        setup_logging(level="ERROR", log_file=log_file, use_json=True)

        logger = get_logger("test")

        with LogContext(experiment="exp1", epoch=5):
            try:
                raise ValueError("Test error")
            except ValueError:
                logger.exception("Error occurred")

        # Verify error log has context and exception
        with open(log_file) as f:
            line = f.readline()
            data = json.loads(line)
            assert "context" in data
            assert data["context"]["experiment"] == "exp1"
            assert "exception" in data
            assert "ValueError" in data["exception"]
