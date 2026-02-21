"""
Enhanced logging configuration utilities for Med-Core framework.

Provides structured logging with context, JSON formatting, and performance tracking.
"""

import json
import logging
import sys
import time
from contextvars import ContextVar
from pathlib import Path
from typing import Any

# Context variables for structured logging
log_context: ContextVar[dict[str, Any] | None] = ContextVar("log_context", default=None)


class ContextFilter(logging.Filter):
    """Filter that adds context information to log records."""

    def filter(self, record: logging.LogRecord) -> bool:
        """Add context to log record."""
        context = log_context.get()
        if context:
            for key, value in context.items():
                setattr(record, key, value)
        return True


class JSONFormatter(logging.Formatter):
    """Formatter that outputs logs as JSON."""

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        log_data = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # Add context fields
        context = log_context.get()
        if context:
            log_data["context"] = context

        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        # Add extra fields
        for key, value in record.__dict__.items():
            if key not in [
                "name",
                "msg",
                "args",
                "created",
                "filename",
                "funcName",
                "levelname",
                "levelno",
                "lineno",
                "module",
                "msecs",
                "message",
                "pathname",
                "process",
                "processName",
                "relativeCreated",
                "thread",
                "threadName",
                "exc_info",
                "exc_text",
                "stack_info",
                "taskName",
            ]:
                log_data[key] = value

        return json.dumps(log_data)


class ColoredFormatter(logging.Formatter):
    """Formatter that adds colors to console output."""

    COLORS = {
        "DEBUG": "\033[36m",  # Cyan
        "INFO": "\033[32m",  # Green
        "WARNING": "\033[33m",  # Yellow
        "ERROR": "\033[31m",  # Red
        "CRITICAL": "\033[35m",  # Magenta
    }
    RESET = "\033[0m"

    def format(self, record: logging.LogRecord) -> str:
        """Format log record with colors."""
        # Add color to level name
        levelname = record.levelname
        if levelname in self.COLORS:
            record.levelname = f"{self.COLORS[levelname]}{levelname}{self.RESET}"

        # Format the message
        formatted = super().format(record)

        # Add context if present
        context = log_context.get()
        if context:
            context_str = " ".join(f"{k}={v}" for k, v in context.items())
            formatted += f" [{context_str}]"

        return formatted


def setup_logging(
    level: str = "INFO",
    log_file: str | Path | None = None,
    format_string: str | None = None,
    use_json: bool = False,
    use_colors: bool = True,
) -> logging.Logger:
    """
    Configure logging for the framework.

    Args:
        level: Logging level ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")
        log_file: Optional path to log file
        format_string: Custom format string for log messages
        use_json: Use JSON formatting for file output
        use_colors: Use colored output for console

    Returns:
        Configured root logger
    """
    if format_string is None:
        format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    # Configure root logger
    logger = logging.getLogger()
    logger.setLevel(getattr(logging, level.upper()))

    # Remove existing handlers
    logger.handlers.clear()

    # Add context filter
    context_filter = ContextFilter()

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, level.upper()))
    console_handler.addFilter(context_filter)

    if use_colors and sys.stdout.isatty():
        console_formatter = ColoredFormatter(format_string)
    else:
        console_formatter = logging.Formatter(format_string)

    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    # File handler if specified
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(getattr(logging, level.upper()))
        file_handler.addFilter(context_filter)

        if use_json:
            file_formatter = JSONFormatter()
        else:
            file_formatter = logging.Formatter(format_string)

        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger with the specified name.

    Args:
        name: Logger name (typically __name__)

    Returns:
        Logger instance
    """
    return logging.getLogger(name)


class LogContext:
    """
    Context manager for adding structured context to logs.

    Usage:
        with LogContext(experiment="exp1", epoch=5):
            logger.info("Training started")
            # Logs will include experiment=exp1, epoch=5
    """

    def __init__(self, **kwargs):
        """Initialize with context key-value pairs."""
        self.context = kwargs
        self.token = None

    def __enter__(self):
        """Enter context and set context variables."""
        current = log_context.get()
        if current is None:
            current = {}
        else:
            current = current.copy()
        current.update(self.context)
        self.token = log_context.set(current)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context and restore previous context."""
        if self.token:
            log_context.reset(self.token)


class PerformanceLogger:
    """
    Context manager for logging performance metrics.

    Usage:
        with PerformanceLogger("data_loading"):
            load_data()
            # Automatically logs execution time
    """

    def __init__(
        self,
        operation: str,
        logger: logging.Logger | None = None,
        level: int = logging.INFO,
        log_args: bool = False,
        **kwargs,
    ):
        """
        Initialize performance logger.

        Args:
            operation: Name of the operation being timed
            logger: Logger to use (default: root logger)
            level: Log level for performance message
            log_args: Whether to log additional kwargs
            **kwargs: Additional context to log
        """
        self.operation = operation
        self.logger = logger or logging.getLogger()
        self.level = level
        self.log_args = log_args
        self.kwargs = kwargs
        self.start_time = None

    def __enter__(self):
        """Start timing."""
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Stop timing and log performance."""
        elapsed = time.perf_counter() - self.start_time

        message = f"{self.operation} completed in {elapsed:.4f}s"

        extra = {"operation": self.operation, "elapsed_time": elapsed}
        if self.log_args:
            extra.update(self.kwargs)

        if exc_type is not None:
            extra["error"] = str(exc_val)
            self.logger.error(
                f"{self.operation} failed after {elapsed:.4f}s", extra=extra
            )
        else:
            self.logger.log(self.level, message, extra=extra)


def log_function_call(logger: logging.Logger | None = None, level: int = logging.DEBUG):
    """
    Decorator to log function calls with arguments and execution time.

    Usage:
        @log_function_call()
        def my_function(arg1, arg2):
            pass
    """

    def decorator(func):
        nonlocal logger
        if logger is None:
            logger = logging.getLogger(func.__module__)

        def wrapper(*args, **kwargs):
            func_name = func.__qualname__

            # Log function call
            logger.log(level, f"Calling {func_name}")

            # Execute with performance tracking
            start_time = time.perf_counter()
            try:
                result = func(*args, **kwargs)
                elapsed = time.perf_counter() - start_time
                logger.log(
                    level,
                    f"{func_name} completed in {elapsed:.4f}s",
                    extra={"function": func_name, "elapsed_time": elapsed},
                )
                return result
            except Exception as e:
                elapsed = time.perf_counter() - start_time
                logger.error(
                    f"{func_name} failed after {elapsed:.4f}s: {e}",
                    extra={
                        "function": func_name,
                        "elapsed_time": elapsed,
                        "error": str(e),
                    },
                    exc_info=True,
                )
                raise

        return wrapper

    return decorator


class MetricsLogger:
    """
    Logger for tracking metrics over time.

    Usage:
        metrics = MetricsLogger("training")
        metrics.log("loss", 0.5, step=100)
        metrics.log("accuracy", 0.85, step=100)
    """

    def __init__(self, name: str, logger: logging.Logger | None = None):
        """
        Initialize metrics logger.

        Args:
            name: Name of the metrics group
            logger: Logger to use (default: root logger)
        """
        self.name = name
        self.logger = logger or logging.getLogger(f"metrics.{name}")
        self.metrics = {}

    def log(self, metric_name: str, value: float, step: int | None = None, **kwargs):
        """
        Log a metric value.

        Args:
            metric_name: Name of the metric
            value: Metric value
            step: Optional step/iteration number
            **kwargs: Additional context
        """
        extra = {
            "metric_group": self.name,
            "metric_name": metric_name,
            "value": value,
        }

        if step is not None:
            extra["step"] = step

        extra.update(kwargs)

        message = f"{self.name}.{metric_name}={value}"
        if step is not None:
            message += f" (step={step})"

        self.logger.info(message, extra=extra)

        # Store for later retrieval
        if metric_name not in self.metrics:
            self.metrics[metric_name] = []
        self.metrics[metric_name].append({"value": value, "step": step, **kwargs})

    def get_metrics(self, metric_name: str | None = None) -> dict | list:
        """
        Get logged metrics.

        Args:
            metric_name: Specific metric to retrieve (None for all)

        Returns:
            Metrics data
        """
        if metric_name:
            return self.metrics.get(metric_name, [])
        return self.metrics

    def summary(self) -> dict[str, dict[str, float]]:
        """
        Get summary statistics for all metrics.

        Returns:
            Dictionary with min, max, mean for each metric
        """
        summary = {}
        for name, values in self.metrics.items():
            vals = [v["value"] for v in values]
            if vals:
                summary[name] = {
                    "min": min(vals),
                    "max": max(vals),
                    "mean": sum(vals) / len(vals),
                    "count": len(vals),
                }
        return summary
