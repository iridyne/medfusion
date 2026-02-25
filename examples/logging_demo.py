"""
Example demonstrating enhanced logging system usage.

This script shows how to use structured logging, context, performance tracking,
and metrics logging.
"""

import time

from med_core.utils.logging import (
    LogContext,
    MetricsLogger,
    PerformanceLogger,
    get_logger,
    log_function_call,
    setup_logging,
)


def example_basic_logging():
    """Example: Basic logging setup."""
    print("=" * 60)
    print("Example 1: Basic Logging")
    print("=" * 60)

    # Setup logging with colors
    setup_logging(level="INFO", use_colors=True)

    logger = get_logger("example")

    logger.debug("This is a debug message")
    logger.info("This is an info message")
    logger.warning("This is a warning message")
    logger.error("This is an error message")
    print()


def example_structured_context():
    """Example: Structured logging with context."""
    print("=" * 60)
    print("Example 2: Structured Context")
    print("=" * 60)

    setup_logging(level="INFO")
    logger = get_logger("training")

    # Add context to all logs within the block
    with LogContext(experiment="lung_cancer_detection", model="resnet50"):
        logger.info("Starting training")

        with LogContext(epoch=1):
            logger.info("Training epoch")
            logger.info("Validation complete")

        with LogContext(epoch=2):
            logger.info("Training epoch")
            logger.info("Validation complete")

    logger.info("Training finished")
    print()


def example_performance_tracking():
    """Example: Performance tracking."""
    print("=" * 60)
    print("Example 3: Performance Tracking")
    print("=" * 60)

    setup_logging(level="INFO")
    logger = get_logger("performance")

    # Track performance of operations
    with PerformanceLogger("data_loading", logger=logger):
        time.sleep(0.1)  # Simulate data loading

    with PerformanceLogger("model_inference", logger=logger):
        time.sleep(0.05)  # Simulate inference

    # Track with additional context
    with PerformanceLogger(
        "batch_processing", logger=logger, log_args=True, batch_size=32
    ):
        time.sleep(0.02)

    print()


def example_function_decorator():
    """Example: Function call logging decorator."""
    print("=" * 60)
    print("Example 4: Function Call Logging")
    print("=" * 60)

    setup_logging(level="INFO")

    @log_function_call()
    def train_model(epochs, learning_rate):
        """Simulate model training."""
        time.sleep(0.05)
        return {"loss": 0.5, "accuracy": 0.85}

    @log_function_call()
    def evaluate_model(model_path):
        """Simulate model evaluation."""
        time.sleep(0.03)
        return {"auc": 0.92}

    # Function calls are automatically logged
    train_result = train_model(epochs=10, learning_rate=0.001)
    eval_result = evaluate_model("outputs/model.pth")

    print(f"Training result: {train_result}")
    print(f"Evaluation result: {eval_result}")
    print()


def example_metrics_logging():
    """Example: Metrics logging."""
    print("=" * 60)
    print("Example 5: Metrics Logging")
    print("=" * 60)

    setup_logging(level="INFO")

    # Create metrics logger
    metrics = MetricsLogger("training")

    # Log metrics during training
    for epoch in range(1, 6):
        loss = 1.0 / epoch  # Simulated decreasing loss
        accuracy = 0.5 + (epoch * 0.08)  # Simulated increasing accuracy

        metrics.log("loss", loss, step=epoch, phase="train")
        metrics.log("accuracy", accuracy, step=epoch, phase="train")

    # Get summary statistics
    summary = metrics.summary()
    print("\nMetrics Summary:")
    for metric_name, stats in summary.items():
        print(f"  {metric_name}:")
        print(f"    Min: {stats['min']:.4f}")
        print(f"    Max: {stats['max']:.4f}")
        print(f"    Mean: {stats['mean']:.4f}")
        print(f"    Count: {stats['count']}")
    print()


def example_json_logging():
    """Example: JSON logging to file."""
    print("=" * 60)
    print("Example 6: JSON Logging")
    print("=" * 60)

    import tempfile
    from pathlib import Path

    # Create temporary log file
    with tempfile.TemporaryDirectory() as tmpdir:
        log_file = Path(tmpdir) / "app.json"

        # Setup JSON logging
        setup_logging(level="INFO", log_file=log_file, use_json=True)
        logger = get_logger("json_example")

        with LogContext(user_id="user123", session_id="sess456"):
            logger.info("User logged in")
            logger.info("Processing request", extra={"request_id": "req789"})
            logger.warning(
                "Rate limit approaching", extra={"requests": 95, "limit": 100}
            )

        # Read and display JSON logs
        print("\nJSON Log Output:")
        with open(log_file) as f:
            import json

            for line in f:
                data = json.loads(line)
                print(f"  {json.dumps(data, indent=2)}")
    print()


def example_combined_features():
    """Example: Combining multiple logging features."""
    print("=" * 60)
    print("Example 7: Combined Features")
    print("=" * 60)

    setup_logging(level="INFO")
    logger = get_logger("combined")
    metrics = MetricsLogger("training")

    # Simulate a training loop with all features
    with LogContext(experiment="exp_001", model="resnet50"):
        logger.info("Experiment started")

        for epoch in range(1, 4):
            with LogContext(epoch=epoch):
                # Track epoch performance
                with PerformanceLogger(f"epoch_{epoch}", logger=logger):
                    # Simulate training
                    time.sleep(0.02)

                    # Log metrics
                    loss = 1.0 / epoch
                    metrics.log("loss", loss, step=epoch)

                    logger.info(f"Epoch {epoch} complete")

        logger.info("Experiment finished")

        # Show summary
        summary = metrics.summary()
        logger.info(f"Final loss: {summary['loss']['min']:.4f}")
    print()


def example_error_logging():
    """Example: Error logging with context."""
    print("=" * 60)
    print("Example 8: Error Logging")
    print("=" * 60)

    setup_logging(level="ERROR")
    logger = get_logger("errors")

    with LogContext(experiment="exp_002", epoch=5):
        try:
            # Simulate an error
            raise ValueError("Loss became NaN")
        except ValueError:
            logger.exception("Training failed")

    print()


def main():
    """Run all examples."""
    print("\n" + "=" * 60)
    print("Enhanced Logging System Examples")
    print("=" * 60 + "\n")

    example_basic_logging()
    example_structured_context()
    example_performance_tracking()
    example_function_decorator()
    example_metrics_logging()
    example_json_logging()
    example_combined_features()
    example_error_logging()

    print("=" * 60)
    print("Summary")
    print("=" * 60)
    print("✅ Structured logging with context")
    print("✅ Performance tracking with timing")
    print("✅ Metrics logging with statistics")
    print("✅ JSON output for log aggregation")
    print("✅ Colored console output")
    print("✅ Function call decoration")
    print()


if __name__ == "__main__":
    main()
