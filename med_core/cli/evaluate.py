"""Evaluation command implementation."""

import argparse
import logging
import sys
from collections.abc import Sequence
from pathlib import Path

from med_core.configs import load_config
from med_core.configs.config_loader import resolve_config_path
from med_core.postprocessing import build_results_artifacts

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def evaluate(
    argv: Sequence[str] | None = None,
    prog: str = "med-evaluate",
) -> None:
    """Command-line entry point for evaluation."""
    parser = argparse.ArgumentParser(
        prog=prog,
        description=(
            "Evaluate a trained checkpoint and generate the canonical result "
            "artifact contract."
        ),
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to config file used for training",
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to model checkpoint",
    )
    parser.add_argument(
        "--output-dir", type=str, default="evaluation_results", help="Output directory",
    )
    parser.add_argument(
        "--split", type=str, default="test", choices=["val", "test", "train"],
    )
    args = parser.parse_args(argv)

    config_path = resolve_config_path(args.config)
    checkpoint_path = Path(args.checkpoint)
    output_dir = Path(args.output_dir)

    try:
        config = load_config(config_path)
    except FileNotFoundError as exc:
        logger.error("Config file not found: %s", exc)
        sys.exit(1)
    except Exception as exc:
        logger.error("Failed to load config: %s", exc)
        sys.exit(1)

    if not checkpoint_path.exists():
        logger.error("Checkpoint not found: %s", checkpoint_path)
        sys.exit(1)

    logger.info(
        "Evaluating %s on split=%s using the canonical artifact contract...",
        checkpoint_path,
        args.split,
    )
    try:
        result = build_results_artifacts(
            config_path=config_path,
            checkpoint_path=checkpoint_path,
            output_dir=output_dir,
            split=args.split,
        )
    except FileNotFoundError as exc:
        logger.error("Evaluation input missing: %s", exc)
        sys.exit(1)
    except Exception as exc:
        logger.error("Evaluation failed: %s", exc)
        sys.exit(1)

    logger.info(
        "Evaluation complete for %s. Output=%s, report=%s, metrics=%s, validation=%s",
        config.experiment_name,
        result.output_dir,
        result.artifact_paths.get("report_path"),
        result.artifact_paths.get("metrics_path"),
        result.artifact_paths.get("validation_path"),
    )
