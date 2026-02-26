"""
Report generation for evaluation results.

Provides a modular report generator that composes metrics calculation
and visualization management.
"""

import json
import logging
import platform
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import torch

from med_core.evaluation.metrics_calculator import MetricsCalculator
from med_core.evaluation.report_visualizer import ReportVisualizer
from med_core.version import __version__

logger = logging.getLogger(__name__)


class ReportGenerator:
    """
    Generates comprehensive evaluation reports.

    Composes MetricsCalculator and ReportVisualizer to produce
    structured markdown reports for experiments.

    Example:
        >>> generator = ReportGenerator(
        ...     experiment_name="My Experiment",
        ...     output_dir="results"
        ... )
        >>> generator.add_metrics(metrics)
        >>> generator.add_plot("ROC Curve", "roc.png")
        >>> report_path = generator.generate()
    """

    def __init__(
        self,
        experiment_name: str,
        output_dir: str | Path,
        description: str = "",
    ):
        """
        Initialize report generator.

        Args:
            experiment_name: Name of the experiment
            output_dir: Directory to save the report
            description: Optional description of the experiment
        """
        self.experiment_name = experiment_name
        self.output_dir = Path(output_dir)
        self.description = description
        self.timestamp = datetime.now()

        # Components
        self.metrics_calculator = MetricsCalculator()
        self.visualizer = ReportVisualizer(output_dir)

        # Data
        self.metrics: object | dict | None = None
        self.config: dict[str, Any] = {}
        self.system_info: dict[str, str] = self._collect_system_info()

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _collect_system_info(self) -> dict[str, str]:
        """Collect system and environment information."""
        return {
            "Timestamp": self.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            "Med-Core Version": __version__,
            "Python Version": sys.version.split()[0],
            "PyTorch Version": torch.__version__,
            "OS": platform.platform(),
            "Device": (
                torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
            ),
        }

    def add_metrics(self, metrics: object | dict) -> None:
        """
        Add evaluation metrics to the report.

        Args:
            metrics: Metrics object or dictionary
        """
        self.metrics = metrics

    def add_plot(self, name: str, path: str | Path) -> None:
        """
        Add a visualization plot to the report.

        Args:
            name: Display name of the plot
            path: Path to the image file
        """
        self.visualizer.add_plot(name, path)

    def add_config(self, config: dict[str, Any]) -> None:
        """
        Add experiment configuration.

        Args:
            config: Configuration dictionary
        """
        self.config = config

    def generate(self, filename: str = "report.md") -> Path:
        """
        Generate and save the markdown report.

        Args:
            filename: Output filename

        Returns:
            Path to the generated report
        """
        report_path = self.output_dir / filename

        with open(report_path, "w", encoding="utf-8") as f:
            # Header
            f.write(f"# Experiment Report: {self.experiment_name}\n\n")
            if self.description:
                f.write(f"**Description:** {self.description}\n\n")

            # System Info
            f.write(self._generate_system_info_markdown())

            # Metrics
            if self.metrics:
                f.write(self._generate_metrics_markdown())

            # Visualizations
            if self.visualizer.has_plots():
                f.write(self.visualizer.generate_markdown())

            # Configuration
            if self.config:
                f.write(self._generate_config_markdown())

        logger.info(f"Report generated at {report_path}")
        return report_path

    def _generate_system_info_markdown(self) -> str:
        """Generate system information section."""
        lines = ["## System Information\n\n"]
        lines.append("| Property | Value |\n")
        lines.append("| :--- | :--- |\n")
        for k, v in self.system_info.items():
            lines.append(f"| {k} | {v} |\n")
        lines.append("\n")
        return "".join(lines)

    def _generate_metrics_markdown(self) -> str:
        """Generate metrics section."""
        lines = ["## Evaluation Metrics\n\n"]

        # Determine metrics type and format accordingly
        if isinstance(self.metrics, dict):
            lines.append(self._format_multiclass_metrics())
        elif hasattr(self.metrics, "auc_roc") or hasattr(self.metrics, "accuracy"):
            lines.append(self._format_binary_metrics())
        else:
            lines.append("Unknown metrics format.\n\n")

        return "".join(lines)

    def _format_binary_metrics(self) -> str:
        """Format binary classification metrics."""
        formatted = self.metrics_calculator.format_binary_metrics(self.metrics)
        lines = []

        # Performance summary
        lines.append("### Performance Summary\n\n")
        lines.append("| Metric | Value | 95% CI |\n")
        lines.append("| :--- | :--- | :--- |\n")

        for metric_name, metric_data in formatted["performance"].items():
            value = metric_data["value"]
            ci = metric_data["ci"]
            if metric_data.get("highlight"):
                lines.append(f"| **{metric_name}** | **{value}** | {ci} |\n")
            else:
                lines.append(f"| {metric_name} | {value} | {ci} |\n")
        lines.append("\n")

        # Confusion matrix
        cm = formatted["confusion_matrix"]
        lines.append("### Confusion Matrix\n\n")
        lines.append("| | Predicted Negative | Predicted Positive |\n")
        lines.append("| :--- | :---: | :---: |\n")
        lines.append(
            f"| **Actual Negative** | {cm['true_negatives']} | {cm['false_positives']} |\n",
        )
        lines.append(
            f"| **Actual Positive** | {cm['false_negatives']} | {cm['true_positives']} |\n",
        )
        lines.append("\n")

        return "".join(lines)

    def _format_multiclass_metrics(self) -> str:
        """Format multiclass classification metrics."""
        formatted = self.metrics_calculator.format_multiclass_metrics(self.metrics)
        lines = []

        # Overall performance
        lines.append("### Performance Summary\n\n")
        lines.append("| Metric | Value |\n")
        lines.append("| :--- | :--- |\n")

        for metric_name, metric_data in formatted["overall"].items():
            value = metric_data["value"]
            if metric_data.get("highlight"):
                lines.append(f"| **{metric_name}** | **{value}** |\n")
            else:
                lines.append(f"| {metric_name} | {value} |\n")
        lines.append("\n")

        # Per-class performance
        if "per_class" in formatted:
            lines.append("### Per-Class Performance\n\n")
            lines.append("| Class | Precision | Recall | F1 Score | Support |\n")
            lines.append("| :--- | :--- | :--- | :--- | :--- |\n")

            for cls, metrics in formatted["per_class"].items():
                lines.append(
                    f"| {cls} | {metrics['precision']} | "
                    f"{metrics['recall']} | {metrics['f1']} | "
                    f"{metrics['support']} |\n",
                )
            lines.append("\n")

        return "".join(lines)

    def _generate_config_markdown(self) -> str:
        """Generate configuration section."""
        lines = ["## Configuration\n\n"]
        lines.append("```json\n")
        lines.append(json.dumps(self.config, indent=2, default=str))
        lines.append("\n```\n\n")
        return "".join(lines)


def generate_evaluation_report(
    metrics: object | dict,
    output_dir: str | Path,
    experiment_name: str = "Evaluation",
    plots: dict[str, Path] | None = None,
    config: dict[str, Any] | None = None,
) -> Path:
    """
    Convenience function to generate a report in one call.

    Args:
        metrics: Calculated metrics object
        output_dir: Directory to save the report
        experiment_name: Name of the experiment
        plots: Dictionary of plot names to paths
        config: Experiment configuration dict

    Returns:
        Path to the generated report
    """
    generator = ReportGenerator(experiment_name, output_dir)
    generator.add_metrics(metrics)

    if plots:
        for name, path in plots.items():
            generator.add_plot(name, path)

    if config:
        generator.add_config(config)

    return generator.generate()
