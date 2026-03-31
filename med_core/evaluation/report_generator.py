"""
Report generation for evaluation results.

Provides a modular report generator that composes metrics calculation
and visualization management.
"""

import json
import os
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


def _relative_markdown_asset_path(
    report_path: str | Path, target_path: str | Path | None
) -> str | None:
    if not target_path:
        return None
    return os.path.relpath(Path(target_path), start=Path(report_path).parent).replace(
        "\\", "/"
    )


def _report_value(value: Any) -> str:
    if value is None or value == "":
        return "-"
    return str(value)


def _format_importance_method(method: Any) -> str:
    if method in {"logistic_surrogate_shap", "ridge_surrogate_shap"}:
        return "SHAP-style surrogate"
    return _report_value(method)


def _format_score_name(score_name: Any) -> str:
    mapping = {
        "positive_class_probability": "阳性预测概率",
        "max_predicted_probability": "最高预测概率",
        "risk_head": "模型风险概率",
        "probability": "预测概率",
    }
    return mapping.get(str(score_name), _report_value(score_name))


def _append_markdown_image(
    lines: list[str], title: str, report_path: str | Path, target_path: str | Path | None
) -> None:
    relative_path = _relative_markdown_asset_path(report_path, target_path)
    if not relative_path:
        return
    lines.extend([f"![{title}]({relative_path})", ""])


def generate_result_artifact_report(
    *,
    report_path: str | Path,
    experiment_name: str,
    config_path: str | Path,
    checkpoint_path: str | Path,
    split: str,
    metrics_payload: dict[str, Any],
    validation_payload: dict[str, Any],
    history_payload: dict[str, Any],
    artifact_paths: dict[str, str],
    artifact_metadata: dict[str, Any],
    backbone: str | None = None,
) -> Path:
    """Generate the canonical markdown report for build-results artifacts."""
    resolved_report_path = Path(report_path)
    resolved_report_path.parent.mkdir(parents=True, exist_ok=True)

    history_entries = history_payload.get("entries", [])
    validation_overview = validation_payload.get("overview", {})
    per_class_rows = validation_payload.get("per_class", [])
    top_misclassifications = validation_payload.get("prediction_summary", {}).get(
        "top_misclassifications", []
    )
    survival_payload = validation_payload.get("survival") or {}
    importance_payload = validation_payload.get("global_feature_importance") or {}

    report_lines = [
        f"# {experiment_name} 结果报告",
        "",
        "## Contract Metadata",
        "",
        f"- Schema Version: {_report_value(artifact_metadata.get('schema_version'))}",
        f"- Generated By: {_report_value(artifact_metadata.get('generated_by'))}",
        f"- Generated At: {_report_value(artifact_metadata.get('generated_at'))}",
        f"- Source Config Path: {_report_value(artifact_metadata.get('source_config_path'))}",
        f"- Checkpoint Path: {_report_value(artifact_metadata.get('checkpoint_path'))}",
        f"- Split: {_report_value(artifact_metadata.get('split'))}",
        "",
        "## 运行来源",
        "",
        f"- Config: {config_path}",
        f"- Checkpoint: {checkpoint_path}",
        f"- Split: {split}",
        "",
        "## 实验摘要",
        "",
        f"- 数据集: {_report_value(validation_payload.get('dataset', {}).get('name'))}",
        f"- Backbone: {_report_value(backbone)}",
        f"- 总体准确率: {_report_value(metrics_payload.get('accuracy'))}",
        f"- 区分能力（AUC）: {_report_value(metrics_payload.get('auc'))}",
        f"- F1: {_report_value(metrics_payload.get('f1_score'))}",
        f"- 平衡准确率: {_report_value(metrics_payload.get('balanced_accuracy'))}",
        f"- C-index: {_report_value(metrics_payload.get('c_index'))}",
        f"- 最佳轮次: {_report_value(metrics_payload.get('best_epoch'))}",
        f"- 最佳准确率: {_report_value(metrics_payload.get('best_accuracy'))}",
        f"- 最佳损失: {_report_value(metrics_payload.get('best_loss'))}",
        "",
        "## Validation 概览",
        "",
        f"- 样本数: {_report_value(validation_overview.get('sample_count'))}",
        f"- 类别数: {_report_value(validation_overview.get('num_classes'))}",
        f"- 正类标签: {_report_value(validation_overview.get('positive_class_label'))}",
        f"- 正类占比: {_report_value(validation_overview.get('positive_prevalence'))}",
        f"- 宏平均 F1: {_report_value(validation_overview.get('macro_f1'))}",
        f"- 加权 F1: {_report_value(validation_overview.get('weighted_f1'))}",
        f"- 平均置信度: {_report_value(validation_overview.get('mean_confidence'))}",
        f"- 错误率: {_report_value(validation_overview.get('error_rate'))}",
        "",
        "## Artifact Index",
        "",
        f"- metrics.json: {_report_value(artifact_paths.get('metrics_path'))}",
        f"- validation.json: {_report_value(artifact_paths.get('validation_path'))}",
        f"- predictions.json: {_report_value(artifact_paths.get('predictions_path') or artifact_paths.get('prediction_path'))}",
        f"- summary.json: {_report_value(artifact_paths.get('summary_path'))}",
        f"- report.md: {_report_value(artifact_paths.get('report_path'))}",
        f"- history.json: {_report_value(artifact_paths.get('history_path'))}",
        f"- training.log: {_report_value(artifact_paths.get('log_path'))}",
        "",
        "## Per-class Metrics",
        "",
        "| Class | Support | Prevalence | Precision | Recall | F1 | Predicted |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        *[
            "| {label} | {support} | {prevalence} | {precision} | {recall} | {f1_score} | {predicted_count} |".format(
                **row
            )
            for row in per_class_rows
        ],
        "",
        "## Visualizations",
        "",
    ]

    for title, path in [
        ("训练曲线", artifact_paths.get("training_curves_plot_path")),
        ("ROC 曲线（区分能力）", artifact_paths.get("roc_curve_plot_path")),
        ("混淆矩阵（阳性/阴性判别情况）", artifact_paths.get("confusion_matrix_plot_path")),
        ("注意力统计图", artifact_paths.get("attention_statistics_plot_path")),
        ("Kaplan-Meier 生存曲线", artifact_paths.get("kaplan_meier_plot_path")),
        ("风险分层分布图", artifact_paths.get("risk_score_distribution_plot_path")),
        ("关键影响因素条形图", artifact_paths.get("feature_importance_bar_plot_path")),
        ("关键影响因素散点图", artifact_paths.get("feature_importance_beeswarm_plot_path")),
    ]:
        _append_markdown_image(report_lines, title, resolved_report_path, path)

    report_lines.extend(
        [
            "## Threshold Analysis",
            "",
            f"- 最优阈值: {_report_value(metrics_payload.get('optimal_threshold'))}",
            f"- 敏感度: {_report_value(metrics_payload.get('sensitivity'))}",
            f"- 特异度: {_report_value(metrics_payload.get('specificity'))}",
            f"- PPV: {_report_value(metrics_payload.get('ppv'))}",
            f"- NPV: {_report_value(metrics_payload.get('npv'))}",
            "",
            "## 常见误分类",
            "",
            *(
                [
                    f"- {item['actual']} -> {item['predicted']}: {item['count']}"
                    for item in top_misclassifications
                ]
                if top_misclassifications
                else ["- 无明显误分类聚集"]
            ),
            "",
            "## History",
            "",
            f"- total_entries: {len(history_entries)}",
        ]
    )

    if survival_payload:
        report_lines.extend(
            [
                "",
                "## 生存分析",
                "",
                f"- 生存分析样本数: {_report_value(survival_payload.get('sample_count'))}",
                f"- 事件发生率: {_report_value(survival_payload.get('event_rate'))}",
                f"- C-index: {_report_value(survival_payload.get('c_index'))}",
                f"- 风险分层依据: {_format_score_name(survival_payload.get('risk_score_source'))}",
            ]
        )

    if importance_payload:
        report_lines.extend(
            [
                "",
                "## 关键影响因素",
                "",
                f"- 方法说明: {_format_importance_method(importance_payload.get('method'))}",
                f"- 关键影响因素评分来源: {_format_score_name(importance_payload.get('score_name'))}",
                *[
                    f"- 关键因素: {item['feature']}: {item['mean_abs_contribution']}"
                    for item in importance_payload.get("top_features", [])[:5]
                ],
            ]
        )

    resolved_report_path.write_text(
        "\n".join(line for line in report_lines if line is not None),
        encoding="utf-8",
    )
    logger.info("Result artifact report generated at %s", resolved_report_path)
    return resolved_report_path


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
