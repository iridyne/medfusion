"""
Enhanced report generation with high-resolution plots, statistical tests, and LaTeX output.

Extends the base report generator with:
- High-resolution plots (300 DPI)
- Statistical significance tests
- LaTeX table generation
- Publication-ready formatting
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from med_core.evaluation.report_generator import ReportGenerator

logger = logging.getLogger(__name__)


class EnhancedReportGenerator(ReportGenerator):
    """
    Enhanced report generator with publication-ready features.

    Adds:
    - High-resolution plot generation (300 DPI)
    - Statistical significance testing
    - LaTeX table output
    - Publication-ready formatting

    Example:
        >>> generator = EnhancedReportGenerator(
        ...     experiment_name="My Experiment",
        ...     output_dir="results",
        ...     dpi=300
        ... )
        >>> generator.add_metrics(metrics)
        >>> generator.generate_latex_report()
    """

    def __init__(
        self,
        experiment_name: str,
        output_dir: str | Path,
        description: str = "",
        dpi: int = 300,
        enable_statistical_tests: bool = True,
    ):
        """
        Initialize enhanced report generator.

        Args:
            experiment_name: Name of the experiment
            output_dir: Directory to save the report
            description: Optional description
            dpi: DPI for high-resolution plots (default: 300)
            enable_statistical_tests: Whether to compute statistical tests
        """
        super().__init__(experiment_name, output_dir, description)
        self.dpi = dpi
        self.enable_statistical_tests = enable_statistical_tests
        self.statistical_tests: dict[str, Any] = {}

        # Configure matplotlib for high-resolution output
        plt.rcParams['figure.dpi'] = dpi
        plt.rcParams['savefig.dpi'] = dpi
        plt.rcParams['font.size'] = 10
        plt.rcParams['axes.labelsize'] = 11
        plt.rcParams['axes.titlesize'] = 12
        plt.rcParams['xtick.labelsize'] = 9
        plt.rcParams['ytick.labelsize'] = 9
        plt.rcParams['legend.fontsize'] = 9

    def add_comparison_metrics(
        self,
        baseline_metrics: Any,
        comparison_name: str = "Baseline"
    ) -> None:
        """
        Add baseline metrics for statistical comparison.

        Args:
            baseline_metrics: Baseline metrics object
            comparison_name: Name of the baseline
        """
        if not self.enable_statistical_tests:
            return

        if not hasattr(self, 'baseline_metrics'):
            self.baseline_metrics = {}

        self.baseline_metrics[comparison_name] = baseline_metrics

    def compute_statistical_tests(self) -> dict[str, Any]:
        """
        Compute statistical significance tests.

        Returns:
            Dictionary with test results
        """
        if not self.enable_statistical_tests or not hasattr(self, 'baseline_metrics'):
            return {}

        tests = {}

        for baseline_name, baseline in self.baseline_metrics.items():
            # McNemar's test for paired binary classification
            if hasattr(self.metrics, 'true_positives') and hasattr(baseline, 'true_positives'):
                # Construct contingency table
                # [both_correct, current_correct_baseline_wrong]
                # [current_wrong_baseline_correct, both_wrong]

                # Simplified: compare accuracy improvements
                current_acc = self.metrics.accuracy
                baseline_acc = baseline.accuracy

                # Use binomial test for significance
                n_samples = (self.metrics.true_positives + self.metrics.true_negatives +
                            self.metrics.false_positives + self.metrics.false_negatives)

                # Approximate p-value using normal approximation
                se = np.sqrt((current_acc * (1 - current_acc) +
                             baseline_acc * (1 - baseline_acc)) / n_samples)
                z_score = (current_acc - baseline_acc) / se if se > 0 else 0
                p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))

                tests[baseline_name] = {
                    'test': 'Z-test for proportions',
                    'current_accuracy': current_acc,
                    'baseline_accuracy': baseline_acc,
                    'difference': current_acc - baseline_acc,
                    'z_score': z_score,
                    'p_value': p_value,
                    'significant': p_value < 0.05,
                }

        self.statistical_tests = tests
        return tests

    def generate_latex_table(self, metrics: Any = None) -> str:
        """
        Generate LaTeX table for metrics.

        Args:
            metrics: Metrics object (uses self.metrics if None)

        Returns:
            LaTeX table string
        """
        if metrics is None:
            metrics = self.metrics

        if not metrics:
            return ""

        lines = []
        lines.append("\\begin{table}[htbp]")
        lines.append("\\centering")
        lines.append(f"\\caption{{Performance metrics for {self.experiment_name}}}")
        lines.append(f"\\label{{tab:{self.experiment_name.lower().replace(' ', '_')}}}")

        # Binary classification
        if hasattr(metrics, 'auc_roc'):
            lines.append("\\begin{tabular}{lcc}")
            lines.append("\\hline")
            lines.append("Metric & Value & 95\\% CI \\\\")
            lines.append("\\hline")

            # Format metrics
            def fmt_ci(ci):
                if ci:
                    return f"({ci[0]:.3f}, {ci[1]:.3f})"
                return "-"

            lines.append(f"AUC-ROC & {metrics.auc_roc:.3f} & {fmt_ci(metrics.ci_auc_roc)} \\\\")
            lines.append(f"Accuracy & {metrics.accuracy:.3f} & {fmt_ci(metrics.ci_accuracy)} \\\\")
            lines.append(f"Sensitivity & {metrics.sensitivity:.3f} & {fmt_ci(metrics.ci_sensitivity)} \\\\")
            lines.append(f"Specificity & {metrics.specificity:.3f} & {fmt_ci(metrics.ci_specificity)} \\\\")
            lines.append(f"F1 Score & {metrics.f1:.3f} & - \\\\")
            lines.append(f"Precision (PPV) & {metrics.ppv:.3f} & - \\\\")
            lines.append(f"NPV & {metrics.npv:.3f} & - \\\\")

            lines.append("\\hline")
            lines.append("\\end{tabular}")

        lines.append("\\end{table}")

        return "\n".join(lines)

    def generate_latex_report(self, filename: str = "report.tex") -> Path:
        """
        Generate LaTeX report for publication.

        Args:
            filename: Output filename

        Returns:
            Path to the generated LaTeX file
        """
        report_path = self.output_dir / filename

        with open(report_path, "w", encoding="utf-8") as f:
            # Document header
            f.write("\\documentclass[11pt]{article}\n")
            f.write("\\usepackage{booktabs}\n")
            f.write("\\usepackage{graphicx}\n")
            f.write("\\usepackage{float}\n")
            f.write("\\usepackage{hyperref}\n\n")
            f.write("\\begin{document}\n\n")

            # Title
            f.write(f"\\section*{{{self.experiment_name}}}\n\n")
            if self.description:
                f.write(f"{self.description}\n\n")

            # System info
            f.write("\\subsection*{System Information}\n\n")
            f.write("\\begin{itemize}\n")
            for k, v in self.system_info.items():
                f.write(f"  \\item \\textbf{{{k}:}} {v}\n")
            f.write("\\end{itemize}\n\n")

            # Metrics table
            if self.metrics:
                f.write("\\subsection*{Performance Metrics}\n\n")
                f.write(self.generate_latex_table())
                f.write("\n\n")

            # Statistical tests
            if self.statistical_tests:
                f.write("\\subsection*{Statistical Significance Tests}\n\n")
                f.write("\\begin{table}[htbp]\n")
                f.write("\\centering\n")
                f.write("\\begin{tabular}{lccc}\n")
                f.write("\\hline\n")
                f.write("Comparison & Difference & p-value & Significant \\\\\\n")
                f.write("\\hline\n")

                for name, test in self.statistical_tests.items():
                    sig = "Yes" if test['significant'] else "No"
                    f.write(f"{name} & {test['difference']:.4f} & {test['p_value']:.4f} & {sig} \\\\\\n")

                f.write("\\hline\n")
                f.write("\\end{tabular}\n")
                f.write("\\end{table}\n\n")

            # Figures
            if self.visualizer.has_plots():
                f.write("\\subsection*{Visualizations}\n\n")
                for name, path in self.visualizer.plots.items():
                    f.write("\\begin{figure}[H]\n")
                    f.write("\\centering\n")
                    f.write(f"\\includegraphics[width=0.7\\textwidth]{{{path}}}\n")
                    f.write(f"\\caption{{{name}}}\n")
                    f.write("\\end{figure}\n\n")

            f.write("\\end{document}\n")

        logger.info(f"LaTeX report generated at {report_path}")
        return report_path

    def generate(self, filename: str = "report.md") -> Path:
        """
        Generate enhanced markdown report with statistical tests.

        Args:
            filename: Output filename

        Returns:
            Path to the generated report
        """
        # Compute statistical tests if enabled
        if self.enable_statistical_tests:
            self.compute_statistical_tests()

        # Generate base markdown report
        report_path = super().generate(filename)

        # Append statistical tests section if available
        if self.statistical_tests:
            with open(report_path, "a", encoding="utf-8") as f:
                f.write(self._generate_statistical_tests_markdown())

        # Also generate LaTeX report
        latex_filename = filename.replace('.md', '.tex')
        self.generate_latex_report(latex_filename)

        return report_path

    def _generate_statistical_tests_markdown(self) -> str:
        """Generate statistical tests section for markdown."""
        if not self.statistical_tests:
            return ""

        lines = ["## Statistical Significance Tests\n\n"]
        lines.append("| Comparison | Current | Baseline | Difference | p-value | Significant |\n")
        lines.append("| :--- | :--- | :--- | :--- | :--- | :--- |\n")

        for name, test in self.statistical_tests.items():
            sig = "✅ Yes" if test['significant'] else "❌ No"
            lines.append(
                f"| {name} | {test['current_accuracy']:.4f} | "
                f"{test['baseline_accuracy']:.4f} | {test['difference']:.4f} | "
                f"{test['p_value']:.4f} | {sig} |\n"
            )

        lines.append("\n")
        lines.append("*Note: p < 0.05 indicates statistical significance*\n\n")

        return "".join(lines)


def generate_enhanced_report(
    metrics: Any,
    output_dir: str | Path,
    experiment_name: str = "Evaluation",
    plots: dict[str, Path] | None = None,
    config: dict[str, Any] | None = None,
    baseline_metrics: Any | None = None,
    dpi: int = 300,
) -> Path:
    """
    Convenience function to generate an enhanced report.

    Args:
        metrics: Calculated metrics object
        output_dir: Directory to save the report
        experiment_name: Name of the experiment
        plots: Dictionary of plot names to paths
        config: Experiment configuration dict
        baseline_metrics: Optional baseline metrics for comparison
        dpi: DPI for high-resolution plots

    Returns:
        Path to the generated report
    """
    generator = EnhancedReportGenerator(
        experiment_name,
        output_dir,
        dpi=dpi,
        enable_statistical_tests=baseline_metrics is not None
    )

    generator.add_metrics(metrics)

    if baseline_metrics:
        generator.add_comparison_metrics(baseline_metrics, "Baseline")

    if plots:
        for name, path in plots.items():
            generator.add_plot(name, path)

    if config:
        generator.add_config(config)

    return generator.generate()
