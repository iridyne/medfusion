"""
Visualization management for evaluation reports.

Handles plot references and formatting for markdown reports.
"""

from pathlib import Path


class ReportVisualizer:
    """
    Manages visualization plots for evaluation reports.

    Separates plot management from report generation logic.

    Example:
        >>> visualizer = ReportVisualizer(output_dir="results")
        >>> visualizer.add_plot("ROC Curve", "roc_curve.png")
        >>> markdown = visualizer.generate_markdown()
    """

    def __init__(self, output_dir: str | Path):
        """
        Initialize report visualizer.

        Args:
            output_dir: Base directory for relative path calculation
        """
        self.output_dir = Path(output_dir)
        self.plots: dict[str, Path] = {}

    def add_plot(self, name: str, path: str | Path) -> None:
        """
        Add a visualization plot.

        Args:
            name: Display name of the plot
            path: Path to the image file
        """
        self.plots[name] = Path(path)

    def generate_markdown(self) -> str:
        """
        Generate markdown for all plots.

        Returns:
            Markdown string with embedded images
        """
        if not self.plots:
            return ""

        lines = ["## Visualizations\n\n"]

        for name, path in self.plots.items():
            # Calculate relative path if possible for portable links
            try:
                rel_path = path.relative_to(self.output_dir)
            except ValueError:
                rel_path = path

            lines.append(f"### {name}\n\n")
            lines.append(f"![{name}]({rel_path})\n\n")

        return "".join(lines)

    def has_plots(self) -> bool:
        """Check if any plots have been added."""
        return len(self.plots) > 0
