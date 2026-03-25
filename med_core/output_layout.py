"""Shared run-output layout helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RunOutputLayout:
    """Canonical directory layout for a training or import run."""

    root_dir: Path

    def __init__(self, root_dir: str | Path):
        object.__setattr__(self, "root_dir", Path(root_dir))

    @property
    def checkpoints_dir(self) -> Path:
        return self.root_dir / "checkpoints"

    @property
    def logs_dir(self) -> Path:
        return self.root_dir / "logs"

    @property
    def reports_dir(self) -> Path:
        return self.root_dir / "reports"

    @property
    def metrics_dir(self) -> Path:
        return self.root_dir / "metrics"

    @property
    def artifacts_dir(self) -> Path:
        return self.root_dir / "artifacts"

    @property
    def history_path(self) -> Path:
        return self.logs_dir / "history.json"

    @property
    def training_log_path(self) -> Path:
        return self.logs_dir / "training.log"

    @property
    def generated_config_path(self) -> Path:
        return self.artifacts_dir / "training-config.yaml"

    @property
    def config_snapshot_path(self) -> Path:
        return self.artifacts_dir / "training-config.json"

    @property
    def summary_path(self) -> Path:
        return self.reports_dir / "summary.json"

    @property
    def report_path(self) -> Path:
        return self.reports_dir / "report.md"

    @property
    def metrics_path(self) -> Path:
        return self.metrics_dir / "metrics.json"

    @property
    def validation_path(self) -> Path:
        return self.metrics_dir / "validation.json"

    @property
    def predictions_path(self) -> Path:
        return self.metrics_dir / "predictions.json"

    @property
    def roc_curve_json_path(self) -> Path:
        return self.metrics_dir / "roc_curve.json"

    @property
    def confusion_matrix_json_path(self) -> Path:
        return self.metrics_dir / "confusion_matrix.json"

    @property
    def visualizations_dir(self) -> Path:
        return self.artifacts_dir / "visualizations"

    def ensure_exists(self) -> RunOutputLayout:
        for path in (
            self.root_dir,
            self.checkpoints_dir,
            self.logs_dir,
            self.reports_dir,
            self.metrics_dir,
            self.artifacts_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)
        return self


def resolve_run_output_dir(
    *,
    config_output_dir: str | Path,
    checkpoint_path: Path,
    override: str | Path | None,
) -> Path:
    """Resolve the run root, preserving explicit overrides."""
    if override is not None:
        return Path(override)
    if checkpoint_path.parent.name == "checkpoints":
        return checkpoint_path.parent.parent
    return Path(config_output_dir)
