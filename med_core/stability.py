"""Reusable helpers for multi-seed experiment stability studies."""

from __future__ import annotations

import csv
import json
import math
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from med_core.output_layout import RunOutputLayout

_MAXIMIZE_METRIC_HINTS = (
    "acc",
    "accuracy",
    "auc",
    "auroc",
    "f1",
    "precision",
    "recall",
    "specificity",
    "sensitivity",
    "dice",
    "iou",
    "c_index",
    "concordance",
)
_MINIMIZE_METRIC_HINTS = (
    "loss",
    "error",
    "mae",
    "mse",
    "rmse",
    "nll",
    "brier",
)


@dataclass(frozen=True)
class SeedRunArtifacts:
    """Artifact locations for one seed run."""

    seed: int
    output_dir: Path
    metrics_path: Path
    history_path: Path | None = None
    report_path: Path | None = None


@dataclass(frozen=True)
class StabilityStudyResult:
    """Generated summary artifact paths for a stability study."""

    study_dir: Path
    seed_dirs: dict[int, Path]
    summary_json_path: Path
    summary_csv_path: Path
    summary_md_path: Path


def parse_seeds(seeds: str | Iterable[int]) -> list[int]:
    """Parse seed values from either a CSV string or an iterable."""
    raw_values: list[Any]
    if isinstance(seeds, str):
        raw_values = [part.strip() for part in seeds.split(",")]
    else:
        raw_values = list(seeds)

    parsed: list[int] = []
    seen: set[int] = set()
    for raw_value in raw_values:
        if raw_value in {None, ""}:
            continue
        seed = int(raw_value)
        if seed in seen:
            continue
        seen.add(seed)
        parsed.append(seed)

    if not parsed:
        raise ValueError("at least one seed is required")

    return parsed


def default_seed_output_dir(study_dir: str | Path, seed: int) -> Path:
    """Return the canonical per-seed output directory for a study."""
    return RunOutputLayout(study_dir).seed_output_dir(seed)


def default_seed_artifacts(seed: int, output_dir: Path) -> SeedRunArtifacts:
    """Resolve canonical MedFusion artifact paths for a seed run."""
    layout = RunOutputLayout(output_dir)
    return SeedRunArtifacts(
        seed=seed,
        output_dir=output_dir,
        metrics_path=layout.metrics_path,
        history_path=layout.history_path,
        report_path=layout.report_path,
    )


def run_stability_study(
    *,
    seeds: str | Sequence[int],
    study_dir: str | Path,
    run_seed: Callable[[int, Path], None],
    resolve_seed_artifacts: Callable[[int, Path], SeedRunArtifacts] | None = None,
    study_name: str | None = None,
) -> StabilityStudyResult:
    """Execute repeated runs for multiple seeds and write aggregate summaries."""
    parsed_seeds = parse_seeds(seeds)
    layout = RunOutputLayout(study_dir)
    layout.root_dir.mkdir(parents=True, exist_ok=True)
    layout.seed_runs_dir.mkdir(parents=True, exist_ok=True)
    layout.stability_dir.mkdir(parents=True, exist_ok=True)

    artifacts_resolver = resolve_seed_artifacts or default_seed_artifacts
    seed_dirs: dict[int, Path] = {}
    per_seed_rows: list[dict[str, Any]] = []

    for seed in parsed_seeds:
        seed_dir = default_seed_output_dir(layout.root_dir, seed)
        seed_dir.mkdir(parents=True, exist_ok=True)
        run_seed(seed, seed_dir)

        artifacts = artifacts_resolver(seed, seed_dir)
        seed_dirs[seed] = seed_dir
        per_seed_rows.append(_load_seed_row(artifacts))

    summary_payload = _build_summary_payload(
        study_dir=layout.root_dir,
        per_seed_rows=per_seed_rows,
        study_name=study_name or "Multi-Seed Stability",
    )

    layout.stability_summary_json_path.write_text(
        json.dumps(summary_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    _write_summary_csv(layout.stability_summary_csv_path, per_seed_rows, summary_payload)
    _write_summary_md(layout.stability_summary_md_path, summary_payload)

    return StabilityStudyResult(
        study_dir=layout.root_dir,
        seed_dirs=seed_dirs,
        summary_json_path=layout.stability_summary_json_path,
        summary_csv_path=layout.stability_summary_csv_path,
        summary_md_path=layout.stability_summary_md_path,
    )


def _load_seed_row(artifacts: SeedRunArtifacts) -> dict[str, Any]:
    metrics = _read_required_json(artifacts.metrics_path)
    history = (
        _read_optional_json(artifacts.history_path)
        if artifacts.history_path is not None
        else {}
    )
    history_summary = _summarize_history(history)
    numeric_metrics = _extract_numeric_scalars(metrics)
    numeric_metrics.update(
        _extract_numeric_scalars(history_summary, prefix="history."),
    )

    artifact_paths = {
        "metrics_path": str(artifacts.metrics_path),
        "history_path": str(artifacts.history_path) if artifacts.history_path else None,
        "report_path": str(artifacts.report_path) if artifacts.report_path else None,
    }
    return {
        "seed": artifacts.seed,
        "output_dir": str(artifacts.output_dir),
        "metrics": metrics,
        "history_summary": history_summary,
        "artifact_paths": artifact_paths,
        "_numeric_metrics": numeric_metrics,
    }


def _build_summary_payload(
    *,
    study_dir: Path,
    per_seed_rows: Sequence[dict[str, Any]],
    study_name: str,
) -> dict[str, Any]:
    metrics_by_name: dict[str, list[tuple[int, float]]] = {}
    for row in per_seed_rows:
        seed = int(row["seed"])
        for metric_name, value in row["_numeric_metrics"].items():
            metrics_by_name.setdefault(metric_name, []).append((seed, float(value)))

    aggregates = {
        metric_name: _aggregate_metric(metric_name, seed_values)
        for metric_name, seed_values in sorted(metrics_by_name.items())
    }

    per_seed = [
        {
            "seed": row["seed"],
            "output_dir": row["output_dir"],
            "metrics": row["metrics"],
            "history_summary": row["history_summary"],
            "artifact_paths": row["artifact_paths"],
        }
        for row in per_seed_rows
    ]

    return {
        "study_name": study_name,
        "study_dir": str(study_dir),
        "seeds": [row["seed"] for row in per_seed],
        "per_seed": per_seed,
        "aggregates": aggregates,
    }


def _aggregate_metric(
    metric_name: str,
    seed_values: Sequence[tuple[int, float]],
) -> dict[str, Any]:
    values = [value for _, value in seed_values]
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    min_seed, min_value = min(seed_values, key=lambda item: item[1])
    max_seed, max_value = max(seed_values, key=lambda item: item[1])
    goal = _metric_goal(metric_name)

    payload = {
        "count": len(values),
        "mean": mean,
        "std": math.sqrt(variance),
        "min": min_value,
        "min_seed": min_seed,
        "max": max_value,
        "max_seed": max_seed,
        "goal": goal,
    }
    if goal == "maximize":
        payload["best_seed"] = max_seed
        payload["best_value"] = max_value
        payload["worst_seed"] = min_seed
        payload["worst_value"] = min_value
    elif goal == "minimize":
        payload["best_seed"] = min_seed
        payload["best_value"] = min_value
        payload["worst_seed"] = max_seed
        payload["worst_value"] = max_value

    return payload


def _metric_goal(metric_name: str) -> str | None:
    lowered = metric_name.lower()
    if any(hint in lowered for hint in _MINIMIZE_METRIC_HINTS):
        return "minimize"
    if any(hint in lowered for hint in _MAXIMIZE_METRIC_HINTS):
        return "maximize"
    return None


def _summarize_history(history: dict[str, Any]) -> dict[str, Any]:
    entries = history.get("entries")
    summary: dict[str, Any] = {}

    if isinstance(entries, list):
        summary["epochs_completed"] = len(entries)

    for key in ("best_val_acc", "best_epoch"):
        value = history.get(key)
        if _is_finite_number(value):
            summary[key] = value

    early_stopping = history.get("early_stopping")
    if isinstance(early_stopping, dict):
        for key in ("enabled", "stopped_early"):
            value = early_stopping.get(key)
            if isinstance(value, bool):
                summary[key] = value

    return summary


def _extract_numeric_scalars(
    payload: dict[str, Any],
    *,
    prefix: str = "",
) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for key, value in payload.items():
        if isinstance(value, bool):
            continue
        if _is_finite_number(value):
            metrics[f"{prefix}{key}"] = float(value)
    return metrics


def _is_finite_number(value: Any) -> bool:
    return isinstance(value, int | float) and math.isfinite(float(value))


def _read_required_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"required artifact not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _read_optional_json(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _write_summary_csv(
    path: Path,
    per_seed_rows: Sequence[dict[str, Any]],
    summary_payload: dict[str, Any],
) -> None:
    metric_names = list(summary_payload["aggregates"].keys())
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "row_type",
                "seed",
                "metric",
                "value",
                "mean",
                "std",
                "min",
                "min_seed",
                "max",
                "max_seed",
                "goal",
                "best_seed",
                "best_value",
                "worst_seed",
                "worst_value",
            ]
        )
        for row in per_seed_rows:
            seed = row["seed"]
            for metric_name in metric_names:
                value = row["_numeric_metrics"].get(metric_name)
                if value is None:
                    continue
                writer.writerow(
                    [
                        "per_seed",
                        seed,
                        metric_name,
                        value,
                        "",
                        "",
                        "",
                        "",
                        "",
                        "",
                        "",
                        "",
                        "",
                        "",
                        "",
                    ]
                )
        for metric_name, aggregate in summary_payload["aggregates"].items():
            writer.writerow(
                [
                    "aggregate",
                    "",
                    metric_name,
                    "",
                    aggregate["mean"],
                    aggregate["std"],
                    aggregate["min"],
                    aggregate["min_seed"],
                    aggregate["max"],
                    aggregate["max_seed"],
                    aggregate.get("goal"),
                    aggregate.get("best_seed"),
                    aggregate.get("best_value"),
                    aggregate.get("worst_seed"),
                    aggregate.get("worst_value"),
                ]
            )


def _write_summary_md(path: Path, summary_payload: dict[str, Any]) -> None:
    per_seed_rows = summary_payload["per_seed"]
    metric_names = list(summary_payload["aggregates"].keys())
    lines = [
        f"# {summary_payload['study_name']} Stability Summary",
        "",
        f"- Study dir: `{summary_payload['study_dir']}`",
        f"- Seeds: `{', '.join(str(seed) for seed in summary_payload['seeds'])}`",
        "",
        "## Per-seed Metrics",
        "",
    ]

    if per_seed_rows:
        header = ["Seed", "Output Dir", *metric_names]
        lines.append("| " + " | ".join(header) + " |")
        lines.append("| " + " | ".join(["---"] * len(header)) + " |")
        study_root = Path(summary_payload["study_dir"])
        for row in per_seed_rows:
            output_dir = Path(row["output_dir"])
            try:
                output_label = output_dir.relative_to(study_root).as_posix()
            except ValueError:
                output_label = str(output_dir)
            numeric_metrics = _extract_numeric_scalars(row["metrics"])
            numeric_metrics.update(
                _extract_numeric_scalars(row["history_summary"], prefix="history."),
            )
            rendered_values = [_format_metric_value(numeric_metrics.get(name)) for name in metric_names]
            lines.append(
                "| "
                + " | ".join([str(row["seed"]), output_label, *rendered_values])
                + " |"
            )
    else:
        lines.append("No completed seed runs were found.")

    lines.extend(["", "## Aggregate Metrics", ""])
    header = [
        "Metric",
        "Mean",
        "Std",
        "Min",
        "Min Seed",
        "Max",
        "Max Seed",
        "Goal",
        "Best Seed",
        "Worst Seed",
    ]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("| " + " | ".join(["---"] * len(header)) + " |")
    for metric_name, aggregate in summary_payload["aggregates"].items():
        lines.append(
            "| "
            + " | ".join(
                [
                    metric_name,
                    _format_metric_value(aggregate.get("mean")),
                    _format_metric_value(aggregate.get("std")),
                    _format_metric_value(aggregate.get("min")),
                    str(aggregate.get("min_seed", "")),
                    _format_metric_value(aggregate.get("max")),
                    str(aggregate.get("max_seed", "")),
                    str(aggregate.get("goal", "")),
                    str(aggregate.get("best_seed", "")),
                    str(aggregate.get("worst_seed", "")),
                ]
            )
            + " |"
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def _format_metric_value(value: Any) -> str:
    if value is None:
        return ""
    if _is_finite_number(value):
        return f"{float(value):.4f}"
    return str(value)
