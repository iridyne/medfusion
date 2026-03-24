"""Compatibility layer for advanced build-results analysis.

This module exposes a stable interface consumed by results.py while using the
lighter-weight advanced analysis helpers internally.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from torch.utils.data import DataLoader

from med_core.datasets import MedicalMultimodalDataset
from med_core.postprocessing.advanced_analysis import (
    build_shap_artifacts,
    build_survival_artifacts as _build_survival_artifacts,
)


def resolve_survival_columns(
    config: Any,
    *,
    override_time_column: str | None = None,
    override_event_column: str | None = None,
) -> tuple[str | None, str | None]:
    time_column = override_time_column or getattr(config.data, "survival_time_column", None)
    event_column = override_event_column or getattr(config.data, "survival_event_column", None)
    return time_column, event_column


def _copy_if_exists(source_path: str | None, target_path: Path) -> str | None:
    if not source_path:
        return None
    source = Path(source_path)
    if not source.exists():
        return None
    target_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target_path)
    return str(target_path)


def build_survival_artifacts(
    dataset: MedicalMultimodalDataset,
    output_dir: Path,
    risk_scores: np.ndarray,
    risk_score_source: str,
    time_column: str | None,
    event_column: str | None,
    *,
    cutoff: float | None = None,
    cutoff_source: str = "current_split_median",
    split: str = "test",
) -> tuple[dict[str, Any] | None, dict[str, str]]:
    metadata_frame = getattr(dataset, "metadata_frame", None)
    payload, artifact_paths, _metric_updates = _build_survival_artifacts(
        output_dir=output_dir,
        metadata_frame=metadata_frame,
        risk_scores=risk_scores,
        time_column=time_column,
        event_column=event_column,
        split=split,
        cutoff=cutoff,
        cutoff_source=cutoff_source,
    )
    if not payload.get("available"):
        return None, {}

    visualization_dir = output_dir / "visualizations"
    km_target = visualization_dir / "kaplan_meier_curve.png"
    risk_target = visualization_dir / "risk_score_distribution.png"
    copied_km = _copy_if_exists(artifact_paths.get("kaplan_meier_plot_path"), km_target)
    copied_risk = _copy_if_exists(artifact_paths.get("risk_score_distribution_plot_path"), risk_target)

    payload["risk_score_source"] = risk_score_source
    payload.setdefault("artifacts", {})
    if copied_km:
        payload["artifacts"]["kaplan_meier_plot_path"] = copied_km
    if copied_risk:
        payload["artifacts"]["risk_score_distribution_plot_path"] = copied_risk

    normalized_paths = {
        "survival_json_path": artifact_paths.get("survival_json_path"),
        "survival_path": artifact_paths.get("survival_json_path"),
        "kaplan_meier_plot_path": copied_km,
        "risk_score_distribution_plot_path": copied_risk,
    }
    normalized_paths = {key: value for key, value in normalized_paths.items() if value}
    return payload, normalized_paths


def build_global_feature_importance_artifacts(
    dataset: MedicalMultimodalDataset,
    device: torch.device,
    output_dir: Path,
    batch_size: int,
    sample_limit: int,
    score_name: str,
    score_fn: Callable[[torch.Tensor, torch.Tensor], np.ndarray],
) -> tuple[dict[str, Any] | None, dict[str, str]]:
    if sample_limit <= 0:
        return {"enabled": True, "available": False, "reason": "disabled"}, {}

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    scores: list[np.ndarray] = []
    for images, tabular, _labels in loader:
        images = images.to(device)
        tabular = tabular.to(device)
        scores.append(np.asarray(score_fn(images, tabular), dtype=float).reshape(-1))

    if not scores:
        return None, {}

    model_scores = np.concatenate(scores, axis=0)
    feature_names = dataset.get_feature_names() if hasattr(dataset, "get_feature_names") else []
    payload, artifact_paths, _metric_updates = build_shap_artifacts(
        output_dir=output_dir,
        feature_names=feature_names,
        tabular_data=dataset.tabular_data.cpu().numpy(),
        model_scores=model_scores,
        y_true=dataset.labels.cpu().numpy(),
        positive_class_label="positive_class",
        max_display=min(10, 1 + len(feature_names)),
        max_samples=max(sample_limit, 1),
    )
    if not payload.get("available"):
        return None, {}

    visualization_dir = output_dir / "visualizations"
    bar_target = visualization_dir / "feature_importance_bar.png"
    beeswarm_target = visualization_dir / "feature_importance_beeswarm.png"
    copied_bar = _copy_if_exists(artifact_paths.get("shap_bar_plot_path"), bar_target)
    copied_beeswarm = _copy_if_exists(artifact_paths.get("shap_beeswarm_plot_path"), beeswarm_target)

    payload = {
        "available": True,
        "method": payload.get("method"),
        "score_name": score_name,
        "top_features": payload.get("features", []),
    }
    feature_importance_json_path = output_dir / "feature_importance.json"
    copied_importance_json = _copy_if_exists(
        artifact_paths.get("shap_summary_json_path"),
        feature_importance_json_path,
    )
    normalized_paths = {
        "global_feature_importance_json_path": copied_importance_json,
        "feature_importance_path": copied_importance_json,
        "feature_importance_bar_plot_path": copied_bar,
        "feature_importance_beeswarm_plot_path": copied_beeswarm,
    }
    normalized_paths = {key: value for key, value in normalized_paths.items() if value}
    return payload, normalized_paths

