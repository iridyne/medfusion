"""Helpers for registering real training runs in the Web model library."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from sqlalchemy.orm import Session

from med_core.configs import load_config
from med_core.postprocessing import build_results_artifacts

from .models import ModelInfo

_BACKBONE_PARAMETER_MAP: dict[str, int] = {
    "resnet18": 11_700_000,
    "resnet34": 21_800_000,
    "resnet50": 25_600_000,
    "resnet101": 44_500_000,
    "efficientnet_b0": 5_300_000,
    "vit_b16": 86_000_000,
    "swin_tiny": 28_300_000,
}


def _read_json(path: str | None) -> dict[str, Any]:
    if not path:
        return {}
    json_path = Path(path)
    if not json_path.exists():
        return {}
    try:
        return json.loads(json_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def _merge_tags(*groups: list[str] | None) -> list[str]:
    merged: list[str] = []
    for group in groups:
        if not group:
            continue
        for item in group:
            if item and item not in merged:
                merged.append(item)
    return merged


def _infer_accuracy(metrics: dict[str, Any], validation: dict[str, Any]) -> float | None:
    overview = validation.get("overview", {})
    for value in (
        metrics.get("best_accuracy"),
        overview.get("accuracy"),
        metrics.get("accuracy"),
        metrics.get("final_accuracy"),
    ):
        if value is not None:
            return float(value)
    return None


def _infer_loss(metrics: dict[str, Any]) -> float | None:
    for value in (
        metrics.get("best_loss"),
        metrics.get("final_loss"),
        metrics.get("loss"),
    ):
        if value is not None:
            return float(value)
    return None


def _infer_trained_epochs(
    explicit_epochs: int | None,
    artifact_paths: dict[str, str],
    summary_payload: dict[str, Any],
) -> int | None:
    if explicit_epochs is not None:
        return explicit_epochs
    total_epochs = summary_payload.get("total_epochs")
    if total_epochs is not None:
        return int(total_epochs)

    history_payload = _read_json(artifact_paths.get("history_path"))
    entries = history_payload.get("entries", [])
    if entries:
        return len(entries)
    return None


def _infer_training_time_seconds(
    explicit_seconds: float | None,
    summary_payload: dict[str, Any],
) -> float | None:
    if explicit_seconds is not None:
        return float(explicit_seconds)
    training_time_seconds = summary_payload.get("training_time_seconds")
    if training_time_seconds is not None:
        return float(training_time_seconds)
    return None


def _build_result_summary(
    *,
    experiment_name: str,
    dataset_name: str | None,
    backbone: str | None,
    metrics: dict[str, Any],
    validation: dict[str, Any],
    split: str | None,
) -> dict[str, Any]:
    overview = validation.get("overview", {})
    return {
        "experiment_name": experiment_name,
        "dataset_name": dataset_name,
        "backbone": backbone,
        "split": split or overview.get("split"),
        "best_accuracy": metrics.get("best_accuracy", overview.get("accuracy")),
        "best_loss": metrics.get("best_loss"),
        "balanced_accuracy": overview.get("balanced_accuracy", metrics.get("balanced_accuracy")),
        "macro_f1": overview.get("macro_f1", metrics.get("macro_f1")),
        "c_index": metrics.get("c_index"),
        "sample_count": overview.get("sample_count"),
    }


def register_model_artifacts(
    db: Session,
    *,
    checkpoint_path: str | Path,
    artifact_paths: dict[str, str],
    metrics: dict[str, Any],
    validation: dict[str, Any],
    name: str,
    description: str | None = None,
    architecture: str | None = None,
    config_path: str | Path | None = None,
    tags: list[str] | None = None,
    trained_epochs: int | None = None,
    training_time: float | None = None,
    num_parameters: int | None = None,
    model_type: str = "classification",
    extra_config: dict[str, Any] | None = None,
) -> ModelInfo:
    """Create or update a ModelInfo row from real result artifacts."""
    checkpoint = Path(checkpoint_path)
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    summary_payload = _read_json(artifact_paths.get("summary_path"))
    validation_dataset = validation.get("dataset", {})
    split = validation.get("overview", {}).get("split") or summary_payload.get("split")
    dataset_name = validation_dataset.get("name") or summary_payload.get("dataset_name")
    resolved_architecture = (
        architecture
        or summary_payload.get("backbone")
        or "unknown"
    )
    experiment_name = summary_payload.get("experiment_name") or checkpoint.stem

    model = (
        db.query(ModelInfo)
        .filter(ModelInfo.checkpoint_path == str(checkpoint))
        .first()
    )
    if model is None:
        model = ModelInfo(
            name=name,
            description=description,
            model_type=model_type,
            architecture=resolved_architecture,
            checkpoint_path=str(checkpoint),
        )
        db.add(model)

    stored_config = dict(extra_config or {})
    stored_config["artifact_paths"] = artifact_paths
    stored_config["result_summary"] = _build_result_summary(
        experiment_name=experiment_name,
        dataset_name=dataset_name,
        backbone=resolved_architecture,
        metrics=metrics,
        validation=validation,
        split=split,
    )

    model.name = name
    model.description = description
    model.model_type = model_type
    model.architecture = resolved_architecture
    model.checkpoint_path = str(checkpoint)
    model.config = stored_config
    model.config_path = str(config_path) if config_path else artifact_paths.get("config_path")
    model.metrics = metrics
    model.accuracy = _infer_accuracy(metrics, validation)
    model.loss = _infer_loss(metrics)
    model.num_parameters = (
        num_parameters
        if num_parameters is not None
        else _BACKBONE_PARAMETER_MAP.get(resolved_architecture)
    )
    model.model_size_mb = checkpoint.stat().st_size / (1024 * 1024)
    model.trained_epochs = _infer_trained_epochs(trained_epochs, artifact_paths, summary_payload)
    model.training_time = _infer_training_time_seconds(training_time, summary_payload)
    model.dataset_name = dataset_name
    model.num_classes = (
        validation_dataset.get("num_classes")
        or summary_payload.get("num_classes")
        or len(validation_dataset.get("labels", []))
        or None
    )
    model.tags = _merge_tags(model.tags, tags)

    db.commit()
    db.refresh(model)
    return model


def import_model_run(
    db: Session,
    *,
    config_path: str | Path,
    checkpoint_path: str | Path,
    output_dir: str | Path | None = None,
    split: str = "test",
    attention_samples: int = 4,
    enable_survival: bool = True,
    survival_time_column: str | None = None,
    survival_event_column: str | None = None,
    enable_importance: bool = True,
    importance_sample_limit: int = 128,
    name: str | None = None,
    description: str | None = None,
    tags: list[str] | None = None,
    import_source: str = "cli",
    source_context: dict[str, Any] | None = None,
) -> ModelInfo:
    """Build artifacts from a real run and register them in ModelInfo."""
    config_path = Path(config_path)
    checkpoint_path = Path(checkpoint_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    config = load_config(config_path)
    result = build_results_artifacts(
        config_path=config_path,
        checkpoint_path=checkpoint_path,
        output_dir=output_dir,
        split=split,
        attention_samples=max(attention_samples, 0),
        enable_survival=enable_survival,
        survival_time_column=survival_time_column,
        survival_event_column=survival_event_column,
        enable_importance=enable_importance,
        importance_sample_limit=max(importance_sample_limit, 0),
    )
    summary_payload = _read_json(result.artifact_paths.get("summary_path"))
    config_snapshot = _read_json(result.artifact_paths.get("config_path"))
    experiment_name = summary_payload.get("experiment_name") or config.experiment_name or checkpoint_path.stem
    model_name = name or (
        experiment_name if experiment_name.endswith("-model") else f"{experiment_name}-model"
    )

    return register_model_artifacts(
        db=db,
        checkpoint_path=checkpoint_path,
        artifact_paths=result.artifact_paths,
        metrics=result.metrics,
        validation=result.validation,
        name=model_name,
        description=description or "由真实 CLI 训练结果导入的模型产物",
        architecture=config.model.vision.backbone,
        config_path=result.artifact_paths.get("config_path"),
        tags=_merge_tags(["imported", f"split:{split}"], tags),
        extra_config={
            "import_source": import_source,
            "source_config_path": str(config_path),
            "config_snapshot": config_snapshot,
            "source_context": source_context or {},
        },
    )
