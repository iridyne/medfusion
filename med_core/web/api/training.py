"""训练任务 API"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import APIRouter, Depends, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_curve,
)
from sqlalchemy.orm import Session

from med_core.shared.visualization import (
    plot_calibration_curve,
    plot_confusion_matrix,
    plot_probability_distribution,
    plot_roc_curve,
    plot_training_curves,
)
from med_core.visualization.attention_viz import (
    plot_attention_statistics,
    visualize_attention_overlay,
)

from ..config import settings
from ..database import SessionLocal, get_db_session
from ..models import ModelInfo, TrainingJob

router = APIRouter()
logger = logging.getLogger(__name__)

_training_tasks: dict[str, asyncio.Task[None]] = {}
_pause_flags: dict[str, bool] = {}
_stop_flags: dict[str, bool] = {}

_BACKBONE_PARAMETER_MAP: dict[str, int] = {
    "resnet18": 11_700_000,
    "resnet34": 21_800_000,
    "resnet50": 25_600_000,
    "resnet101": 44_500_000,
    "efficientnet_b0": 5_300_000,
    "vit_b16": 86_000_000,
    "swin_tiny": 28_300_000,
}


class TrainingConfig(BaseModel):
    """训练配置"""

    experiment_name: str
    training_model_config: dict[str, Any]
    dataset_config: dict[str, Any]
    training_config: dict[str, Any]


class TrainingJobResponse(BaseModel):
    """训练任务响应"""

    id: int
    job_id: str
    experiment_name: str
    dataset_name: str | None
    backbone: str | None
    status: str
    progress: float
    current_epoch: int
    total_epochs: int
    current_loss: float | None
    current_accuracy: float | None
    created_at: str

    class Config:
        from_attributes = True


def _get_job_or_404(db: Session, job_id: str) -> TrainingJob:
    job = db.query(TrainingJob).filter(TrainingJob.job_id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="训练任务不存在")
    return job


def _extract_job_metadata(job: TrainingJob) -> dict[str, Any]:
    config = job.config or {}
    training_model_config = config.get("training_model_config", {})
    dataset_config = config.get("dataset_config", {})
    return {
        "experiment_name": config.get("experiment_name") or f"training-{job.job_id[:8]}",
        "dataset_name": dataset_config.get("dataset")
        or dataset_config.get("dataset_name")
        or dataset_config.get("name"),
        "backbone": training_model_config.get("backbone"),
        "num_classes": training_model_config.get("num_classes")
        or dataset_config.get("num_classes"),
    }


def _estimate_num_parameters(backbone: str | None) -> int | None:
    if not backbone:
        return None
    return _BACKBONE_PARAMETER_MAP.get(backbone)


def _prepare_job_output(job_id: str) -> tuple[str, str]:
    output_dir = settings.data_dir / "experiments" / job_id
    output_dir.mkdir(parents=True, exist_ok=True)
    log_file = output_dir / "training.log"
    if not log_file.exists():
        log_file.write_text("training started\n", encoding="utf-8")
    return str(output_dir), str(log_file)


def _seed_from_text(text: str) -> int:
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return int(digest[:16], 16) % (2**32)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        logger.warning("Failed to parse JSON artifact: %s", path)
        return {}


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _history_path_for_job(job: TrainingJob) -> Path:
    output_dir = Path(job.output_dir or settings.data_dir / "experiments" / job.job_id)
    return output_dir / "history.json"


def _append_history_entry(job: TrainingJob, entry: dict[str, Any]) -> None:
    history_path = _history_path_for_job(job)
    payload = _read_json(history_path)
    entries = payload.get("entries", [])
    entries.append(entry)
    payload["entries"] = entries
    payload["job_id"] = job.job_id
    _write_json(history_path, payload)


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def _generate_prediction_payload(
    job: TrainingJob,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    num_classes = max(int(metadata.get("num_classes") or 2), 2)
    sample_count = max(160, num_classes * 80)
    labels = [f"Class {index}" for index in range(num_classes)]
    rng = np.random.default_rng(_seed_from_text(f"predictions:{job.job_id}"))

    y_true = np.arange(sample_count, dtype=int) % num_classes
    rng.shuffle(y_true)

    target_accuracy = _clamp(
        float(job.best_accuracy or job.current_accuracy or 0.85),
        0.55,
        0.98,
    )
    probabilities = np.zeros((sample_count, num_classes), dtype=float)
    y_pred = np.zeros(sample_count, dtype=int)

    for index, true_label in enumerate(y_true):
        is_correct = bool(rng.random() < target_accuracy)
        if is_correct:
            predicted_label = int(true_label)
        else:
            predicted_label = int(rng.integers(0, num_classes - 1))
            if predicted_label >= true_label:
                predicted_label += 1

        raw_scores = rng.uniform(0.01, 0.18, size=num_classes)
        raw_scores[predicted_label] += rng.uniform(0.55, 0.95)
        if predicted_label != true_label:
            raw_scores[true_label] += rng.uniform(0.08, 0.28)
        else:
            raw_scores[true_label] += rng.uniform(0.15, 0.32)

        normalized = raw_scores / raw_scores.sum()
        probabilities[index] = normalized
        y_pred[index] = int(np.argmax(normalized))

    positive_class_index = 1 if num_classes > 1 else 0
    return {
        "labels": labels,
        "num_classes": num_classes,
        "positive_class_index": positive_class_index,
        "y_true": y_true.tolist(),
        "y_pred": y_pred.tolist(),
        "probabilities": probabilities.round(6).tolist(),
    }


def _downsample_attention(attention: np.ndarray, size: int = 8) -> list[list[float]]:
    if attention.ndim != 2:
        raise ValueError("Attention map must be 2D")
    height, width = attention.shape
    if height % size == 0 and width % size == 0:
        block_h = height // size
        block_w = width // size
        reduced = attention.reshape(size, block_h, size, block_w).mean(axis=(1, 3))
    else:
        row_indices = np.linspace(0, height - 1, size).astype(int)
        col_indices = np.linspace(0, width - 1, size).astype(int)
        reduced = attention[np.ix_(row_indices, col_indices)]
    return np.round(reduced, 4).tolist()


def _generate_base_image(seed: str, size: int = 64) -> np.ndarray:
    rng = np.random.default_rng(_seed_from_text(seed))
    y_axis, x_axis = np.mgrid[0:size, 0:size]
    center_x = rng.uniform(size * 0.25, size * 0.75)
    center_y = rng.uniform(size * 0.25, size * 0.75)
    radius = rng.uniform(size * 0.12, size * 0.26)

    gaussian = np.exp(
        -(((x_axis - center_x) ** 2 + (y_axis - center_y) ** 2) / (2 * radius**2))
    )
    diagonal = (x_axis + y_axis) / (2 * size)
    noise = rng.uniform(0.0, 0.12, size=(size, size))
    image = gaussian * 0.7 + diagonal * 0.2 + noise
    image = (image - image.min()) / (image.max() - image.min() + 1e-8)
    return image


def _generate_attention_map(seed: str, size: int = 64) -> np.ndarray:
    rng = np.random.default_rng(_seed_from_text(seed))
    y_axis, x_axis = np.mgrid[0:size, 0:size]
    centers = [
        (
            rng.uniform(size * 0.2, size * 0.8),
            rng.uniform(size * 0.2, size * 0.8),
            rng.uniform(size * 0.08, size * 0.18),
            rng.uniform(0.4, 0.8),
        )
        for _ in range(2)
    ]
    attention = np.zeros((size, size), dtype=float)
    for center_x, center_y, spread, weight in centers:
        distance = (x_axis - center_x) ** 2 + (y_axis - center_y) ** 2
        attention += weight * np.exp(-distance / (2 * (size * spread) ** 2))

    attention += rng.uniform(0.0, 0.05, size=(size, size))
    attention = (attention - attention.min()) / (attention.max() - attention.min() + 1e-8)
    return attention


def _generate_attention_artifacts(output_dir: Path, job_id: str) -> dict[str, Any]:
    attention_dir = output_dir / "visualizations" / "attention"
    attention_dir.mkdir(parents=True, exist_ok=True)

    modalities = [
        ("image", "影像模态注意力"),
        ("tabular", "临床表格注意力"),
        ("text", "病历文本注意力"),
        ("fusion", "融合层注意力"),
    ]
    manifest_items: list[dict[str, Any]] = []
    attention_history: list[np.ndarray] = []

    for modality, title in modalities:
        base_image = _generate_base_image(f"{job_id}:{modality}:image")
        attention_map = _generate_attention_map(f"{job_id}:{modality}:attention")
        artifact_key = f"attention_{modality}_overlay"
        image_path = attention_dir / f"{artifact_key}.png"

        figure = visualize_attention_overlay(
            image=base_image,
            attention=attention_map,
            alpha=0.55,
            title=title,
            save_path=image_path,
        )
        figure.clf()

        attention_history.append(attention_map)
        manifest_items.append(
            {
                "title": title,
                "modality": modality,
                "artifact_key": artifact_key,
                "image_path": str(image_path),
                "grid": _downsample_attention(attention_map),
                "mean_attention": round(float(attention_map.mean()), 4),
                "peak_attention": round(float(attention_map.max()), 4),
            }
        )

    statistics_path = attention_dir / "attention_statistics.png"
    statistics_figure = plot_attention_statistics(
        attention_history,
        labels=[item["modality"] for item in manifest_items],
        save_path=statistics_path,
    )
    statistics_figure.clf()

    manifest = {
        "items": manifest_items,
        "statistics_artifact_key": "attention_statistics_plot",
        "statistics_plot_path": str(statistics_path),
    }
    manifest_path = attention_dir / "attention_maps.json"
    _write_json(manifest_path, manifest)

    return {
        "attention_manifest_path": str(manifest_path),
        "attention_statistics_plot_path": str(statistics_path),
    }


def _generate_visualization_artifacts(
    output_dir: Path,
    job: TrainingJob,
    metadata: dict[str, Any],
    history_payload: dict[str, Any],
) -> dict[str, Any]:
    visualization_dir = output_dir / "visualizations"
    visualization_dir.mkdir(parents=True, exist_ok=True)

    predictions_payload = _generate_prediction_payload(job, metadata)
    predictions_path = output_dir / "predictions.json"
    _write_json(predictions_path, predictions_payload)

    labels = predictions_payload["labels"]
    num_classes = predictions_payload["num_classes"]
    positive_class_index = predictions_payload["positive_class_index"]
    y_true = np.asarray(predictions_payload["y_true"], dtype=int)
    y_pred = np.asarray(predictions_payload["y_pred"], dtype=int)
    probabilities = np.asarray(predictions_payload["probabilities"], dtype=float)

    average = "binary" if num_classes == 2 else "macro"
    positive_class_label = labels[positive_class_index]
    y_true_binary = (y_true == positive_class_index).astype(int)
    positive_scores = probabilities[:, positive_class_index]

    precision, recall, f1_score, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average=average,
        zero_division=0,
    )
    accuracy = accuracy_score(y_true, y_pred)
    matrix = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))

    roc_points: list[dict[str, float]] = []
    roc_auc: float | None = None
    roc_curve_path = visualization_dir / "roc_curve.png"
    if np.unique(y_true_binary).size > 1:
        fpr, tpr, thresholds = roc_curve(y_true_binary, positive_scores)
        roc_auc = float(auc(fpr, tpr))
        roc_points = [
            {
                "fpr": round(float(fpr_value), 4),
                "tpr": round(float(tpr_value), 4),
                "threshold": round(float(threshold_value), 4),
            }
            for fpr_value, tpr_value, threshold_value in zip(fpr, tpr, thresholds, strict=False)
        ]
        roc_figure, _ = plot_roc_curve(
            y_true_binary,
            positive_scores,
            title=f"ROC Curve ({positive_class_label} vs Rest)",
            save_path=roc_curve_path,
        )
        roc_figure.clf()
    else:
        roc_curve_path = None

    confusion_matrix_path = visualization_dir / "confusion_matrix.png"
    confusion_figure, _ = plot_confusion_matrix(
        y_true,
        y_pred,
        class_names=labels,
        title="Confusion Matrix",
        save_path=confusion_matrix_path,
    )
    confusion_figure.clf()

    normalized_confusion_matrix_path = visualization_dir / "confusion_matrix_normalized.png"
    normalized_confusion_figure, _ = plot_confusion_matrix(
        y_true,
        y_pred,
        class_names=labels,
        title="Normalized Confusion Matrix",
        normalize="true",
        save_path=normalized_confusion_matrix_path,
    )
    normalized_confusion_figure.clf()

    history_entries = history_payload.get("entries", [])
    training_curves_path = visualization_dir / "training_curves.png"
    training_curve_figure, _ = plot_training_curves(
        [entry["train_loss"] for entry in history_entries],
        val_losses=[entry["val_loss"] for entry in history_entries],
        train_metrics={"Accuracy": [entry["train_accuracy"] for entry in history_entries]},
        val_metrics={"Accuracy": [entry["val_accuracy"] for entry in history_entries]},
        title="Training Curves",
        save_path=training_curves_path,
    )
    training_curve_figure.clf()

    calibration_curve_path = None
    probability_distribution_path = None
    if np.unique(y_true_binary).size > 1:
        calibration_curve_path = visualization_dir / "calibration_curve.png"
        calibration_figure, _ = plot_calibration_curve(
            y_true_binary,
            positive_scores,
            title=f"Calibration Curve ({positive_class_label})",
            save_path=calibration_curve_path,
        )
        calibration_figure.clf()

        probability_distribution_path = visualization_dir / "probability_distribution.png"
        probability_figure, _ = plot_probability_distribution(
            y_true_binary,
            positive_scores,
            title=f"Prediction Probability Distribution ({positive_class_label})",
            save_path=probability_distribution_path,
        )
        probability_figure.clf()

    roc_payload = {
        "artifact_key": "roc_curve_plot",
        "positive_class_label": positive_class_label,
        "auc": round(float(roc_auc), 4) if roc_auc is not None else None,
        "points": roc_points,
        "plot_path": str(roc_curve_path) if roc_curve_path else None,
    }
    confusion_payload = {
        "artifact_key": "confusion_matrix_plot",
        "labels": labels,
        "matrix": matrix.tolist(),
        "plot_path": str(confusion_matrix_path),
        "normalized_plot_path": str(normalized_confusion_matrix_path),
    }
    metrics_payload = {
        "accuracy": round(float(accuracy), 4),
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "f1_score": round(float(f1_score), 4),
        "auc": round(float(roc_auc), 4) if roc_auc is not None else None,
        "best_accuracy": round(float(job.best_accuracy or accuracy), 4),
        "best_loss": round(float(job.best_loss or job.current_loss or 0.0), 4),
        "best_epoch": job.best_epoch,
        "final_accuracy": round(float(job.current_accuracy or accuracy), 4),
        "final_loss": round(float(job.current_loss or 0.0), 4),
        "progress": round(float(job.progress), 2),
    }

    roc_curve_json_path = output_dir / "roc_curve.json"
    confusion_matrix_json_path = output_dir / "confusion_matrix.json"
    metrics_path = output_dir / "metrics.json"

    _write_json(roc_curve_json_path, roc_payload)
    _write_json(confusion_matrix_json_path, confusion_payload)
    _write_json(metrics_path, metrics_payload)

    attention_artifacts = _generate_attention_artifacts(output_dir, job.job_id)

    return {
        "metrics": metrics_payload,
        "prediction_path": str(predictions_path),
        "roc_curve_json_path": str(roc_curve_json_path),
        "roc_curve_plot_path": str(roc_curve_path) if roc_curve_path else None,
        "confusion_matrix_json_path": str(confusion_matrix_json_path),
        "confusion_matrix_plot_path": str(confusion_matrix_path),
        "confusion_matrix_normalized_plot_path": str(normalized_confusion_matrix_path),
        "training_curves_plot_path": str(training_curves_path),
        "calibration_curve_plot_path": str(calibration_curve_path) if calibration_curve_path else None,
        "probability_distribution_plot_path": str(probability_distribution_path)
        if probability_distribution_path
        else None,
        **attention_artifacts,
    }


def _write_demo_checkpoint(job: TrainingJob, metadata: dict[str, Any]) -> Path:
    target_dir = settings.data_dir / "models"
    target_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = target_dir / f"{job.job_id}.pth"
    checkpoint_payload = {
        "job_id": job.job_id,
        "status": job.status,
        "experiment_name": metadata["experiment_name"],
        "dataset_name": metadata["dataset_name"],
        "backbone": metadata["backbone"],
        "best_accuracy": job.best_accuracy,
        "best_loss": job.best_loss,
        "completed_at": job.completed_at.isoformat() if job.completed_at else None,
    }
    checkpoint_path.write_text(
        json.dumps(checkpoint_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return checkpoint_path


def _write_result_artifacts(
    job: TrainingJob,
    metadata: dict[str, Any],
    checkpoint_path: Path,
) -> dict[str, str]:
    output_dir = Path(job.output_dir or settings.data_dir / "experiments" / job.job_id)
    output_dir.mkdir(parents=True, exist_ok=True)

    config_path = output_dir / "training-config.json"
    summary_path = output_dir / "summary.json"
    report_path = output_dir / "report.md"
    log_path = Path(job.log_file) if job.log_file else output_dir / "training.log"
    history_path = _history_path_for_job(job)
    history_payload = _read_json(history_path)
    history_entries = history_payload.get("entries", [])

    visualization_artifacts = _generate_visualization_artifacts(
        output_dir=output_dir,
        job=job,
        metadata=metadata,
        history_payload=history_payload,
    )
    metrics_payload = visualization_artifacts["metrics"]

    summary_payload = {
        "job_id": job.job_id,
        "experiment_name": metadata["experiment_name"],
        "dataset_name": metadata["dataset_name"],
        "backbone": metadata["backbone"],
        "num_classes": metadata["num_classes"],
        "total_epochs": job.total_epochs,
        "training_time_seconds": (
            (job.completed_at - job.started_at).total_seconds()
            if job.completed_at and job.started_at
            else None
        ),
        "artifacts": {
            "checkpoint_path": str(checkpoint_path),
            "config_path": str(config_path),
            "summary_path": str(summary_path),
            "report_path": str(report_path),
            "log_path": str(log_path),
            "history_path": str(history_path),
            **{
                key: value
                for key, value in visualization_artifacts.items()
                if key != "metrics" and value
            },
        },
        "metrics": metrics_payload,
    }

    config_path.write_text(
        json.dumps(job.config or {}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    summary_path.write_text(
        json.dumps(summary_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    report_path.write_text(
        "\n".join(
            [
                f"# {metadata['experiment_name']} Result",
                "",
                f"- dataset: {metadata['dataset_name'] or 'unknown'}",
                f"- backbone: {metadata['backbone'] or 'unknown'}",
                f"- accuracy: {metrics_payload['accuracy']}",
                f"- precision: {metrics_payload['precision']}",
                f"- recall: {metrics_payload['recall']}",
                f"- f1_score: {metrics_payload['f1_score']}",
                f"- auc: {metrics_payload['auc']}",
                f"- best_accuracy: {metrics_payload['best_accuracy']}",
                f"- best_loss: {metrics_payload['best_loss']}",
                f"- checkpoint: {checkpoint_path}",
                "",
                "## Visualizations",
                "",
                "![Training Curves](visualizations/training_curves.png)",
                "",
                "![ROC Curve](visualizations/roc_curve.png)"
                if visualization_artifacts.get("roc_curve_plot_path")
                else "",
                "",
                "![Confusion Matrix](visualizations/confusion_matrix.png)",
                "",
                "![Attention Statistics](visualizations/attention/attention_statistics.png)",
                "",
                "## History",
                "",
                f"- total_entries: {len(history_entries)}",
            ],
        ),
        encoding="utf-8",
    )
    log_path.write_text(
        log_path.read_text(encoding="utf-8") + "training completed\n",
        encoding="utf-8",
    )

    return {
        "output_dir": str(output_dir),
        "config_path": str(config_path),
        "metrics_path": str(output_dir / "metrics.json"),
        "summary_path": str(summary_path),
        "report_path": str(report_path),
        "log_path": str(log_path),
        "history_path": str(history_path),
        **{
            key: value
            for key, value in visualization_artifacts.items()
            if key != "metrics" and value
        },
    }


def _sync_completed_model(db: Session, job: TrainingJob) -> None:
    metadata = _extract_job_metadata(job)
    checkpoint_path = _write_demo_checkpoint(job, metadata)
    artifact_paths = _write_result_artifacts(job, metadata, checkpoint_path)
    file_size = checkpoint_path.stat().st_size
    training_time = (
        (job.completed_at - job.started_at).total_seconds()
        if job.completed_at and job.started_at
        else None
    )

    model = (
        db.query(ModelInfo)
        .filter(ModelInfo.checkpoint_path == str(checkpoint_path))
        .first()
    )
    if model is None:
        model = ModelInfo(
            name=f"{metadata['experiment_name']}-model",
            description="演示型 MVP 自动生成的训练产物",
            model_type="classification",
            architecture=metadata["backbone"] or "resnet18",
            checkpoint_path=str(checkpoint_path),
        )
        db.add(model)

    model.config = {
        "job_id": job.job_id,
        "training_config": (job.config or {}).get("training_config", {}),
        "dataset_config": (job.config or {}).get("dataset_config", {}),
        "artifact_paths": artifact_paths,
        "result_summary": {
            "experiment_name": metadata["experiment_name"],
            "dataset_name": metadata["dataset_name"],
            "backbone": metadata["backbone"],
            "best_accuracy": job.best_accuracy,
            "best_loss": job.best_loss,
        },
    }
    model.config_path = artifact_paths["config_path"]
    model.metrics = {
        **_read_json(Path(artifact_paths["metrics_path"])),
    }
    model.accuracy = job.best_accuracy or job.current_accuracy
    model.loss = job.best_loss or job.current_loss
    model.num_parameters = _estimate_num_parameters(metadata["backbone"])
    model.model_size_mb = file_size / (1024 * 1024)
    model.trained_epochs = job.total_epochs
    model.training_time = training_time
    model.dataset_name = metadata["dataset_name"]
    model.num_classes = metadata["num_classes"]
    model.tags = [
        "demo-mvp",
        f"job:{job.job_id}",
        *(["auto-generated"] if True else []),
    ]


async def _simulate_training(job_id: str, total_epochs: int) -> None:
    """轻量训练模拟器：用于 Web UI 最小可用闭环。"""
    db = SessionLocal()
    try:
        for epoch in range(1, total_epochs + 1):
            while _pause_flags.get(job_id, False):
                await asyncio.sleep(0.5)

            if _stop_flags.get(job_id, False):
                job = _get_job_or_404(db, job_id)
                job.status = "stopped"
                job.completed_at = datetime.utcnow()
                if job.log_file:
                    Path(job.log_file).write_text(
                        Path(job.log_file).read_text(encoding="utf-8") + "training stopped\n",
                        encoding="utf-8",
                    )
                db.commit()
                return

            await asyncio.sleep(1.0)
            job = _get_job_or_404(db, job_id)

            progress = round(epoch / total_epochs * 100, 2)
            current_loss = max(0.01, 1.0 - (epoch / total_epochs) * 0.9)
            current_accuracy = min(0.99, 0.5 + (epoch / total_epochs) * 0.45)
            learning_rate = max(
                float((job.config or {}).get("training_config", {}).get("learningRate", 0.001))
                * (0.94 ** max(epoch - 1, 0)),
                0.00001,
            )
            val_loss = min(1.2, current_loss * 1.06)
            val_accuracy = max(0.0, current_accuracy - 0.03)

            job.status = "running"
            job.current_epoch = epoch
            job.progress = progress
            job.current_loss = current_loss
            job.current_accuracy = current_accuracy
            job.current_lr = learning_rate
            job.best_loss = (
                current_loss
                if job.best_loss is None
                else min(job.best_loss, current_loss)
            )
            job.best_accuracy = (
                current_accuracy
                if job.best_accuracy is None
                else max(job.best_accuracy, current_accuracy)
            )
            if job.best_accuracy == current_accuracy:
                job.best_epoch = epoch
            if job.log_file:
                Path(job.log_file).write_text(
                    Path(job.log_file).read_text(encoding="utf-8")
                    + f"epoch {epoch}: loss={current_loss:.4f}, accuracy={current_accuracy:.4f}, lr={learning_rate:.6f}\n",
                    encoding="utf-8",
                )
            _append_history_entry(
                job,
                {
                    "epoch": epoch,
                    "train_loss": round(float(current_loss), 4),
                    "val_loss": round(float(val_loss), 4),
                    "train_accuracy": round(float(min(current_accuracy + 0.02, 0.999)), 4),
                    "val_accuracy": round(float(val_accuracy), 4),
                    "learning_rate": round(float(learning_rate), 6),
                    "best_so_far": job.best_epoch == epoch,
                },
            )

            if epoch >= total_epochs:
                job.status = "completed"
                job.completed_at = datetime.utcnow()
                job.progress = 100.0
                _sync_completed_model(db, job)

            db.commit()
    except Exception as e:
        logger.error(f"训练任务模拟失败: {job_id}, 错误: {e}")
        job = db.query(TrainingJob).filter(TrainingJob.job_id == job_id).first()
        if job:
            job.status = "failed"
            job.error_message = str(e)
            job.completed_at = datetime.utcnow()
            db.commit()
    finally:
        db.close()
        _training_tasks.pop(job_id, None)
        _pause_flags.pop(job_id, None)
        _stop_flags.pop(job_id, None)


@router.post("/start")
async def start_training(
    config: TrainingConfig,
    db: Session = Depends(get_db_session),
) -> dict[str, Any]:
    """开始训练任务"""
    try:
        job_id = str(uuid.uuid4())
        total_epochs = int(config.training_config.get("epochs", 20))
        output_dir, log_file = _prepare_job_output(job_id)

        job = TrainingJob(
            job_id=job_id,
            config=config.model_dump(),
            total_epochs=total_epochs,
            status="running",
            progress=0.0,
            current_epoch=0,
            created_at=datetime.utcnow(),
            started_at=datetime.utcnow(),
            current_loss=1.0,
            current_accuracy=0.5,
            output_dir=output_dir,
            log_file=log_file,
        )
        db.add(job)
        db.commit()
        db.refresh(job)

        _pause_flags[job_id] = False
        _stop_flags[job_id] = False
        _training_tasks[job_id] = asyncio.create_task(_simulate_training(job_id, total_epochs))

        logger.info(f"训练任务已创建: {job_id}")
        return {"job_id": job_id, "status": "running", "message": "训练任务已启动"}
    except Exception as e:
        logger.error(f"创建训练任务失败: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/jobs")
async def list_training_jobs(
    skip: int = 0,
    limit: int = 20,
    db: Session = Depends(get_db_session),
) -> list[TrainingJobResponse]:
    """获取训练任务列表"""
    jobs = (
        db.query(TrainingJob)
        .order_by(TrainingJob.created_at.desc())
        .offset(skip)
        .limit(limit)
        .all()
    )

    return [
        TrainingJobResponse(
            id=job.id,
            job_id=job.job_id,
            experiment_name=_extract_job_metadata(job)["experiment_name"],
            dataset_name=_extract_job_metadata(job)["dataset_name"],
            backbone=_extract_job_metadata(job)["backbone"],
            status=job.status,
            progress=job.progress,
            current_epoch=job.current_epoch,
            total_epochs=job.total_epochs,
            current_loss=job.current_loss,
            current_accuracy=job.current_accuracy,
            created_at=job.created_at.isoformat(),
        )
        for job in jobs
    ]


@router.get("/{job_id}/status")
async def get_training_status(
    job_id: str,
    db: Session = Depends(get_db_session),
) -> dict[str, Any]:
    """获取训练任务状态"""
    job = _get_job_or_404(db, job_id)
    metadata = _extract_job_metadata(job)
    return {
        "job_id": job.job_id,
        "experiment_name": metadata["experiment_name"],
        "dataset_name": metadata["dataset_name"],
        "backbone": metadata["backbone"],
        "status": job.status,
        "progress": job.progress,
        "current_epoch": job.current_epoch,
        "total_epochs": job.total_epochs,
        "current_loss": job.current_loss,
        "current_accuracy": job.current_accuracy,
        "best_loss": job.best_loss,
        "best_accuracy": job.best_accuracy,
        "gpu_usage": job.gpu_usage,
        "gpu_memory": job.gpu_memory,
        "error_message": job.error_message,
    }


@router.post("/{job_id}/pause")
async def pause_training(
    job_id: str,
    db: Session = Depends(get_db_session),
) -> dict[str, str]:
    """暂停训练任务"""
    job = _get_job_or_404(db, job_id)
    if job.status != "running":
        raise HTTPException(status_code=400, detail="只能暂停正在运行的任务")

    _pause_flags[job_id] = True
    job.status = "paused"
    db.commit()
    return {"message": "训练任务已暂停"}


@router.post("/{job_id}/resume")
async def resume_training(
    job_id: str,
    db: Session = Depends(get_db_session),
) -> dict[str, str]:
    """恢复训练任务"""
    job = _get_job_or_404(db, job_id)
    if job.status != "paused":
        raise HTTPException(status_code=400, detail="只能恢复已暂停的任务")

    _pause_flags[job_id] = False
    job.status = "running"
    db.commit()
    return {"message": "训练任务已恢复"}


@router.post("/{job_id}/stop")
async def stop_training(
    job_id: str,
    db: Session = Depends(get_db_session),
) -> dict[str, str]:
    """停止训练任务"""
    job = _get_job_or_404(db, job_id)
    if job.status not in {"running", "paused", "queued"}:
        raise HTTPException(status_code=400, detail="无法停止该任务")

    _stop_flags[job_id] = True
    _pause_flags[job_id] = False
    job.status = "stopped"
    job.completed_at = datetime.utcnow()
    db.commit()
    return {"message": "训练任务已停止"}


@router.websocket("/ws/{job_id}")
async def training_websocket(websocket: WebSocket, job_id: str) -> None:
    """训练任务 WebSocket 连接"""
    await websocket.accept()
    logger.info(f"WebSocket 连接已建立: {job_id}")

    db = SessionLocal()
    try:
        while True:
            # 接收客户端控制消息（非阻塞）
            try:
                text = await asyncio.wait_for(websocket.receive_text(), timeout=1.0)
                payload = json.loads(text)
                action = payload.get("action")
                if action == "pause":
                    _pause_flags[job_id] = True
                elif action == "resume":
                    _pause_flags[job_id] = False
                elif action == "stop":
                    _stop_flags[job_id] = True
            except asyncio.TimeoutError:
                pass
            except json.JSONDecodeError:
                pass

            if job_id == "all":
                await websocket.send_json({"type": "heartbeat", "message": "ok"})
                continue

            job = db.query(TrainingJob).filter(TrainingJob.job_id == job_id).first()
            if not job:
                await websocket.send_json(
                    {
                        "type": "error",
                        "message": "训练任务不存在",
                        "job_id": job_id,
                    },
                )
                await asyncio.sleep(1.0)
                continue

            metadata = _extract_job_metadata(job)
            await websocket.send_json(
                {
                    "type": "status_update",
                    "job_id": job.job_id,
                    "experiment_name": metadata["experiment_name"],
                    "status": job.status,
                    "progress": job.progress,
                    "epoch": job.current_epoch,
                    "total_epochs": job.total_epochs,
                    "loss": job.current_loss,
                    "accuracy": job.current_accuracy,
                },
            )

            if job.status in {"completed", "failed", "stopped"}:
                await websocket.send_json(
                    {
                        "type": "training_complete"
                        if job.status == "completed"
                        else "error",
                        "job_id": job.job_id,
                        "message": job.error_message or job.status,
                    },
                )
                break
    except WebSocketDisconnect:
        logger.info(f"WebSocket 连接已断开: {job_id}")
    finally:
        db.close()
