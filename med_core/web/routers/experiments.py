"""
Experiments Router - API endpoints for experiment comparison and reporting
"""

import logging
import json
from datetime import datetime
from operator import attrgetter, itemgetter
from pathlib import Path
from typing import Any
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from med_core.web.report_generator import ReportGenerator

from ..database import get_db_session
from ..models import Experiment as ExperimentRecord
from ..models import ModelInfo

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/experiments", tags=["experiments"])


# ============================================================================
# Pydantic Models
# ============================================================================


class ExperimentConfig(BaseModel):
    """Experiment configuration"""

    backbone: str
    fusion: str
    aggregator: str | None = None
    learning_rate: float
    batch_size: int
    epochs: int
    optimizer: str = "adam"
    scheduler: str | None = None


class ExperimentMetrics(BaseModel):
    """Experiment performance metrics"""

    accuracy: float
    precision: float
    recall: float
    f1_score: float
    auc: float | None = None
    loss: float


class Experiment(BaseModel):
    """Experiment model"""

    id: str
    name: str
    status: str = Field(..., pattern="^(completed|running|failed|pending)$")
    config: ExperimentConfig
    metrics: ExperimentMetrics
    training_time: int  # seconds
    created_at: str
    updated_at: str
    is_favorite: bool = False
    description: str | None = None


class ExperimentListResponse(BaseModel):
    """Response for experiment list"""

    experiments: list[Experiment]
    total: int
    page: int
    page_size: int


class ComparisonMetric(BaseModel):
    """Single metric comparison across experiments"""

    metric: str
    experiments: dict[str, float]
    best_experiment: str
    worst_experiment: str


class ComparisonResponse(BaseModel):
    """Response for experiment comparison"""

    experiments: list[Experiment]
    metrics: list[ComparisonMetric]
    summary: dict[str, Any]


class TrainingHistory(BaseModel):
    """Training history data"""

    epoch: int
    train_loss: float
    val_loss: float
    train_accuracy: float
    val_accuracy: float
    learning_rate: float


class MetricsHistoryResponse(BaseModel):
    """Response for training metrics history"""

    experiment_id: str
    experiment_name: str
    history: list[TrainingHistory]


class ConfusionMatrixData(BaseModel):
    """Confusion matrix data"""

    classes: list[str]
    matrix: list[list[int]]
    total: int


class ROCPoint(BaseModel):
    """Single point on ROC curve"""

    fpr: float
    tpr: float
    threshold: float


class ROCCurveData(BaseModel):
    """ROC curve data for one experiment"""

    experiment_id: str
    experiment_name: str
    auc: float
    points: list[ROCPoint]


class ReportRequest(BaseModel):
    """Request for report generation"""

    experiment_ids: list[str]
    format: str = Field(..., pattern="^(word|pdf)$")
    include_charts: bool = True
    include_config: bool = True
    include_metrics: bool = True


class ReportResponse(BaseModel):
    """Response for report generation"""

    report_id: str
    download_url: str
    format: str
    created_at: str


# ============================================================================
# Mock Data (Replace with actual database queries)
# ============================================================================


def get_mock_experiments() -> list[Experiment]:
    """Generate mock experiment data"""
    return [
        Experiment(
            id="exp-001",
            name="ResNet50 + Concatenate",
            status="completed",
            config=ExperimentConfig(
                backbone="resnet50",
                fusion="concatenate",
                learning_rate=0.001,
                batch_size=32,
                epochs=50,
                optimizer="adam",
            ),
            metrics=ExperimentMetrics(
                accuracy=0.892,
                precision=0.885,
                recall=0.878,
                f1_score=0.881,
                auc=0.945,
                loss=0.234,
            ),
            training_time=3600,
            created_at="2026-02-15T10:30:00Z",
            updated_at="2026-02-15T12:30:00Z",
            is_favorite=True,
            description="Baseline model with ResNet50 backbone",
        ),
        Experiment(
            id="exp-002",
            name="ViT-Base + Attention",
            status="completed",
            config=ExperimentConfig(
                backbone="vit_base",
                fusion="attention",
                learning_rate=0.0001,
                batch_size=16,
                epochs=50,
                optimizer="adamw",
                scheduler="cosine",
            ),
            metrics=ExperimentMetrics(
                accuracy=0.915,
                precision=0.908,
                recall=0.902,
                f1_score=0.905,
                auc=0.962,
                loss=0.198,
            ),
            training_time=5400,
            created_at="2026-02-16T14:20:00Z",
            updated_at="2026-02-16T16:50:00Z",
            is_favorite=False,
            description="Vision Transformer with attention fusion",
        ),
        Experiment(
            id="exp-003",
            name="Swin-Base + Gated",
            status="completed",
            config=ExperimentConfig(
                backbone="swin_base",
                fusion="gated",
                aggregator="attention",
                learning_rate=0.0005,
                batch_size=24,
                epochs=50,
                optimizer="adamw",
                scheduler="step",
            ),
            metrics=ExperimentMetrics(
                accuracy=0.928,
                precision=0.922,
                recall=0.918,
                f1_score=0.920,
                auc=0.971,
                loss=0.176,
            ),
            training_time=7200,
            created_at="2026-02-17T09:15:00Z",
            updated_at="2026-02-17T11:15:00Z",
            is_favorite=True,
            description="Swin Transformer with gated fusion",
        ),
        Experiment(
            id="exp-004",
            name="EfficientNet-B3 + Bilinear",
            status="completed",
            config=ExperimentConfig(
                backbone="efficientnet_b3",
                fusion="bilinear",
                learning_rate=0.001,
                batch_size=32,
                epochs=50,
                optimizer="adam",
            ),
            metrics=ExperimentMetrics(
                accuracy=0.901,
                precision=0.895,
                recall=0.889,
                f1_score=0.892,
                auc=0.953,
                loss=0.215,
            ),
            training_time=4200,
            created_at="2026-02-18T11:45:00Z",
            updated_at="2026-02-18T13:15:00Z",
            is_favorite=False,
            description="EfficientNet with bilinear pooling",
        ),
    ]


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _extract_status(raw_status: str | None) -> str:
    status = (raw_status or "").lower()
    status_map = {
        "created": "pending",
        "queued": "pending",
        "running": "running",
        "paused": "running",
        "completed": "completed",
        "failed": "failed",
        "stopped": "failed",
    }
    return status_map.get(status, "pending")


def _parse_numeric_experiment_id(experiment_id: str) -> int | None:
    if experiment_id.startswith("model-"):
        suffix = experiment_id.removeprefix("model-")
        return int(suffix) if suffix.isdigit() else None
    if experiment_id.startswith("run-"):
        suffix = experiment_id.removeprefix("run-")
        return int(suffix) if suffix.isdigit() else None
    return int(experiment_id) if experiment_id.isdigit() else None


def _experiment_from_model(model: ModelInfo) -> Experiment:
    metrics_payload = model.metrics or {}
    validation_overview = (
        ((model.config or {}).get("validation") or {}).get("overview", {})
        if isinstance(model.config, dict)
        else {}
    )

    accuracy = _safe_float(
        model.accuracy
        if model.accuracy is not None
        else metrics_payload.get("accuracy", validation_overview.get("accuracy")),
    )
    precision = _safe_float(
        metrics_payload.get("precision", validation_overview.get("precision_macro", accuracy)),
        default=accuracy,
    )
    recall = _safe_float(
        metrics_payload.get("recall", validation_overview.get("recall_macro", accuracy)),
        default=accuracy,
    )
    f1_score = _safe_float(
        metrics_payload.get("f1_score", validation_overview.get("macro_f1", accuracy)),
        default=accuracy,
    )
    auc = metrics_payload.get("auc")
    if auc is None:
        auc = metrics_payload.get("auc_roc", validation_overview.get("auc"))
    loss = _safe_float(
        model.loss
        if model.loss is not None
        else metrics_payload.get("loss", metrics_payload.get("best_loss", 0.0)),
    )

    tags = model.tags if isinstance(model.tags, list) else []
    created_at = model.created_at.isoformat() if model.created_at else datetime.now().isoformat()
    updated_at = (
        model.updated_at.isoformat()
        if model.updated_at is not None
        else created_at
    )

    return Experiment(
        id=f"model-{model.id}",
        name=model.name,
        status="completed",
        config=ExperimentConfig(
            backbone=model.architecture or "unknown",
            fusion="concatenate",
            aggregator=None,
            learning_rate=0.001,
            batch_size=16,
            epochs=model.trained_epochs or 1,
            optimizer="adam",
            scheduler=None,
        ),
        metrics=ExperimentMetrics(
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1_score,
            auc=_safe_float(auc) if auc is not None else None,
            loss=loss,
        ),
        training_time=int(model.training_time or 0),
        created_at=created_at,
        updated_at=updated_at,
        is_favorite="favorite" in tags,
        description=model.description,
    )


def _experiment_from_record(record: ExperimentRecord) -> Experiment:
    config_payload = record.config if isinstance(record.config, dict) else {}
    model_payload = config_payload.get("model", {})
    training_payload = config_payload.get("training", {})
    optimizer_payload = training_payload.get("optimizer", {})
    scheduler_payload = training_payload.get("scheduler", {})
    data_payload = config_payload.get("data", {})

    metrics_payload = record.metrics if isinstance(record.metrics, dict) else {}
    accuracy = _safe_float(metrics_payload.get("accuracy"))
    precision = _safe_float(metrics_payload.get("precision"), default=accuracy)
    recall = _safe_float(metrics_payload.get("recall"), default=accuracy)
    f1_score = _safe_float(
        metrics_payload.get("f1_score", metrics_payload.get("macro_f1", accuracy)),
        default=accuracy,
    )
    auc = metrics_payload.get("auc")
    if auc is None:
        auc = metrics_payload.get("auc_roc")
    loss = _safe_float(
        metrics_payload.get("loss", metrics_payload.get("best_loss", 0.0)),
    )

    created_at = record.created_at.isoformat() if record.created_at else datetime.now().isoformat()
    updated_reference = record.completed_at or record.started_at or record.created_at
    updated_at = updated_reference.isoformat() if updated_reference else created_at
    training_time = 0
    if record.started_at and record.completed_at:
        training_time = int((record.completed_at - record.started_at).total_seconds())

    tags = record.tags if isinstance(record.tags, list) else []

    return Experiment(
        id=f"run-{record.id}",
        name=record.name,
        status=_extract_status(record.status),
        config=ExperimentConfig(
            backbone=(
                model_payload.get("vision", {}).get("backbone")
                if isinstance(model_payload.get("vision"), dict)
                else model_payload.get("backbone", "unknown")
            )
            or "unknown",
            fusion=(
                model_payload.get("fusion", {}).get("fusion_type")
                if isinstance(model_payload.get("fusion"), dict)
                else model_payload.get("fusion_type", "concatenate")
            )
            or "concatenate",
            aggregator=model_payload.get("aggregator"),
            learning_rate=_safe_float(
                optimizer_payload.get("learning_rate", config_payload.get("learning_rate", 0.001)),
                default=0.001,
            ),
            batch_size=int(data_payload.get("batch_size", config_payload.get("batch_size", 16))),
            epochs=int(training_payload.get("num_epochs", config_payload.get("epochs", 1))),
            optimizer=str(optimizer_payload.get("optimizer", "adam")),
            scheduler=(
                str(scheduler_payload.get("scheduler"))
                if scheduler_payload.get("scheduler") is not None
                else None
            ),
        ),
        metrics=ExperimentMetrics(
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1_score,
            auc=_safe_float(auc) if auc is not None else None,
            loss=loss,
        ),
        training_time=training_time,
        created_at=created_at,
        updated_at=updated_at,
        is_favorite="favorite" in tags,
        description=record.description,
    )


def _load_real_experiments(db: Session) -> list[Experiment]:
    experiment_rows = (
        db.query(ExperimentRecord).order_by(ExperimentRecord.created_at.desc()).all()
    )
    if experiment_rows:
        return [_experiment_from_record(item) for item in experiment_rows]

    model_rows = db.query(ModelInfo).order_by(ModelInfo.created_at.desc()).all()
    return [_experiment_from_model(item) for item in model_rows]


def _resolve_experiments(db: Session) -> list[Experiment]:
    real = _load_real_experiments(db)
    if real:
        return real
    return get_mock_experiments()


def _load_json_payload(path: str | None) -> dict[str, Any] | list[Any] | None:
    if not path:
        return None
    payload_path = Path(path)
    if not payload_path.exists():
        return None
    try:
        return json.loads(payload_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        logger.warning("Invalid JSON payload at %s", payload_path)
        return None


def _artifact_paths_from_model(model: ModelInfo | None) -> dict[str, Any]:
    if model is None or not isinstance(model.config, dict):
        return {}
    artifact_paths = model.config.get("artifact_paths")
    return artifact_paths if isinstance(artifact_paths, dict) else {}


def _resolve_model_for_experiment_id(
    db: Session,
    experiment_id: str,
) -> ModelInfo | None:
    numeric_id = _parse_numeric_experiment_id(experiment_id)

    if experiment_id.startswith("model-") and numeric_id is not None:
        return db.query(ModelInfo).filter(ModelInfo.id == numeric_id).first()

    if experiment_id.startswith("run-") and numeric_id is not None:
        record = (
            db.query(ExperimentRecord).filter(ExperimentRecord.id == numeric_id).first()
        )
        if record is not None and record.checkpoint_path:
            matched = (
                db.query(ModelInfo)
                .filter(ModelInfo.checkpoint_path == record.checkpoint_path)
                .order_by(ModelInfo.id.desc())
                .first()
            )
            if matched is not None:
                return matched
        return None

    if numeric_id is not None:
        return db.query(ModelInfo).filter(ModelInfo.id == numeric_id).first()

    return None


def _parse_history_entries(payload: dict[str, Any] | list[Any] | None) -> list[dict[str, Any]]:
    if payload is None:
        return []
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    entries = payload.get("entries")
    if isinstance(entries, list):
        return [item for item in entries if isinstance(item, dict)]
    return []


def _parse_roc_points(raw_points: Any) -> list[ROCPoint]:
    if not isinstance(raw_points, list):
        return []

    parsed: list[ROCPoint] = []
    for index, point in enumerate(raw_points):
        if isinstance(point, dict):
            fpr = point.get("fpr")
            tpr = point.get("tpr")
            threshold = point.get("threshold")
        elif (
            isinstance(point, list) or isinstance(point, tuple)
        ) and len(point) >= 2:
            fpr = point[0]
            tpr = point[1]
            threshold = point[2] if len(point) >= 3 else (1.0 - float(fpr))
        else:
            continue

        try:
            fpr_value = float(fpr)
            tpr_value = float(tpr)
            threshold_value = float(threshold) if threshold is not None else 1.0 - fpr_value
        except (TypeError, ValueError):
            continue

        parsed.append(
            ROCPoint(
                fpr=round(fpr_value, 3),
                tpr=round(tpr_value, 3),
                threshold=round(threshold_value, 3),
            )
        )

        # Prevent runaway payloads; frontend chart does not need huge point counts.
        if index >= 2048:
            break

    return parsed


# ============================================================================
# API Endpoints
# ============================================================================


@router.get("/", response_model=ExperimentListResponse)
async def list_experiments(
    page: int = Query(1, ge=1),
    page_size: int = Query(10, ge=1, le=100),
    status: str | None = Query(None, pattern="^(completed|running|failed|pending)$"),
    sort_by: str | None = Query("created_at", pattern="^(created_at|accuracy|name)$"),
    order: str = Query("desc", pattern="^(asc|desc)$"),
    db: Session = Depends(get_db_session),
) -> ExperimentListResponse:
    """
    List all experiments with pagination and filtering
    """
    try:
        experiments = _resolve_experiments(db)

        # Filter by status
        if status:
            experiments = [exp for exp in experiments if exp.status == status]

        # Sort
        reverse = order == "desc"
        if sort_by == "accuracy":
            experiments.sort(key=attrgetter("metrics.accuracy"), reverse=reverse)
        elif sort_by == "name":
            experiments.sort(key=attrgetter("name"), reverse=reverse)
        else:  # created_at
            experiments.sort(key=attrgetter("created_at"), reverse=reverse)

        # Pagination
        total = len(experiments)
        start = (page - 1) * page_size
        end = start + page_size
        paginated = experiments[start:end]

        return ExperimentListResponse(
            experiments=paginated,
            total=total,
            page=page,
            page_size=page_size,
        )

    except Exception as e:
        logger.error(f"Failed to list experiments: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/{experiment_id}", response_model=Experiment)
async def get_experiment(
    experiment_id: str,
    db: Session = Depends(get_db_session),
) -> Experiment:
    """
    Get details of a specific experiment
    """
    try:
        # Keep compatibility for both legacy mock IDs and real DB-backed IDs.
        experiments = _resolve_experiments(db)
        experiment = next((exp for exp in experiments if exp.id == experiment_id), None)

        if not experiment:
            raise HTTPException(status_code=404, detail="Experiment not found")

        return experiment

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get experiment {experiment_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/compare", response_model=ComparisonResponse)
async def compare_experiments(
    experiment_ids: list[str],
    db: Session = Depends(get_db_session),
) -> ComparisonResponse:
    """
    Compare multiple experiments
    """
    try:
        if len(experiment_ids) < 2:
            raise HTTPException(
                status_code=400, detail="At least 2 experiments required for comparison",
            )

        experiments = _resolve_experiments(db)
        selected = [exp for exp in experiments if exp.id in experiment_ids]

        if len(selected) != len(experiment_ids):
            raise HTTPException(status_code=404, detail="Some experiments not found")

        # Build comparison metrics
        metrics = []
        metric_names = ["accuracy", "precision", "recall", "f1_score", "auc", "loss"]

        for metric_name in metric_names:
            values = {}
            for exp in selected:
                value = getattr(exp.metrics, metric_name)
                if value is not None:
                    values[exp.name] = value

            if not values:
                continue

            # Find best and worst
            is_lower_better = metric_name == "loss"
            sorted_items = sorted(
                values.items(), key=itemgetter(1), reverse=not is_lower_better,
            )
            best = sorted_items[0][0]
            worst = sorted_items[-1][0]

            metrics.append(
                ComparisonMetric(
                    metric=metric_name.replace("_", " ").title(),
                    experiments=values,
                    best_experiment=best,
                    worst_experiment=worst,
                ),
            )

        # Build summary
        best_overall = max(selected, key=attrgetter("metrics.accuracy"))
        summary = {
            "best_experiment": best_overall.name,
            "best_accuracy": best_overall.metrics.accuracy,
            "avg_accuracy": sum(exp.metrics.accuracy for exp in selected)
            / len(selected),
            "avg_training_time": sum(exp.training_time for exp in selected)
            / len(selected),
        }

        return ComparisonResponse(
            experiments=selected,
            metrics=metrics,
            summary=summary,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to compare experiments: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/{experiment_id}/metrics", response_model=MetricsHistoryResponse)
async def get_metrics_history(
    experiment_id: str,
    db: Session = Depends(get_db_session),
) -> MetricsHistoryResponse:
    """
    Get training metrics history for an experiment
    """
    try:
        experiments = _resolve_experiments(db)
        experiment = next((exp for exp in experiments if exp.id == experiment_id), None)

        if not experiment:
            raise HTTPException(status_code=404, detail="Experiment not found")

        history: list[TrainingHistory] = []
        model = _resolve_model_for_experiment_id(db, experiment_id)
        artifact_paths = _artifact_paths_from_model(model)
        history_payload = _load_json_payload(artifact_paths.get("history_path"))
        entries = _parse_history_entries(history_payload)

        if entries:
            for index, entry in enumerate(entries):
                epoch_value = entry.get("epoch")
                try:
                    epoch = int(epoch_value) if epoch_value is not None else index + 1
                except (TypeError, ValueError):
                    epoch = index + 1

                history.append(
                    TrainingHistory(
                        epoch=epoch,
                        train_loss=round(_safe_float(entry.get("train_loss")), 4),
                        val_loss=round(_safe_float(entry.get("val_loss")), 4),
                        train_accuracy=round(_safe_float(entry.get("train_accuracy")), 4),
                        val_accuracy=round(_safe_float(entry.get("val_accuracy")), 4),
                        learning_rate=_safe_float(
                            entry.get("learning_rate", experiment.config.learning_rate),
                            default=experiment.config.learning_rate,
                        ),
                    )
                )
        else:
            # Fallback synthetic curve only when no real history artifact is available.
            epochs = experiment.config.epochs
            final_loss = experiment.metrics.loss
            final_acc = experiment.metrics.accuracy

            for epoch in range(1, epochs + 1):
                progress = epoch / epochs
                train_loss = final_loss + (1.0 - final_loss) * (1 - progress) ** 2
                val_loss = final_loss + (1.2 - final_loss) * (1 - progress) ** 2
                train_acc = final_acc * (0.5 + 0.5 * progress)
                val_acc = final_acc * (0.4 + 0.6 * progress)
                lr = experiment.config.learning_rate * (0.1 ** (epoch // 20))

                history.append(
                    TrainingHistory(
                        epoch=epoch,
                        train_loss=round(train_loss, 4),
                        val_loss=round(val_loss, 4),
                        train_accuracy=round(train_acc, 4),
                        val_accuracy=round(val_acc, 4),
                        learning_rate=lr,
                    ),
                )

        return MetricsHistoryResponse(
            experiment_id=experiment.id,
            experiment_name=experiment.name,
            history=history,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get metrics history: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/{experiment_id}/confusion-matrix", response_model=ConfusionMatrixData)
async def get_confusion_matrix(
    experiment_id: str,
    db: Session = Depends(get_db_session),
) -> ConfusionMatrixData:
    """
    Get confusion matrix for an experiment
    """
    try:
        experiments = _resolve_experiments(db)
        experiment = next((exp for exp in experiments if exp.id == experiment_id), None)

        if not experiment:
            raise HTTPException(status_code=404, detail="Experiment not found")

        model = _resolve_model_for_experiment_id(db, experiment_id)
        artifact_paths = _artifact_paths_from_model(model)
        confusion_payload = _load_json_payload(
            artifact_paths.get("confusion_matrix_json_path")
        )
        classes: list[str] | None = None
        matrix: list[list[int]] | None = None

        if isinstance(confusion_payload, dict):
            raw_labels = confusion_payload.get("labels") or confusion_payload.get("classes")
            raw_matrix = confusion_payload.get("matrix")
            if isinstance(raw_labels, list) and isinstance(raw_matrix, list):
                parsed_matrix: list[list[int]] = []
                for row in raw_matrix:
                    if not isinstance(row, list):
                        continue
                    parsed_row: list[int] = []
                    for cell in row:
                        try:
                            parsed_row.append(int(cell))
                        except (TypeError, ValueError):
                            parsed_row.append(0)
                    parsed_matrix.append(parsed_row)

                if parsed_matrix and all(len(row) == len(parsed_matrix[0]) for row in parsed_matrix):
                    classes = [str(item) for item in raw_labels]
                    matrix = parsed_matrix

        if classes is None or matrix is None:
            classes = ["Class 0", "Class 1", "Class 2", "Class 3"]
            accuracy = experiment.metrics.accuracy
            matrix = [
                [
                    int(150 * accuracy)
                    if i == j
                    else int(150 * (1 - accuracy) / (len(classes) - 1))
                    for j in range(len(classes))
                ]
                for i in range(len(classes))
            ]

        total = sum(sum(row) for row in matrix)

        return ConfusionMatrixData(
            classes=classes,
            matrix=matrix,
            total=total,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get confusion matrix: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/{experiment_id}/roc-curve", response_model=ROCCurveData)
async def get_roc_curve(
    experiment_id: str,
    db: Session = Depends(get_db_session),
) -> ROCCurveData:
    """
    Get ROC curve data for an experiment
    """
    try:
        experiments = _resolve_experiments(db)
        experiment = next((exp for exp in experiments if exp.id == experiment_id), None)

        if not experiment:
            raise HTTPException(status_code=404, detail="Experiment not found")

        model = _resolve_model_for_experiment_id(db, experiment_id)
        artifact_paths = _artifact_paths_from_model(model)
        roc_payload = _load_json_payload(artifact_paths.get("roc_curve_json_path"))

        auc = experiment.metrics.auc or 0.85
        points: list[ROCPoint] = []
        if isinstance(roc_payload, dict):
            auc_value = roc_payload.get("auc", roc_payload.get("auc_roc"))
            if auc_value is not None:
                auc = _safe_float(auc_value, default=auc)
            points = _parse_roc_points(roc_payload.get("points"))

        if not points:
            num_points = 50
            for i in range(num_points + 1):
                fpr = i / num_points
                tpr = min(1.0, fpr + (auc - 0.5) * 2 * (1 - fpr))
                threshold = 1.0 - i / num_points
                points.append(
                    ROCPoint(
                        fpr=round(fpr, 3),
                        tpr=round(tpr, 3),
                        threshold=round(threshold, 3),
                    ),
                )

        return ROCCurveData(
            experiment_id=experiment.id,
            experiment_name=experiment.name,
            auc=auc,
            points=points,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get ROC curve: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/report", response_model=ReportResponse)
async def generate_report(
    request: ReportRequest,
    db: Session = Depends(get_db_session),
) -> ReportResponse:
    """
    Generate comparison report (Word or PDF)
    """
    try:
        experiments = _resolve_experiments(db)
        selected = [exp for exp in experiments if exp.id in request.experiment_ids]

        if not selected:
            raise HTTPException(
                status_code=404, detail="No experiments found with provided IDs",
            )

        # Prepare data for report generation
        experiments_data = [exp.model_dump() for exp in selected]

        # Mock comparison data (in real implementation, compute from actual data)
        comparison_data = {
            "statistical_tests": {
                "t_test": {"p_value": 0.032, "statistic": 2.45},
                "wilcoxon": {"p_value": 0.028, "statistic": 15.0},
            },
        }

        # Generate report using ReportGenerator
        report_generator = ReportGenerator(
            output_dir=Path.home() / ".medfusion" / "reports",
        )

        report_id = str(uuid4())
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        if request.format == "word":
            filename = f"report_{report_id}_{timestamp}.docx"
            report_path = report_generator.generate_word_report(
                experiments=experiments_data,
                comparison_data=comparison_data,
                output_filename=filename,
            )
        else:  # pdf
            filename = f"report_{report_id}_{timestamp}.pdf"
            report_path = report_generator.generate_pdf_report(
                experiments=experiments_data,
                comparison_data=comparison_data,
                output_filename=filename,
            )

        download_url = f"/api/experiments/reports/{report_path.name}"

        logger.info(
            f"Generated {request.format} report for {len(selected)} experiments: {report_path}",
        )

        return ReportResponse(
            report_id=report_id,
            download_url=download_url,
            format=request.format,
            created_at=datetime.now().isoformat(),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to generate report: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/reports/{filename}")
async def download_report(filename: str) -> FileResponse:
    """
    Download a generated report file
    """
    try:
        report_path = Path.home() / ".medfusion" / "reports" / filename

        if not report_path.exists():
            raise HTTPException(status_code=404, detail="Report not found")

        # Determine media type based on extension
        media_type = (
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
            if filename.endswith(".docx")
            else "application/pdf"
        )

        return FileResponse(
            path=str(report_path),
            media_type=media_type,
            filename=filename,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to download report: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.patch("/{experiment_id}/favorite")
async def toggle_favorite(
    experiment_id: str,
    db: Session = Depends(get_db_session),
) -> dict[str, Any]:
    """
    Toggle favorite status of an experiment
    """
    try:
        numeric_id = _parse_numeric_experiment_id(experiment_id)

        model_record: ModelInfo | None = None
        experiment_record: ExperimentRecord | None = None
        if experiment_id.startswith("model-") and numeric_id is not None:
            model_record = db.query(ModelInfo).filter(ModelInfo.id == numeric_id).first()
        elif experiment_id.startswith("run-") and numeric_id is not None:
            experiment_record = (
                db.query(ExperimentRecord).filter(ExperimentRecord.id == numeric_id).first()
            )
        elif numeric_id is not None:
            experiment_record = (
                db.query(ExperimentRecord).filter(ExperimentRecord.id == numeric_id).first()
            )
            if experiment_record is None:
                model_record = db.query(ModelInfo).filter(ModelInfo.id == numeric_id).first()
        elif experiment_id.startswith("exp-"):
            # Legacy mock ids: keep backward-compatible no-op success.
            return {"success": True, "experiment_id": experiment_id, "is_favorite": False}

        if model_record is None and experiment_record is None:
            raise HTTPException(status_code=404, detail="Experiment not found")

        if model_record is not None:
            tags = model_record.tags if isinstance(model_record.tags, list) else []
            is_favorite = "favorite" in tags
            if is_favorite:
                tags = [tag for tag in tags if tag != "favorite"]
            else:
                tags = [*tags, "favorite"]
            model_record.tags = tags
            db.commit()
            return {
                "success": True,
                "experiment_id": experiment_id,
                "is_favorite": not is_favorite,
            }

        tags = (
            experiment_record.tags
            if isinstance(experiment_record.tags, list)
            else []
        )
        is_favorite = "favorite" in tags
        if is_favorite:
            tags = [tag for tag in tags if tag != "favorite"]
        else:
            tags = [*tags, "favorite"]
        experiment_record.tags = tags
        db.commit()
        return {
            "success": True,
            "experiment_id": experiment_id,
            "is_favorite": not is_favorite,
        }

    except Exception as e:
        logger.error(f"Failed to toggle favorite: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.delete("/{experiment_id}")
async def delete_experiment(
    experiment_id: str,
    db: Session = Depends(get_db_session),
) -> dict[str, Any]:
    """
    Delete an experiment
    """
    try:
        numeric_id = _parse_numeric_experiment_id(experiment_id)
        if experiment_id.startswith("exp-"):
            # Legacy mock ids: keep backward-compatible no-op success.
            return {"success": True, "experiment_id": experiment_id}

        if experiment_id.startswith("model-") and numeric_id is not None:
            record = db.query(ModelInfo).filter(ModelInfo.id == numeric_id).first()
            if record is None:
                raise HTTPException(status_code=404, detail="Experiment not found")
            db.delete(record)
            db.commit()
            return {"success": True, "experiment_id": experiment_id}

        if experiment_id.startswith("run-") and numeric_id is not None:
            record = (
                db.query(ExperimentRecord).filter(ExperimentRecord.id == numeric_id).first()
            )
            if record is None:
                raise HTTPException(status_code=404, detail="Experiment not found")
            db.delete(record)
            db.commit()
            return {"success": True, "experiment_id": experiment_id}

        if numeric_id is not None:
            exp_record = (
                db.query(ExperimentRecord).filter(ExperimentRecord.id == numeric_id).first()
            )
            if exp_record is not None:
                db.delete(exp_record)
                db.commit()
                return {"success": True, "experiment_id": experiment_id}
            model_record = db.query(ModelInfo).filter(ModelInfo.id == numeric_id).first()
            if model_record is not None:
                db.delete(model_record)
                db.commit()
                return {"success": True, "experiment_id": experiment_id}

        raise HTTPException(status_code=404, detail="Experiment not found")

    except Exception as e:
        logger.error(f"Failed to delete experiment: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e
