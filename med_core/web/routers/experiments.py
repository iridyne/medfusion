"""
Experiments Router - API endpoints for experiment comparison and reporting
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import uuid4

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from med_core.web.report_generator import ReportGenerator

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/experiments", tags=["experiments"])


# ============================================================================
# Pydantic Models
# ============================================================================


class ExperimentConfig(BaseModel):
    """Experiment configuration"""

    backbone: str
    fusion: str
    aggregator: Optional[str] = None
    learning_rate: float
    batch_size: int
    epochs: int
    optimizer: str = "adam"
    scheduler: Optional[str] = None


class ExperimentMetrics(BaseModel):
    """Experiment performance metrics"""

    accuracy: float
    precision: float
    recall: float
    f1_score: float
    auc: Optional[float] = None
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
    description: Optional[str] = None


class ExperimentListResponse(BaseModel):
    """Response for experiment list"""

    experiments: List[Experiment]
    total: int
    page: int
    page_size: int


class ComparisonMetric(BaseModel):
    """Single metric comparison across experiments"""

    metric: str
    experiments: Dict[str, float]
    best_experiment: str
    worst_experiment: str


class ComparisonResponse(BaseModel):
    """Response for experiment comparison"""

    experiments: List[Experiment]
    metrics: List[ComparisonMetric]
    summary: Dict[str, Any]


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
    history: List[TrainingHistory]


class ConfusionMatrixData(BaseModel):
    """Confusion matrix data"""

    classes: List[str]
    matrix: List[List[int]]
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
    points: List[ROCPoint]


class ReportRequest(BaseModel):
    """Request for report generation"""

    experiment_ids: List[str]
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


def get_mock_experiments() -> List[Experiment]:
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


# ============================================================================
# API Endpoints
# ============================================================================


@router.get("/", response_model=ExperimentListResponse)
async def list_experiments(
    page: int = Query(1, ge=1),
    page_size: int = Query(10, ge=1, le=100),
    status: Optional[str] = Query(None, pattern="^(completed|running|failed|pending)$"),
    sort_by: Optional[str] = Query(
        "created_at", pattern="^(created_at|accuracy|name)$"
    ),
    order: str = Query("desc", pattern="^(asc|desc)$"),
):
    """
    List all experiments with pagination and filtering
    """
    try:
        experiments = get_mock_experiments()

        # Filter by status
        if status:
            experiments = [exp for exp in experiments if exp.status == status]

        # Sort
        reverse = order == "desc"
        if sort_by == "accuracy":
            experiments.sort(key=lambda x: x.metrics.accuracy, reverse=reverse)
        elif sort_by == "name":
            experiments.sort(key=lambda x: x.name, reverse=reverse)
        else:  # created_at
            experiments.sort(key=lambda x: x.created_at, reverse=reverse)

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
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{experiment_id}", response_model=Experiment)
async def get_experiment(experiment_id: str):
    """
    Get details of a specific experiment
    """
    try:
        experiments = get_mock_experiments()
        experiment = next((exp for exp in experiments if exp.id == experiment_id), None)

        if not experiment:
            raise HTTPException(status_code=404, detail="Experiment not found")

        return experiment

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get experiment {experiment_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/compare", response_model=ComparisonResponse)
async def compare_experiments(experiment_ids: List[str]):
    """
    Compare multiple experiments
    """
    try:
        if len(experiment_ids) < 2:
            raise HTTPException(
                status_code=400, detail="At least 2 experiments required for comparison"
            )

        experiments = get_mock_experiments()
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
                values.items(), key=lambda x: x[1], reverse=not is_lower_better
            )
            best = sorted_items[0][0]
            worst = sorted_items[-1][0]

            metrics.append(
                ComparisonMetric(
                    metric=metric_name.replace("_", " ").title(),
                    experiments=values,
                    best_experiment=best,
                    worst_experiment=worst,
                )
            )

        # Build summary
        best_overall = max(selected, key=lambda x: x.metrics.accuracy)
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
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{experiment_id}/metrics", response_model=MetricsHistoryResponse)
async def get_metrics_history(experiment_id: str):
    """
    Get training metrics history for an experiment
    """
    try:
        experiments = get_mock_experiments()
        experiment = next((exp for exp in experiments if exp.id == experiment_id), None)

        if not experiment:
            raise HTTPException(status_code=404, detail="Experiment not found")

        # Generate mock training history
        history = []
        epochs = experiment.config.epochs
        final_loss = experiment.metrics.loss
        final_acc = experiment.metrics.accuracy

        for epoch in range(1, epochs + 1):
            progress = epoch / epochs
            # Simulate learning curve
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
                )
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
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{experiment_id}/confusion-matrix", response_model=ConfusionMatrixData)
async def get_confusion_matrix(experiment_id: str):
    """
    Get confusion matrix for an experiment
    """
    try:
        experiments = get_mock_experiments()
        experiment = next((exp for exp in experiments if exp.id == experiment_id), None)

        if not experiment:
            raise HTTPException(status_code=404, detail="Experiment not found")

        # Generate mock confusion matrix
        classes = ["Class 0", "Class 1", "Class 2", "Class 3"]
        accuracy = experiment.metrics.accuracy

        # Generate realistic confusion matrix based on accuracy
        matrix = []
        total = 0
        for i in range(len(classes)):
            row = []
            for j in range(len(classes)):
                if i == j:
                    # Diagonal (correct predictions)
                    value = int(150 * accuracy)
                else:
                    # Off-diagonal (misclassifications)
                    value = int(150 * (1 - accuracy) / (len(classes) - 1))
                row.append(value)
                total += value
            matrix.append(row)

        return ConfusionMatrixData(
            classes=classes,
            matrix=matrix,
            total=total,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get confusion matrix: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{experiment_id}/roc-curve", response_model=ROCCurveData)
async def get_roc_curve(experiment_id: str):
    """
    Get ROC curve data for an experiment
    """
    try:
        experiments = get_mock_experiments()
        experiment = next((exp for exp in experiments if exp.id == experiment_id), None)

        if not experiment:
            raise HTTPException(status_code=404, detail="Experiment not found")

        auc = experiment.metrics.auc or 0.85

        # Generate ROC curve points
        points = []
        num_points = 50

        for i in range(num_points + 1):
            fpr = i / num_points
            # Generate TPR based on AUC
            tpr = min(1.0, fpr + (auc - 0.5) * 2 * (1 - fpr))
            threshold = 1.0 - i / num_points

            points.append(
                ROCPoint(
                    fpr=round(fpr, 3),
                    tpr=round(tpr, 3),
                    threshold=round(threshold, 3),
                )
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
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/report", response_model=ReportResponse)
async def generate_report(request: ReportRequest):
    """
    Generate comparison report (Word or PDF)
    """
    try:
        experiments = get_mock_experiments()
        selected = [exp for exp in experiments if exp.id in request.experiment_ids]

        if not selected:
            raise HTTPException(
                status_code=404, detail="No experiments found with provided IDs"
            )

        # Prepare data for report generation
        experiments_data = [exp.model_dump() for exp in selected]

        # Mock comparison data (in real implementation, compute from actual data)
        comparison_data = {
            "statistical_tests": {
                "t_test": {"p_value": 0.032, "statistic": 2.45},
                "wilcoxon": {"p_value": 0.028, "statistic": 15.0},
            }
        }

        # Generate report using ReportGenerator
        report_generator = ReportGenerator(
            output_dir=Path.home() / ".medfusion" / "reports"
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
            f"Generated {request.format} report for {len(selected)} experiments: {report_path}"
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
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/reports/{filename}")
async def download_report(filename: str):
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
        raise HTTPException(status_code=500, detail=str(e))


@router.patch("/{experiment_id}/favorite")
async def toggle_favorite(experiment_id: str):
    """
    Toggle favorite status of an experiment
    """
    try:
        # In real implementation, update database
        return {"success": True, "experiment_id": experiment_id}

    except Exception as e:
        logger.error(f"Failed to toggle favorite: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/{experiment_id}")
async def delete_experiment(experiment_id: str):
    """
    Delete an experiment
    """
    try:
        # In real implementation, delete from database
        return {"success": True, "experiment_id": experiment_id}

    except Exception as e:
        logger.error(f"Failed to delete experiment: {e}")
        raise HTTPException(status_code=500, detail=str(e))
