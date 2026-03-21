"""模型管理 API"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel
from sqlalchemy import asc, desc, func, or_
from sqlalchemy.orm import Session

from ..config import settings
from ..database import get_db_session
from ..models import ModelInfo

router = APIRouter()


class ModelCreate(BaseModel):
    name: str
    description: str | None = None
    backbone: str
    num_classes: int
    accuracy: float | None = None
    loss: float | None = None
    metrics: dict[str, Any] | None = None
    format: str | None = "pytorch"
    input_shape: list[int] | None = None
    trained_epochs: int | None = None
    tags: list[str] | None = None
    model_path: str | None = None


class ModelUpdate(BaseModel):
    name: str | None = None
    description: str | None = None
    accuracy: float | None = None
    loss: float | None = None
    metrics: dict[str, Any] | None = None
    tags: list[str] | None = None
    trained_epochs: int | None = None
    num_classes: int | None = None


def _infer_format(path: str | None) -> str:
    if not path:
        return "pytorch"
    suffix = Path(path).suffix.lower()
    if suffix == ".onnx":
        return "onnx"
    if suffix in {".ts", ".pt", ".pth"}:
        return "pytorch"
    return "pytorch"


def _artifact_download_url(model_id: int, artifact_key: str) -> str:
    return f"/api/models/{model_id}/artifacts/{artifact_key}"


def _safe_load_json(path: str | None) -> dict[str, Any]:
    if not path:
        return {}
    artifact_path = Path(path)
    if not artifact_path.exists():
        return {}
    try:
        return json.loads(artifact_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def _is_image_artifact(path: str | None) -> bool:
    if not path:
        return False
    return Path(path).suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}


def _build_artifact_index(model: ModelInfo) -> dict[str, dict[str, Any]]:
    config = model.config or {}
    artifact_paths = config.get("artifact_paths", {})

    candidates: list[tuple[str, str, str | None]] = [
        ("checkpoint", "模型权重", model.checkpoint_path),
        ("config", "训练配置", model.config_path or artifact_paths.get("config_path")),
        ("summary", "结果摘要", artifact_paths.get("summary_path")),
        ("metrics", "指标文件", artifact_paths.get("metrics_path")),
        ("validation", "验证摘要", artifact_paths.get("validation_path")),
        ("report", "结果报告", artifact_paths.get("report_path")),
        ("log", "训练日志", artifact_paths.get("log_path")),
        ("history", "训练历史", artifact_paths.get("history_path")),
        ("predictions", "预测样本", artifact_paths.get("prediction_path")),
        ("roc_curve_json", "ROC 数据", artifact_paths.get("roc_curve_json_path")),
        ("roc_curve_plot", "ROC 曲线图", artifact_paths.get("roc_curve_plot_path")),
        (
            "confusion_matrix_json",
            "混淆矩阵数据",
            artifact_paths.get("confusion_matrix_json_path"),
        ),
        (
            "confusion_matrix_plot",
            "混淆矩阵图",
            artifact_paths.get("confusion_matrix_plot_path"),
        ),
        (
            "confusion_matrix_normalized_plot",
            "归一化混淆矩阵图",
            artifact_paths.get("confusion_matrix_normalized_plot_path"),
        ),
        (
            "training_curves_plot",
            "训练曲线图",
            artifact_paths.get("training_curves_plot_path"),
        ),
        (
            "calibration_curve_plot",
            "校准曲线图",
            artifact_paths.get("calibration_curve_plot_path"),
        ),
        (
            "probability_distribution_plot",
            "概率分布图",
            artifact_paths.get("probability_distribution_plot_path"),
        ),
    ]

    artifact_index: dict[str, dict[str, Any]] = {}
    for key, label, path in candidates:
        if not path or key in artifact_index:
            continue
        artifact_index[key] = {
            "key": key,
            "label": label,
            "path": path,
            "exists": Path(path).exists(),
            "download_url": _artifact_download_url(model.id, key),
            "is_image": _is_image_artifact(path),
            "preview_url": _artifact_download_url(model.id, key)
            if _is_image_artifact(path)
            else None,
        }

    attention_manifest = _safe_load_json(artifact_paths.get("attention_manifest_path"))
    for item in attention_manifest.get("items", []):
        artifact_key = item.get("artifact_key")
        image_path = item.get("image_path")
        if not artifact_key or not image_path or artifact_key in artifact_index:
            continue
        artifact_index[artifact_key] = {
            "key": artifact_key,
            "label": item.get("title", artifact_key),
            "path": image_path,
            "exists": Path(image_path).exists(),
            "download_url": _artifact_download_url(model.id, artifact_key),
            "is_image": True,
            "preview_url": _artifact_download_url(model.id, artifact_key),
        }

    statistics_key = attention_manifest.get("statistics_artifact_key")
    statistics_path = attention_manifest.get("statistics_plot_path")
    if statistics_key and statistics_path and statistics_key not in artifact_index:
        artifact_index[statistics_key] = {
            "key": statistics_key,
            "label": "注意力统计图",
            "path": statistics_path,
            "exists": Path(statistics_path).exists(),
            "download_url": _artifact_download_url(model.id, statistics_key),
            "is_image": True,
            "preview_url": _artifact_download_url(model.id, statistics_key),
        }

    return artifact_index


def _collect_result_files(model: ModelInfo) -> list[dict[str, Any]]:
    result_files = list(_build_artifact_index(model).values())
    result_files.sort(
        key=lambda item: (
            0
            if item["key"]
            in {
                "checkpoint",
                "config",
                "summary",
                "metrics",
                "validation",
                "history",
                "roc_curve_plot",
                "confusion_matrix_plot",
                "training_curves_plot",
            }
            else 1,
            item["label"],
        ),
    )
    return result_files


def _load_training_history(model: ModelInfo) -> dict[str, Any] | None:
    artifact_paths = (model.config or {}).get("artifact_paths", {})
    history_payload = _safe_load_json(artifact_paths.get("history_path"))
    entries = history_payload.get("entries")
    if not entries:
        return None

    plot_path = artifact_paths.get("training_curves_plot_path")
    return {
        "entries": entries,
        "plot_artifact_key": "training_curves_plot" if plot_path else None,
        "plot_url": _artifact_download_url(model.id, "training_curves_plot")
        if plot_path and Path(plot_path).exists()
        else None,
    }


def _load_visualizations(model: ModelInfo) -> dict[str, Any]:
    config = model.config or {}
    artifact_paths = config.get("artifact_paths", {})
    visualizations: dict[str, Any] = {}

    roc_payload = _safe_load_json(artifact_paths.get("roc_curve_json_path"))
    if roc_payload:
        plot_path = artifact_paths.get("roc_curve_plot_path")
        visualizations["roc_curve"] = {
            **roc_payload,
            "plot_artifact_key": "roc_curve_plot" if plot_path else None,
            "plot_url": _artifact_download_url(model.id, "roc_curve_plot")
            if plot_path and Path(plot_path).exists()
            else None,
        }

    confusion_payload = _safe_load_json(artifact_paths.get("confusion_matrix_json_path"))
    if confusion_payload:
        plot_path = artifact_paths.get("confusion_matrix_plot_path")
        normalized_plot_path = artifact_paths.get("confusion_matrix_normalized_plot_path")
        visualizations["confusion_matrix"] = {
            **confusion_payload,
            "plot_artifact_key": "confusion_matrix_plot" if plot_path else None,
            "plot_url": _artifact_download_url(model.id, "confusion_matrix_plot")
            if plot_path and Path(plot_path).exists()
            else None,
            "normalized_plot_artifact_key": (
                "confusion_matrix_normalized_plot" if normalized_plot_path else None
            ),
            "normalized_plot_url": _artifact_download_url(
                model.id,
                "confusion_matrix_normalized_plot",
            )
            if normalized_plot_path and Path(normalized_plot_path).exists()
            else None,
        }

    attention_manifest = _safe_load_json(artifact_paths.get("attention_manifest_path"))
    if attention_manifest:
        attention_items = []
        for item in attention_manifest.get("items", []):
            artifact_key = item.get("artifact_key")
            attention_items.append(
                {
                    **item,
                    "image_url": _artifact_download_url(model.id, artifact_key)
                    if artifact_key and item.get("image_path")
                    else None,
                }
            )
        visualizations["attention_maps"] = attention_items

        statistics_key = attention_manifest.get("statistics_artifact_key")
        statistics_path = attention_manifest.get("statistics_plot_path")
        if statistics_key and statistics_path and Path(statistics_path).exists():
            visualizations["attention_statistics"] = {
                "artifact_key": statistics_key,
                "image_url": _artifact_download_url(model.id, statistics_key),
            }

    calibration_path = artifact_paths.get("calibration_curve_plot_path")
    if calibration_path and Path(calibration_path).exists():
        visualizations["calibration_curve"] = {
            "artifact_key": "calibration_curve_plot",
            "image_url": _artifact_download_url(model.id, "calibration_curve_plot"),
        }

    probability_distribution_path = artifact_paths.get("probability_distribution_plot_path")
    if probability_distribution_path and Path(probability_distribution_path).exists():
        visualizations["probability_distribution"] = {
            "artifact_key": "probability_distribution_plot",
            "image_url": _artifact_download_url(model.id, "probability_distribution_plot"),
        }

    training_curves_path = artifact_paths.get("training_curves_plot_path")
    if training_curves_path and Path(training_curves_path).exists():
        visualizations["training_curves"] = {
            "artifact_key": "training_curves_plot",
            "image_url": _artifact_download_url(model.id, "training_curves_plot"),
        }

    return visualizations


def _load_validation(model: ModelInfo) -> dict[str, Any] | None:
    artifact_paths = (model.config or {}).get("artifact_paths", {})
    validation_payload = _safe_load_json(artifact_paths.get("validation_path"))
    return validation_payload or None


def _to_payload(model: ModelInfo) -> dict[str, Any]:
    checkpoint_path = model.checkpoint_path
    file_size = Path(checkpoint_path).stat().st_size if checkpoint_path and Path(checkpoint_path).exists() else None
    model_format = _infer_format(checkpoint_path)
    metrics = model.metrics or {}
    visualizations = _load_visualizations(model)
    training_history = _load_training_history(model)
    validation = _load_validation(model)

    return {
        "id": model.id,
        "name": model.name,
        "description": model.description,
        "model_type": model.model_type,
        "architecture": model.architecture,
        "backbone": model.architecture,
        "num_classes": model.num_classes,
        "accuracy": model.accuracy,
        "loss": model.loss,
        "metrics": metrics,
        "num_parameters": model.num_parameters,
        "params": model.num_parameters,
        "model_size_mb": model.model_size_mb,
        "file_size": file_size,
        "model_path": checkpoint_path,
        "checkpoint_path": checkpoint_path,
        "config": model.config,
        "config_path": model.config_path,
        "trained_epochs": model.trained_epochs,
        "training_time": model.training_time,
        "dataset_name": model.dataset_name,
        "tags": model.tags,
        "result_files": _collect_result_files(model),
        "training_history": training_history,
        "visualizations": visualizations,
        "validation": validation,
        "format": model_format,
        "created_at": model.created_at.isoformat(),
        "updated_at": model.updated_at.isoformat() if model.updated_at else None,
    }


@router.get("/")
async def list_models(
    skip: int = 0,
    limit: int = 20,
    backbone: str | None = None,
    format: str | None = None,
    sort_by: str = "created_at",
    order: str = "desc",
    db: Session = Depends(get_db_session),
) -> list[dict[str, Any]]:
    """获取模型列表"""
    query = db.query(ModelInfo)

    if backbone:
        query = query.filter(ModelInfo.architecture == backbone)

    sort_column = getattr(ModelInfo, sort_by, ModelInfo.created_at)
    sort_expr = desc(sort_column) if order.lower() == "desc" else asc(sort_column)
    models = query.order_by(sort_expr).offset(skip).limit(limit).all()

    payloads = [_to_payload(model) for model in models]
    if format:
        payloads = [item for item in payloads if item["format"] == format]
    return payloads


@router.get("/search")
async def search_models(
    keyword: str,
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db_session),
) -> list[dict[str, Any]]:
    """按名称/描述搜索模型"""
    pattern = f"%{keyword}%"
    models = (
        db.query(ModelInfo)
        .filter(
            or_(
                ModelInfo.name.ilike(pattern),
                ModelInfo.description.ilike(pattern),
            ),
        )
        .order_by(ModelInfo.created_at.desc())
        .offset(skip)
        .limit(limit)
        .all()
    )
    return [_to_payload(model) for model in models]


@router.get("/statistics")
async def get_model_statistics(db: Session = Depends(get_db_session)) -> dict[str, Any]:
    """获取模型统计信息"""
    total_models = db.query(func.count(ModelInfo.id)).scalar() or 0

    models = db.query(ModelInfo).all()
    total_size = 0
    accuracy_values = []
    for model in models:
        if model.checkpoint_path and Path(model.checkpoint_path).exists():
            total_size += Path(model.checkpoint_path).stat().st_size
        if model.accuracy is not None:
            accuracy_values.append(model.accuracy)

    avg_accuracy = sum(accuracy_values) / len(accuracy_values) if accuracy_values else 0.0
    return {
        "total_models": int(total_models),
        "total_size": int(total_size),
        "avg_accuracy": avg_accuracy,
    }


@router.get("/backbones")
async def get_backbones(db: Session = Depends(get_db_session)) -> dict[str, list[str]]:
    """获取已有模型中使用的 backbone"""
    rows = db.query(ModelInfo.architecture).distinct().all()
    values = sorted({row[0] for row in rows if row[0]})
    return {"backbones": values}


@router.get("/formats")
async def get_formats(db: Session = Depends(get_db_session)) -> dict[str, list[str]]:
    """获取已有模型的格式列表"""
    models = db.query(ModelInfo).all()
    formats = sorted({_infer_format(model.checkpoint_path) for model in models})
    return {"formats": formats}


@router.get("/{model_id}")
async def get_model(model_id: int, db: Session = Depends(get_db_session)) -> dict[str, Any]:
    """获取模型详情"""
    model = db.query(ModelInfo).filter(ModelInfo.id == model_id).first()

    if not model:
        raise HTTPException(status_code=404, detail="模型不存在")
    return _to_payload(model)


@router.post("/")
async def create_model(
    payload: ModelCreate,
    db: Session = Depends(get_db_session),
) -> dict[str, Any]:
    """创建模型记录"""
    checkpoint_path = payload.model_path or str(
        settings.data_dir / "models" / f"{payload.name}_{int(datetime.utcnow().timestamp())}.pth",
    )
    model = ModelInfo(
        name=payload.name,
        description=payload.description,
        model_type="classification",
        architecture=payload.backbone,
        config={"input_shape": payload.input_shape, "format": payload.format},
        metrics=payload.metrics,
        accuracy=payload.accuracy,
        loss=payload.loss,
        checkpoint_path=checkpoint_path,
        trained_epochs=payload.trained_epochs,
        num_classes=payload.num_classes,
        tags=payload.tags,
    )
    db.add(model)
    db.commit()
    db.refresh(model)
    return _to_payload(model)


@router.put("/{model_id}")
async def update_model(
    model_id: int,
    payload: ModelUpdate,
    db: Session = Depends(get_db_session),
) -> dict[str, Any]:
    """更新模型信息"""
    model = db.query(ModelInfo).filter(ModelInfo.id == model_id).first()
    if not model:
        raise HTTPException(status_code=404, detail="模型不存在")

    updates = payload.model_dump(exclude_unset=True)
    for key, value in updates.items():
        if key == "num_classes":
            model.num_classes = value
        else:
            setattr(model, key, value)

    db.commit()
    db.refresh(model)
    return _to_payload(model)


@router.post("/{model_id}/upload")
async def upload_model_file(
    model_id: int,
    file: UploadFile = File(...),
    db: Session = Depends(get_db_session),
) -> dict[str, Any]:
    """上传模型文件"""
    model = db.query(ModelInfo).filter(ModelInfo.id == model_id).first()
    if not model:
        raise HTTPException(status_code=404, detail="模型不存在")

    target_dir = settings.data_dir / "models"
    target_dir.mkdir(parents=True, exist_ok=True)
    target_path = target_dir / f"{model.id}_{file.filename}"

    with target_path.open("wb") as f:
        f.write(await file.read())

    file_size = target_path.stat().st_size
    model.checkpoint_path = str(target_path)
    model.model_size_mb = file_size / (1024 * 1024)
    db.commit()
    db.refresh(model)

    return {
        "message": "模型文件上传成功",
        "model_id": model.id,
        "model_path": str(target_path),
        "file_size": file_size,
    }


@router.get("/{model_id}/download")
@router.head("/{model_id}/download", include_in_schema=False)
async def download_model(
    model_id: int,
    db: Session = Depends(get_db_session),
) -> FileResponse:
    """下载模型文件"""
    model = db.query(ModelInfo).filter(ModelInfo.id == model_id).first()
    if not model:
        raise HTTPException(status_code=404, detail="模型不存在")

    model_path = Path(model.checkpoint_path)
    if not model_path.exists():
        raise HTTPException(status_code=404, detail="模型文件不存在")

    return FileResponse(path=model_path, filename=model_path.name)


@router.get("/{model_id}/artifacts/{artifact_key}")
@router.head("/{model_id}/artifacts/{artifact_key}", include_in_schema=False)
async def download_model_artifact(
    model_id: int,
    artifact_key: str,
    db: Session = Depends(get_db_session),
) -> FileResponse:
    """下载模型结果产物"""
    model = db.query(ModelInfo).filter(ModelInfo.id == model_id).first()
    if not model:
        raise HTTPException(status_code=404, detail="模型不存在")

    artifact = _build_artifact_index(model).get(artifact_key)
    if not artifact:
        raise HTTPException(status_code=404, detail="结果文件不存在")

    artifact_path = Path(artifact["path"])
    if not artifact_path.exists():
        raise HTTPException(status_code=404, detail="结果文件不存在")

    return FileResponse(path=artifact_path, filename=artifact_path.name)


@router.delete("/{model_id}")
async def delete_model(
    model_id: int,
    db: Session = Depends(get_db_session),
) -> dict[str, str]:
    """删除模型"""
    model = db.query(ModelInfo).filter(ModelInfo.id == model_id).first()

    if not model:
        raise HTTPException(status_code=404, detail="模型不存在")

    db.delete(model)
    db.commit()
    return {"message": "模型已删除"}
