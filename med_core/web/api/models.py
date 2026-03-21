"""模型管理 API"""

from __future__ import annotations

import random
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


def _collect_result_files(model: ModelInfo) -> list[dict[str, Any]]:
    config = model.config or {}
    artifact_paths = config.get("artifact_paths", {})
    candidates = [
        ("checkpoint", "模型权重", model.checkpoint_path),
        ("config", "训练配置", model.config_path or artifact_paths.get("config_path")),
        ("summary", "结果摘要", artifact_paths.get("summary_path")),
        ("metrics", "指标文件", artifact_paths.get("metrics_path")),
        ("report", "结果报告", artifact_paths.get("report_path")),
        ("log", "训练日志", artifact_paths.get("log_path")),
    ]

    result_files = []
    for key, label, path in candidates:
        if not path:
            continue
        result_files.append(
            {
                "key": key,
                "label": label,
                "path": path,
                "exists": Path(path).exists(),
            },
        )
    return result_files


def _build_demo_roc_curve(model: ModelInfo) -> dict[str, Any]:
    metrics = model.metrics or {}
    base_accuracy = model.accuracy or metrics.get("best_accuracy") or 0.85
    auc = metrics.get("auc") or min(0.995, max(0.7, base_accuracy + 0.035))
    points = []

    for index in range(21):
        fpr = index / 20
        tpr = min(1.0, max(0.0, fpr + (auc - 0.5) * 2 * (1 - fpr)))
        points.append(
            {
                "fpr": round(fpr, 3),
                "tpr": round(tpr, 3),
            },
        )

    return {"auc": round(auc, 4), "points": points}


def _build_demo_confusion_matrix(model: ModelInfo) -> dict[str, Any]:
    num_classes = min(max(model.num_classes or 2, 2), 4)
    accuracy = min(0.98, max(0.6, model.accuracy or 0.85))
    rng = random.Random(f"cm-{model.id}")
    labels = [f"Class {idx}" for idx in range(num_classes)]
    matrix = []

    for _ in range(num_classes):
        row_total = 100
        correct = int(row_total * accuracy)
        remaining = row_total - correct
        row = [0] * num_classes

        for col in range(num_classes):
            if remaining <= 0:
                break
            noise = rng.randint(0, remaining if col < num_classes - 1 else remaining)
            row[col] = noise
            remaining -= noise

        row[rng.randrange(num_classes)] += remaining
        matrix.append(row)

    for row_index, row in enumerate(matrix):
        diagonal_value = max(row[row_index], int(100 * accuracy))
        off_diagonal_sum = sum(value for idx, value in enumerate(row) if idx != row_index)
        diagonal_value = min(100, diagonal_value)
        scale = max(0, 100 - diagonal_value)

        if off_diagonal_sum > 0:
            adjusted = []
            running = 0
            for idx, value in enumerate(row):
                if idx == row_index:
                    adjusted.append(diagonal_value)
                    continue
                new_value = round(value / off_diagonal_sum * scale)
                adjusted.append(new_value)
                running += new_value
            diff = 100 - sum(adjusted)
            for idx in range(num_classes):
                if idx != row_index and diff != 0:
                    adjusted[idx] += 1 if diff > 0 else -1
                    diff += -1 if diff > 0 else 1
                    if diff == 0:
                        break
            matrix[row_index] = adjusted
        else:
            row[row_index] = 100
            matrix[row_index] = row

    return {"labels": labels, "matrix": matrix}


def _build_attention_grid(seed: str, size: int = 8) -> list[list[float]]:
    rng = random.Random(seed)
    center_x = rng.uniform(1.5, size - 2.0)
    center_y = rng.uniform(1.5, size - 2.0)
    secondary_x = rng.uniform(1.0, size - 1.0)
    secondary_y = rng.uniform(1.0, size - 1.0)

    grid: list[list[float]] = []
    for row in range(size):
        row_values = []
        for col in range(size):
            primary = max(0.0, 1.0 - (((row - center_y) ** 2 + (col - center_x) ** 2) / 18))
            secondary = max(
                0.0,
                0.65 - (((row - secondary_y) ** 2 + (col - secondary_x) ** 2) / 22),
            )
            value = min(1.0, primary + secondary + rng.uniform(0.0, 0.08))
            row_values.append(round(value, 3))
        grid.append(row_values)
    return grid


def _build_demo_attention_maps(model: ModelInfo) -> list[dict[str, Any]]:
    return [
        {
            "title": "影像模态注意力",
            "modality": "image",
            "grid": _build_attention_grid(f"image-{model.id}"),
        },
        {
            "title": "临床表格注意力",
            "modality": "tabular",
            "grid": _build_attention_grid(f"tabular-{model.id}"),
        },
        {
            "title": "病历文本注意力",
            "modality": "text",
            "grid": _build_attention_grid(f"text-{model.id}"),
        },
        {
            "title": "融合层注意力",
            "modality": "fusion",
            "grid": _build_attention_grid(f"fusion-{model.id}"),
        },
    ]


def _build_demo_visualizations(model: ModelInfo) -> dict[str, Any]:
    return {
        "roc_curve": _build_demo_roc_curve(model),
        "confusion_matrix": _build_demo_confusion_matrix(model),
        "attention_maps": _build_demo_attention_maps(model),
    }


def _to_payload(model: ModelInfo) -> dict[str, Any]:
    checkpoint_path = model.checkpoint_path
    file_size = Path(checkpoint_path).stat().st_size if checkpoint_path and Path(checkpoint_path).exists() else None
    model_format = _infer_format(checkpoint_path)

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
        "metrics": model.metrics,
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
        "visualizations": _build_demo_visualizations(model),
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
async def download_model_artifact(
    model_id: int,
    artifact_key: str,
    db: Session = Depends(get_db_session),
) -> FileResponse:
    """下载模型结果产物"""
    model = db.query(ModelInfo).filter(ModelInfo.id == model_id).first()
    if not model:
        raise HTTPException(status_code=404, detail="模型不存在")

    artifacts = {item["key"]: item for item in _collect_result_files(model)}
    artifact = artifacts.get(artifact_key)
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
