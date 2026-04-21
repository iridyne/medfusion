"""数据集管理 API"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy import asc, desc, func, or_
from sqlalchemy.orm import Session

from ..application.training_jobs import (
    _IMAGE_COLUMN_CANDIDATES,
    _PATIENT_ID_COLUMN_CANDIDATES,
    _TARGET_COLUMN_CANDIDATES,
    _infer_csv_path,
    _infer_feature_columns,
    _infer_image_dir,
    _infer_num_classes,
    _pick_column,
    _read_csv_preview,
)
from ..database import get_db_session
from ..models import DatasetInfo

router = APIRouter()


class DatasetCreate(BaseModel):
    name: str
    description: str | None = None
    data_path: str
    dataset_type: str = "image"
    status: str = "ready"
    size_bytes: int | None = None
    num_samples: int | None = None
    num_classes: int | None = None
    train_samples: int | None = None
    val_samples: int | None = None
    test_samples: int | None = None
    class_distribution: dict[str, Any] | None = None
    tags: list[str] | None = None
    created_by: str | None = None


class DatasetUpdate(BaseModel):
    name: str | None = None
    description: str | None = None
    data_path: str | None = None
    dataset_type: str | None = None
    status: str | None = None
    size_bytes: int | None = None
    num_samples: int | None = None
    num_classes: int | None = None
    train_samples: int | None = None
    val_samples: int | None = None
    test_samples: int | None = None
    class_distribution: dict[str, Any] | None = None
    tags: list[str] | None = None
    created_by: str | None = None


class DatasetInspectRequest(BaseModel):
    data_path: str
    dataset_type: str = "image"
    csv_path: str | None = None
    image_dir: str | None = None
    image_path_column: str | None = None
    target_column: str | None = None
    patient_id_column: str | None = None
    numerical_features: list[str] | None = None
    categorical_features: list[str] | None = None
    num_classes: int | None = None


def _estimate_path_size(data_path: str) -> int | None:
    path = Path(data_path)
    if not path.exists():
        return None
    if path.is_file():
        return path.stat().st_size

    total = 0
    try:
        for p in path.rglob("*"):
            if p.is_file():
                total += p.stat().st_size
    except Exception:
        return None
    return total


def _count_rows(csv_path: Path) -> int:
    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        next(reader, None)
        return sum(1 for _ in reader)


def _probe_image_paths(
    *,
    image_dir: Path,
    sample_rows: list[dict[str, str]],
    image_column: str,
    limit: int = 8,
) -> dict[str, Any]:
    probed: list[dict[str, Any]] = []
    existing = 0

    for row in sample_rows:
        relative_path = row.get(image_column)
        if not relative_path:
            continue
        resolved = Path(relative_path)
        if not resolved.is_absolute():
            resolved = image_dir / resolved
        exists = resolved.exists()
        if exists:
            existing += 1
        probed.append(
            {
                "value": relative_path,
                "resolved_path": str(resolved.resolve()),
                "exists": exists,
            }
        )
        if len(probed) >= limit:
            break

    return {
        "checked": len(probed),
        "existing": existing,
        "missing": len(probed) - existing,
        "samples": probed,
    }


def _build_readiness_summary(
    *,
    dataset_type: str,
    dataset_path: Path,
    csv_path: Path | None,
    headers: list[str],
    sample_rows: list[dict[str, str]],
    image_path_column: str | None,
    target_column: str | None,
    patient_id_column: str | None,
    numerical_features: list[str],
    categorical_features: list[str],
    num_classes: int | None,
    image_probe: dict[str, Any] | None,
) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    checks: list[dict[str, Any]] = []

    path_exists = dataset_path.exists()
    checks.append(
        {
            "key": "path_exists",
            "label": "本地路径存在",
            "status": "pass" if path_exists else "fail",
            "detail": str(dataset_path),
        }
    )
    if not path_exists:
        errors.append(f"数据路径不存在：{dataset_path}")

    if csv_path is None:
        checks.append(
            {
                "key": "csv_manifest",
                "label": "CSV / manifest",
                "status": "fail",
                "detail": "未找到可读的 CSV 描述文件",
            }
        )
        errors.append("未找到可读的 CSV / manifest 文件")
    else:
        checks.append(
            {
                "key": "csv_manifest",
                "label": "CSV / manifest",
                "status": "pass",
                "detail": str(csv_path),
            }
        )

    if target_column:
        checks.append(
            {
                "key": "target_column",
                "label": "标签列",
                "status": "pass",
                "detail": target_column,
            }
        )
    else:
        checks.append(
            {
                "key": "target_column",
                "label": "标签列",
                "status": "fail",
                "detail": "未识别 target_column",
            }
        )
        errors.append("未识别标签列 target_column")

    if dataset_type in {"image", "multimodal"}:
        if image_path_column:
            checks.append(
                {
                    "key": "image_column",
                    "label": "图像路径列",
                    "status": "pass",
                    "detail": image_path_column,
                }
            )
        else:
            checks.append(
                {
                    "key": "image_column",
                    "label": "图像路径列",
                    "status": "fail",
                    "detail": "未识别 image_path_column",
                }
            )
            errors.append("未识别图像路径列 image_path_column")

    if dataset_type in {"tabular", "multimodal"}:
        feature_count = len(numerical_features) + len(categorical_features)
        checks.append(
            {
                "key": "tabular_features",
                "label": "表格特征列",
                "status": "pass" if feature_count > 0 else "warning",
                "detail": (
                    f"数值 {len(numerical_features)} / 类别 {len(categorical_features)}"
                ),
            }
        )
        if feature_count == 0:
            warnings.append("未识别到明确的表格特征列，训练时可能只能走图像支路")

    if patient_id_column is None:
        warnings.append("未识别 patient_id_column，后续 cohort / case 维度分析会更弱")

    if not sample_rows:
        errors.append("CSV 为空，无法预览样本")

    if num_classes is not None and num_classes < 2:
        warnings.append("推断出的类别数小于 2，请确认标签列是否正确")

    if image_probe is not None:
        if image_probe["checked"] == 0:
            warnings.append("未从预览样本中找到可探测的图像路径")
        elif image_probe["missing"] > 0:
            errors.append(
                f"抽样检查到 {image_probe['missing']} 条图像路径不存在，请确认 image_dir 和 CSV 内容"
            )
        checks.append(
            {
                "key": "image_probe",
                "label": "图像路径抽样",
                "status": "pass" if image_probe["missing"] == 0 else "fail",
                "detail": f"已检查 {image_probe['checked']} 条，存在 {image_probe['existing']} 条",
            }
        )

    status = "ready" if not errors else "blocked"
    if status == "ready" and warnings:
        status = "warning"

    return {
        "status": status,
        "can_enter_training": status != "blocked",
        "errors": errors,
        "warnings": warnings,
        "checks": checks,
        "summary": {
            "headers": len(headers),
            "preview_rows": len(sample_rows),
            "num_classes": num_classes,
            "numerical_features": len(numerical_features),
            "categorical_features": len(categorical_features),
        },
        "next_step": (
            "可进入训练或 RunSpec 向导"
            if status != "blocked"
            else "先修复路径 / CSV / 列名问题，再进入训练"
        ),
    }


def _inspect_dataset_payload(
    *,
    payload: DatasetInspectRequest,
) -> dict[str, Any]:
    dataset_path = Path(payload.data_path).expanduser()
    size_bytes = _estimate_path_size(str(dataset_path))
    path_payload = {
        "path": str(dataset_path),
        "exists": dataset_path.exists(),
        "kind": (
            "file" if dataset_path.is_file() else "directory" if dataset_path.is_dir() else "missing"
        ),
        "size_bytes": size_bytes,
        "estimated_size_mb": round((size_bytes or 0) / (1024 * 1024), 2),
    }

    headers: list[str] = []
    sample_rows: list[dict[str, str]] = []
    csv_path: Path | None = None
    image_dir: Path | None = None
    image_path_column: str | None = None
    target_column: str | None = None
    patient_id_column: str | None = None
    numerical_features = payload.numerical_features or []
    categorical_features = payload.categorical_features or []
    num_classes = payload.num_classes
    csv_error: str | None = None
    image_probe: dict[str, Any] | None = None

    try:
        csv_path = _infer_csv_path(
            project_root=Path.cwd(),
            dataset_path=dataset_path if dataset_path.exists() else None,
            explicit_csv_path=payload.csv_path,
        )
        headers, sample_rows = _read_csv_preview(csv_path)
        image_path_column = _pick_column(
            headers,
            preferred=payload.image_path_column,
            candidates=_IMAGE_COLUMN_CANDIDATES,
        )
        target_column = _pick_column(
            headers,
            preferred=payload.target_column,
            candidates=_TARGET_COLUMN_CANDIDATES,
        )
        patient_id_column = _pick_column(
            headers,
            preferred=payload.patient_id_column,
            candidates=_PATIENT_ID_COLUMN_CANDIDATES,
        )
        if not numerical_features and not categorical_features and headers:
            numerical_features, categorical_features = _infer_feature_columns(
                headers,
                sample_rows,
                excluded_columns={
                    image_path_column or "",
                    target_column or "",
                    patient_id_column or "",
                },
            )
        if target_column:
            num_classes = _infer_num_classes(csv_path, target_column, payload.num_classes)
        if image_path_column:
            image_dir = _infer_image_dir(
                project_root=Path.cwd(),
                dataset_path=dataset_path if dataset_path.exists() else None,
                csv_path=csv_path,
                sample_rows=sample_rows,
                image_column=image_path_column,
                explicit_image_dir=payload.image_dir,
            )
            image_probe = _probe_image_paths(
                image_dir=image_dir,
                sample_rows=sample_rows,
                image_column=image_path_column,
            )
    except Exception as exc:
        csv_error = str(exc)

    readiness = _build_readiness_summary(
        dataset_type=payload.dataset_type,
        dataset_path=dataset_path,
        csv_path=csv_path,
        headers=headers,
        sample_rows=sample_rows,
        image_path_column=image_path_column,
        target_column=target_column,
        patient_id_column=patient_id_column,
        numerical_features=numerical_features,
        categorical_features=categorical_features,
        num_classes=num_classes,
        image_probe=image_probe,
    )
    if csv_error:
        readiness["status"] = "blocked"
        readiness["can_enter_training"] = False
        readiness["errors"] = [*readiness["errors"], csv_error]

    row_count = None
    if csv_path and csv_path.exists():
        try:
            row_count = _count_rows(csv_path)
        except Exception:
            row_count = None

    return {
        "path": path_payload,
        "csv": {
            "path": str(csv_path) if csv_path else None,
            "error": csv_error,
            "headers": headers,
            "row_count": row_count,
            "preview_rows": sample_rows[:5],
        },
        "schema": {
            "image_path_column": image_path_column,
            "target_column": target_column,
            "patient_id_column": patient_id_column,
            "numerical_features": numerical_features,
            "categorical_features": categorical_features,
            "num_classes": num_classes,
            "image_dir": str(image_dir) if image_dir else None,
        },
        "image_probe": image_probe,
        "readiness": readiness,
    }


def _dataset_to_inspect_request(dataset: DatasetInfo) -> DatasetInspectRequest:
    analysis = dataset.analysis or {}
    schema = analysis.get("schema", {}) if isinstance(analysis, dict) else {}
    return DatasetInspectRequest(
        data_path=dataset.data_path,
        dataset_type=dataset.dataset_type,
        csv_path=schema.get("csv_path"),
        image_dir=schema.get("image_dir"),
        image_path_column=schema.get("image_path_column"),
        target_column=schema.get("target_column"),
        patient_id_column=schema.get("patient_id_column"),
        numerical_features=schema.get("numerical_features"),
        categorical_features=schema.get("categorical_features"),
        num_classes=dataset.num_classes,
    )


def _to_payload(dataset: DatasetInfo) -> dict[str, Any]:
    return {
        "id": dataset.id,
        "name": dataset.name,
        "description": dataset.description,
        "data_path": dataset.data_path,
        "dataset_type": dataset.dataset_type,
        "status": dataset.status,
        "size_bytes": dataset.size_bytes,
        "num_samples": dataset.num_samples,
        "num_classes": dataset.num_classes,
        "train_samples": dataset.train_samples,
        "val_samples": dataset.val_samples,
        "test_samples": dataset.test_samples,
        "class_distribution": dataset.class_distribution,
        "tags": dataset.tags,
        "created_by": dataset.created_by,
        "analysis": dataset.analysis,
        "created_at": dataset.created_at.isoformat(),
        "updated_at": dataset.updated_at.isoformat() if dataset.updated_at else None,
    }


@router.get("/")
async def list_datasets(
    skip: int = 0,
    limit: int = 20,
    num_classes: int | None = None,
    sort_by: str = "created_at",
    order: str = "desc",
    db: Session = Depends(get_db_session),
) -> list[dict[str, Any]]:
    """获取数据集列表"""
    query = db.query(DatasetInfo)

    if num_classes is not None:
        query = query.filter(DatasetInfo.num_classes == num_classes)

    sort_column = getattr(DatasetInfo, sort_by, DatasetInfo.created_at)
    sort_expr = desc(sort_column) if order.lower() == "desc" else asc(sort_column)

    datasets = query.order_by(sort_expr).offset(skip).limit(limit).all()
    return [_to_payload(dataset) for dataset in datasets]


@router.get("/search")
async def search_datasets(
    keyword: str = Query(..., min_length=1),
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db_session),
) -> list[dict[str, Any]]:
    """按名称/描述搜索数据集"""
    pattern = f"%{keyword}%"
    datasets = (
        db.query(DatasetInfo)
        .filter(
            or_(
                DatasetInfo.name.ilike(pattern),
                DatasetInfo.description.ilike(pattern),
            ),
        )
        .order_by(DatasetInfo.created_at.desc())
        .offset(skip)
        .limit(limit)
        .all()
    )
    return [_to_payload(dataset) for dataset in datasets]


@router.get("/statistics")
async def get_dataset_statistics(
    db: Session = Depends(get_db_session),
) -> dict[str, Any]:
    """获取数据集统计信息"""
    total_datasets = db.query(func.count(DatasetInfo.id)).scalar() or 0
    total_samples = db.query(func.coalesce(func.sum(DatasetInfo.num_samples), 0)).scalar() or 0
    avg_samples = float(total_samples / total_datasets) if total_datasets > 0 else 0.0

    return {
        "total_datasets": int(total_datasets),
        "total_samples": int(total_samples),
        "avg_samples": avg_samples,
    }


@router.get("/class-counts")
async def get_class_counts(
    db: Session = Depends(get_db_session),
) -> dict[str, list[int]]:
    """获取所有已使用的类别数"""
    rows = (
        db.query(DatasetInfo.num_classes)
        .filter(DatasetInfo.num_classes.isnot(None))
        .distinct()
        .all()
    )
    values = sorted({int(row[0]) for row in rows if row[0] is not None})
    return {"class_counts": values}


@router.get("/{dataset_id}")
async def get_dataset(
    dataset_id: int,
    db: Session = Depends(get_db_session),
) -> dict[str, Any]:
    """获取数据集详情"""
    dataset = db.query(DatasetInfo).filter(DatasetInfo.id == dataset_id).first()
    if not dataset:
        raise HTTPException(status_code=404, detail="数据集不存在")
    return _to_payload(dataset)


@router.post("/")
async def create_dataset(
    payload: DatasetCreate,
    db: Session = Depends(get_db_session),
) -> dict[str, Any]:
    """创建数据集记录"""
    size_bytes = payload.size_bytes
    if size_bytes is None:
        size_bytes = _estimate_path_size(payload.data_path)

    dataset = DatasetInfo(
        name=payload.name,
        description=payload.description,
        data_path=payload.data_path,
        dataset_type=payload.dataset_type,
        status=payload.status,
        size_bytes=size_bytes,
        num_samples=payload.num_samples,
        num_classes=payload.num_classes,
        train_samples=payload.train_samples,
        val_samples=payload.val_samples,
        test_samples=payload.test_samples,
        class_distribution=payload.class_distribution,
        tags=payload.tags,
        created_by=payload.created_by,
    )
    db.add(dataset)
    db.commit()
    db.refresh(dataset)
    return _to_payload(dataset)


@router.post("/inspect")
async def inspect_dataset(payload: DatasetInspectRequest) -> dict[str, Any]:
    """Inspect an arbitrary dataset path before or after registration."""
    return _inspect_dataset_payload(payload=payload)


@router.put("/{dataset_id}")
async def update_dataset(
    dataset_id: int,
    payload: DatasetUpdate,
    db: Session = Depends(get_db_session),
) -> dict[str, Any]:
    """更新数据集信息"""
    dataset = db.query(DatasetInfo).filter(DatasetInfo.id == dataset_id).first()
    if not dataset:
        raise HTTPException(status_code=404, detail="数据集不存在")

    updates = payload.model_dump(exclude_unset=True)
    for key, value in updates.items():
        setattr(dataset, key, value)

    if "data_path" in updates and "size_bytes" not in updates:
        dataset.size_bytes = _estimate_path_size(dataset.data_path)

    db.commit()
    db.refresh(dataset)
    return _to_payload(dataset)


@router.delete("/{dataset_id}")
async def delete_dataset(
    dataset_id: int,
    db: Session = Depends(get_db_session),
) -> dict[str, str]:
    """删除数据集"""
    dataset = db.query(DatasetInfo).filter(DatasetInfo.id == dataset_id).first()
    if not dataset:
        raise HTTPException(status_code=404, detail="数据集不存在")

    db.delete(dataset)
    db.commit()
    return {"message": "数据集已删除"}


@router.post("/{dataset_id}/analyze")
async def analyze_dataset(
    dataset_id: int,
    db: Session = Depends(get_db_session),
) -> dict[str, Any]:
    """分析数据集并写回 readiness / schema 结果。"""
    dataset = db.query(DatasetInfo).filter(DatasetInfo.id == dataset_id).first()
    if not dataset:
        raise HTTPException(status_code=404, detail="数据集不存在")

    inspection = _inspect_dataset_payload(
        payload=_dataset_to_inspect_request(dataset),
    )
    analysis = {
        "has_split": all(
            value is not None
            for value in [dataset.train_samples, dataset.val_samples, dataset.test_samples]
        ),
        "estimated_size_mb": round((dataset.size_bytes or 0) / (1024 * 1024), 2),
        "num_samples": inspection["csv"].get("row_count") or dataset.num_samples or 0,
        "num_classes": inspection["schema"].get("num_classes") or dataset.num_classes or 0,
        "path": inspection["path"],
        "schema": {
            "csv_path": inspection["csv"].get("path"),
            "image_dir": inspection["schema"].get("image_dir"),
            "image_path_column": inspection["schema"].get("image_path_column"),
            "target_column": inspection["schema"].get("target_column"),
            "patient_id_column": inspection["schema"].get("patient_id_column"),
            "numerical_features": inspection["schema"].get("numerical_features"),
            "categorical_features": inspection["schema"].get("categorical_features"),
        },
        "readiness": inspection["readiness"],
        "image_probe": inspection.get("image_probe"),
        "csv_preview": inspection["csv"].get("preview_rows"),
    }

    dataset.analysis = analysis
    dataset.status = "ready" if inspection["readiness"]["can_enter_training"] else "error"
    dataset.num_samples = analysis["num_samples"] or dataset.num_samples
    dataset.num_classes = analysis["num_classes"] or dataset.num_classes
    db.commit()
    db.refresh(dataset)

    return {"dataset_id": dataset_id, "analysis": analysis}


@router.get("/{dataset_id}/readiness")
async def get_dataset_readiness(
    dataset_id: int,
    db: Session = Depends(get_db_session),
) -> dict[str, Any]:
    dataset = db.query(DatasetInfo).filter(DatasetInfo.id == dataset_id).first()
    if not dataset:
        raise HTTPException(status_code=404, detail="数据集不存在")

    inspection = _inspect_dataset_payload(
        payload=_dataset_to_inspect_request(dataset),
    )
    return {
        "dataset_id": dataset_id,
        "dataset_name": dataset.name,
        **inspection,
    }
