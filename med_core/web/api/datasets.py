"""数据集管理 API"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy import asc, desc, func, or_
from sqlalchemy.orm import Session

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
    """分析数据集并写回简要结果"""
    dataset = db.query(DatasetInfo).filter(DatasetInfo.id == dataset_id).first()
    if not dataset:
        raise HTTPException(status_code=404, detail="数据集不存在")

    analysis = {
        "has_split": all(
            value is not None
            for value in [dataset.train_samples, dataset.val_samples, dataset.test_samples]
        ),
        "estimated_size_mb": round((dataset.size_bytes or 0) / (1024 * 1024), 2),
        "num_samples": dataset.num_samples or 0,
        "num_classes": dataset.num_classes or 0,
    }

    dataset.analysis = analysis
    db.commit()
    db.refresh(dataset)

    return {"dataset_id": dataset_id, "analysis": analysis}
