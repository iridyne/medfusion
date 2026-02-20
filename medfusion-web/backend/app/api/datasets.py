"""数据集 API"""
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.crud import DatasetCRUD

router = APIRouter()


class DatasetCreate(BaseModel):
    """创建数据集请求"""
    name: str
    description: str | None = None
    data_path: str
    num_samples: int | None = None
    num_classes: int | None = None
    train_samples: int | None = None
    val_samples: int | None = None
    test_samples: int | None = None
    class_distribution: dict[str, Any] | None = None
    tags: list[str] | None = None


class DatasetUpdate(BaseModel):
    """更新数据集请求"""
    name: str | None = None
    description: str | None = None
    num_samples: int | None = None
    num_classes: int | None = None
    train_samples: int | None = None
    val_samples: int | None = None
    test_samples: int | None = None
    class_distribution: dict[str, Any] | None = None
    tags: list[str] | None = None


# 数据集存储目录
DATASET_STORAGE_DIR = Path("./storage/datasets")
DATASET_STORAGE_DIR.mkdir(parents=True, exist_ok=True)


@router.get("/")
async def list_datasets(
    skip: int = 0,
    limit: int = 100,
    num_classes: int | None = None,
    sort_by: str = "created_at",
    order: str = "desc",
    db: Session = Depends(get_db)
):
    """获取数据集列表"""
    datasets = DatasetCRUD.list(
        db=db,
        skip=skip,
        limit=limit,
        num_classes=num_classes,
        sort_by=sort_by,
        order=order,
    )

    return {
        "datasets": [
            {
                "id": dataset.id,
                "name": dataset.name,
                "description": dataset.description,
                "data_path": dataset.data_path,
                "num_samples": dataset.num_samples,
                "num_classes": dataset.num_classes,
                "train_samples": dataset.train_samples,
                "val_samples": dataset.val_samples,
                "test_samples": dataset.test_samples,
                "class_distribution": dataset.class_distribution,
                "tags": dataset.tags,
                "created_at": dataset.created_at.isoformat(),
            }
            for dataset in datasets
        ],
        "total": len(datasets),
    }


@router.get("/search")
async def search_datasets(
    keyword: str,
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db)
):
    """搜索数据集"""
    datasets = DatasetCRUD.search(db=db, keyword=keyword, skip=skip, limit=limit)

    return {
        "datasets": [
            {
                "id": dataset.id,
                "name": dataset.name,
                "description": dataset.description,
                "num_samples": dataset.num_samples,
                "num_classes": dataset.num_classes,
                "created_at": dataset.created_at.isoformat(),
            }
            for dataset in datasets
        ],
        "total": len(datasets),
    }


@router.get("/statistics")
async def get_statistics(db: Session = Depends(get_db)):
    """获取数据集统计信息"""
    stats = DatasetCRUD.get_statistics(db)

    return {
        "total_datasets": stats["total_count"],
        "total_samples": stats["total_samples"],
        "avg_samples": stats["avg_samples"],
    }


@router.get("/class-counts")
async def get_class_counts(db: Session = Depends(get_db)):
    """获取所有数据集的类别数"""
    class_counts = DatasetCRUD.get_class_counts(db)
    return {"class_counts": class_counts}


@router.get("/{dataset_id}")
async def get_dataset(dataset_id: int, db: Session = Depends(get_db)):
    """获取数据集详情"""
    dataset = DatasetCRUD.get(db, dataset_id)

    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")

    return {
        "id": dataset.id,
        "name": dataset.name,
        "description": dataset.description,
        "data_path": dataset.data_path,
        "num_samples": dataset.num_samples,
        "num_classes": dataset.num_classes,
        "train_samples": dataset.train_samples,
        "val_samples": dataset.val_samples,
        "test_samples": dataset.test_samples,
        "class_distribution": dataset.class_distribution,
        "tags": dataset.tags,
        "created_at": dataset.created_at.isoformat(),
        "created_by": dataset.created_by,
    }


@router.post("/")
async def create_dataset(
    dataset: DatasetCreate,
    db: Session = Depends(get_db)
):
    """创建数据集记录"""
    # 检查名称是否已存在
    existing = DatasetCRUD.get_by_name(db, dataset.name)
    if existing:
        raise HTTPException(status_code=400, detail=f"Dataset with name '{dataset.name}' already exists")

    db_dataset = DatasetCRUD.create(
        db=db,
        name=dataset.name,
        description=dataset.description,
        data_path=dataset.data_path,
        num_samples=dataset.num_samples,
        num_classes=dataset.num_classes,
        train_samples=dataset.train_samples,
        val_samples=dataset.val_samples,
        test_samples=dataset.test_samples,
        class_distribution=dataset.class_distribution,
        tags=dataset.tags,
    )

    return {
        "id": db_dataset.id,
        "name": db_dataset.name,
        "status": "created",
        "created_at": db_dataset.created_at.isoformat(),
    }


@router.put("/{dataset_id}")
async def update_dataset(
    dataset_id: int,
    dataset: DatasetUpdate,
    db: Session = Depends(get_db)
):
    """更新数据集信息"""
    db_dataset = DatasetCRUD.update(
        db=db,
        dataset_id=dataset_id,
        **dataset.dict(exclude_unset=True)
    )

    if not db_dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")

    return {
        "id": db_dataset.id,
        "name": db_dataset.name,
        "status": "updated",
    }


@router.delete("/{dataset_id}")
async def delete_dataset(dataset_id: int, db: Session = Depends(get_db)):
    """删除数据集"""
    dataset = DatasetCRUD.get(db, dataset_id)

    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")

    # 删除数据库记录
    success = DatasetCRUD.delete(db, dataset_id)

    if not success:
        raise HTTPException(status_code=500, detail="Failed to delete dataset")

    return {
        "status": "deleted",
        "id": dataset_id,
    }


@router.post("/{dataset_id}/analyze")
async def analyze_dataset(dataset_id: int, db: Session = Depends(get_db)):
    """分析数据集（计算统计信息）"""
    dataset = DatasetCRUD.get(db, dataset_id)

    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")

    # 这里可以添加实际的数据集分析逻辑
    # 例如：扫描数据目录，计算样本数、类别分布等

    return {
        "status": "analyzed",
        "dataset_id": dataset_id,
        "message": "Dataset analysis completed",
    }
