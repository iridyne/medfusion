"""模型 API"""
import os
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.crud import ModelCRUD

router = APIRouter()


class ModelCreate(BaseModel):
    """创建模型请求"""
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


class ModelUpdate(BaseModel):
    """更新模型请求"""
    name: str | None = None
    description: str | None = None
    accuracy: float | None = None
    loss: float | None = None
    metrics: dict[str, Any] | None = None
    tags: list[str] | None = None


# 模型存储目录
MODEL_STORAGE_DIR = Path("./storage/models")
MODEL_STORAGE_DIR.mkdir(parents=True, exist_ok=True)


@router.get("/")
async def list_models(
    skip: int = 0,
    limit: int = 100,
    backbone: str | None = None,
    format: str | None = None,
    sort_by: str = "created_at",
    order: str = "desc",
    db: Session = Depends(get_db)
):
    """获取模型列表"""
    models = ModelCRUD.list(
        db=db,
        skip=skip,
        limit=limit,
        backbone=backbone,
        format=format,
        sort_by=sort_by,
        order=order,
    )

    return {
        "models": [
            {
                "id": model.id,
                "name": model.name,
                "description": model.description,
                "backbone": model.backbone,
                "num_classes": model.num_classes,
                "accuracy": model.accuracy,
                "loss": model.loss,
                "metrics": model.metrics,
                "file_size": model.file_size,
                "format": model.format,
                "trained_epochs": model.trained_epochs,
                "tags": model.tags,
                "created_at": model.created_at.isoformat(),
            }
            for model in models
        ],
        "total": len(models),
    }


@router.get("/search")
async def search_models(
    keyword: str,
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db)
):
    """搜索模型"""
    models = ModelCRUD.search(db=db, keyword=keyword, skip=skip, limit=limit)

    return {
        "models": [
            {
                "id": model.id,
                "name": model.name,
                "description": model.description,
                "backbone": model.backbone,
                "accuracy": model.accuracy,
                "created_at": model.created_at.isoformat(),
            }
            for model in models
        ],
        "total": len(models),
    }


@router.get("/statistics")
async def get_statistics(db: Session = Depends(get_db)):
    """获取模型统计信息"""
    stats = ModelCRUD.get_statistics(db)

    return {
        "total_models": stats["total_count"],
        "total_size": stats["total_size"],
        "avg_accuracy": stats["avg_accuracy"],
    }


@router.get("/backbones")
async def get_backbones(db: Session = Depends(get_db)):
    """获取所有使用的 Backbone"""
    backbones = ModelCRUD.get_backbones(db)
    return {"backbones": backbones}


@router.get("/formats")
async def get_formats(db: Session = Depends(get_db)):
    """获取所有模型格式"""
    formats = ModelCRUD.get_formats(db)
    return {"formats": formats}


@router.get("/{model_id}")
async def get_model(model_id: int, db: Session = Depends(get_db)):
    """获取模型详情"""
    model = ModelCRUD.get(db, model_id)

    if not model:
        raise HTTPException(status_code=404, detail="Model not found")

    return {
        "id": model.id,
        "name": model.name,
        "description": model.description,
        "backbone": model.backbone,
        "num_classes": model.num_classes,
        "input_shape": model.input_shape,
        "accuracy": model.accuracy,
        "loss": model.loss,
        "metrics": model.metrics,
        "model_path": model.model_path,
        "file_size": model.file_size,
        "format": model.format,
        "training_job_id": model.training_job_id,
        "trained_epochs": model.trained_epochs,
        "tags": model.tags,
        "created_at": model.created_at.isoformat(),
        "created_by": model.created_by,
    }


@router.post("/")
async def create_model(
    model: ModelCreate,
    db: Session = Depends(get_db)
):
    """创建模型记录（不包含文件上传）"""
    # 检查名称是否已存在
    existing = ModelCRUD.get_by_name(db, model.name)
    if existing:
        raise HTTPException(status_code=400, detail=f"Model with name '{model.name}' already exists")

    # 创建模型路径（实际文件需要通过上传接口上传）
    model_path = str(MODEL_STORAGE_DIR / f"{model.name}.pth")

    db_model = ModelCRUD.create(
        db=db,
        name=model.name,
        description=model.description,
        backbone=model.backbone,
        num_classes=model.num_classes,
        model_path=model_path,
        accuracy=model.accuracy,
        loss=model.loss,
        metrics=model.metrics,
        format=model.format,
        input_shape=model.input_shape,
        trained_epochs=model.trained_epochs,
        tags=model.tags,
    )

    return {
        "id": db_model.id,
        "name": db_model.name,
        "status": "created",
        "created_at": db_model.created_at.isoformat(),
    }


@router.post("/{model_id}/upload")
async def upload_model_file(
    model_id: int,
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    """上传模型文件

    限制：
    - 文件大小：最大 500MB
    - 文件类型：.pth, .pt, .onnx, .h5, .pb
    """
    # 文件大小限制（500MB）
    MAX_FILE_SIZE = 500 * 1024 * 1024

    # 允许的文件扩展名
    ALLOWED_EXTENSIONS = {".pth", ".pt", ".onnx", ".h5", ".pb"}

    # 验证文件扩展名
    file_ext = os.path.splitext(file.filename)[1].lower()
    if file_ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type. Allowed types: {', '.join(ALLOWED_EXTENSIONS)}"
        )

    model = ModelCRUD.get(db, model_id)

    if not model:
        raise HTTPException(status_code=404, detail="Model not found")

    # 保存文件并验证大小
    file_path = MODEL_STORAGE_DIR / f"{model.name}_{model_id}{file_ext}"

    # 分块读取并验证文件大小
    total_size = 0
    chunk_size = 1024 * 1024  # 1MB chunks

    try:
        with open(file_path, "wb") as buffer:
            while chunk := await file.read(chunk_size):
                total_size += len(chunk)

                # 检查文件大小
                if total_size > MAX_FILE_SIZE:
                    buffer.close()
                    os.remove(file_path)  # 删除部分上传的文件
                    raise HTTPException(
                        status_code=413,
                        detail=f"File too large. Maximum size: {MAX_FILE_SIZE / (1024*1024):.0f}MB"
                    )

                buffer.write(chunk)
    except Exception:
        # 清理失败的上传
        if os.path.exists(file_path):
            os.remove(file_path)
        raise

    # 获取文件大小
    file_size = os.path.getsize(file_path)

    # 更新模型记录
    ModelCRUD.update(
        db=db,
        model_id=model_id,
        model_path=str(file_path),
        file_size=file_size,
    )

    return {
        "status": "uploaded",
        "file_path": str(file_path),
        "file_size": file_size,
        "file_type": file_ext,
    }


@router.get("/{model_id}/download")
async def download_model(model_id: int, db: Session = Depends(get_db)):
    """下载模型文件"""
    model = ModelCRUD.get(db, model_id)

    if not model:
        raise HTTPException(status_code=404, detail="Model not found")

    if not model.model_path or not os.path.exists(model.model_path):
        raise HTTPException(status_code=404, detail="Model file not found")

    return FileResponse(
        path=model.model_path,
        filename=f"{model.name}.pth",
        media_type="application/octet-stream"
    )


@router.put("/{model_id}")
async def update_model(
    model_id: int,
    model: ModelUpdate,
    db: Session = Depends(get_db)
):
    """更新模型信息"""
    db_model = ModelCRUD.update(
        db=db,
        model_id=model_id,
        **model.dict(exclude_unset=True)
    )

    if not db_model:
        raise HTTPException(status_code=404, detail="Model not found")

    return {
        "id": db_model.id,
        "name": db_model.name,
        "status": "updated",
    }


@router.delete("/{model_id}")
async def delete_model(model_id: int, db: Session = Depends(get_db)):
    """删除模型"""
    model = ModelCRUD.get(db, model_id)

    if not model:
        raise HTTPException(status_code=404, detail="Model not found")

    # 删除文件
    if model.model_path and os.path.exists(model.model_path):
        os.remove(model.model_path)

    # 删除数据库记录
    success = ModelCRUD.delete(db, model_id)

    if not success:
        raise HTTPException(status_code=500, detail="Failed to delete model")

    return {
        "status": "deleted",
        "id": model_id,
    }

