"""模型管理 API"""

from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from ..database import get_db_session
from ..models import ModelInfo

router = APIRouter()


class ModelResponse(BaseModel):
    """模型响应"""

    id: int
    name: str
    description: str | None
    model_type: str
    architecture: str
    accuracy: float | None
    num_parameters: int | None
    created_at: str

    class Config:
        from_attributes = True


@router.get("")
async def list_models(
    skip: int = 0, limit: int = 20, db: Session = Depends(get_db_session),
) -> list[ModelResponse]:
    """获取模型列表"""
    models = (
        db.query(ModelInfo)
        .order_by(ModelInfo.created_at.desc())
        .offset(skip)
        .limit(limit)
        .all()
    )

    return [
        ModelResponse(
            id=model.id,
            name=model.name,
            description=model.description,
            model_type=model.model_type,
            architecture=model.architecture,
            accuracy=model.accuracy,
            num_parameters=model.num_parameters,
            created_at=model.created_at.isoformat(),
        )
        for model in models
    ]


@router.get("/{model_id}")
async def get_model(
    model_id: int, db: Session = Depends(get_db_session),
) -> dict[str, Any]:
    """获取模型详情"""
    model = db.query(ModelInfo).filter(ModelInfo.id == model_id).first()

    if not model:
        raise HTTPException(status_code=404, detail="模型不存在")

    return {
        "id": model.id,
        "name": model.name,
        "description": model.description,
        "model_type": model.model_type,
        "architecture": model.architecture,
        "config": model.config,
        "metrics": model.metrics,
        "accuracy": model.accuracy,
        "loss": model.loss,
        "num_parameters": model.num_parameters,
        "model_size_mb": model.model_size_mb,
        "checkpoint_path": model.checkpoint_path,
        "trained_epochs": model.trained_epochs,
        "dataset_name": model.dataset_name,
        "num_classes": model.num_classes,
        "tags": model.tags,
        "created_at": model.created_at.isoformat(),
    }


@router.delete("/{model_id}")
async def delete_model(
    model_id: int, db: Session = Depends(get_db_session),
) -> dict[str, str]:
    """删除模型"""
    model = db.query(ModelInfo).filter(ModelInfo.id == model_id).first()

    if not model:
        raise HTTPException(status_code=404, detail="模型不存在")

    db.delete(model)
    db.commit()

    return {"message": "模型已删除"}
