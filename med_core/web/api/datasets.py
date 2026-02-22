"""数据集管理 API"""

from typing import Any

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from ..database import get_db_session

router = APIRouter()


@router.get("")
async def list_datasets(
    skip: int = 0, limit: int = 20, db: Session = Depends(get_db_session)
) -> list[dict[str, Any]]:
    """获取数据集列表"""
    # TODO: 实现数据集列表
    return []


@router.get("/{dataset_id}")
async def get_dataset(
    dataset_id: int, db: Session = Depends(get_db_session)
) -> dict[str, Any]:
    """获取数据集详情"""
    # TODO: 实现数据集详情
    return {}
