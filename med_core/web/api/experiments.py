"""实验管理 API"""

from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from ..database import get_db_session
from ..models import Experiment

router = APIRouter()


class ExperimentResponse(BaseModel):
    """实验响应"""

    id: int
    name: str
    description: str | None
    status: str
    created_at: str

    class Config:
        from_attributes = True


@router.get("")
async def list_experiments(
    skip: int = 0, limit: int = 20, db: Session = Depends(get_db_session)
) -> list[ExperimentResponse]:
    """获取实验列表"""
    experiments = (
        db.query(Experiment)
        .order_by(Experiment.created_at.desc())
        .offset(skip)
        .limit(limit)
        .all()
    )

    return [
        ExperimentResponse(
            id=exp.id,
            name=exp.name,
            description=exp.description,
            status=exp.status,
            created_at=exp.created_at.isoformat(),
        )
        for exp in experiments
    ]


@router.get("/{experiment_id}")
async def get_experiment(
    experiment_id: int, db: Session = Depends(get_db_session)
) -> dict[str, Any]:
    """获取实验详情"""
    experiment = db.query(Experiment).filter(Experiment.id == experiment_id).first()

    if not experiment:
        raise HTTPException(status_code=404, detail="实验不存在")

    return {
        "id": experiment.id,
        "name": experiment.name,
        "description": experiment.description,
        "config": experiment.config,
        "status": experiment.status,
        "metrics": experiment.metrics,
        "output_dir": experiment.output_dir,
        "created_at": experiment.created_at.isoformat(),
    }


@router.delete("/{experiment_id}")
async def delete_experiment(
    experiment_id: int, db: Session = Depends(get_db_session)
) -> dict[str, str]:
    """删除实验"""
    experiment = db.query(Experiment).filter(Experiment.id == experiment_id).first()

    if not experiment:
        raise HTTPException(status_code=404, detail="实验不存在")

    db.delete(experiment)
    db.commit()

    return {"message": "实验已删除"}
