"""训练任务 API"""

import logging
import uuid
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel
from sqlalchemy.orm import Session

from ..database import get_db_session
from ..models import TrainingJob
from ..services.training_service import TrainingService

router = APIRouter()
logger = logging.getLogger(__name__)


# Pydantic 模型
class TrainingConfig(BaseModel):
    """训练配置"""

    experiment_name: str
    training_model_config: dict[str, Any]  # 重命名避免冲突
    dataset_config: dict[str, Any]
    training_config: dict[str, Any]


class TrainingJobResponse(BaseModel):
    """训练任务响应"""

    id: int
    job_id: str
    status: str
    progress: float
    current_epoch: int
    total_epochs: int
    current_loss: float | None
    current_accuracy: float | None
    created_at: str

    class Config:
        from_attributes = True


@router.post("/start")
async def start_training(
    config: TrainingConfig, db: Session = Depends(get_db_session),
) -> dict[str, Any]:
    """开始训练任务"""
    try:
        # 生成任务 ID
        job_id = str(uuid.uuid4())

        # 创建训练任务记录
        job = TrainingJob(
            job_id=job_id,
            config=config.model_dump(),  # 使用 model_dump 替代 dict
            total_epochs=config.training_config.get("epochs", 100),
            status="queued",
        )
        db.add(job)
        db.commit()
        db.refresh(job)

        # 提交训练任务到后台
        training_service = TrainingService()
        training_service.submit_job(job_id, config.model_dump())  # 使用 model_dump

        logger.info(f"训练任务已创建: {job_id}")

        return {"job_id": job_id, "status": "queued", "message": "训练任务已提交"}
    except Exception as e:
        logger.error(f"创建训练任务失败: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/jobs")
async def list_training_jobs(
    skip: int = 0, limit: int = 20, db: Session = Depends(get_db_session),
) -> list[TrainingJobResponse]:
    """获取训练任务列表"""
    jobs = (
        db.query(TrainingJob)
        .order_by(TrainingJob.created_at.desc())
        .offset(skip)
        .limit(limit)
        .all()
    )

    return [
        TrainingJobResponse(
            id=job.id,
            job_id=job.job_id,
            status=job.status,
            progress=job.progress,
            current_epoch=job.current_epoch,
            total_epochs=job.total_epochs,
            current_loss=job.current_loss,
            current_accuracy=job.current_accuracy,
            created_at=job.created_at.isoformat(),
        )
        for job in jobs
    ]


@router.get("/{job_id}/status")
async def get_training_status(
    job_id: str, db: Session = Depends(get_db_session),
) -> dict[str, Any]:
    """获取训练任务状态"""
    job = db.query(TrainingJob).filter(TrainingJob.job_id == job_id).first()

    if not job:
        raise HTTPException(status_code=404, detail="训练任务不存在")

    return {
        "job_id": job.job_id,
        "status": job.status,
        "progress": job.progress,
        "current_epoch": job.current_epoch,
        "total_epochs": job.total_epochs,
        "current_loss": job.current_loss,
        "current_accuracy": job.current_accuracy,
        "best_loss": job.best_loss,
        "best_accuracy": job.best_accuracy,
        "gpu_usage": job.gpu_usage,
        "gpu_memory": job.gpu_memory,
    }


@router.post("/{job_id}/pause")
async def pause_training(
    job_id: str, db: Session = Depends(get_db_session),
) -> dict[str, str]:
    """暂停训练任务"""
    job = db.query(TrainingJob).filter(TrainingJob.job_id == job_id).first()

    if not job:
        raise HTTPException(status_code=404, detail="训练任务不存在")

    if job.status != "running":
        raise HTTPException(status_code=400, detail="只能暂停正在运行的任务")

    # TODO: 实现暂停逻辑
    job.status = "paused"
    db.commit()

    return {"message": "训练任务已暂停"}


@router.post("/{job_id}/resume")
async def resume_training(
    job_id: str, db: Session = Depends(get_db_session),
) -> dict[str, str]:
    """恢复训练任务"""
    job = db.query(TrainingJob).filter(TrainingJob.job_id == job_id).first()

    if not job:
        raise HTTPException(status_code=404, detail="训练任务不存在")

    if job.status != "paused":
        raise HTTPException(status_code=400, detail="只能恢复已暂停的任务")

    # TODO: 实现恢复逻辑
    job.status = "running"
    db.commit()

    return {"message": "训练任务已恢复"}


@router.post("/{job_id}/stop")
async def stop_training(
    job_id: str, db: Session = Depends(get_db_session),
) -> dict[str, str]:
    """停止训练任务"""
    job = db.query(TrainingJob).filter(TrainingJob.job_id == job_id).first()

    if not job:
        raise HTTPException(status_code=404, detail="训练任务不存在")

    if job.status not in ["running", "paused", "queued"]:
        raise HTTPException(status_code=400, detail="无法停止该任务")

    # TODO: 实现停止逻辑
    job.status = "stopped"
    db.commit()

    return {"message": "训练任务已停止"}


@router.websocket("/ws/{job_id}")
async def training_websocket(websocket: WebSocket, job_id: str) -> None:
    """训练任务 WebSocket 连接"""
    await websocket.accept()
    logger.info(f"WebSocket 连接已建立: {job_id}")

    try:
        # TODO: 实现实时推送训练指标
        while True:
            # 等待客户端消息
            data = await websocket.receive_text()
            logger.debug(f"收到消息: {data}")

            # 发送测试消息
            await websocket.send_json({"type": "ping", "message": "pong"})
    except WebSocketDisconnect:
        logger.info(f"WebSocket 连接已断开: {job_id}")
