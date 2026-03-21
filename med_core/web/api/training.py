"""训练任务 API"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from datetime import datetime
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel
from sqlalchemy.orm import Session

from ..database import SessionLocal, get_db_session
from ..models import TrainingJob

router = APIRouter()
logger = logging.getLogger(__name__)

_training_tasks: dict[str, asyncio.Task[None]] = {}
_pause_flags: dict[str, bool] = {}
_stop_flags: dict[str, bool] = {}


class TrainingConfig(BaseModel):
    """训练配置"""

    experiment_name: str
    training_model_config: dict[str, Any]
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


def _get_job_or_404(db: Session, job_id: str) -> TrainingJob:
    job = db.query(TrainingJob).filter(TrainingJob.job_id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="训练任务不存在")
    return job


async def _simulate_training(job_id: str, total_epochs: int) -> None:
    """轻量训练模拟器：用于 Web UI 最小可用闭环。"""
    db = SessionLocal()
    try:
        for epoch in range(1, total_epochs + 1):
            while _pause_flags.get(job_id, False):
                await asyncio.sleep(0.5)

            if _stop_flags.get(job_id, False):
                job = _get_job_or_404(db, job_id)
                job.status = "stopped"
                job.completed_at = datetime.utcnow()
                db.commit()
                return

            await asyncio.sleep(1.0)
            job = _get_job_or_404(db, job_id)

            progress = round(epoch / total_epochs * 100, 2)
            current_loss = max(0.01, 1.0 - (epoch / total_epochs) * 0.9)
            current_accuracy = min(0.99, 0.5 + (epoch / total_epochs) * 0.45)

            job.status = "running"
            job.current_epoch = epoch
            job.progress = progress
            job.current_loss = current_loss
            job.current_accuracy = current_accuracy
            job.best_loss = (
                current_loss
                if job.best_loss is None
                else min(job.best_loss, current_loss)
            )
            job.best_accuracy = (
                current_accuracy
                if job.best_accuracy is None
                else max(job.best_accuracy, current_accuracy)
            )
            if job.best_accuracy == current_accuracy:
                job.best_epoch = epoch

            if epoch >= total_epochs:
                job.status = "completed"
                job.completed_at = datetime.utcnow()
                job.progress = 100.0

            db.commit()
    except Exception as e:
        logger.error(f"训练任务模拟失败: {job_id}, 错误: {e}")
        job = db.query(TrainingJob).filter(TrainingJob.job_id == job_id).first()
        if job:
            job.status = "failed"
            job.error_message = str(e)
            job.completed_at = datetime.utcnow()
            db.commit()
    finally:
        db.close()
        _training_tasks.pop(job_id, None)
        _pause_flags.pop(job_id, None)
        _stop_flags.pop(job_id, None)


@router.post("/start")
async def start_training(
    config: TrainingConfig,
    db: Session = Depends(get_db_session),
) -> dict[str, Any]:
    """开始训练任务"""
    try:
        job_id = str(uuid.uuid4())
        total_epochs = int(config.training_config.get("epochs", 20))

        job = TrainingJob(
            job_id=job_id,
            config=config.model_dump(),
            total_epochs=total_epochs,
            status="running",
            progress=0.0,
            current_epoch=0,
            created_at=datetime.utcnow(),
            started_at=datetime.utcnow(),
            current_loss=1.0,
            current_accuracy=0.5,
        )
        db.add(job)
        db.commit()
        db.refresh(job)

        _pause_flags[job_id] = False
        _stop_flags[job_id] = False
        _training_tasks[job_id] = asyncio.create_task(_simulate_training(job_id, total_epochs))

        logger.info(f"训练任务已创建: {job_id}")
        return {"job_id": job_id, "status": "running", "message": "训练任务已启动"}
    except Exception as e:
        logger.error(f"创建训练任务失败: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/jobs")
async def list_training_jobs(
    skip: int = 0,
    limit: int = 20,
    db: Session = Depends(get_db_session),
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
    job_id: str,
    db: Session = Depends(get_db_session),
) -> dict[str, Any]:
    """获取训练任务状态"""
    job = _get_job_or_404(db, job_id)
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
        "error_message": job.error_message,
    }


@router.post("/{job_id}/pause")
async def pause_training(
    job_id: str,
    db: Session = Depends(get_db_session),
) -> dict[str, str]:
    """暂停训练任务"""
    job = _get_job_or_404(db, job_id)
    if job.status != "running":
        raise HTTPException(status_code=400, detail="只能暂停正在运行的任务")

    _pause_flags[job_id] = True
    job.status = "paused"
    db.commit()
    return {"message": "训练任务已暂停"}


@router.post("/{job_id}/resume")
async def resume_training(
    job_id: str,
    db: Session = Depends(get_db_session),
) -> dict[str, str]:
    """恢复训练任务"""
    job = _get_job_or_404(db, job_id)
    if job.status != "paused":
        raise HTTPException(status_code=400, detail="只能恢复已暂停的任务")

    _pause_flags[job_id] = False
    job.status = "running"
    db.commit()
    return {"message": "训练任务已恢复"}


@router.post("/{job_id}/stop")
async def stop_training(
    job_id: str,
    db: Session = Depends(get_db_session),
) -> dict[str, str]:
    """停止训练任务"""
    job = _get_job_or_404(db, job_id)
    if job.status not in {"running", "paused", "queued"}:
        raise HTTPException(status_code=400, detail="无法停止该任务")

    _stop_flags[job_id] = True
    _pause_flags[job_id] = False
    job.status = "stopped"
    job.completed_at = datetime.utcnow()
    db.commit()
    return {"message": "训练任务已停止"}


@router.websocket("/ws/{job_id}")
async def training_websocket(websocket: WebSocket, job_id: str) -> None:
    """训练任务 WebSocket 连接"""
    await websocket.accept()
    logger.info(f"WebSocket 连接已建立: {job_id}")

    db = SessionLocal()
    try:
        while True:
            # 接收客户端控制消息（非阻塞）
            try:
                text = await asyncio.wait_for(websocket.receive_text(), timeout=1.0)
                payload = json.loads(text)
                action = payload.get("action")
                if action == "pause":
                    _pause_flags[job_id] = True
                elif action == "resume":
                    _pause_flags[job_id] = False
                elif action == "stop":
                    _stop_flags[job_id] = True
            except asyncio.TimeoutError:
                pass
            except json.JSONDecodeError:
                pass

            if job_id == "all":
                await websocket.send_json({"type": "heartbeat", "message": "ok"})
                continue

            job = db.query(TrainingJob).filter(TrainingJob.job_id == job_id).first()
            if not job:
                await websocket.send_json(
                    {
                        "type": "error",
                        "message": "训练任务不存在",
                        "job_id": job_id,
                    },
                )
                await asyncio.sleep(1.0)
                continue

            await websocket.send_json(
                {
                    "type": "status_update",
                    "job_id": job.job_id,
                    "status": job.status,
                    "progress": job.progress,
                    "epoch": job.current_epoch,
                    "loss": job.current_loss,
                    "accuracy": job.current_accuracy,
                },
            )

            if job.status in {"completed", "failed", "stopped"}:
                await websocket.send_json(
                    {
                        "type": "training_complete"
                        if job.status == "completed"
                        else "error",
                        "job_id": job.job_id,
                        "message": job.error_message or job.status,
                    },
                )
                break
    except WebSocketDisconnect:
        logger.info(f"WebSocket 连接已断开: {job_id}")
    finally:
        db.close()
