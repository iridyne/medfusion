"""训练 API"""
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, Optional
from sqlalchemy.orm import Session
import asyncio
import json

from app.services.training_service import TrainingService
from app.core.database import get_db
from app.crud import TrainingJobCRUD

router = APIRouter()


class TrainingConfig(BaseModel):
    """训练配置"""
    name: Optional[str] = None
    description: Optional[str] = None
    model_config: Dict[str, Any]
    data_config: Dict[str, Any]
    training_config: Dict[str, Any]


# 全局训练任务字典（内存中的运行时服务）
training_jobs: Dict[str, TrainingService] = {}


@router.post("/start")
async def start_training(config: TrainingConfig, db: Session = Depends(get_db)):
    """开始训练"""
    job_id = f"job_{len(training_jobs) + 1:04d}"
    
    # 保存到数据库
    db_job = TrainingJobCRUD.create(
        db=db,
        job_id=job_id,
        name=config.name,
        description=config.description,
        model_config=config.model_config,
        data_config=config.data_config,
        training_config=config.training_config,
    )
    
    # 创建训练服务
    service = TrainingService(job_id, config.dict())
    training_jobs[job_id] = service
    
    # 更新状态为 initializing
    TrainingJobCRUD.update_status(db, job_id, "initializing")
    
    # 在后台运行训练，带数据库更新回调
    async def run_with_db_updates():
        try:
            # 更新状态为 running
            TrainingJobCRUD.update_status(db, job_id, "running")
            
            # 运行训练
            await service.run(progress_callback=lambda data: update_db_progress(db, job_id, data))
            
            # 更新状态为 completed
            TrainingJobCRUD.update_status(db, job_id, "completed")
        except Exception as e:
            # 更新状态为 failed
            TrainingJobCRUD.update_status(db, job_id, "failed", error=str(e))
    
    asyncio.create_task(run_with_db_updates())
    
    return {
        "job_id": job_id,
        "status": "started",
        "database_id": db_job.id,
    }


async def update_db_progress(db: Session, job_id: str, data: Dict[str, Any]):
    """更新数据库中的训练进度"""
    if data.get("type") == "epoch_completed":
        TrainingJobCRUD.update_progress(
            db=db,
            job_id=job_id,
            progress=data.get("progress", 0.0),
            current_epoch=data.get("epoch", 0),
            current_metrics=data.get("metrics"),
        )


@router.get("/status/{job_id}")
async def get_training_status(job_id: str, db: Session = Depends(get_db)):
    """获取训练状态"""
    # 先从数据库获取
    db_job = TrainingJobCRUD.get(db, job_id)
    
    if not db_job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    # 如果任务在运行中，从内存获取最新状态
    if job_id in training_jobs:
        service = training_jobs[job_id]
        runtime_status = service.get_status()
        
        return {
            **runtime_status,
            "database_id": db_job.id,
            "created_at": db_job.created_at.isoformat(),
        }
    
    # 否则返回数据库中的状态
    return {
        "job_id": db_job.job_id,
        "status": db_job.status,
        "progress": db_job.progress,
        "current_epoch": db_job.current_epoch,
        "total_epochs": db_job.total_epochs,
        "current_metrics": db_job.current_metrics,
        "error": db_job.error,
        "created_at": db_job.created_at.isoformat(),
        "started_at": db_job.started_at.isoformat() if db_job.started_at else None,
        "completed_at": db_job.completed_at.isoformat() if db_job.completed_at else None,
    }


@router.get("/list")
async def list_training_jobs(
    skip: int = 0,
    limit: int = 100,
    status: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """列出所有训练任务"""
    jobs = TrainingJobCRUD.list(db, skip=skip, limit=limit, status=status)
    
    return {
        "jobs": [
            {
                "id": job.id,
                "job_id": job.job_id,
                "name": job.name,
                "status": job.status,
                "progress": job.progress,
                "current_epoch": job.current_epoch,
                "total_epochs": job.total_epochs,
                "current_metrics": job.current_metrics,
                "created_at": job.created_at.isoformat(),
                "started_at": job.started_at.isoformat() if job.started_at else None,
                "completed_at": job.completed_at.isoformat() if job.completed_at else None,
                "duration": job.duration,
            }
            for job in jobs
        ],
        "total": len(jobs),
    }


@router.post("/stop/{job_id}")
async def stop_training(job_id: str, db: Session = Depends(get_db)):
    """停止训练"""
    if job_id not in training_jobs:
        raise HTTPException(status_code=404, detail="Job not found or not running")
    
    service = training_jobs[job_id]
    service.stop()
    
    # 更新数据库状态
    TrainingJobCRUD.update_status(db, job_id, "stopped")
    
    return {
        "job_id": job_id,
        "status": "stopped",
    }


@router.post("/pause/{job_id}")
async def pause_training(job_id: str, db: Session = Depends(get_db)):
    """暂停训练"""
    if job_id not in training_jobs:
        raise HTTPException(status_code=404, detail="Job not found or not running")
    
    service = training_jobs[job_id]
    service.pause()
    
    # 更新数据库状态
    TrainingJobCRUD.update_status(db, job_id, "paused")
    
    return {
        "job_id": job_id,
        "status": "paused",
    }


@router.post("/resume/{job_id}")
async def resume_training(job_id: str, db: Session = Depends(get_db)):
    """恢复训练"""
    if job_id not in training_jobs:
        raise HTTPException(status_code=404, detail="Job not found or not running")
    
    service = training_jobs[job_id]
    service.resume()
    
    # 更新数据库状态
    TrainingJobCRUD.update_status(db, job_id, "running")
    
    return {
        "job_id": job_id,
        "status": "resumed",
    }


@router.websocket("/ws/{job_id}")
async def training_websocket(websocket: WebSocket, job_id: str):
    """训练 WebSocket 连接，实时推送训练进度"""
    await websocket.accept()
    
    try:
        if job_id in training_jobs:
            service = training_jobs[job_id]
            
            # 定义进度回调
            async def progress_callback(data: Dict[str, Any]):
                try:
                    await websocket.send_json(data)
                except:
                    pass
            
            # 如果任务还未开始，启动它
            if service.status == "pending":
                await service.run(progress_callback=progress_callback)
            else:
                # 发送当前状态
                await websocket.send_json({
                    "type": "training_status",
                    **service.get_status()
                })
                
                # 持续发送状态更新
                while service.status in ["initializing", "running", "paused"]:
                    await asyncio.sleep(1)
                    await websocket.send_json({
                        "type": "status_update",
                        **service.get_status()
                    })
        else:
            await websocket.send_json({
                "type": "error",
                "error": "Job not found"
            })
        
        # 保持连接
        while True:
            try:
                data = await asyncio.wait_for(websocket.receive_text(), timeout=1.0)
                # 处理客户端消息（如暂停/恢复命令）
                message = json.loads(data)
                command = message.get("command")
                
                if command == "pause":
                    service.pause()
                elif command == "resume":
                    service.resume()
                elif command == "stop":
                    service.stop()
                    break
            except asyncio.TimeoutError:
                # 发送心跳
                await websocket.send_json({"type": "heartbeat"})
            
    except WebSocketDisconnect:
        print(f"WebSocket disconnected for job {job_id}")
    except Exception as e:
        print(f"WebSocket error for job {job_id}: {e}")
    finally:
        try:
            await websocket.close()
        except:
            pass

