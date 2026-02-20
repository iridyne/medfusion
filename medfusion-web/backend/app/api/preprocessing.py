"""预处理 API 端点"""

import asyncio
import logging
import uuid
from datetime import datetime, timezone
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.crud.preprocessing import PreprocessingTaskCRUD
from app.services.preprocessing_service import preprocessing_service

logger = logging.getLogger(__name__)

router = APIRouter()


# ==================== Pydantic 模型 ====================


class PreprocessingConfig(BaseModel):
    """预处理配置"""

    size: int = Field(default=224, ge=32, le=1024, description="目标图像大小")
    normalize: str = Field(
        default="percentile",
        description="归一化方法",
        pattern="^(minmax|zscore|percentile|none)$",
    )
    remove_artifacts: bool = Field(default=False, description="是否去除伪影")
    enhance_contrast: bool = Field(default=False, description="是否增强对比度")


class PreprocessingTaskCreate(BaseModel):
    """创建预处理任务请求"""

    name: str = Field(..., min_length=1, max_length=255, description="任务名称")
    description: Optional[str] = Field(None, description="任务描述")
    input_dir: str = Field(..., min_length=1, description="输入目录路径")
    output_dir: str = Field(..., min_length=1, description="输出目录路径")
    config: PreprocessingConfig = Field(..., description="预处理配置")


class PreprocessingTaskResponse(BaseModel):
    """预处理任务响应"""

    id: int
    task_id: str
    name: str
    description: Optional[str]
    input_dir: str
    output_dir: str
    config: dict
    status: str
    progress: float
    total_images: int
    processed_images: int
    failed_images: int
    error: Optional[str]
    created_at: datetime
    started_at: Optional[datetime]
    completed_at: Optional[datetime]
    duration: Optional[float]

    class Config:
        from_attributes = True


class PreprocessingTaskListResponse(BaseModel):
    """预处理任务列表响应"""

    tasks: List[PreprocessingTaskResponse]
    total: int
    skip: int
    limit: int


class PreprocessingStatistics(BaseModel):
    """预处理统计信息"""

    total_tasks: int
    status_counts: dict
    total_processed_images: int
    total_failed_images: int


# ==================== API 端点 ====================


@router.post("/start", response_model=PreprocessingTaskResponse)
async def start_preprocessing(
    task_data: PreprocessingTaskCreate,
    db: Session = Depends(get_db),
):
    """启动预处理任务

    创建预处理任务并异步执行
    """
    try:
        # 生成任务 ID
        task_id = f"preprocess_{uuid.uuid4().hex[:8]}"

        # 创建数据库记录
        task = PreprocessingTaskCRUD.create(
            db=db,
            task_id=task_id,
            name=task_data.name,
            description=task_data.description,
            input_dir=task_data.input_dir,
            output_dir=task_data.output_dir,
            config=task_data.config.model_dump(),
        )

        logger.info(f"Created preprocessing task: {task_id}")

        # 定义进度回调
        async def progress_callback(data: dict):
            """更新数据库中的任务状态"""
            try:
                if data["type"] == "started":
                    PreprocessingTaskCRUD.update(
                        db,
                        task.id,
                        status="running",
                        started_at=datetime.now(timezone.utc),
                        total_images=data["total_images"],
                    )
                elif data["type"] == "progress":
                    PreprocessingTaskCRUD.update_progress(
                        db,
                        task_id,
                        progress=data["progress"],
                        processed_images=data["processed_images"],
                        failed_images=data["failed_images"],
                    )
                elif data["type"] == "completed":
                    result = data["result"]
                    PreprocessingTaskCRUD.update(
                        db,
                        task.id,
                        status="completed",
                        progress=1.0,
                        completed_at=datetime.now(timezone.utc),
                        duration=result["duration"],
                        result=result,
                    )
                elif data["type"] == "cancelled":
                    PreprocessingTaskCRUD.update(
                        db,
                        task.id,
                        status="cancelled",
                        completed_at=datetime.now(timezone.utc),
                    )
            except Exception as e:
                logger.error(f"Error in progress callback: {e}")

        # 启动异步任务
        async_task = asyncio.create_task(
            preprocessing_service.start_preprocessing(
                task_id=task_id,
                input_dir=task_data.input_dir,
                output_dir=task_data.output_dir,
                config=task_data.config.model_dump(),
                progress_callback=progress_callback,
            )
        )

        # 注册任务
        preprocessing_service.register_task(task_id, async_task)

        # 添加完成回调
        def task_done_callback(future):
            try:
                if future.exception():
                    error = str(future.exception())
                    logger.error(f"Task {task_id} failed: {error}")
                    PreprocessingTaskCRUD.update(
                        db,
                        task.id,
                        status="failed",
                        error=error,
                        completed_at=datetime.now(timezone.utc),
                    )
            except Exception as e:
                logger.error(f"Error in task done callback: {e}")
            finally:
                preprocessing_service.cleanup_task(task_id)

        async_task.add_done_callback(task_done_callback)

        return task

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to start preprocessing: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to start preprocessing: {e}"
        )


@router.get("/status/{task_id}", response_model=PreprocessingTaskResponse)
async def get_preprocessing_status(
    task_id: str,
    db: Session = Depends(get_db),
):
    """获取预处理任务状态"""
    task = PreprocessingTaskCRUD.get_by_task_id(db, task_id)
    if not task:
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")
    return task


@router.get("/list", response_model=PreprocessingTaskListResponse)
async def list_preprocessing_tasks(
    skip: int = 0,
    limit: int = 100,
    status: Optional[str] = None,
    sort_by: str = "created_at",
    order: str = "desc",
    db: Session = Depends(get_db),
):
    """列出预处理任务"""
    tasks = PreprocessingTaskCRUD.list(
        db=db,
        skip=skip,
        limit=limit,
        status=status,
        sort_by=sort_by,
        order=order,
    )
    total = PreprocessingTaskCRUD.count(db, status=status)

    return PreprocessingTaskListResponse(
        tasks=tasks,
        total=total,
        skip=skip,
        limit=limit,
    )


@router.post("/cancel/{task_id}")
async def cancel_preprocessing(
    task_id: str,
    db: Session = Depends(get_db),
):
    """取消预处理任务"""
    # 检查任务是否存在
    task = PreprocessingTaskCRUD.get_by_task_id(db, task_id)
    if not task:
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")

    # 检查任务状态
    if task.status not in ["pending", "running"]:
        raise HTTPException(
            status_code=400,
            detail=f"Cannot cancel task in status: {task.status}",
        )

    # 取消任务
    success = preprocessing_service.cancel_task(task_id)
    if not success:
        raise HTTPException(status_code=400, detail="Failed to cancel task")

    return {"message": f"Task {task_id} cancellation requested"}


@router.delete("/{task_id}")
async def delete_preprocessing_task(
    task_id: int,
    db: Session = Depends(get_db),
):
    """删除预处理任务"""
    success = PreprocessingTaskCRUD.delete(db, task_id)
    if not success:
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")

    return {"message": f"Task {task_id} deleted successfully"}


@router.get("/statistics", response_model=PreprocessingStatistics)
async def get_preprocessing_statistics(
    db: Session = Depends(get_db),
):
    """获取预处理统计信息"""
    stats = PreprocessingTaskCRUD.get_statistics(db)
    return PreprocessingStatistics(**stats)


@router.websocket("/ws/{task_id}")
async def preprocessing_websocket(
    websocket: WebSocket,
    task_id: str,
    db: Session = Depends(get_db),
):
    """WebSocket 实时监控预处理任务

    客户端连接后会持续接收任务状态更新
    """
    await websocket.accept()
    logger.info(f"WebSocket connected for task {task_id}")

    try:
        # 检查任务是否存在
        task = PreprocessingTaskCRUD.get_by_task_id(db, task_id)
        if not task:
            await websocket.send_json(
                {
                    "type": "error",
                    "message": f"Task {task_id} not found",
                }
            )
            await websocket.close()
            return

        # 发送初始状态
        await websocket.send_json(
            {
                "type": "status",
                "task_id": task_id,
                "status": task.status,
                "progress": task.progress,
                "processed_images": task.processed_images,
                "failed_images": task.failed_images,
                "total_images": task.total_images,
            }
        )

        # 持续发送状态更新
        while True:
            # 从数据库获取最新状态
            db.refresh(task)

            # 发送状态更新
            await websocket.send_json(
                {
                    "type": "status",
                    "task_id": task_id,
                    "status": task.status,
                    "progress": task.progress,
                    "processed_images": task.processed_images,
                    "failed_images": task.failed_images,
                    "total_images": task.total_images,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
            )

            # 如果任务已完成，发送最终状态并关闭连接
            if task.status in ["completed", "failed", "cancelled"]:
                await websocket.send_json(
                    {
                        "type": "finished",
                        "task_id": task_id,
                        "status": task.status,
                        "result": task.result,
                        "error": task.error,
                    }
                )
                break

            # 等待一段时间再更新
            await asyncio.sleep(1)

    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected for task {task_id}")
    except Exception as e:
        logger.error(f"WebSocket error for task {task_id}: {e}")
        try:
            await websocket.send_json(
                {
                    "type": "error",
                    "message": str(e),
                }
            )
        except:
            pass
    finally:
        try:
            await websocket.close()
        except:
            pass
