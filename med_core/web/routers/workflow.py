"""
工作流 API 路由

提供工作流的验证、执行、状态查询等功能。
"""

import asyncio
import logging
import uuid
from typing import Any, Dict, Optional

from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field

from ..config import settings
from ..workflow_engine import (
    NodeStatus,
    WorkflowEngine,
    WorkflowExecutionError,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/workflows", tags=["workflows"])

# 存储正在执行的工作流
active_workflows: Dict[str, WorkflowEngine] = {}


class WorkflowData(BaseModel):
    """工作流数据模型"""

    nodes: list[Dict[str, Any]] = Field(..., description="节点列表")
    edges: list[Dict[str, Any]] = Field(..., description="边列表")


class WorkflowValidateRequest(BaseModel):
    """工作流验证请求"""

    workflow: WorkflowData


class WorkflowValidateResponse(BaseModel):
    """工作流验证响应"""

    valid: bool
    errors: list[str] = []


class WorkflowExecuteRequest(BaseModel):
    """工作流执行请求"""

    workflow: WorkflowData
    name: Optional[str] = None


class WorkflowExecuteResponse(BaseModel):
    """工作流执行响应"""

    workflow_id: str
    status: str
    message: str


class WorkflowStatusResponse(BaseModel):
    """工作流状态响应"""

    workflow_id: str
    status: Dict[str, Any]
    results: Optional[Dict[str, Any]] = None


@router.post("/validate", response_model=WorkflowValidateResponse)
async def validate_workflow(request: WorkflowValidateRequest):
    """
    验证工作流合法性

    检查：
    - 节点和边的完整性
    - 循环依赖
    - 端口类型匹配
    - 必需输入
    """
    try:
        engine = WorkflowEngine(data_dir=settings.data_dir)
        engine.load_workflow(request.workflow.model_dump())

        is_valid, errors = engine.validate()

        return WorkflowValidateResponse(valid=is_valid, errors=errors)

    except Exception as e:
        logger.error(f"Workflow validation error: {e}")
        return WorkflowValidateResponse(valid=False, errors=[str(e)])


@router.post("/execute", response_model=WorkflowExecuteResponse)
async def execute_workflow(request: WorkflowExecuteRequest):
    """
    执行工作流

    创建一个新的工作流执行实例，并在后台异步执行。
    返回 workflow_id 用于查询执行状态。
    """
    try:
        # 生成唯一 ID
        workflow_id = str(uuid.uuid4())

        # 创建引擎
        engine = WorkflowEngine(data_dir=settings.data_dir)
        engine.load_workflow(request.workflow.model_dump())

        # 验证工作流
        is_valid, errors = engine.validate()
        if not is_valid:
            raise HTTPException(
                status_code=400, detail=f"工作流验证失败: {', '.join(errors)}"
            )

        # 保存到活动工作流
        active_workflows[workflow_id] = engine

        # 在后台执行
        asyncio.create_task(_execute_workflow_background(workflow_id, engine))

        return WorkflowExecuteResponse(
            workflow_id=workflow_id,
            status="started",
            message=f"工作流 {request.name or workflow_id} 已开始执行",
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to start workflow execution: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


async def _execute_workflow_background(workflow_id: str, engine: WorkflowEngine):
    """后台执行工作流"""
    try:
        logger.info(f"Starting background execution for workflow {workflow_id}")
        await engine.execute(workflow_id=workflow_id)
        logger.info(f"Workflow {workflow_id} completed successfully")
    except Exception as e:
        logger.error(f"Workflow {workflow_id} execution failed: {e}")


@router.get("/{workflow_id}/status", response_model=WorkflowStatusResponse)
async def get_workflow_status(workflow_id: str):
    """
    查询工作流执行状态

    返回：
    - 各节点的执行状态
    - 完成进度
    - 执行结果（如果已完成）
    """
    if workflow_id not in active_workflows:
        raise HTTPException(status_code=404, detail=f"工作流 {workflow_id} 不存在")

    engine = active_workflows[workflow_id]
    status = engine.get_status()

    # 收集结果
    results = None
    all_completed = all(
        node.status == NodeStatus.COMPLETED for node in engine.nodes.values()
    )
    if all_completed:
        results = {node_id: node.result for node_id, node in engine.nodes.items()}

    return WorkflowStatusResponse(
        workflow_id=workflow_id, status=status, results=results
    )


@router.delete("/{workflow_id}")
async def delete_workflow(workflow_id: str):
    """
    删除工作流执行记录

    清理已完成或失败的工作流。
    """
    if workflow_id not in active_workflows:
        raise HTTPException(status_code=404, detail=f"工作流 {workflow_id} 不存在")

    del active_workflows[workflow_id]
    return {"message": f"工作流 {workflow_id} 已删除"}


@router.websocket("/{workflow_id}/progress")
async def workflow_progress_websocket(websocket: WebSocket, workflow_id: str):
    """
    工作流执行进度 WebSocket

    实时推送节点执行状态和进度。
    """
    await websocket.accept()

    if workflow_id not in active_workflows:
        await websocket.send_json({"error": f"工作流 {workflow_id} 不存在"})
        await websocket.close()
        return

    engine = active_workflows[workflow_id]

    try:
        # 定义进度回调
        async def progress_callback(node_id: str, status: NodeStatus, progress: float):
            await websocket.send_json(
                {
                    "node_id": node_id,
                    "status": status.value,
                    "progress": progress,
                    "timestamp": asyncio.get_event_loop().time(),
                }
            )

        # 发送初始状态
        await websocket.send_json(
            {"type": "status", "data": engine.get_status(), "message": "连接成功"}
        )

        # 执行工作流（如果还未执行）
        if all(node.status == NodeStatus.PENDING for node in engine.nodes.values()):
            try:
                results = await engine.execute(progress_callback=progress_callback)
                await websocket.send_json(
                    {"type": "completed", "results": results, "message": "执行完成"}
                )
            except WorkflowExecutionError as e:
                await websocket.send_json(
                    {"type": "error", "error": str(e), "message": "执行失败"}
                )
        else:
            # 工作流已在执行，只发送当前状态
            while True:
                status = engine.get_status()
                await websocket.send_json({"type": "status", "data": status})

                # 检查是否完成
                if status["completed"] + status["failed"] == status["total"]:
                    break

                await asyncio.sleep(1)

    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected for workflow {workflow_id}")
    except Exception as e:
        logger.error(f"WebSocket error for workflow {workflow_id}: {e}")
        try:
            await websocket.send_json({"type": "error", "error": str(e)})
        except Exception:
            pass
    finally:
        try:
            await websocket.close()
        except Exception:
            pass


@router.get("/")
async def list_workflows():
    """
    列出所有活动的工作流

    返回当前正在执行或已完成的工作流列表。
    """
    workflows = []
    for workflow_id, engine in active_workflows.items():
        status = engine.get_status()
        workflows.append(
            {
                "workflow_id": workflow_id,
                "total_nodes": status["total"],
                "completed": status["completed"],
                "failed": status["failed"],
                "progress": status["completed"] / status["total"]
                if status["total"] > 0
                else 0,
            }
        )

    return {"workflows": workflows, "total": len(workflows)}


@router.get("/{workflow_id}/resources")
async def get_workflow_resources(workflow_id: str, duration: Optional[int] = None):
    """
    获取工作流执行期间的资源使用统计

    Args:
        workflow_id: 工作流 ID
        duration: 可选的时间范围（秒）

    Returns:
        资源统计信息
    """
    if workflow_id not in active_workflows:
        raise HTTPException(status_code=404, detail=f"工作流 {workflow_id} 不存在")

    engine = active_workflows[workflow_id]

    # 获取当前状态
    current_status = (
        engine.resource_monitor.get_current_status() if engine.resource_monitor else {}
    )

    # 获取统计信息
    statistics = engine.get_resource_statistics(duration)

    # 获取历史记录
    history = (
        engine.resource_monitor.get_history(duration) if engine.resource_monitor else []
    )

    return {
        "workflow_id": workflow_id,
        "current": current_status,
        "statistics": statistics,
        "history": history,
    }


@router.get("/{workflow_id}/checkpoints")
async def list_workflow_checkpoints(workflow_id: str):
    """
    列出工作流的所有检查点

    Args:
        workflow_id: 工作流 ID

    Returns:
        检查点列表
    """
    if workflow_id not in active_workflows:
        raise HTTPException(status_code=404, detail=f"工作流 {workflow_id} 不存在")

    engine = active_workflows[workflow_id]

    if not engine.checkpoint_manager:
        raise HTTPException(status_code=400, detail="检查点功能未启用")

    checkpoints = engine.checkpoint_manager.list_checkpoints(workflow_id)

    return {
        "workflow_id": workflow_id,
        "checkpoints": checkpoints,
        "total": len(checkpoints),
    }


@router.post("/{workflow_id}/resume")
async def resume_workflow(workflow_id: str, checkpoint_path: Optional[str] = None):
    """
    从检查点恢复工作流执行

    Args:
        workflow_id: 工作流 ID
        checkpoint_path: 可选的检查点路径，如果不提供则使用最新的

    Returns:
        恢复结果
    """
    # 检查工作流是否已存在
    if workflow_id in active_workflows:
        raise HTTPException(status_code=400, detail=f"工作流 {workflow_id} 已在运行中")

    try:
        # 创建新引擎
        engine = WorkflowEngine(data_dir=settings.data_dir)

        if not engine.checkpoint_manager:
            raise HTTPException(status_code=400, detail="检查点功能未启用")

        # 恢复工作流
        from pathlib import Path

        cp_path = Path(checkpoint_path) if checkpoint_path else None
        restored_data = engine.checkpoint_manager.resume_workflow(workflow_id, cp_path)

        # 加载工作流
        engine.load_workflow(restored_data["workflow"])

        # 保存到活动工作流
        active_workflows[workflow_id] = engine

        # 在后台继续执行
        asyncio.create_task(_execute_workflow_background(workflow_id, engine))

        return {
            "workflow_id": workflow_id,
            "status": "resumed",
            "message": f"工作流 {workflow_id} 已从检查点恢复并继续执行",
            "checkpoint": str(cp_path) if cp_path else "latest",
        }

    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to resume workflow: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e
