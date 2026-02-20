"""工作流 API"""
from fastapi import APIRouter, HTTPException, WebSocket, Depends
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from sqlalchemy.orm import Session

from app.core.node_registry import node_registry
from app.core.database import get_db
from app.crud import WorkflowCRUD, WorkflowExecutionCRUD

router = APIRouter()


class Node(BaseModel):
    """节点模型"""
    id: str
    type: str
    position: Dict[str, float]
    data: Dict[str, Any]


class Edge(BaseModel):
    """连接边模型"""
    id: str
    source: str
    target: str
    sourceHandle: Optional[str] = None
    targetHandle: Optional[str] = None


class Workflow(BaseModel):
    """工作流模型"""
    id: Optional[str] = None
    name: str
    description: Optional[str] = ""
    nodes: List[Node]
    edges: List[Edge]


class WorkflowExecuteRequest(BaseModel):
    """工作流执行请求"""
    workflow: Workflow


@router.get("/nodes")
async def list_nodes():
    """获取所有可用节点"""
    return {
        "nodes": node_registry.list_nodes()
    }


@router.get("/nodes/category/{category}")
async def get_nodes_by_category(category: str):
    """按类别获取节点"""
    return {
        "nodes": node_registry.get_by_category(category)
    }


@router.post("/")
async def create_workflow(workflow: Workflow, db: Session = Depends(get_db)):
    """创建工作流"""
    # 检查名称是否已存在
    existing = WorkflowCRUD.get_by_name(db, workflow.name)
    if existing:
        raise HTTPException(status_code=400, detail=f"Workflow with name '{workflow.name}' already exists")
    
    # 保存到数据库
    db_workflow = WorkflowCRUD.create(
        db=db,
        name=workflow.name,
        description=workflow.description,
        nodes=[node.dict() for node in workflow.nodes],
        edges=[edge.dict() for edge in workflow.edges],
    )
    
    return {
        "id": db_workflow.id,
        "name": db_workflow.name,
        "description": db_workflow.description,
        "status": "created",
        "created_at": db_workflow.created_at.isoformat(),
    }


@router.get("/")
async def list_workflows(
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db)
):
    """列出所有工作流"""
    workflows = WorkflowCRUD.list(db, skip=skip, limit=limit)
    
    return {
        "workflows": [
            {
                "id": wf.id,
                "name": wf.name,
                "description": wf.description,
                "execution_count": wf.execution_count,
                "last_executed_at": wf.last_executed_at.isoformat() if wf.last_executed_at else None,
                "created_at": wf.created_at.isoformat(),
                "updated_at": wf.updated_at.isoformat(),
            }
            for wf in workflows
        ],
        "total": len(workflows),
    }


@router.get("/{workflow_id}")
async def get_workflow(workflow_id: int, db: Session = Depends(get_db)):
    """获取工作流详情"""
    workflow = WorkflowCRUD.get(db, workflow_id)
    
    if not workflow:
        raise HTTPException(status_code=404, detail="Workflow not found")
    
    return {
        "id": workflow.id,
        "name": workflow.name,
        "description": workflow.description,
        "nodes": workflow.nodes,
        "edges": workflow.edges,
        "execution_count": workflow.execution_count,
        "last_executed_at": workflow.last_executed_at.isoformat() if workflow.last_executed_at else None,
        "created_at": workflow.created_at.isoformat(),
        "updated_at": workflow.updated_at.isoformat(),
    }


@router.put("/{workflow_id}")
async def update_workflow(
    workflow_id: int,
    workflow: Workflow,
    db: Session = Depends(get_db)
):
    """更新工作流"""
    db_workflow = WorkflowCRUD.update(
        db=db,
        workflow_id=workflow_id,
        name=workflow.name,
        description=workflow.description,
        nodes=[node.dict() for node in workflow.nodes],
        edges=[edge.dict() for edge in workflow.edges],
    )
    
    if not db_workflow:
        raise HTTPException(status_code=404, detail="Workflow not found")
    
    return {
        "id": db_workflow.id,
        "name": db_workflow.name,
        "description": db_workflow.description,
        "status": "updated",
        "updated_at": db_workflow.updated_at.isoformat(),
    }


@router.delete("/{workflow_id}")
async def delete_workflow(workflow_id: int, db: Session = Depends(get_db)):
    """删除工作流"""
    success = WorkflowCRUD.delete(db, workflow_id)
    
    if not success:
        raise HTTPException(status_code=404, detail="Workflow not found")
    
    return {"status": "deleted", "id": workflow_id}


@router.post("/execute")
async def execute_workflow(request: WorkflowExecuteRequest):
    """执行工作流
    
    使用工作流引擎执行，支持：
    - 依赖关系解析
    - 并行执行
    - 错误处理
    """
    from app.core.workflow_engine import WorkflowEngine
    
    workflow_dict = {
        "nodes": [node.dict() for node in request.workflow.nodes],
        "edges": [edge.dict() for edge in request.workflow.edges],
    }
    
    # 创建执行引擎
    engine = WorkflowEngine(workflow_dict)
    
    # 执行工作流
    result = await engine.execute()
    
    return result


@router.delete("/{workflow_id}")
async def delete_workflow(workflow_id: str):
    """删除工作流"""
    # TODO: 从数据库删除
    return {
        "status": "deleted",
        "id": workflow_id,
    }


@router.websocket("/ws/execute")
async def execute_workflow_ws(websocket: WebSocket):
    """通过 WebSocket 执行工作流，实时推送进度"""
    from app.core.workflow_engine import WorkflowEngine
    
    await websocket.accept()
    
    try:
        # 接收工作流定义
        data = await websocket.receive_json()
        
        workflow_dict = {
            "nodes": data.get("nodes", []),
            "edges": data.get("edges", []),
        }
        
        # 创建执行引擎
        engine = WorkflowEngine(workflow_dict)
        
        # 定义进度回调
        async def progress_callback(node_id, status, execution, progress):
            await websocket.send_json({
                "type": "node_progress",
                "node_id": node_id,
                "status": status,
                "progress": progress,
                "execution": {
                    "inputs": execution.inputs if execution else {},
                    "outputs": execution.outputs if execution else {},
                    "error": execution.error if execution else None,
                    "duration": execution.duration if execution else None,
                },
            })
        
        # 发送开始消息
        await websocket.send_json({
            "type": "workflow_started",
            "total_nodes": len(workflow_dict["nodes"]),
        })
        
        # 执行工作流
        result = await engine.execute(progress_callback=progress_callback)
        
        # 发送完成消息
        await websocket.send_json({
            "type": "workflow_completed",
            "result": result,
        })
        
    except Exception as e:
        await websocket.send_json({
            "type": "workflow_error",
            "error": str(e),
        })
    
    finally:
        await websocket.close()
