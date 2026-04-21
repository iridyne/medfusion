"""Workflow API routes for the constrained preview workflow editor."""

from __future__ import annotations

import asyncio
import logging
import uuid
from dataclasses import dataclass
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from ..config import settings
from ..database import SessionLocal, get_db_session
from ..workflow_engine import NodeStatus, WorkflowEngine

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/workflows", tags=["workflows"])


@dataclass
class WorkflowRunState:
    workflow_id: str
    name: str | None
    engine: WorkflowEngine
    training_job_id: str
    node_roles: dict[str, str]


active_workflows: dict[str, WorkflowRunState] = {}


def _workflow_disabled_detail() -> dict[str, Any]:
    return {
        "code": "workflow_experimental_disabled",
        "message": (
            "Workflow editor is disabled. Set "
            "MEDFUSION_ENABLE_EXPERIMENTAL_WORKFLOW=true to expose the preview."
        ),
        "enable_env": "MEDFUSION_ENABLE_EXPERIMENTAL_WORKFLOW=true",
        "recommended_primary_flow": [
            "Open Getting Started with `medfusion start`",
            "Create a real config in the Run Wizard",
            "Use training monitor and model library for the stable flow",
        ],
    }


def _workflow_preview_unavailable_detail() -> dict[str, Any]:
    return {
        "code": "workflow_preview_scope_limited",
        "message": (
            "当前工作流 preview 只支持单条主线："
            "dataLoader -> model -> training -> optional evaluation。"
        ),
        "recommended_graph": [
            "一个数据节点",
            "一个模型节点",
            "一个训练节点",
            "可选一个评估节点",
        ],
    }


def _ensure_workflow_enabled() -> None:
    if not settings.enable_experimental_workflow:
        raise HTTPException(status_code=503, detail=_workflow_disabled_detail())


def _node_ids_by_type(engine: WorkflowEngine, node_type: str) -> list[str]:
    return [node.id for node in engine.nodes.values() if node.type == node_type]


def _has_edge(
    engine: WorkflowEngine,
    *,
    source: str,
    target: str,
    source_handle: str | None = None,
    target_handle: str | None = None,
) -> bool:
    for edge in engine.edges:
        if edge.source != source or edge.target != target:
            continue
        if source_handle is not None and edge.source_handle != source_handle:
            continue
        if target_handle is not None and edge.target_handle != target_handle:
            continue
        return True
    return False


def _validate_preview_graph(engine: WorkflowEngine) -> tuple[dict[str, str], list[str]]:
    errors: list[str] = []
    roles: dict[str, str] = {}

    data_loader_ids = _node_ids_by_type(engine, "dataLoader")
    model_ids = _node_ids_by_type(engine, "model")
    training_ids = _node_ids_by_type(engine, "training")
    evaluation_ids = _node_ids_by_type(engine, "evaluation")

    if len(data_loader_ids) != 1:
        errors.append("当前 preview 要求且只允许一个数据节点（dataLoader）")
    else:
        roles["data_loader"] = data_loader_ids[0]

    if len(model_ids) != 1:
        errors.append("当前 preview 要求且只允许一个模型节点（model）")
    else:
        roles["model"] = model_ids[0]

    if len(training_ids) != 1:
        errors.append("当前 preview 要求且只允许一个训练节点（training）")
    else:
        roles["training"] = training_ids[0]

    if len(evaluation_ids) > 1:
        errors.append("当前 preview 最多只允许一个评估节点（evaluation）")
    elif evaluation_ids:
        roles["evaluation"] = evaluation_ids[0]

    if errors:
        return roles, errors

    if not _has_edge(
        engine,
        source=roles["model"],
        target=roles["training"],
        source_handle="model",
        target_handle="model",
    ):
        errors.append("训练节点必须接收来自模型节点的 model 输出")

    if not _has_edge(
        engine,
        source=roles["data_loader"],
        target=roles["training"],
        source_handle="dataset",
        target_handle="train_data",
    ):
        errors.append("训练节点必须接收来自数据节点的 train_data 输入")

    if "evaluation" in roles:
        if not _has_edge(
            engine,
            source=roles["training"],
            target=roles["evaluation"],
            source_handle="trained_model",
            target_handle="model",
        ):
            errors.append("评估节点必须接收来自训练节点的 trained_model 输出")
        if not _has_edge(
            engine,
            source=roles["data_loader"],
            target=roles["evaluation"],
            source_handle="dataset",
            target_handle="test_data",
        ):
            errors.append("评估节点必须接收来自数据节点的 test_data 输入")

    return roles, errors


def _build_training_payload(
    *,
    workflow_id: str,
    workflow_name: str | None,
    engine: WorkflowEngine,
    node_roles: dict[str, str],
) -> dict[str, Any]:
    data_node = engine.nodes[node_roles["data_loader"]].data
    model_node = engine.nodes[node_roles["model"]].data
    training_node = engine.nodes[node_roles["training"]].data

    experiment_name = (
        str(workflow_name or training_node.get("experimentName") or "").strip()
        or f"workflow-{workflow_id[:8]}"
    )

    dataset_name = (
        data_node.get("datasetName")
        or data_node.get("label")
        or data_node.get("datasetId")
        or "workflow-dataset"
    )

    dataset_config = {
        "dataset": dataset_name,
        "dataset_id": data_node.get("datasetId"),
        "data_path": data_node.get("dataPath"),
        "csv_path": data_node.get("csvPath"),
        "image_dir": data_node.get("imageDir"),
        "image_path_column": data_node.get("imagePathColumn"),
        "target_column": data_node.get("targetColumn"),
        "patient_id_column": data_node.get("patientIdColumn"),
        "numerical_features": data_node.get("numericalFeatures"),
        "categorical_features": data_node.get("categoricalFeatures"),
        "num_classes": model_node.get("numClasses") or data_node.get("numClasses"),
    }

    training_model_config = {
        "backbone": model_node.get("backbone"),
        "num_classes": model_node.get("numClasses"),
        "pretrained": model_node.get("pretrained", True),
        "freeze_backbone": model_node.get("freezeBackbone", False),
        "fusion_type": model_node.get("fusion"),
        "feature_dim": model_node.get("featureDim"),
        "tabular_output_dim": model_node.get("tabularOutputDim"),
    }

    training_config = {
        "epochs": training_node.get("epochs"),
        "batch_size": data_node.get("batchSize"),
        "num_workers": data_node.get("numWorkers"),
        "learning_rate": training_node.get("learningRate"),
        "optimizer": training_node.get("optimizer"),
        "mixed_precision": training_node.get("useAmp", False),
        "use_progressive_training": training_node.get(
            "useProgressiveTraining", False
        ),
        "image_size": data_node.get("imageSize"),
        "monitor": training_node.get("monitor"),
        "mode": training_node.get("mode"),
    }

    return {
        "experiment_name": experiment_name,
        "training_model_config": training_model_config,
        "dataset_config": dataset_config,
        "training_config": training_config,
        "source_context": {
            "source_type": "workflow",
            "entrypoint": "workflow-editor",
            "workflow_id": workflow_id,
            "workflow_name": workflow_name,
        },
    }


def _set_initial_preview_node_states(
    state: WorkflowRunState,
) -> None:
    for node in state.engine.nodes.values():
        node.status = NodeStatus.PENDING
        node.error = None
        node.result = None

    state.engine.nodes[state.node_roles["data_loader"]].status = NodeStatus.COMPLETED
    state.engine.nodes[state.node_roles["model"]].status = NodeStatus.COMPLETED
    state.engine.nodes[state.node_roles["training"]].status = NodeStatus.RUNNING
    if "evaluation" in state.node_roles:
        state.engine.nodes[state.node_roles["evaluation"]].status = NodeStatus.PENDING


def _sync_preview_state(
    state: WorkflowRunState,
    db: Session,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    from ..api import training as training_api

    training_status = training_api._training_job_service().get_training_status(
        db=db,
        job_id=state.training_job_id,
    )
    training_node = state.engine.nodes[state.node_roles["training"]]
    evaluation_node = (
        state.engine.nodes[state.node_roles["evaluation"]]
        if "evaluation" in state.node_roles
        else None
    )

    raw_status = training_status["status"]
    if raw_status in {"running", "paused"}:
        training_node.status = NodeStatus.RUNNING
        training_node.error = None
        if evaluation_node is not None:
            evaluation_node.status = NodeStatus.PENDING
            evaluation_node.error = None
    elif raw_status == "completed":
        training_node.status = NodeStatus.COMPLETED
        training_node.error = None
        training_node.result = {
            "job_id": state.training_job_id,
            "result_model_id": training_status.get("result_model_id"),
            "result_model_name": training_status.get("result_model_name"),
        }
        if evaluation_node is not None:
            if training_status.get("result_model_id") is not None:
                evaluation_node.status = NodeStatus.COMPLETED
                evaluation_node.error = None
                evaluation_node.result = {
                    "result_model_id": training_status.get("result_model_id"),
                    "result_model_name": training_status.get("result_model_name"),
                }
            else:
                evaluation_node.status = NodeStatus.FAILED
                evaluation_node.error = "训练已完成，但结果回流未生成 model handoff"
    else:
        training_node.status = NodeStatus.FAILED
        training_node.error = training_status.get("error_message") or raw_status
        if evaluation_node is not None:
            evaluation_node.status = NodeStatus.FAILED
            evaluation_node.error = training_node.error

    results = None
    if training_node.status == NodeStatus.COMPLETED:
        results = {
            "training_job_id": state.training_job_id,
            "result_model_id": training_status.get("result_model_id"),
            "result_model_name": training_status.get("result_model_name"),
            "training_status": raw_status,
        }

    return training_status, results


def _validate_preview_request(
    workflow: dict[str, Any],
) -> tuple[WorkflowEngine, dict[str, str], list[str]]:
    engine = WorkflowEngine(data_dir=settings.data_dir, enable_monitoring=False)
    engine.load_workflow(workflow)
    is_valid, base_errors = engine.validate(include_runtime_readiness=False)
    node_roles, preview_errors = _validate_preview_graph(engine)
    errors = list(dict.fromkeys([*base_errors, *preview_errors]))
    return engine, node_roles, ([] if is_valid and not preview_errors else errors)


class WorkflowData(BaseModel):
    nodes: list[dict[str, Any]] = Field(..., description="节点列表")
    edges: list[dict[str, Any]] = Field(..., description="边列表")


class WorkflowValidateRequest(BaseModel):
    workflow: WorkflowData


class WorkflowValidateResponse(BaseModel):
    valid: bool
    errors: list[str] = []


class WorkflowExecuteRequest(BaseModel):
    workflow: WorkflowData
    name: str | None = None


class WorkflowExecuteResponse(BaseModel):
    workflow_id: str
    status: str
    message: str
    training_job_id: str | None = None


class WorkflowStatusResponse(BaseModel):
    workflow_id: str
    status: dict[str, Any]
    results: dict[str, Any] | None = None
    training_job_id: str | None = None
    training_status: dict[str, Any] | None = None


@router.post("/validate", response_model=WorkflowValidateResponse)
async def validate_workflow(request: WorkflowValidateRequest) -> WorkflowValidateResponse:
    _ensure_workflow_enabled()
    try:
        _engine, _node_roles, errors = _validate_preview_request(
            request.workflow.model_dump()
        )
        return WorkflowValidateResponse(valid=not errors, errors=errors)
    except Exception as exc:
        logger.error("Workflow validation error: %s", exc)
        return WorkflowValidateResponse(valid=False, errors=[str(exc)])


@router.post("/execute", response_model=WorkflowExecuteResponse)
async def execute_workflow(
    request: WorkflowExecuteRequest,
    db: Session = Depends(get_db_session),
) -> WorkflowExecuteResponse:
    _ensure_workflow_enabled()
    workflow_id = str(uuid.uuid4())

    try:
        engine, node_roles, errors = _validate_preview_request(
            request.workflow.model_dump()
        )
        if errors:
            raise HTTPException(
                status_code=400,
                detail={
                    **_workflow_preview_unavailable_detail(),
                    "errors": errors,
                },
            )

        from ..api import training as training_api

        payload = _build_training_payload(
            workflow_id=workflow_id,
            workflow_name=request.name,
            engine=engine,
            node_roles=node_roles,
        )
        training_response = await training_api.start_training(
            training_api.TrainingConfig.model_validate(payload),
            db,
        )

        state = WorkflowRunState(
            workflow_id=workflow_id,
            name=request.name,
            engine=engine,
            training_job_id=training_response["job_id"],
            node_roles=node_roles,
        )
        _set_initial_preview_node_states(state)
        active_workflows[workflow_id] = state

        return WorkflowExecuteResponse(
            workflow_id=workflow_id,
            status="started",
            message=f"工作流 {request.name or workflow_id} 已启动真实训练任务",
            training_job_id=training_response["job_id"],
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Failed to start workflow execution: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get("/{workflow_id}/status", response_model=WorkflowStatusResponse)
async def get_workflow_status(
    workflow_id: str,
    db: Session = Depends(get_db_session),
) -> WorkflowStatusResponse:
    _ensure_workflow_enabled()
    state = active_workflows.get(workflow_id)
    if state is None:
        raise HTTPException(status_code=404, detail=f"工作流 {workflow_id} 不存在")

    training_status, results = _sync_preview_state(state, db)
    return WorkflowStatusResponse(
        workflow_id=workflow_id,
        status=state.engine.get_status(),
        results=results,
        training_job_id=state.training_job_id,
        training_status=training_status,
    )


@router.delete("/{workflow_id}")
async def delete_workflow(workflow_id: str) -> dict[str, str]:
    _ensure_workflow_enabled()
    if workflow_id not in active_workflows:
        raise HTTPException(status_code=404, detail=f"工作流 {workflow_id} 不存在")
    del active_workflows[workflow_id]
    return {"message": f"工作流 {workflow_id} 已删除"}


@router.websocket("/{workflow_id}/progress")
async def workflow_progress_websocket(websocket: WebSocket, workflow_id: str) -> None:
    await websocket.accept()

    if not settings.enable_experimental_workflow:
        await websocket.send_json({"type": "error", "detail": _workflow_disabled_detail()})
        await websocket.close()
        return

    state = active_workflows.get(workflow_id)
    if state is None:
        await websocket.send_json({"type": "error", "error": f"工作流 {workflow_id} 不存在"})
        await websocket.close()
        return

    db = SessionLocal()
    try:
        while True:
            training_status, results = _sync_preview_state(state, db)
            status = state.engine.get_status()
            await websocket.send_json(
                {
                    "type": "status",
                    "data": status,
                    "training_job_id": state.training_job_id,
                    "training_status": training_status,
                    "results": results,
                }
            )
            if status["completed"] + status["failed"] == status["total"]:
                break
            await asyncio.sleep(1.0)
    except WebSocketDisconnect:
        logger.info("WebSocket disconnected for workflow %s", workflow_id)
    except Exception as exc:
        logger.error("WebSocket error for workflow %s: %s", workflow_id, exc)
        try:
            await websocket.send_json({"type": "error", "error": str(exc)})
        except Exception:
            logger.debug("Failed to send workflow websocket error", exc_info=True)
    finally:
        db.close()
        try:
            await websocket.close()
        except Exception:
            logger.debug("Failed to close workflow websocket cleanly", exc_info=True)


@router.get("/")
async def list_workflows(
    db: Session = Depends(get_db_session),
) -> dict[str, Any]:
    _ensure_workflow_enabled()
    workflows: list[dict[str, Any]] = []
    for workflow_id, state in active_workflows.items():
        training_status, _results = _sync_preview_state(state, db)
        status = state.engine.get_status()
        workflows.append(
            {
                "workflow_id": workflow_id,
                "name": state.name,
                "training_job_id": state.training_job_id,
                "training_status": training_status["status"],
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
async def get_workflow_resources(
    workflow_id: str,
    duration: int | None = None,
) -> dict[str, Any]:
    _ensure_workflow_enabled()
    if workflow_id not in active_workflows:
        raise HTTPException(status_code=404, detail=f"工作流 {workflow_id} 不存在")
    return {
        "workflow_id": workflow_id,
        "duration": duration,
        "message": "当前 preview 不单独暴露 workflow 资源轨迹，请改看 /training 与 /system。",
    }


@router.get("/{workflow_id}/checkpoints")
async def list_workflow_checkpoints(workflow_id: str) -> dict[str, Any]:
    _ensure_workflow_enabled()
    if workflow_id not in active_workflows:
        raise HTTPException(status_code=404, detail=f"工作流 {workflow_id} 不存在")
    return {
        "workflow_id": workflow_id,
        "checkpoints": [],
        "total": 0,
        "message": "当前 preview 直接复用训练主链，检查点请查看对应 training job 的输出目录。",
    }


@router.post("/{workflow_id}/resume")
async def resume_workflow(
    workflow_id: str,
    checkpoint_path: str | None = None,
) -> dict[str, Any]:
    _ensure_workflow_enabled()
    raise HTTPException(
        status_code=400,
        detail={
            "code": "workflow_preview_resume_unsupported",
            "message": "当前 workflow preview 不支持从旧检查点恢复图执行，请直接重启对应训练任务。",
            "workflow_id": workflow_id,
            "checkpoint_path": checkpoint_path,
        },
    )
