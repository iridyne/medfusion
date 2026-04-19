"""Advanced builder API/BFF routes."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from ..application.advanced_builder import (
    build_training_payload_from_runspec,
    compile_graph_to_runspec,
    export_catalog,
)
from ..database import get_db_session

router = APIRouter()


class AdvancedBuilderNodePayload(BaseModel):
    id: str
    type: str | None = None
    data: dict[str, Any] = Field(default_factory=dict)
    position: dict[str, float] = Field(default_factory=dict)


class AdvancedBuilderEdgePayload(BaseModel):
    id: str | None = None
    source: str
    target: str
    sourceHandle: str | None = None
    targetHandle: str | None = None
    label: str | None = None


class AdvancedBuilderCompileRequest(BaseModel):
    nodes: list[AdvancedBuilderNodePayload]
    edges: list[AdvancedBuilderEdgePayload]
    blueprint_id: str | None = None


@router.get("/catalog")
async def get_advanced_builder_catalog() -> dict[str, Any]:
    """Return the formal-release advanced builder registry."""
    return export_catalog()


@router.post("/compile")
async def compile_advanced_builder(
    request: AdvancedBuilderCompileRequest,
) -> dict[str, Any]:
    """Compile a constrained GraphSpec into a RunSpec draft."""
    return compile_graph_to_runspec(
        nodes=[node.model_dump() for node in request.nodes],
        edges=[edge.model_dump() for edge in request.edges],
    )


@router.post("/start-training")
async def start_training_from_advanced_builder(
    request: AdvancedBuilderCompileRequest,
    db: Session = Depends(get_db_session),
) -> dict[str, Any]:
    """Compile a graph and immediately create a real training job."""
    compiled = compile_graph_to_runspec(
        nodes=[node.model_dump() for node in request.nodes],
        edges=[edge.model_dump() for edge in request.edges],
    )

    run_spec = compiled.get("run_spec")
    contract_validation = compiled.get("contract_validation")
    if not run_spec or not contract_validation or not contract_validation.get("ok"):
        raise HTTPException(
            status_code=400,
            detail={
                "code": "advanced_builder_compile_not_ready",
                "message": "当前节点图还没有通过正式配置校验，不能直接创建训练任务。",
                "compile_result": compiled,
            },
        )

    from ..api import training as training_api

    payload = build_training_payload_from_runspec(run_spec)
    source_context = payload.setdefault("source_context", {})
    source_context["preset"] = compiled["preset"]
    if request.blueprint_id:
        source_context["blueprint_id"] = request.blueprint_id
    job_response = await training_api.start_training(
        training_api.TrainingConfig.model_validate(payload),
        db,
    )
    return {
        **job_response,
        "preset": compiled["preset"],
        "experiment_name": run_spec["experimentName"],
        "compile_result": {
            "mainline_contract": compiled.get("mainline_contract"),
            "contract_validation": contract_validation,
        },
    }
