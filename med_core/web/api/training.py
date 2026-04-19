"""训练任务 API."""

from __future__ import annotations

import asyncio
import json
import logging
import signal
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, ConfigDict
from sqlalchemy.orm import Session

from ..application import TrainingJobApplicationService, TrainingJobRequestError
from ..application.training_runtime import (
    TrainingRuntimeContext,
    build_train_command as _build_train_command_impl,
    extract_job_metadata as _extract_job_metadata_impl,
    extract_result_handoff as _extract_result_handoff_impl,
    get_job_or_404 as _get_job_or_404_impl,
    history_path_for_job as _history_path_for_job_impl,
    read_history_entries as _read_history_entries_impl,
    run_real_training_job as _run_real_training_job_impl,
    sync_job_from_history as _sync_job_from_history_impl,
)
from ..config import settings
from ..database import SessionLocal, get_db_session
from ..model_registry import import_model_run
from ..models import TrainingJob
from ..workers import local_training_worker_registry

router = APIRouter()
logger = logging.getLogger(__name__)

_training_tasks: dict[str, asyncio.Task[None]] = {}

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_TRAINING_PAUSE_SIGNAL = getattr(signal, "SIGSTOP", 19)
_TRAINING_RESUME_SIGNAL = getattr(signal, "SIGCONT", 18)
_TRAINING_TERMINATE_SIGNAL = getattr(signal, "SIGTERM", 15)
if not hasattr(signal, "SIGSTOP"):
    setattr(signal, "SIGSTOP", _TRAINING_PAUSE_SIGNAL)
if not hasattr(signal, "SIGCONT"):
    setattr(signal, "SIGCONT", _TRAINING_RESUME_SIGNAL)


class TrainingConfig(BaseModel):
    """训练配置."""

    experiment_name: str
    training_model_config: dict[str, Any]
    dataset_config: dict[str, Any]
    training_config: dict[str, Any]
    source_context: dict[str, Any] | None = None


class TrainingJobResponse(BaseModel):
    """训练任务响应."""

    model_config = ConfigDict(from_attributes=True)

    id: int
    job_id: str
    experiment_name: str
    dataset_name: str | None
    backbone: str | None
    status: str
    progress: float
    current_epoch: int
    total_epochs: int
    current_loss: float | None
    current_accuracy: float | None
    result_model_id: int | None = None
    result_model_name: str | None = None
    created_at: str


def _to_http_exception(exc: TrainingJobRequestError) -> HTTPException:
    return HTTPException(status_code=exc.status_code, detail=exc.detail)


def _get_job_or_404(db: Session, job_id: str) -> TrainingJob:
    try:
        return _get_job_or_404_impl(db, job_id)
    except RuntimeError as exc:
        raise TrainingJobRequestError(404, "训练任务不存在") from exc


def _extract_job_metadata(job: TrainingJob) -> dict[str, Any]:
    return _extract_job_metadata_impl(job)


def _extract_result_handoff(job: TrainingJob) -> dict[str, Any]:
    return _extract_result_handoff_impl(job)


def _clear_task_state(job_id: str) -> None:
    _training_tasks.pop(job_id, None)


def _prepare_job_output(job_id: str) -> tuple[str, str]:
    return _training_job_service().prepare_job_output(job_id)


def _training_runtime_context() -> TrainingRuntimeContext:
    return TrainingRuntimeContext(
        session_factory=SessionLocal,
        worker_registry=local_training_worker_registry,
        project_root=_PROJECT_ROOT,
        data_dir=settings.data_dir,
        import_model_run_fn=import_model_run,
        clear_task_state=_clear_task_state,
    )


def _schedule_training_task(job_id: str) -> None:
    _training_tasks[job_id] = asyncio.create_task(_run_real_training_job(job_id))


def _training_job_service() -> TrainingJobApplicationService:
    return TrainingJobApplicationService(
        project_root=_PROJECT_ROOT,
        data_dir=settings.data_dir,
        schedule_background_run=_schedule_training_task,
        signal_process=_signal_process,
        pause_signal=_TRAINING_PAUSE_SIGNAL,
        resume_signal=_TRAINING_RESUME_SIGNAL,
        terminate_signal=_TRAINING_TERMINATE_SIGNAL,
        get_job_or_404=_get_job_or_404,
        extract_job_metadata=_extract_job_metadata,
        extract_result_handoff=_extract_result_handoff,
        read_history_entries=_read_history_entries,
        sync_job_from_history=_sync_job_from_history,
    )


def _build_training_config_artifact(
    db: Session,
    payload: TrainingConfig | dict[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    payload_dict = payload.model_dump() if isinstance(payload, BaseModel) else payload
    return _training_job_service().build_training_config_artifact(
        db=db,
        payload=payload_dict,
        output_dir=output_dir,
    )


def _build_train_command(config_path: Path, output_dir: Path) -> list[str]:
    return _build_train_command_impl(config_path, output_dir)


def _signal_process(job_id: str, process_signal: signal.Signals | int) -> bool:
    return local_training_worker_registry.signal(job_id, process_signal)


def _dispatch_ws_control_signal(
    *,
    route_job_id: str,
    action: str | None,
    payload_job_id: str | None,
) -> bool:
    return _training_job_service().dispatch_ws_control_signal(
        route_job_id=route_job_id,
        action=action,
        payload_job_id=payload_job_id,
    )


def _read_history_entries(job: TrainingJob) -> list[dict[str, Any]]:
    return _read_history_entries_impl(job, settings.data_dir)


def _sync_job_from_history(job: TrainingJob) -> dict[str, Any] | None:
    return _sync_job_from_history_impl(job, default_data_dir=settings.data_dir)


def _history_path_for_job(job: TrainingJob) -> Path:
    return _history_path_for_job_impl(job, settings.data_dir)


async def _run_real_training_job(job_id: str) -> None:
    await _run_real_training_job_impl(job_id, _training_runtime_context())


@router.post("/start")
async def start_training(
    config: TrainingConfig,
    db: Session = Depends(get_db_session),
) -> dict[str, Any]:
    """开始训练任务."""
    try:
        return _training_job_service().start_training(db=db, payload=config.model_dump())
    except TrainingJobRequestError as exc:
        raise _to_http_exception(exc) from exc
    except Exception as exc:
        logger.error("创建训练任务失败: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get("/jobs")
async def list_training_jobs(
    skip: int = 0,
    limit: int = 20,
    db: Session = Depends(get_db_session),
) -> list[TrainingJobResponse]:
    """获取训练任务列表."""
    items = _training_job_service().list_training_jobs(db=db, skip=skip, limit=limit)
    return [TrainingJobResponse.model_validate(item) for item in items]


@router.get("/{job_id}/status")
async def get_training_status(
    job_id: str,
    db: Session = Depends(get_db_session),
) -> dict[str, Any]:
    """获取训练任务状态."""
    try:
        return _training_job_service().get_training_status(db=db, job_id=job_id)
    except TrainingJobRequestError as exc:
        raise _to_http_exception(exc) from exc


@router.get("/{job_id}/history")
async def get_training_history(
    job_id: str,
    db: Session = Depends(get_db_session),
) -> dict[str, Any]:
    """获取训练历史，用于真实曲线回放。"""
    try:
        return _training_job_service().get_training_history(db=db, job_id=job_id)
    except TrainingJobRequestError as exc:
        raise _to_http_exception(exc) from exc


@router.post("/{job_id}/pause")
async def pause_training(
    job_id: str,
    db: Session = Depends(get_db_session),
) -> dict[str, str]:
    """暂停训练任务."""
    try:
        return _training_job_service().pause_training(db=db, job_id=job_id)
    except TrainingJobRequestError as exc:
        raise _to_http_exception(exc) from exc


@router.post("/{job_id}/resume")
async def resume_training(
    job_id: str,
    db: Session = Depends(get_db_session),
) -> dict[str, str]:
    """恢复训练任务."""
    try:
        return _training_job_service().resume_training(db=db, job_id=job_id)
    except TrainingJobRequestError as exc:
        raise _to_http_exception(exc) from exc


@router.post("/{job_id}/stop")
async def stop_training(
    job_id: str,
    db: Session = Depends(get_db_session),
) -> dict[str, str]:
    """停止训练任务."""
    try:
        return _training_job_service().stop_training(db=db, job_id=job_id)
    except TrainingJobRequestError as exc:
        raise _to_http_exception(exc) from exc


@router.websocket("/ws/{job_id}")
async def training_websocket(websocket: WebSocket, job_id: str) -> None:
    """训练任务 WebSocket 连接."""
    await websocket.accept()
    logger.info("WebSocket 连接已建立: %s", job_id)

    db = SessionLocal()
    service = _training_job_service()
    try:
        while True:
            try:
                text = await asyncio.wait_for(websocket.receive_text(), timeout=1.0)
                payload = json.loads(text)
                service.dispatch_ws_control_signal(
                    route_job_id=job_id,
                    action=payload.get("action"),
                    payload_job_id=payload.get("job_id"),
                )
            except TimeoutError:
                pass
            except json.JSONDecodeError:
                pass

            messages, should_close = service.build_websocket_messages(db=db, job_id=job_id)
            for message in messages:
                await websocket.send_json(message)

            if should_close:
                break

            if any(message.get("type") == "error" for message in messages):
                await asyncio.sleep(1.0)
    except WebSocketDisconnect:
        logger.info("WebSocket 连接已断开: %s", job_id)
    finally:
        db.close()
