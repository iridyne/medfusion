"""Application-layer training runtime orchestration.

This layer sits between FastAPI routes and the concrete worker implementation.
The API/BFF layer should translate requests into job records and responses; the
application layer owns training lifecycle, artifact synchronization, and result
handoff semantics for the current local worker mode.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import shlex
import shutil
import signal
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np
from sqlalchemy.orm import Session

from med_core.output_layout import RunOutputLayout

from ..models import TrainingJob
from ..time_utils import utcnow
from ..workers.local_training_worker import LocalTrainingWorkerRegistry

logger = logging.getLogger(__name__)


class ImportModelRunFn(Protocol):
    def __call__(self, **kwargs: Any) -> Any: ...


@dataclass(frozen=True)
class TrainingRuntimeContext:
    """Dependencies required to run one training job in the background."""

    session_factory: Any
    worker_registry: LocalTrainingWorkerRegistry
    project_root: Path
    data_dir: Path
    import_model_run_fn: ImportModelRunFn
    clear_task_state: Any


def get_job_or_404(db: Session, job_id: str) -> TrainingJob:
    job = db.query(TrainingJob).filter(TrainingJob.job_id == job_id).first()
    if not job:
        raise RuntimeError(f"训练任务不存在: {job_id}")
    return job


def extract_job_metadata(job: TrainingJob) -> dict[str, Any]:
    config = job.config or {}
    training_model_config = config.get("training_model_config", {})
    dataset_config = config.get("dataset_config", {})
    return {
        "experiment_name": config.get("experiment_name")
        or f"training-{job.job_id[:8]}",
        "dataset_name": dataset_config.get("dataset")
        or dataset_config.get("dataset_name")
        or dataset_config.get("name"),
        "backbone": training_model_config.get("backbone"),
        "num_classes": training_model_config.get("num_classes")
        or dataset_config.get("num_classes"),
    }


def extract_result_handoff(job: TrainingJob) -> dict[str, Any]:
    config = job.config or {}
    handoff = config.get("result_handoff", {})
    return {
        "result_model_id": handoff.get("model_id"),
        "result_model_name": handoff.get("model_name"),
    }


def extract_source_context(job: TrainingJob) -> dict[str, Any]:
    config = job.config or {}
    source_context = config.get("source_context", {})
    return source_context if isinstance(source_context, dict) else {}


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        logger.warning("Failed to parse JSON artifact: %s", path)
        return {}


def build_train_command(config_path: Path, output_dir: Path) -> list[str]:
    if shutil.which("uv"):
        return [
            "uv",
            "run",
            "medfusion",
            "train",
            "--config",
            str(config_path),
            "--output-dir",
            str(output_dir),
        ]

    medfusion_executable = shutil.which("medfusion")
    if medfusion_executable:
        return [
            medfusion_executable,
            "train",
            "--config",
            str(config_path),
            "--output-dir",
            str(output_dir),
        ]

    raise RuntimeError("找不到 medfusion 或 uv，可执行真实训练命令")


def resolve_checkpoint_path(output_dir: Path) -> Path | None:
    checkpoint_dir = RunOutputLayout(output_dir).checkpoints_dir
    preferred_candidates = [
        checkpoint_dir / "best.pth",
        checkpoint_dir / "last.pth",
    ]
    for candidate in preferred_candidates:
        if candidate.exists():
            return candidate
    checkpoints = sorted(checkpoint_dir.glob("*.pth"))
    return checkpoints[0] if checkpoints else None


def history_path_for_job(job: TrainingJob, default_data_dir: Path) -> Path:
    output_dir = Path(job.output_dir or default_data_dir / "experiments" / job.job_id)
    return RunOutputLayout(output_dir).history_path


def read_history_entries(job: TrainingJob, default_data_dir: Path) -> list[dict[str, Any]]:
    history_payload = _read_json(history_path_for_job(job, default_data_dir))
    entries = history_payload.get("entries", [])
    return entries if isinstance(entries, list) else []


def _round_metric(value: float | None, digits: int = 4) -> float | None:
    if value is None:
        return None
    if bool(np.isnan(value)):
        return None
    return round(float(value), digits)


def _safe_rate(numerator: float | int, denominator: float | int) -> float:
    if not denominator:
        return 0.0
    return float(numerator) / float(denominator)


def sync_job_from_history(
    job: TrainingJob,
    *,
    default_data_dir: Path,
) -> dict[str, Any] | None:
    entries = read_history_entries(job, default_data_dir)
    if not entries:
        return None

    latest_entry = entries[-1]
    current_epoch = int(latest_entry.get("epoch") or job.current_epoch or 0)
    job.current_epoch = current_epoch
    job.progress = round(_safe_rate(current_epoch, max(job.total_epochs, 1)) * 100, 2)
    job.current_loss = _round_metric(
        latest_entry.get("val_loss")
        if latest_entry.get("val_loss") is not None
        else latest_entry.get("train_loss")
    )
    job.current_accuracy = _round_metric(
        latest_entry.get("val_accuracy")
        if latest_entry.get("val_accuracy") is not None
        else latest_entry.get("train_accuracy")
    )
    job.current_lr = _round_metric(latest_entry.get("learning_rate"), digits=6)

    val_losses = [
        (int(entry.get("epoch", index + 1)), float(entry["val_loss"]))
        for index, entry in enumerate(entries)
        if entry.get("val_loss") is not None
    ]
    if val_losses:
        best_loss_epoch, best_loss = min(val_losses, key=lambda item: item[1])
        job.best_loss = best_loss
        if job.best_epoch is None:
            job.best_epoch = best_loss_epoch

    val_accuracies = [
        (int(entry.get("epoch", index + 1)), float(entry["val_accuracy"]))
        for index, entry in enumerate(entries)
        if entry.get("val_accuracy") is not None
    ]
    if val_accuracies:
        best_accuracy_epoch, best_accuracy = max(
            val_accuracies, key=lambda item: item[1]
        )
        job.best_accuracy = best_accuracy
        job.best_epoch = best_accuracy_epoch

    return latest_entry


def tail_log(log_path: Path | None, max_lines: int = 40) -> str | None:
    if log_path is None or not log_path.exists():
        return None
    lines = log_path.read_text(encoding="utf-8").splitlines()
    if not lines:
        return None
    return "\n".join(lines[-max_lines:])


async def run_real_training_job(job_id: str, context: TrainingRuntimeContext) -> None:
    """Run a real Python training worker and sync artifacts back into metadata."""
    db = context.session_factory()
    process = None
    try:
        job = get_job_or_404(db, job_id)
        resolved_run = (job.config or {}).get("resolved_run", {})
        config_path = Path(resolved_run["config_path"])
        output_dir = Path(job.output_dir or context.data_dir / "experiments" / job.job_id)
        layout = RunOutputLayout(output_dir).ensure_exists()
        log_path = Path(job.log_file) if job.log_file else layout.training_log_path
        command = build_train_command(config_path, output_dir)

        job.status = "running"
        job.started_at = job.started_at or utcnow()
        job.error_message = None
        db.commit()

        env = os.environ.copy()
        env.setdefault("PYTHONUNBUFFERED", "1")
        env.setdefault("UV_CACHE_DIR", str(context.data_dir / "uv-cache"))

        log_path.write_text(
            f"command: {' '.join(shlex.quote(part) for part in command)}\n",
            encoding="utf-8",
        )
        process = context.worker_registry.start(
            job_id=job_id,
            command=command,
            cwd=context.project_root,
            log_path=log_path,
            env=env,
        )

        while True:
            await asyncio.sleep(1.0)
            db.expire_all()
            job = get_job_or_404(db, job_id)
            sync_job_from_history(job, default_data_dir=context.data_dir)
            return_code = process.poll()

            if return_code is None:
                if job.status not in {"paused", "stopped"}:
                    job.status = "running"
                db.commit()
                continue

            if job.status == "stopped":
                job.completed_at = job.completed_at or utcnow()
                db.commit()
                return

            if return_code != 0:
                job.status = "failed"
                job.completed_at = utcnow()
                job.error_message = tail_log(log_path) or f"训练进程退出码: {return_code}"
                db.commit()
                return

            checkpoint_path = resolve_checkpoint_path(output_dir)
            if checkpoint_path is None:
                raise FileNotFoundError(
                    f"训练完成但未找到 checkpoint: {output_dir / 'checkpoints'}"
                )

            metadata = extract_job_metadata(job)
            source_context = extract_source_context(job)
            imported_model = context.import_model_run_fn(
                db=db,
                config_path=config_path,
                checkpoint_path=checkpoint_path,
                output_dir=output_dir,
                split="test",
                attention_samples=4,
                name=f"{metadata['experiment_name']}-model",
                description="Web UI 触发的真实训练产物",
                tags=[
                    "web-ui",
                    f"job:{job.job_id}",
                    *(["advanced-builder"] if source_context.get("source_type") == "advanced_builder" else []),
                    *(
                        [f"blueprint:{source_context['blueprint_id']}"]
                        if source_context.get("blueprint_id")
                        else []
                    ),
                ],
                import_source="web",
                source_context=source_context,
            )

            sync_job_from_history(job, default_data_dir=context.data_dir)
            existing_config = dict(job.config or {})
            existing_config["result_handoff"] = {
                "model_id": getattr(imported_model, "id", None),
                "model_name": getattr(imported_model, "name", None),
            }
            job.config = existing_config
            job.status = "completed"
            job.progress = 100.0
            job.completed_at = utcnow()
            job.error_message = None
            db.commit()
            return
    except Exception as e:
        logger.exception("真实训练任务失败: %s", job_id)
        db.rollback()
        job = db.query(TrainingJob).filter(TrainingJob.job_id == job_id).first()
        if job and job.status != "stopped":
            job.status = "failed"
            job.error_message = str(e)
            job.completed_at = utcnow()
            db.commit()
    finally:
        if process is not None and process.poll() is None:
            context.worker_registry.signal(job_id, signal.SIGTERM)
        db.close()
        context.worker_registry.remove(job_id)
        context.clear_task_state(job_id)
