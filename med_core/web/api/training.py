"""训练任务 API"""

from __future__ import annotations

import asyncio
import csv
import json
import logging
import os
import shlex
import shutil
import signal
import subprocess
import uuid
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import APIRouter, Depends, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, ConfigDict
from sqlalchemy.orm import Session

from med_core.configs import load_config, save_config

from ..config import settings
from ..database import SessionLocal, get_db_session
from ..model_registry import import_model_run
from ..models import DatasetInfo, TrainingJob
from ..time_utils import utcnow

router = APIRouter()
logger = logging.getLogger(__name__)

_training_tasks: dict[str, asyncio.Task[None]] = {}
_training_processes: dict[str, subprocess.Popen[str]] = {}
_pause_flags: dict[str, bool] = {}
_stop_flags: dict[str, bool] = {}

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_BACKBONE_ALIASES: dict[str, str] = {
    "vit_b16": "vit_b_16",
    "vit_b_16": "vit_b_16",
    "swin_tiny": "swin_t",
    "swin_t": "swin_t",
}
_IMAGE_COLUMN_CANDIDATES = (
    "image_path",
    "image",
    "img_path",
    "scan_path",
    "filepath",
    "file_path",
)
_TARGET_COLUMN_CANDIDATES = (
    "label",
    "diagnosis",
    "diagnosis_binary",
    "target",
    "class",
    "y",
)
_PATIENT_ID_COLUMN_CANDIDATES = (
    "patient_id",
    "patient",
    "subject_id",
    "study_id",
    "id",
)

_BACKBONE_PARAMETER_MAP: dict[str, int] = {
    "resnet18": 11_700_000,
    "resnet34": 21_800_000,
    "resnet50": 25_600_000,
    "resnet101": 44_500_000,
    "mobilenetv2": 3_500_000,
    "efficientnet_b0": 5_300_000,
    "vit_b_16": 86_000_000,
    "vit_b16": 86_000_000,
    "swin_t": 28_300_000,
    "swin_tiny": 28_300_000,
}


class TrainingConfig(BaseModel):
    """训练配置"""

    experiment_name: str
    training_model_config: dict[str, Any]
    dataset_config: dict[str, Any]
    training_config: dict[str, Any]


class TrainingJobResponse(BaseModel):
    """训练任务响应"""

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
    created_at: str


def _get_job_or_404(db: Session, job_id: str) -> TrainingJob:
    job = db.query(TrainingJob).filter(TrainingJob.job_id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="训练任务不存在")
    return job


def _extract_job_metadata(job: TrainingJob) -> dict[str, Any]:
    config = job.config or {}
    training_model_config = config.get("training_model_config", {})
    dataset_config = config.get("dataset_config", {})
    return {
        "experiment_name": config.get("experiment_name") or f"training-{job.job_id[:8]}",
        "dataset_name": dataset_config.get("dataset")
        or dataset_config.get("dataset_name")
        or dataset_config.get("name"),
        "backbone": training_model_config.get("backbone"),
        "num_classes": training_model_config.get("num_classes")
        or dataset_config.get("num_classes"),
    }


def _estimate_num_parameters(backbone: str | None) -> int | None:
    if not backbone:
        return None
    return _BACKBONE_PARAMETER_MAP.get(backbone)


def _prepare_job_output(job_id: str) -> tuple[str, str]:
    output_dir = settings.data_dir / "experiments" / job_id
    output_dir.mkdir(parents=True, exist_ok=True)
    log_file = output_dir / "training.log"
    if not log_file.exists():
        log_file.write_text("training started\n", encoding="utf-8")
    return str(output_dir), str(log_file)


def _coerce_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value if str(item).strip()]
    if isinstance(value, tuple):
        return [str(item) for item in value if str(item).strip()]
    if isinstance(value, str):
        if not value.strip():
            return []
        if "," in value:
            return [item.strip() for item in value.split(",") if item.strip()]
        return [value]
    return [str(value)]


def _normalize_backbone_name(backbone: str | None) -> str | None:
    if not backbone:
        return None
    return _BACKBONE_ALIASES.get(backbone, backbone)


def _resolve_path(path_value: str | Path | None) -> Path | None:
    if not path_value:
        return None
    path = Path(path_value).expanduser()
    if path.is_absolute():
        return path

    candidates = [
        (_PROJECT_ROOT / path),
        (Path.cwd() / path),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()

    return (_PROJECT_ROOT / path).resolve()


def _pick_column(
    headers: list[str],
    *,
    preferred: str | None = None,
    candidates: tuple[str, ...],
) -> str | None:
    if preferred and preferred in headers:
        return preferred
    for candidate in candidates:
        if candidate in headers:
            return candidate
    return None


def _read_csv_preview(
    csv_path: Path,
    limit: int = 64,
) -> tuple[list[str], list[dict[str, str]]]:
    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        headers = list(reader.fieldnames or [])
        rows: list[dict[str, str]] = []
        for index, row in enumerate(reader):
            rows.append(
                {
                    key: (value.strip() if isinstance(value, str) else value)
                    for key, value in row.items()
                }
            )
            if index + 1 >= limit:
                break
    return headers, rows


def _infer_csv_path(dataset_path: Path | None, explicit_csv_path: str | None) -> Path:
    if explicit_csv_path:
        resolved = _resolve_path(explicit_csv_path)
        if resolved is not None and resolved.exists():
            return resolved
        raise HTTPException(status_code=400, detail=f"CSV 文件不存在: {explicit_csv_path}")

    if dataset_path is None:
        raise HTTPException(status_code=400, detail="缺少 data_path 或 csv_path，无法生成真实训练配置")

    if dataset_path.is_file():
        if dataset_path.suffix.lower() != ".csv":
            raise HTTPException(status_code=400, detail=f"不支持的数据描述文件: {dataset_path}")
        return dataset_path

    candidates = [
        dataset_path / "metadata.csv",
        dataset_path / "dataset.csv",
        dataset_path / "labels.csv",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    csv_files = sorted(dataset_path.glob("*.csv"))
    if len(csv_files) == 1:
        return csv_files[0]

    raise HTTPException(
        status_code=400,
        detail=f"无法从 {dataset_path} 自动识别 CSV，请显式提供 csv_path",
    )


def _is_numeric_value(value: str) -> bool:
    try:
        float(value)
    except (TypeError, ValueError):
        return False
    return True


def _infer_feature_columns(
    headers: list[str],
    sample_rows: list[dict[str, str]],
    excluded_columns: set[str],
) -> tuple[list[str], list[str]]:
    numerical_features: list[str] = []
    categorical_features: list[str] = []

    for column in headers:
        if column in excluded_columns:
            continue

        values = [
            row[column]
            for row in sample_rows
            if column in row and row[column] not in {None, ""}
        ]
        if not values:
            continue

        if all(_is_numeric_value(value) for value in values):
            unique_values = {float(value) for value in values}
            all_integer_like = all(float(value).is_integer() for value in values)
            if all_integer_like and len(unique_values) <= min(10, max(2, len(values) // 2)):
                categorical_features.append(column)
            else:
                numerical_features.append(column)
            continue

        categorical_features.append(column)

    return numerical_features, categorical_features


def _infer_num_classes(csv_path: Path, target_column: str, fallback: int | None) -> int:
    if fallback is not None:
        return int(fallback)

    unique_values: set[str] = set()
    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            value = row.get(target_column)
            if value not in {None, ""}:
                unique_values.add(str(value))
    return max(len(unique_values), 2)


def _infer_image_dir(
    dataset_path: Path | None,
    csv_path: Path,
    sample_rows: list[dict[str, str]],
    image_column: str,
    explicit_image_dir: str | None,
) -> Path:
    if explicit_image_dir:
        resolved = _resolve_path(explicit_image_dir)
        if resolved is not None:
            return resolved

    sample_image_path = next(
        (
            row.get(image_column)
            for row in sample_rows
            if row.get(image_column) not in {None, ""}
        ),
        None,
    )
    if sample_image_path:
        sample_path = Path(sample_image_path)
        if sample_path.is_absolute():
            return sample_path.parent

    candidate_paths: list[Path] = []
    if dataset_path is not None:
        candidate_paths.extend(
            [
                dataset_path if dataset_path.is_dir() else dataset_path.parent,
                (dataset_path / "images") if dataset_path.is_dir() else dataset_path.parent / "images",
            ]
        )
    candidate_paths.extend([csv_path.parent, csv_path.parent / "images"])

    checked: set[Path] = set()
    for candidate in candidate_paths:
        resolved = candidate.resolve()
        if resolved in checked:
            continue
        checked.add(resolved)
        if sample_image_path and (resolved / sample_image_path).exists():
            return resolved

    if dataset_path is not None and dataset_path.is_dir():
        return dataset_path.resolve()
    return csv_path.parent.resolve()


def _resolve_dataset_spec(
    db: Session,
    dataset_config: dict[str, Any],
) -> dict[str, Any]:
    dataset_record: DatasetInfo | None = None
    dataset_id = dataset_config.get("dataset_id")
    if dataset_id is not None:
        try:
            dataset_record = (
                db.query(DatasetInfo)
                .filter(DatasetInfo.id == int(dataset_id))
                .first()
            )
        except (TypeError, ValueError):
            dataset_record = None

    dataset_path = _resolve_path(
        dataset_config.get("data_path")
        or (dataset_record.data_path if dataset_record else None)
    )
    csv_path = _infer_csv_path(dataset_path, dataset_config.get("csv_path"))
    headers, sample_rows = _read_csv_preview(csv_path)

    image_path_column = _pick_column(
        headers,
        preferred=dataset_config.get("image_path_column"),
        candidates=_IMAGE_COLUMN_CANDIDATES,
    )
    if image_path_column is None:
        raise HTTPException(status_code=400, detail=f"CSV 中缺少图像路径列: {csv_path}")

    target_column = _pick_column(
        headers,
        preferred=dataset_config.get("target_column"),
        candidates=_TARGET_COLUMN_CANDIDATES,
    )
    if target_column is None:
        raise HTTPException(status_code=400, detail=f"CSV 中缺少标签列: {csv_path}")

    patient_id_column = _pick_column(
        headers,
        preferred=dataset_config.get("patient_id_column"),
        candidates=_PATIENT_ID_COLUMN_CANDIDATES,
    )

    numerical_features = _coerce_list(dataset_config.get("numerical_features"))
    categorical_features = _coerce_list(dataset_config.get("categorical_features"))
    if not numerical_features and not categorical_features:
        numerical_features, categorical_features = _infer_feature_columns(
            headers,
            sample_rows,
            excluded_columns={
                image_path_column,
                target_column,
                patient_id_column or "",
            },
        )

    num_classes = _infer_num_classes(
        csv_path=csv_path,
        target_column=target_column,
        fallback=dataset_config.get("num_classes")
        or (dataset_record.num_classes if dataset_record else None),
    )

    image_dir = _infer_image_dir(
        dataset_path=dataset_path,
        csv_path=csv_path,
        sample_rows=sample_rows,
        image_column=image_path_column,
        explicit_image_dir=dataset_config.get("image_dir"),
    )

    return {
        "dataset_name": dataset_config.get("dataset")
        or dataset_config.get("dataset_name")
        or (dataset_record.name if dataset_record else None)
        or csv_path.stem,
        "csv_path": str(csv_path.resolve()),
        "image_dir": str(image_dir.resolve()),
        "image_path_column": image_path_column,
        "target_column": target_column,
        "patient_id_column": patient_id_column,
        "numerical_features": numerical_features,
        "categorical_features": categorical_features,
        "num_classes": num_classes,
        "data_path": str(dataset_path.resolve()) if dataset_path is not None else str(csv_path.parent.resolve()),
    }


def _build_training_config_artifact(
    db: Session,
    payload: TrainingConfig,
    output_dir: Path,
) -> dict[str, Any]:
    config = load_config(_PROJECT_ROOT / "configs" / "starter" / "quickstart.yaml")

    training_model_config = payload.training_model_config or {}
    dataset_config = payload.dataset_config or {}
    training_config = payload.training_config or {}
    dataset_spec = _resolve_dataset_spec(db, dataset_config)

    experiment_name = payload.experiment_name.strip() or f"training-{uuid.uuid4().hex[:8]}"
    backbone = _normalize_backbone_name(
        training_model_config.get("backbone") or config.model.vision.backbone
    )
    if backbone:
        config.model.vision.backbone = backbone
    config.model.vision.pretrained = bool(training_model_config.get("pretrained", False))
    config.model.vision.freeze_backbone = bool(
        training_model_config.get("freeze_backbone", False)
    )

    config.project_name = "web-training"
    config.experiment_name = experiment_name
    config.logging.experiment_name = experiment_name
    config.logging.output_dir = str(output_dir.resolve())
    config.logging.use_tensorboard = bool(training_config.get("use_tensorboard", False))
    config.logging.use_wandb = False

    config.data.csv_path = dataset_spec["csv_path"]
    config.data.image_dir = dataset_spec["image_dir"]
    config.data.image_path_column = dataset_spec["image_path_column"]
    config.data.target_column = dataset_spec["target_column"]
    config.data.patient_id_column = dataset_spec["patient_id_column"]
    config.data.numerical_features = dataset_spec["numerical_features"]
    config.data.categorical_features = dataset_spec["categorical_features"]
    config.data.batch_size = int(
        training_config.get("batch_size")
        or training_config.get("batchSize")
        or config.data.batch_size
    )
    config.data.num_workers = int(
        training_config.get("num_workers")
        or training_config.get("numWorkers")
        or 0
    )
    config.data.pin_memory = bool(training_config.get("pin_memory", False))
    if training_config.get("image_size") is not None:
        config.data.image_size = int(training_config["image_size"])

    resolved_num_classes = int(
        training_model_config.get("num_classes")
        or dataset_spec["num_classes"]
        or config.model.num_classes
    )
    config.model.num_classes = resolved_num_classes

    config.training.num_epochs = int(
        training_config.get("epochs")
        or training_config.get("num_epochs")
        or training_config.get("numEpochs")
        or config.training.num_epochs
    )
    config.training.use_progressive_training = bool(
        training_config.get("use_progressive_training", False)
    )
    config.training.mixed_precision = bool(training_config.get("mixed_precision", False))
    config.training.monitor = str(training_config.get("monitor") or "accuracy")
    config.training.mode = str(
        training_config.get("mode")
        or ("min" if config.training.monitor == "loss" else "max")
    )
    config.training.optimizer.optimizer = str(
        training_config.get("optimizer")
        or config.training.optimizer.optimizer
    )
    config.training.optimizer.learning_rate = float(
        training_config.get("learning_rate")
        or training_config.get("learningRate")
        or config.training.optimizer.learning_rate
    )
    if training_config.get("weight_decay") is not None:
        config.training.optimizer.weight_decay = float(training_config["weight_decay"])
    if training_config.get("scheduler") is not None:
        config.training.scheduler.scheduler = str(training_config["scheduler"])
    if training_config.get("step_size") is not None:
        config.training.scheduler.step_size = int(training_config["step_size"])
    if training_config.get("patience") is not None:
        config.training.scheduler.patience = int(training_config["patience"])

    if training_model_config.get("fusion_type") is not None:
        config.model.fusion.fusion_type = str(training_model_config["fusion_type"])
    if training_model_config.get("feature_dim") is not None:
        config.model.vision.feature_dim = int(training_model_config["feature_dim"])
    if training_model_config.get("tabular_output_dim") is not None:
        config.model.tabular.output_dim = int(training_model_config["tabular_output_dim"])
    if config.model.fusion.fusion_type == "concatenate":
        config.model.fusion.hidden_dim = (
            config.model.vision.feature_dim + config.model.tabular.output_dim
        )

    config_path = output_dir / "training-config.yaml"
    save_config(config, config_path)

    return {
        "config_path": str(config_path.resolve()),
        "dataset_spec": dataset_spec,
        "backbone": config.model.vision.backbone,
        "num_classes": resolved_num_classes,
        "total_epochs": config.training.num_epochs,
    }
def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        logger.warning("Failed to parse JSON artifact: %s", path)
        return {}


def _build_train_command(config_path: Path, output_dir: Path) -> list[str]:
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


def _signal_process(job_id: str, process_signal: signal.Signals) -> bool:
    process = _training_processes.get(job_id)
    if process is None or process.poll() is not None:
        return False

    try:
        os.killpg(process.pid, process_signal)
    except ProcessLookupError:
        return False

    return True


def _resolve_checkpoint_path(output_dir: Path) -> Path | None:
    checkpoint_dir = output_dir / "checkpoints"
    preferred_candidates = [
        checkpoint_dir / "best.pth",
        checkpoint_dir / "last.pth",
    ]
    for candidate in preferred_candidates:
        if candidate.exists():
            return candidate
    checkpoints = sorted(checkpoint_dir.glob("*.pth"))
    return checkpoints[0] if checkpoints else None


def _read_history_entries(job: TrainingJob) -> list[dict[str, Any]]:
    history_payload = _read_json(_history_path_for_job(job))
    entries = history_payload.get("entries", [])
    return entries if isinstance(entries, list) else []


def _sync_job_from_history(job: TrainingJob) -> dict[str, Any] | None:
    entries = _read_history_entries(job)
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
        best_accuracy_epoch, best_accuracy = max(val_accuracies, key=lambda item: item[1])
        job.best_accuracy = best_accuracy
        job.best_epoch = best_accuracy_epoch

    return latest_entry


def _tail_log(log_path: Path | None, max_lines: int = 40) -> str | None:
    if log_path is None or not log_path.exists():
        return None
    lines = log_path.read_text(encoding="utf-8").splitlines()
    if not lines:
        return None
    return "\n".join(lines[-max_lines:])


def _history_path_for_job(job: TrainingJob) -> Path:
    output_dir = Path(job.output_dir or settings.data_dir / "experiments" / job.job_id)
    return output_dir / "history.json"

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


async def _run_real_training_job(job_id: str) -> None:
    """Run a real training process in the background and sync its artifacts."""
    db = SessionLocal()
    process: subprocess.Popen[str] | None = None
    log_handle: Any | None = None
    try:
        job = _get_job_or_404(db, job_id)
        resolved_run = (job.config or {}).get("resolved_run", {})
        config_path = Path(resolved_run["config_path"])
        output_dir = Path(job.output_dir or settings.data_dir / "experiments" / job.job_id)
        log_path = Path(job.log_file) if job.log_file else output_dir / "training.log"
        command = _build_train_command(config_path, output_dir)

        job.status = "running"
        job.started_at = job.started_at or utcnow()
        job.error_message = None
        db.commit()

        log_handle = log_path.open("a", encoding="utf-8", buffering=1)
        log_handle.write(f"command: {' '.join(shlex.quote(part) for part in command)}\n")
        env = os.environ.copy()
        env.setdefault("PYTHONUNBUFFERED", "1")

        process = subprocess.Popen(
            command,
            cwd=str(_PROJECT_ROOT),
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
            start_new_session=True,
        )
        _training_processes[job_id] = process

        while True:
            await asyncio.sleep(1.0)
            db.expire_all()
            job = _get_job_or_404(db, job_id)
            _sync_job_from_history(job)
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
                job.error_message = _tail_log(log_path) or f"训练进程退出码: {return_code}"
                db.commit()
                return

            checkpoint_path = _resolve_checkpoint_path(output_dir)
            if checkpoint_path is None:
                raise FileNotFoundError(f"训练完成但未找到 checkpoint: {output_dir / 'checkpoints'}")

            metadata = _extract_job_metadata(job)
            import_model_run(
                db=db,
                config_path=config_path,
                checkpoint_path=checkpoint_path,
                output_dir=output_dir,
                split="test",
                attention_samples=4,
                name=f"{metadata['experiment_name']}-model",
                description="Web UI 触发的真实训练产物",
                tags=["web-ui", f"job:{job.job_id}"],
                import_source="web",
            )

            _sync_job_from_history(job)
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
            _signal_process(job_id, signal.SIGTERM)
        if log_handle is not None:
            log_handle.close()
        db.close()
        _training_processes.pop(job_id, None)
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
        output_dir, log_file = _prepare_job_output(job_id)
        resolved_run = _build_training_config_artifact(
            db=db,
            payload=config,
            output_dir=Path(output_dir),
        )
        total_epochs = int(resolved_run["total_epochs"])

        job_payload = config.model_dump()
        job_payload.setdefault("training_model_config", {})
        job_payload.setdefault("dataset_config", {})
        job_payload["training_model_config"]["backbone"] = resolved_run["backbone"]
        job_payload["training_model_config"]["num_classes"] = resolved_run["num_classes"]
        job_payload["dataset_config"].update(
            {
                "dataset": resolved_run["dataset_spec"]["dataset_name"],
                "data_path": resolved_run["dataset_spec"]["data_path"],
                "csv_path": resolved_run["dataset_spec"]["csv_path"],
                "image_dir": resolved_run["dataset_spec"]["image_dir"],
                "image_path_column": resolved_run["dataset_spec"]["image_path_column"],
                "target_column": resolved_run["dataset_spec"]["target_column"],
                "patient_id_column": resolved_run["dataset_spec"]["patient_id_column"],
                "numerical_features": resolved_run["dataset_spec"]["numerical_features"],
                "categorical_features": resolved_run["dataset_spec"]["categorical_features"],
                "num_classes": resolved_run["dataset_spec"]["num_classes"],
            }
        )
        job_payload["resolved_run"] = resolved_run

        job = TrainingJob(
            job_id=job_id,
            config=job_payload,
            total_epochs=total_epochs,
            status="running",
            progress=0.0,
            current_epoch=0,
            created_at=utcnow(),
            started_at=utcnow(),
            current_loss=None,
            current_accuracy=None,
            output_dir=output_dir,
            log_file=log_file,
        )
        db.add(job)
        db.commit()
        db.refresh(job)

        _pause_flags[job_id] = False
        _stop_flags[job_id] = False
        _training_tasks[job_id] = asyncio.create_task(_run_real_training_job(job_id))

        logger.info(f"训练任务已创建: {job_id}")
        return {"job_id": job_id, "status": "running", "message": "训练任务已启动"}
    except HTTPException:
        raise
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
            experiment_name=_extract_job_metadata(job)["experiment_name"],
            dataset_name=_extract_job_metadata(job)["dataset_name"],
            backbone=_extract_job_metadata(job)["backbone"],
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
    metadata = _extract_job_metadata(job)
    latest_entry = _sync_job_from_history(job)
    db.commit()
    return {
        "job_id": job.job_id,
        "experiment_name": metadata["experiment_name"],
        "dataset_name": metadata["dataset_name"],
        "backbone": metadata["backbone"],
        "status": job.status,
        "progress": job.progress,
        "current_epoch": job.current_epoch,
        "total_epochs": job.total_epochs,
        "current_loss": job.current_loss,
        "current_accuracy": job.current_accuracy,
        "current_lr": job.current_lr,
        "best_loss": job.best_loss,
        "best_accuracy": job.best_accuracy,
        "gpu_usage": job.gpu_usage,
        "gpu_memory": job.gpu_memory,
        "error_message": job.error_message,
        "latest_history": latest_entry,
    }


@router.get("/{job_id}/history")
async def get_training_history(
    job_id: str,
    db: Session = Depends(get_db_session),
) -> dict[str, Any]:
    """获取训练历史，用于真实曲线回放。"""
    job = _get_job_or_404(db, job_id)
    return {
        "job_id": job.job_id,
        "entries": _read_history_entries(job),
        "total_epochs": job.total_epochs,
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

    if not _signal_process(job_id, signal.SIGSTOP):
        raise HTTPException(status_code=400, detail="训练进程未运行，无法暂停")
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

    if not _signal_process(job_id, signal.SIGCONT):
        raise HTTPException(status_code=400, detail="训练进程未运行，无法恢复")
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

    _signal_process(job_id, signal.SIGCONT)
    _signal_process(job_id, signal.SIGTERM)
    job.status = "stopped"
    job.completed_at = utcnow()
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
                    _signal_process(job_id, signal.SIGSTOP)
                elif action == "resume":
                    _signal_process(job_id, signal.SIGCONT)
                elif action == "stop":
                    _signal_process(job_id, signal.SIGCONT)
                    _signal_process(job_id, signal.SIGTERM)
            except TimeoutError:
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

            metadata = _extract_job_metadata(job)
            latest_entry = _sync_job_from_history(job)
            db.commit()
            await websocket.send_json(
                {
                    "type": "status_update",
                    "job_id": job.job_id,
                    "experiment_name": metadata["experiment_name"],
                    "status": job.status,
                    "progress": job.progress,
                    "epoch": job.current_epoch,
                    "total_epochs": job.total_epochs,
                    "loss": job.current_loss,
                    "accuracy": job.current_accuracy,
                    "train_loss": latest_entry.get("train_loss") if latest_entry else None,
                    "val_loss": latest_entry.get("val_loss") if latest_entry else None,
                    "train_accuracy": latest_entry.get("train_accuracy") if latest_entry else None,
                    "val_accuracy": latest_entry.get("val_accuracy") if latest_entry else None,
                    "learning_rate": latest_entry.get("learning_rate") if latest_entry else None,
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
