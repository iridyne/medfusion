"""Application-layer training job orchestration for the web API.

This module keeps the FastAPI layer thin. It owns job creation, config
materialization, training control transitions, and websocket status payloads
while preserving the current external route contracts.
"""

from __future__ import annotations

import copy
import csv
import logging
import signal
import uuid
from pathlib import Path
from typing import Any, Callable

from sqlalchemy.orm import Session

from med_core.configs import load_config, save_config
from med_core.output_layout import RunOutputLayout

from ..models import DatasetInfo, TrainingJob
from ..time_utils import utcnow

logger = logging.getLogger(__name__)

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


class TrainingJobRequestError(RuntimeError):
    """Application-layer request error that maps cleanly to HTTP responses."""

    def __init__(self, status_code: int, detail: str) -> None:
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail


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


def _resolve_path(project_root: Path, path_value: str | Path | None) -> Path | None:
    if not path_value:
        return None
    path = Path(path_value).expanduser()
    if path.is_absolute():
        return path

    candidates = [
        project_root / path,
        Path.cwd() / path,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()

    return (project_root / path).resolve()


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


def _infer_csv_path(
    *,
    project_root: Path,
    dataset_path: Path | None,
    explicit_csv_path: str | None,
) -> Path:
    if explicit_csv_path:
        resolved = _resolve_path(project_root, explicit_csv_path)
        if resolved is not None and resolved.exists():
            return resolved
        raise TrainingJobRequestError(400, f"CSV 文件不存在: {explicit_csv_path}")

    if dataset_path is None:
        raise TrainingJobRequestError(400, "缺少 data_path 或 csv_path，无法生成真实训练配置")

    if dataset_path.is_file():
        if dataset_path.suffix.lower() != ".csv":
            raise TrainingJobRequestError(400, f"不支持的数据描述文件: {dataset_path}")
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

    raise TrainingJobRequestError(
        400,
        f"无法从 {dataset_path} 自动识别 CSV，请显式提供 csv_path",
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
            if all_integer_like and len(unique_values) <= min(
                10, max(2, len(values) // 2)
            ):
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
    *,
    project_root: Path,
    dataset_path: Path | None,
    csv_path: Path,
    sample_rows: list[dict[str, str]],
    image_column: str,
    explicit_image_dir: str | None,
) -> Path:
    if explicit_image_dir:
        resolved = _resolve_path(project_root, explicit_image_dir)
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
                (dataset_path / "images")
                if dataset_path.is_dir()
                else dataset_path.parent / "images",
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


def _build_job_payload(
    payload: dict[str, Any],
    resolved_run: dict[str, Any],
) -> dict[str, Any]:
    job_payload = copy.deepcopy(payload)
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
    return job_payload


class TrainingJobApplicationService:
    """Own training job lifecycle orchestration for the web app."""

    def __init__(
        self,
        *,
        project_root: Path,
        data_dir: Path,
        schedule_background_run: Callable[[str], None],
        signal_process: Callable[[str, signal.Signals | int], bool],
        pause_signal: signal.Signals | int,
        resume_signal: signal.Signals | int,
        terminate_signal: signal.Signals | int,
        get_job_or_404: Callable[[Session, str], TrainingJob],
        extract_job_metadata: Callable[[TrainingJob], dict[str, Any]],
        extract_result_handoff: Callable[[TrainingJob], dict[str, Any]],
        read_history_entries: Callable[[TrainingJob], list[dict[str, Any]]],
        sync_job_from_history: Callable[[TrainingJob], dict[str, Any] | None],
    ) -> None:
        self.project_root = project_root
        self.data_dir = data_dir
        self.schedule_background_run = schedule_background_run
        self.signal_process = signal_process
        self.pause_signal = pause_signal
        self.resume_signal = resume_signal
        self.terminate_signal = terminate_signal
        self.get_job_or_404 = get_job_or_404
        self.extract_job_metadata = extract_job_metadata
        self.extract_result_handoff = extract_result_handoff
        self.read_history_entries = read_history_entries
        self.sync_job_from_history = sync_job_from_history

    def prepare_job_output(self, job_id: str) -> tuple[str, str]:
        layout = RunOutputLayout(self.data_dir / "experiments" / job_id).ensure_exists()
        log_file = layout.training_log_path
        if not log_file.exists():
            log_file.write_text("training started\n", encoding="utf-8")
        return str(layout.root_dir), str(log_file)

    def _resolve_dataset_spec(
        self,
        db: Session,
        dataset_config: dict[str, Any],
    ) -> dict[str, Any]:
        dataset_record: DatasetInfo | None = None
        dataset_id = dataset_config.get("dataset_id")
        if dataset_id is not None:
            try:
                dataset_record = (
                    db.query(DatasetInfo).filter(DatasetInfo.id == int(dataset_id)).first()
                )
            except (TypeError, ValueError):
                dataset_record = None

        dataset_path = _resolve_path(
            self.project_root,
            dataset_config.get("data_path")
            or (dataset_record.data_path if dataset_record else None),
        )
        csv_path = _infer_csv_path(
            project_root=self.project_root,
            dataset_path=dataset_path,
            explicit_csv_path=dataset_config.get("csv_path"),
        )
        headers, sample_rows = _read_csv_preview(csv_path)

        image_path_column = _pick_column(
            headers,
            preferred=dataset_config.get("image_path_column"),
            candidates=_IMAGE_COLUMN_CANDIDATES,
        )
        if image_path_column is None:
            raise TrainingJobRequestError(400, f"CSV 中缺少图像路径列: {csv_path}")

        target_column = _pick_column(
            headers,
            preferred=dataset_config.get("target_column"),
            candidates=_TARGET_COLUMN_CANDIDATES,
        )
        if target_column is None:
            raise TrainingJobRequestError(400, f"CSV 中缺少标签列: {csv_path}")

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
            project_root=self.project_root,
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
            "data_path": str(dataset_path.resolve())
            if dataset_path is not None
            else str(csv_path.parent.resolve()),
        }

    def build_training_config_artifact(
        self,
        *,
        db: Session,
        payload: dict[str, Any],
        output_dir: Path,
    ) -> dict[str, Any]:
        layout = RunOutputLayout(output_dir).ensure_exists()
        config = load_config(self.project_root / "configs" / "starter" / "quickstart.yaml")

        training_model_config = payload.get("training_model_config") or {}
        dataset_config = payload.get("dataset_config") or {}
        training_config = payload.get("training_config") or {}
        dataset_spec = self._resolve_dataset_spec(db, dataset_config)

        experiment_name = (
            str(payload.get("experiment_name") or "").strip()
            or f"training-{uuid.uuid4().hex[:8]}"
        )
        backbone = _normalize_backbone_name(
            training_model_config.get("backbone") or config.model.vision.backbone
        )
        if backbone:
            config.model.vision.backbone = backbone
        config.model.vision.pretrained = bool(
            training_model_config.get("pretrained", False)
        )
        config.model.vision.freeze_backbone = bool(
            training_model_config.get("freeze_backbone", False)
        )

        config.project_name = "web-training"
        config.experiment_name = experiment_name
        config.logging.experiment_name = experiment_name
        config.logging.output_dir = str(layout.root_dir.resolve())
        config.logging.use_tensorboard = bool(
            training_config.get("use_tensorboard", False)
        )
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
        config.training.mixed_precision = bool(
            training_config.get("mixed_precision", False)
        )
        config.training.monitor = str(training_config.get("monitor") or "accuracy")
        config.training.mode = str(
            training_config.get("mode")
            or ("min" if config.training.monitor == "loss" else "max")
        )
        config.training.optimizer.optimizer = str(
            training_config.get("optimizer") or config.training.optimizer.optimizer
        )
        config.training.optimizer.learning_rate = float(
            training_config.get("learning_rate")
            or training_config.get("learningRate")
            or config.training.optimizer.learning_rate
        )
        if training_config.get("weight_decay") is not None:
            config.training.optimizer.weight_decay = float(
                training_config["weight_decay"]
            )
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
            config.model.tabular.output_dim = int(
                training_model_config["tabular_output_dim"]
            )
        if config.model.fusion.fusion_type == "concatenate":
            config.model.fusion.hidden_dim = (
                config.model.vision.feature_dim + config.model.tabular.output_dim
            )

        config_path = layout.generated_config_path
        save_config(config, config_path)

        return {
            "config_path": str(config_path.resolve()),
            "dataset_spec": dataset_spec,
            "backbone": config.model.vision.backbone,
            "num_classes": resolved_num_classes,
            "total_epochs": config.training.num_epochs,
        }

    def start_training(self, *, db: Session, payload: dict[str, Any]) -> dict[str, Any]:
        try:
            job_id = str(uuid.uuid4())
            output_dir, log_file = self.prepare_job_output(job_id)
            resolved_run = self.build_training_config_artifact(
                db=db,
                payload=payload,
                output_dir=Path(output_dir),
            )
            job = TrainingJob(
                job_id=job_id,
                config=_build_job_payload(payload, resolved_run),
                total_epochs=int(resolved_run["total_epochs"]),
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
            self.schedule_background_run(job_id)
            logger.info("训练任务已创建: %s", job_id)
            return {"job_id": job_id, "status": "running", "message": "训练任务已启动"}
        except Exception:
            db.rollback()
            raise

    def list_training_jobs(
        self,
        *,
        db: Session,
        skip: int = 0,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        jobs = (
            db.query(TrainingJob)
            .order_by(TrainingJob.created_at.desc())
            .offset(skip)
            .limit(limit)
            .all()
        )
        return [self._serialize_job_summary(job) for job in jobs]

    def get_training_status(self, *, db: Session, job_id: str) -> dict[str, Any]:
        job = self.get_job_or_404(db, job_id)
        latest_entry = self.sync_job_from_history(job)
        db.commit()
        return self._serialize_job_status(job, latest_entry)

    def get_training_history(self, *, db: Session, job_id: str) -> dict[str, Any]:
        job = self.get_job_or_404(db, job_id)
        return {
            "job_id": job.job_id,
            "entries": self.read_history_entries(job),
            "total_epochs": job.total_epochs,
        }

    def pause_training(self, *, db: Session, job_id: str) -> dict[str, str]:
        job = self.get_job_or_404(db, job_id)
        if job.status != "running":
            raise TrainingJobRequestError(400, "只能暂停正在运行的任务")
        if not self.signal_process(job_id, self.pause_signal):
            raise TrainingJobRequestError(400, "训练进程未运行，无法暂停")
        job.status = "paused"
        db.commit()
        return {"message": "训练任务已暂停"}

    def resume_training(self, *, db: Session, job_id: str) -> dict[str, str]:
        job = self.get_job_or_404(db, job_id)
        if job.status != "paused":
            raise TrainingJobRequestError(400, "只能恢复已暂停的任务")
        if not self.signal_process(job_id, self.resume_signal):
            raise TrainingJobRequestError(400, "训练进程未运行，无法恢复")
        job.status = "running"
        db.commit()
        return {"message": "训练任务已恢复"}

    def stop_training(self, *, db: Session, job_id: str) -> dict[str, str]:
        job = self.get_job_or_404(db, job_id)
        if job.status not in {"running", "paused", "queued"}:
            raise TrainingJobRequestError(400, "无法停止该任务")
        self.signal_process(job_id, self.resume_signal)
        self.signal_process(job_id, self.terminate_signal)
        job.status = "stopped"
        job.completed_at = utcnow()
        db.commit()
        return {"message": "训练任务已停止"}

    def dispatch_ws_control_signal(
        self,
        *,
        route_job_id: str,
        action: str | None,
        payload_job_id: str | None,
    ) -> bool:
        if action not in {"pause", "resume", "stop"}:
            return False

        if payload_job_id and payload_job_id != route_job_id:
            logger.warning(
                "Ignoring websocket control action for mismatched job: route=%s payload=%s action=%s",
                route_job_id,
                payload_job_id,
                action,
            )
            return False

        if action == "pause":
            return self.signal_process(route_job_id, self.pause_signal)
        if action == "resume":
            return self.signal_process(route_job_id, self.resume_signal)

        continued = self.signal_process(route_job_id, self.resume_signal)
        terminated = self.signal_process(route_job_id, self.terminate_signal)
        return continued or terminated

    def build_websocket_messages(
        self,
        *,
        db: Session,
        job_id: str,
    ) -> tuple[list[dict[str, Any]], bool]:
        if job_id == "all":
            return ([{"type": "heartbeat", "message": "ok"}], False)

        job = db.query(TrainingJob).filter(TrainingJob.job_id == job_id).first()
        if not job:
            return (
                [
                    {
                        "type": "error",
                        "message": "训练任务不存在",
                        "job_id": job_id,
                    }
                ],
                False,
            )

        latest_entry = self.sync_job_from_history(job)
        db.commit()
        result_handoff = self.extract_result_handoff(job)

        status_message = {
            "type": "status_update",
            "job_id": job.job_id,
            "experiment_name": self.extract_job_metadata(job)["experiment_name"],
            "status": job.status,
            "progress": job.progress,
            "epoch": job.current_epoch,
            "total_epochs": job.total_epochs,
            "result_model_id": result_handoff["result_model_id"],
            "result_model_name": result_handoff["result_model_name"],
            "loss": job.current_loss,
            "accuracy": job.current_accuracy,
            "train_loss": latest_entry.get("train_loss") if latest_entry else None,
            "val_loss": latest_entry.get("val_loss") if latest_entry else None,
            "train_accuracy": latest_entry.get("train_accuracy")
            if latest_entry
            else None,
            "val_accuracy": latest_entry.get("val_accuracy") if latest_entry else None,
            "learning_rate": latest_entry.get("learning_rate")
            if latest_entry
            else None,
        }

        if job.status not in {"completed", "failed", "stopped"}:
            return ([status_message], False)

        terminal_message = {
            "type": "training_complete" if job.status == "completed" else "error",
            "job_id": job.job_id,
            "result_model_id": result_handoff["result_model_id"],
            "result_model_name": result_handoff["result_model_name"],
            "message": job.error_message or job.status,
        }
        return ([status_message, terminal_message], True)

    def _serialize_job_summary(self, job: TrainingJob) -> dict[str, Any]:
        metadata = self.extract_job_metadata(job)
        result_handoff = self.extract_result_handoff(job)
        return {
            "id": job.id,
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
            "result_model_id": result_handoff["result_model_id"],
            "result_model_name": result_handoff["result_model_name"],
            "created_at": job.created_at.isoformat(),
        }

    def _serialize_job_status(
        self,
        job: TrainingJob,
        latest_entry: dict[str, Any] | None,
    ) -> dict[str, Any]:
        metadata = self.extract_job_metadata(job)
        result_handoff = self.extract_result_handoff(job)
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
            "result_model_id": result_handoff["result_model_id"],
            "result_model_name": result_handoff["result_model_name"],
            "gpu_usage": job.gpu_usage,
            "gpu_memory": job.gpu_memory,
            "error_message": job.error_message,
            "latest_history": latest_entry,
        }
