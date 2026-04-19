"""Tests for the training job application service."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest

os.environ.setdefault(
    "MEDFUSION_DATA_DIR", tempfile.mkdtemp(prefix="medfusion-training-job-app-test-")
)

from med_core.web.application.training_jobs import TrainingJobApplicationService
from med_core.web.application.training_runtime import (
    extract_job_metadata,
    extract_result_handoff,
    get_job_or_404,
    read_history_entries,
    sync_job_from_history,
)
from med_core.web.database import SessionLocal, init_db
from med_core.web.models import TrainingJob


@pytest.fixture(scope="module", autouse=True)
def _prepare_web_storage() -> None:
    init_db()


def test_service_start_training_creates_job_and_schedules_background_run(
    tmp_path: Path,
) -> None:
    scheduled_job_ids: list[str] = []
    repo_root = Path(__file__).resolve().parents[1]
    mock_dataset_path = repo_root / "data" / "mock"

    service = TrainingJobApplicationService(
        project_root=repo_root,
        data_dir=tmp_path,
        schedule_background_run=scheduled_job_ids.append,
        signal_process=lambda *_args, **_kwargs: True,
        pause_signal=19,
        resume_signal=18,
        terminate_signal=15,
        get_job_or_404=get_job_or_404,
        extract_job_metadata=extract_job_metadata,
        extract_result_handoff=extract_result_handoff,
        read_history_entries=lambda job: read_history_entries(job, tmp_path),
        sync_job_from_history=lambda job: sync_job_from_history(
            job, default_data_dir=tmp_path
        ),
    )

    payload = {
        "experiment_name": "service-start-job",
        "training_model_config": {
            "backbone": "mobilenetv2",
            "num_classes": 2,
            "pretrained": False,
        },
        "dataset_config": {
            "dataset": "mock-dataset",
            "data_path": str(mock_dataset_path),
            "num_classes": 2,
        },
        "training_config": {
            "epochs": 3,
            "batch_size": 4,
            "learning_rate": 0.001,
            "image_size": 64,
            "num_workers": 0,
            "mixed_precision": False,
        },
    }

    db = SessionLocal()
    try:
        response = service.start_training(db=db, payload=payload)
        assert response["status"] == "running"
        assert response["message"] == "训练任务已启动"
        assert scheduled_job_ids == [response["job_id"]]

        job = db.query(TrainingJob).filter(TrainingJob.job_id == response["job_id"]).first()
        assert job is not None
        assert job.status == "running"
        assert job.total_epochs == 3
        assert job.output_dir == str(tmp_path / "experiments" / response["job_id"])
        assert job.log_file == str(
            tmp_path / "experiments" / response["job_id"] / "logs" / "training.log"
        )
        assert Path(job.log_file).exists()
        assert Path(job.config["resolved_run"]["config_path"]).exists()
        assert job.config["dataset_config"]["csv_path"].endswith("metadata.csv")
        assert job.config["training_model_config"]["backbone"] == "mobilenetv2"
    finally:
        db.close()
