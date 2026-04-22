"""Tests for web backup helpers."""

from __future__ import annotations

import tarfile
from pathlib import Path

from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from med_core.web.backup import (
    create_backup_archive,
    discover_backup_contents,
    export_database_snapshot,
    load_database_snapshot,
    restore_database_snapshot,
)
from med_core.web.database import Base
from med_core.web.models.dataset_info import DatasetInfo
from med_core.web.models.experiment import Experiment
from med_core.web.models.training_job import TrainingJob


def _sqlite_engine(tmp_path: Path):
    db_path = tmp_path / "backup-test.db"
    test_engine = create_engine(
        f"sqlite:///{db_path}",
        connect_args={"check_same_thread": False},
    )
    Base.metadata.create_all(bind=test_engine)
    return test_engine


def test_database_snapshot_export_and_restore_roundtrip(tmp_path: Path) -> None:
    test_engine = _sqlite_engine(tmp_path)
    with Session(test_engine) as session:
        exp = Experiment(
            name="exp-roundtrip",
            description="snapshot",
            config={"lr": 0.001},
            status="created",
        )
        session.add(exp)
        session.flush()
        session.add(
            TrainingJob(
                job_id="job-roundtrip-1",
                experiment_id=exp.id,
                config={"epochs": 10},
                total_epochs=10,
                status="queued",
            )
        )
        session.add(
            DatasetInfo(
                name="dataset-roundtrip",
                data_path="/tmp/dataset.csv",
                dataset_type="image",
                status="ready",
            )
        )
        session.commit()

    snapshot = export_database_snapshot(db_engine=test_engine)
    assert snapshot["row_counts"]["experiments"] == 1
    assert snapshot["row_counts"]["training_jobs"] == 1
    assert snapshot["row_counts"]["dataset_info"] == 1

    with Session(test_engine) as session:
        session.add(
            Experiment(
                name="exp-extra",
                description="extra",
                config={"lr": 0.01},
                status="created",
            )
        )
        session.commit()

    restored_counts = restore_database_snapshot(
        snapshot,
        db_engine=test_engine,
        truncate_existing=True,
    )
    assert restored_counts["experiments"] == 1
    assert restored_counts["training_jobs"] == 1
    assert restored_counts["dataset_info"] == 1

    with Session(test_engine) as session:
        experiments = session.execute(select(Experiment)).scalars().all()
        training_jobs = session.execute(select(TrainingJob)).scalars().all()
        datasets = session.execute(select(DatasetInfo)).scalars().all()

    assert len(experiments) == 1
    assert experiments[0].name == "exp-roundtrip"
    assert len(training_jobs) == 1
    assert training_jobs[0].job_id == "job-roundtrip-1"
    assert len(datasets) == 1
    assert datasets[0].name == "dataset-roundtrip"


def test_create_backup_archive_includes_manifest_data_and_db_snapshot(
    tmp_path: Path,
) -> None:
    source_data_dir = tmp_path / "source-data"
    (source_data_dir / "logs").mkdir(parents=True)
    (source_data_dir / "logs" / "run.log").write_text("ok", encoding="utf-8")

    test_engine = _sqlite_engine(tmp_path)
    with Session(test_engine) as session:
        session.add(
            Experiment(
                name="exp-backup",
                description="archive",
                config={"batch": 8},
                status="created",
            )
        )
        session.commit()

    archive_path = create_backup_archive(
        tmp_path / "medfusion-backup",
        include_db_snapshot=True,
        source_data_dir=source_data_dir,
        db_engine=test_engine,
    )
    assert archive_path.exists()
    assert archive_path.name.endswith(".tar.gz")

    extract_root = tmp_path / "extract"
    extract_root.mkdir(parents=True)
    with tarfile.open(archive_path, "r:*") as tar:
        try:
            tar.extractall(path=extract_root, filter="data")
        except TypeError:
            tar.extractall(path=extract_root)

    data_root, manifest, db_snapshot_path = discover_backup_contents(extract_root)
    assert manifest is not None
    assert manifest["schema_version"] == "1.0"
    assert (data_root / "logs" / "run.log").read_text(encoding="utf-8") == "ok"
    assert db_snapshot_path is not None

    snapshot = load_database_snapshot(db_snapshot_path)
    assert snapshot["row_counts"]["experiments"] == 1
    assert "tables" in snapshot
