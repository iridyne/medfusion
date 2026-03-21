"""Minimal Web API contract tests for dataset/model/training routes."""

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from med_core.web.app import app


def test_web_basic_routes() -> None:
    with TestClient(app) as client:
        # Health
        health = client.get("/health")
        assert health.status_code == 200
        assert health.json()["status"] == "healthy"

        # Dataset CRUD + analyze
        created_dataset = client.post(
            "/api/datasets/",
            json={
                "name": "ci-dataset",
                "data_path": "/tmp/ci-dataset",
                "dataset_type": "image",
                "num_samples": 12,
                "num_classes": 2,
            },
        )
        assert created_dataset.status_code == 200
        dataset_id = created_dataset.json()["id"]

        dataset_list = client.get("/api/datasets/")
        assert dataset_list.status_code == 200
        assert isinstance(dataset_list.json(), list)

        dataset_stats = client.get("/api/datasets/statistics")
        assert dataset_stats.status_code == 200
        assert "total_datasets" in dataset_stats.json()

        dataset_analysis = client.post(f"/api/datasets/{dataset_id}/analyze")
        assert dataset_analysis.status_code == 200

        # Model CRUD
        created_model = client.post(
            "/api/models/",
            json={
                "name": "ci-model",
                "backbone": "resnet18",
                "num_classes": 2,
                "accuracy": 0.88,
            },
        )
        assert created_model.status_code == 200
        model_id = created_model.json()["id"]

        model_list = client.get("/api/models/")
        assert model_list.status_code == 200
        assert isinstance(model_list.json(), list)

        model_stats = client.get("/api/models/statistics")
        assert model_stats.status_code == 200
        assert "total_models" in model_stats.json()

        # Training lifecycle (start + status)
        started_job = client.post(
            "/api/training/start",
            json={
                "experiment_name": "ci-exp",
                "training_model_config": {"backbone": "resnet18"},
                "dataset_config": {"dataset": "ci-dataset"},
                "training_config": {"epochs": 2},
            },
        )
        assert started_job.status_code == 200
        job_id = started_job.json()["job_id"]

        job_status = client.get(f"/api/training/{job_id}/status")
        assert job_status.status_code == 200
        assert job_status.json()["job_id"] == job_id

        # Cleanup
        deleted_model = client.delete(f"/api/models/{model_id}")
        assert deleted_model.status_code == 200

        deleted_dataset = client.delete(f"/api/datasets/{dataset_id}")
        assert deleted_dataset.status_code == 200
