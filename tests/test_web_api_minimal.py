"""Minimal Web API contract tests for dataset/model/training routes."""

import time

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

        for _ in range(6):
            current_status = client.get(f"/api/training/{job_id}/status")
            assert current_status.status_code == 200
            if current_status.json()["status"] == "completed":
                break
            time.sleep(0.6)
        else:
            raise AssertionError("training job did not complete in expected time")

        refreshed_models = client.get("/api/models/")
        assert refreshed_models.status_code == 200
        generated_model = next(
            item for item in refreshed_models.json() if item["name"] == "ci-exp-model"
        )
        assert generated_model["validation"]["overview"]["sample_count"] > 0
        assert len(generated_model["validation"]["per_class"]) == 2
        assert generated_model["metrics"]["balanced_accuracy"] >= 0
        assert any(
            artifact["key"] == "validation" for artifact in generated_model["result_files"]
        )

        generated_model_detail = client.get(f"/api/models/{generated_model['id']}")
        assert generated_model_detail.status_code == 200
        detail_payload = generated_model_detail.json()
        assert detail_payload["validation"]["prediction_summary"]["error_count"] >= 0
        assert detail_payload["validation"]["dataset"]["num_classes"] == 2

        # Cleanup
        deleted_model = client.delete(f"/api/models/{model_id}")
        assert deleted_model.status_code == 200

        deleted_generated_model = client.delete(f"/api/models/{generated_model['id']}")
        assert deleted_generated_model.status_code == 200

        deleted_dataset = client.delete(f"/api/datasets/{dataset_id}")
        assert deleted_dataset.status_code == 200
