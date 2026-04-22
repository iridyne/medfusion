"""Minimal Web API contract tests for dataset/model/training routes."""

import asyncio
import os
import tempfile
from pathlib import Path

import httpx
import pytest

pytest.importorskip("fastapi")

os.environ.setdefault(
    "MEDFUSION_DATA_DIR", tempfile.mkdtemp(prefix="medfusion-web-test-")
)

from test_build_results import _create_checkpoint_and_logs

from med_core.web.app import app
from med_core.web.config import settings
from med_core.web.database import init_db


@pytest.fixture(scope="module", autouse=True)
def _prepare_web_storage() -> None:
    settings.initialize_directories()
    init_db()


@pytest.fixture
async def api_client():
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://testserver",
    ) as client:
        yield client


async def test_web_basic_routes(api_client) -> None:
    mock_dataset_path = Path(__file__).resolve().parents[1] / "data" / "mock"

    # Health
    health = await api_client.get("/health")
    assert health.status_code == 200
    assert health.json()["status"] == "healthy"

    ui_preferences = await api_client.get("/api/system/preferences")
    assert ui_preferences.status_code == 200
    assert ui_preferences.json()["preferences"]["history_display_mode"] in {
        "friendly",
        "technical",
    }
    assert ui_preferences.json()["history_display_scope"] == "custom_model_history_only"
    assert ui_preferences.json()["preferences"]["language"] in {"zh", "en"}
    assert ui_preferences.json()["preferences"]["theme_mode"] in {"light", "dark", "auto"}
    assert ui_preferences.json()["storage"] == "filesystem"
    assert ui_preferences.json()["path"].endswith("settings\\ui-preferences.json")
    update_ui_preferences = await api_client.put(
        "/api/system/preferences",
        json={
            "history_display_mode": "technical",
            "language": "en",
            "theme_mode": "light",
        },
    )
    assert update_ui_preferences.status_code == 200
    assert (
        update_ui_preferences.json()["preferences"]["history_display_mode"]
        == "technical"
    )
    assert update_ui_preferences.json()["history_display_scope"] == "custom_model_history_only"
    assert update_ui_preferences.json()["preferences"]["language"] == "en"
    assert update_ui_preferences.json()["preferences"]["theme_mode"] == "light"
    reset_ui_preferences = await api_client.delete("/api/system/preferences")
    assert reset_ui_preferences.status_code == 200
    assert reset_ui_preferences.json()["preferences"]["history_display_mode"] == "friendly"
    assert reset_ui_preferences.json()["preferences"]["language"] == "zh"
    assert reset_ui_preferences.json()["preferences"]["theme_mode"] == "auto"

    # Dataset CRUD + analyze
    created_dataset = await api_client.post(
        "/api/datasets/",
        json={
            "name": "ci-dataset",
            "data_path": str(mock_dataset_path),
            "dataset_type": "multimodal",
            "num_samples": 30,
            "num_classes": 2,
        },
    )
    assert created_dataset.status_code == 200
    dataset_id = created_dataset.json()["id"]

    dataset_list = await api_client.get("/api/datasets/")
    assert dataset_list.status_code == 200
    assert isinstance(dataset_list.json(), list)

    dataset_stats = await api_client.get("/api/datasets/statistics")
    assert dataset_stats.status_code == 200
    assert "total_datasets" in dataset_stats.json()

    dataset_inspect = await api_client.post(
        "/api/datasets/inspect",
        json={
            "data_path": str(mock_dataset_path),
            "dataset_type": "multimodal",
        },
    )
    assert dataset_inspect.status_code == 200
    inspect_payload = dataset_inspect.json()
    assert inspect_payload["path"]["exists"] is True
    assert inspect_payload["readiness"]["can_enter_training"] is True
    assert inspect_payload["schema"]["target_column"] is not None

    dataset_analysis = await api_client.post(f"/api/datasets/{dataset_id}/analyze")
    assert dataset_analysis.status_code == 200
    assert dataset_analysis.json()["analysis"]["readiness"]["can_enter_training"] is True

    dataset_readiness = await api_client.get(f"/api/datasets/{dataset_id}/readiness")
    assert dataset_readiness.status_code == 200
    readiness_payload = dataset_readiness.json()
    assert readiness_payload["dataset_name"] == "ci-dataset"
    assert readiness_payload["readiness"]["status"] in {"ready", "warning"}
    assert readiness_payload["csv"]["path"]
    assert readiness_payload["schema"]["image_path_column"] is not None

    # Model CRUD
    created_model = await api_client.post(
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

    model_list = await api_client.get("/api/models/")
    assert model_list.status_code == 200
    assert isinstance(model_list.json(), list)

    model_stats = await api_client.get("/api/models/statistics")
    assert model_stats.status_code == 200
    assert "total_models" in model_stats.json()

    model_catalog = await api_client.get("/api/models/catalog")
    assert model_catalog.status_code == 200
    model_catalog_payload = model_catalog.json()
    assert model_catalog_payload["sources"]["official"]["enabled"] is True
    assert model_catalog_payload["sources"]["custom"]["enabled"] is True
    assert (
        model_catalog_payload["advanced_builder"]["family_projection"]["data_bundle"][
            "advanced_family"
        ]
        == "data_input"
    )
    assert "training_strategy" in model_catalog_payload["advanced_builder"]["required_families"]
    assert any(
        rule["from_family"] == "fusion" and rule["to_family"] == "head"
        for rule in model_catalog_payload["advanced_builder"]["connection_rules"]
    )
    assert any(
        item["id"] == "quickstart_multimodal"
        for item in model_catalog_payload["templates"]
    )
    quickstart_template = next(
        item
        for item in model_catalog_payload["templates"]
        if item["id"] == "quickstart_multimodal"
    )
    assert (
        quickstart_template["advanced_builder_contract"]["recommended_preset"]
        == "quickstart"
    )
    attention_component = next(
        item
        for item in model_catalog_payload["components"]
        if item["id"] == "attention_cbam_encoder_bundle"
    )
    assert (
        attention_component["advanced_builder_contract"]["compile_boundary"]
        == "conditional_attention_path"
    )
    assert (
        attention_component["advanced_builder_contract"]["warning_metadata"][0]["code"]
        == "ABG-W001"
    )
    stable_advanced_components = [
        item
        for item in model_catalog_payload["components"]
        if item.get("advanced_builder_component_id")
        and item.get("status") in {"compile_ready", "conditional"}
    ]
    assert stable_advanced_components
    for component in stable_advanced_components:
        patch_contract = component["advanced_builder_contract"].get("patch_contract")
        assert isinstance(patch_contract, list)
        assert patch_contract, f"missing patch_contract for {component['id']}"
    progressive_component = next(
        item for item in model_catalog_payload["components"] if item["id"] == "progressive_training_bundle"
    )
    assert any(
        operation["op"] == "derive_sum"
        and operation["path"] == "training.numEpochs"
        for operation in progressive_component["advanced_builder_contract"]["patch_contract"]
    )
    assert any(
        item["id"] == "resnet18_encoder_bundle"
        for item in model_catalog_payload["components"]
    )

    custom_model = {
        "schema_version": "0.1",
        "id": "custom-ci-model",
        "source": "custom",
        "label": "CI Custom Model",
        "description": "filesystem-backed custom model",
        "status": "local_custom",
        "based_on_model_id": "quickstart_multimodal",
        "unit_map": {
            "vision_encoder": "resnet18_encoder_bundle",
            "tabular_encoder": "mlp_tabular_encoder_bundle",
            "fusion_bundle": "concatenate_fusion_bundle",
            "task_head": "classification_head_bundle",
            "training_strategy": "standard_training_bundle",
        },
        "editable_slots": [
            "vision_encoder",
            "tabular_encoder",
            "fusion_bundle",
            "task_head",
            "training_strategy",
        ],
        "component_ids": [
            "resnet18_encoder_bundle",
            "mlp_tabular_encoder_bundle",
            "concatenate_fusion_bundle",
            "classification_head_bundle",
            "standard_training_bundle",
        ],
        "data_requirements": ["CSV + image_dir", "至少 1 个表格特征"],
        "compute_profile": {
            "tier": "light",
            "gpu_vram_hint": "8GB+",
            "notes": "ci custom model",
        },
        "wizard_prefill": {
            "modelTemplateId": "quickstart_multimodal",
            "customModelLabel": "CI Custom Model",
        },
        "created_at": "2026-04-21T00:00:00",
        "updated_at": "2026-04-21T00:00:00",
    }
    save_custom = await api_client.post("/api/models/custom", json=custom_model)
    assert save_custom.status_code == 200
    assert save_custom.json()["schema_version"] == "0.1"
    assert save_custom.json()["history_backend"] in {"git", "none"}
    listed_custom = await api_client.get("/api/models/custom")
    assert listed_custom.status_code == 200
    assert listed_custom.json()["schema_version"] == "0.1"
    assert listed_custom.json()["format_contract"] == "internal_state_file"
    assert listed_custom.json()["retention_policy"]["mode"] == "count"
    assert listed_custom.json()["retention_policy"]["max_count"] == 40
    assert listed_custom.json()["retention_policy"]["min_count_per_model"] == 3
    assert listed_custom.json()["retention_floor_scope"] == "active_models_only"
    assert any(item["id"] == "custom-ci-model" for item in listed_custom.json()["items"])

    update_policy = await api_client.put(
        "/api/models/custom/policy",
        json={
            "mode": "count",
            "max_count": 40,
            "max_age_days": 90,
            "min_count_per_model": 3,
        },
    )
    assert update_policy.status_code == 200
    assert update_policy.json()["policy"]["max_count"] == 40
    assert update_policy.json()["policy"]["min_count_per_model"] == 3
    assert update_policy.json()["retention_floor_scope"] == "active_models_only"

    updated_custom_model = {
        **custom_model,
        "label": "CI Custom Model v2",
        "updated_at": "2026-04-21T00:10:00",
    }
    save_custom_v2 = await api_client.post("/api/models/custom", json=updated_custom_model)
    assert save_custom_v2.status_code == 200

    custom_history = await api_client.get("/api/models/custom/custom-ci-model/history")
    assert custom_history.status_code == 200
    assert len(custom_history.json()["items"]) >= 1

    history_items = custom_history.json()["items"]
    oldest_revision = history_items[-1]["commit"]
    restore_custom = await api_client.post(
        "/api/models/custom/custom-ci-model/restore",
        json={"revision": oldest_revision},
    )
    assert restore_custom.status_code == 200
    assert restore_custom.json()["restore_behavior"] == "overwrite_current"
    restored_custom_list = await api_client.get("/api/models/custom")
    assert restored_custom_list.status_code == 200
    restored_entry = next(
        item for item in restored_custom_list.json()["items"] if item["id"] == "custom-ci-model"
    )
    assert restored_entry["label"] == "CI Custom Model"

    model_inspect = await api_client.post(
        "/api/models/inspect-config",
        json={
            "num_classes": 2,
            "use_auxiliary_heads": True,
            "vision": {
                "backbone": "resnet18",
                "pretrained": True,
                "freeze_backbone": False,
                "feature_dim": 128,
                "dropout": 0.3,
                "attention_type": "cbam",
            },
            "tabular": {
                "hidden_dims": [32],
                "output_dim": 16,
                "dropout": 0.2,
            },
            "fusion": {
                "fusion_type": "concatenate",
                "hidden_dim": 144,
                "dropout": 0.3,
                "num_heads": 4,
            },
            "numerical_features": ["age"],
            "categorical_features": ["gender"],
            "image_size": 64,
            "use_attention_supervision": False,
            "num_epochs": 3,
        },
    )
    assert model_inspect.status_code == 200
    model_inspect_payload = model_inspect.json()
    assert model_inspect_payload["can_enter_training"] is True
    assert model_inspect_payload["summary"]["backbone"] == "resnet18"
    assert model_inspect_payload["runtime"]["total_params"] > 0
    assert model_inspect_payload["runtime"]["tabular_input_dim"] == 2

    # Training lifecycle (start + status)
    started_job = await api_client.post(
        "/api/training/start",
        json={
            "experiment_name": "ci-exp",
            "training_model_config": {
                "backbone": "mobilenetv2",
                "num_classes": 2,
                "pretrained": False,
            },
            "dataset_config": {
                "dataset": "ci-dataset",
                "data_path": str(mock_dataset_path),
                "num_classes": 2,
            },
            "training_config": {
                "epochs": 1,
                "batch_size": 8,
                "learning_rate": 0.001,
                "image_size": 64,
                "num_workers": 0,
                "mixed_precision": False,
            },
        },
    )
    assert started_job.status_code == 200
    job_id = started_job.json()["job_id"]

    job_status = await api_client.get(f"/api/training/{job_id}/status")
    assert job_status.status_code == 200
    assert job_status.json()["job_id"] == job_id

    for _ in range(45):
        current_status = await api_client.get(f"/api/training/{job_id}/status")
        assert current_status.status_code == 200
        if current_status.json()["status"] == "completed":
            break
        if current_status.json()["status"] == "failed":
            raise AssertionError(current_status.json().get("error_message"))
        await asyncio.sleep(1.0)
    else:
        raise AssertionError("training job did not complete in expected time")

    history_payload = await api_client.get(f"/api/training/{job_id}/history")
    assert history_payload.status_code == 200
    assert len(history_payload.json()["entries"]) >= 1

    completed_status = await api_client.get(f"/api/training/{job_id}/status")
    assert completed_status.status_code == 200
    assert completed_status.json()["result_model_id"] is not None

    refreshed_models = await api_client.get("/api/models/")
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

    generated_model_detail = await api_client.get(
        f"/api/models/{generated_model['id']}"
    )
    assert generated_model_detail.status_code == 200
    detail_payload = generated_model_detail.json()
    assert detail_payload["validation"]["prediction_summary"]["error_count"] >= 0
    assert detail_payload["validation"]["dataset"]["num_classes"] == 2

    # Cleanup
    deleted_model = await api_client.delete(f"/api/models/{model_id}")
    assert deleted_model.status_code == 200

    deleted_generated_model = await api_client.delete(
        f"/api/models/{generated_model['id']}"
    )
    assert deleted_generated_model.status_code == 200

    deleted_custom = await api_client.delete("/api/models/custom/custom-ci-model")
    assert deleted_custom.status_code == 200

    deleted_dataset = await api_client.delete(f"/api/datasets/{dataset_id}")
    assert deleted_dataset.status_code == 200


async def test_web_can_import_real_cli_run(api_client, tmp_path) -> None:
    config_path, checkpoint_path = _create_checkpoint_and_logs(
        tmp_path,
        include_survival=True,
    )

    imported = await api_client.post(
        "/api/models/import-run",
        json={
            "config_path": str(config_path),
            "checkpoint_path": str(checkpoint_path),
            "split": "train",
            "attention_samples": 2,
            "survival_time_column": "survival_time",
            "survival_event_column": "event",
            "importance_sample_limit": 8,
            "name": "ci-imported-real-run",
            "tags": ["ci-import"],
        },
    )
    assert imported.status_code == 200
    payload = imported.json()
    assert payload["name"] == "ci-imported-real-run"
    assert payload["validation"]["overview"]["split"] == "train"
    assert payload["visualizations"]["confusion_matrix"]["plot_url"]
    assert payload["visualizations"]["attention_maps"]
    assert payload["validation"]["survival"]["c_index"] is not None
    assert payload["visualizations"]["survival_curve"]["image_url"]
    assert payload["visualizations"]["risk_score_distribution"]["image_url"]
    assert payload["validation"]["global_feature_importance"]["top_features"]
    assert payload["visualizations"]["feature_importance_bar"]["image_url"]
    assert payload["visualizations"]["feature_importance_beeswarm"]["image_url"]
    assert any(artifact["key"] == "survival" for artifact in payload["result_files"])
    assert any(
        artifact["key"] == "feature_importance" for artifact in payload["result_files"]
    )
    assert any(artifact["key"] == "report" for artifact in payload["result_files"])

    imported_detail = await api_client.get(f"/api/models/{payload['id']}")
    assert imported_detail.status_code == 200
    detail_payload = imported_detail.json()
    assert detail_payload["validation"]["overview"]["split"] == "train"
    assert any(artifact["key"] == "summary" for artifact in detail_payload["result_files"])
    assert any(artifact["key"] == "validation" for artifact in detail_payload["result_files"])
    assert any(artifact["key"] == "report" for artifact in detail_payload["result_files"])

    for artifact_key in ("summary", "validation", "report", "feature_importance"):
        artifact_response = await api_client.get(
            f"/api/models/{payload['id']}/artifacts/{artifact_key}"
        )
        assert artifact_response.status_code == 200
        assert artifact_response.content

    listed = await api_client.get("/api/models/")
    assert listed.status_code == 200
    assert any(item["id"] == payload["id"] for item in listed.json())

    deleted = await api_client.delete(f"/api/models/{payload['id']}")
    assert deleted.status_code == 200
