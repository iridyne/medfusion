"""Contract tests for model result payloads consumed by the result panel."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from med_core.web.api import models as models_api
from med_core.web.models import ModelInfo


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def test_model_payload_keeps_result_panel_contract(tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "checkpoints" / "best.pth"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path.write_bytes(b"checkpoint")

    config_path = tmp_path / "artifacts" / "training-config.json"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text("{}", encoding="utf-8")

    summary_path = tmp_path / "reports" / "summary.json"
    metrics_path = tmp_path / "metrics" / "metrics.json"
    validation_path = tmp_path / "metrics" / "validation.json"
    report_path = tmp_path / "reports" / "report.md"
    history_path = tmp_path / "logs" / "history.json"
    roc_curve_json_path = tmp_path / "metrics" / "roc_curve.json"
    roc_curve_plot_path = tmp_path / "artifacts" / "visualizations" / "roc_curve.png"
    confusion_matrix_json_path = tmp_path / "metrics" / "confusion_matrix.json"
    confusion_matrix_plot_path = (
        tmp_path / "artifacts" / "visualizations" / "confusion_matrix.png"
    )
    training_curves_plot_path = (
        tmp_path / "artifacts" / "visualizations" / "training_curves.png"
    )
    survival_path = tmp_path / "metrics" / "survival.json"
    kaplan_meier_plot_path = (
        tmp_path / "artifacts" / "visualizations" / "kaplan_meier_curve.png"
    )
    risk_score_distribution_plot_path = (
        tmp_path / "artifacts" / "visualizations" / "risk_score_distribution.png"
    )
    feature_importance_path = tmp_path / "metrics" / "feature_importance.json"
    feature_importance_bar_plot_path = (
        tmp_path / "artifacts" / "visualizations" / "feature_importance_bar.png"
    )
    feature_importance_beeswarm_plot_path = (
        tmp_path / "artifacts" / "visualizations" / "feature_importance_beeswarm.png"
    )

    _write_json(
        summary_path,
        {
            "meta": {"schema_version": "1.0"},
            "metrics": {"accuracy": 0.71},
        },
    )
    _write_json(metrics_path, {"accuracy": 0.71, "auc": 0.81})
    _write_json(
        validation_path,
        {
            "overview": {
                "sample_count": 24,
                "auc": 0.81,
                "macro_f1": 0.73,
                "balanced_accuracy": 0.7,
                "best_epoch": 3,
            },
            "dataset": {"num_classes": 2},
            "prediction_summary": {"error_count": 4, "error_rate": 0.16},
            "per_class": [{"label": "0", "precision": 0.8}],
            "survival": {"c_index": 0.66},
            "global_feature_importance": {
                "top_features": [{"feature": "age", "importance": 0.4}]
            },
        },
    )
    _write_json(
        history_path,
        {
            "entries": [
                {
                    "epoch": 1,
                    "train_loss": 0.8,
                    "val_loss": 0.7,
                    "train_accuracy": 0.61,
                    "val_accuracy": 0.64,
                    "learning_rate": 0.001,
                }
            ]
        },
    )
    _write_json(roc_curve_json_path, {"auc": 0.81, "points": [[0.0, 0.0], [1.0, 1.0]]})
    _write_json(
        confusion_matrix_json_path,
        {"matrix": [[10, 2], [3, 9]], "labels": ["negative", "positive"]},
    )
    _write_json(survival_path, {"c_index": 0.66})
    _write_json(
        feature_importance_path,
        {"top_features": [{"feature": "age", "importance": 0.4}]},
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("# report", encoding="utf-8")
    for image_path in (
        roc_curve_plot_path,
        confusion_matrix_plot_path,
        training_curves_plot_path,
        kaplan_meier_plot_path,
        risk_score_distribution_plot_path,
        feature_importance_bar_plot_path,
        feature_importance_beeswarm_plot_path,
    ):
        image_path.parent.mkdir(parents=True, exist_ok=True)
        image_path.write_bytes(b"png")

    model = ModelInfo(
        id=7,
        name="panel-contract-model",
        description="",
        model_type="classification",
        architecture="resnet18",
        config={
            "artifact_paths": {
                "summary_path": str(summary_path),
                "metrics_path": str(metrics_path),
                "validation_path": str(validation_path),
                "report_path": str(report_path),
                "history_path": str(history_path),
                "roc_curve_json_path": str(roc_curve_json_path),
                "roc_curve_plot_path": str(roc_curve_plot_path),
                "confusion_matrix_json_path": str(confusion_matrix_json_path),
                "confusion_matrix_plot_path": str(confusion_matrix_plot_path),
                "training_curves_plot_path": str(training_curves_plot_path),
                "survival_path": str(survival_path),
                "kaplan_meier_plot_path": str(kaplan_meier_plot_path),
                "risk_score_distribution_plot_path": str(
                    risk_score_distribution_plot_path
                ),
                "feature_importance_path": str(feature_importance_path),
                "feature_importance_bar_plot_path": str(
                    feature_importance_bar_plot_path
                ),
                "feature_importance_beeswarm_plot_path": str(
                    feature_importance_beeswarm_plot_path
                ),
            },
            "result_summary": {"best_accuracy": 0.71},
        },
        metrics={"auc": 0.81, "balanced_accuracy": 0.7},
        accuracy=0.71,
        loss=0.42,
        checkpoint_path=str(checkpoint_path),
        config_path=str(config_path),
        training_time=88.0,
        created_at=datetime(2026, 3, 28, 10, 0, 0),
        updated_at=datetime(2026, 3, 28, 10, 5, 0),
    )

    payload = models_api._to_payload(model)

    assert payload["training_history"] is not None
    assert len(payload["training_history"]["entries"]) == 1
    assert payload["validation"]["overview"]["sample_count"] == 24
    assert payload["validation"]["global_feature_importance"]["top_features"]
    assert payload["visualizations"]["roc_curve"]["plot_url"] == (
        "/api/models/7/artifacts/roc_curve_plot"
    )
    assert payload["visualizations"]["training_curves"]["image_url"] == (
        "/api/models/7/artifacts/training_curves_plot"
    )
    assert payload["visualizations"]["survival_curve"]["image_url"] == (
        "/api/models/7/artifacts/kaplan_meier_plot"
    )
    assert payload["visualizations"]["feature_importance_bar"]["image_url"] == (
        "/api/models/7/artifacts/feature_importance_bar_plot"
    )
    result_file_keys = {item["key"] for item in payload["result_files"]}
    assert {"summary", "validation", "report", "history"} <= result_file_keys
