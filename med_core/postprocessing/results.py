"""Build real validation artifacts from a trained config + checkpoint."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_curve,
)
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from torch.utils.data import DataLoader

from med_core.backbones import create_tabular_backbone, create_vision_backbone
from med_core.configs import load_config
from med_core.datasets import (
    MedicalMultimodalDataset,
    ThreePhaseCTCaseDataset,
    get_val_transforms,
    split_dataset,
)
from med_core.evaluation.interpretability import GradCAM
from med_core.evaluation.report_generator import generate_result_artifact_report
from med_core.fusion import MultiModalFusionModel, create_fusion_module
from med_core.output_layout import RunOutputLayout, resolve_run_output_dir
from med_core.postprocessing.analysis import (
    build_global_feature_importance_artifacts,
    build_survival_artifacts,
    resolve_survival_columns,
)
from med_core.postprocessing.advanced_analysis import build_shap_artifacts
from med_core.models import ThreePhaseCTFusionModel
from med_core.shared.visualization import (
    plot_calibration_curve,
    plot_confusion_matrix,
    plot_probability_distribution,
    plot_roc_curve,
    plot_training_curves,
)
from med_core.visualization.attention_viz import (
    plot_attention_statistics,
    visualize_attention_overlay,
)

logger = logging.getLogger(__name__)

_BUILD_RESULTS_SCHEMA_VERSION = "1.0"
_BUILD_RESULTS_GENERATED_BY = "medfusion.build_results"


@dataclass
class InferenceArtifacts:
    """In-memory inference payload for artifact generation."""

    labels: list[str]
    num_classes: int
    positive_class_index: int
    y_true: np.ndarray
    y_pred: np.ndarray
    probabilities: np.ndarray
    risk_scores: np.ndarray
    risk_score_source: str
    sample_records: list[dict[str, Any]]


@dataclass
class BuildResultsOutput:
    """Final artifact metadata returned by build_results_artifacts."""

    output_dir: str
    artifact_paths: dict[str, str]
    metrics: dict[str, Any]
    validation: dict[str, Any]


def _read_manifest_dataframe(config: Any) -> pd.DataFrame:
    read_csv_kwargs = {}
    if config.data.patient_id_column:
        read_csv_kwargs["dtype"] = {config.data.patient_id_column: "string"}
    return pd.read_csv(config.data.csv_path, **read_csv_kwargs)


def _resolve_device(device_name: str) -> torch.device:
    if device_name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device_name)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        logger.warning("Failed to parse JSON artifact: %s", path)
        return {}


def _safe_rate(numerator: float | int, denominator: float | int) -> float:
    if not denominator:
        return 0.0
    return float(numerator) / float(denominator)


def _round_metric(value: float | None, digits: int = 4) -> float | None:
    if value is None:
        return None
    if bool(np.isnan(value)):
        return None
    return round(float(value), digits)


def _format_importance_method(method: str | None) -> str:
    if not method:
        return "-"
    if method in {"logistic_surrogate_shap", "ridge_surrogate_shap"}:
        return "SHAP-style surrogate"
    return method.replace("_", " ")


def _build_artifact_metadata(
    *,
    config_path: Path,
    checkpoint_path: Path,
    split: str,
) -> dict[str, Any]:
    return {
        "schema_version": _BUILD_RESULTS_SCHEMA_VERSION,
        "generated_by": _BUILD_RESULTS_GENERATED_BY,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_config_path": str(config_path),
        "checkpoint_path": str(checkpoint_path),
        "split": split,
    }


def _load_history(layout: RunOutputLayout) -> dict[str, Any]:
    history_path = layout.history_path
    if history_path.exists():
        payload = _read_json(history_path)
        if payload.get("entries"):
            return payload

    log_dir = layout.logs_dir
    event_files = sorted(log_dir.glob("events.out.tfevents.*"))
    if not event_files:
        return {"entries": []}

    accumulator = EventAccumulator(str(event_files[-1]))
    accumulator.Reload()
    scalar_tags = set(accumulator.Tags().get("scalars", []))

    tag_mapping = {
        "train/loss": "train_loss",
        "val/loss": "val_loss",
        "train/accuracy": "train_accuracy",
        "val/accuracy": "val_accuracy",
        "learning_rate": "learning_rate",
    }
    by_step: dict[int, dict[str, Any]] = {}
    for tensorboard_tag, output_key in tag_mapping.items():
        if tensorboard_tag not in scalar_tags:
            continue
        for event in accumulator.Scalars(tensorboard_tag):
            entry = by_step.setdefault(int(event.step), {"epoch": int(event.step)})
            entry[output_key] = float(event.value)

    entries = [by_step[index] for index in sorted(by_step)]
    best_val_loss = float("inf")
    for entry in entries:
        current_val_loss = entry.get("val_loss")
        if current_val_loss is not None and current_val_loss <= best_val_loss:
            best_val_loss = current_val_loss
            entry["best_so_far"] = True

    payload = {"entries": entries}
    if entries:
        _write_json(history_path, payload)
    return payload


def _build_model(config: Any, tabular_dim: int) -> MultiModalFusionModel:
    vision_backbone = create_vision_backbone(
        backbone_name=config.model.vision.backbone,
        pretrained=config.model.vision.pretrained,
        freeze=config.model.vision.freeze_backbone,
        feature_dim=config.model.vision.feature_dim,
        dropout=config.model.vision.dropout,
        attention_type=config.model.vision.attention_type,
        enable_attention_supervision=config.model.vision.enable_attention_supervision,
    )
    tabular_backbone = create_tabular_backbone(
        input_dim=tabular_dim,
        output_dim=config.model.tabular.output_dim,
        hidden_dims=config.model.tabular.hidden_dims,
        dropout=config.model.tabular.dropout,
    )
    fusion_module = create_fusion_module(
        fusion_type=config.model.fusion.fusion_type,
        vision_dim=config.model.vision.feature_dim,
        tabular_dim=config.model.tabular.output_dim,
        output_dim=config.model.fusion.hidden_dim,
        dropout=config.model.fusion.dropout,
    )
    return MultiModalFusionModel(
        vision_backbone=vision_backbone,
        tabular_backbone=tabular_backbone,
        fusion_module=fusion_module,
        num_classes=config.model.num_classes,
        use_auxiliary_heads=config.model.use_auxiliary_heads,
    )


def _load_split_dataset(config: Any, split: str) -> MedicalMultimodalDataset:
    transform = get_val_transforms(image_size=config.data.image_size)
    full_dataset, _ = MedicalMultimodalDataset.from_csv(
        csv_path=config.data.csv_path,
        image_dir=config.data.image_dir,
        image_column=config.data.image_path_column,
        target_column=config.data.target_column,
        numerical_features=config.data.numerical_features,
        categorical_features=config.data.categorical_features,
        patient_id_column=config.data.patient_id_column,
        transform=None,
    )
    train_ds, val_ds, test_ds = split_dataset(
        full_dataset,
        train_ratio=config.data.train_ratio,
        val_ratio=config.data.val_ratio,
        test_ratio=config.data.test_ratio,
        random_seed=config.data.random_seed,
    )
    datasets = {"train": train_ds, "val": val_ds, "test": test_ds}
    selected = datasets[split]
    selected.transform = transform
    return selected


def _split_three_phase_records(
    config: Any,
    dataset: ThreePhaseCTCaseDataset,
    split: str,
) -> ThreePhaseCTCaseDataset:
    dataset_size = len(dataset.records)
    generator = torch.Generator().manual_seed(config.data.random_seed)
    indices = torch.randperm(dataset_size, generator=generator).tolist()

    train_count = int(round(dataset_size * config.data.train_ratio))
    val_count = int(round(dataset_size * config.data.val_ratio))
    if train_count <= 0 and dataset_size > 0:
        train_count = 1
    if train_count + val_count > dataset_size:
        val_count = max(dataset_size - train_count, 0)

    split_indices = {
        "train": indices[:train_count],
        "val": indices[train_count : train_count + val_count],
        "test": indices[train_count + val_count :],
    }
    selected_indices = split_indices[split]
    if not selected_indices:
        selected_indices = split_indices["val"] or split_indices["train"]

    selected_records = [dataset.records[index] for index in selected_indices]
    return ThreePhaseCTCaseDataset(
        records=selected_records,
        target_shape=dataset.target_shape,
        window_preset=dataset.window_preset,
    )


def _build_three_phase_results(
    *,
    config: Any,
    config_path: Path,
    checkpoint_path: Path,
    layout: RunOutputLayout,
    split: str,
    enable_importance: bool,
    importance_sample_limit: int,
) -> BuildResultsOutput:
    dataframe = _read_manifest_dataframe(config)
    full_dataset = ThreePhaseCTCaseDataset.from_manifest_dataframe(
        dataframe,
        phase_dir_columns=config.data.phase_dir_columns,
        clinical_feature_columns=config.data.clinical_feature_columns,
        target_column=config.data.target_column,
        patient_id_column=config.data.patient_id_column or "case_id",
        target_shape=tuple(config.data.target_shape or [16, 64, 64]),
        window_preset=config.data.window_preset,
    )
    dataset = _split_three_phase_records(config, full_dataset, split)
    device = _resolve_device(config.device)
    model = ThreePhaseCTFusionModel(
        phase_feature_dim=config.model.phase_feature_dim,
        clinical_input_dim=len(config.data.clinical_feature_columns),
        clinical_hidden_dim=config.model.tabular.output_dim,
        fusion_hidden_dim=config.model.fusion.hidden_dim,
        phase_fusion_type=config.model.phase_fusion_type,
        share_phase_encoder=config.model.share_phase_encoder,
        freeze_phase_encoder=config.model.vision.freeze_backbone,
        use_risk_head=config.model.use_risk_head,
    )
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)

    loader = DataLoader(
        dataset,
        batch_size=max(config.data.batch_size, 1),
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    cases: list[dict[str, Any]] = []
    y_true: list[int] = []
    y_pred: list[int] = []
    y_prob: list[float] = []
    risk_scores: list[float] = []
    tabular_rows: list[list[float]] = []

    model.eval()
    with torch.inference_mode():
        for batch in loader:
            outputs = model(
                arterial=batch["arterial"].to(device),
                portal=batch["portal"].to(device),
                noncontrast=batch["noncontrast"].to(device),
                clinical=batch["clinical"].to(device),
            )
            probabilities = outputs["probability"].detach().cpu().numpy().tolist()
            predictions = (outputs["probability"] >= 0.5).long().cpu().numpy().tolist()
            labels = batch["label"].cpu().numpy().tolist()
            batch_risk_scores = outputs["risk_score"].detach().cpu().numpy().tolist()
            case_ids = list(batch["case_id"])
            batch_clinical = batch["clinical"].cpu().numpy().tolist()
            for case_id, label, pred, prob, risk in zip(
                case_ids,
                labels,
                predictions,
                probabilities,
                batch_risk_scores,
                strict=True,
            ):
                y_true.append(int(label))
                y_pred.append(int(pred))
                y_prob.append(float(prob))
                risk_scores.append(float(risk))
                cases.append(
                    {
                        "case_id": case_id,
                        "true_label": int(label),
                        "predicted_label": int(pred),
                        "pred_probability": float(prob),
                        "risk_score": float(risk),
                    }
                )
            tabular_rows.extend(
                [list(map(float, row)) for row in batch_clinical]
            )

    accuracy = accuracy_score(y_true, y_pred) if y_true else 0.0
    auc_value: float | None = None
    visualization_dir = layout.visualizations_dir
    visualization_dir.mkdir(parents=True, exist_ok=True)
    roc_curve_path: Path | None = None
    if len(set(y_true)) > 1 and y_prob:
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc_value = float(auc(fpr, tpr))
        roc_curve_path = visualization_dir / "roc_curve.png"
        roc_figure, _ = plot_roc_curve(
            np.asarray(y_true, dtype=int),
            np.asarray(y_prob, dtype=float),
            title="ROC Curve (Three-Phase CT MVI)",
            save_path=roc_curve_path,
        )
        plt.close(roc_figure)

    confusion_matrix_path = visualization_dir / "confusion_matrix.png"
    confusion_figure, _ = plot_confusion_matrix(
        np.asarray(y_true, dtype=int),
        np.asarray(y_pred, dtype=int),
        class_names=["Negative", "Positive"],
        title="Confusion Matrix",
        save_path=confusion_matrix_path,
    )
    plt.close(confusion_figure)

    roc_payload = {
        "artifact_key": "roc_curve_plot",
        "auc": _round_metric(auc_value),
        "plot_path": str(roc_curve_path) if roc_curve_path else None,
    }
    confusion_payload = {
        "artifact_key": "confusion_matrix_plot",
        "labels": ["Negative", "Positive"],
        "matrix": confusion_matrix(
            np.asarray(y_true, dtype=int),
            np.asarray(y_pred, dtype=int),
            labels=[0, 1],
        ).astype(int).tolist(),
        "plot_path": str(confusion_matrix_path),
    }

    _write_json(layout.roc_curve_json_path, roc_payload)
    _write_json(layout.confusion_matrix_json_path, confusion_payload)

    predictions_payload = {
        "labels": ["Negative", "Positive"],
        "positive_class_index": 1,
        "y_true": y_true,
        "y_pred": y_pred,
        "probabilities": [[round(1.0 - prob, 6), round(prob, 6)] for prob in y_prob],
        "risk_scores": [round(float(score), 6) for score in risk_scores],
        "risk_score_source": "risk_head" if config.model.use_risk_head else "probability",
        "samples": cases,
    }
    _write_json(layout.predictions_path, predictions_payload)
    layout.config_snapshot_path.write_text(
        json.dumps(config.to_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    importance_payload: dict[str, Any] | None = None
    importance_artifact_paths: dict[str, str] = {}
    if enable_importance and importance_sample_limit > 0:
        importance_payload_raw, importance_artifact_paths, _metric_updates = (
            build_shap_artifacts(
                output_dir=layout.artifacts_dir,
                feature_names=list(config.data.clinical_feature_columns),
                tabular_data=np.asarray(tabular_rows, dtype=float),
                model_scores=np.asarray(risk_scores, dtype=float),
                y_true=np.asarray(y_true, dtype=int),
                positive_class_label="Positive",
                max_display=min(10, 1 + len(config.data.clinical_feature_columns)),
                max_samples=max(importance_sample_limit, 1),
            )
        )
        if importance_payload_raw.get("available"):
            importance_artifact_paths = {
                **importance_artifact_paths,
                "feature_importance_path": importance_artifact_paths.get(
                    "shap_summary_json_path"
                ),
                "feature_importance_bar_plot_path": importance_artifact_paths.get(
                    "shap_bar_plot_path"
                ),
                "feature_importance_beeswarm_plot_path": importance_artifact_paths.get(
                    "shap_beeswarm_plot_path"
                ),
            }
            importance_artifact_paths = {
                key: value for key, value in importance_artifact_paths.items() if value
            }
            importance_payload = {
                "available": True,
                "method": importance_payload_raw.get("method"),
                "score_name": predictions_payload["risk_score_source"],
                "top_features": importance_payload_raw.get("features", []),
                "artifacts": importance_payload_raw.get("artifacts", {}),
            }
        else:
            importance_artifact_paths = {}

    artifact_metadata = _build_artifact_metadata(
        config_path=config_path,
        checkpoint_path=checkpoint_path,
        split=split,
    )
    history_payload = _load_history(layout)
    metrics_payload = {
        **artifact_metadata,
        "task": "three_phase_ct_tabular_classification",
        "accuracy": round(float(accuracy), 4),
        "auc": _round_metric(auc_value),
        "sample_count": len(cases),
        "global_feature_importance": importance_payload,
    }
    validation_payload = {
        **artifact_metadata,
        "overview": {"split": split, "sample_count": len(cases)},
        "cases": cases,
        "global_feature_importance": importance_payload,
    }
    summary_payload = {
        **artifact_metadata,
        "run_name": config.experiment_name,
        "task": "classification",
        "primary_metric": "accuracy" if auc_value is None else "auc",
        "primary_metric_value": round(float(accuracy), 4)
        if auc_value is None
        else _round_metric(auc_value),
        "checkpoint": str(checkpoint_path),
        "artifacts": {
            "config_path": str(layout.config_snapshot_path),
            "metrics_path": str(layout.metrics_path),
            "validation_path": str(layout.validation_path),
            "predictions_path": str(layout.predictions_path),
            "prediction_path": str(layout.predictions_path),
            "summary_path": str(layout.summary_path),
            "report_path": str(layout.report_path),
            "history_path": str(layout.history_path),
            "roc_curve_json_path": str(layout.roc_curve_json_path),
            "confusion_matrix_json_path": str(layout.confusion_matrix_json_path),
            "roc_curve_plot_path": str(roc_curve_path) if roc_curve_path else None,
            "confusion_matrix_plot_path": str(confusion_matrix_path),
            **importance_artifact_paths,
        },
        "global_feature_importance": importance_payload,
    }
    report_lines = [
        "# MedFusion Result Report",
        "",
        "## Overview",
        "",
        f"- Experiment: {config.experiment_name}",
        f"- Split: {split}",
        f"- Samples: {len(cases)}",
        "",
        "## Metrics",
        "",
        f"- Accuracy: {metrics_payload['accuracy']}",
        f"- AUC: {metrics_payload['auc']}",
        "",
        "## Data Summary",
        "",
        f"- Positive Cases: {sum(y_true)}",
        f"- Negative Cases: {len(y_true) - sum(y_true)}",
        f"- Clinical Features: {', '.join(config.data.clinical_feature_columns)}",
    ]
    report_lines.extend(["", "## Visual Artifacts", ""])
    if roc_curve_path is not None:
        report_lines.append(f"- ROC Curve: {roc_curve_path}")
    report_lines.append(f"- Confusion Matrix: {confusion_matrix_path}")
    if importance_payload is not None:
        report_lines.extend(["", "## Feature Importance", ""])
        report_lines.append(
            f"- Method: {_format_importance_method(importance_payload.get('method'))}"
        )
        top_features = importance_payload.get("top_features", [])[:5]
        if top_features:
            for item in top_features:
                report_lines.append(
                    "- "
                    f"{item.get('feature')}: "
                    f"{item.get('mean_abs_contribution')}"
                )
        report_lines.extend(["", "## Artifact Paths", ""])
    else:
        report_lines.extend(["", "## Artifact Paths", ""])
    if importance_payload is not None:
        report_lines.append(
            "- Feature Importance Bar: "
            f"{importance_artifact_paths.get('feature_importance_bar_plot_path')}"
        )
        report_lines.append(
            "- Feature Importance Beeswarm: "
            f"{importance_artifact_paths.get('feature_importance_beeswarm_plot_path')}"
        )
    report_lines.append(f"- Config Snapshot: {layout.config_snapshot_path}")
    report_lines.append(f"- Metrics JSON: {layout.metrics_path}")
    report_lines.append(f"- Validation JSON: {layout.validation_path}")
    report_lines.append(f"- Predictions JSON: {layout.predictions_path}")
    report_lines.append(f"- History JSON: {layout.history_path}")

    _write_json(layout.metrics_path, metrics_payload)
    _write_json(layout.validation_path, validation_payload)
    _write_json(layout.summary_path, summary_payload)
    if history_payload.get("entries"):
        _write_json(layout.history_path, history_payload)
    layout.report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    artifact_paths = {
        "config_path": str(layout.config_snapshot_path),
        "metrics_path": str(layout.metrics_path),
        "validation_path": str(layout.validation_path),
        "predictions_path": str(layout.predictions_path),
        "prediction_path": str(layout.predictions_path),
        "summary_path": str(layout.summary_path),
        "report_path": str(layout.report_path),
        "history_path": str(layout.history_path),
        "roc_curve_json_path": str(layout.roc_curve_json_path),
        "confusion_matrix_json_path": str(layout.confusion_matrix_json_path),
        "roc_curve_plot_path": str(roc_curve_path) if roc_curve_path else None,
        "confusion_matrix_plot_path": str(confusion_matrix_path),
        **importance_artifact_paths,
    }
    artifact_paths = {key: value for key, value in artifact_paths.items() if value}
    return BuildResultsOutput(
        output_dir=str(layout.root_dir),
        artifact_paths=artifact_paths,
        metrics=metrics_payload,
        validation=validation_payload,
    )


def _run_inference(
    model: MultiModalFusionModel,
    dataset: MedicalMultimodalDataset,
    device: torch.device,
    batch_size: int,
) -> InferenceArtifacts:
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    y_true: list[int] = []
    y_pred: list[int] = []
    probabilities: list[list[float]] = []

    model.eval()
    with torch.inference_mode():
        for images, tabular, labels in loader:
            images = images.to(device)
            tabular = tabular.to(device)
            outputs = model(images, tabular)
            logits = outputs["logits"] if isinstance(outputs, dict) else outputs
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)

            y_true.extend(labels.cpu().numpy().tolist())
            y_pred.extend(preds.cpu().numpy().tolist())
            probabilities.extend(probs.cpu().numpy().tolist())

    y_true_array = np.asarray(y_true, dtype=int)
    probabilities_array = np.asarray(probabilities, dtype=float)
    y_pred_array = np.asarray(y_pred, dtype=int)
    num_classes = probabilities_array.shape[1] if probabilities_array.ndim == 2 else 1
    labels = [str(label) for label in sorted(np.unique(y_true_array).tolist())]
    if len(labels) != num_classes:
        labels = [f"Class {index}" for index in range(num_classes)]

    confidence = probabilities_array.max(axis=1)
    if num_classes == 2:
        risk_scores = probabilities_array[:, 1]
        risk_score_source = "positive_class_probability"
    else:
        risk_scores = confidence
        risk_score_source = "max_predicted_probability"
    sample_records = [
        {
            "index": index,
            "patient_id": dataset.get_patient_id(index)
            if hasattr(dataset, "get_patient_id")
            else None,
            "true_label": labels[int(y_true_array[index])],
            "predicted_label": labels[int(y_pred_array[index])],
            "confidence": round(float(confidence[index]), 4),
            "risk_score": round(float(risk_scores[index]), 6),
        }
        for index in range(len(y_true_array))
    ]

    return InferenceArtifacts(
        labels=labels,
        num_classes=num_classes,
        positive_class_index=1 if num_classes > 1 else 0,
        y_true=y_true_array,
        y_pred=y_pred_array,
        probabilities=probabilities_array,
        risk_scores=risk_scores,
        risk_score_source=risk_score_source,
        sample_records=sample_records,
    )


def _resolve_survival_cutoff(
    config: Any,
    model: MultiModalFusionModel,
    device: torch.device,
    batch_size: int,
    split: str,
    time_column: str | None,
    event_column: str | None,
) -> tuple[float | None, str]:
    if not time_column or not event_column or split == "train":
        return None, "current_split_median"

    try:
        train_dataset = _load_split_dataset(config, "train")
        train_inference = _run_inference(
            model=model,
            dataset=train_dataset,
            device=device,
            batch_size=batch_size,
        )
        return float(np.median(train_inference.risk_scores)), "train_split_median"
    except Exception as exc:
        logger.warning(
            "Failed to resolve train survival cutoff, fallback to current split median: %s",
            exc,
        )
        return None, "current_split_median"


def _compute_expected_calibration_error(
    y_true_binary: np.ndarray,
    positive_scores: np.ndarray,
    n_bins: int = 10,
) -> tuple[float, list[dict[str, Any]]]:
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_indices = np.digitize(positive_scores, bin_edges[1:-1], right=True)
    summaries: list[dict[str, Any]] = []
    ece = 0.0

    for bin_index in range(n_bins):
        mask = bin_indices == bin_index
        count = int(mask.sum())
        if count == 0:
            continue

        confidence = float(np.mean(positive_scores[mask]))
        accuracy = float(np.mean(y_true_binary[mask]))
        gap = abs(confidence - accuracy)
        ece += gap * (count / len(positive_scores))
        summaries.append(
            {
                "bin_index": bin_index,
                "range_start": round(float(bin_edges[bin_index]), 4),
                "range_end": round(float(bin_edges[bin_index + 1]), 4),
                "count": count,
                "mean_confidence": round(confidence, 4),
                "empirical_accuracy": round(accuracy, 4),
                "gap": round(gap, 4),
            }
        )

    return round(float(ece), 4), summaries


def _select_attention_indices(
    y_true: np.ndarray, y_pred: np.ndarray, probabilities: np.ndarray, limit: int
) -> list[int]:
    confidence = probabilities.max(axis=1)
    correct_indices = [
        int(index)
        for index in np.argsort(-confidence)
        if y_true[int(index)] == y_pred[int(index)]
    ]
    error_indices = [
        int(index)
        for index in np.argsort(-confidence)
        if y_true[int(index)] != y_pred[int(index)]
    ]
    selected = error_indices[: max(limit // 2, 1)] + correct_indices[:limit]
    deduped: list[int] = []
    for index in selected:
        if index not in deduped:
            deduped.append(index)
        if len(deduped) >= limit:
            break
    return deduped


def _resize_original_image(image_path: Path, target_size: int) -> np.ndarray:
    image = Image.open(image_path).convert("RGB")
    if image.size != (target_size, target_size):
        image = image.resize((target_size, target_size))
    return np.asarray(image, dtype=np.float32) / 255.0


def _extract_cbam_attention(
    model: MultiModalFusionModel,
    image_tensor: torch.Tensor,
) -> np.ndarray | None:
    if not getattr(model.vision_backbone, "enable_attention_supervision", False):
        return None
    outputs = model.vision_backbone(image_tensor, return_intermediates=True)
    if not isinstance(outputs, dict):
        return None
    attention_weights = outputs.get("attention_weights")
    if attention_weights is None:
        return None
    return attention_weights[0, 0].detach().cpu().numpy()


def _generate_attention_artifacts(
    model: MultiModalFusionModel,
    dataset: MedicalMultimodalDataset,
    device: torch.device,
    layout: RunOutputLayout,
    config: Any,
    inference: InferenceArtifacts,
    sample_limit: int,
) -> dict[str, Any]:
    if sample_limit <= 0:
        return {}

    attention_dir = layout.visualizations_dir / "attention"
    attention_dir.mkdir(parents=True, exist_ok=True)
    selected_indices = _select_attention_indices(
        inference.y_true,
        inference.y_pred,
        inference.probabilities,
        sample_limit,
    )
    if not selected_indices:
        return {}

    gradcam = None
    manifest_items: list[dict[str, Any]] = []
    heatmaps: list[np.ndarray] = []

    for order, sample_index in enumerate(selected_indices, start=1):
        image_tensor, tabular_tensor, _ = dataset[sample_index]
        image_batch = image_tensor.unsqueeze(0).to(device)
        tabular_batch = tabular_tensor.unsqueeze(0).to(device)

        attention_map = _extract_cbam_attention(model, image_batch)
        method = "vision_attention"
        title_prefix = "Vision Attention"
        if attention_map is None:
            if gradcam is None:
                gradcam = GradCAM(model)
            attention_map = gradcam.generate(
                image_batch,
                target_class=int(inference.y_pred[sample_index]),
                additional_inputs={"tabular": tabular_batch},
            )
            method = "gradcam"
            title_prefix = "Grad-CAM"

        original_image = _resize_original_image(
            dataset.image_paths[sample_index],
            config.data.image_size,
        )
        artifact_key = f"attention_sample_{order}_{method}"
        image_path = attention_dir / f"{artifact_key}.png"
        figure = visualize_attention_overlay(
            image=original_image,
            attention=attention_map,
            alpha=0.55,
            title=(
                f"{title_prefix} #{order} | "
                f"true={inference.sample_records[sample_index]['true_label']} | "
                f"pred={inference.sample_records[sample_index]['predicted_label']}"
            ),
            save_path=image_path,
        )
        plt.close(figure)

        heatmaps.append(attention_map)
        manifest_items.append(
            {
                "title": f"{title_prefix} #{order}",
                "modality": "image",
                "method": method,
                "sample_index": sample_index,
                "artifact_key": artifact_key,
                "image_path": str(image_path),
                "mean_attention": round(float(np.mean(attention_map)), 4),
                "peak_attention": round(float(np.max(attention_map)), 4),
                "true_label": inference.sample_records[sample_index]["true_label"],
                "predicted_label": inference.sample_records[sample_index][
                    "predicted_label"
                ],
                "confidence": inference.sample_records[sample_index]["confidence"],
            }
        )

    statistics_path = attention_dir / "attention_statistics.png"
    statistics_figure = plot_attention_statistics(
        heatmaps,
        labels=[item["title"] for item in manifest_items],
        save_path=statistics_path,
    )
    plt.close(statistics_figure)

    manifest = {
        "items": manifest_items,
        "statistics_artifact_key": "attention_statistics_plot",
        "statistics_plot_path": str(statistics_path),
    }
    manifest_path = attention_dir / "attention_maps.json"
    _write_json(manifest_path, manifest)

    return {
        "attention_manifest_path": str(manifest_path),
        "attention_statistics_plot_path": str(statistics_path),
    }


def _build_metrics_payload(
    layout: RunOutputLayout,
    inference: InferenceArtifacts,
    history_entries: list[dict[str, Any]],
    dataset_name: str | None,
    split: str,
    attention_artifacts: dict[str, Any],
    survival_payload: dict[str, Any] | None,
    importance_payload: dict[str, Any] | None,
    artifact_metadata: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, str]]:
    labels = inference.labels
    num_classes = inference.num_classes
    positive_class_index = inference.positive_class_index
    y_true = inference.y_true
    y_pred = inference.y_pred
    probabilities = inference.probabilities

    confidence = probabilities.max(axis=1)
    correct_mask = y_true == y_pred
    average = "binary" if num_classes == 2 else "macro"
    positive_class_label = labels[positive_class_index]
    y_true_binary = (y_true == positive_class_index).astype(int)
    positive_scores = probabilities[:, positive_class_index]

    precision, recall, f1_score, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average=average,
        zero_division=0,
    )
    accuracy = accuracy_score(y_true, y_pred)
    matrix = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    (
        per_class_precision,
        per_class_recall,
        per_class_f1,
        per_class_support,
    ) = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=list(range(num_classes)),
        average=None,
        zero_division=0,
    )
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )
    _, _, f1_weighted, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0,
    )
    balanced_accuracy = (
        float(np.mean(per_class_recall)) if len(per_class_recall) else 0.0
    )
    class_supports = np.bincount(y_true, minlength=num_classes)
    predicted_distribution = np.bincount(y_pred, minlength=num_classes)
    error_count = int((~correct_mask).sum())
    misclassification_pairs = [
        {
            "actual": labels[row_index],
            "predicted": labels[col_index],
            "count": int(matrix[row_index, col_index]),
        }
        for row_index in range(num_classes)
        for col_index in range(num_classes)
        if row_index != col_index and int(matrix[row_index, col_index]) > 0
    ]
    misclassification_pairs.sort(key=lambda item: item["count"], reverse=True)
    mean_confidence = float(np.mean(confidence))
    mean_confidence_correct = (
        float(np.mean(confidence[correct_mask])) if bool(correct_mask.any()) else None
    )
    mean_confidence_error = (
        float(np.mean(confidence[~correct_mask])) if error_count > 0 else None
    )

    per_class_payload = [
        {
            "label": labels[class_index],
            "support": int(per_class_support[class_index]),
            "prevalence": round(
                _safe_rate(class_supports[class_index], len(y_true)), 4
            ),
            "precision": round(float(per_class_precision[class_index]), 4),
            "recall": round(float(per_class_recall[class_index]), 4),
            "f1_score": round(float(per_class_f1[class_index]), 4),
            "predicted_count": int(predicted_distribution[class_index]),
            "predicted_rate": round(
                _safe_rate(predicted_distribution[class_index], len(y_pred)), 4
            ),
        }
        for class_index in range(num_classes)
    ]

    roc_points: list[dict[str, float]] = []
    roc_auc: float | None = None
    threshold_analysis: dict[str, Any] | None = None
    calibration_summary: dict[str, Any] | None = None

    visualization_dir = layout.visualizations_dir
    visualization_dir.mkdir(parents=True, exist_ok=True)

    roc_curve_path = visualization_dir / "roc_curve.png"
    if np.unique(y_true_binary).size > 1:
        fpr, tpr, thresholds = roc_curve(y_true_binary, positive_scores)
        roc_auc = float(auc(fpr, tpr))
        roc_points = [
            {
                "fpr": round(float(fpr_value), 4),
                "tpr": round(float(tpr_value), 4),
                "threshold": round(float(threshold_value), 4),
            }
            for fpr_value, tpr_value, threshold_value in zip(
                fpr, tpr, thresholds, strict=False
            )
        ]
        roc_figure, _ = plot_roc_curve(
            y_true_binary,
            positive_scores,
            title=f"ROC Curve ({positive_class_label} vs Rest)",
            save_path=roc_curve_path,
        )
        plt.close(roc_figure)

        finite_threshold_indices = np.flatnonzero(np.isfinite(thresholds))
        if finite_threshold_indices.size > 0:
            optimal_index = int(
                finite_threshold_indices[
                    np.argmax((tpr - fpr)[finite_threshold_indices])
                ]
            )
            optimal_threshold = float(thresholds[optimal_index])
            optimal_prediction = (positive_scores >= optimal_threshold).astype(int)
            threshold_matrix = confusion_matrix(
                y_true_binary, optimal_prediction, labels=[0, 1]
            )
            tn, fp, fn, tp = threshold_matrix.ravel()
            threshold_analysis = {
                "threshold": round(optimal_threshold, 4),
                "youden_j": round(float(tpr[optimal_index] - fpr[optimal_index]), 4),
                "sensitivity": round(_safe_rate(tp, tp + fn), 4),
                "specificity": round(_safe_rate(tn, tn + fp), 4),
                "ppv": round(_safe_rate(tp, tp + fp), 4),
                "npv": round(_safe_rate(tn, tn + fn), 4),
                "confusion_matrix": threshold_matrix.astype(int).tolist(),
            }

        brier_score = float(np.mean((positive_scores - y_true_binary) ** 2))
        ece, calibration_bins = _compute_expected_calibration_error(
            y_true_binary, positive_scores
        )
        calibration_summary = {
            "positive_class_label": positive_class_label,
            "brier_score": round(brier_score, 4),
            "ece": ece,
            "n_bins": 10,
            "bins": calibration_bins,
        }
    else:
        roc_curve_path = None

    confusion_matrix_path = visualization_dir / "confusion_matrix.png"
    confusion_figure, _ = plot_confusion_matrix(
        y_true,
        y_pred,
        class_names=labels,
        title="Confusion Matrix",
        save_path=confusion_matrix_path,
    )
    plt.close(confusion_figure)

    normalized_confusion_matrix_path = (
        visualization_dir / "confusion_matrix_normalized.png"
    )
    normalized_confusion_figure, _ = plot_confusion_matrix(
        y_true,
        y_pred,
        class_names=labels,
        title="Normalized Confusion Matrix",
        normalize="true",
        save_path=normalized_confusion_matrix_path,
    )
    plt.close(normalized_confusion_figure)

    has_train_accuracy = any(
        entry.get("train_accuracy") is not None for entry in history_entries
    )
    has_val_accuracy = any(
        entry.get("val_accuracy") is not None for entry in history_entries
    )

    training_curves_path = None
    if history_entries:
        training_curves_path = visualization_dir / "training_curves.png"
        training_curve_figure, _ = plot_training_curves(
            [entry.get("train_loss", 0.0) for entry in history_entries],
            val_losses=[entry.get("val_loss", 0.0) for entry in history_entries],
            train_metrics=(
                {
                    "Accuracy": [
                        entry.get("train_accuracy", 0.0) for entry in history_entries
                    ]
                }
                if has_train_accuracy
                else None
            ),
            val_metrics=(
                {
                    "Accuracy": [
                        entry.get("val_accuracy", 0.0) for entry in history_entries
                    ]
                }
                if has_val_accuracy
                else None
            ),
            title="Training Curves",
            save_path=training_curves_path,
        )
        plt.close(training_curve_figure)

    calibration_curve_path = None
    probability_distribution_path = None
    if np.unique(y_true_binary).size > 1:
        calibration_curve_path = visualization_dir / "calibration_curve.png"
        calibration_figure, _ = plot_calibration_curve(
            y_true_binary,
            positive_scores,
            title=f"Calibration Curve ({positive_class_label})",
            save_path=calibration_curve_path,
        )
        plt.close(calibration_figure)

        probability_distribution_path = (
            visualization_dir / "probability_distribution.png"
        )
        probability_figure, _ = plot_probability_distribution(
            y_true_binary,
            positive_scores,
            title=f"Prediction Probability Distribution ({positive_class_label})",
            save_path=probability_distribution_path,
        )
        plt.close(probability_figure)

    roc_payload = {
        "artifact_key": "roc_curve_plot",
        "positive_class_label": positive_class_label,
        "auc": round(float(roc_auc), 4) if roc_auc is not None else None,
        "points": roc_points,
        "plot_path": str(roc_curve_path) if roc_curve_path else None,
    }
    confusion_payload = {
        "artifact_key": "confusion_matrix_plot",
        "labels": labels,
        "matrix": matrix.tolist(),
        "plot_path": str(confusion_matrix_path),
        "normalized_plot_path": str(normalized_confusion_matrix_path),
    }
    validation_overview = {
        "split": split,
        "sample_count": int(len(y_true)),
        "num_classes": num_classes,
        "positive_class_label": positive_class_label,
        "positive_prevalence": round(
            _safe_rate(y_true_binary.sum(), len(y_true_binary)), 4
        ),
        "accuracy": round(float(accuracy), 4),
        "balanced_accuracy": round(balanced_accuracy, 4),
        "precision_macro": round(float(precision_macro), 4),
        "recall_macro": round(float(recall_macro), 4),
        "macro_f1": round(float(f1_macro), 4),
        "weighted_f1": round(float(f1_weighted), 4),
        "auc": round(float(roc_auc), 4) if roc_auc is not None else None,
        "mean_confidence": round(mean_confidence, 4),
        "error_count": error_count,
        "error_rate": round(_safe_rate(error_count, len(y_true)), 4),
        "best_epoch": max(
            (
                entry.get("epoch", 0)
                for entry in history_entries
                if entry.get("best_so_far")
            ),
            default=None,
        ),
    }
    metrics_payload = {
        "meta": dict(artifact_metadata),
        "accuracy": round(float(accuracy), 4),
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "f1_score": round(float(f1_score), 4),
        "auc": round(float(roc_auc), 4) if roc_auc is not None else None,
        "balanced_accuracy": round(balanced_accuracy, 4),
        "precision_macro": round(float(precision_macro), 4),
        "recall_macro": round(float(recall_macro), 4),
        "macro_f1": round(float(f1_macro), 4),
        "weighted_f1": round(float(f1_weighted), 4),
        "mean_confidence": round(mean_confidence, 4),
        "mean_confidence_correct": _round_metric(mean_confidence_correct),
        "mean_confidence_error": _round_metric(mean_confidence_error),
        "error_rate": round(_safe_rate(error_count, len(y_true)), 4),
        "best_accuracy": round(
            float(
                max(
                    (entry.get("val_accuracy", 0.0) for entry in history_entries),
                    default=accuracy,
                )
            ),
            4,
        ),
        "best_loss": round(
            float(
                min(
                    (entry.get("val_loss", accuracy) for entry in history_entries),
                    default=0.0,
                )
            ),
            4,
        ),
        "best_epoch": validation_overview["best_epoch"],
        "final_accuracy": round(
            float(history_entries[-1].get("val_accuracy", accuracy)), 4
        )
        if history_entries
        else round(float(accuracy), 4),
        "final_loss": round(float(history_entries[-1].get("val_loss", 0.0)), 4)
        if history_entries
        else None,
        "progress": 100.0,
    }
    if threshold_analysis:
        metrics_payload.update(
            {
                "sensitivity": threshold_analysis["sensitivity"],
                "specificity": threshold_analysis["specificity"],
                "ppv": threshold_analysis["ppv"],
                "npv": threshold_analysis["npv"],
                "optimal_threshold": threshold_analysis["threshold"],
            }
        )
    if calibration_summary:
        metrics_payload.update(
            {
                "brier_score": calibration_summary["brier_score"],
                "ece": calibration_summary["ece"],
            }
        )
    if survival_payload:
        metrics_payload["c_index"] = survival_payload.get("c_index")

    validation_payload = {
        "meta": dict(artifact_metadata),
        "dataset": {
            "name": dataset_name,
            "labels": labels,
            "num_classes": num_classes,
            "sample_count": int(len(y_true)),
            "class_distribution": [
                {
                    "label": label,
                    "count": int(class_supports[index]),
                    "rate": round(_safe_rate(class_supports[index], len(y_true)), 4),
                }
                for index, label in enumerate(labels)
            ],
        },
        "overview": validation_overview,
        "per_class": per_class_payload,
        "prediction_summary": {
            "mean_confidence": round(mean_confidence, 4),
            "mean_confidence_correct": _round_metric(mean_confidence_correct),
            "mean_confidence_error": _round_metric(mean_confidence_error),
            "error_count": error_count,
            "error_rate": round(_safe_rate(error_count, len(y_true)), 4),
            "top_misclassifications": misclassification_pairs[:5],
        },
        "threshold_analysis": threshold_analysis,
        "calibration": calibration_summary,
        "survival": survival_payload,
        "global_feature_importance": importance_payload,
    }

    predictions_payload = {
        "labels": labels,
        "num_classes": num_classes,
        "positive_class_index": positive_class_index,
        "y_true": y_true.tolist(),
        "y_pred": y_pred.tolist(),
        "probabilities": np.round(probabilities, 6).tolist(),
        "risk_scores": np.round(inference.risk_scores, 6).tolist(),
        "risk_score_source": inference.risk_score_source,
        "samples": inference.sample_records,
    }

    roc_curve_json_path = layout.roc_curve_json_path
    confusion_matrix_json_path = layout.confusion_matrix_json_path
    metrics_path = layout.metrics_path
    validation_path = layout.validation_path
    predictions_path = layout.predictions_path

    _write_json(roc_curve_json_path, roc_payload)
    _write_json(confusion_matrix_json_path, confusion_payload)
    _write_json(metrics_path, metrics_payload)
    _write_json(validation_path, validation_payload)
    _write_json(predictions_path, predictions_payload)

    artifact_paths = {
        "prediction_path": str(predictions_path),
        "predictions_path": str(predictions_path),
        "roc_curve_json_path": str(roc_curve_json_path),
        "roc_curve_plot_path": str(roc_curve_path) if roc_curve_path else None,
        "confusion_matrix_json_path": str(confusion_matrix_json_path),
        "confusion_matrix_plot_path": str(confusion_matrix_path),
        "confusion_matrix_normalized_plot_path": str(normalized_confusion_matrix_path),
        "training_curves_plot_path": str(training_curves_path)
        if training_curves_path
        else None,
        "validation_path": str(validation_path),
        "calibration_curve_plot_path": str(calibration_curve_path)
        if calibration_curve_path
        else None,
        "probability_distribution_plot_path": str(probability_distribution_path)
        if probability_distribution_path
        else None,
        **attention_artifacts,
    }
    artifact_paths = {key: value for key, value in artifact_paths.items() if value}
    return metrics_payload, validation_payload, artifact_paths


def _write_summary_and_report(
    layout: RunOutputLayout,
    config_path: Path,
    checkpoint_path: Path,
    split: str,
    config: Any,
    metrics_payload: dict[str, Any],
    validation_payload: dict[str, Any],
    history_payload: dict[str, Any],
    artifact_paths: dict[str, str],
    artifact_metadata: dict[str, Any],
) -> dict[str, str]:
    config_snapshot_path = layout.config_snapshot_path
    summary_path = layout.summary_path
    report_path = layout.report_path
    metrics_path = layout.metrics_path
    history_path = layout.history_path

    config_snapshot_path.write_text(
        json.dumps(config.to_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    summary_artifact_paths = {
        "checkpoint_path": str(checkpoint_path),
        "config_path": str(config_snapshot_path),
        "metrics_path": str(metrics_path),
        "validation_path": str(layout.validation_path),
        "prediction_path": str(layout.predictions_path),
        "predictions_path": str(layout.predictions_path),
        "summary_path": str(summary_path),
        "report_path": str(report_path),
        "log_path": str(layout.training_log_path),
        "history_path": str(history_path),
        **artifact_paths,
    }
    validation_overview = validation_payload.get("overview", {})
    summary_payload = {
        "meta": dict(artifact_metadata),
        "experiment_name": config.experiment_name,
        "dataset_name": validation_payload.get("dataset", {}).get("name"),
        "backbone": config.model.vision.backbone,
        "num_classes": config.model.num_classes,
        "split": split,
        "checkpoint_path": str(checkpoint_path),
        "source_config_path": str(config_path),
        "artifacts": summary_artifact_paths,
        "metrics": metrics_payload,
        "validation_overview": validation_overview,
        "survival": validation_payload.get("survival") or None,
        "global_feature_importance": validation_payload.get(
            "global_feature_importance"
        )
        or None,
    }
    _write_json(summary_path, summary_payload)

    generate_result_artifact_report(
        report_path=report_path,
        experiment_name=config.experiment_name,
        config_path=config_path,
        checkpoint_path=checkpoint_path,
        split=split,
        metrics_payload=metrics_payload,
        validation_payload=validation_payload,
        history_payload=history_payload,
        artifact_paths=summary_artifact_paths,
        artifact_metadata=artifact_metadata,
        backbone=config.model.vision.backbone,
    )

    return {
        "config_path": str(config_snapshot_path),
        "metrics_path": str(metrics_path),
        "summary_path": str(summary_path),
        "report_path": str(report_path),
        "log_path": str(layout.training_log_path),
        "history_path": str(history_path),
    }


def build_results_artifacts(
    config_path: str | Path,
    checkpoint_path: str | Path,
    output_dir: str | Path | None = None,
    split: str = "test",
    attention_samples: int = 4,
    enable_survival: bool = True,
    survival_time_column: str | None = None,
    survival_event_column: str | None = None,
    enable_importance: bool = True,
    importance_sample_limit: int = 128,
) -> BuildResultsOutput:
    """Generate validation artifacts from a real config and checkpoint."""
    if split not in {"train", "val", "test"}:
        raise ValueError(f"Unsupported split: {split}")

    config_path = Path(config_path)
    checkpoint_path = Path(checkpoint_path)
    config = load_config(config_path)
    actual_output_dir = resolve_run_output_dir(
        config_output_dir=config.logging.output_dir,
        checkpoint_path=checkpoint_path,
        override=output_dir,
    )
    layout = RunOutputLayout(actual_output_dir).ensure_exists()

    if (
        config.data.dataset_type == "three_phase_ct_tabular"
        or config.model.model_type == "three_phase_ct_fusion"
    ):
        return _build_three_phase_results(
            config=config,
            config_path=config_path,
            checkpoint_path=checkpoint_path,
            layout=layout,
            split=split,
            enable_importance=enable_importance,
            importance_sample_limit=importance_sample_limit,
        )

    dataset = _load_split_dataset(config, split)
    device = _resolve_device(config.device)
    model = _build_model(config, dataset.get_tabular_dim())
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)

    history_payload = _load_history(layout)
    if history_payload.get("entries"):
        _write_json(layout.history_path, history_payload)

    inference = _run_inference(
        model=model,
        dataset=dataset,
        device=device,
        batch_size=config.data.batch_size,
    )
    resolved_time_column, resolved_event_column = resolve_survival_columns(
        config,
        override_time_column=survival_time_column,
        override_event_column=survival_event_column,
    )
    survival_cutoff, survival_cutoff_source = _resolve_survival_cutoff(
        config=config,
        model=model,
        device=device,
        batch_size=config.data.batch_size,
        split=split,
        time_column=resolved_time_column,
        event_column=resolved_event_column,
    )
    attention_artifacts = _generate_attention_artifacts(
        model=model,
        dataset=dataset,
        device=device,
        layout=layout,
        config=config,
        inference=inference,
        sample_limit=attention_samples,
    )
    artifact_metadata = _build_artifact_metadata(
        config_path=config_path,
        checkpoint_path=checkpoint_path,
        split=split,
    )
    survival_payload, survival_artifact_paths = (None, {})
    if enable_survival:
        survival_payload, survival_artifact_paths = build_survival_artifacts(
            dataset=dataset,
            output_dir=layout.artifacts_dir,
            risk_scores=inference.risk_scores,
            risk_score_source=inference.risk_score_source,
            time_column=resolved_time_column,
            event_column=resolved_event_column,
            cutoff=survival_cutoff,
            cutoff_source=survival_cutoff_source,
            split=split,
        )

    def _score_fn(image_batch: torch.Tensor, tabular_batch: torch.Tensor) -> np.ndarray:
        outputs = model(image_batch, tabular_batch)
        logits = outputs["logits"] if isinstance(outputs, dict) else outputs
        probabilities = torch.softmax(logits, dim=1)
        if probabilities.shape[1] == 2:
            return probabilities[:, 1].detach().cpu().numpy()
        return probabilities.max(dim=1).values.detach().cpu().numpy()

    importance_payload, importance_artifact_paths = (None, {})
    if enable_importance and importance_sample_limit > 0:
        importance_payload, importance_artifact_paths = (
            build_global_feature_importance_artifacts(
                dataset=dataset,
                device=device,
                output_dir=layout.artifacts_dir,
                batch_size=min(max(config.data.batch_size, 1), 16),
                sample_limit=max(importance_sample_limit, 0),
                score_name=inference.risk_score_source,
                score_fn=_score_fn,
            )
        )
    metrics_payload, validation_payload, visualization_paths = _build_metrics_payload(
        layout=layout,
        inference=inference,
        history_entries=history_payload.get("entries", []),
        dataset_name=Path(config.data.csv_path).stem,
        split=split,
        attention_artifacts=attention_artifacts,
        survival_payload=survival_payload,
        importance_payload=importance_payload,
        artifact_metadata=artifact_metadata,
    )
    visualization_paths = {
        **visualization_paths,
        **survival_artifact_paths,
        **importance_artifact_paths,
    }
    summary_paths = _write_summary_and_report(
        layout=layout,
        config_path=config_path,
        checkpoint_path=checkpoint_path,
        split=split,
        config=config,
        metrics_payload=metrics_payload,
        validation_payload=validation_payload,
        history_payload=history_payload,
        artifact_paths=visualization_paths,
        artifact_metadata=artifact_metadata,
    )
    artifact_paths = {
        **summary_paths,
        **visualization_paths,
    }
    return BuildResultsOutput(
        output_dir=str(layout.root_dir),
        artifact_paths=artifact_paths,
        metrics=metrics_payload,
        validation=validation_payload,
    )
