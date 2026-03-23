"""Build real validation artifacts from a trained config + checkpoint."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
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
    get_val_transforms,
    split_dataset,
)
from med_core.evaluation.interpretability import GradCAM
from med_core.fusion import MultiModalFusionModel, create_fusion_module
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


@dataclass
class InferenceArtifacts:
    """In-memory inference payload for artifact generation."""

    labels: list[str]
    num_classes: int
    positive_class_index: int
    y_true: np.ndarray
    y_pred: np.ndarray
    probabilities: np.ndarray
    sample_records: list[dict[str, Any]]


@dataclass
class BuildResultsOutput:
    """Final artifact metadata returned by build_results_artifacts."""

    output_dir: str
    artifact_paths: dict[str, str]
    metrics: dict[str, Any]
    validation: dict[str, Any]


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


def _infer_output_dir(config_output_dir: str, checkpoint_path: Path, override: str | Path | None) -> Path:
    if override is not None:
        return Path(override)
    if checkpoint_path.parent.name == "checkpoints":
        return checkpoint_path.parent.parent
    return Path(config_output_dir)


def _load_history(output_dir: Path) -> dict[str, Any]:
    history_path = output_dir / "history.json"
    if history_path.exists():
        payload = _read_json(history_path)
        if payload.get("entries"):
            return payload

    log_dir = output_dir / "logs"
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
    sample_records = [
        {
            "index": index,
            "true_label": labels[int(y_true_array[index])],
            "predicted_label": labels[int(y_pred_array[index])],
            "confidence": round(float(confidence[index]), 4),
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
        sample_records=sample_records,
    )


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


def _select_attention_indices(y_true: np.ndarray, y_pred: np.ndarray, probabilities: np.ndarray, limit: int) -> list[int]:
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
    output_dir: Path,
    config: Any,
    inference: InferenceArtifacts,
    sample_limit: int,
) -> dict[str, Any]:
    if sample_limit <= 0:
        return {}

    attention_dir = output_dir / "visualizations" / "attention"
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
                "predicted_label": inference.sample_records[sample_index]["predicted_label"],
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
    output_dir: Path,
    inference: InferenceArtifacts,
    history_entries: list[dict[str, Any]],
    dataset_name: str | None,
    split: str,
    attention_artifacts: dict[str, Any],
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
    balanced_accuracy = float(np.mean(per_class_recall)) if len(per_class_recall) else 0.0
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
    mean_confidence_correct = float(np.mean(confidence[correct_mask])) if bool(correct_mask.any()) else None
    mean_confidence_error = float(np.mean(confidence[~correct_mask])) if error_count > 0 else None

    per_class_payload = [
        {
            "label": labels[class_index],
            "support": int(per_class_support[class_index]),
            "prevalence": round(_safe_rate(class_supports[class_index], len(y_true)), 4),
            "precision": round(float(per_class_precision[class_index]), 4),
            "recall": round(float(per_class_recall[class_index]), 4),
            "f1_score": round(float(per_class_f1[class_index]), 4),
            "predicted_count": int(predicted_distribution[class_index]),
            "predicted_rate": round(_safe_rate(predicted_distribution[class_index], len(y_pred)), 4),
        }
        for class_index in range(num_classes)
    ]

    roc_points: list[dict[str, float]] = []
    roc_auc: float | None = None
    threshold_analysis: dict[str, Any] | None = None
    calibration_summary: dict[str, Any] | None = None

    visualization_dir = output_dir / "visualizations"
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
            for fpr_value, tpr_value, threshold_value in zip(fpr, tpr, thresholds, strict=False)
        ]
        roc_figure, _ = plot_roc_curve(
            y_true_binary,
            positive_scores,
            title=f"ROC Curve ({positive_class_label} vs Rest)",
            save_path=roc_curve_path,
        )
        roc_figure.clf()

        finite_threshold_indices = np.flatnonzero(np.isfinite(thresholds))
        if finite_threshold_indices.size > 0:
            optimal_index = int(
                finite_threshold_indices[np.argmax((tpr - fpr)[finite_threshold_indices])]
            )
            optimal_threshold = float(thresholds[optimal_index])
            optimal_prediction = (positive_scores >= optimal_threshold).astype(int)
            threshold_matrix = confusion_matrix(y_true_binary, optimal_prediction, labels=[0, 1])
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
        ece, calibration_bins = _compute_expected_calibration_error(y_true_binary, positive_scores)
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
    confusion_figure.clf()

    normalized_confusion_matrix_path = visualization_dir / "confusion_matrix_normalized.png"
    normalized_confusion_figure, _ = plot_confusion_matrix(
        y_true,
        y_pred,
        class_names=labels,
        title="Normalized Confusion Matrix",
        normalize="true",
        save_path=normalized_confusion_matrix_path,
    )
    normalized_confusion_figure.clf()

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
                        entry.get("train_accuracy", 0.0)
                        for entry in history_entries
                    ]
                }
                if has_train_accuracy
                else None
            ),
            val_metrics=(
                {
                    "Accuracy": [
                        entry.get("val_accuracy", 0.0)
                        for entry in history_entries
                    ]
                }
                if has_val_accuracy
                else None
            ),
            title="Training Curves",
            save_path=training_curves_path,
        )
        training_curve_figure.clf()

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
        calibration_figure.clf()

        probability_distribution_path = visualization_dir / "probability_distribution.png"
        probability_figure, _ = plot_probability_distribution(
            y_true_binary,
            positive_scores,
            title=f"Prediction Probability Distribution ({positive_class_label})",
            save_path=probability_distribution_path,
        )
        probability_figure.clf()

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
        "positive_prevalence": round(_safe_rate(y_true_binary.sum(), len(y_true_binary)), 4),
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
        "best_epoch": max((entry.get("epoch", 0) for entry in history_entries if entry.get("best_so_far")), default=None),
    }
    metrics_payload = {
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
        "best_accuracy": round(float(max((entry.get("val_accuracy", 0.0) for entry in history_entries), default=accuracy)), 4),
        "best_loss": round(float(min((entry.get("val_loss", accuracy) for entry in history_entries), default=0.0)), 4),
        "best_epoch": validation_overview["best_epoch"],
        "final_accuracy": round(float(history_entries[-1].get("val_accuracy", accuracy)), 4) if history_entries else round(float(accuracy), 4),
        "final_loss": round(float(history_entries[-1].get("val_loss", 0.0)), 4) if history_entries else None,
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

    validation_payload = {
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
    }

    predictions_payload = {
        "labels": labels,
        "num_classes": num_classes,
        "positive_class_index": positive_class_index,
        "y_true": y_true.tolist(),
        "y_pred": y_pred.tolist(),
        "probabilities": np.round(probabilities, 6).tolist(),
        "samples": inference.sample_records,
    }

    roc_curve_json_path = output_dir / "roc_curve.json"
    confusion_matrix_json_path = output_dir / "confusion_matrix.json"
    metrics_path = output_dir / "metrics.json"
    validation_path = output_dir / "validation.json"
    predictions_path = output_dir / "predictions.json"

    _write_json(roc_curve_json_path, roc_payload)
    _write_json(confusion_matrix_json_path, confusion_payload)
    _write_json(metrics_path, metrics_payload)
    _write_json(validation_path, validation_payload)
    _write_json(predictions_path, predictions_payload)

    artifact_paths = {
        "prediction_path": str(predictions_path),
        "roc_curve_json_path": str(roc_curve_json_path),
        "roc_curve_plot_path": str(roc_curve_path) if roc_curve_path else None,
        "confusion_matrix_json_path": str(confusion_matrix_json_path),
        "confusion_matrix_plot_path": str(confusion_matrix_path),
        "confusion_matrix_normalized_plot_path": str(normalized_confusion_matrix_path),
        "training_curves_plot_path": str(training_curves_path) if training_curves_path else None,
        "validation_path": str(validation_path),
        "calibration_curve_plot_path": str(calibration_curve_path) if calibration_curve_path else None,
        "probability_distribution_plot_path": str(probability_distribution_path) if probability_distribution_path else None,
        **attention_artifacts,
    }
    artifact_paths = {key: value for key, value in artifact_paths.items() if value}
    return metrics_payload, validation_payload, artifact_paths


def _write_summary_and_report(
    output_dir: Path,
    config_path: Path,
    checkpoint_path: Path,
    split: str,
    config: Any,
    metrics_payload: dict[str, Any],
    validation_payload: dict[str, Any],
    history_payload: dict[str, Any],
    artifact_paths: dict[str, str],
) -> dict[str, str]:
    config_snapshot_path = output_dir / "training-config.json"
    summary_path = output_dir / "summary.json"
    report_path = output_dir / "report.md"
    metrics_path = output_dir / "metrics.json"
    history_path = output_dir / "history.json"

    config_snapshot_path.write_text(
        json.dumps(config.to_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    history_entries = history_payload.get("entries", [])
    validation_overview = validation_payload.get("overview", {})
    per_class_rows = validation_payload.get("per_class", [])
    top_misclassifications = validation_payload.get("prediction_summary", {}).get("top_misclassifications", [])

    summary_payload = {
        "experiment_name": config.experiment_name,
        "dataset_name": validation_payload.get("dataset", {}).get("name"),
        "backbone": config.model.vision.backbone,
        "num_classes": config.model.num_classes,
        "split": split,
        "checkpoint_path": str(checkpoint_path),
        "source_config_path": str(config_path),
        "artifacts": {
            "checkpoint_path": str(checkpoint_path),
            "config_path": str(config_snapshot_path),
            "summary_path": str(summary_path),
            "report_path": str(report_path),
            "history_path": str(history_path),
            **artifact_paths,
        },
        "metrics": metrics_payload,
        "validation_overview": validation_overview,
    }
    _write_json(summary_path, summary_payload)

    report_lines = [
        f"# {config.experiment_name} 结果报告",
        "",
        "## 运行来源",
        "",
        f"- Config: {config_path}",
        f"- Checkpoint: {checkpoint_path}",
        f"- Split: {split}",
        "",
        "## 实验摘要",
        "",
        f"- 数据集: {validation_payload.get('dataset', {}).get('name') or 'unknown'}",
        f"- Backbone: {config.model.vision.backbone}",
        f"- Accuracy: {metrics_payload.get('accuracy')}",
        f"- AUC: {metrics_payload.get('auc')}",
        f"- F1: {metrics_payload.get('f1_score')}",
        f"- Balanced Accuracy: {metrics_payload.get('balanced_accuracy', '-')}",
        f"- Best Epoch: {metrics_payload.get('best_epoch', '-')}",
        f"- Best Accuracy: {metrics_payload.get('best_accuracy', '-')}",
        f"- Best Loss: {metrics_payload.get('best_loss', '-')}",
        "",
        "## Validation 概览",
        "",
        f"- 样本数: {validation_overview.get('sample_count', '-')}",
        f"- 类别数: {validation_overview.get('num_classes', '-')}",
        f"- 正类标签: {validation_overview.get('positive_class_label', '-')}",
        f"- 正类占比: {validation_overview.get('positive_prevalence', '-')}",
        f"- 宏平均 F1: {validation_overview.get('macro_f1', '-')}",
        f"- 加权 F1: {validation_overview.get('weighted_f1', '-')}",
        f"- 平均置信度: {validation_overview.get('mean_confidence', '-')}",
        f"- 错误率: {validation_overview.get('error_rate', '-')}",
        "",
        "## Per-class Metrics",
        "",
        "| Class | Support | Prevalence | Precision | Recall | F1 | Predicted |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        *[
            "| {label} | {support} | {prevalence} | {precision} | {recall} | {f1_score} | {predicted_count} |".format(
                **row
            )
            for row in per_class_rows
        ],
        "",
        "## Visualizations",
        "",
        "![Training Curves](visualizations/training_curves.png)" if artifact_paths.get("training_curves_plot_path") else "",
        "",
        "![ROC Curve](visualizations/roc_curve.png)" if artifact_paths.get("roc_curve_plot_path") else "",
        "",
        "![Confusion Matrix](visualizations/confusion_matrix.png)",
        "",
        "![Attention Statistics](visualizations/attention/attention_statistics.png)" if artifact_paths.get("attention_statistics_plot_path") else "",
        "",
        "## Threshold Analysis",
        "",
        f"- Optimal Threshold: {metrics_payload.get('optimal_threshold', '-')}",
        f"- Sensitivity: {metrics_payload.get('sensitivity', '-')}",
        f"- Specificity: {metrics_payload.get('specificity', '-')}",
        f"- PPV: {metrics_payload.get('ppv', '-')}",
        f"- NPV: {metrics_payload.get('npv', '-')}",
        "",
        "## 常见误分类",
        "",
        *(
            [f"- {item['actual']} -> {item['predicted']}: {item['count']}" for item in top_misclassifications]
            if top_misclassifications
            else ["- 无明显误分类聚集"]
        ),
        "",
        "## History",
        "",
        f"- total_entries: {len(history_entries)}",
    ]
    report_path.write_text("\n".join(line for line in report_lines if line is not None), encoding="utf-8")

    return {
        "config_path": str(config_snapshot_path),
        "metrics_path": str(metrics_path),
        "summary_path": str(summary_path),
        "report_path": str(report_path),
        "history_path": str(history_path),
    }


def build_results_artifacts(
    config_path: str | Path,
    checkpoint_path: str | Path,
    output_dir: str | Path | None = None,
    split: str = "test",
    attention_samples: int = 4,
) -> BuildResultsOutput:
    """Generate validation artifacts from a real config and checkpoint."""
    if split not in {"train", "val", "test"}:
        raise ValueError(f"Unsupported split: {split}")

    config_path = Path(config_path)
    checkpoint_path = Path(checkpoint_path)
    config = load_config(config_path)
    actual_output_dir = _infer_output_dir(config.logging.output_dir, checkpoint_path, output_dir)
    actual_output_dir.mkdir(parents=True, exist_ok=True)

    dataset = _load_split_dataset(config, split)
    device = torch.device(config.device)
    model = _build_model(config, dataset.get_tabular_dim())
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)

    history_payload = _load_history(actual_output_dir)
    if history_payload.get("entries"):
        _write_json(actual_output_dir / "history.json", history_payload)

    inference = _run_inference(
        model=model,
        dataset=dataset,
        device=device,
        batch_size=config.data.batch_size,
    )
    attention_artifacts = _generate_attention_artifacts(
        model=model,
        dataset=dataset,
        device=device,
        output_dir=actual_output_dir,
        config=config,
        inference=inference,
        sample_limit=attention_samples,
    )
    metrics_payload, validation_payload, visualization_paths = _build_metrics_payload(
        output_dir=actual_output_dir,
        inference=inference,
        history_entries=history_payload.get("entries", []),
        dataset_name=Path(config.data.csv_path).stem,
        split=split,
        attention_artifacts=attention_artifacts,
    )
    summary_paths = _write_summary_and_report(
        output_dir=actual_output_dir,
        config_path=config_path,
        checkpoint_path=checkpoint_path,
        split=split,
        config=config,
        metrics_payload=metrics_payload,
        validation_payload=validation_payload,
        history_payload=history_payload,
        artifact_paths=visualization_paths,
    )
    artifact_paths = {
        **summary_paths,
        **visualization_paths,
    }
    return BuildResultsOutput(
        output_dir=str(actual_output_dir),
        artifact_paths=artifact_paths,
        metrics=metrics_payload,
        validation=validation_payload,
    )
