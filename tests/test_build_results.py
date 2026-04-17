"""Tests for post-training artifact generation."""

import json
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.tensorboard import SummaryWriter

from med_core.backbones import create_tabular_backbone, create_vision_backbone
from med_core.configs import load_config, save_config
from med_core.datasets import MedicalMultimodalDataset, split_dataset
from med_core.fusion import MultiModalFusionModel, create_fusion_module
from med_core.postprocessing import build_results_artifacts


def _create_mock_dataset_root(tmp_path: Path, sample_count: int = 20) -> tuple[Path, Path]:
    data_root = tmp_path / "mock-data"
    image_dir = data_root / "images"
    image_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    for index in range(sample_count):
        image_name = f"patient_{index:03d}.png"
        image_path = image_dir / image_name
        pixel_value = 32 + (index * 7) % 180
        Image.new("RGB", (64, 64), color=(pixel_value, pixel_value, pixel_value)).save(
            image_path
        )
        rows.append(
            {
                "patient_id": f"P{index:03d}",
                "image_path": f"images/{image_name}",
                "age": 30 + index,
                "gender": index % 2,
                "diagnosis": index % 2,
            }
        )

    csv_path = data_root / "metadata.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    return csv_path, data_root


def _create_checkpoint_and_logs(
    tmp_path: Path,
    *,
    include_survival: bool = False,
) -> tuple[Path, Path]:
    config = load_config("configs/starter/quickstart.yaml")
    mock_csv_path, mock_image_dir = _create_mock_dataset_root(tmp_path)
    config.logging.output_dir = str(tmp_path / "run")
    config.data.image_size = 64
    config.data.batch_size = 8
    config.data.num_workers = 0
    config.data.csv_path = str(mock_csv_path)
    config.data.image_dir = str(mock_image_dir)
    config.model.vision.pretrained = False
    if include_survival:
        source_csv = Path(config.data.csv_path)
        dataframe = pd.read_csv(source_csv)
        dataframe["survival_time"] = [
            30 + (index * 17) % 720 for index in range(len(dataframe))
        ]
        dataframe["event"] = [1 if index % 3 else 0 for index in range(len(dataframe))]
        survival_csv_path = tmp_path / "metadata_with_survival.csv"
        dataframe.to_csv(survival_csv_path, index=False)
        config.data.csv_path = str(survival_csv_path)
        config.data.survival_time_column = "survival_time"
        config.data.survival_event_column = "event"

    config_path = tmp_path / "quickstart-test.yaml"
    save_config(config, config_path)

    dataset, _ = MedicalMultimodalDataset.from_csv(
        csv_path=config.data.csv_path,
        image_dir=config.data.image_dir,
        image_column=config.data.image_path_column,
        target_column=config.data.target_column,
        numerical_features=config.data.numerical_features,
        categorical_features=config.data.categorical_features,
        patient_id_column=config.data.patient_id_column,
        transform=None,
    )
    train_ds, _, _ = split_dataset(
        dataset,
        train_ratio=config.data.train_ratio,
        val_ratio=config.data.val_ratio,
        test_ratio=config.data.test_ratio,
        random_seed=config.data.random_seed,
    )

    model = MultiModalFusionModel(
        vision_backbone=create_vision_backbone(
            backbone_name=config.model.vision.backbone,
            pretrained=False,
            freeze=config.model.vision.freeze_backbone,
            feature_dim=config.model.vision.feature_dim,
            dropout=config.model.vision.dropout,
            attention_type=config.model.vision.attention_type,
            enable_attention_supervision=config.model.vision.enable_attention_supervision,
        ),
        tabular_backbone=create_tabular_backbone(
            input_dim=train_ds.get_tabular_dim(),
            output_dim=config.model.tabular.output_dim,
            hidden_dims=config.model.tabular.hidden_dims,
            dropout=config.model.tabular.dropout,
        ),
        fusion_module=create_fusion_module(
            fusion_type=config.model.fusion.fusion_type,
            vision_dim=config.model.vision.feature_dim,
            tabular_dim=config.model.tabular.output_dim,
            output_dim=config.model.fusion.hidden_dim,
            dropout=config.model.fusion.dropout,
        ),
        num_classes=config.model.num_classes,
        use_auxiliary_heads=config.model.use_auxiliary_heads,
    )

    checkpoint_dir = Path(config.logging.output_dir) / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / "best.pth"
    torch.save(
        {
            "epoch": 2,
            "global_step": 12,
            "model_state_dict": model.state_dict(),
            "metrics": {"accuracy": 0.5},
            "config": config.to_dict(),
        },
        checkpoint_path,
    )

    writer = SummaryWriter(log_dir=str(Path(config.logging.output_dir) / "logs"))
    for step, train_loss, val_loss, val_accuracy in [
        (0, 0.92, 0.81, 0.52),
        (1, 0.78, 0.71, 0.61),
        (2, 0.66, 0.64, 0.69),
    ]:
        writer.add_scalar("train/loss", train_loss, step)
        writer.add_scalar("val/loss", val_loss, step)
        writer.add_scalar("val/accuracy", val_accuracy, step)
        writer.add_scalar("learning_rate", 0.001 * (0.8**step), step)
    writer.close()

    return config_path, checkpoint_path


def _assert_contract_meta(
    payload: dict[str, object],
    *,
    split: str,
    config_path: Path,
    checkpoint_path: Path,
) -> None:
    meta = payload["meta"]
    assert meta["schema_version"] == "1.0"
    assert meta["generated_by"] == "medfusion.build_results"
    assert meta["generated_at"]
    assert meta["source_config_path"] == str(config_path)
    assert meta["checkpoint_path"] == str(checkpoint_path)
    assert meta["split"] == split


def _assert_report_contract(
    report_text: str,
    *,
    output_dir: Path,
    split: str,
) -> None:
    assert "## Contract Metadata" in report_text
    assert "- Schema Version: 1.0" in report_text
    assert "- Generated By: medfusion.build_results" in report_text
    assert f"- Split: {split}" in report_text
    assert "## Artifact Index" in report_text
    assert f"- metrics.json: {output_dir / 'metrics' / 'metrics.json'}" in report_text
    assert f"- validation.json: {output_dir / 'metrics' / 'validation.json'}" in report_text
    assert f"- predictions.json: {output_dir / 'metrics' / 'predictions.json'}" in report_text
    assert f"- summary.json: {output_dir / 'reports' / 'summary.json'}" in report_text
    assert f"- report.md: {output_dir / 'reports' / 'report.md'}" in report_text
    assert "## Per-class Metrics" in report_text
    assert "## Threshold Analysis" in report_text
    assert "## History" in report_text
    assert "- 总体准确率:" in report_text
    assert "- 区分能力（AUC）:" in report_text
    assert "- 最优阈值:" in report_text
    assert "- 敏感度:" in report_text
    assert "- 特异度:" in report_text
    assert "![ROC 曲线（区分能力）](../artifacts/visualizations/roc_curve.png)" in report_text
    assert "![训练曲线](../artifacts/visualizations/training_curves.png)" in report_text
    assert "![混淆矩阵（阳性/阴性判别情况）](../artifacts/visualizations/confusion_matrix.png)" in report_text


def test_build_results_artifacts_from_real_checkpoint(tmp_path: Path) -> None:
    config_path, checkpoint_path = _create_checkpoint_and_logs(tmp_path)

    result = build_results_artifacts(
        config_path=config_path,
        checkpoint_path=checkpoint_path,
        split="train",
        attention_samples=2,
    )

    output_dir = Path(result.output_dir)
    assert output_dir.exists()
    assert result.validation["overview"]["split"] == "train"
    assert result.validation["overview"]["sample_count"] > 0
    assert 0.0 <= result.metrics["accuracy"] <= 1.0
    assert result.validation.get("survival") is None
    assert result.validation["global_feature_importance"]["top_features"]
    assert "validation_path" in result.artifact_paths

    expected_files = [
        output_dir / "metrics" / "metrics.json",
        output_dir / "metrics" / "validation.json",
        output_dir / "reports" / "summary.json",
        output_dir / "reports" / "report.md",
        output_dir / "logs" / "history.json",
        output_dir / "artifacts" / "visualizations" / "confusion_matrix.png",
        output_dir / "artifacts" / "visualizations" / "training_curves.png",
        output_dir
        / "artifacts"
        / "visualizations"
        / "attention"
        / "attention_maps.json",
    ]
    for path in expected_files:
        assert path.exists(), f"Missing artifact: {path}"

    metrics_data = json.loads(
        (output_dir / "metrics" / "metrics.json").read_text(encoding="utf-8")
    )
    validation_data = json.loads(
        (output_dir / "metrics" / "validation.json").read_text(encoding="utf-8")
    )
    summary_data = json.loads(
        (output_dir / "reports" / "summary.json").read_text(encoding="utf-8")
    )
    report_text = (output_dir / "reports" / "report.md").read_text(encoding="utf-8")

    for payload in (metrics_data, validation_data, summary_data):
        _assert_contract_meta(
            payload,
            split="train",
            config_path=config_path,
            checkpoint_path=checkpoint_path,
        )

    assert result.artifact_paths["prediction_path"] == str(
        output_dir / "metrics" / "predictions.json"
    )
    assert result.artifact_paths["predictions_path"] == str(
        output_dir / "metrics" / "predictions.json"
    )
    assert summary_data["meta"] == metrics_data["meta"]
    assert summary_data["meta"] == validation_data["meta"]
    assert summary_data["validation_overview"] == validation_data["overview"]
    assert summary_data["metrics"]["accuracy"] == metrics_data["accuracy"]
    assert summary_data["artifacts"]["metrics_path"] == str(
        output_dir / "metrics" / "metrics.json"
    )
    assert summary_data["artifacts"]["validation_path"] == str(
        output_dir / "metrics" / "validation.json"
    )
    assert summary_data["artifacts"]["prediction_path"] == str(
        output_dir / "metrics" / "predictions.json"
    )
    assert summary_data["artifacts"]["predictions_path"] == str(
        output_dir / "metrics" / "predictions.json"
    )
    assert summary_data["artifacts"]["summary_path"] == str(
        output_dir / "reports" / "summary.json"
    )
    assert summary_data["artifacts"]["report_path"] == str(
        output_dir / "reports" / "report.md"
    )
    _assert_report_contract(report_text, output_dir=output_dir, split="train")


def test_build_results_artifacts_include_survival_and_importance_when_configured(
    tmp_path: Path,
) -> None:
    config_path, checkpoint_path = _create_checkpoint_and_logs(
        tmp_path,
        include_survival=True,
    )

    result = build_results_artifacts(
        config_path=config_path,
        checkpoint_path=checkpoint_path,
        split="train",
        attention_samples=2,
        importance_sample_limit=8,
    )

    output_dir = Path(result.output_dir)
    report_path = output_dir / "reports" / "report.md"
    report_text = report_path.read_text(encoding="utf-8")

    assert result.validation["survival"]["c_index"] is not None
    assert result.validation["survival"]["sample_count"] > 0
    assert result.validation["global_feature_importance"]["top_features"]
    assert result.metrics["c_index"] is not None
    assert result.artifact_paths["survival_path"].endswith("survival.json")
    assert result.artifact_paths["kaplan_meier_plot_path"].endswith(
        "kaplan_meier_curve.png"
    )
    assert result.artifact_paths["risk_score_distribution_plot_path"].endswith(
        "risk_score_distribution.png"
    )
    assert result.artifact_paths["feature_importance_path"].endswith(
        "feature_importance.json"
    )
    assert result.artifact_paths["feature_importance_bar_plot_path"].endswith(
        "feature_importance_bar.png"
    )
    assert result.artifact_paths["feature_importance_beeswarm_plot_path"].endswith(
        "feature_importance_beeswarm.png"
    )
    assert "## 生存分析" in report_text
    assert "## 关键影响因素" in report_text
    assert "- 风险分层依据:" in report_text
    assert "- 方法说明:" in report_text
    assert "- 关键影响因素评分来源:" in report_text
    assert (
        "![Kaplan-Meier 生存曲线](../artifacts/visualizations/kaplan_meier_curve.png)"
        in report_text
    )
    assert (
        "![关键影响因素条形图](../artifacts/visualizations/feature_importance_bar.png)"
        in report_text
    )
