"""Tests for post-training artifact generation."""

from pathlib import Path

import torch
from torch.utils.tensorboard import SummaryWriter

from med_core.backbones import create_tabular_backbone, create_vision_backbone
from med_core.configs import load_config, save_config
from med_core.datasets import MedicalMultimodalDataset, split_dataset
from med_core.fusion import MultiModalFusionModel, create_fusion_module
from med_core.postprocessing import build_results_artifacts


def _create_checkpoint_and_logs(tmp_path: Path) -> tuple[Path, Path]:
    config = load_config("configs/starter/quickstart.yaml")
    config.logging.output_dir = str(tmp_path / "run")
    config.data.image_size = 64
    config.data.batch_size = 8
    config.data.num_workers = 0
    config.model.vision.pretrained = False

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
    assert "validation_path" in result.artifact_paths

    expected_files = [
        output_dir / "metrics.json",
        output_dir / "validation.json",
        output_dir / "summary.json",
        output_dir / "report.md",
        output_dir / "history.json",
        output_dir / "visualizations" / "confusion_matrix.png",
        output_dir / "visualizations" / "training_curves.png",
        output_dir / "visualizations" / "attention" / "attention_maps.json",
    ]
    for path in expected_files:
        assert path.exists(), f"Missing artifact: {path}"
