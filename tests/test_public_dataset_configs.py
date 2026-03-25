"""Tests for public dataset quick-validation configs."""

from pathlib import Path

from med_core.configs import load_config


def test_pathmnist_quickstart_config_loads() -> None:
    config = load_config("configs/public_datasets/pathmnist_quickstart.yaml")

    assert config.data.csv_path == "data/public/medmnist/pathmnist-demo/metadata.csv"
    assert config.data.image_dir == "data/public/medmnist/pathmnist-demo"
    assert config.data.target_column == "label"
    assert config.data.numerical_features == []
    assert config.data.categorical_features == []
    assert config.training.monitor == "accuracy"
    assert config.training.use_progressive_training is False


def test_breastmnist_quickstart_config_loads() -> None:
    config = load_config("configs/public_datasets/breastmnist_quickstart.yaml")

    assert config.data.csv_path == "data/public/medmnist/breastmnist-demo/metadata.csv"
    assert config.data.image_dir == "data/public/medmnist/breastmnist-demo"
    assert config.data.target_column == "label"
    assert config.data.numerical_features == []
    assert config.data.categorical_features == []
    assert config.model.num_classes == 2
    assert config.training.monitor == "accuracy"
    assert Path(config.logging.output_dir).name == "breastmnist_quickstart"


def test_uci_heart_quickstart_config_loads() -> None:
    config = load_config("configs/public_datasets/uci_heart_disease_quickstart.yaml")

    assert config.data.csv_path == "data/public/uci/heart-disease-demo/metadata.csv"
    assert config.data.image_dir == "data/public/uci/heart-disease-demo"
    assert config.data.target_column == "diagnosis_binary"
    assert "age" in config.data.numerical_features
    assert "thal" in config.data.categorical_features
    assert config.model.vision.freeze_backbone is True
    assert config.training.monitor == "accuracy"
    assert Path(config.logging.output_dir).name == "uci_heart_disease_quickstart"
