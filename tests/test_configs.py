"""
Tests for configuration loading and validation.

Tests cover:
- Configuration dataclass creation
- YAML config loading
- Config validation
- Config serialization/deserialization
"""

import tempfile
import unittest
from pathlib import Path

import yaml

from med_core.configs import (
    DataConfig,
    ExperimentConfig,
    FusionConfig,
    LoggingConfig,
    ModelConfig,
    TabularConfig,
    VisionConfig,
    load_config,
    save_config,
)


class TestConfigDataclasses(unittest.TestCase):
    """Test configuration dataclass creation and defaults."""

    def test_data_config_defaults(self):
        """Test DataConfig default values."""
        config = DataConfig()
        self.assertEqual(config.batch_size, 16)
        self.assertEqual(config.image_size, 224)
        self.assertEqual(config.train_ratio, 0.7)
        self.assertEqual(config.val_ratio, 0.15)
        self.assertEqual(config.test_ratio, 0.15)

    def test_vision_config_defaults(self):
        """Test VisionConfig default values."""
        config = VisionConfig()
        self.assertEqual(config.backbone, "resnet18")
        self.assertTrue(config.pretrained)
        self.assertEqual(config.feature_dim, 128)
        self.assertEqual(config.attention_type, "cbam")

    def test_tabular_config_defaults(self):
        """Test TabularConfig default values."""
        config = TabularConfig()
        self.assertEqual(config.hidden_dims, [64, 64])
        self.assertEqual(config.output_dim, 32)
        self.assertTrue(config.use_batch_norm)

    def test_fusion_config_defaults(self):
        """Test FusionConfig default values."""
        config = FusionConfig()
        self.assertEqual(config.fusion_type, "gated")
        self.assertEqual(config.hidden_dim, 96)
        self.assertTrue(config.learnable_weights)

    def test_model_config_composition(self):
        """Test ModelConfig with nested configs."""
        config = ModelConfig(
            num_classes=3,
            vision=VisionConfig(backbone="resnet50"),
            tabular=TabularConfig(output_dim=64),
        )
        self.assertEqual(config.num_classes, 3)
        self.assertEqual(config.vision.backbone, "resnet50")
        self.assertEqual(config.tabular.output_dim, 64)

    def test_experiment_config_full(self):
        """Test complete ExperimentConfig."""
        config = ExperimentConfig(
            experiment_name="test_exp",
            seed=123,
        )
        self.assertEqual(config.experiment_name, "test_exp")
        self.assertEqual(config.seed, 123)
        self.assertIsInstance(config.data, DataConfig)
        self.assertIsInstance(config.model, ModelConfig)


class TestConfigSerialization(unittest.TestCase):
    """Test config to_dict and from_dict methods."""

    def test_data_config_to_dict(self):
        """Test DataConfig serialization."""
        config = DataConfig(batch_size=32, image_size=256)
        config_dict = config.to_dict()
        self.assertIsInstance(config_dict, dict)
        self.assertEqual(config_dict["batch_size"], 32)
        self.assertEqual(config_dict["image_size"], 256)

    def test_nested_config_to_dict(self):
        """Test nested config serialization."""
        config = ModelConfig(
            num_classes=2,
            vision=VisionConfig(backbone="resnet34"),
        )
        config_dict = config.to_dict()
        self.assertIn("vision", config_dict)
        self.assertEqual(config_dict["vision"]["backbone"], "resnet34")

    def test_experiment_config_to_dict(self):
        """Test full experiment config serialization."""
        config = ExperimentConfig(experiment_name="test")
        config_dict = config.to_dict()
        self.assertIn("data", config_dict)
        self.assertIn("model", config_dict)
        self.assertIn("training", config_dict)


class TestConfigYAMLLoading(unittest.TestCase):
    """Test loading configs from YAML files."""

    def setUp(self):
        """Create temporary directory for test configs."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.config_dir = Path(self.temp_dir.name)

    def tearDown(self):
        """Clean up temporary directory."""
        self.temp_dir.cleanup()

    def test_load_minimal_config(self):
        """Test loading minimal YAML config."""
        config_data = {
            "experiment_name": "minimal_test",
            "seed": 42,
        }
        config_path = self.config_dir / "minimal.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config_data, f)

        config = load_config(str(config_path))
        self.assertEqual(config.experiment_name, "minimal_test")
        self.assertEqual(config.seed, 42)

    def test_load_full_config(self):
        """Test loading complete YAML config."""
        config_data = {
            "experiment_name": "full_test",
            "seed": 123,
            "data": {
                "batch_size": 64,
                "image_size": 256,
                "train_ratio": 0.8,
            },
            "model": {
                "num_classes": 3,
                "vision": {
                    "backbone": "resnet50",
                    "feature_dim": 256,
                },
                "tabular": {
                    "output_dim": 64,
                },
            },
        }
        config_path = self.config_dir / "full.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config_data, f)

        config = load_config(str(config_path))
        self.assertEqual(config.data.batch_size, 64)
        self.assertEqual(config.model.vision.backbone, "resnet50")
        self.assertEqual(config.model.tabular.output_dim, 64)

    def test_save_and_load_config(self):
        """Test saving and loading config roundtrip."""
        original_config = ExperimentConfig(
            experiment_name="roundtrip_test",
            seed=999,
        )
        config_path = self.config_dir / "roundtrip.yaml"

        save_config(original_config, str(config_path))
        loaded_config = load_config(str(config_path))

        self.assertEqual(loaded_config.experiment_name, "roundtrip_test")
        self.assertEqual(loaded_config.seed, 999)

    def test_load_nonexistent_file(self):
        """Test loading non-existent config file."""
        with self.assertRaises(FileNotFoundError):
            load_config(str(self.config_dir / "nonexistent.yaml"))


class TestConfigValidation(unittest.TestCase):
    """Test configuration validation logic."""

    def test_split_ratios_sum(self):
        """Test that split ratios should sum to 1.0."""
        # Valid ratios
        config = DataConfig(train_ratio=0.7, val_ratio=0.2, test_ratio=0.1)
        total = config.train_ratio + config.val_ratio + config.test_ratio
        self.assertAlmostEqual(total, 1.0, places=6)

    def test_device_detection(self):
        """Test automatic device detection."""
        config = ExperimentConfig(device="auto")
        self.assertIn(config.device, ["cuda", "cpu", "mps"])

    def test_output_directory_creation(self):
        """Test that output directories are created."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = ExperimentConfig(
                logging=LoggingConfig(output_dir=str(Path(temp_dir) / "test_output"))
            )
            self.assertTrue(config.checkpoint_dir.exists())
            self.assertTrue(config.log_dir.exists())
            self.assertTrue(config.results_dir.exists())


if __name__ == "__main__":
    unittest.main()
