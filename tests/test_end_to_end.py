"""
End-to-end integration tests for the entire framework.

Tests complete workflows from data loading to model training and evaluation.
"""

import tempfile
from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader

from med_core.configs import create_ct_multiview_config, create_default_config
from med_core.datasets import MedicalMultimodalDataset, MedicalMultiViewDataset
from med_core.datasets.multiview_types import MultiViewConfig
from med_core.fusion import create_fusion_model, create_multiview_fusion_model
from med_core.trainers import MultimodalTrainer, MultiViewMultimodalTrainer


class TestSingleViewEndToEnd:
    """Test complete single-view workflow."""

    def test_complete_single_view_pipeline(self, sample_csv_data):
        """Test complete pipeline: data -> model -> train -> evaluate."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # 1. Load dataset
            train_dataset, train_scaler = MedicalMultimodalDataset.from_csv(
                csv_path=sample_csv_data["csv_path"],
                image_dir=sample_csv_data["image_dir"],
                image_column="image_path",
                target_column="label",
                numerical_features=["age", "bmi"],
                categorical_features=["gender"],
                normalize_features=True,
                transform=sample_csv_data["transform"],
            )

            val_dataset, _ = MedicalMultimodalDataset.from_csv(
                csv_path=sample_csv_data["csv_path"],
                image_dir=sample_csv_data["image_dir"],
                image_column="image_path",
                target_column="label",
                numerical_features=["age", "bmi"],
                categorical_features=["gender"],
                normalize_features=True,
                scaler=train_scaler,
                transform=sample_csv_data["transform"],
            )

            assert len(train_dataset) > 0
            assert len(val_dataset) > 0

            # 2. Create dataloaders
            train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)

            # 3. Create model
            tabular_dim = len(train_dataset.get_feature_names())
            model = create_fusion_model(
                vision_backbone_name="resnet18",
                tabular_input_dim=tabular_dim,
                num_classes=2,
                fusion_type="attention",
                pretrained=False,
            )

            assert model is not None

            # 4. Create config
            config = create_default_config()
            config.training.num_epochs = 2
            config.training.mixed_precision = False

            # 5. Create trainer
            trainer = MultimodalTrainer(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                config=config,
                device="cpu",
                log_dir=tmpdir,
            )

            # 6. Train model
            history = trainer.train()

            assert history is not None
            assert len(history["train_loss"]) == 2
            assert len(history["val_loss"]) == 2

            # 7. Evaluate model (simple validation)
            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for batch in val_loader:
                    images, tabular, labels = batch
                    outputs = model(images, tabular)
                    predictions = torch.argmax(outputs["logits"], dim=1)
                    correct += (predictions == labels).sum().item()
                    total += labels.size(0)

            accuracy = correct / total
            assert 0 <= accuracy <= 1

    def test_single_view_with_different_fusion_strategies(self, sample_csv_data):
        """Test pipeline with different fusion strategies."""
        fusion_types = ["concatenate", "gated", "attention"]

        for fusion_type in fusion_types:
            # Load dataset
            dataset, _ = MedicalMultimodalDataset.from_csv(
                csv_path=sample_csv_data["csv_path"],
                image_dir=sample_csv_data["image_dir"],
                image_column="image_path",
                target_column="label",
                numerical_features=["age", "bmi"],
                categorical_features=["gender"],
                transform=sample_csv_data["transform"],
            )

            loader = DataLoader(dataset, batch_size=4)

            # Create model with specific fusion type
            tabular_dim = len(dataset.get_feature_names())
            model = create_fusion_model(
                vision_backbone_name="resnet18",
                tabular_input_dim=tabular_dim,
                num_classes=2,
                fusion_type=fusion_type,
                pretrained=False,
            )

            # Test forward pass
            batch = next(iter(loader))
            images, tabular, labels = batch

            model.eval()
            with torch.no_grad():
                outputs = model(images, tabular)

            assert "logits" in outputs
            assert outputs["logits"].shape[0] == images.shape[0]
            assert outputs["logits"].shape[1] == 2

    def test_single_view_progressive_training(self, sample_csv_data):
        """Test progressive training workflow."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Load dataset
            dataset, _ = MedicalMultimodalDataset.from_csv(
                csv_path=sample_csv_data["csv_path"],
                image_dir=sample_csv_data["image_dir"],
                image_column="image_path",
                target_column="label",
                numerical_features=["age", "bmi"],
                transform=sample_csv_data["transform"],
            )

            train_loader = DataLoader(dataset, batch_size=4)
            val_loader = DataLoader(dataset, batch_size=4)

            # Create model
            tabular_dim = len(dataset.get_feature_names())
            model = create_fusion_model(
                vision_backbone_name="resnet18",
                tabular_input_dim=tabular_dim,
                num_classes=2,
                pretrained=False,
            )

            # Create config with progressive training
            config = create_default_config()
            config.training.num_epochs = 3
            config.training.use_progressive_training = True
            config.training.stage1_epochs = 1
            config.training.stage2_epochs = 1
            config.training.mixed_precision = False

            # Train
            trainer = MultimodalTrainer(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                config=config,
                device="cpu",
                log_dir=tmpdir,
            )

            history = trainer.train()
            assert len(history["train_loss"]) == 3


class TestMultiViewEndToEnd:
    """Test complete multi-view workflow."""

    def test_complete_multiview_pipeline(self, sample_multiview_csv_data):
        """Test complete multi-view pipeline: data -> model -> train -> evaluate."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # 1. Load multi-view dataset
            view_config = MultiViewConfig(
                view_names=["axial", "coronal", "sagittal"],
                required_views=["axial"],
                handle_missing="zero",
            )

            train_dataset, train_scaler = MedicalMultiViewDataset.from_csv_multiview(
                csv_path=sample_multiview_csv_data["csv_path"],
                image_dir=sample_multiview_csv_data["image_dir"],
                view_columns=sample_multiview_csv_data["view_columns"],
                target_column="label",
                numerical_features=["age", "bmi"],
                categorical_features=["gender"],
                view_config=view_config,
                normalize_features=True,
                transform=sample_multiview_csv_data["transform"],
            )

            val_dataset, _ = MedicalMultiViewDataset.from_csv_multiview(
                csv_path=sample_multiview_csv_data["csv_path"],
                image_dir=sample_multiview_csv_data["image_dir"],
                view_columns=sample_multiview_csv_data["view_columns"],
                target_column="label",
                numerical_features=["age", "bmi"],
                categorical_features=["gender"],
                view_config=view_config,
                normalize_features=True,
                scaler=train_scaler,
                transform=sample_multiview_csv_data["transform"],
            )

            assert len(train_dataset) > 0
            assert len(val_dataset) > 0

            # 2. Create dataloaders
            train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False)

            # 3. Create multi-view model
            tabular_dim = len(train_dataset.get_feature_names())
            model = create_multiview_fusion_model(
                vision_backbone_name="resnet18",
                tabular_input_dim=tabular_dim,
                num_classes=2,
                view_names=["axial", "coronal", "sagittal"],
                aggregator_type="attention",
                fusion_type="attention",
                pretrained=False,
            )

            assert model is not None

            # 4. Create config
            config = create_ct_multiview_config(
                view_names=["axial", "coronal", "sagittal"],
                aggregator_type="attention",
            )
            config.training.num_epochs = 2
            config.training.mixed_precision = False

            # 5. Create multi-view trainer
            trainer = MultiViewMultimodalTrainer(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                config=config,
                device="cpu",
                log_dir=tmpdir,
            )

            # 6. Train model
            history = trainer.train()

            assert history is not None
            assert len(history["train_loss"]) == 2
            assert len(history["val_loss"]) == 2

    def test_multiview_with_different_aggregators(self, sample_multiview_csv_data):
        """Test multi-view pipeline with different aggregators."""
        aggregator_types = ["mean", "max", "attention"]

        for agg_type in aggregator_types:
            # Load dataset
            view_config = MultiViewConfig(
                view_names=["axial", "coronal", "sagittal"],
                required_views=[],
                handle_missing="zero",
            )

            dataset, _ = MedicalMultiViewDataset.from_csv_multiview(
                csv_path=sample_multiview_csv_data["csv_path"],
                image_dir=sample_multiview_csv_data["image_dir"],
                view_columns=sample_multiview_csv_data["view_columns"],
                target_column="label",
                numerical_features=["age", "bmi"],
                view_config=view_config,
                transform=sample_multiview_csv_data["transform"],
            )

            loader = DataLoader(dataset, batch_size=2)

            # Create model with specific aggregator
            tabular_dim = len(dataset.get_feature_names())
            model = create_multiview_fusion_model(
                vision_backbone_name="resnet18",
                tabular_input_dim=tabular_dim,
                num_classes=2,
                view_names=["axial", "coronal", "sagittal"],
                aggregator_type=agg_type,
                fusion_type="concatenate",
                pretrained=False,
            )

            # Test forward pass
            batch = next(iter(loader))
            views, tabular, labels = batch

            model.eval()
            with torch.no_grad():
                outputs = model(views, tabular)

            assert "logits" in outputs
            assert outputs["logits"].shape[0] == 2
            assert outputs["logits"].shape[1] == 2

    def test_multiview_progressive_views_training(self, sample_multiview_csv_data):
        """Test progressive view training workflow."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Load dataset
            view_config = MultiViewConfig(
                view_names=["axial", "coronal", "sagittal"],
                required_views=[],
                handle_missing="zero",
            )

            dataset, _ = MedicalMultiViewDataset.from_csv_multiview(
                csv_path=sample_multiview_csv_data["csv_path"],
                image_dir=sample_multiview_csv_data["image_dir"],
                view_columns=sample_multiview_csv_data["view_columns"],
                target_column="label",
                numerical_features=["age", "bmi"],
                view_config=view_config,
                transform=sample_multiview_csv_data["transform"],
            )

            train_loader = DataLoader(dataset, batch_size=2)
            val_loader = DataLoader(dataset, batch_size=2)

            # Create model
            tabular_dim = len(dataset.get_feature_names())
            model = create_multiview_fusion_model(
                vision_backbone_name="resnet18",
                tabular_input_dim=tabular_dim,
                num_classes=2,
                view_names=["axial", "coronal", "sagittal"],
                aggregator_type="attention",
                pretrained=False,
            )

            # Create config with progressive views
            config = create_ct_multiview_config(
                view_names=["axial", "coronal", "sagittal"],
                aggregator_type="attention",
            )
            config.training.num_epochs = 3
            config.training.use_progressive_views = True
            config.training.progressive_view_schedule = {
                0: ["axial"],
                1: ["axial", "coronal"],
                2: ["axial", "coronal", "sagittal"],
            }
            config.training.mixed_precision = False

            # Train
            trainer = MultiViewMultimodalTrainer(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                config=config,
                device="cpu",
                log_dir=tmpdir,
            )

            history = trainer.train()
            assert len(history["train_loss"]) == 3


class TestModelSavingAndLoading:
    """Test model checkpoint saving and loading."""

    def test_save_and_load_single_view_model(self, sample_csv_data):
        """Test saving and loading single-view model."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create and train model
            dataset, _ = MedicalMultimodalDataset.from_csv(
                csv_path=sample_csv_data["csv_path"],
                image_dir=sample_csv_data["image_dir"],
                image_column="image_path",
                target_column="label",
                numerical_features=["age", "bmi"],
                transform=sample_csv_data["transform"],
            )

            loader = DataLoader(dataset, batch_size=4)

            tabular_dim = len(dataset.get_feature_names())
            model = create_fusion_model(
                vision_backbone_name="resnet18",
                tabular_input_dim=tabular_dim,
                num_classes=2,
                pretrained=False,
            )

            # Save model
            checkpoint_path = Path(tmpdir) / "model.pt"
            torch.save(model.state_dict(), checkpoint_path)

            # Load model
            loaded_model = create_fusion_model(
                vision_backbone_name="resnet18",
                tabular_input_dim=tabular_dim,
                num_classes=2,
                pretrained=False,
            )
            loaded_model.load_state_dict(torch.load(checkpoint_path))

            # Test both models produce same output
            batch = next(iter(loader))
            images, tabular, _ = batch

            model.eval()
            loaded_model.eval()

            with torch.no_grad():
                output1 = model(images, tabular)
                output2 = loaded_model(images, tabular)

            assert torch.allclose(output1["logits"], output2["logits"], atol=1e-6)

    def test_save_and_load_multiview_model(self, sample_multiview_csv_data):
        """Test saving and loading multi-view model."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create model
            view_config = MultiViewConfig(
                view_names=["axial", "coronal", "sagittal"],
                required_views=[],
                handle_missing="zero",
            )

            dataset, _ = MedicalMultiViewDataset.from_csv_multiview(
                csv_path=sample_multiview_csv_data["csv_path"],
                image_dir=sample_multiview_csv_data["image_dir"],
                view_columns=sample_multiview_csv_data["view_columns"],
                target_column="label",
                numerical_features=["age", "bmi"],
                view_config=view_config,
                transform=sample_multiview_csv_data["transform"],
            )

            loader = DataLoader(dataset, batch_size=2)

            tabular_dim = len(dataset.get_feature_names())
            model = create_multiview_fusion_model(
                vision_backbone_name="resnet18",
                tabular_input_dim=tabular_dim,
                num_classes=2,
                view_names=["axial", "coronal", "sagittal"],
                aggregator_type="attention",
                pretrained=False,
            )

            # Save model
            checkpoint_path = Path(tmpdir) / "multiview_model.pt"
            torch.save(model.state_dict(), checkpoint_path)

            # Load model
            loaded_model = create_multiview_fusion_model(
                vision_backbone_name="resnet18",
                tabular_input_dim=tabular_dim,
                num_classes=2,
                view_names=["axial", "coronal", "sagittal"],
                aggregator_type="attention",
                pretrained=False,
            )
            loaded_model.load_state_dict(torch.load(checkpoint_path))

            # Test both models produce same output
            batch = next(iter(loader))
            views, tabular, _ = batch

            model.eval()
            loaded_model.eval()

            with torch.no_grad():
                output1 = model(views, tabular)
                output2 = loaded_model(views, tabular)

            assert torch.allclose(output1["logits"], output2["logits"], atol=1e-6)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
