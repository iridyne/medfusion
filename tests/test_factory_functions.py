"""
Tests for all factory functions in the framework.

Tests:
- create_vision_backbone()
- create_tabular_backbone()
- create_fusion_module()
- create_fusion_model()
- create_multiview_fusion_model()
- create_view_aggregator()
- create_trainer()
- create_multiview_trainer()
- create_*_config()
"""

import pytest
import torch

from med_core.backbones import (
    create_multiview_vision_backbone,
    create_tabular_backbone,
    create_view_aggregator,
    create_vision_backbone,
)
from med_core.configs import (
    create_ct_multiview_config,
    create_default_config,
    create_temporal_multiview_config,
)
from med_core.fusion import (
    create_fusion_model,
    create_fusion_module,
    create_multiview_fusion_model,
)


class TestBackboneFactories:
    """Test backbone factory functions."""

    def test_create_vision_backbone_resnet(self):
        """Test creating ResNet backbone."""
        backbone = create_vision_backbone(
            name="resnet18",
            pretrained=False,
            feature_dim=128,
        )
        assert backbone is not None
        assert backbone.output_dim == 128

        # Test forward pass
        x = torch.randn(2, 3, 224, 224)
        output = backbone(x)
        assert output.shape == (2, 128)

    def test_create_vision_backbone_mobilenet(self):
        """Test creating MobileNet backbone."""
        backbone = create_vision_backbone(
            name="mobilenetv2",
            pretrained=False,
            feature_dim=64,
        )
        assert backbone is not None
        assert backbone.output_dim == 64

    def test_create_vision_backbone_efficientnet(self):
        """Test creating EfficientNet backbone."""
        backbone = create_vision_backbone(
            name="efficientnet_b0",
            pretrained=False,
            feature_dim=256,
        )
        assert backbone is not None
        assert backbone.output_dim == 256

    def test_create_tabular_backbone_mlp(self):
        """Test creating MLP tabular backbone."""
        backbone = create_tabular_backbone(
            input_dim=10,
            output_dim=32,
            hidden_dims=[64, 64],
            backbone_type="mlp",
        )
        assert backbone is not None
        assert backbone.output_dim == 32

        # Test forward pass
        x = torch.randn(4, 10)
        output = backbone(x)
        assert output.shape == (4, 32)

    def test_create_tabular_backbone_residual(self):
        """Test creating residual tabular backbone."""
        backbone = create_tabular_backbone(
            input_dim=10,
            output_dim=32,
            hidden_dims=[32, 32],
            backbone_type="residual",
        )
        assert backbone is not None
        assert backbone.output_dim == 32

    def test_create_view_aggregator_all_types(self, aggregator_types):
        """Test creating all aggregator types."""
        for agg_type in aggregator_types:
            aggregator = create_view_aggregator(
                aggregator_type=agg_type,
                feature_dim=128,
                view_names=["view1", "view2", "view3"],
            )
            assert aggregator is not None

            # Test forward pass
            views = {
                "view1": torch.randn(4, 128),
                "view2": torch.randn(4, 128),
                "view3": torch.randn(4, 128),
            }
            output = aggregator(views)
            assert output.shape == (4, 128)

    def test_create_multiview_vision_backbone(self):
        """Test creating multi-view vision backbone."""
        backbone = create_multiview_vision_backbone(
            backbone_name="resnet18",
            view_names=["axial", "coronal", "sagittal"],
            aggregator_type="attention",
            feature_dim=128,
            pretrained=False,
        )
        assert backbone is not None
        assert backbone.output_dim == 128

        # Test forward pass
        views = {
            "axial": torch.randn(2, 3, 224, 224),
            "coronal": torch.randn(2, 3, 224, 224),
            "sagittal": torch.randn(2, 3, 224, 224),
        }
        output = backbone(views)
        assert output.shape == (2, 128)


class TestFusionFactories:
    """Test fusion factory functions."""

    def test_create_fusion_module_all_types(self, fusion_types):
        """Test creating all fusion module types."""
        for fusion_type in fusion_types:
            fusion = create_fusion_module(
                fusion_type=fusion_type,
                vision_dim=128,
                tabular_dim=32,
                output_dim=64,
            )
            assert fusion is not None
            assert fusion.output_dim == 64

            # Test forward pass
            vision_features = torch.randn(4, 128)
            tabular_features = torch.randn(4, 32)
            output, aux = fusion(vision_features, tabular_features)
            assert output.shape == (4, 64)

    def test_create_fusion_model(self):
        """Test creating complete fusion model."""
        model = create_fusion_model(
            vision_backbone_name="resnet18",
            tabular_input_dim=10,
            num_classes=2,
            fusion_type="attention",
            pretrained=False,
        )
        assert model is not None

        # Test forward pass
        images = torch.randn(2, 3, 224, 224)
        tabular = torch.randn(2, 10)
        outputs = model(images, tabular)

        assert "logits" in outputs
        assert outputs["logits"].shape == (2, 2)
        assert "vision_features" in outputs
        assert "tabular_features" in outputs
        assert "fused_features" in outputs

    def test_create_fusion_model_no_auxiliary_heads(self):
        """Test creating fusion model without auxiliary heads."""
        model = create_fusion_model(
            vision_backbone_name="resnet18",
            tabular_input_dim=10,
            num_classes=2,
            fusion_type="concatenate",
            use_auxiliary_heads=False,
            pretrained=False,
        )

        images = torch.randn(2, 3, 224, 224)
        tabular = torch.randn(2, 10)
        outputs = model(images, tabular)

        assert "vision_logits" not in outputs
        assert "tabular_logits" not in outputs

    def test_create_multiview_fusion_model(self):
        """Test creating multi-view fusion model."""
        model = create_multiview_fusion_model(
            vision_backbone_name="resnet18",
            tabular_input_dim=10,
            num_classes=2,
            view_names=["axial", "coronal", "sagittal"],
            aggregator_type="attention",
            fusion_type="attention",
            pretrained=False,
        )
        assert model is not None

        # Test forward pass
        views = {
            "axial": torch.randn(2, 3, 224, 224),
            "coronal": torch.randn(2, 3, 224, 224),
            "sagittal": torch.randn(2, 3, 224, 224),
        }
        tabular = torch.randn(2, 10)
        outputs = model(views, tabular)

        assert "logits" in outputs
        assert outputs["logits"].shape == (2, 2)

    def test_create_multiview_fusion_model_with_shared_backbone(self):
        """Test creating multi-view model with shared backbone."""
        model = create_multiview_fusion_model(
            vision_backbone_name="resnet18",
            tabular_input_dim=10,
            num_classes=2,
            view_names=["view1", "view2"],
            aggregator_type="mean",
            fusion_type="gated",
            share_backbone=True,
            pretrained=False,
        )
        assert model is not None


class TestConfigFactories:
    """Test config factory functions."""

    def test_create_default_config(self):
        """Test creating default config."""
        config = create_default_config()
        assert config is not None
        assert hasattr(config, "data")
        assert hasattr(config, "model")
        assert hasattr(config, "training")

    def test_create_ct_multiview_config(self):
        """Test creating CT multi-view config."""
        config = create_ct_multiview_config(
            data_root="./data",
            csv_path="./data.csv",
            image_dir="./images",
        )
        assert config is not None
        assert hasattr(config, "data")
        assert hasattr(config.data, "view_names")
        assert "axial" in config.data.view_names
        assert "coronal" in config.data.view_names
        assert "sagittal" in config.data.view_names

    def test_create_temporal_multiview_config(self):
        """Test creating temporal multi-view config."""
        config = create_temporal_multiview_config(
            data_root="./data",
            csv_path="./data.csv",
            image_dir="./images",
            num_timepoints=3,
        )
        assert config is not None
        assert hasattr(config.data, "view_names")
        assert len(config.data.view_names) == 3


class TestTrainerFactories:
    """Test trainer factory functions."""

    def test_create_trainer(self):
        """Test creating single-view trainer."""
        from torch.utils.data import DataLoader, TensorDataset

        from med_core.trainers import create_trainer

        # Create dummy model
        model = create_fusion_model(
            vision_backbone_name="resnet18",
            tabular_input_dim=10,
            num_classes=2,
            pretrained=False,
        )

        # Create dummy dataloaders
        images = torch.randn(20, 3, 224, 224)
        tabular = torch.randn(20, 10)
        labels = torch.randint(0, 2, (20,))
        dataset = TensorDataset(images, tabular, labels)
        train_loader = DataLoader(dataset, batch_size=4)
        val_loader = DataLoader(dataset, batch_size=4)

        # Create config
        config = create_default_config()

        # Create trainer
        trainer = create_trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=config,
            device="cpu",
        )
        assert trainer is not None

    def test_create_multiview_trainer(self):
        """Test creating multi-view trainer."""
        from torch.utils.data import DataLoader, TensorDataset

        from med_core.trainers import create_multiview_trainer

        # Create dummy model
        model = create_multiview_fusion_model(
            vision_backbone_name="resnet18",
            tabular_input_dim=10,
            num_classes=2,
            view_names=["view1", "view2"],
            pretrained=False,
        )

        # Create dummy dataloaders (simplified - real multi-view needs dict)
        images = torch.randn(20, 3, 224, 224)
        tabular = torch.randn(20, 10)
        labels = torch.randint(0, 2, (20,))
        dataset = TensorDataset(images, tabular, labels)
        train_loader = DataLoader(dataset, batch_size=4)
        val_loader = DataLoader(dataset, batch_size=4)

        # Create config
        config = create_ct_multiview_config(
            data_root="./data",
            csv_path="./data.csv",
            image_dir="./images",
        )

        # Create trainer
        trainer = create_multiview_trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=config,
            device="cpu",
        )
        assert trainer is not None


class TestFactoryParameterValidation:
    """Test factory functions handle invalid parameters correctly."""

    def test_invalid_vision_backbone_name(self):
        """Test invalid vision backbone name raises error."""
        with pytest.raises((ValueError, KeyError)):
            create_vision_backbone(
                name="invalid_backbone",
                pretrained=False,
            )

    def test_invalid_fusion_type(self):
        """Test invalid fusion type raises error."""
        with pytest.raises((ValueError, KeyError)):
            create_fusion_module(
                fusion_type="invalid_fusion",
                vision_dim=128,
                tabular_dim=32,
                output_dim=64,
            )

    def test_invalid_aggregator_type(self):
        """Test invalid aggregator type raises error."""
        with pytest.raises((ValueError, KeyError)):
            create_view_aggregator(
                aggregator_type="invalid_aggregator",
                feature_dim=128,
                view_names=["view1", "view2"],
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
