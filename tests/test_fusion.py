"""
Tests for fusion modules and complete fusion models.

Tests:
- All fusion module types (concatenate, gated, attention, cross_attention, bilinear)
- Complete single-view fusion models
- Complete multi-view fusion models
- Auxiliary heads functionality
- Model forward pass and output structure
"""

import pytest
import torch

from med_core.fusion import (
    MultiModalFusionModel,
    MultiViewMultiModalFusionModel,
    create_fusion_model,
    create_fusion_module,
    create_multiview_fusion_model,
)


class TestFusionModules:
    """Test individual fusion modules."""

    @pytest.fixture
    def fusion_features(self):
        """Create sample features for fusion testing."""
        batch_size = 4
        vision_dim = 128
        tabular_dim = 32

        return {
            "batch_size": batch_size,
            "vision_dim": vision_dim,
            "tabular_dim": tabular_dim,
            "vision_features": torch.randn(batch_size, vision_dim),
            "tabular_features": torch.randn(batch_size, tabular_dim),
        }

    def test_concatenate_fusion(self, fusion_features):
        """Test concatenate fusion module."""
        fusion = create_fusion_module(
            "concatenate",
            vision_dim=fusion_features["vision_dim"],
            tabular_dim=fusion_features["tabular_dim"],
            output_dim=64,
        )
        output, aux = fusion(
            fusion_features["vision_features"],
            fusion_features["tabular_features"],
        )
        assert output.shape == (fusion_features["batch_size"], 64)
        assert aux is None

    def test_gated_fusion(self, fusion_features):
        """Test gated fusion module."""
        fusion = create_fusion_module(
            "gated",
            vision_dim=fusion_features["vision_dim"],
            tabular_dim=fusion_features["tabular_dim"],
            output_dim=64,
        )
        output, aux = fusion(
            fusion_features["vision_features"],
            fusion_features["tabular_features"],
        )
        assert output.shape == (fusion_features["batch_size"], 64)
        assert "gate_values" in aux
        assert aux["gate_values"].shape == (fusion_features["batch_size"], 64)

    def test_attention_fusion(self, fusion_features):
        """Test attention fusion module."""
        fusion = create_fusion_module(
            "attention",
            vision_dim=fusion_features["vision_dim"],
            tabular_dim=fusion_features["tabular_dim"],
            output_dim=64,
            num_heads=4,
        )
        output, aux = fusion(
            fusion_features["vision_features"],
            fusion_features["tabular_features"],
        )
        assert output.shape == (fusion_features["batch_size"], 64)
        assert "attention_weights" in aux

    def test_cross_attention_fusion(self, fusion_features):
        """Test cross-attention fusion module."""
        fusion = create_fusion_module(
            "cross_attention",
            vision_dim=fusion_features["vision_dim"],
            tabular_dim=fusion_features["tabular_dim"],
            output_dim=64,
            num_heads=4,
        )
        output, aux = fusion(
            fusion_features["vision_features"],
            fusion_features["tabular_features"],
        )
        assert output.shape == (fusion_features["batch_size"], 64)
        assert "vision_to_tabular_attn" in aux
        assert "tabular_to_vision_attn" in aux

    def test_bilinear_fusion(self, fusion_features):
        """Test bilinear fusion module."""
        fusion = create_fusion_module(
            "bilinear",
            vision_dim=fusion_features["vision_dim"],
            tabular_dim=fusion_features["tabular_dim"],
            output_dim=64,
            rank=16,
        )
        output, aux = fusion(
            fusion_features["vision_features"],
            fusion_features["tabular_features"],
        )
        assert output.shape == (fusion_features["batch_size"], 64)


class TestCompleteFusionModels:
    """Test complete fusion models (backbone + fusion + classifier)."""

    def test_single_view_fusion_model_creation(self):
        """Test creating complete single-view fusion model."""
        model = create_fusion_model(
            vision_backbone_name="resnet18",
            tabular_input_dim=10,
            num_classes=2,
            fusion_type="attention",
            pretrained=False,
        )

        assert model is not None
        assert isinstance(model, MultiModalFusionModel)
        assert hasattr(model, "vision_backbone")
        assert hasattr(model, "tabular_backbone")
        assert hasattr(model, "fusion_module")
        assert hasattr(model, "classifier")

    def test_single_view_fusion_model_forward(self):
        """Test forward pass of single-view fusion model."""
        model = create_fusion_model(
            vision_backbone_name="resnet18",
            tabular_input_dim=10,
            num_classes=2,
            fusion_type="concatenate",
            pretrained=False,
        )

        images = torch.randn(4, 3, 224, 224)
        tabular = torch.randn(4, 10)

        model.eval()
        with torch.no_grad():
            outputs = model(images, tabular)

        assert "logits" in outputs
        assert outputs["logits"].shape == (4, 2)
        assert "vision_features" in outputs
        assert "tabular_features" in outputs
        assert "fused_features" in outputs

    def test_single_view_model_with_auxiliary_heads(self):
        """Test model with auxiliary classification heads."""
        model = create_fusion_model(
            vision_backbone_name="resnet18",
            tabular_input_dim=10,
            num_classes=2,
            fusion_type="gated",
            use_auxiliary_heads=True,
            pretrained=False,
        )

        images = torch.randn(4, 3, 224, 224)
        tabular = torch.randn(4, 10)

        model.eval()
        with torch.no_grad():
            outputs = model(images, tabular)

        assert "logits" in outputs
        assert "vision_logits" in outputs
        assert "tabular_logits" in outputs
        assert outputs["vision_logits"].shape == (4, 2)
        assert outputs["tabular_logits"].shape == (4, 2)

    def test_single_view_model_without_auxiliary_heads(self):
        """Test model without auxiliary heads."""
        model = create_fusion_model(
            vision_backbone_name="resnet18",
            tabular_input_dim=10,
            num_classes=2,
            fusion_type="concatenate",
            use_auxiliary_heads=False,
            pretrained=False,
        )

        images = torch.randn(4, 3, 224, 224)
        tabular = torch.randn(4, 10)

        model.eval()
        with torch.no_grad():
            outputs = model(images, tabular)

        assert "logits" in outputs
        assert "vision_logits" not in outputs
        assert "tabular_logits" not in outputs

    def test_single_view_model_different_backbones(self):
        """Test model with different vision backbones."""
        backbones = ["resnet18", "mobilenetv2"]

        for backbone_name in backbones:
            model = create_fusion_model(
                vision_backbone_name=backbone_name,
                tabular_input_dim=10,
                num_classes=2,
                pretrained=False,
            )

            images = torch.randn(2, 3, 224, 224)
            tabular = torch.randn(2, 10)

            model.eval()
            with torch.no_grad():
                outputs = model(images, tabular)

            assert outputs["logits"].shape == (2, 2)


class TestMultiViewFusionModels:
    """Test multi-view fusion models."""

    def test_multiview_fusion_model_creation(self):
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
        assert isinstance(model, MultiViewMultiModalFusionModel)
        assert hasattr(model, "vision_backbone")
        assert hasattr(model, "tabular_backbone")
        assert hasattr(model, "fusion_module")
        assert hasattr(model, "classifier")

    def test_multiview_fusion_model_forward(self):
        """Test forward pass of multi-view fusion model."""
        model = create_multiview_fusion_model(
            vision_backbone_name="resnet18",
            tabular_input_dim=10,
            num_classes=2,
            view_names=["view1", "view2", "view3"],
            aggregator_type="mean",
            fusion_type="concatenate",
            pretrained=False,
        )

        views = {
            "view1": torch.randn(4, 3, 224, 224),
            "view2": torch.randn(4, 3, 224, 224),
            "view3": torch.randn(4, 3, 224, 224),
        }
        tabular = torch.randn(4, 10)

        model.eval()
        with torch.no_grad():
            outputs = model(views, tabular)

        assert "logits" in outputs
        assert outputs["logits"].shape == (4, 2)
        assert "vision_features" in outputs
        assert "tabular_features" in outputs
        assert "fused_features" in outputs

    def test_multiview_model_different_aggregators(self):
        """Test multi-view model with different aggregators."""
        aggregators = ["max", "mean", "attention"]

        for agg_type in aggregators:
            model = create_multiview_fusion_model(
                vision_backbone_name="resnet18",
                tabular_input_dim=10,
                num_classes=2,
                view_names=["view1", "view2"],
                aggregator_type=agg_type,
                fusion_type="concatenate",
                pretrained=False,
            )

            views = {
                "view1": torch.randn(2, 3, 224, 224),
                "view2": torch.randn(2, 3, 224, 224),
            }
            tabular = torch.randn(2, 10)

            model.eval()
            with torch.no_grad():
                outputs = model(views, tabular)

            assert outputs["logits"].shape == (2, 2)

    def test_multiview_model_with_shared_backbone(self):
        """Test multi-view model with shared backbone weights."""
        model = create_multiview_fusion_model(
            vision_backbone_name="resnet18",
            tabular_input_dim=10,
            num_classes=2,
            view_names=["view1", "view2"],
            aggregator_type="mean",
            share_backbone=True,
            pretrained=False,
        )

        views = {
            "view1": torch.randn(2, 3, 224, 224),
            "view2": torch.randn(2, 3, 224, 224),
        }
        tabular = torch.randn(2, 10)

        model.eval()
        with torch.no_grad():
            outputs = model(views, tabular)

        assert outputs["logits"].shape == (2, 2)

    def test_multiview_model_with_auxiliary_heads(self):
        """Test multi-view model with auxiliary heads."""
        model = create_multiview_fusion_model(
            vision_backbone_name="resnet18",
            tabular_input_dim=10,
            num_classes=2,
            view_names=["view1", "view2"],
            aggregator_type="attention",
            fusion_type="gated",
            use_auxiliary_heads=True,
            pretrained=False,
        )

        views = {
            "view1": torch.randn(2, 3, 224, 224),
            "view2": torch.randn(2, 3, 224, 224),
        }
        tabular = torch.randn(2, 10)

        model.eval()
        with torch.no_grad():
            outputs = model(views, tabular)

        assert "logits" in outputs
        assert "vision_logits" in outputs
        assert "tabular_logits" in outputs


class TestModelGradients:
    """Test gradient flow through models."""

    def test_single_view_model_gradients(self):
        """Test gradients flow through single-view model."""
        model = create_fusion_model(
            vision_backbone_name="resnet18",
            tabular_input_dim=10,
            num_classes=2,
            pretrained=False,
        )

        images = torch.randn(2, 3, 224, 224, requires_grad=True)
        tabular = torch.randn(2, 10, requires_grad=True)
        labels = torch.tensor([0, 1])

        outputs = model(images, tabular)
        loss = torch.nn.functional.cross_entropy(outputs["logits"], labels)
        loss.backward()

        # Check gradients exist
        assert images.grad is not None
        assert tabular.grad is not None

        # Check model parameters have gradients
        for param in model.parameters():
            if param.requires_grad:
                assert param.grad is not None

    def test_multiview_model_gradients(self):
        """Test gradients flow through multi-view model."""
        model = create_multiview_fusion_model(
            vision_backbone_name="resnet18",
            tabular_input_dim=10,
            num_classes=2,
            view_names=["view1", "view2"],
            pretrained=False,
        )

        views = {
            "view1": torch.randn(2, 3, 224, 224, requires_grad=True),
            "view2": torch.randn(2, 3, 224, 224, requires_grad=True),
        }
        tabular = torch.randn(2, 10, requires_grad=True)
        labels = torch.tensor([0, 1])

        outputs = model(views, tabular)
        loss = torch.nn.functional.cross_entropy(outputs["logits"], labels)
        loss.backward()

        # Check gradients exist
        assert views["view1"].grad is not None
        assert views["view2"].grad is not None
        assert tabular.grad is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
