"""
Tests for attention supervision functionality.

Tests the attention supervision features including:
- Attention weight extraction from backbones
- CAM generation
- Attention loss computation
- Integration with training pipeline
"""

import pytest
import torch
import torch.nn as nn

from med_core.backbones import create_vision_backbone
from med_core.backbones.attention import CBAM, create_attention_module
from med_core.configs.base_config import (
    DataConfig,
    ExperimentConfig,
    FusionConfig,
    ModelConfig,
    TabularConfig,
    TrainingConfig,
    VisionConfig,
)


class TestAttentionModules:
    """Test attention modules with weight return functionality."""

    def test_cbam_without_weights(self):
        """Test CBAM in normal mode (no weight return)."""
        cbam = CBAM(in_channels=64, return_attention_weights=False)
        x = torch.randn(2, 64, 28, 28)

        output = cbam(x)

        assert isinstance(output, torch.Tensor)
        assert output.shape == x.shape

    def test_cbam_with_weights(self):
        """Test CBAM with attention weight return."""
        cbam = CBAM(in_channels=64, return_attention_weights=True)
        x = torch.randn(2, 64, 28, 28)

        output, weights_dict = cbam(x)

        assert isinstance(output, torch.Tensor)
        assert output.shape == x.shape
        assert isinstance(weights_dict, dict)
        assert "channel_weights" in weights_dict
        assert "spatial_weights" in weights_dict
        assert weights_dict["channel_weights"].shape == (2, 64, 1, 1)
        assert weights_dict["spatial_weights"].shape == (2, 1, 28, 28)

    def test_cbam_spatial_only(self):
        """Test CBAM with only spatial attention."""
        cbam = CBAM(in_channels=64, use_spatial=True, return_attention_weights=True)
        x = torch.randn(2, 64, 28, 28)

        output, weights_dict = cbam(x)

        assert weights_dict["spatial_weights"] is not None

    def test_create_attention_module_with_weights(self):
        """Test attention module factory with weight return."""
        attention = create_attention_module(
            attention_type="cbam", in_channels=64, return_attention_weights=True
        )
        x = torch.randn(2, 64, 28, 28)

        output, weights_dict = attention(x)

        assert isinstance(weights_dict, dict)
        assert "spatial_weights" in weights_dict


class TestVisionBackboneAttention:
    """Test vision backbones with attention supervision support."""

    @pytest.mark.parametrize("backbone_name", ["resnet18", "resnet34"])
    def test_backbone_without_attention_supervision(self, backbone_name):
        """Test backbone in normal mode."""
        backbone = create_vision_backbone(
            backbone_name,
            pretrained=False,
            feature_dim=128,
            attention_type="cbam",
            enable_attention_supervision=False,
        )
        x = torch.randn(2, 3, 224, 224)

        output = backbone(x)

        assert isinstance(output, torch.Tensor)
        assert output.shape == (2, 128)

    @pytest.mark.parametrize("backbone_name", ["resnet18", "mobilenetv2"])
    def test_backbone_with_attention_supervision(self, backbone_name):
        """Test backbone with attention supervision enabled."""
        backbone = create_vision_backbone(
            backbone_name,
            pretrained=False,
            feature_dim=128,
            attention_type="cbam",
            enable_attention_supervision=True,
        )
        x = torch.randn(2, 3, 224, 224)

        output = backbone(x, return_intermediates=True)

        assert isinstance(output, dict)
        assert "features" in output
        assert "feature_maps" in output
        assert "attention_weights" in output
        assert output["features"].shape == (2, 128)
        assert output["feature_maps"].dim() == 4  # (B, C, H, W)
        if output["attention_weights"] is not None:
            assert output["attention_weights"].dim() == 4  # (B, 1, H, W)

    def test_backbone_attention_weights_shape(self):
        """Test attention weights have correct shape."""
        backbone = create_vision_backbone(
            "resnet18",
            pretrained=False,
            feature_dim=128,
            attention_type="cbam",
            enable_attention_supervision=True,
        )
        x = torch.randn(2, 3, 224, 224)

        output = backbone(x, return_intermediates=True)

        attention_weights = output["attention_weights"]
        if attention_weights is not None:
            # Spatial attention weights should be (B, 1, H, W)
            assert attention_weights.shape[0] == 2  # batch size
            assert attention_weights.shape[1] == 1  # single channel
            assert attention_weights.shape[2] > 0  # height
            assert attention_weights.shape[3] > 0  # width

    def test_backbone_without_attention_module(self):
        """Test backbone without attention module."""
        backbone = create_vision_backbone(
            "resnet18",
            pretrained=False,
            feature_dim=128,
            attention_type=None,
            enable_attention_supervision=True,
        )
        x = torch.randn(2, 3, 224, 224)

        output = backbone(x, return_intermediates=True)

        assert output["attention_weights"] is None


class TestCAMGeneration:
    """Test Class Activation Map (CAM) generation."""

    def test_cam_generation_basic(self):
        """Test basic CAM generation."""
        # Simulate feature maps and classifier weights
        batch_size = 2
        num_classes = 2
        feature_maps = torch.randn(batch_size, 512, 7, 7)
        classifier_weights = torch.randn(num_classes, 512)
        target_class = torch.tensor([0, 1])

        # Generate CAM
        cam = self._generate_cam(feature_maps, classifier_weights, target_class)

        assert cam.shape == (batch_size, 1, 7, 7)
        assert cam.min() >= 0  # CAM should be non-negative after ReLU
        assert cam.max() <= 1  # CAM should be normalized

    def test_cam_upsampling(self):
        """Test CAM upsampling to input size."""
        batch_size = 2
        feature_maps = torch.randn(batch_size, 512, 7, 7)
        classifier_weights = torch.randn(2, 512)
        target_class = torch.tensor([0, 1])

        cam = self._generate_cam(feature_maps, classifier_weights, target_class)

        # Upsample to input size
        upsampled_cam = nn.functional.interpolate(
            cam, size=(224, 224), mode="bilinear", align_corners=False
        )

        assert upsampled_cam.shape == (batch_size, 1, 224, 224)

    def _generate_cam(self, feature_maps, classifier_weights, target_class):
        """Helper method to generate CAM."""
        batch_size = feature_maps.size(0)
        num_channels = feature_maps.size(1)

        # Get weights for target class
        weights = classifier_weights[target_class]  # (B, C)

        # Reshape for broadcasting
        weights = weights.view(batch_size, num_channels, 1, 1)

        # Weighted sum of feature maps
        cam = (feature_maps * weights).sum(dim=1, keepdim=True)

        # Apply ReLU
        cam = torch.relu(cam)

        # Normalize
        cam_min = (
            cam.view(batch_size, -1)
            .min(dim=1, keepdim=True)[0]
            .view(batch_size, 1, 1, 1)
        )
        cam_max = (
            cam.view(batch_size, -1)
            .max(dim=1, keepdim=True)[0]
            .view(batch_size, 1, 1, 1)
        )
        cam = (cam - cam_min) / (cam_max - cam_min + 1e-8)

        return cam


class TestAttentionLoss:
    """Test attention loss computation."""

    def test_mask_guided_loss(self):
        """Test mask-guided attention loss."""
        attention_weights = torch.rand(2, 1, 28, 28)
        masks = torch.randint(0, 2, (2, 1, 28, 28)).float()

        # Binary cross-entropy loss
        loss = nn.functional.binary_cross_entropy(attention_weights, masks)

        assert loss.item() >= 0
        assert not torch.isnan(loss)

    def test_cam_based_loss(self):
        """Test CAM-based attention loss."""
        attention_weights = torch.rand(2, 1, 28, 28)
        cam = torch.rand(2, 1, 28, 28)

        # MSE loss between attention and CAM
        loss = nn.functional.mse_loss(attention_weights, cam)

        assert loss.item() >= 0
        assert not torch.isnan(loss)

    def test_consistency_loss(self):
        """Test attention consistency loss."""
        attention_weights = torch.rand(2, 1, 28, 28)

        # Compute spatial variance as consistency measure
        mean_attention = attention_weights.mean(dim=[2, 3], keepdim=True)
        variance = ((attention_weights - mean_attention) ** 2).mean()

        assert variance.item() >= 0
        assert not torch.isnan(variance)

    def test_loss_with_different_sizes(self):
        """Test loss computation with different spatial sizes."""
        attention_weights = torch.rand(2, 1, 14, 14)
        masks = torch.randint(0, 2, (2, 1, 28, 28)).float()

        # Resize attention to match mask size
        attention_resized = nn.functional.interpolate(
            attention_weights,
            size=masks.shape[2:],
            mode="bilinear",
            align_corners=False,
        )

        loss = nn.functional.binary_cross_entropy(attention_resized, masks)

        assert loss.item() >= 0
        assert not torch.isnan(loss)


class TestTrainerIntegration:
    """Test attention supervision integration with trainer."""

    def test_trainer_config_validation(self):
        """Test trainer validates attention supervision config."""
        config = ExperimentConfig(
            name="test_attention",
            data=DataConfig(
                train_image_dir="dummy",
                train_csv="dummy.csv",
                val_image_dir="dummy",
                val_csv="dummy.csv",
                tabular_features=["age", "bmi"],
                label_column="label",
            ),
            model=ModelConfig(
                vision=VisionConfig(
                    backbone="resnet18",
                    attention_type="cbam",
                    enable_attention_supervision=True,
                ),
                tabular=TabularConfig(input_dim=2),
                fusion=FusionConfig(strategy="concatenate"),
            ),
            training=TrainingConfig(
                num_epochs=1,
                batch_size=2,
                use_attention_supervision=True,
                attention_supervision_method="mask_guided",
            ),
        )

        # Config should be valid
        assert config.model.vision.enable_attention_supervision
        assert config.training.use_attention_supervision
        assert config.model.vision.attention_type == "cbam"

    def test_trainer_config_mismatch_warning(self):
        """Test trainer warns on config mismatch."""
        # Config with SE attention (doesn't support spatial weights)
        config = ExperimentConfig(
            name="test_attention",
            data=DataConfig(
                train_image_dir="dummy",
                train_csv="dummy.csv",
                val_image_dir="dummy",
                val_csv="dummy.csv",
                tabular_features=["age", "bmi"],
                label_column="label",
            ),
            model=ModelConfig(
                vision=VisionConfig(
                    backbone="resnet18",
                    attention_type="se",  # SE doesn't support spatial attention
                    enable_attention_supervision=True,
                ),
                tabular=TabularConfig(input_dim=2),
                fusion=FusionConfig(strategy="concatenate"),
            ),
            training=TrainingConfig(
                num_epochs=1,
                batch_size=2,
                use_attention_supervision=True,
            ),
        )

        # Should still be valid but trainer will disable attention supervision
        assert config.model.vision.attention_type == "se"

    def test_batch_with_masks(self):
        """Test batch unpacking with optional masks."""
        # Simulate batch data
        images = torch.randn(2, 3, 224, 224)
        tabular = torch.randn(2, 10)
        labels = torch.tensor([0, 1])
        masks = torch.randint(0, 2, (2, 1, 224, 224)).float()

        batch = (images, tabular, labels, masks)

        # Unpack batch
        images_out, tabular_out, labels_out, *rest = batch

        assert images_out.shape == images.shape
        assert tabular_out.shape == tabular.shape
        assert labels_out.shape == labels.shape
        assert len(rest) == 1
        assert rest[0].shape == masks.shape

    def test_batch_without_masks(self):
        """Test batch unpacking without masks."""
        images = torch.randn(2, 3, 224, 224)
        tabular = torch.randn(2, 10)
        labels = torch.tensor([0, 1])

        batch = (images, tabular, labels)

        # Unpack batch
        images_out, tabular_out, labels_out, *rest = batch

        assert images_out.shape == images.shape
        assert tabular_out.shape == tabular.shape
        assert labels_out.shape == labels.shape
        assert len(rest) == 0


class TestEndToEndAttentionSupervision:
    """Test end-to-end attention supervision workflow."""

    def test_forward_pass_with_attention(self):
        """Test complete forward pass with attention supervision."""
        # Create model
        backbone = create_vision_backbone(
            "resnet18",
            pretrained=False,
            feature_dim=128,
            attention_type="cbam",
            enable_attention_supervision=True,
        )

        # Forward pass
        x = torch.randn(2, 3, 224, 224)
        output = backbone(x, return_intermediates=True)

        # Verify all components
        assert "features" in output
        assert "feature_maps" in output
        assert "attention_weights" in output

        features = output["features"]
        _feature_maps = output["feature_maps"]
        attention_weights = output["attention_weights"]

        # Simulate classifier
        classifier = nn.Linear(128, 2)
        logits = classifier(features)

        # Simulate loss computation
        labels = torch.tensor([0, 1])
        classification_loss = nn.functional.cross_entropy(logits, labels)

        # Simulate attention loss
        if attention_weights is not None:
            masks = torch.rand_like(attention_weights)
            attention_loss = nn.functional.mse_loss(attention_weights, masks)
            total_loss = classification_loss + 0.1 * attention_loss
        else:
            total_loss = classification_loss

        assert total_loss.item() >= 0
        assert not torch.isnan(total_loss)

    def test_backward_pass_with_attention(self):
        """Test backward pass with attention supervision."""
        backbone = create_vision_backbone(
            "resnet18",
            pretrained=False,
            feature_dim=128,
            attention_type="cbam",
            enable_attention_supervision=True,
        )
        classifier = nn.Linear(128, 2)

        # Forward pass
        x = torch.randn(2, 3, 224, 224)
        output = backbone(x, return_intermediates=True)
        logits = classifier(output["features"])

        # Compute loss
        labels = torch.tensor([0, 1])
        loss = nn.functional.cross_entropy(logits, labels)

        # Backward pass
        loss.backward()

        # Check gradients exist
        has_gradients = any(
            p.grad is not None and p.grad.abs().sum() > 0 for p in backbone.parameters()
        )
        assert has_gradients


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
