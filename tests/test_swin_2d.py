"""
Unit tests for 2D Swin Transformer Backbone.
"""

import pytest
import torch

from med_core.backbones.swin_2d import (
    SwinTransformer2DBackbone,
    swin2d_tiny,
    swin2d_small,
    swin2d_base,
)


class TestSwinTransformer2DBackbone:
    """Test suite for SwinTransformer2DBackbone."""

    def test_tiny_variant_forward(self):
        """Test forward pass with tiny variant."""
        backbone = swin2d_tiny(in_channels=3, feature_dim=512)
        x = torch.randn(2, 3, 224, 224)

        output = backbone(x)

        assert output.shape == (2, 512), f"Expected shape (2, 512), got {output.shape}"

    def test_small_variant_forward(self):
        """Test forward pass with small variant."""
        backbone = swin2d_small(in_channels=3, feature_dim=256)
        x = torch.randn(1, 3, 224, 224)

        output = backbone(x)

        assert output.shape == (1, 256), f"Expected shape (1, 256), got {output.shape}"

    def test_multi_channel_input(self):
        """Test with different number of input channels."""
        backbone = swin2d_tiny(in_channels=1, feature_dim=128)  # Grayscale
        x = torch.randn(2, 1, 224, 224)

        output = backbone(x)

        assert output.shape == (2, 128)

    def test_return_intermediates(self):
        """Test returning intermediate features."""
        backbone = swin2d_tiny(in_channels=3, feature_dim=128, return_intermediate=True)
        x = torch.randn(1, 3, 224, 224)

        output = backbone(x)

        assert isinstance(output, dict)
        assert "features" in output
        assert "intermediate" in output
        assert "hidden_states" in output
        assert output["features"].shape == (1, 128)
        assert len(output["intermediate"]) == 4  # 4 stages

    def test_gradient_flow(self):
        """Test gradient flow through the network."""
        backbone = swin2d_tiny(in_channels=3, feature_dim=128)
        x = torch.randn(2, 3, 224, 224, requires_grad=True)

        output = backbone(x)
        loss = output.sum()
        loss.backward()

        assert x.grad is not None
        assert not torch.isnan(x.grad).any()

    def test_freeze_backbone(self):
        """Test freezing backbone parameters."""
        backbone = swin2d_tiny(in_channels=3, feature_dim=128, freeze=True)

        # Check that backbone parameters are frozen
        for param in backbone._backbone.parameters():
            assert not param.requires_grad

        # Check that projection head is not frozen
        for param in backbone._projection.parameters():
            if isinstance(param, torch.nn.Parameter):
                assert param.requires_grad

    def test_different_input_sizes(self):
        """Test with different input sizes."""
        backbone = swin2d_tiny(in_channels=3, feature_dim=128, img_size=(256, 256))
        x = torch.randn(1, 3, 256, 256)

        output = backbone(x)

        assert output.shape == (1, 128)

    def test_batch_size_one(self):
        """Test with batch size of 1."""
        backbone = swin2d_tiny(in_channels=3, feature_dim=128)
        x = torch.randn(1, 3, 224, 224)

        output = backbone(x)

        assert output.shape == (1, 128)

    def test_config_serialization(self):
        """Test configuration serialization."""
        backbone = swin2d_tiny(in_channels=3, feature_dim=256)
        config = backbone.get_config()

        assert config["variant"] == "tiny"
        assert config["in_channels"] == 3
        assert config["feature_dim"] == 256

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda_forward(self):
        """Test forward pass on CUDA."""
        backbone = swin2d_tiny(in_channels=3, feature_dim=128).cuda()
        x = torch.randn(2, 3, 224, 224).cuda()

        output = backbone(x)

        assert output.shape == (2, 128)
        assert output.device.type == "cuda"

    def test_output_dim_property(self):
        """Test output_dim property."""
        backbone = swin2d_tiny(in_channels=3, feature_dim=256)

        assert backbone.output_dim == 256

    def test_base_variant(self):
        """Test base variant."""
        backbone = swin2d_base(in_channels=3, feature_dim=512)
        x = torch.randn(1, 3, 224, 224)

        output = backbone(x)

        assert output.shape == (1, 512)

    def test_dropout(self):
        """Test with dropout enabled."""
        backbone = swin2d_tiny(in_channels=3, feature_dim=128, dropout=0.5)
        backbone.train()  # Enable dropout
        x = torch.randn(2, 3, 224, 224)

        output = backbone(x)

        assert output.shape == (2, 128)

    def test_unfreeze_backbone(self):
        """Test unfreezing backbone."""
        backbone = swin2d_tiny(in_channels=3, feature_dim=128, freeze=True)
        backbone.unfreeze_backbone()

        # Check that all parameters are trainable
        for param in backbone._backbone.parameters():
            assert param.requires_grad
