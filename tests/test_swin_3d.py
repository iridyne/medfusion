"""
Unit tests for 3D Swin Transformer backbone.
"""

import pytest
import torch

from med_core.backbones.swin_3d import (
    swin3d_small,
    swin3d_tiny,
)


class TestSwinTransformer3DBackbone:
    """Test suite for 3D Swin Transformer backbone."""

    def test_tiny_variant_forward(self):
        """Test forward pass with tiny variant."""
        backbone = swin3d_tiny(in_channels=1, feature_dim=128)
        x = torch.randn(2, 1, 32, 64, 64)

        output = backbone(x)

        assert output.shape == (2, 128), f"Expected shape (2, 128), got {output.shape}"
        assert not torch.isnan(output).any(), "Output contains NaN values"
        assert not torch.isinf(output).any(), "Output contains Inf values"

    def test_small_variant_forward(self):
        """Test forward pass with small variant."""
        backbone = swin3d_small(in_channels=1, feature_dim=256)
        x = torch.randn(1, 1, 32, 64, 64)  # Larger input for 4 stages

        output = backbone(x)

        assert output.shape == (1, 256), f"Expected shape (1, 256), got {output.shape}"

    def test_multi_channel_input(self):
        """Test with multi-channel input (e.g., multi-sequence MRI)."""
        backbone = swin3d_tiny(in_channels=4, feature_dim=128)
        x = torch.randn(2, 4, 32, 64, 64)

        output = backbone(x)

        assert output.shape == (2, 128)

    def test_return_intermediates(self):
        """Test returning intermediate features."""
        backbone = swin3d_tiny(in_channels=1, feature_dim=128, return_intermediate=True)
        x = torch.randn(2, 1, 32, 64, 64)

        output = backbone(x)

        assert isinstance(output, dict), (
            "Expected dict output when return_intermediate=True"
        )
        assert "features" in output, "Missing 'features' key"
        assert "intermediate" in output, "Missing 'intermediate' key"
        assert "hidden_states" in output, "Missing 'hidden_states' key"
        assert output["features"].shape == (2, 128)

    def test_gradient_flow(self):
        """Test that gradients flow correctly."""
        backbone = swin3d_tiny(in_channels=1, feature_dim=128)
        x = torch.randn(2, 1, 32, 64, 64, requires_grad=True)

        output = backbone(x)
        loss = output.sum()
        loss.backward()

        assert x.grad is not None, "Gradients not flowing to input"
        assert not torch.isnan(x.grad).any(), "Gradient contains NaN"

    def test_freeze_backbone(self):
        """Test freezing backbone parameters."""
        backbone = swin3d_tiny(in_channels=1, feature_dim=128, freeze=True)

        # Check that backbone parameters are frozen
        for name, param in backbone.named_parameters():
            if "_backbone" in name:
                assert not param.requires_grad, f"Parameter {name} should be frozen"

    def test_different_input_sizes(self):
        """Test with different input spatial sizes."""
        backbone = swin3d_tiny(in_channels=1, feature_dim=128)

        # Test various sizes
        sizes = [
            (1, 1, 16, 32, 32),
            (1, 1, 32, 64, 64),
            (1, 1, 64, 128, 128),
        ]

        for size in sizes:
            x = torch.randn(*size)
            output = backbone(x)
            assert output.shape == (size[0], 128), f"Failed for input size {size}"

    def test_batch_size_one(self):
        """Test with batch size of 1."""
        backbone = swin3d_tiny(in_channels=1, feature_dim=128)
        x = torch.randn(1, 1, 32, 64, 64)

        output = backbone(x)

        assert output.shape == (1, 128)

    def test_config_serialization(self):
        """Test configuration serialization."""
        backbone = swin3d_tiny(in_channels=1, feature_dim=128)
        config = backbone.get_config()

        assert "variant" in config
        assert "in_channels" in config
        assert "feature_dim" in config
        assert config["variant"] == "tiny"
        assert config["in_channels"] == 1
        assert config["feature_dim"] == 128

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda_forward(self):
        """Test forward pass on CUDA."""
        backbone = swin3d_tiny(in_channels=1, feature_dim=128).cuda()
        x = torch.randn(2, 1, 32, 64, 64).cuda()

        output = backbone(x)

        assert output.is_cuda, "Output should be on CUDA"
        assert output.shape == (2, 128)

    def test_output_dim_property(self):
        """Test output_dim property."""
        backbone = swin3d_tiny(in_channels=1, feature_dim=256)

        assert backbone.output_dim == 256
        assert backbone.feature_dim == 256


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
