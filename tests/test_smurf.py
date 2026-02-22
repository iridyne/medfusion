"""
Unit tests for SMuRF Model.
"""

import pytest

# Skip this test file if med_core.models cannot be imported
pytest.importorskip("med_core.models")

import torch

from med_core.models.smurf import (
    SMuRFModel,
    SMuRFWithMIL,
    smurf_base,
    smurf_small,
    smurf_with_mil_small,
)


class TestSMuRFModel:
    """Test suite for SMuRFModel."""

    def test_basic_forward(self):
        """Test basic forward pass."""
        model = SMuRFModel(
            radiology_backbone={"variant": "tiny"},
            pathology_backbone={"variant": "tiny"},
            fusion_strategy="concat",
            num_classes=4,
        )

        ct = torch.randn(2, 1, 32, 64, 64)
        pathology = torch.randn(2, 3, 64, 64)

        logits = model(ct, pathology)

        assert logits.shape == (2, 4)

    def test_concat_fusion(self):
        """Test concatenation fusion."""
        model = SMuRFModel(
            radiology_backbone={"variant": "tiny"},
            pathology_backbone={"variant": "tiny"},
            fusion_strategy="concat",
            num_classes=4,
        )

        ct = torch.randn(2, 1, 32, 64, 64)
        pathology = torch.randn(2, 3, 64, 64)

        logits = model(ct, pathology)

        assert logits.shape == (2, 4)

    def test_kronecker_fusion(self):
        """Test Kronecker product fusion."""
        model = SMuRFModel(
            radiology_backbone={"variant": "tiny"},
            pathology_backbone={"variant": "tiny"},
            fusion_strategy="kronecker",
            num_classes=4,
        )

        ct = torch.randn(2, 1, 32, 64, 64)
        pathology = torch.randn(2, 3, 64, 64)

        logits = model(ct, pathology)

        assert logits.shape == (2, 4)

    def test_fused_attention_fusion(self):
        """Test fused attention fusion."""
        model = SMuRFModel(
            radiology_backbone={"variant": "tiny"},
            pathology_backbone={"variant": "tiny"},
            fusion_strategy="fused_attention",
            num_classes=4,
        )

        ct = torch.randn(2, 1, 32, 64, 64)
        pathology = torch.randn(2, 3, 64, 64)

        logits = model(ct, pathology)

        assert logits.shape == (2, 4)

    def test_return_features(self):
        """Test returning intermediate features."""
        model = SMuRFModel(
            radiology_backbone={"variant": "tiny"},
            pathology_backbone={"variant": "tiny"},
            fusion_strategy="fused_attention",
            num_classes=4,
        )

        ct = torch.randn(2, 1, 32, 64, 64)
        pathology = torch.randn(2, 3, 64, 64)

        logits, features = model(ct, pathology, return_features=True)

        assert logits.shape == (2, 4)
        assert "radiology" in features
        assert "pathology" in features
        assert "fused" in features
        assert features["radiology"].shape[0] == 2
        assert features["pathology"].shape[0] == 2
        assert features["fused"].shape[0] == 2

    def test_different_num_classes(self):
        """Test with different number of classes."""
        for num_classes in [2, 5, 10]:
            model = SMuRFModel(
                radiology_backbone={"variant": "tiny"},
                pathology_backbone={"variant": "tiny"},
                fusion_strategy="concat",
                num_classes=num_classes,
            )

            ct = torch.randn(2, 1, 32, 64, 64)
            pathology = torch.randn(2, 3, 64, 64)

            logits = model(ct, pathology)

            assert logits.shape == (2, num_classes)

    def test_gradient_flow(self):
        """Test gradient flow through the model."""
        model = SMuRFModel(
            radiology_backbone={"variant": "tiny"},
            pathology_backbone={"variant": "tiny"},
            fusion_strategy="fused_attention",
            num_classes=4,
        )

        ct = torch.randn(2, 1, 32, 64, 64, requires_grad=True)
        pathology = torch.randn(2, 3, 64, 64, requires_grad=True)

        logits = model(ct, pathology)
        loss = logits.sum()
        loss.backward()

        assert ct.grad is not None
        assert pathology.grad is not None
        assert not torch.isnan(ct.grad).any()
        assert not torch.isnan(pathology.grad).any()

    def test_batch_size_one(self):
        """Test with batch size of 1."""
        model = SMuRFModel(
            radiology_backbone={"variant": "tiny"},
            pathology_backbone={"variant": "tiny"},
            fusion_strategy="concat",
            num_classes=4,
        )

        ct = torch.randn(1, 1, 32, 64, 64)
        pathology = torch.randn(1, 3, 64, 64)

        logits = model(ct, pathology)

        assert logits.shape == (1, 4)

    def test_invalid_fusion_strategy(self):
        """Test error handling for invalid fusion strategy."""
        with pytest.raises(ValueError):
            SMuRFModel(
                radiology_backbone={"variant": "tiny"},
                pathology_backbone={"variant": "tiny"},
                fusion_strategy="invalid",
                num_classes=4,
            )


class TestSMuRFWithMIL:
    """Test suite for SMuRFWithMIL."""

    def test_basic_forward_with_mil(self):
        """Test basic forward pass with MIL."""
        model = SMuRFWithMIL(
            radiology_backbone={"variant": "tiny"},
            pathology_backbone={"variant": "tiny"},
            fusion_strategy="concat",
            num_classes=4,
        )

        ct = torch.randn(2, 1, 32, 64, 64)
        pathology_patches = torch.randn(2, 5, 3, 64, 64)  # 5 patches

        logits = model(ct, pathology_patches)

        assert logits.shape == (2, 4)

    def test_mil_with_different_num_patches(self):
        """Test MIL with different number of patches."""
        model = SMuRFWithMIL(
            radiology_backbone={"variant": "tiny"},
            pathology_backbone={"variant": "tiny"},
            fusion_strategy="concat",
            num_classes=4,
        )

        ct = torch.randn(2, 1, 32, 64, 64)

        for num_patches in [1, 5, 10, 20]:
            pathology_patches = torch.randn(2, num_patches, 3, 64, 64)
            logits = model(ct, pathology_patches)
            assert logits.shape == (2, 4)

    def test_mil_return_features(self):
        """Test returning features with MIL."""
        model = SMuRFWithMIL(
            radiology_backbone={"variant": "tiny"},
            pathology_backbone={"variant": "tiny"},
            fusion_strategy="fused_attention",
            num_classes=4,
        )

        ct = torch.randn(2, 1, 32, 64, 64)
        pathology_patches = torch.randn(2, 5, 3, 64, 64)

        logits, features = model(ct, pathology_patches, return_features=True)

        assert logits.shape == (2, 4)
        assert "radiology" in features
        assert "pathology_patches" in features
        assert "pathology_aggregated" in features
        assert "attention_weights" in features
        assert "fused" in features

        # Check shapes
        assert features["pathology_patches"].shape == (2, 5, 512)
        assert features["pathology_aggregated"].shape == (2, 512)
        assert features["attention_weights"].shape == (2, 5, 1)

        # Attention weights should sum to 1
        assert torch.allclose(
            features["attention_weights"].sum(dim=1),
            torch.ones(2, 1),
            atol=1e-5,
        )

    def test_mil_gradient_flow(self):
        """Test gradient flow with MIL."""
        model = SMuRFWithMIL(
            radiology_backbone={"variant": "tiny"},
            pathology_backbone={"variant": "tiny"},
            fusion_strategy="fused_attention",
            num_classes=4,
        )

        ct = torch.randn(2, 1, 32, 64, 64, requires_grad=True)
        pathology_patches = torch.randn(2, 5, 3, 64, 64, requires_grad=True)

        logits = model(ct, pathology_patches)
        loss = logits.sum()
        loss.backward()

        assert ct.grad is not None
        assert pathology_patches.grad is not None

    def test_mil_with_different_fusion_strategies(self):
        """Test MIL with different fusion strategies."""
        for fusion_strategy in ["concat", "kronecker", "fused_attention"]:
            model = SMuRFWithMIL(
                radiology_backbone={"variant": "tiny"},
                pathology_backbone={"variant": "tiny"},
                fusion_strategy=fusion_strategy,
                num_classes=4,
            )

            ct = torch.randn(2, 1, 32, 64, 64)
            pathology_patches = torch.randn(2, 5, 3, 64, 64)

            logits = model(ct, pathology_patches)

            assert logits.shape == (2, 4)


class TestPrebuiltModels:
    """Test suite for prebuilt model functions."""

    def test_smurf_small(self):
        """Test smurf_small function."""
        model = smurf_small(num_classes=4, fusion_strategy="concat")

        ct = torch.randn(2, 1, 32, 64, 64)
        pathology = torch.randn(2, 3, 64, 64)

        logits = model(ct, pathology)

        assert logits.shape == (2, 4)

    def test_smurf_base(self):
        """Test smurf_base function."""
        model = smurf_base(num_classes=4, fusion_strategy="concat")

        ct = torch.randn(2, 1, 32, 64, 64)
        pathology = torch.randn(2, 3, 64, 64)

        logits = model(ct, pathology)

        assert logits.shape == (2, 4)

    def test_smurf_with_mil_small(self):
        """Test smurf_with_mil_small function."""
        model = smurf_with_mil_small(num_classes=4, fusion_strategy="concat")

        ct = torch.randn(2, 1, 32, 64, 64)
        pathology_patches = torch.randn(2, 5, 3, 64, 64)

        logits = model(ct, pathology_patches)

        assert logits.shape == (2, 4)

    def test_prebuilt_with_different_fusion(self):
        """Test prebuilt models with different fusion strategies."""
        for fusion_strategy in ["concat", "kronecker", "fused_attention"]:
            model = smurf_small(num_classes=4, fusion_strategy=fusion_strategy)

            ct = torch.randn(2, 1, 32, 64, 64)
            pathology = torch.randn(2, 3, 64, 64)

            logits = model(ct, pathology)

            assert logits.shape == (2, 4)


class TestIntegration:
    """Integration tests for SMuRF models."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda_forward(self):
        """Test forward pass on CUDA."""
        model = smurf_small(num_classes=4).cuda()

        ct = torch.randn(2, 1, 32, 64, 64).cuda()
        pathology = torch.randn(2, 3, 64, 64).cuda()

        logits = model(ct, pathology)

        assert logits.shape == (2, 4)
        assert logits.device.type == "cuda"

    def test_training_mode(self):
        """Test model in training mode."""
        model = smurf_small(num_classes=4)
        model.train()

        ct = torch.randn(2, 1, 32, 64, 64)
        pathology = torch.randn(2, 3, 64, 64)

        logits = model(ct, pathology)

        assert logits.shape == (2, 4)
        assert model.training

    def test_eval_mode(self):
        """Test model in eval mode."""
        model = smurf_small(num_classes=4)
        model.eval()

        ct = torch.randn(2, 1, 32, 64, 64)
        pathology = torch.randn(2, 3, 64, 64)

        with torch.no_grad():
            logits = model(ct, pathology)

        assert logits.shape == (2, 4)
        assert not model.training

    def test_parameter_count(self):
        """Test parameter count is reasonable."""
        model = smurf_small(num_classes=4)

        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        # Should have millions of parameters but not too many
        assert 1_000_000 < total_params < 100_000_000
        assert trainable_params > 0

    def test_different_input_sizes(self):
        """Test with different input sizes."""
        model = smurf_small(num_classes=4)

        # Different CT sizes (must be divisible by window sizes)
        for d, h, w in [(32, 64, 64), (64, 128, 128)]:
            ct = torch.randn(2, 1, d, h, w)
            pathology = torch.randn(2, 3, 64, 64)
            logits = model(ct, pathology)
            assert logits.shape == (2, 4)

    def test_multimodal_fusion_effectiveness(self):
        """Test that fusion combines both modalities."""
        model = smurf_small(num_classes=4, fusion_strategy="concat")

        ct = torch.randn(2, 1, 32, 64, 64)
        pathology = torch.randn(2, 3, 64, 64)

        # Get features
        logits, features = model(ct, pathology, return_features=True)

        # Check that we have features from both modalities
        assert features["radiology"].shape == (2, 512)
        assert features["pathology"].shape == (2, 512)
        assert features["fused"].shape == (2, 1024)  # concat: 512 + 512
