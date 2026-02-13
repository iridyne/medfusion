import unittest

import torch

from med_core.backbones import create_tabular_backbone, create_vision_backbone


class TestVisionBackbones(unittest.TestCase):
    def setUp(self):
        self.batch_size = 2
        self.channels = 3
        self.height = 224
        self.width = 224
        self.input_tensor = torch.randn(self.batch_size, self.channels, self.height, self.width)

    def test_resnet18_creation(self):
        feature_dim = 128
        backbone = create_vision_backbone(
            "resnet18",
            pretrained=False,
            feature_dim=feature_dim
        )
        output = backbone(self.input_tensor)
        self.assertEqual(output.shape, (self.batch_size, feature_dim))

    def test_mobilenet_creation(self):
        feature_dim = 64
        backbone = create_vision_backbone(
            "mobilenetv2",
            pretrained=False,
            feature_dim=feature_dim
        )
        output = backbone(self.input_tensor)
        self.assertEqual(output.shape, (self.batch_size, feature_dim))

    def test_freezing(self):
        backbone = create_vision_backbone(
            "resnet18",
            pretrained=False,
            freeze=True
        )
        # Check if parameters require grad
        # Note: The projection head should remain trainable even if backbone is frozen
        # But the underlying backbone layers should be frozen

        # Access the internal backbone module (implementation detail check)
        if hasattr(backbone, "_backbone"):
            for param in backbone._backbone.parameters():
                self.assertFalse(param.requires_grad)

class TestTabularBackbones(unittest.TestCase):
    def setUp(self):
        self.batch_size = 4
        self.input_dim = 10
        self.input_tensor = torch.randn(self.batch_size, self.input_dim)

    def test_mlp_creation(self):
        output_dim = 32
        backbone = create_tabular_backbone(
            input_dim=self.input_dim,
            output_dim=output_dim,
            hidden_dims=[64, 64],
            backbone_type="mlp"
        )
        output = backbone(self.input_tensor)
        self.assertEqual(output.shape, (self.batch_size, output_dim))

    def test_residual_creation(self):
        output_dim = 16
        backbone = create_tabular_backbone(
            input_dim=self.input_dim,
            output_dim=output_dim,
            hidden_dims=[32, 32],
            backbone_type="residual"
        )
        output = backbone(self.input_tensor)
        self.assertEqual(output.shape, (self.batch_size, output_dim))

if __name__ == "__main__":
    unittest.main()
