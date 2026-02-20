"""
Multi-Modal Model Builder Demo

This script demonstrates how to use the MultiModalModelBuilder to create
various multi-modal models for medical imaging tasks.
"""

import torch

from med_core.models import (
    MultiModalModelBuilder,
    smurf_small,
)


def example_basic_usage():
    """Example 1: Basic multi-modal model with builder API."""
    print("=" * 80)
    print("Example 1: Basic Multi-Modal Model")
    print("=" * 80)

    # Build a simple 2-modality model
    model = (
        MultiModalModelBuilder()
        .add_modality("xray", backbone="resnet18", modality_type="vision")
        .add_modality("clinical", backbone="mlp", modality_type="tabular", input_dim=10)
        .set_fusion("attention")
        .set_head("classification", num_classes=2)
        .build()
    )

    # Test forward pass
    xray = torch.randn(4, 3, 224, 224)
    clinical = torch.randn(4, 10)

    inputs = {"xray": xray, "clinical": clinical}
    logits = model(inputs)

    print(f"X-ray input shape: {xray.shape}")
    print(f"Clinical input shape: {clinical.shape}")
    print(f"Output logits shape: {logits.shape}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print()


def example_smurf_equivalent():
    """Example 2: Build SMuRF model using generic builder."""
    print("=" * 80)
    print("Example 2: SMuRF Model with Generic Builder")
    print("=" * 80)

    # Build SMuRF using generic builder
    model = (
        MultiModalModelBuilder()
        .add_modality(
            "radiology",
            backbone="swin3d_small",
            modality_type="vision3d",
            in_channels=1,
            feature_dim=512,
        )
        .add_modality(
            "pathology",
            backbone="swin2d_small",
            modality_type="vision",
            in_channels=3,
            feature_dim=512,
        )
        .set_fusion("fused_attention", num_heads=8, use_kronecker=True, output_dim=256)
        .set_head("classification", num_classes=4, dropout=0.3)
        .build()
    )

    # Compare with original SMuRF
    smurf_original = smurf_small(num_classes=4, fusion_strategy="fused_attention")

    # Test forward pass
    ct = torch.randn(2, 1, 64, 128, 128)
    pathology = torch.randn(2, 3, 224, 224)

    inputs = {"radiology": ct, "pathology": pathology}
    logits = model(inputs)

    print(f"CT input shape: {ct.shape}")
    print(f"Pathology input shape: {pathology.shape}")
    print(f"Output logits shape: {logits.shape}")
    print(f"Generic builder params: {sum(p.numel() for p in model.parameters()):,}")
    print(
        f"Original SMuRF params: {sum(p.numel() for p in smurf_original.parameters()):,}"
    )
    print()


def example_mil_aggregation():
    """Example 3: Multi-modal model with MIL aggregation."""
    print("=" * 80)
    print("Example 3: Multi-Modal Model with MIL")
    print("=" * 80)

    # Build model with MIL for pathology
    model = (
        MultiModalModelBuilder()
        .add_modality(
            "radiology",
            backbone="swin3d_small",
            modality_type="vision3d",
            in_channels=1,
            feature_dim=512,
        )
        .add_modality(
            "pathology",
            backbone="swin2d_small",
            modality_type="vision",
            in_channels=3,
            feature_dim=512,
        )
        .add_mil_aggregation("pathology", strategy="attention", attention_dim=128)
        .set_fusion("fused_attention", num_heads=8, output_dim=256)
        .set_head("classification", num_classes=4)
        .build()
    )

    # Test with multiple pathology patches
    ct = torch.randn(2, 1, 64, 128, 128)
    pathology_patches = torch.randn(2, 10, 3, 224, 224)  # 10 patches per sample

    inputs = {"radiology": ct, "pathology": pathology_patches}
    logits, features = model(inputs, return_features=True)

    print(f"CT input shape: {ct.shape}")
    print(f"Pathology patches shape: {pathology_patches.shape}")
    print(f"Output logits shape: {logits.shape}")

    if "mil_attention_weights" in features:
        attention = features["mil_attention_weights"]["pathology"]
        print(f"MIL attention weights shape: {attention.shape}")
        print(f"Attention weights (sample 0): {attention[0].squeeze().tolist()}")
    print()


def example_different_fusion_strategies():
    """Example 4: Compare different fusion strategies."""
    print("=" * 80)
    print("Example 4: Different Fusion Strategies")
    print("=" * 80)

    strategies = ["concat", "gated", "attention", "kronecker", "fused_attention"]

    # Prepare test data
    xray = torch.randn(4, 3, 224, 224)
    clinical = torch.randn(4, 10)
    inputs = {"xray": xray, "clinical": clinical}

    for strategy in strategies:
        model = (
            MultiModalModelBuilder()
            .add_modality("xray", backbone="resnet18", modality_type="vision")
            .add_modality(
                "clinical", backbone="mlp", modality_type="tabular", input_dim=10
            )
            .set_fusion(strategy)
            .set_head("classification", num_classes=2)
            .build()
        )

        logits = model(inputs)
        num_params = sum(p.numel() for p in model.parameters())

        print(f"{strategy:20s} - Params: {num_params:,} - Output: {logits.shape}")

    print()


def example_three_modalities():
    """Example 5: Model with three modalities."""
    print("=" * 80)
    print("Example 5: Three-Modality Model")
    print("=" * 80)

    # Build model with 3 modalities
    model = (
        MultiModalModelBuilder()
        .add_modality(
            "xray",
            backbone="resnet18",
            modality_type="vision",
            feature_dim=256,
        )
        .add_modality(
            "ct",
            backbone="swin3d_tiny",
            modality_type="vision3d",
            in_channels=1,
            feature_dim=256,
        )
        .add_modality(
            "clinical",
            backbone="mlp",
            modality_type="tabular",
            input_dim=15,
            feature_dim=256,
        )
        .set_fusion("concat")  # For >2 modalities, uses mean pooling internally
        .set_head("classification", num_classes=3)
        .build()
    )

    # Test forward pass
    xray = torch.randn(2, 3, 224, 224)
    ct = torch.randn(2, 1, 32, 64, 64)
    clinical = torch.randn(2, 15)

    inputs = {"xray": xray, "ct": ct, "clinical": clinical}
    logits, features = model(inputs, return_features=True)

    print(f"X-ray input shape: {xray.shape}")
    print(f"CT input shape: {ct.shape}")
    print(f"Clinical input shape: {clinical.shape}")
    print(f"Output logits shape: {logits.shape}")
    print(f"Fused features shape: {features['fused_features'].shape}")

    # Get modality contributions
    contributions = model.get_modality_contribution()
    print("\nModality contributions:")
    for modality, contrib in contributions.items():
        print(f"  {modality}: {contrib:.3f}")
    print()


def example_survival_analysis():
    """Example 6: Multi-modal model for survival analysis."""
    print("=" * 80)
    print("Example 6: Survival Analysis Model")
    print("=" * 80)

    # Build model for Cox survival analysis
    model = (
        MultiModalModelBuilder()
        .add_modality(
            "radiology",
            backbone="swin3d_small",
            modality_type="vision3d",
            in_channels=1,
            feature_dim=512,
        )
        .add_modality(
            "pathology",
            backbone="swin2d_small",
            modality_type="vision",
            in_channels=3,
            feature_dim=512,
        )
        .set_fusion("fused_attention", num_heads=8, output_dim=256)
        .set_head("survival_cox", hidden_dims=[128, 64])
        .build()
    )

    # Test forward pass
    ct = torch.randn(2, 1, 64, 128, 128)
    pathology = torch.randn(2, 3, 224, 224)

    inputs = {"radiology": ct, "pathology": pathology}
    risk_scores = model(inputs)

    print(f"CT input shape: {ct.shape}")
    print(f"Pathology input shape: {pathology.shape}")
    print(f"Risk scores shape: {risk_scores.shape}")
    print(f"Risk scores: {risk_scores.squeeze().tolist()}")
    print()


def example_from_config():
    """Example 7: Build model from configuration dict."""
    print("=" * 80)
    print("Example 7: Build Model from Configuration")
    print("=" * 80)

    # Define configuration
    config = {
        "modalities": {
            "xray": {
                "backbone": "resnet18",
                "modality_type": "vision",
                "feature_dim": 256,
            },
            "clinical": {
                "backbone": "mlp",
                "modality_type": "tabular",
                "input_dim": 10,
                "feature_dim": 64,
            },
        },
        "fusion": {"strategy": "attention", "output_dim": 128},
        "head": {"task_type": "classification", "num_classes": 2, "dropout": 0.3},
    }

    # Build from config
    builder = MultiModalModelBuilder.from_config(config)
    model = builder.build()

    # Test forward pass
    xray = torch.randn(4, 3, 224, 224)
    clinical = torch.randn(4, 10)

    inputs = {"xray": xray, "clinical": clinical}
    logits = model(inputs)

    print("Configuration:")
    print(f"  Modalities: {list(config['modalities'].keys())}")
    print(f"  Fusion: {config['fusion']['strategy']}")
    print(f"  Task: {config['head']['task_type']}")
    print(f"\nOutput logits shape: {logits.shape}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print()


def example_feature_extraction():
    """Example 8: Extract intermediate features."""
    print("=" * 80)
    print("Example 8: Feature Extraction")
    print("=" * 80)

    model = (
        MultiModalModelBuilder()
        .add_modality(
            "xray", backbone="resnet18", modality_type="vision", in_channels=1
        )
        .add_modality("clinical", backbone="mlp", modality_type="tabular", input_dim=10)
        .set_fusion("attention")
        .set_head("classification", num_classes=2)
        .build()
    )

    # Extract features
    xray = torch.randn(4, 3, 224, 224)
    clinical = torch.randn(4, 10)

    inputs = {"xray": xray, "clinical": clinical}
    logits, features = model(inputs, return_features=True)

    print("Extracted features:")
    print(f"  X-ray features: {features['modality_features']['xray'].shape}")
    print(f"  Clinical features: {features['modality_features']['clinical'].shape}")
    print(f"  Fused features: {features['fused_features'].shape}")

    if "fusion_aux" in features and features["fusion_aux"] is not None:
        print(f"  Fusion auxiliary outputs: {list(features['fusion_aux'].keys())}")

    print()


def example_different_mil_strategies():
    """Example 9: Compare different MIL aggregation strategies."""
    print("=" * 80)
    print("Example 9: Different MIL Aggregation Strategies")
    print("=" * 80)

    strategies = ["mean", "max", "attention", "gated", "deepsets", "transformer"]

    # Prepare test data
    ct = torch.randn(2, 1, 64, 128, 128)
    pathology_patches = torch.randn(2, 10, 3, 224, 224)
    inputs = {"radiology": ct, "pathology": pathology_patches}

    for strategy in strategies:
        model = (
            MultiModalModelBuilder()
            .add_modality(
                "radiology",
                backbone="swin3d_tiny",
                modality_type="vision3d",
                in_channels=1,
                feature_dim=256,
            )
            .add_modality(
                "pathology",
                backbone="resnet18",
                modality_type="vision",
                feature_dim=256,
            )
            .add_mil_aggregation("pathology", strategy=strategy)
            .set_fusion("concat")
            .set_head("classification", num_classes=4)
            .build()
        )

        logits = model(inputs)
        num_params = sum(p.numel() for p in model.parameters())

        print(f"{strategy:15s} - Params: {num_params:,} - Output: {logits.shape}")

    print()


def example_custom_backbone():
    """Example 10: Use custom backbone module."""
    print("=" * 80)
    print("Example 10: Custom Backbone Module")
    print("=" * 80)

    # Define custom backbone
    class CustomBackbone(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = torch.nn.Conv2d(3, 64, 3, padding=1)
            self.pool = torch.nn.AdaptiveAvgPool2d(1)
            self.fc = torch.nn.Linear(64, 128)
            self.output_dim = 128

        def forward(self, x):
            x = self.conv(x)
            x = torch.relu(x)
            x = self.pool(x)
            x = x.flatten(1)
            x = self.fc(x)
            return x

    # Use custom backbone
    custom_backbone = CustomBackbone()

    model = (
        MultiModalModelBuilder()
        .add_modality("xray", backbone=custom_backbone, modality_type="custom")
        .add_modality("clinical", backbone="mlp", modality_type="tabular", input_dim=10)
        .set_fusion("concat")
        .set_head("classification", num_classes=2)
        .build()
    )

    # Test forward pass
    xray = torch.randn(4, 3, 224, 224)
    clinical = torch.randn(4, 10)

    inputs = {"xray": xray, "clinical": clinical}
    logits = model(inputs)

    print(f"Custom backbone output dim: {custom_backbone.output_dim}")
    print(f"Output logits shape: {logits.shape}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print()


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("Multi-Modal Model Builder Demo")
    print("=" * 80 + "\n")

    example_basic_usage()
    example_smurf_equivalent()
    example_mil_aggregation()
    example_different_fusion_strategies()
    example_three_modalities()
    example_survival_analysis()
    example_from_config()
    example_feature_extraction()
    example_different_mil_strategies()
    example_custom_backbone()

    print("=" * 80)
    print("All examples completed successfully!")
    print("=" * 80)
