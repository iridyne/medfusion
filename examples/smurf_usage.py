"""
SMuRF Model Usage Examples

This script demonstrates how to use the SMuRF (Survival Multimodal
Radiology-Pathology Fusion) model for multimodal medical imaging analysis.
"""

import torch

from med_core.models import smurf_small, smurf_with_mil_small


def example_basic_usage():
    """Basic usage of SMuRF model."""
    print("=" * 60)
    print("Example 1: Basic SMuRF Model")
    print("=" * 60)

    # Create model
    model = smurf_small(num_classes=4, fusion_strategy="fused_attention")

    # Prepare dummy data
    ct_scan = torch.randn(2, 1, 64, 128, 128)  # [B, C, D, H, W]
    pathology = torch.randn(2, 3, 224, 224)  # [B, C, H, W]

    # Forward pass
    model.eval()
    with torch.no_grad():
        logits = model(ct_scan, pathology)

    print(f"Input CT shape: {ct_scan.shape}")
    print(f"Input pathology shape: {pathology.shape}")
    print(f"Output logits shape: {logits.shape}")
    print(f"Predicted classes: {logits.argmax(dim=1)}")
    print()


def example_with_features():
    """Get intermediate features from SMuRF model."""
    print("=" * 60)
    print("Example 2: SMuRF with Feature Extraction")
    print("=" * 60)

    # Create model
    model = smurf_small(num_classes=4, fusion_strategy="fused_attention")

    # Prepare data
    ct_scan = torch.randn(2, 1, 64, 128, 128)
    pathology = torch.randn(2, 3, 224, 224)

    # Get features
    model.eval()
    with torch.no_grad():
        logits, features = model(ct_scan, pathology, return_features=True)

    print(f"Radiology features: {features['radiology'].shape}")
    print(f"Pathology features: {features['pathology'].shape}")
    print(f"Fused features: {features['fused'].shape}")
    print(f"Output logits: {logits.shape}")
    print()


def example_mil():
    """SMuRF with Multiple Instance Learning for pathology."""
    print("=" * 60)
    print("Example 3: SMuRF with MIL")
    print("=" * 60)

    # Create model with MIL
    model = smurf_with_mil_small(num_classes=4, fusion_strategy="fused_attention")

    # Prepare data with multiple pathology patches
    ct_scan = torch.randn(2, 1, 64, 128, 128)
    pathology_patches = torch.randn(2, 10, 3, 224, 224)  # 10 patches per sample

    # Forward pass
    model.eval()
    with torch.no_grad():
        logits, features = model(ct_scan, pathology_patches, return_features=True)

    print(f"Input CT shape: {ct_scan.shape}")
    print(f"Input pathology patches: {pathology_patches.shape}")
    print(f"Pathology patch features: {features['pathology_patches'].shape}")
    print(f"Aggregated pathology features: {features['pathology_aggregated'].shape}")
    print(f"Attention weights: {features['attention_weights'].shape}")
    print(f"Output logits: {logits.shape}")

    # Show attention weights for first sample
    attention = features["attention_weights"][0].squeeze()
    print("\nAttention weights for first sample:")
    for i, weight in enumerate(attention):
        print(f"  Patch {i}: {weight.item():.4f}")
    print()


def example_different_fusion_strategies():
    """Compare different fusion strategies."""
    print("=" * 60)
    print("Example 4: Different Fusion Strategies")
    print("=" * 60)

    # Prepare data
    ct_scan = torch.randn(2, 1, 64, 128, 128)
    pathology = torch.randn(2, 3, 224, 224)

    strategies = ["concat", "kronecker", "fused_attention"]

    for strategy in strategies:
        model = smurf_small(num_classes=4, fusion_strategy=strategy)
        model.eval()

        with torch.no_grad():
            logits = model(ct_scan, pathology)

        num_params = sum(p.numel() for p in model.parameters())
        print(f"{strategy:20s} - Params: {num_params:,} - Output: {logits.shape}")

    print()


def example_training_setup():
    """Example training setup."""
    print("=" * 60)
    print("Example 5: Training Setup")
    print("=" * 60)

    # Create model
    model = smurf_small(num_classes=4, fusion_strategy="fused_attention")

    # Setup optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)

    # Setup loss
    criterion = torch.nn.CrossEntropyLoss()

    # Dummy training data
    ct_scan = torch.randn(4, 1, 64, 128, 128)
    pathology = torch.randn(4, 3, 224, 224)
    labels = torch.randint(0, 4, (4,))

    # Training step
    model.train()
    optimizer.zero_grad()

    logits = model(ct_scan, pathology)
    loss = criterion(logits, labels)

    loss.backward()
    optimizer.step()

    print(f"Batch size: {ct_scan.shape[0]}")
    print(f"Loss: {loss.item():.4f}")
    print(f"Predictions: {logits.argmax(dim=1)}")
    print(f"Ground truth: {labels}")
    print()


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("SMuRF Model Examples")
    print("=" * 60 + "\n")

    example_basic_usage()
    example_with_features()
    example_mil()
    example_different_fusion_strategies()
    example_training_setup()

    print("=" * 60)
    print("All examples completed successfully!")
    print("=" * 60)
