"""
Integration test example for multi-view multimodal framework.

This script demonstrates the complete pipeline from data loading to training.
It uses synthetic data for testing purposes.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

# Test if imports work
print("Testing imports...")
from med_core.backbones import create_multiview_vision_backbone
from med_core.configs import create_ct_multiview_config
from med_core.datasets.multiview_types import MultiViewConfig, ViewDict
from med_core.fusion import create_multiview_fusion_model
from med_core.trainers import create_multiview_trainer

print("✓ All imports successful")


class SyntheticMultiViewDataset(Dataset):
    """Synthetic dataset for testing multi-view functionality."""

    def __init__(
        self,
        num_samples: int = 100,
        view_names: list[str] = None,
        image_size: int = 224,
        num_tabular_features: int = 10,
        num_classes: int = 2,
    ):
        self.num_samples = num_samples
        self.view_names = view_names or ["axial", "coronal", "sagittal"]
        self.image_size = image_size
        self.num_tabular_features = num_tabular_features
        self.num_classes = num_classes

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Generate synthetic multi-view images
        images = {
            view_name: torch.randn(3, self.image_size, self.image_size)
            for view_name in self.view_names
        }

        # Generate synthetic tabular data
        tabular = torch.randn(self.num_tabular_features)

        # Generate random label
        label = torch.randint(0, self.num_classes, (1,)).item()

        return images, tabular, label


def test_multiview_pipeline():
    """Test the complete multi-view pipeline."""

    print("\n" + "="*60)
    print("Multi-View Multimodal Framework Integration Test")
    print("="*60)

    # 1. Configuration
    print("\n[1/6] Creating configuration...")
    config = create_ct_multiview_config(
        view_names=["axial", "coronal", "sagittal"],
        aggregator_type="attention",
        backbone="resnet18",
    )
    config.data.batch_size = 4
    config.training.num_epochs = 2
    config.training.use_progressive_training = False
    config.model.vision.pretrained = False  # Faster for testing
    print(f"✓ Config created: {len(config.data.view_names)} views")

    # 2. Dataset
    print("\n[2/6] Creating synthetic datasets...")
    train_dataset = SyntheticMultiViewDataset(
        num_samples=20,
        view_names=config.data.view_names,
        num_tabular_features=10,
    )
    val_dataset = SyntheticMultiViewDataset(
        num_samples=10,
        view_names=config.data.view_names,
        num_tabular_features=10,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.data.batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.data.batch_size,
        shuffle=False,
    )
    print(f"✓ Train: {len(train_dataset)} samples, Val: {len(val_dataset)} samples")

    # 3. Model
    print("\n[3/6] Creating multi-view fusion model...")
    model = create_multiview_fusion_model(
        vision_backbone_name="resnet18",
        tabular_input_dim=10,
        fusion_type="gated",
        num_classes=2,
        vision_feature_dim=128,
        tabular_feature_dim=32,
        fusion_output_dim=64,
        aggregator_type="attention",
        share_weights=True,
        view_names=config.data.view_names,
        vision_kwargs={"pretrained": False},
    )
    print(f"✓ Model created with {sum(p.numel() for p in model.parameters()):,} parameters")

    # 4. Test forward pass
    print("\n[4/6] Testing forward pass...")
    sample_images, sample_tabular, sample_label = train_dataset[0]

    # Add batch dimension
    batch_images = {k: v.unsqueeze(0) for k, v in sample_images.items()}
    batch_tabular = sample_tabular.unsqueeze(0)

    model.eval()
    with torch.no_grad():
        outputs = model(batch_images, batch_tabular)

    print(f"✓ Forward pass successful")
    print(f"  - Logits shape: {outputs['logits'].shape}")
    print(f"  - Vision features shape: {outputs['vision_features'].shape}")
    print(f"  - Tabular features shape: {outputs['tabular_features'].shape}")
    print(f"  - Fused features shape: {outputs['fused_features'].shape}")

    if "view_aggregation_aux" in outputs:
        print(f"  - View aggregation info available: {list(outputs['view_aggregation_aux'].keys())}")

    # 5. Trainer
    print("\n[5/6] Creating trainer...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Using device: {device}")

    trainer = create_multiview_trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=device,
    )
    print(f"✓ Trainer created")

    # 6. Training (just 1 epoch for testing)
    print("\n[6/6] Running training test (1 epoch)...")
    try:
        # Override to just 1 epoch for quick test
        original_epochs = config.training.num_epochs
        config.training.num_epochs = 1

        trainer.train()

        print(f"✓ Training completed successfully")

        # Restore original epochs
        config.training.num_epochs = original_epochs

    except Exception as e:
        print(f"✗ Training failed: {e}")
        raise

    print("\n" + "="*60)
    print("✓ All tests passed!")
    print("="*60)

    return model, trainer


def test_single_view_backward_compatibility():
    """Test that single-view inputs still work (backward compatibility)."""

    print("\n" + "="*60)
    print("Testing Backward Compatibility (Single-View)")
    print("="*60)

    from med_core.backbones import ResNetBackbone, create_tabular_backbone
    from med_core.fusion import GatedFusion, MultiModalFusionModel

    # Create single-view model (old way)
    vision_backbone = ResNetBackbone(
        variant="resnet18",
        pretrained=False,
        feature_dim=128,
    )
    tabular_backbone = create_tabular_backbone(
        input_dim=10,
        output_dim=32,
    )
    fusion_module = GatedFusion(
        vision_dim=128,
        tabular_dim=32,
        output_dim=64,
    )

    model = MultiModalFusionModel(
        vision_backbone=vision_backbone,
        tabular_backbone=tabular_backbone,
        fusion_module=fusion_module,
        num_classes=2,
    )

    # Test with single-view input
    single_image = torch.randn(4, 3, 224, 224)
    tabular = torch.randn(4, 10)

    model.eval()
    with torch.no_grad():
        outputs = model(single_image, tabular)

    print(f"✓ Single-view forward pass successful")
    print(f"  - Logits shape: {outputs['logits'].shape}")

    print("\n✓ Backward compatibility maintained!")
    print("="*60)


if __name__ == "__main__":
    # Run integration test
    model, trainer = test_multiview_pipeline()

    # Test backward compatibility
    test_single_view_backward_compatibility()

    print("\n🎉 All integration tests passed!")
    print("\nMulti-view framework is ready to use!")
    print("\nNext steps:")
    print("  1. Prepare your multi-view dataset")
    print("  2. Create a config using create_ct_multiview_config() or create_temporal_multiview_config()")
    print("  3. Build your model using create_multiview_fusion_model()")
    print("  4. Train using create_multiview_trainer()")
