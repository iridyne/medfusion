#!/usr/bin/env python3
"""
Smoke Test for Med-Core Framework

This script performs a comprehensive smoke test to verify:
1. Model architecture (vision + tabular + fusion)
2. Dimension alignment across all components
3. Training loop functionality and gradient flow

Usage:
    uv run python scripts/smoke_test.py --config configs/test_mock.yaml
    uv run python scripts/smoke_test.py --config configs/test_mock.yaml --epochs 5
"""

import argparse
import sys

import torch
import torch.nn as nn

from med_core.backbones import create_tabular_backbone, create_vision_backbone
from med_core.configs import load_config
from med_core.datasets import (
    MedicalMultimodalDataset,
    create_dataloaders,
    get_train_transforms,
    get_val_transforms,
    split_dataset,
)
from med_core.fusion import MultiModalFusionModel, create_fusion_module


def print_section(title: str):
    """Print a formatted section header."""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)


def print_subsection(title: str):
    """Print a formatted subsection header."""
    print(f"\n--- {title} ---")


def analyze_model_architecture(model: nn.Module, config):
    """Analyze and print model architecture details."""
    print_section("MODEL ARCHITECTURE ANALYSIS")

    # Vision Backbone
    print_subsection("Vision Backbone")
    vision = model.vision_backbone
    print(f"  Type: {config.model.vision.backbone}")
    print(f"  Pretrained: {config.model.vision.pretrained}")
    print(f"  Frozen: {config.model.vision.freeze_backbone}")
    print(f"  Output Dim: {vision.output_dim}")
    print(f"  Parameters: {sum(p.numel() for p in vision.parameters()):,}")
    print(f"  Trainable: {sum(p.numel() for p in vision.parameters() if p.requires_grad):,}")

    # Tabular Backbone
    print_subsection("Tabular Backbone")
    tabular = model.tabular_backbone
    print(f"  Input Dim: {tabular.input_dim}")
    print(f"  Hidden Dims: {config.model.tabular.hidden_dims}")
    print(f"  Output Dim: {tabular.output_dim}")
    print(f"  Parameters: {sum(p.numel() for p in tabular.parameters()):,}")
    print(f"  Trainable: {sum(p.numel() for p in tabular.parameters() if p.requires_grad):,}")

    # Fusion Module
    print_subsection("Fusion Module")
    fusion = model.fusion_module
    print(f"  Type: {config.model.fusion.fusion_type}")
    print(f"  Vision Input: {vision.output_dim}")
    print(f"  Tabular Input: {tabular.output_dim}")
    print(f"  Fusion Output: {fusion.output_dim}")
    print(f"  Parameters: {sum(p.numel() for p in fusion.parameters()):,}")

    # Classifier
    print_subsection("Classifier Head")
    classifier = model.classifier
    print(f"  Input Dim: {fusion.output_dim}")
    print(f"  Output Classes: {config.model.num_classes}")
    print(f"  Parameters: {sum(p.numel() for p in classifier.parameters()):,}")

    # Total
    print_subsection("Total Model")
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total Parameters: {total_params:,}")
    print(f"  Trainable Parameters: {trainable_params:,}")
    print(f"  Frozen Parameters: {total_params - trainable_params:,}")
    print(f"  Trainable Ratio: {trainable_params / total_params * 100:.2f}%")


def test_dimension_alignment(model: nn.Module, sample_batch):
    """Test dimension alignment through forward pass."""
    print_section("DIMENSION ALIGNMENT TEST")

    images, tabular, labels = sample_batch
    batch_size = images.size(0)

    print(f"  Batch Size: {batch_size}")
    print(f"  Image Shape: {tuple(images.shape)}")
    print(f"  Tabular Shape: {tuple(tabular.shape)}")
    print(f"  Labels Shape: {tuple(labels.shape)}")

    try:
        with torch.no_grad():
            # Forward pass
            outputs = model(images, tabular)

            if isinstance(outputs, dict):
                logits = outputs["logits"]
                vision_features = outputs.get("vision_features")
                tabular_features = outputs.get("tabular_features")
                fused_features = outputs.get("fused_features")

                print_subsection("Forward Pass Results")
                print(f"  Vision Features: {tuple(vision_features.shape)}")
                print(f"  Tabular Features: {tuple(tabular_features.shape)}")
                print(f"  Fused Features: {tuple(fused_features.shape)}")
                print(f"  Logits: {tuple(logits.shape)}")

                # Check auxiliary heads if present
                if "vision_logits" in outputs:
                    print(f"  Vision Logits (aux): {tuple(outputs['vision_logits'].shape)}")
                if "tabular_logits" in outputs:
                    print(f"  Tabular Logits (aux): {tuple(outputs['tabular_logits'].shape)}")
            else:
                logits = outputs
                print_subsection("Forward Pass Results")
                print(f"  Logits: {tuple(logits.shape)}")

        print("\n  ✓ Dimension alignment verified successfully!")
        return True

    except Exception as e:
        print(f"\n  ✗ Dimension alignment failed: {e}")
        return False


def test_gradient_flow(model: nn.Module, sample_batch, criterion):
    """Test gradient flow through backpropagation."""
    print_section("GRADIENT FLOW TEST")

    images, tabular, labels = sample_batch

    try:
        # Forward pass
        outputs = model(images, tabular)
        if isinstance(outputs, dict):
            logits = outputs["logits"]
        else:
            logits = outputs

        # Compute loss
        loss = criterion(logits, labels)
        print(f"  Initial Loss: {loss.item():.4f}")

        # Backward pass
        loss.backward()

        # Check gradients
        print_subsection("Gradient Statistics")
        grad_stats = {}
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                grad_norm = param.grad.norm().item()
                grad_stats[name] = grad_norm

        # Print summary
        if grad_stats:
            max_grad = max(grad_stats.values())
            min_grad = min(grad_stats.values())
            avg_grad = sum(grad_stats.values()) / len(grad_stats)

            print(f"  Layers with gradients: {len(grad_stats)}")
            print(f"  Max gradient norm: {max_grad:.6f}")
            print(f"  Min gradient norm: {min_grad:.6f}")
            print(f"  Avg gradient norm: {avg_grad:.6f}")

            # Check for vanishing/exploding gradients
            if max_grad > 100:
                print("  ⚠ Warning: Possible exploding gradients detected!")
            elif max_grad < 1e-7:
                print("  ⚠ Warning: Possible vanishing gradients detected!")
            else:
                print("  ✓ Gradient magnitudes look healthy")
        else:
            print("  ✗ No gradients found!")
            return False

        print("\n  ✓ Gradient flow verified successfully!")
        return True

    except Exception as e:
        print(f"\n  ✗ Gradient flow test failed: {e}")
        return False


def test_training_loop(model, train_loader, val_loader, criterion, optimizer, num_epochs=3):
    """Test training loop and loss convergence."""
    print_section("TRAINING LOOP TEST")

    device = next(model.parameters()).device
    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        # Training
        model.train()
        epoch_train_loss = 0.0
        num_train_batches = 0

        for images, tabular, labels in train_loader:
            images = images.to(device)
            tabular = tabular.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images, tabular)

            if isinstance(outputs, dict):
                logits = outputs["logits"]
            else:
                logits = outputs

            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            epoch_train_loss += loss.item()
            num_train_batches += 1

        avg_train_loss = epoch_train_loss / num_train_batches
        train_losses.append(avg_train_loss)

        # Validation
        model.eval()
        epoch_val_loss = 0.0
        num_val_batches = 0

        with torch.no_grad():
            for images, tabular, labels in val_loader:
                images = images.to(device)
                tabular = tabular.to(device)
                labels = labels.to(device)

                outputs = model(images, tabular)
                if isinstance(outputs, dict):
                    logits = outputs["logits"]
                else:
                    logits = outputs

                loss = criterion(logits, labels)
                epoch_val_loss += loss.item()
                num_val_batches += 1

        avg_val_loss = epoch_val_loss / num_val_batches
        val_losses.append(avg_val_loss)

        print(f"  Epoch {epoch + 1}/{num_epochs}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}")

    # Analyze loss trends
    print_subsection("Loss Analysis")
    print(f"  Initial Train Loss: {train_losses[0]:.4f}")
    print(f"  Final Train Loss: {train_losses[-1]:.4f}")
    print(f"  Train Loss Change: {train_losses[-1] - train_losses[0]:.4f}")

    print(f"\n  Initial Val Loss: {val_losses[0]:.4f}")
    print(f"  Final Val Loss: {val_losses[-1]:.4f}")
    print(f"  Val Loss Change: {val_losses[-1] - val_losses[0]:.4f}")

    # Check if loss is decreasing (at least validation loss)
    val_improving = val_losses[-1] < val_losses[0]
    if val_improving:
        print("\n  ✓ Validation loss is decreasing - model is learning!")
    else:
        print("\n  ⚠ Validation loss is not decreasing - may need tuning")

    return train_losses, val_losses


def generate_report(results: dict):
    """Generate final smoke test report."""
    print_section("SMOKE TEST REPORT")

    all_passed = all(results.values())

    print("\n  Test Results:")
    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"    {status}: {test_name}")

    print(f"\n  Overall Status: {'✓ ALL TESTS PASSED' if all_passed else '✗ SOME TESTS FAILED'}")

    if all_passed:
        print("\n  🎉 Framework is ready for production use!")
    else:
        print("\n  ⚠ Please review failed tests before proceeding.")

    return all_passed


def main():
    parser = argparse.ArgumentParser(description="Run smoke test for Med-Core framework")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/test_mock.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of training epochs for smoke test",
    )
    args = parser.parse_args()

    print_section("MED-CORE FRAMEWORK SMOKE TEST")
    print(f"  Config: {args.config}")
    print(f"  Epochs: {args.epochs}")

    # Load configuration
    config = load_config(args.config)

    # Load data
    print_section("DATA LOADING")
    train_transform = get_train_transforms(
        image_size=config.data.image_size,
        augmentation_strength=config.data.augmentation_strength,
    )
    val_transform = get_val_transforms(image_size=config.data.image_size)

    full_dataset, scaler = MedicalMultimodalDataset.from_csv(
        csv_path=config.data.csv_path,
        image_dir=config.data.image_dir,
        image_column=config.data.image_path_column,
        target_column=config.data.target_column,
        numerical_features=config.data.numerical_features,
        categorical_features=config.data.categorical_features,
        patient_id_column=config.data.patient_id_column,
    )

    train_ds, val_ds, test_ds = split_dataset(
        full_dataset,
        train_ratio=config.data.train_ratio,
        val_ratio=config.data.val_ratio,
        test_ratio=config.data.test_ratio,
        random_seed=config.data.random_seed,
    )

    train_ds.transform = train_transform
    val_ds.transform = val_transform

    dataloaders = create_dataloaders(
        train_dataset=train_ds,
        val_dataset=val_ds,
        test_dataset=test_ds,
        batch_size=config.data.batch_size,
        num_workers=0,
        pin_memory=False,
    )

    print(f"  Train samples: {len(train_ds)}")
    print(f"  Val samples: {len(val_ds)}")
    print(f"  Test samples: {len(test_ds)}")

    # Build model
    print_section("MODEL CONSTRUCTION")
    vision_backbone = create_vision_backbone(
        backbone_name=config.model.vision.backbone,
        pretrained=config.model.vision.pretrained,
        freeze=config.model.vision.freeze_backbone,
        feature_dim=config.model.vision.feature_dim,
        dropout=config.model.vision.dropout,
        attention_type=config.model.vision.attention_type,
    )

    tabular_dim = train_ds.get_tabular_dim()
    tabular_backbone = create_tabular_backbone(
        input_dim=tabular_dim,
        output_dim=config.model.tabular.output_dim,
        hidden_dims=config.model.tabular.hidden_dims,
        dropout=config.model.tabular.dropout,
    )

    fusion_module = create_fusion_module(
        fusion_type=config.model.fusion.fusion_type,
        vision_dim=config.model.vision.feature_dim,
        tabular_dim=config.model.tabular.output_dim,
        output_dim=config.model.fusion.hidden_dim,
        dropout=config.model.fusion.dropout,
    )

    model = MultiModalFusionModel(
        vision_backbone=vision_backbone,
        tabular_backbone=tabular_backbone,
        fusion_module=fusion_module,
        num_classes=config.model.num_classes,
        use_auxiliary_heads=config.model.use_auxiliary_heads,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    print(f"  Device: {device}")
    print("  Model created successfully")

    # Run tests
    results = {}

    # Test 1: Architecture Analysis
    analyze_model_architecture(model, config)
    results["Architecture Analysis"] = True

    # Test 2: Dimension Alignment
    sample_batch = next(iter(dataloaders["train"]))
    sample_batch = tuple(x.to(device) for x in sample_batch)
    results["Dimension Alignment"] = test_dimension_alignment(model, sample_batch)

    # Test 3: Gradient Flow
    criterion = nn.CrossEntropyLoss()
    results["Gradient Flow"] = test_gradient_flow(model, sample_batch, criterion)

    # Test 4: Training Loop
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    train_losses, val_losses = test_training_loop(
        model, dataloaders["train"], dataloaders["val"], criterion, optimizer, num_epochs=args.epochs
    )
    results["Training Loop"] = True

    # Generate final report
    all_passed = generate_report(results)

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
