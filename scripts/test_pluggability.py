#!/usr/bin/env python3
"""
Pluggability Validation Script for Med-Core Framework

This script systematically tests the plug-and-play capability of the framework
by switching between different components without modifying code.

Tests:
1. Vision Backbones: ResNet18, MobileNetV2, EfficientNet-B0
2. Fusion Strategies: Concatenate, Gated, Attention
3. Tabular Backbones: Different hidden layer configurations

Usage:
    uv run python scripts/test_pluggability.py
    uv run python scripts/test_pluggability.py --quick
"""

import argparse
import os
import sys
import tempfile
from pathlib import Path

# When this module is imported by pytest for collection, avoid exposing script-style
# test functions (which start with `test_`) to pytest's collector. Instead skip the
# entire module during collection. When executed as a script (`__name__ == "__main__"`),
# the script runs normally.
if "PYTEST_CURRENT_TEST" in os.environ or "pytest" in sys.modules:
    try:
        import pytest  # type: ignore
        pytest.skip("Script-style tests are skipped during pytest collection", allow_module_level=True)
    except Exception:
        # If pytest cannot be imported for some reason, simply exit early to avoid collection.
        # This branch is conservative: in normal script execution pytest won't be in sys.modules
        # and this block will not be executed.
        raise SystemExit(0) from None

import torch
import yaml

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


def print_header(title: str):
    """Print formatted header."""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)


def print_result(component: str, value: str, status: str):
    """Print test result."""
    symbol = "✓" if status == "PASS" else "✗"
    print(f"  {symbol} {component:20s} → {value:30s} [{status}]")


def test_component_combination(
    config_dict: dict,
    vision_backbone: str,
    fusion_type: str,
    tabular_hidden: list,
    dataloaders: dict,
) -> dict:
    """Test a specific combination of components."""

    # Update config
    config_dict["model"]["vision"]["backbone"] = vision_backbone
    config_dict["model"]["fusion"]["fusion_type"] = fusion_type
    config_dict["model"]["tabular"]["hidden_dims"] = tabular_hidden

    # Save temporary config
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(config_dict, f)
        temp_config_path = f.name

    try:
        # Load config
        config = load_config(temp_config_path)

        # Get sample batch
        sample_batch = next(iter(dataloaders["train"]))
        images, tabular, _ = sample_batch

        # Build model
        vision = create_vision_backbone(
            backbone_name=config.model.vision.backbone,
            pretrained=False,  # Faster for testing
            freeze=False,
            feature_dim=config.model.vision.feature_dim,
            dropout=config.model.vision.dropout,
            attention_type=config.model.vision.attention_type,
        )

        tabular_dim = tabular.shape[1]
        tabular_net = create_tabular_backbone(
            input_dim=tabular_dim,
            output_dim=config.model.tabular.output_dim,
            hidden_dims=config.model.tabular.hidden_dims,
            dropout=config.model.tabular.dropout,
        )

        fusion = create_fusion_module(
            fusion_type=config.model.fusion.fusion_type,
            vision_dim=config.model.vision.feature_dim,
            tabular_dim=config.model.tabular.output_dim,
            output_dim=config.model.fusion.hidden_dim,
            dropout=config.model.fusion.dropout,
        )

        model = MultiModalFusionModel(
            vision_backbone=vision,
            tabular_backbone=tabular_net,
            fusion_module=fusion,
            num_classes=config.model.num_classes,
            use_auxiliary_heads=False,
        )

        # Test forward pass
        with torch.no_grad():
            outputs = model(images, tabular)
            if isinstance(outputs, dict):
                logits = outputs["logits"]
            else:
                logits = outputs

        # Verify output shape
        expected_shape = (images.size(0), config.model.num_classes)
        actual_shape = tuple(logits.shape)

        if actual_shape == expected_shape:
            status = "PASS"
            error = None
        else:
            status = "FAIL"
            error = f"Shape mismatch: expected {expected_shape}, got {actual_shape}"

        # Get model info
        total_params = sum(p.numel() for p in model.parameters())

        result = {
            "status": status,
            "error": error,
            "params": total_params,
            "vision_params": sum(p.numel() for p in vision.parameters()),
            "tabular_params": sum(p.numel() for p in tabular_net.parameters()),
            "fusion_params": sum(p.numel() for p in fusion.parameters()),
        }

    except Exception as e:
        result = {
            "status": "FAIL",
            "error": str(e),
            "params": 0,
            "vision_params": 0,
            "tabular_params": 0,
            "fusion_params": 0,
        }

    finally:
        # Clean up temp file
        Path(temp_config_path).unlink(missing_ok=True)

    return result


def main():
    parser = argparse.ArgumentParser(description="Test framework pluggability")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/test_mock.yaml",
        help="Base configuration file",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run quick test with fewer combinations",
    )
    args = parser.parse_args()

    print_header("MED-CORE PLUGGABILITY VALIDATION")

    # Load base config
    base_config = load_config(args.config)
    config_dict = base_config.to_dict()

    # Load data once
    print("\nLoading data...")
    train_transform = get_train_transforms(
        image_size=base_config.data.image_size,
        augmentation_strength="light",
    )
    val_transform = get_val_transforms(image_size=base_config.data.image_size)

    full_dataset, _ = MedicalMultimodalDataset.from_csv(
        csv_path=base_config.data.csv_path,
        image_dir=base_config.data.image_dir,
        image_column=base_config.data.image_path_column,
        target_column=base_config.data.target_column,
        numerical_features=base_config.data.numerical_features,
        categorical_features=base_config.data.categorical_features,
        patient_id_column=base_config.data.patient_id_column,
    )

    train_ds, val_ds, test_ds = split_dataset(
        full_dataset,
        train_ratio=base_config.data.train_ratio,
        val_ratio=base_config.data.val_ratio,
        test_ratio=base_config.data.test_ratio,
        random_seed=base_config.data.random_seed,
    )

    train_ds.transform = train_transform
    val_ds.transform = val_transform

    dataloaders = create_dataloaders(
        train_dataset=train_ds,
        val_dataset=val_ds,
        test_dataset=test_ds,
        batch_size=2,
        num_workers=0,
        pin_memory=False,
    )

    print(f"  Loaded {len(train_ds)} training samples")

    # Define test combinations
    if args.quick:
        vision_backbones = ["resnet18", "mobilenetv2"]
        fusion_types = ["concatenate", "gated"]
        tabular_configs = [[32], [64, 32]]
    else:
        vision_backbones = ["resnet18", "mobilenetv2", "efficientnet_b0"]
        fusion_types = ["concatenate", "gated", "attention"]
        tabular_configs = [[32], [64, 32], [128, 64, 32]]

    # Run tests
    print_header("TESTING COMPONENT COMBINATIONS")

    results = []
    total_tests = len(vision_backbones) * len(fusion_types) * len(tabular_configs)
    current_test = 0

    for vision in vision_backbones:
        for fusion in fusion_types:
            for tabular in tabular_configs:
                current_test += 1

                print(f"\n[{current_test}/{total_tests}] Testing combination:")
                print(f"  Vision: {vision}")
                print(f"  Fusion: {fusion}")
                print(f"  Tabular: {tabular}")

                result = test_component_combination(
                    config_dict.copy(),
                    vision,
                    fusion,
                    tabular,
                    dataloaders,
                )

                result["vision"] = vision
                result["fusion"] = fusion
                result["tabular"] = str(tabular)
                results.append(result)

                if result["status"] == "PASS":
                    print(f"  ✓ PASS - {result['params']:,} parameters")
                else:
                    print(f"  ✗ FAIL - {result['error']}")

    # Generate report
    print_header("PLUGGABILITY TEST REPORT")

    passed = sum(1 for r in results if r["status"] == "PASS")
    failed = sum(1 for r in results if r["status"] == "FAIL")

    print(f"\n  Total Tests: {total_tests}")
    print(f"  Passed: {passed} ({passed/total_tests*100:.1f}%)")
    print(f"  Failed: {failed} ({failed/total_tests*100:.1f}%)")

    if failed > 0:
        print("\n  Failed Combinations:")
        for r in results:
            if r["status"] == "FAIL":
                print(f"    ✗ {r['vision']} + {r['fusion']} + {r['tabular']}")
                print(f"      Error: {r['error']}")

    # Component-specific analysis
    print_header("COMPONENT ANALYSIS")

    print("\n  Vision Backbones:")
    for vision in vision_backbones:
        vision_results = [r for r in results if r["vision"] == vision]
        vision_passed = sum(1 for r in vision_results if r["status"] == "PASS")
        status = "PASS" if vision_passed == len(vision_results) else "PARTIAL" if vision_passed > 0 else "FAIL"
        print_result("Vision", vision, status)

    print("\n  Fusion Strategies:")
    for fusion in fusion_types:
        fusion_results = [r for r in results if r["fusion"] == fusion]
        fusion_passed = sum(1 for r in fusion_results if r["status"] == "PASS")
        status = "PASS" if fusion_passed == len(fusion_results) else "PARTIAL" if fusion_passed > 0 else "FAIL"
        print_result("Fusion", fusion, status)

    print("\n  Tabular Configurations:")
    for tabular in tabular_configs:
        tabular_str = str(tabular)
        tabular_results = [r for r in results if r["tabular"] == tabular_str]
        tabular_passed = sum(1 for r in tabular_results if r["status"] == "PASS")
        status = "PASS" if tabular_passed == len(tabular_results) else "PARTIAL" if tabular_passed > 0 else "FAIL"
        print_result("Tabular", tabular_str, status)

    # Final verdict
    print_header("FINAL VERDICT")

    if failed == 0:
        print("\n  🎉 ALL TESTS PASSED!")
        print("  Framework demonstrates excellent plug-and-play capability.")
        print("  All component combinations work without code modification.")
        return 0
    elif passed > failed:
        print("\n  ⚠ MOSTLY PASSED")
        print(f"  {passed}/{total_tests} combinations work correctly.")
        print("  Some component combinations need attention.")
        return 1
    else:
        print("\n  ✗ TESTS FAILED")
        print("  Framework pluggability needs improvement.")
        return 2


if __name__ == "__main__":
    sys.exit(main())
