"""
Example demonstrating configuration validation usage.

This script shows how to use the config validation system to catch
configuration errors before training starts.
"""

from med_core.configs import (
    DataConfig,
    ExperimentConfig,
    FusionConfig,
    LoggingConfig,
    ModelConfig,
    TabularConfig,
    TrainingConfig,
    VisionConfig,
    validate_config,
)


def example_valid_config():
    """Example of a valid configuration."""
    print("=" * 60)
    print("Example 1: Valid Configuration")
    print("=" * 60)

    config = ExperimentConfig(
        project_name="lung_cancer_detection",
        experiment_name="resnet50_baseline",
        model=ModelConfig(
            num_classes=2,
            vision=VisionConfig(
                backbone="resnet50",
                feature_dim=256,
                dropout=0.3,
                attention_type="cbam",
            ),
            tabular=TabularConfig(
                hidden_dims=[128, 64],
                output_dim=32,
            ),
            fusion=FusionConfig(
                fusion_type="gated",
                hidden_dim=128,
            ),
        ),
        data=DataConfig(
            data_root="data/lung_cancer",
            csv_path="data/lung_cancer/metadata.csv",
            image_dir="data/lung_cancer/images",
            train_ratio=0.7,
            val_ratio=0.15,
            test_ratio=0.15,
            batch_size=32,
            image_size=224,
        ),
        training=TrainingConfig(
            num_epochs=100,
        ),
        logging=LoggingConfig(
            output_dir="outputs/lung_cancer",
            use_tensorboard=True,
            use_wandb=False,
        ),
    )

    errors = validate_config(config)
    if errors:
        print("❌ Validation failed:")
        for error in errors:
            print(f"  [{error.error_code}] {error.path}: {error.message}")
    else:
        print("✅ Configuration is valid!")
    print()


def example_invalid_backbone():
    """Example with invalid backbone."""
    print("=" * 60)
    print("Example 2: Invalid Backbone")
    print("=" * 60)

    config = ExperimentConfig(
        project_name="test",
        experiment_name="test",
        model=ModelConfig(
            num_classes=2,
            vision=VisionConfig(
                backbone="resnet999",  # Invalid!
                feature_dim=128,
            ),
            tabular=TabularConfig(hidden_dims=[64], output_dim=32),
            fusion=FusionConfig(fusion_type="concatenate", hidden_dim=96),
        ),
        data=DataConfig(
            data_root="data/",
            csv_path="data.csv",
            image_dir="images/",
            batch_size=32,
            image_size=224,
        ),
        training=TrainingConfig(num_epochs=50),
        logging=LoggingConfig(output_dir="outputs/"),
    )

    errors = validate_config(config)
    if errors:
        print("❌ Validation failed:")
        for error in errors:
            print(f"  [{error.error_code}] {error.path}")
            print(f"    ❌ {error.message}")
            if error.suggestion:
                print(f"    💡 {error.suggestion}")
    print()


def example_invalid_split_ratios():
    """Example with invalid split ratios."""
    print("=" * 60)
    print("Example 3: Invalid Split Ratios")
    print("=" * 60)

    config = ExperimentConfig(
        project_name="test",
        experiment_name="test",
        model=ModelConfig(
            num_classes=2,
            vision=VisionConfig(backbone="resnet18", feature_dim=128),
            tabular=TabularConfig(hidden_dims=[64], output_dim=32),
            fusion=FusionConfig(fusion_type="concatenate", hidden_dim=96),
        ),
        data=DataConfig(
            data_root="data/",
            csv_path="data.csv",
            image_dir="images/",
            train_ratio=0.6,
            val_ratio=0.3,
            test_ratio=0.3,  # Sum = 1.2, invalid!
            batch_size=32,
            image_size=224,
        ),
        training=TrainingConfig(num_epochs=50),
        logging=LoggingConfig(output_dir="outputs/"),
    )

    errors = validate_config(config)
    if errors:
        print("❌ Validation failed:")
        for error in errors:
            print(f"  [{error.error_code}] {error.path}")
            print(f"    ❌ {error.message}")
            if error.suggestion:
                print(f"    💡 {error.suggestion}")
    print()


def example_attention_supervision_without_cbam():
    """Example with attention supervision but no CBAM."""
    print("=" * 60)
    print("Example 4: Attention Supervision Without CBAM")
    print("=" * 60)

    config = ExperimentConfig(
        project_name="test",
        experiment_name="test",
        model=ModelConfig(
            num_classes=2,
            vision=VisionConfig(
                backbone="resnet18",
                feature_dim=128,
                attention_type="se",  # Not CBAM!
                enable_attention_supervision=False,
            ),
            tabular=TabularConfig(hidden_dims=[64], output_dim=32),
            fusion=FusionConfig(fusion_type="concatenate", hidden_dim=96),
        ),
        data=DataConfig(
            data_root="data/",
            csv_path="data.csv",
            image_dir="images/",
            batch_size=32,
            image_size=224,
        ),
        training=TrainingConfig(
            num_epochs=50,
            use_attention_supervision=True,  # Enabled!
        ),
        logging=LoggingConfig(output_dir="outputs/"),
    )

    errors = validate_config(config)
    if errors:
        print("❌ Validation failed:")
        for error in errors:
            print(f"  [{error.error_code}] {error.path}")
            print(f"    ❌ {error.message}")
            if error.suggestion:
                print(f"    💡 {error.suggestion}")
    print()


def example_progressive_training_mismatch():
    """Example with progressive training epoch mismatch."""
    print("=" * 60)
    print("Example 5: Progressive Training Epoch Mismatch")
    print("=" * 60)

    config = ExperimentConfig(
        project_name="test",
        experiment_name="test",
        model=ModelConfig(
            num_classes=2,
            vision=VisionConfig(backbone="resnet18", feature_dim=128),
            tabular=TabularConfig(hidden_dims=[64], output_dim=32),
            fusion=FusionConfig(fusion_type="concatenate", hidden_dim=96),
        ),
        data=DataConfig(
            data_root="data/",
            csv_path="data.csv",
            image_dir="images/",
            batch_size=32,
            image_size=224,
        ),
        training=TrainingConfig(
            num_epochs=100,
            use_progressive_training=True,
            stage1_epochs=20,
            stage2_epochs=30,
            stage3_epochs=20,  # Sum = 70, but num_epochs = 100!
        ),
        logging=LoggingConfig(output_dir="outputs/"),
    )

    errors = validate_config(config)
    if errors:
        print("❌ Validation failed:")
        for error in errors:
            print(f"  [{error.error_code}] {error.path}")
            print(f"    ❌ {error.message}")
            if error.suggestion:
                print(f"    💡 {error.suggestion}")
    print()


def main():
    """Run all examples."""
    print("\n" + "=" * 60)
    print("Configuration Validation Examples")
    print("=" * 60 + "\n")

    example_valid_config()
    example_invalid_backbone()
    example_invalid_split_ratios()
    example_attention_supervision_without_cbam()
    example_progressive_training_mismatch()

    print("=" * 60)
    print("Summary")
    print("=" * 60)
    print("✅ Use validate_config() to check configurations")
    print("✅ Use validate_config_or_exit() in CLI scripts")
    print("✅ All errors include error codes and helpful suggestions")
    print()


if __name__ == "__main__":
    main()
