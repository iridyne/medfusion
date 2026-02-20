"""
Example demonstrating enhanced exception system usage.

This script shows how to use the improved exception classes with
error codes, context, and helpful suggestions.
"""

from med_core.exceptions import (
    BackboneNotFoundError,
    CheckpointNotFoundError,
    ConfigurationError,
    DatasetNotFoundError,
    DimensionMismatchError,
    IncompatibleConfigError,
    MissingColumnError,
    MissingDependencyError,
    TrainingError,
    format_error_report,
)


def example_backbone_not_found():
    """Example: Backbone not found error."""
    print("=" * 60)
    print("Example 1: Backbone Not Found")
    print("=" * 60)
    
    try:
        available_backbones = [
            "resnet18", "resnet50", "efficientnet_b0", 
            "vit_b_16", "swin_t"
        ]
        raise BackboneNotFoundError("resnet999", available_backbones)
    except BackboneNotFoundError as e:
        print(format_error_report(e))


def example_dimension_mismatch():
    """Example: Dimension mismatch error."""
    print("=" * 60)
    print("Example 2: Dimension Mismatch")
    print("=" * 60)
    
    try:
        raise DimensionMismatchError(
            expected=(32, 3, 224, 224),
            actual=(32, 3, 256, 256),
            tensor_name="input_images",
            suggestion="Resize images to 224x224 or update model input size"
        )
    except DimensionMismatchError as e:
        print(format_error_report(e))


def example_missing_column():
    """Example: Missing column error."""
    print("=" * 60)
    print("Example 3: Missing Column in Dataset")
    print("=" * 60)
    
    try:
        available_columns = [
            "patient_id", "age", "gender", "image_path", "diagnosis"
        ]
        raise MissingColumnError("bmi", available_columns)
    except MissingColumnError as e:
        print(format_error_report(e))


def example_training_error():
    """Example: Training error with context."""
    print("=" * 60)
    print("Example 4: Training Error")
    print("=" * 60)
    
    try:
        raise TrainingError(
            "Loss became NaN",
            epoch=15,
            step=2500,
            suggestion="Reduce learning rate or enable gradient clipping"
        )
    except TrainingError as e:
        print(format_error_report(e))


def example_incompatible_config():
    """Example: Incompatible configuration."""
    print("=" * 60)
    print("Example 5: Incompatible Configuration")
    print("=" * 60)
    
    try:
        raise IncompatibleConfigError(
            "Attention supervision requires CBAM attention mechanism",
            conflicting_options=[
                "training.use_attention_supervision=True",
                "model.vision.attention_type='se'"
            ],
            suggestion="Set model.vision.attention_type='cbam'"
        )
    except IncompatibleConfigError as e:
        print(format_error_report(e))


def example_missing_dependency():
    """Example: Missing dependency error."""
    print("=" * 60)
    print("Example 6: Missing Dependency")
    print("=" * 60)
    
    try:
        raise MissingDependencyError(
            "torch",
            install_cmd="pip install torch torchvision"
        )
    except MissingDependencyError as e:
        print(format_error_report(e))


def example_dataset_not_found():
    """Example: Dataset not found error."""
    print("=" * 60)
    print("Example 7: Dataset Not Found")
    print("=" * 60)
    
    try:
        raise DatasetNotFoundError("/data/medical/lung_cancer/metadata.csv")
    except DatasetNotFoundError as e:
        print(format_error_report(e))


def example_checkpoint_not_found():
    """Example: Checkpoint not found error."""
    print("=" * 60)
    print("Example 8: Checkpoint Not Found")
    print("=" * 60)
    
    try:
        raise CheckpointNotFoundError("outputs/best_model.pth")
    except CheckpointNotFoundError as e:
        print(format_error_report(e))


def example_configuration_error():
    """Example: Configuration error."""
    print("=" * 60)
    print("Example 9: Configuration Error")
    print("=" * 60)
    
    try:
        raise ConfigurationError(
            "Invalid batch size",
            config_path="training.batch_size",
            invalid_value=-1,
            suggestion="Batch size must be a positive integer (e.g., 16, 32, 64)"
        )
    except ConfigurationError as e:
        print(format_error_report(e))


def example_error_handling_in_function():
    """Example: Using exceptions in a function."""
    print("=" * 60)
    print("Example 10: Error Handling in Function")
    print("=" * 60)
    
    def load_backbone(name: str):
        """Load a backbone model."""
        available = ["resnet18", "resnet50", "efficientnet_b0"]
        
        if name not in available:
            raise BackboneNotFoundError(name, available)
        
        return f"Loaded {name}"
    
    # Try with valid backbone
    try:
        result = load_backbone("resnet50")
        print(f"✅ Success: {result}\n")
    except BackboneNotFoundError as e:
        print(format_error_report(e))
    
    # Try with invalid backbone
    try:
        result = load_backbone("invalid_model")
        print(f"✅ Success: {result}\n")
    except BackboneNotFoundError as e:
        print(format_error_report(e))


def main():
    """Run all examples."""
    print("\n" + "=" * 60)
    print("Enhanced Exception System Examples")
    print("=" * 60 + "\n")
    
    example_backbone_not_found()
    example_dimension_mismatch()
    example_missing_column()
    example_training_error()
    example_incompatible_config()
    example_missing_dependency()
    example_dataset_not_found()
    example_checkpoint_not_found()
    example_configuration_error()
    example_error_handling_in_function()
    
    print("=" * 60)
    print("Summary")
    print("=" * 60)
    print("✅ All exceptions include error codes (E100-E1000)")
    print("✅ Context information helps with debugging")
    print("✅ Suggestions guide users to solutions")
    print("✅ Use format_error_report() for user-friendly output")
    print()


if __name__ == "__main__":
    main()
