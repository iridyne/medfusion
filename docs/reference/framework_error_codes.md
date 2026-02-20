# Med-Core Framework Error Codes Reference

**Version:** 2.0.0  
**Last Updated:** 2026-02-18  
**Purpose:** Comprehensive error code reference for Med-Core framework exceptions

---

## Overview

The Med-Core framework uses a structured exception system with:

- **Error Codes**: Unique identifiers (E000-E1000+)
- **Context Information**: Relevant details for debugging
- **Helpful Suggestions**: Actionable guidance for resolution
- **Formatted Output**: User-friendly error reports

All exceptions inherit from `MedCoreError` and include these features automatically.

---

## Error Code Structure

Format: `E[CATEGORY][NUMBER]`

- **E000**: Base/Generic errors
- **E1xx**: Configuration errors
- **E2xx**: Dataset errors
- **E3xx**: Model errors
- **E4xx**: Training errors
- **E5xx**: Preprocessing errors
- **E6xx**: Evaluation errors
- **E7xx**: Input validation errors
- **E8xx**: Dependency errors
- **E9xx**: Attention supervision errors
- **E10xx**: Multi-view errors

---

## Error Codes by Category

### Base Errors (E000)

#### E000: MedCoreError
**Class:** `MedCoreError`  
**Description:** Base exception for all Med-Core errors  
**Usage:** Generic errors or as base class for custom exceptions

```python
raise MedCoreError(
    "Something went wrong",
    error_code="E000",
    context={"key": "value"},
    suggestion="Try this instead"
)
```

---

### Configuration Errors (E100-E199)

#### E100: ConfigurationError
**Class:** `ConfigurationError`  
**Description:** General configuration error  
**Common Causes:**
- Invalid configuration values
- Missing required configuration
- Type mismatches

```python
raise ConfigurationError(
    "Invalid batch size",
    config_path="training.batch_size",
    invalid_value=-1,
    suggestion="Batch size must be positive (e.g., 16, 32, 64)"
)
```

#### E101: IncompatibleConfigError
**Class:** `IncompatibleConfigError`  
**Description:** Configuration options are incompatible  
**Common Causes:**
- Attention supervision without CBAM
- Conflicting feature flags
- Incompatible model settings

```python
raise IncompatibleConfigError(
    "Attention supervision requires CBAM",
    conflicting_options=["use_attention_supervision=True", "attention_type='se'"],
    suggestion="Set attention_type='cbam'"
)
```

---

### Dataset Errors (E200-E299)

#### E200: DatasetError
**Class:** `DatasetError`  
**Description:** General dataset error  
**Common Causes:**
- Data loading failures
- Corrupted data files
- Invalid data format

```python
raise DatasetError(
    "Failed to load dataset",
    dataset_path="/path/to/data",
    suggestion="Check file format and permissions"
)
```

#### E201: DatasetNotFoundError
**Class:** `DatasetNotFoundError`  
**Description:** Dataset file or directory not found  
**Common Causes:**
- Incorrect path
- Missing files
- Permission issues

```python
raise DatasetNotFoundError("/path/to/dataset.csv")
```

#### E202: MissingColumnError
**Class:** `MissingColumnError`  
**Description:** Required column missing from dataset  
**Common Causes:**
- CSV missing expected columns
- Renamed columns
- Wrong dataset file

```python
raise MissingColumnError(
    "patient_id",
    available_columns=["id", "age", "gender"]
)
```

---

### Model Errors (E300-E399)

#### E300: ModelError
**Class:** `ModelError`  
**Description:** General model error  
**Common Causes:**
- Model construction failures
- Invalid model configuration
- Architecture mismatches

```python
raise ModelError(
    "Failed to build model",
    model_name="MultimodalClassifier",
    suggestion="Check model configuration"
)
```

#### E310: BackboneError
**Class:** `BackboneError`  
**Description:** Backbone-related error  
**Common Causes:**
- Invalid backbone name
- Backbone loading failure
- Incompatible backbone version

```python
raise BackboneError(
    "Failed to load backbone",
    backbone_name="resnet50",
    suggestion="Check backbone name and pretrained weights"
)
```

#### E311: BackboneNotFoundError
**Class:** `BackboneNotFoundError`  
**Description:** Requested backbone not available  
**Common Causes:**
- Typo in backbone name
- Unsupported backbone
- Missing backbone implementation

```python
raise BackboneNotFoundError(
    "resnet999",
    available_backbones=["resnet18", "resnet50", "efficientnet_b0"]
)
```

#### E320: FusionError
**Class:** `FusionError`  
**Description:** Fusion module error  
**Common Causes:**
- Invalid fusion type
- Dimension mismatches
- Fusion configuration issues

```python
raise FusionError(
    "Fusion failed",
    fusion_type="gated",
    suggestion="Check input dimensions"
)
```

#### E321: FusionNotFoundError
**Class:** `FusionNotFoundError`  
**Description:** Requested fusion strategy not available  
**Common Causes:**
- Invalid fusion type name
- Unsupported fusion method
- Missing fusion implementation

```python
raise FusionNotFoundError(
    "invalid_fusion",
    available_fusions=["concatenate", "gated", "attention"]
)
```

---

### Training Errors (E400-E499)

#### E400: TrainingError
**Class:** `TrainingError`  
**Description:** General training error  
**Common Causes:**
- Loss becomes NaN
- Gradient explosion
- Training instability

```python
raise TrainingError(
    "Loss became NaN",
    epoch=10,
    step=500,
    suggestion="Reduce learning rate or enable gradient clipping"
)
```

#### E410: CheckpointError
**Class:** `CheckpointError`  
**Description:** Checkpoint loading/saving error  
**Common Causes:**
- Corrupted checkpoint
- Incompatible checkpoint format
- Disk space issues

```python
raise CheckpointError(
    "Failed to save checkpoint",
    checkpoint_path="outputs/model.pth",
    suggestion="Check disk space and permissions"
)
```

#### E411: CheckpointNotFoundError
**Class:** `CheckpointNotFoundError`  
**Description:** Checkpoint file not found  
**Common Causes:**
- Incorrect checkpoint path
- Checkpoint not created yet
- File deleted

```python
raise CheckpointNotFoundError("outputs/best_model.pth")
```

---

### Preprocessing Errors (E500-E599)

#### E500: PreprocessingError
**Class:** `PreprocessingError`  
**Description:** Image preprocessing error  
**Common Causes:**
- Invalid image format
- Corrupted image file
- Unsupported image type

```python
raise PreprocessingError(
    "Failed to load image",
    image_path="/path/to/image.jpg",
    suggestion="Check image format and file integrity"
)
```

---

### Evaluation Errors (E600-E699)

#### E600: EvaluationError
**Class:** `EvaluationError`  
**Description:** Model evaluation error  
**Common Causes:**
- Invalid metric configuration
- Missing ground truth
- Incompatible predictions

```python
raise EvaluationError(
    "Failed to compute metric",
    metric_name="auc",
    suggestion="Check that predictions and labels are compatible"
)
```

---

### Input Validation Errors (E700-E799)

#### E700: InvalidInputError
**Class:** `InvalidInputError`  
**Description:** Invalid input data  
**Common Causes:**
- Wrong data type
- Invalid value range
- Malformed input

```python
raise InvalidInputError(
    "Invalid input type",
    input_name="learning_rate",
    expected_type="float",
    actual_type="str",
    suggestion="Provide a float value (e.g., 0.001)"
)
```

#### E701: DimensionMismatchError
**Class:** `DimensionMismatchError`  
**Description:** Tensor dimensions don't match  
**Common Causes:**
- Wrong input shape
- Batch size mismatch
- Channel mismatch

```python
raise DimensionMismatchError(
    expected=(32, 3, 224, 224),
    actual=(32, 3, 256, 256),
    tensor_name="input_images",
    suggestion="Resize images to 224x224"
)
```

---

### Dependency Errors (E800-E899)

#### E800: MissingDependencyError
**Class:** `MissingDependencyError`  
**Description:** Required package not installed  
**Common Causes:**
- Missing optional dependency
- Incomplete installation
- Version incompatibility

```python
raise MissingDependencyError(
    "torch",
    install_cmd="pip install torch torchvision"
)
```

---

### Attention Supervision Errors (E900-E999)

#### E900: AttentionSupervisionError
**Class:** `AttentionSupervisionError`  
**Description:** Attention supervision error  
**Common Causes:**
- Incompatible attention type
- Missing attention masks
- Invalid supervision method

```python
raise AttentionSupervisionError(
    "Attention supervision requires CBAM",
    attention_type="se",
    suggestion="Use CBAM attention for supervision"
)
```

---

### Multi-View Errors (E1000-E1099)

#### E1000: MultiViewError
**Class:** `MultiViewError`  
**Description:** Multi-view processing error  
**Common Causes:**
- Missing views
- Inconsistent view dimensions
- Invalid view configuration

```python
raise MultiViewError(
    "Missing required view",
    view_name="coronal",
    num_views=2,
    suggestion="Provide all required views"
)
```

---

## Usage Examples

### Basic Exception Handling

```python
from med_core.exceptions import BackboneNotFoundError, format_error_report

try:
    backbone = load_backbone("invalid_name")
except BackboneNotFoundError as e:
    print(format_error_report(e))
    # Output:
    # ❌ Error [E311]: Backbone 'invalid_name' not found
    # 
    # 📋 Context:
    #   • model_name: invalid_name
    #   • available_backbones: ['resnet18', 'resnet50', ...]
    # 
    # 💡 Suggestion: Available backbones: resnet18, resnet50...
```

### Custom Error with Context

```python
from med_core.exceptions import TrainingError

raise TrainingError(
    "Gradient explosion detected",
    epoch=5,
    step=1000,
    suggestion="Enable gradient clipping with max_norm=1.0"
)
```

### Catching Multiple Exception Types

```python
from med_core.exceptions import (
    DatasetError,
    ModelError,
    TrainingError,
    format_error_report
)

try:
    train_model(config)
except (DatasetError, ModelError, TrainingError) as e:
    logger.error(format_error_report(e))
    # All Med-Core exceptions have error_code, context, suggestion
    if e.error_code.startswith("E2"):
        # Handle dataset errors
        pass
    elif e.error_code.startswith("E3"):
        # Handle model errors
        pass
```

---

## Error Code Quick Reference

| Code | Exception | Description |
|------|-----------|-------------|
| E000 | MedCoreError | Base exception |
| E100 | ConfigurationError | Configuration error |
| E101 | IncompatibleConfigError | Incompatible config options |
| E200 | DatasetError | Dataset error |
| E201 | DatasetNotFoundError | Dataset not found |
| E202 | MissingColumnError | Missing column |
| E300 | ModelError | Model error |
| E310 | BackboneError | Backbone error |
| E311 | BackboneNotFoundError | Backbone not found |
| E320 | FusionError | Fusion error |
| E321 | FusionNotFoundError | Fusion not found |
| E400 | TrainingError | Training error |
| E410 | CheckpointError | Checkpoint error |
| E411 | CheckpointNotFoundError | Checkpoint not found |
| E500 | PreprocessingError | Preprocessing error |
| E600 | EvaluationError | Evaluation error |
| E700 | InvalidInputError | Invalid input |
| E701 | DimensionMismatchError | Dimension mismatch |
| E800 | MissingDependencyError | Missing dependency |
| E900 | AttentionSupervisionError | Attention supervision error |
| E1000 | MultiViewError | Multi-view error |

---

## Best Practices

### 1. Always Include Context

```python
# ❌ Bad
raise ModelError("Model failed")

# ✅ Good
raise ModelError(
    "Model failed to initialize",
    model_name="ResNet50",
    suggestion="Check that pretrained weights are available"
)
```

### 2. Provide Actionable Suggestions

```python
# ❌ Bad
raise DatasetNotFoundError("/path/to/data")

# ✅ Good (already done in DatasetNotFoundError)
# Automatically includes: "Check that the path exists and is accessible"
```

### 3. Use Specific Exception Types

```python
# ❌ Bad
raise MedCoreError("Backbone not found")

# ✅ Good
raise BackboneNotFoundError("resnet999", available_backbones)
```

### 4. Format Errors for Users

```python
from med_core.exceptions import format_error_report

try:
    process_data()
except MedCoreError as e:
    # Use format_error_report for user-friendly output
    print(format_error_report(e))
    # Log the full exception for debugging
    logger.exception("Error during processing")
```

---

## Adding New Error Codes

When adding new exceptions:

1. **Choose appropriate error code range** based on category
2. **Inherit from appropriate base class**
3. **Include helpful context and suggestions**
4. **Add tests** in `tests/test_exceptions.py`
5. **Update this documentation**

Example:

```python
class CustomError(MedCoreError):
    """Custom error for specific use case."""
    
    def __init__(
        self,
        message: str,
        custom_field: str | None = None,
        suggestion: str | None = None,
    ):
        context = {}
        if custom_field:
            context["custom_field"] = custom_field
        
        super().__init__(
            message=message,
            error_code="E1100",  # Choose unused code
            context=context,
            suggestion=suggestion,
        )
```

---

## See Also

- [Configuration Validation](../guides/configuration.md)
- [Error Handling Examples](../../examples/exception_handling_demo.py)
- [Medical Error Codes](error_codes.md) - Medical-specific error codes

---

**Document Maintainer:** Med-Core Engineering Team  
**Review Cycle:** Monthly  
**Last Review:** 2026-02-18
