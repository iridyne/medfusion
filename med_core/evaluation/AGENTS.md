# Evaluation Module

<!-- Parent: ../AGENTS.md -->

**Last Updated:** 2025-02-27

## Purpose

Comprehensive evaluation tools for medical multimodal models, including metrics calculation, visualization, interpretability, and report generation.

## Key Components

### Metrics (`metrics.py`, `metrics_calculator.py`)
- **calculate_binary_metrics()**: Calculate accuracy, AUC, F1, sensitivity, specificity
- **MetricsCalculator**: Comprehensive metrics calculation for classification and survival tasks

### Visualization (`visualization.py`, `report_visualizer.py`)
- **plot_confusion_matrix()**: Generate confusion matrix plots
- **plot_roc_curve()**: Generate ROC curves with AUC
- **ReportVisualizer**: Comprehensive visualization suite

### Interpretability (`interpretability.py`)
- **GradCAM**: Gradient-weighted Class Activation Mapping
- **visualize_gradcam()**: Generate Grad-CAM heatmaps
- **visualize_attention_weights()**: Visualize attention maps

### Report Generation

1. **Basic Reports** (`report.py`, `report_generator.py`)
   - **EvaluationReport**: Legacy report class
   - **ReportGenerator**: Generate evaluation reports
   - **generate_evaluation_report()**: Factory function

2. **Enhanced Reports** (`report_generator_enhanced.py`)
   - **EnhancedReportGenerator**: Publication-ready reports
   - **generate_enhanced_report()**: Generate comprehensive reports
   - Includes statistical tests, confidence intervals, subgroup analysis

## Architecture

```
Model Predictions → Metrics Calculation → Visualization → Report Generation
```

## Usage Patterns

### Calculate Metrics
```python
from med_core.evaluation import calculate_binary_metrics

metrics = calculate_binary_metrics(
    y_true=labels,
    y_pred=predictions,
    y_score=probabilities
)
print(f"Accuracy: {metrics['accuracy']:.4f}")
print(f"AUC: {metrics['auc']:.4f}")
print(f"F1: {metrics['f1']:.4f}")
```

### Generate Grad-CAM
```python
from med_core.evaluation import GradCAM, visualize_gradcam

gradcam = GradCAM(model, target_layer='layer4')
cam = gradcam(images, class_idx=1)
visualize_gradcam(images[0], cam[0], save_path='outputs/gradcam.png')
```

### Generate Evaluation Report
```python
from med_core.evaluation import generate_enhanced_report

report = generate_enhanced_report(
    model=model,
    dataloader=test_loader,
    output_dir='outputs/evaluation',
    device='cuda'
)
```

### Visualize ROC Curve
```python
from med_core.evaluation import plot_roc_curve

plot_roc_curve(
    y_true=labels,
    y_score=probabilities,
    save_path='outputs/roc_curve.png'
)
```

## Key Files

- `metrics.py`: Basic metrics calculation
- `metrics_calculator.py`: Advanced metrics calculator
- `visualization.py`: Basic visualization functions
- `report_visualizer.py`: Comprehensive visualization suite
- `interpretability.py`: Grad-CAM and attention visualization
- `report.py`: Legacy report generation
- `report_generator.py`: Standard report generation
- `report_generator_enhanced.py`: Enhanced report generation

## Dependencies

- PyTorch (model inference)
- scikit-learn (metrics calculation)
- matplotlib, seaborn (visualization)
- numpy, pandas (data processing)
- Used by: `med_core.cli`, `med_core.trainers`

## Testing

Run tests with:
```bash
uv run pytest tests/test_evaluation.py -v
```

## Common Issues

1. **Grad-CAM target layer**: Choose the last convolutional layer for best results
2. **Memory usage**: Grad-CAM can be memory-intensive for large images
3. **Class imbalance**: Use balanced metrics (F1, AUC) instead of accuracy
4. **Confidence intervals**: Require sufficient test samples (>100) for reliability

## Report Contents

### Standard Report
- Classification metrics (accuracy, precision, recall, F1, AUC)
- Confusion matrix
- ROC curve
- Per-class performance

### Enhanced Report
- All standard metrics
- Statistical significance tests
- Confidence intervals (bootstrap)
- Subgroup analysis (by age, sex, etc.)
- Calibration curves
- Grad-CAM visualizations
- Attention weight analysis

## Related Modules

- `trainers/`: Uses evaluation during training
- `cli/evaluate.py`: CLI interface for evaluation
- `models/`: Models being evaluated
- `datasets/`: Test data for evaluation
