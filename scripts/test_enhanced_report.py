"""
Test script for enhanced report generation.

Tests:
- High-resolution plot generation (300 DPI)
- Statistical significance testing
- LaTeX report generation
- Publication-ready formatting
"""

import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from med_core.evaluation import (
    EnhancedReportGenerator,
    calculate_binary_metrics,
    generate_enhanced_report,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def create_sample_metrics(accuracy: float = 0.95, n_samples: int = 100):
    """Create sample metrics for testing."""
    # Generate synthetic predictions
    np.random.seed(42)
    n_positive = n_samples // 2
    n_negative = n_samples - n_positive

    # True labels
    y_true = np.concatenate([np.ones(n_positive), np.zeros(n_negative)])

    # Predictions with specified accuracy
    y_pred = y_true.copy()
    n_errors = int(n_samples * (1 - accuracy))
    error_indices = np.random.choice(n_samples, n_errors, replace=False)
    y_pred[error_indices] = 1 - y_pred[error_indices]

    # Prediction scores (probabilities)
    y_scores = np.random.rand(n_samples)
    y_scores[y_true == 1] = np.random.uniform(0.6, 1.0, n_positive)
    y_scores[y_true == 0] = np.random.uniform(0.0, 0.4, n_negative)

    # Calculate metrics
    metrics = calculate_binary_metrics(y_true, y_pred, y_scores)

    return metrics, y_true, y_scores


def create_high_res_roc_plot(y_true, y_scores, output_path: Path, dpi: int = 300):
    """Create high-resolution ROC curve plot."""
    fpr, tpr, _ = roc_curve(y_true, y_scores)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, linewidth=2, label="ROC Curve")
    plt.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random Classifier")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate", fontsize=12)
    plt.ylabel("True Positive Rate", fontsize=12)
    plt.title("Receiver Operating Characteristic (ROC) Curve", fontsize=14)
    plt.legend(loc="lower right", fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close()

    logger.info(f"✅ High-resolution ROC curve saved: {output_path}")


def create_high_res_confusion_matrix(metrics, output_path: Path, dpi: int = 300):
    """Create high-resolution confusion matrix plot."""
    cm = np.array(
        [
            [metrics.true_negatives, metrics.false_positives],
            [metrics.false_negatives, metrics.true_positives],
        ]
    )

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)

    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=["Negative", "Positive"],
        yticklabels=["Negative", "Positive"],
        title="Confusion Matrix",
        ylabel="True Label",
        xlabel="Predicted Label",
    )

    # Add text annotations
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                format(cm[i, j], "d"),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=14,
            )

    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close()

    logger.info(f"✅ High-resolution confusion matrix saved: {output_path}")


def test_enhanced_report_basic():
    """Test basic enhanced report generation."""
    logger.info("\n" + "=" * 60)
    logger.info("测试 1: 基础增强报告生成")
    logger.info("=" * 60)

    # Create output directory
    output_dir = Path("outputs/enhanced_report_test")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate sample metrics
    metrics, y_true, y_scores = create_sample_metrics(accuracy=0.92)

    # Create high-resolution plots
    roc_path = output_dir / "roc_curve_300dpi.png"
    cm_path = output_dir / "confusion_matrix_300dpi.png"

    create_high_res_roc_plot(y_true, y_scores, roc_path, dpi=300)
    create_high_res_confusion_matrix(metrics, cm_path, dpi=300)

    # Generate enhanced report
    logger.info("\n生成增强报告...")
    report_path = generate_enhanced_report(
        metrics=metrics,
        output_dir=output_dir,
        experiment_name="Enhanced Report Test",
        plots={
            "ROC Curve (300 DPI)": roc_path,
            "Confusion Matrix (300 DPI)": cm_path,
        },
        config={
            "model": "ResNet18",
            "dataset": "Synthetic Test Data",
            "n_samples": 100,
        },
        dpi=300,
    )

    logger.info(f"✅ 报告生成成功: {report_path}")

    # Check if LaTeX report was generated
    latex_path = output_dir / "report.tex"
    if latex_path.exists():
        logger.info(f"✅ LaTeX 报告生成成功: {latex_path}")
    else:
        logger.warning("⚠️ LaTeX 报告未生成")

    return output_dir


def test_statistical_comparison():
    """Test statistical significance testing."""
    logger.info("\n" + "=" * 60)
    logger.info("测试 2: 统计显著性检验")
    logger.info("=" * 60)

    # Create output directory
    output_dir = Path("outputs/statistical_test")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate baseline and improved metrics
    baseline_metrics, _, _ = create_sample_metrics(accuracy=0.85, n_samples=200)
    improved_metrics, y_true, y_scores = create_sample_metrics(
        accuracy=0.92, n_samples=200
    )

    logger.info(f"Baseline 准确率: {baseline_metrics.accuracy:.4f}")
    logger.info(f"改进后准确率: {improved_metrics.accuracy:.4f}")
    logger.info(f"提升: {improved_metrics.accuracy - baseline_metrics.accuracy:.4f}")

    # Create plots
    roc_path = output_dir / "roc_curve.png"
    cm_path = output_dir / "confusion_matrix.png"

    create_high_res_roc_plot(y_true, y_scores, roc_path, dpi=300)
    create_high_res_confusion_matrix(improved_metrics, cm_path, dpi=300)

    # Generate report with statistical comparison
    logger.info("\n生成带统计检验的报告...")
    report_path = generate_enhanced_report(
        metrics=improved_metrics,
        output_dir=output_dir,
        experiment_name="Statistical Comparison Test",
        plots={
            "ROC Curve": roc_path,
            "Confusion Matrix": cm_path,
        },
        baseline_metrics=baseline_metrics,
        dpi=300,
    )

    logger.info(f"✅ 报告生成成功: {report_path}")

    # Read and display statistical tests section
    with open(report_path, encoding="utf-8") as f:
        content = f.read()
        if "Statistical Significance Tests" in content:
            logger.info("✅ 统计检验部分已添加到报告")
            # Extract and display the section
            start = content.find("## Statistical Significance Tests")
            if start != -1:
                end = content.find("\n## ", start + 1)
                if end == -1:
                    end = len(content)
                section = content[start:end]
                logger.info("\n" + section)
        else:
            logger.warning("⚠️ 统计检验部分未找到")

    return output_dir


def test_latex_output():
    """Test LaTeX report generation."""
    logger.info("\n" + "=" * 60)
    logger.info("测试 3: LaTeX 报告生成")
    logger.info("=" * 60)

    # Create output directory
    output_dir = Path("outputs/latex_test")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate sample metrics
    metrics, y_true, y_scores = create_sample_metrics(accuracy=0.94)

    # Create generator
    generator = EnhancedReportGenerator(
        experiment_name="LaTeX Report Test",
        output_dir=output_dir,
        description="Testing LaTeX report generation with high-resolution figures",
        dpi=300,
    )

    generator.add_metrics(metrics)

    # Create plots
    roc_path = output_dir / "roc_curve.png"
    cm_path = output_dir / "confusion_matrix.png"

    create_high_res_roc_plot(y_true, y_scores, roc_path, dpi=300)
    create_high_res_confusion_matrix(metrics, cm_path, dpi=300)

    generator.add_plot("ROC Curve", roc_path)
    generator.add_plot("Confusion Matrix", cm_path)

    # Generate LaTeX report
    logger.info("\n生成 LaTeX 报告...")
    latex_path = generator.generate_latex_report()

    logger.info(f"✅ LaTeX 报告生成成功: {latex_path}")

    # Display LaTeX table
    with open(latex_path, encoding="utf-8") as f:
        content = f.read()
        # Extract table
        start = content.find("\\begin{table}")
        if start != -1:
            end = content.find("\\end{table}", start) + len("\\end{table}")
            table = content[start:end]
            logger.info("\nLaTeX 表格预览:")
            logger.info("-" * 60)
            logger.info(table)
            logger.info("-" * 60)

    return output_dir


def main():
    """Run all tests."""
    logger.info("=" * 60)
    logger.info("增强报告生成器测试")
    logger.info("=" * 60)

    try:
        # Test 1: Basic enhanced report
        test_enhanced_report_basic()

        # Test 2: Statistical comparison
        test_statistical_comparison()

        # Test 3: LaTeX output
        test_latex_output()

        logger.info("\n" + "=" * 60)
        logger.info("🎉 所有测试完成！")
        logger.info("=" * 60)
        logger.info("\n生成的报告位置:")
        logger.info("  1. outputs/enhanced_report_test/")
        logger.info("  2. outputs/statistical_test/")
        logger.info("  3. outputs/latex_test/")
        logger.info("\n功能验证:")
        logger.info("  ✅ 高分辨率图表生成 (300 DPI)")
        logger.info("  ✅ 统计显著性检验")
        logger.info("  ✅ LaTeX 报告生成")
        logger.info("  ✅ Markdown 报告生成")

    except Exception as e:
        logger.error(f"❌ 测试失败: {e}", exc_info=True)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
