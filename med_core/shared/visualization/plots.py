"""
Medical ML Visualization Utilities
==================================
Publication-ready plots for model evaluation and analysis.
"""

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import matplotlib.pyplot as plt  # type: ignore
import numpy as np  # type: ignore
import seaborn as sns  # type: ignore
from sklearn.calibration import calibration_curve  # type: ignore
from sklearn.metrics import (  # type: ignore
    auc,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
)

# Help static type-checkers in environments where third-party stubs are unavailable.
# At runtime the normal imports above are used; during type-checking we expose a
# lightweight alias `NPArray` for numpy arrays and Figure/Axes aliases to reduce
# false-positive noise from unresolved third-party stubs.
if TYPE_CHECKING:
    from matplotlib.axes import Axes  # type: ignore
    from matplotlib.figure import Figure  # type: ignore
    from numpy import ndarray as NPArray  # type: ignore
else:
    NPArray = object  # type: ignore
    Figure = object  # type: ignore
    Axes = object  # type: ignore

logger = logging.getLogger(__name__)

# Publication-quality defaults
plt.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})


def plot_roc_curve(
    y_true: "NPArray",
    y_prob: "NPArray",
    title: str = "ROC Curve",
    save_path: str | Path | None = None,
    show: bool = False,
    figsize: tuple[float, float] = (8, 6),
    ax: "Axes | None" = None,
) -> tuple["Figure", "Axes"]:
    """Plot ROC curve with optimal threshold."""
    y_true = np.asarray(y_true).ravel()
    y_prob = np.asarray(y_prob).ravel()

    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    ax.plot(fpr, tpr, color="darkorange", lw=2, label=f"AUC = {roc_auc:.4f}")
    ax.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--", label="Random")

    # Optimal threshold (Youden's J)
    optimal_idx = np.argmax(tpr - fpr)
    ax.scatter(fpr[optimal_idx], tpr[optimal_idx], marker="o", color="red", s=100, zorder=5)

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path)
        logger.info(f"ROC curve saved to {save_path}")

    if show:
        plt.show()

    return fig, ax


def plot_pr_curve(
    y_true: "NPArray",
    y_prob: "NPArray",
    title: str = "Precision-Recall Curve",
    save_path: str | Path | None = None,
    show: bool = False,
    figsize: tuple[float, float] = (8, 6),
    ax: "Axes | None" = None,
) -> tuple["Figure", "Axes"]:
    """Plot Precision-Recall curve."""
    y_true = np.asarray(y_true).ravel()
    y_prob = np.asarray(y_prob).ravel()

    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = auc(recall, precision)
    baseline = np.sum(y_true) / len(y_true)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    ax.plot(recall, precision, color="green", lw=2, label=f"AUC = {pr_auc:.4f}")
    ax.axhline(y=baseline, color="navy", lw=2, linestyle="--", label=f"Baseline ({baseline:.2f})")

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(title)
    ax.legend(loc="lower left")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path)
        logger.info(f"PR curve saved to {save_path}")

    if show:
        plt.show()

    return fig, ax


def plot_confusion_matrix(
    y_true: "NPArray",
    y_pred: "NPArray",
    class_names: list[str] | None = None,
    title: str = "Confusion Matrix",
    save_path: str | Path | None = None,
    show: bool = False,
    figsize: tuple[float, float] = (8, 6),
    normalize: Literal["true", "pred", "all", None] = None,
    cmap: str = "Blues",
) -> tuple["Figure", "Axes"]:
    """Plot confusion matrix heatmap."""
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()

    cm = confusion_matrix(y_true, y_pred)

    if normalize == "true":
        cm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        fmt = ".2f"
    elif normalize == "pred":
        cm = cm.astype(float) / cm.sum(axis=0, keepdims=True)
        fmt = ".2f"
    elif normalize == "all":
        cm = cm.astype(float) / cm.sum()
        fmt = ".2f"
    else:
        fmt = "d"

    cm = np.nan_to_num(cm)

    if class_names is None:
        n_classes = cm.shape[0]
        class_names = ["Negative", "Positive"] if n_classes == 2 else [f"Class {i}" for i in range(n_classes)]

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(cm, annot=True, fmt=fmt, cmap=cmap, xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title(title)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path)
        logger.info(f"Confusion matrix saved to {save_path}")

    if show:
        plt.show()

    return fig, ax


def plot_training_curves(
    train_losses: list[float],
    val_losses: list[float] | None = None,
    train_metrics: dict[str, list[float]] | None = None,
    val_metrics: dict[str, list[float]] | None = None,
    title: str = "Training Curves",
    save_path: str | Path | None = None,
    show: bool = False,
    figsize: tuple[float, float] = (12, 5),
) -> tuple["Figure", list["Axes"]]:
    """Plot training and validation curves."""
    n_plots = 1 + (1 if (train_metrics or val_metrics) else 0)
    fig, axes = plt.subplots(1, n_plots, figsize=figsize)
    if n_plots == 1:
        axes = [axes]

    epochs = range(1, len(train_losses) + 1)

    # Loss plot
    ax_loss = axes[0]
    ax_loss.plot(epochs, train_losses, "b-", label="Train Loss", lw=2)
    if val_losses:
        ax_loss.plot(epochs, val_losses, "r-", label="Val Loss", lw=2)
        best_epoch = np.argmin(val_losses) + 1
        ax_loss.axvline(x=best_epoch, color="gray", linestyle="--", alpha=0.5)
        ax_loss.scatter([best_epoch], [min(val_losses)], color="red", s=100, zorder=5, marker="*")

    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Loss")
    ax_loss.set_title("Loss")
    ax_loss.legend()
    ax_loss.grid(True, alpha=0.3)

    # Metrics plot
    if n_plots > 1:
        ax_metrics = axes[1]
        colors = plt.cm.tab10(np.linspace(0, 1, 10))
        color_idx = 0

        if train_metrics:
            for name, values in train_metrics.items():
                ax_metrics.plot(epochs[:len(values)], values, "-", color=colors[color_idx], label=f"Train {name}", lw=2)
                color_idx = (color_idx + 1) % 10

        if val_metrics:
            for name, values in val_metrics.items():
                ax_metrics.plot(epochs[:len(values)], values, "--", color=colors[color_idx], label=f"Val {name}", lw=2)
                color_idx = (color_idx + 1) % 10

        ax_metrics.set_xlabel("Epoch")
        ax_metrics.set_ylabel("Metric Value")
        ax_metrics.set_title("Metrics")
        ax_metrics.legend()
        ax_metrics.grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=14, y=1.02)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path)
        logger.info(f"Training curves saved to {save_path}")

    if show:
        plt.show()

    return fig, axes


def plot_calibration_curve(
    y_true: "NPArray",
    y_prob: "NPArray",
    n_bins: int = 10,
    title: str = "Calibration Curve",
    save_path: str | Path | None = None,
    show: bool = False,
    figsize: tuple[float, float] = (8, 6),
) -> tuple["Figure", "Axes"]:
    """Plot calibration curve (reliability diagram)."""
    y_true = np.asarray(y_true).ravel()
    y_prob = np.asarray(y_prob).ravel()

    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy="uniform")

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(prob_pred, prob_true, "s-", color="blue", lw=2, label="Model")
    ax.plot([0, 1], [0, 1], "k--", lw=2, label="Perfect")

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.0])
    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Fraction of Positives")
    ax.set_title(title)
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)

    # Histogram overlay
    ax2 = ax.twinx()
    ax2.hist(y_prob, bins=n_bins, range=(0, 1), alpha=0.3, color="gray")
    ax2.set_ylabel("Count", color="gray")
    ax2.tick_params(axis="y", labelcolor="gray")

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path)
        logger.info(f"Calibration curve saved to {save_path}")

    if show:
        plt.show()

    return fig, ax


def plot_probability_distribution(
    y_true: "NPArray",
    y_prob: "NPArray",
    title: str = "Prediction Probability Distribution",
    save_path: str | Path | None = None,
    show: bool = False,
    figsize: tuple[float, float] = (10, 6),
) -> tuple["Figure", "Axes"]:
    """Plot prediction probability distribution by true class."""
    y_true = np.asarray(y_true).ravel()
    y_prob = np.asarray(y_prob).ravel()

    fig, ax = plt.subplots(figsize=figsize)
    ax.hist(y_prob[y_true == 0], bins=20, alpha=0.6, label="Negative (True)", color="blue")
    ax.hist(y_prob[y_true == 1], bins=20, alpha=0.6, label="Positive (True)", color="red")

    ax.set_xlabel("Predicted Probability (Positive)")
    ax.set_ylabel("Count")
    ax.set_title(title)
    ax.legend(loc="upper center")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path)
        logger.info(f"Probability distribution saved to {save_path}")

    if show:
        plt.show()

    return fig, ax
