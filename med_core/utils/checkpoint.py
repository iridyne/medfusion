"""Checkpoint management utilities for model saving and loading."""

import logging
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    filepath: str | Path,
    metrics: dict[str, float] | None = None,
    scheduler: Any | None = None,
    **kwargs,
) -> None:
    """
    Save model checkpoint.

    Args:
        model: Model to save
        optimizer: Optimizer state
        epoch: Current epoch number
        filepath: Path to save checkpoint
        metrics: Optional metrics dictionary
        scheduler: Optional learning rate scheduler
        **kwargs: Additional items to save
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "metrics": metrics or {},
    }

    if scheduler is not None:
        checkpoint["scheduler_state_dict"] = scheduler.state_dict()

    checkpoint.update(kwargs)

    torch.save(checkpoint, filepath)
    logger.info(f"Checkpoint saved to {filepath}")


def load_checkpoint(
    filepath: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any | None = None,
    device: str = "cpu",
) -> dict[str, Any]:
    """
    Load model checkpoint.

    Args:
        filepath: Path to checkpoint file
        model: Model to load weights into
        optimizer: Optional optimizer to load state into
        scheduler: Optional scheduler to load state into
        device: Device to load checkpoint to

    Returns:
        Dictionary containing checkpoint metadata
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Checkpoint not found: {filepath}")

    checkpoint = torch.load(filepath, map_location=device)

    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    logger.info(f"Checkpoint loaded from {filepath}")

    return {
        "epoch": checkpoint.get("epoch", 0),
        "metrics": checkpoint.get("metrics", {}),
    }


def find_best_checkpoint(
    checkpoint_dir: str | Path,
    metric_name: str = "val_auc",
    mode: str = "max",
) -> Path | None:
    """
    Find best checkpoint based on metric.

    Args:
        checkpoint_dir: Directory containing checkpoints
        metric_name: Metric to compare
        mode: "max" or "min"

    Returns:
        Path to best checkpoint or None
    """
    checkpoint_dir = Path(checkpoint_dir)
    if not checkpoint_dir.exists():
        return None

    checkpoints = list(checkpoint_dir.glob("*.pth"))
    if not checkpoints:
        return None

    best_checkpoint = None
    best_value = float("-inf") if mode == "max" else float("inf")

    for ckpt_path in checkpoints:
        try:
            checkpoint = torch.load(ckpt_path, map_location="cpu")
            metrics = checkpoint.get("metrics", {})
            value = metrics.get(metric_name)

            if value is None:
                continue

            if (mode == "max" and value > best_value) or (mode == "min" and value < best_value):
                best_value = value
                best_checkpoint = ckpt_path

        except Exception as e:
            logger.warning(f"Failed to load checkpoint {ckpt_path}: {e}")

    return best_checkpoint


def cleanup_checkpoints(
    checkpoint_dir: str | Path,
    keep_top_k: int = 3,
    metric_name: str = "val_auc",
    mode: str = "max",
    keep_last: bool = True,
) -> None:
    """
    Remove old checkpoints, keeping only the best ones.

    Args:
        checkpoint_dir: Directory containing checkpoints
        keep_top_k: Number of best checkpoints to keep
        metric_name: Metric to rank by
        mode: "max" or "min"
        keep_last: Whether to always keep the last checkpoint
    """
    checkpoint_dir = Path(checkpoint_dir)
    if not checkpoint_dir.exists():
        return

    checkpoints = list(checkpoint_dir.glob("*.pth"))
    if len(checkpoints) <= keep_top_k:
        return

    # Load and rank checkpoints
    checkpoint_data = []
    for ckpt_path in checkpoints:
        try:
            checkpoint = torch.load(ckpt_path, map_location="cpu")
            metrics = checkpoint.get("metrics", {})
            value = metrics.get(metric_name, float("-inf") if mode == "max" else float("inf"))
            epoch = checkpoint.get("epoch", 0)
            checkpoint_data.append((ckpt_path, value, epoch))
        except Exception:
            continue

    # Sort by metric
    reverse = mode == "max"
    checkpoint_data.sort(key=lambda x: x[1], reverse=reverse)

    # Keep top k
    to_keep = {item[0] for item in checkpoint_data[:keep_top_k]}

    # Keep last checkpoint
    if keep_last and checkpoint_data:
        last_checkpoint = max(checkpoint_data, key=lambda x: x[2])[0]
        to_keep.add(last_checkpoint)

    # Remove others
    for ckpt_path in checkpoints:
        if ckpt_path not in to_keep:
            ckpt_path.unlink()
            logger.info(f"Removed checkpoint: {ckpt_path}")
