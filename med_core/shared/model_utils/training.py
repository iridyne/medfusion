"""
Medical ML Training Utilities
=============================
Common training utilities including early stopping and checkpointing.
"""

import logging
from pathlib import Path
from typing import Any, Literal

import torch
from torch import nn
from torch.optim import Optimizer

logger = logging.getLogger(__name__)


class EarlyStopping:
    """
    Early stopping to stop training when monitored metric stops improving.

    Example:
        >>> early_stopping = EarlyStopping(patience=5, mode="min")
        >>> for epoch in range(num_epochs):
        ...     val_loss = train_epoch()
        ...     early_stopping(val_loss, model)
        ...     if early_stopping.early_stop:
        ...         break
    """

    def __init__(
        self,
        patience: int = 7,
        mode: Literal["min", "max"] = "min",
        delta: float = 0.0,
        verbose: bool = True,
    ):
        """
        Args:
            patience: Number of epochs to wait before stopping
            mode: "min" to minimize metric, "max" to maximize
            delta: Minimum change to qualify as improvement
            verbose: Whether to log messages
        """
        self.patience: int = patience
        self.mode: Literal["min", "max"] = mode
        self.delta: float = delta
        self.verbose: bool = verbose

        self.counter: int = 0
        self.best_score: float | None = None
        self.early_stop: bool = False
        self.best_model_state: dict[str, torch.Tensor] | None = None

    def __call__(self, score: float, model: nn.Module) -> None:
        """
        Check if training should stop.

        Args:
            score: Current metric value
            model: Model to save state from
        """
        current_score = score if self.mode == "max" else -score

        if self.best_score is None:
            self.best_score = current_score
            self._save_checkpoint(model)
        elif current_score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                logger.info(f"EarlyStopping counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = current_score
            self._save_checkpoint(model)
            self.counter = 0

    def _save_checkpoint(self, model: nn.Module) -> None:
        """Save model state."""
        self.best_model_state = {
            k: v.cpu().clone() for k, v in model.state_dict().items()
        }
        if self.verbose:
            logger.info("Validation metric improved, saving model")

    def load_best_model(self, model: nn.Module) -> None:
        """Load best model state back into model."""
        if self.best_model_state is not None:
            model.load_state_dict(self.best_model_state)
            if self.verbose:
                logger.info("Loaded best model state")


class ModelCheckpoint:
    """
    Save model checkpoints during training.

    Example:
        >>> checkpoint = ModelCheckpoint(save_dir="./checkpoints", monitor="val_loss")
        >>> checkpoint.save(model, epoch, val_loss)
    """

    def __init__(
        self,
        save_dir: str | Path,
        monitor: str = "val_loss",
        mode: Literal["min", "max"] = "min",
        save_top_k: int = 1,
        save_last: bool = True,
    ):
        """
        Args:
            save_dir: Directory to save checkpoints
            monitor: Metric to monitor
            mode: "min" to minimize, "max" to maximize
            save_top_k: Number of best models to keep
            save_last: Whether to save last checkpoint
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.monitor = monitor
        self.mode = mode
        self.save_top_k = save_top_k
        self.save_last = save_last

        self.best_scores: list[float] = []
        self.best_model_path: str | None = None

    def save(
        self,
        model: nn.Module,
        epoch: int,
        metrics: dict[str, Any],
        optimizer: Optimizer | None = None,
    ) -> None:
        """
        Save checkpoint if metric improved.

        Args:
            model: Model to save
            epoch: Current epoch
            metrics: Dictionary of metrics
            optimizer: Optimizer state to save (optional)
        """
        score = metrics.get(self.monitor)
        if score is None:
            logger.warning(f"Metric {self.monitor} not found in metrics")
            return

        # Check if this is a top-k model
        is_best = False
        if len(self.best_scores) < self.save_top_k:
            is_best = True
        else:
            worst_best = (
                min(self.best_scores) if self.mode == "max" else max(self.best_scores)
            )
            if (self.mode == "max" and score > worst_best) or (
                self.mode == "min" and score < worst_best
            ):
                is_best = True

        if is_best:
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "metrics": metrics,
            }
            if optimizer:
                checkpoint["optimizer_state_dict"] = optimizer.state_dict()

            filename = f"epoch_{epoch}_{self.monitor}_{score:.4f}.pth"
            path = self.save_dir / filename
            torch.save(checkpoint, path)

            self.best_scores.append(score)
            self.best_scores.sort(reverse=(self.mode == "max"))
            self.best_scores = self.best_scores[: self.save_top_k]
            self.best_model_path = str(path)

            logger.info(f"Saved checkpoint: {path}")

        # Save last checkpoint
        if self.save_last:
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "metrics": metrics,
            }
            if optimizer:
                checkpoint["optimizer_state_dict"] = optimizer.state_dict()

            path = self.save_dir / "last.pth"
            torch.save(checkpoint, path)


def load_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: Optimizer | None = None,
    device: str = "cpu",
) -> dict[str, Any]:
    """
    Load model checkpoint.

    Args:
        path: Path to checkpoint file
        model: Model to load state into
        optimizer: Optimizer to load state into (optional)
        device: Device to map tensors to

    Returns:
        Dictionary containing checkpoint metadata
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    checkpoint = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    logger.info(f"Loaded checkpoint from {path}")
    return checkpoint
