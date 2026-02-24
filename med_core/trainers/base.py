"""
Base trainer class for medical deep learning models.

Defines the standard training loop, validation, checkpointing, and logging structure.
"""

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from med_core.configs import ExperimentConfig

logger = logging.getLogger(__name__)


class BaseTrainer(ABC):
    """
    Abstract base trainer class.

    Implements the skeleton of the training process:
    - Epoch loops
    - Batch iteration
    - Logging
    - Checkpointing
    - Early stopping logic
    """

    def __init__(
        self,
        config: ExperimentConfig,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: torch.optim.Optimizer | None = None,
        scheduler: Any | None = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.config = config
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader

        # Create optimizer if not provided
        if optimizer is None:
            lr = config.training.optimizer.learning_rate
            weight_decay = getattr(config.training.optimizer, "weight_decay", 0.0)
            self.optimizer = optim.AdamW(
                model.parameters(),
                lr=lr,
                weight_decay=weight_decay,
            )
            logger.info(
                f"Auto-created AdamW optimizer with lr={lr}, weight_decay={weight_decay}"
            )
        else:
            self.optimizer = optimizer

        self.scheduler = scheduler
        self.device = torch.device(device)

        # Move model to device
        self.model.to(self.device)

        # Setup logging
        self.log_dir = config.log_dir
        self.writer = SummaryWriter(log_dir=str(self.log_dir))

        # Setup checkpointing
        self.checkpoint_dir = config.checkpoint_dir

        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_metric = (
            float("-inf") if config.training.mode == "max" else float("inf")
        )
        self.patience_counter = 0

    @abstractmethod
    def training_step(self, batch: Any, batch_idx: int) -> dict[str, torch.Tensor]:
        """
        Perform a single training step.

        Args:
            batch: Data batch from dataloader
            batch_idx: Index of the batch

        Returns:
            Dictionary containing 'loss' and any other metrics to log
        """
        pass

    @abstractmethod
    def validation_step(self, batch: Any, batch_idx: int) -> dict[str, torch.Tensor]:
        """
        Perform a single validation step.

        Args:
            batch: Data batch from dataloader
            batch_idx: Index of the batch

        Returns:
            Dictionary containing metrics to aggregate
        """
        pass

    def on_train_start(self):
        """Hook called at the start of training."""
        logger.info(f"Starting training on device: {self.device}")
        logger.info(
            f"Model parameters: {sum(p.numel() for p in self.model.parameters())}"
        )

    def on_epoch_start(self):
        """Hook called at the start of each epoch."""
        # Provide a minimal non-empty default implementation to satisfy linters
        # and to give a useful debug message at runtime. Subclasses can override.
        try:
            epoch = getattr(self, "current_epoch", None)
            logger.debug(f"BaseTrainer.on_epoch_start invoked (epoch={epoch})")
        except Exception:
            # Be defensive: do not raise if logging fails for any reason.
            pass

    def on_epoch_end(
        self, train_metrics: dict[str, float], val_metrics: dict[str, float]
    ):
        """
        Hook called at the end of each epoch.

        Args:
            train_metrics: Aggregated training metrics
            val_metrics: Aggregated validation metrics
        """
        # Update scheduler
        if self.scheduler is not None:
            if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(val_metrics.get(self.config.training.monitor, 0.0))
            else:
                self.scheduler.step()

        # Logging
        current_lr = self.optimizer.param_groups[0]["lr"]
        self.writer.add_scalar("learning_rate", current_lr, self.current_epoch)

        for k, v in train_metrics.items():
            self.writer.add_scalar(f"train/{k}", v, self.current_epoch)

        for k, v in val_metrics.items():
            self.writer.add_scalar(f"val/{k}", v, self.current_epoch)

        logger.info(
            f"Epoch {self.current_epoch}: "
            f"Train Loss: {train_metrics.get('loss', 0):.4f} | "
            f"Val Loss: {val_metrics.get('loss', 0):.4f} | "
            f"Val Metric ({self.config.training.monitor}): {val_metrics.get(self.config.training.monitor, 0):.4f}"
        )

        # Checkpointing & Early Stopping
        self._handle_checkpointing(val_metrics)

    def train(self) -> dict[str, list[float]]:
        """
        Main training loop.

        Returns:
            Dictionary containing training history with keys like 'train_loss', 'val_loss', etc.
        """
        self.on_train_start()

        # Initialize history tracking
        history = {
            "train_loss": [],
            "val_loss": [],
        }

        for epoch in range(self.config.training.num_epochs):
            self.current_epoch = epoch
            self.on_epoch_start()

            # Training Loop
            self.model.train()
            train_metrics = self._run_epoch(self.train_loader, training=True)

            # Validation Loop
            self.model.eval()
            with torch.no_grad():
                val_metrics = self._run_epoch(self.val_loader, training=False)

            self.on_epoch_end(train_metrics, val_metrics)

            # Record history
            history["train_loss"].append(train_metrics.get("loss", 0.0))
            history["val_loss"].append(val_metrics.get("loss", 0.0))

            # Record other metrics
            for key, value in train_metrics.items():
                if key != "loss":
                    history_key = f"train_{key}"
                    if history_key not in history:
                        history[history_key] = []
                    history[history_key].append(value)

            for key, value in val_metrics.items():
                if key != "loss":
                    history_key = f"val_{key}"
                    if history_key not in history:
                        history[history_key] = []
                    history[history_key].append(value)

            if self.patience_counter >= self.config.training.patience:
                logger.info("Early stopping triggered")
                break

        self.writer.close()
        return history

    def _run_epoch(self, loader: DataLoader, training: bool = True) -> dict[str, float]:
        """Run a single epoch of training or validation."""
        metrics_sum = {}
        num_batches = len(loader)

        pbar = tqdm(
            loader,
            desc=f"Epoch {self.current_epoch} [{'Train' if training else 'Val'}]",
            leave=False,
        )

        for batch_idx, batch in enumerate(pbar):
            if training:
                # Zero gradients
                self.optimizer.zero_grad()

                # Forward + Backward
                step_metrics = self.training_step(batch, batch_idx)
                loss = step_metrics["loss"]
                loss.backward()

                # Gradient clipping
                if self.config.training.gradient_clip:
                    nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.config.training.gradient_clip
                    )

                # Optimizer step
                self.optimizer.step()
                self.global_step += 1

            else:
                step_metrics = self.validation_step(batch, batch_idx)

            # Aggregate metrics
            for k, v in step_metrics.items():
                val = v.item() if isinstance(v, torch.Tensor) else v
                metrics_sum[k] = metrics_sum.get(k, 0.0) + val

            # Update progress bar
            pbar.set_postfix(
                {k: f"{v.item():.4f}" for k, v in step_metrics.items() if k == "loss"}
            )

        return {k: v / num_batches for k, v in metrics_sum.items()}

    def _handle_checkpointing(self, val_metrics: dict[str, float]):
        """Handle model checkpointing and early stopping logic."""
        current_metric = val_metrics.get(
            self.config.training.monitor, val_metrics.get("loss")
        )

        if current_metric is None:
            logger.warning(
                f"Metric {self.config.training.monitor} not found in validation metrics."
            )
            return

        is_improvement = False
        if self.config.training.mode == "min":
            if current_metric < self.best_metric - self.config.training.min_delta:
                is_improvement = True
        else:
            if current_metric > self.best_metric + self.config.training.min_delta:
                is_improvement = True

        if is_improvement:
            self.best_metric = current_metric
            self.patience_counter = 0
            self._save_checkpoint("best.pth", val_metrics)
            logger.info(
                f"New best model saved with {self.config.training.monitor}: {self.best_metric:.4f}"
            )
        else:
            self.patience_counter += 1

        if self.config.training.save_last:
            self._save_checkpoint("last.pth", val_metrics)

    def _save_checkpoint(self, filename: str, metrics: dict[str, float]):
        """Save model checkpoint with atomic write to prevent corruption."""
        checkpoint = {
            "epoch": self.current_epoch,
            "global_step": self.global_step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "metrics": metrics,
            "config": self.config.to_dict(),
        }

        path = self.checkpoint_dir / filename
        temp_path = path.with_suffix(".tmp")

        # Write to temporary file first
        torch.save(checkpoint, temp_path)
        # Atomic rename
        temp_path.replace(path)

    def load_checkpoint(self, path: str | Path):
        """Load model checkpoint."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        checkpoint = torch.load(path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.current_epoch = checkpoint["epoch"]
        self.global_step = checkpoint["global_step"]

        logger.info(f"Loaded checkpoint from {path}")
