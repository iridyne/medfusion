"""
Medical Training Engine Module

This module implements a robust training engine for medical multimodal models.

Key Features:
- Automatic Mixed Precision (AMP) training
- Medical-specific metrics (AUC-ROC, F1-Score, Specificity)
- TensorBoard and JSON logging
- Early stopping and checkpointing
- Learning rate scheduling
- Type-safe implementation with Python 3.12+ hints
"""

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


class MedicalTrainer:
    """
    Robust training engine for medical multimodal models.

    Features:
    - AMP training for memory efficiency
    - Medical metrics: AUC-ROC, F1, Specificity, Sensitivity
    - TensorBoard and JSON logging
    - Early stopping
    - Model checkpointing
    - Learning rate scheduling
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        num_epochs: int = 100,
        use_amp: bool = True,
        log_dir: str | Path = "logs",
        checkpoint_dir: str | Path = "checkpoints",
        early_stopping_patience: int = 10,
        scheduler: torch.optim.lr_scheduler._LRScheduler | None = None,
        gradient_clip: float | None = 1.0,
    ):
        """
        Initialize medical trainer.

        Args:
            model: Neural network model
            train_loader: Training data loader
            val_loader: Validation data loader
            criterion: Loss function
            optimizer: Optimizer
            device: Device to train on
            num_epochs: Number of training epochs
            use_amp: Use automatic mixed precision
            log_dir: Directory for logs
            checkpoint_dir: Directory for model checkpoints
            early_stopping_patience: Patience for early stopping
            scheduler: Learning rate scheduler
            gradient_clip: Gradient clipping value (None to disable)
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.num_epochs = num_epochs
        self.use_amp = use_amp
        self.scheduler = scheduler
        self.gradient_clip = gradient_clip

        # Setup directories
        self.log_dir = Path(log_dir)
        self.checkpoint_dir = Path(checkpoint_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Setup AMP
        self.scaler = GradScaler(
            "cuda" if device.type == "cuda" else "cpu", enabled=use_amp
        )

        # Setup logging
        self.writer = SummaryWriter(log_dir=str(self.log_dir))
        self.metrics_history: list[dict] = []

        # Early stopping
        self.early_stopping_patience = early_stopping_patience
        self.best_val_loss = float("inf")
        self.best_val_auc = 0.0
        self.patience_counter = 0

        # Training state
        self.current_epoch = 0
        self.global_step = 0

    def _compute_medical_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: np.ndarray,
    ) -> dict[str, float]:
        """
        Compute medical-specific metrics.

        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_prob: Prediction probabilities

        Returns:
            Dictionary of metrics
        """
        metrics = {}

        # AUC-ROC (binary or multiclass)
        try:
            if len(np.unique(y_true)) == 2:
                # Binary classification
                metrics["auc_roc"] = roc_auc_score(y_true, y_prob[:, 1])
            else:
                # Multiclass
                metrics["auc_roc"] = roc_auc_score(
                    y_true, y_prob, multi_class="ovr", average="macro"
                )
        except Exception:
            metrics["auc_roc"] = 0.0

        # F1 Score
        metrics["f1_score"] = f1_score(y_true, y_pred, average="macro", zero_division=0)

        # Confusion matrix for sensitivity and specificity
        cm = confusion_matrix(y_true, y_pred)

        if cm.shape[0] == 2:
            # Binary classification
            tn, fp, fn, tp = cm.ravel()
            metrics["sensitivity"] = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            metrics["specificity"] = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            metrics["precision"] = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            metrics["npv"] = tn / (tn + fn) if (tn + fn) > 0 else 0.0
        else:
            # Multiclass - compute macro-averaged metrics
            sensitivities = []
            specificities = []
            for i in range(cm.shape[0]):
                tp = cm[i, i]
                fn = cm[i, :].sum() - tp
                fp = cm[:, i].sum() - tp
                tn = cm.sum() - tp - fn - fp

                sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0

                sensitivities.append(sens)
                specificities.append(spec)

            metrics["sensitivity"] = np.mean(sensitivities)
            metrics["specificity"] = np.mean(specificities)

        # Accuracy
        metrics["accuracy"] = (y_true == y_pred).mean()

        return metrics

    def _train_epoch(self) -> dict[str, float]:
        """
        Train for one epoch.

        Returns:
            Dictionary of training metrics
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        pbar = tqdm(
            self.train_loader,
            desc=f"Epoch {self.current_epoch + 1}/{self.num_epochs} [Train]",
        )

        for images, features, labels in pbar:
            images = images.to(self.device)
            features = features.to(self.device)
            labels = labels.to(self.device)

            # Zero gradients
            self.optimizer.zero_grad()

            # Forward pass with AMP
            with autocast(self.device.type, enabled=self.use_amp):
                outputs = self.model(images, features)
                loss = self.criterion(outputs, labels)

            # Backward pass with gradient scaling
            self.scaler.scale(loss).backward()

            # Gradient clipping
            if self.gradient_clip is not None:
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)

            # Optimizer step
            self.scaler.step(self.optimizer)
            self.scaler.update()

            # Update metrics
            total_loss += loss.item()
            num_batches += 1
            self.global_step += 1

            # Update progress bar
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

            # Log to TensorBoard
            if self.global_step % 10 == 0:
                self.writer.add_scalar(
                    "train/batch_loss", loss.item(), self.global_step
                )

        avg_loss = total_loss / num_batches
        return {"loss": avg_loss}

    def _validate_epoch(self) -> dict[str, float]:
        """
        Validate for one epoch.

        Returns:
            Dictionary of validation metrics
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        all_labels = []
        all_preds = []
        all_probs = []

        pbar = tqdm(
            self.val_loader,
            desc=f"Epoch {self.current_epoch + 1}/{self.num_epochs} [Val]",
        )

        with torch.no_grad():
            for images, features, labels in pbar:
                images = images.to(self.device)
                features = features.to(self.device)
                labels = labels.to(self.device)

                # Forward pass
                with autocast(self.device.type, enabled=self.use_amp):
                    outputs = self.model(images, features)
                    loss = self.criterion(outputs, labels)

                # Compute predictions
                probs = torch.softmax(outputs, dim=1)
                preds = torch.argmax(outputs, dim=1)

                # Collect for metrics
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())

                total_loss += loss.item()
                num_batches += 1

        # Compute metrics
        avg_loss = total_loss / num_batches
        y_true = np.array(all_labels)
        y_pred = np.array(all_preds)
        y_prob = np.array(all_probs)

        metrics = self._compute_medical_metrics(y_true, y_pred, y_prob)
        metrics["loss"] = avg_loss

        return metrics

    def _save_checkpoint(self, filename: str, metrics: dict[str, float]) -> None:
        """
        Save model checkpoint.

        Args:
            filename: Checkpoint filename
            metrics: Current metrics
        """
        checkpoint = {
            "epoch": self.current_epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scaler_state_dict": self.scaler.state_dict(),
            "metrics": metrics,
            "best_val_loss": self.best_val_loss,
            "best_val_auc": self.best_val_auc,
        }

        if self.scheduler is not None:
            checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()

        path = self.checkpoint_dir / filename
        torch.save(checkpoint, path)

    def _log_metrics(
        self, train_metrics: dict[str, float], val_metrics: dict[str, float]
    ) -> None:
        """
        Log metrics to TensorBoard and JSON.

        Args:
            train_metrics: Training metrics
            val_metrics: Validation metrics
        """
        # TensorBoard
        for key, value in train_metrics.items():
            self.writer.add_scalar(f"train/{key}", value, self.current_epoch)

        for key, value in val_metrics.items():
            self.writer.add_scalar(f"val/{key}", value, self.current_epoch)

        # Learning rate
        if self.scheduler is not None:
            lr = self.scheduler.get_last_lr()[0]
        else:
            lr = self.optimizer.param_groups[0]["lr"]
        self.writer.add_scalar("learning_rate", lr, self.current_epoch)

        # JSON history
        epoch_metrics = {
            "epoch": self.current_epoch,
            "train": train_metrics,
            "val": val_metrics,
            "lr": lr,
        }
        self.metrics_history.append(epoch_metrics)

        # Save JSON
        json_path = self.log_dir / "metrics_history.json"
        with open(json_path, "w") as f:
            json.dump(self.metrics_history, f, indent=2)

    def train(self) -> dict[str, float]:
        """
        Run full training loop.

        Returns:
            Best validation metrics
        """
        print(f"Starting training for {self.num_epochs} epochs...")
        print(f"Device: {self.device}")
        print(f"AMP enabled: {self.use_amp}")

        for epoch in range(self.num_epochs):
            self.current_epoch = epoch

            # Train
            train_metrics = self._train_epoch()

            # Validate
            val_metrics = self._validate_epoch()

            # Log metrics
            self._log_metrics(train_metrics, val_metrics)

            # Print summary
            print(f"\nEpoch {epoch + 1}/{self.num_epochs}:")
            print(f"  Train Loss: {train_metrics['loss']:.4f}")
            print(f"  Val Loss: {val_metrics['loss']:.4f}")
            print(f"  Val AUC-ROC: {val_metrics['auc_roc']:.4f}")
            print(f"  Val F1: {val_metrics['f1_score']:.4f}")
            print(f"  Val Sensitivity: {val_metrics['sensitivity']:.4f}")
            print(f"  Val Specificity: {val_metrics['specificity']:.4f}")

            # Learning rate scheduling
            if self.scheduler is not None:
                if isinstance(
                    self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau
                ):
                    self.scheduler.step(val_metrics["loss"])
                else:
                    self.scheduler.step()

            # Save best model
            if val_metrics["auc_roc"] > self.best_val_auc:
                self.best_val_auc = val_metrics["auc_roc"]
                self.best_val_loss = val_metrics["loss"]
                self._save_checkpoint("best_model.pth", val_metrics)
                print(f"  ✓ New best model saved (AUC: {self.best_val_auc:.4f})")
                self.patience_counter = 0
            else:
                self.patience_counter += 1

            # Save last model
            self._save_checkpoint("last_model.pth", val_metrics)

            # Early stopping
            if self.patience_counter >= self.early_stopping_patience:
                print(f"\nEarly stopping triggered after {epoch + 1} epochs")
                break

        # Close writer
        self.writer.close()

        print("\nTraining completed!")
        print(f"Best validation AUC-ROC: {self.best_val_auc:.4f}")
        print(f"Best validation loss: {self.best_val_loss:.4f}")

        return {
            "best_val_auc": self.best_val_auc,
            "best_val_loss": self.best_val_loss,
        }

    def load_checkpoint(self, checkpoint_path: str | Path) -> None:
        """
        Load model checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scaler.load_state_dict(checkpoint["scaler_state_dict"])

        if self.scheduler is not None and "scheduler_state_dict" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        self.current_epoch = checkpoint["epoch"]
        self.best_val_loss = checkpoint.get("best_val_loss", float("inf"))
        self.best_val_auc = checkpoint.get("best_val_auc", 0.0)

        print(f"Loaded checkpoint from epoch {self.current_epoch}")
