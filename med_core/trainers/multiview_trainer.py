"""
Multi-view multimodal trainer implementation.

Extends MultimodalTrainer to handle multi-view vision inputs.
"""

import logging
from typing import Any

import torch
import torch.nn as nn
from torch.amp import autocast

from med_core.datasets.multiview_types import ViewTensor
from med_core.trainers.multimodal import MultimodalTrainer

logger = logging.getLogger(__name__)


class MultiViewMultimodalTrainer(MultimodalTrainer):
    """
    Trainer for multi-view multimodal medical models.

    Extends MultimodalTrainer to handle:
    - Multi-view image inputs (dict or stacked tensor)
    - View-level auxiliary losses
    - View attention weight logging
    - Progressive view training (optional)

    The trainer automatically detects multi-view inputs and handles them appropriately.
    Single-view inputs are still supported for backward compatibility.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

        # Track if we're using multi-view
        self.is_multiview = getattr(self.config.data, "enable_multiview", False)

        # Progressive view training settings
        self.use_progressive_views = getattr(
            self.config.model.vision, "use_progressive_view_training", False
        )
        if self.use_progressive_views:
            self.initial_views = self.config.model.vision.initial_views
            self.add_views_every_n_epochs = (
                self.config.model.vision.add_views_every_n_epochs
            )
            self.all_view_names = self.config.data.view_names
            self.current_active_views = set(self.initial_views)

    def on_epoch_start(self) -> None:
        """Handle progressive training stages and progressive view training."""
        super().on_epoch_start()

        # Progressive view training
        if self.use_progressive_views:
            self._update_active_views()

    def _update_active_views(self) -> None:
        """Update active views based on current epoch."""
        epoch = self.current_epoch

        # Calculate how many views to activate
        views_to_add = epoch // self.add_views_every_n_epochs
        target_num_views = min(
            len(self.initial_views) + views_to_add, len(self.all_view_names)
        )

        # Add views progressively
        if len(self.current_active_views) < target_num_views:
            remaining_views = [
                v for v in self.all_view_names if v not in self.current_active_views
            ]
            views_to_activate = remaining_views[
                : target_num_views - len(self.current_active_views)
            ]
            self.current_active_views.update(views_to_activate)

            logger.info(
                f"Epoch {epoch}: Activated views {views_to_activate}. "
                f"Active views: {sorted(self.current_active_views)}"
            )

    def _filter_views(self, images: ViewTensor) -> ViewTensor:
        """Filter images to only include active views (for progressive training)."""
        if not self.use_progressive_views:
            return images

        if isinstance(images, dict):
            return {
                view_name: view_tensor
                for view_name, view_tensor in images.items()
                if view_name in self.current_active_views
            }
        else:
            # For stacked tensor, we can't easily filter, so return as-is
            return images

    def training_step(self, batch: Any, batch_idx: int) -> dict[str, torch.Tensor]:
        """
        Perform forward pass with multi-view support.

        Args:
            batch: Can be:
                - (images, tabular, labels) where images is dict or tensor
                - (images, tabular, labels, view_mask) with optional view mask

        Returns:
            Dictionary with loss and optional auxiliary losses
        """
        # Unpack batch
        if len(batch) == 4:
            images, tabular, labels, view_mask = batch
            view_mask = view_mask.to(self.device) if view_mask is not None else None
        else:
            images, tabular, labels = batch
            view_mask = None

        # Move to device
        if isinstance(images, dict):
            images = {k: v.to(self.device) for k, v in images.items()}
            # Apply progressive view filtering
            images = self._filter_views(images)
        else:
            images = images.to(self.device)

        tabular = tabular.to(self.device)
        labels = labels.to(self.device)

        with autocast(self.amp_device, enabled=self.use_amp):
            # Forward pass
            if view_mask is not None:
                outputs = self.model(images, tabular, view_mask=view_mask)
            else:
                outputs = self.model(images, tabular)

            if isinstance(outputs, dict):
                logits = outputs["logits"]
                aux_outputs = outputs
            else:
                logits = outputs
                aux_outputs = {}

            # Main loss
            loss = self.criterion(logits, labels)

            # Auxiliary losses
            if self.config.model.use_auxiliary_heads:
                if "vision_logits" in aux_outputs:
                    loss += 0.3 * self.criterion(aux_outputs["vision_logits"], labels)
                if "tabular_logits" in aux_outputs:
                    loss += 0.3 * self.criterion(aux_outputs["tabular_logits"], labels)

            # View-specific auxiliary losses (if available)
            if "view_logits" in aux_outputs:
                for view_logits in aux_outputs["view_logits"].values():
                    loss += 0.1 * self.criterion(view_logits, labels)

        metrics = {"loss": loss}

        # Log view attention weights (for monitoring)
        if "view_aggregation_aux" in aux_outputs:
            view_aux = aux_outputs["view_aggregation_aux"]
            if "view_attention_weights" in view_aux:
                # Store for logging (will be averaged across batch)
                attn_weights = view_aux["view_attention_weights"]
                if attn_weights is not None:
                    metrics["view_attention_mean"] = attn_weights.mean()
                    metrics["view_attention_std"] = attn_weights.std()

        return metrics

    def validation_step(self, batch: Any, batch_idx: int) -> dict[str, torch.Tensor]:
        """
        Perform validation step with multi-view support.

        Args:
            batch: Can be:
                - (images, tabular, labels) where images is dict or tensor
                - (images, tabular, labels, view_mask) with optional view mask

        Returns:
            Dictionary with loss and accuracy
        """
        # Unpack batch
        if len(batch) == 4:
            images, tabular, labels, view_mask = batch
            view_mask = view_mask.to(self.device) if view_mask is not None else None
        else:
            images, tabular, labels = batch
            view_mask = None

        # Move to device
        if isinstance(images, dict):
            images = {k: v.to(self.device) for k, v in images.items()}
        else:
            images = images.to(self.device)

        tabular = tabular.to(self.device)
        labels = labels.to(self.device)

        with autocast(self.amp_device, enabled=self.use_amp):
            # Forward pass
            if view_mask is not None:
                outputs = self.model(images, tabular, view_mask=view_mask)
            else:
                outputs = self.model(images, tabular)

            if isinstance(outputs, dict):
                logits = outputs["logits"]
            else:
                logits = outputs

            loss = self.criterion(logits, labels)

        # Calculate accuracy
        preds = torch.argmax(logits, dim=1)
        acc = (preds == labels).float().mean()

        return {"loss": loss, "accuracy": acc}

    def on_epoch_end(
        self, train_metrics: dict[str, float], val_metrics: dict[str, float]
    ) -> None:
        """Log additional multi-view specific metrics."""
        super().on_epoch_end(train_metrics, val_metrics)

        # Log active views if using progressive training
        if self.use_progressive_views:
            logger.info(
                f"Epoch {self.current_epoch} completed with active views: "
                f"{sorted(self.current_active_views)}"
            )


def create_multiview_trainer(
    model: nn.Module,
    train_loader: Any,
    val_loader: Any,
    config: Any,
    device: torch.device | None = None,
) -> MultiViewMultimodalTrainer:
    """
    Factory function to create a multi-view multimodal trainer.

    Args:
        model: Multi-view multimodal model
        train_loader: Training data loader
        val_loader: Validation data loader
        config: Experiment configuration
        device: Device to use (auto-detected if None)

    Returns:
        Configured MultiViewMultimodalTrainer instance

    Example:
        >>> from med_core.fusion import create_multiview_fusion_model
        >>> from med_core.configs import create_ct_multiview_config
        >>>
        >>> config = create_ct_multiview_config()
        >>> model = create_multiview_fusion_model(
        ...     vision_backbone_name="resnet18",
        ...     tabular_input_dim=10,
        ...     num_classes=2,
        ... )
        >>>
        >>> trainer = create_multiview_trainer(
        ...     model=model,
        ...     train_loader=train_loader,
        ...     val_loader=val_loader,
        ...     config=config,
        ... )
        >>> trainer.train()
    """
    if device is None:
        device = torch.device(config.device)

    trainer = MultiViewMultimodalTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=device,
    )

    return trainer
