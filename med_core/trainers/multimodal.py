"""
Multimodal trainer implementation.

Handles training of models with both vision and tabular inputs.
Supports mixed precision training and progressive freezing strategies.
"""

import logging
from typing import Any

import torch
from torch import nn
from torch.amp import GradScaler, autocast

from med_core.trainers.base import BaseTrainer

logger = logging.getLogger(__name__)


class MultimodalTrainer(BaseTrainer):
    """
    Trainer for multimodal medical models.

    Specializes BaseTrainer to handle:
    - Tuple inputs (image, tabular, label)
    - Mixed precision training
    - Progressive training stages (freezing/unfreezing backbones)
    - Dictionary outputs from MultimodalFusionModel
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        # Filter out parameters that should come from config
        kwargs.pop("log_dir", None)
        super().__init__(*args, **kwargs)

        # Mixed precision scaler
        self.use_amp = self.config.training.mixed_precision
        self.amp_device = "cuda" if self.device.type == "cuda" else "cpu"
        self.scaler = GradScaler(self.amp_device, enabled=self.use_amp)

        # Loss function
        class_weights = self.config.training.class_weights
        if class_weights:
            weight = torch.tensor(class_weights, device=self.device)
        else:
            weight = None

        self.criterion = nn.CrossEntropyLoss(
            weight=weight,
            label_smoothing=self.config.training.label_smoothing,
        )

        # Attention supervision settings
        self.use_attention_supervision = self.config.training.use_attention_supervision
        self.attention_loss_weight = self.config.training.attention_loss_weight
        self.attention_supervision_method = (
            self.config.training.attention_supervision_method
        )

        # Validate attention supervision configuration
        if self.use_attention_supervision:
            if not self.config.model.vision.enable_attention_supervision:
                logger.warning(
                    "use_attention_supervision=True but vision.enable_attention_supervision=False. "
                    "Attention supervision will not work. Set vision.enable_attention_supervision=True.",
                )
            if self.config.model.vision.attention_type != "cbam":
                logger.warning(
                    f"Attention supervision only works with CBAM, but attention_type={self.config.model.vision.attention_type}. "
                    "Attention supervision will be disabled.",
                )
                self.use_attention_supervision = False

    def on_epoch_start(self) -> None:
        """Handle progressive training stages."""
        super().on_epoch_start()

        if not self.config.training.use_progressive_training:
            return

        epoch = self.current_epoch
        stage1 = self.config.training.stage1_epochs
        stage2 = self.config.training.stage2_epochs

        # Determine current stage
        if epoch < stage1:
            self._set_stage_1()
        elif epoch < stage1 + stage2:
            self._set_stage_2()
        else:
            self._set_stage_3()

    def _set_stage_1(self) -> None:
        """Stage 1: Train vision backbone (or fusion), freeze tabular."""
        # This implementation depends on specific strategy requirements.
        # Here we implement a common strategy: Warmup vision, freeze tabular.
        # Alternatively: Train only fusion, freeze backbones.
        # Following the user's demo reference: "Stage 1: Freeze tabular stream, train image stream"

        # Access backbones via model attributes (assuming MultiModalFusionModel structure)
        if hasattr(self.model, "tabular_backbone"):
            for param in self.model.tabular_backbone.parameters():
                param.requires_grad = False

        if hasattr(self.model, "vision_backbone"):
            for param in self.model.vision_backbone.parameters():
                param.requires_grad = True

    def _set_stage_2(self) -> None:
        """Stage 2: Full model fine-tuning."""
        for param in self.model.parameters():
            param.requires_grad = True

    def _set_stage_3(self) -> None:
        """Stage 3: Fine-tune fusion layer only."""
        # Freeze backbones
        if hasattr(self.model, "vision_backbone"):
            for param in self.model.vision_backbone.parameters():
                param.requires_grad = False

        if hasattr(self.model, "tabular_backbone"):
            for param in self.model.tabular_backbone.parameters():
                param.requires_grad = False

        # Ensure fusion module and classifiers are trainable
        if hasattr(self.model, "fusion_module"):
            for param in self.model.fusion_module.parameters():
                param.requires_grad = True

        if hasattr(self.model, "classifier"):
            for param in self.model.classifier.parameters():
                param.requires_grad = True

    def _run_epoch(self, loader: Any, training: bool = True) -> dict[str, float]:
        """
        Run a single epoch (Overridden for AMP support).
        """
        metrics_sum = {}
        num_batches = len(loader)

        from tqdm import tqdm

        pbar = tqdm(
            loader,
            desc=f"Epoch {self.current_epoch} [{'Train' if training else 'Val'}]",
            leave=False,
        )

        for batch_idx, batch in enumerate(pbar):
            if training:
                self.optimizer.zero_grad()

                # Custom training step logic to handle forward + backward + scaler
                step_metrics = self.training_step(batch, batch_idx)
                loss = step_metrics["loss"]

                # Backward with scaler
                if self.use_amp:
                    self.scaler.scale(loss).backward()

                    if self.config.training.gradient_clip:
                        self.scaler.unscale_(self.optimizer)
                        nn.utils.clip_grad_norm_(
                            self.model.parameters(), self.config.training.gradient_clip,
                        )

                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    loss.backward()
                    if self.config.training.gradient_clip:
                        nn.utils.clip_grad_norm_(
                            self.model.parameters(), self.config.training.gradient_clip,
                        )
                    self.optimizer.step()

                self.global_step += 1

            else:
                # Validation
                step_metrics = self.validation_step(batch, batch_idx)

            # Aggregate metrics
            for k, v in step_metrics.items():
                val = v.item() if isinstance(v, torch.Tensor) else v
                # Only aggregate scalar metrics
                if isinstance(val, (int, float)):
                    metrics_sum[k] = metrics_sum.get(k, 0.0) + val

            # Update progress bar
            pbar.set_postfix(
                {k: f"{v.item():.4f}" for k, v in step_metrics.items() if k == "loss"},
            )

        return {k: v / num_batches for k, v in metrics_sum.items()}

    def training_step(self, batch: Any, batch_idx: int) -> dict[str, torch.Tensor]:
        """
        Perform forward pass with optional attention supervision.
        Note: Backward pass is handled in _run_epoch for AMP support.
        """
        # Unpack batch (may include masks for attention supervision)
        if len(batch) == 3:
            images, tabular, labels = batch
            masks = None
        elif len(batch) == 4:
            images, tabular, labels, masks = batch
            masks = masks.to(self.device) if masks is not None else None
        else:
            raise ValueError(f"Expected batch with 3 or 4 elements, got {len(batch)}")

        images = images.to(self.device)
        tabular = tabular.to(self.device)
        labels = labels.to(self.device)

        with autocast(self.amp_device, enabled=self.use_amp):
            # Forward pass with optional intermediate outputs
            if (
                self.use_attention_supervision
                and self.config.model.vision.enable_attention_supervision
            ):
                # Need to get intermediate outputs from vision backbone
                # This requires the model to support return_intermediates
                outputs = self._forward_with_attention(images, tabular)
            else:
                outputs = self.model(images, tabular)

            if isinstance(outputs, dict):
                logits = outputs["logits"]
                aux_outputs = outputs
            else:
                logits = outputs
                aux_outputs = {}

            # Classification loss
            loss = self.criterion(logits, labels)

            # Auxiliary head losses
            if self.config.model.use_auxiliary_heads:
                if "vision_logits" in aux_outputs:
                    loss += 0.3 * self.criterion(aux_outputs["vision_logits"], labels)
                if "tabular_logits" in aux_outputs:
                    loss += 0.3 * self.criterion(aux_outputs["tabular_logits"], labels)

            # Attention supervision loss
            if self.use_attention_supervision and "attention_weights" in aux_outputs:
                attention_loss = self._compute_attention_loss(
                    aux_outputs["attention_weights"],
                    aux_outputs.get("feature_maps"),
                    labels,
                    masks,
                )
                if attention_loss is not None:
                    loss += self.attention_loss_weight * attention_loss

        return {"loss": loss}

    def _forward_with_attention(
        self, images: torch.Tensor, tabular: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        Forward pass that extracts attention weights from vision backbone.

        This method handles models that may not directly support return_intermediates.
        It attempts to extract intermediate outputs from the vision backbone.
        """
        # Check if model has vision_backbone attribute
        if not hasattr(self.model, "vision_backbone"):
            logger.warning(
                "Model does not have vision_backbone attribute. Falling back to standard forward.",
            )
            return self.model(images, tabular)

        # Extract vision features with intermediates
        vision_outputs = self.model.vision_backbone(images, return_intermediates=True)

        if isinstance(vision_outputs, dict):
            vision_features = vision_outputs["features"]
            feature_maps = vision_outputs.get("feature_maps")
            attention_weights = vision_outputs.get("attention_weights")
        else:
            # Backbone doesn't support return_intermediates
            vision_features = vision_outputs
            feature_maps = None
            attention_weights = None

        # Extract tabular features
        if hasattr(self.model, "tabular_backbone"):
            tabular_features = self.model.tabular_backbone(tabular)
        else:
            tabular_features = tabular

        # Fusion
        if hasattr(self.model, "fusion_module"):
            fused_features, fusion_aux = self.model.fusion_module(
                vision_features, tabular_features,
            )
        else:
            fused_features = torch.cat([vision_features, tabular_features], dim=1)
            fusion_aux = {}  # noqa: F841

        # Classification
        if hasattr(self.model, "classifier"):
            logits = self.model.classifier(fused_features)
        else:
            logits = fused_features

        # Build output dictionary
        outputs = {
            "logits": logits,
            "feature_maps": feature_maps,
            "attention_weights": attention_weights,
        }

        # Add auxiliary outputs if available
        if (
            hasattr(self.model, "vision_classifier")
            and self.config.model.use_auxiliary_heads
        ):
            outputs["vision_logits"] = self.model.vision_classifier(vision_features)
        if (
            hasattr(self.model, "tabular_classifier")
            and self.config.model.use_auxiliary_heads
        ):
            outputs["tabular_logits"] = self.model.tabular_classifier(tabular_features)

        return outputs

    def _compute_attention_loss(
        self,
        attention_weights: torch.Tensor | None,
        feature_maps: torch.Tensor | None,
        labels: torch.Tensor,
        masks: torch.Tensor | None = None,
    ) -> torch.Tensor | None:
        """
        Compute attention supervision loss.

        Args:
            attention_weights: Spatial attention weights from CBAM (B, 1, H, W)
            feature_maps: Feature maps before pooling (B, C, H, W)
            labels: Ground truth labels (B,)
            masks: Optional ground truth masks (B, 1, H_mask, W_mask)

        Returns:
            Attention loss or None if cannot compute
        """
        if self.attention_supervision_method == "none":
            return None

        if self.attention_supervision_method == "mask":
            # Mask-based supervision: requires ground truth masks
            if masks is None or attention_weights is None:
                logger.warning(
                    "Mask-based attention supervision requires masks in dataset. Skipping.",
                )
                return None

            # Resize masks to match attention_weights size
            if masks.shape[-2:] != attention_weights.shape[-2:]:
                masks = torch.nn.functional.interpolate(
                    masks.float(),
                    size=attention_weights.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )

            # Binary cross-entropy loss between attention and masks
            loss = torch.nn.functional.binary_cross_entropy(
                attention_weights, masks, reduction="mean",
            )
            return loss

        if self.attention_supervision_method == "cam":
            # CAM-based supervision: generate CAM from feature_maps
            if feature_maps is None:
                logger.warning(
                    "CAM-based attention supervision requires feature_maps. Skipping.",
                )
                return None

            if not hasattr(self.model, "classifier") or not hasattr(
                self.model.classifier, "weight",
            ):
                logger.warning(
                    "CAM requires classifier with weight attribute. Skipping.",
                )
                return None

            # Generate CAM
            cam = self._generate_cam(feature_maps, labels)

            if cam is None or attention_weights is None:
                return None

            # Resize CAM to match attention_weights size
            if cam.shape[-2:] != attention_weights.shape[-2:]:
                cam = torch.nn.functional.interpolate(
                    cam,
                    size=attention_weights.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )

            # MSE loss between attention and CAM
            loss = torch.nn.functional.mse_loss(
                attention_weights, cam, reduction="mean",
            )
            return loss

        return None

    def _generate_cam(
        self, feature_maps: torch.Tensor, labels: torch.Tensor,
    ) -> torch.Tensor | None:
        """
        Generate Class Activation Map (CAM) from feature maps.

        Args:
            feature_maps: Feature maps (B, C, H, W)
            labels: Ground truth labels (B,)

        Returns:
            CAM (B, 1, H, W) or None if cannot generate
        """
        try:
            # Get classifier weights for the predicted classes
            classifier_weight = (
                self.model.classifier.weight
            )  # (num_classes, feature_dim)

            # Note: feature_maps are before pooling, but classifier expects pooled features
            # We need to match dimensions. Typically, classifier input = pooled feature_maps
            # So we need to project feature_maps to match classifier input dimension

            # Get the weight corresponding to ground truth labels
            batch_size = feature_maps.size(0)
            class_weights = classifier_weight[labels]  # (B, feature_dim)

            # Global average pooling to get per-channel importance
            # feature_maps: (B, C, H, W)
            # We want to compute: CAM = sum_c (w_c * feature_map_c)
            # But classifier expects pooled features, so dimensions might not match

            # Simple approach: use global average pooling on feature_maps
            pooled_features = torch.nn.functional.adaptive_avg_pool2d(
                feature_maps, 1,
            )  # (B, C, 1, 1)
            pooled_features = pooled_features.view(batch_size, -1)  # (B, C)

            # Check dimension match
            if pooled_features.size(1) != class_weights.size(1):
                logger.warning(
                    f"Dimension mismatch: pooled_features {pooled_features.shape} vs "
                    f"class_weights {class_weights.shape}. Cannot generate CAM.",
                )
                return None

            # Compute CAM: weighted sum of feature maps
            # class_weights: (B, C)
            # feature_maps: (B, C, H, W)
            # CAM: (B, 1, H, W)
            cam = torch.einsum("bc,bchw->bhw", class_weights, feature_maps)  # (B, H, W)
            cam = cam.unsqueeze(1)  # (B, 1, H, W)

            # Apply ReLU (only positive contributions)
            cam = torch.nn.functional.relu(cam)

            # Normalize to [0, 1]
            batch_size = cam.size(0)
            for i in range(batch_size):
                cam_min = cam[i].min()
                cam_max = cam[i].max()
                if cam_max > cam_min:
                    cam[i] = (cam[i] - cam_min) / (cam_max - cam_min + 1e-8)

            return cam

        except Exception as e:
            logger.warning(f"Failed to generate CAM: {e}")
            return None

    def validation_step(self, batch: Any, batch_idx: int) -> dict[str, torch.Tensor]:
        """
        Perform a single validation step.

        Args:
            batch: Tuple of (images, tabular, labels)
            batch_idx: Batch index

        Returns:
            Dictionary with loss and accuracy
        """
        images, tabular, labels = batch
        images = images.to(self.device)
        tabular = tabular.to(self.device)
        labels = labels.to(self.device)

        with autocast(self.amp_device, enabled=self.use_amp):
            outputs = self.model(images, tabular)

            logits = outputs["logits"] if isinstance(outputs, dict) else outputs

            loss = self.criterion(logits, labels)

        # Calculate accuracy
        preds = torch.argmax(logits, dim=1)
        acc = (preds == labels).float().mean()

        return {"loss": loss, "accuracy": acc}


def create_trainer(
    model: nn.Module,
    train_loader: Any,
    val_loader: Any,
    config: Any,
    device: str | torch.device = "cuda",
    log_dir: str | None = None,
) -> MultimodalTrainer:
    """
    Factory function to create a multimodal trainer.

    This is a convenience function that creates a trainer with standard configuration.

    Args:
        model: The model to train (typically MultiModalFusionModel)
        train_loader: Training data loader
        val_loader: Validation data loader
        config: Training configuration (ExperimentConfig)
        device: Device to use for training ("cuda" or "cpu")
        log_dir: Optional directory for logging

    Returns:
        Configured MultimodalTrainer ready for training

    Example:
        >>> from med_core.fusion import create_fusion_model
        >>> from med_core.trainers import create_trainer
        >>> from med_core.configs import create_default_config
        >>>
        >>> model = create_fusion_model(
        ...     vision_backbone_name="resnet18",
        ...     tabular_input_dim=10,
        ...     num_classes=2,
        ... )
        >>> config = create_default_config()
        >>> trainer = create_trainer(model, train_loader, val_loader, config)
        >>> trainer.train()
    """
    trainer = MultimodalTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=device,
        log_dir=log_dir,
    )

    return trainer
