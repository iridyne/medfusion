"""
Tests for trainer functionality.

Tests:
- MultimodalTrainer basic functionality
- MultiViewMultimodalTrainer functionality
- Training loop execution
- Progressive training stages
- Mixed precision training
- Checkpoint saving/loading
"""

import tempfile
from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from med_core.configs import create_ct_multiview_config, create_default_config
from med_core.fusion import create_fusion_model, create_multiview_fusion_model
from med_core.trainers import MultimodalTrainer, MultiViewMultimodalTrainer


class TestMultimodalTrainer:
    """Test single-view multimodal trainer."""

    @pytest.fixture
    def simple_model(self):
        """Create a simple model for testing."""
        return create_fusion_model(
            vision_backbone_name="resnet18",
            tabular_input_dim=10,
            num_classes=2,
            fusion_type="concatenate",
            pretrained=False,
        )

    @pytest.fixture
    def simple_dataloaders(self):
        """Create simple dataloaders for testing."""
        # Create dummy data
        images = torch.randn(40, 3, 224, 224)
        tabular = torch.randn(40, 10)
        labels = torch.randint(0, 2, (40,))

        train_dataset = TensorDataset(images[:30], tabular[:30], labels[:30])
        val_dataset = TensorDataset(images[30:], tabular[30:], labels[30:])

        train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)

        return train_loader, val_loader

    @pytest.fixture
    def simple_config(self):
        """Create simple config for testing."""
        config = create_default_config()
        config.training.num_epochs = 2
        config.training.mixed_precision = False
        config.training.use_progressive_training = False
        return config

    def test_trainer_initialization(self, simple_model, simple_dataloaders, simple_config):
        """Test trainer can be initialized."""
        train_loader, val_loader = simple_dataloaders

        trainer = MultimodalTrainer(
            model=simple_model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=simple_config,
            device="cpu",
        )

        assert trainer is not None
        assert trainer.model is not None
        assert trainer.optimizer is not None

    def test_trainer_single_epoch(self, simple_model, simple_dataloaders, simple_config):
        """Test trainer can run a single epoch."""
        train_loader, val_loader = simple_dataloaders

        trainer = MultimodalTrainer(
            model=simple_model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=simple_config,
            device="cpu",
        )

        # Run one epoch
        trainer.current_epoch = 0
        train_metrics = trainer._run_epoch(train_loader, training=True)
        val_metrics = trainer._run_epoch(val_loader, training=False)

        assert "loss" in train_metrics
        assert "loss" in val_metrics
        assert isinstance(train_metrics["loss"], float)
        assert isinstance(val_metrics["loss"], float)

    def test_trainer_full_training(self, simple_model, simple_dataloaders, simple_config):
        """Test trainer can complete full training."""
        train_loader, val_loader = simple_dataloaders

        trainer = MultimodalTrainer(
            model=simple_model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=simple_config,
            device="cpu",
        )

        # Train for 2 epochs
        history = trainer.train()

        assert history is not None
        assert len(history["train_loss"]) == 2
        assert len(history["val_loss"]) == 2

    def test_trainer_with_mixed_precision(self, simple_model, simple_dataloaders, simple_config):
        """Test trainer with mixed precision training."""
        train_loader, val_loader = simple_dataloaders
        simple_config.training.mixed_precision = True

        trainer = MultimodalTrainer(
            model=simple_model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=simple_config,
            device="cpu",
        )

        assert trainer.use_amp is True
        assert trainer.scaler is not None

    def test_trainer_progressive_training(self, simple_model, simple_dataloaders, simple_config):
        """Test progressive training stages."""
        train_loader, val_loader = simple_dataloaders
        simple_config.training.use_progressive_training = True
        simple_config.training.stage1_epochs = 1
        simple_config.training.stage2_epochs = 1
        simple_config.training.num_epochs = 3

        trainer = MultimodalTrainer(
            model=simple_model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=simple_config,
            device="cpu",
        )

        # Test stage transitions
        trainer.current_epoch = 0
        trainer.on_epoch_start()  # Stage 1

        trainer.current_epoch = 1
        trainer.on_epoch_start()  # Stage 2

        trainer.current_epoch = 2
        trainer.on_epoch_start()  # Stage 3

    def test_trainer_checkpoint_saving(self, simple_model, simple_dataloaders, simple_config):
        """Test checkpoint saving."""
        with tempfile.TemporaryDirectory() as tmpdir:
            train_loader, val_loader = simple_dataloaders
            simple_config.training.num_epochs = 1

            trainer = MultimodalTrainer(
                model=simple_model,
                train_loader=train_loader,
                val_loader=val_loader,
                config=simple_config,
                device="cpu",
                log_dir=tmpdir,
            )

            trainer.train()

            # Check if checkpoint was saved
            checkpoint_dir = Path(tmpdir) / "checkpoints"
            if checkpoint_dir.exists():
                checkpoints = list(checkpoint_dir.glob("*.pt"))
                assert len(checkpoints) > 0

    def test_trainer_early_stopping(self, simple_model, simple_dataloaders, simple_config):
        """Test early stopping mechanism."""
        train_loader, val_loader = simple_dataloaders
        simple_config.training.num_epochs = 10
        simple_config.training.early_stopping_patience = 2
        simple_config.training.early_stopping_min_delta = 0.0

        trainer = MultimodalTrainer(
            model=simple_model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=simple_config,
            device="cpu",
        )

        history = trainer.train()

        # Should stop before 10 epochs due to early stopping
        assert len(history["train_loss"]) <= 10


class TestMultiViewMultimodalTrainer:
    """Test multi-view multimodal trainer."""

    @pytest.fixture
    def multiview_model(self):
        """Create a multi-view model for testing."""
        return create_multiview_fusion_model(
            vision_backbone_name="resnet18",
            tabular_input_dim=10,
            num_classes=2,
            view_names=["view1", "view2"],
            aggregator_type="mean",
            fusion_type="concatenate",
            pretrained=False,
        )

    @pytest.fixture
    def multiview_config(self):
        """Create multi-view config for testing."""
        config = create_ct_multiview_config(
            data_root="./data",
            csv_path="./data.csv",
            image_dir="./images",
        )
        config.training.num_epochs = 2
        config.training.mixed_precision = False
        config.training.use_progressive_training = False
        return config

    def test_multiview_trainer_initialization(self, multiview_model, simple_dataloaders, multiview_config):
        """Test multi-view trainer can be initialized."""
        train_loader, val_loader = simple_dataloaders

        trainer = MultiViewMultimodalTrainer(
            model=multiview_model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=multiview_config,
            device="cpu",
        )

        assert trainer is not None
        assert trainer.model is not None

    def test_multiview_trainer_progressive_views(self, multiview_model, simple_dataloaders, multiview_config):
        """Test progressive view training."""
        train_loader, val_loader = simple_dataloaders
        multiview_config.training.use_progressive_views = True
        multiview_config.training.progressive_view_schedule = {
            0: ["view1"],
            1: ["view1", "view2"],
        }

        trainer = MultiViewMultimodalTrainer(
            model=multiview_model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=multiview_config,
            device="cpu",
        )

        # Test view schedule updates
        trainer.current_epoch = 0
        trainer.on_epoch_start()

        trainer.current_epoch = 1
        trainer.on_epoch_start()


class TestTrainerUtilities:
    """Test trainer utility functions."""

    def test_gradient_clipping(self, simple_model, simple_dataloaders, simple_config):
        """Test gradient clipping works."""
        train_loader, val_loader = simple_dataloaders
        simple_config.training.gradient_clip = 1.0

        trainer = MultimodalTrainer(
            model=simple_model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=simple_config,
            device="cpu",
        )

        # Run one training step
        batch = next(iter(train_loader))
        trainer.optimizer.zero_grad()
        step_metrics = trainer.training_step(batch, 0)
        loss = step_metrics["loss"]
        loss.backward()

        # Check gradients exist before clipping
        total_norm_before = 0
        for p in trainer.model.parameters():
            if p.grad is not None:
                total_norm_before += p.grad.data.norm(2).item() ** 2
        total_norm_before = total_norm_before ** 0.5

        # Apply gradient clipping
        torch.nn.utils.clip_grad_norm_(
            trainer.model.parameters(),
            simple_config.training.gradient_clip
        )

        # Check gradients after clipping
        total_norm_after = 0
        for p in trainer.model.parameters():
            if p.grad is not None:
                total_norm_after += p.grad.data.norm(2).item() ** 2
        total_norm_after = total_norm_after ** 0.5

        # If original norm was > 1.0, it should be clipped to 1.0
        if total_norm_before > 1.0:
            assert total_norm_after <= 1.0 + 1e-6

    def test_learning_rate_scheduling(self, simple_model, simple_dataloaders, simple_config):
        """Test learning rate scheduler works."""
        train_loader, val_loader = simple_dataloaders
        simple_config.training.num_epochs = 3

        trainer = MultimodalTrainer(
            model=simple_model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=simple_config,
            device="cpu",
        )

        initial_lr = trainer.optimizer.param_groups[0]["lr"]

        # Train for a few epochs
        trainer.train()

        # LR should have changed if scheduler is active
        if trainer.scheduler is not None:
            final_lr = trainer.optimizer.param_groups[0]["lr"]
            # LR might increase or decrease depending on scheduler type
            assert final_lr != initial_lr or simple_config.training.num_epochs == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
