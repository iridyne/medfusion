"""
Unit tests for Survival Analysis Heads.
"""

import pytest
import torch

from med_core.heads.survival import (
    CoxSurvivalHead,
    DeepSurvivalHead,
    DiscreteTimeSurvivalHead,
    MultiTaskSurvivalHead,
    RankingSurvivalHead,
)


class TestCoxSurvivalHead:
    """Test suite for CoxSurvivalHead."""

    def test_basic_forward(self):
        """Test basic forward pass."""
        head = CoxSurvivalHead(input_dim=512)
        features = torch.randn(8, 512)

        hazard = head(features)

        assert hazard.shape == (8, 1)

    def test_with_hidden_layers(self):
        """Test with hidden layers."""
        head = CoxSurvivalHead(input_dim=512, hidden_dims=[256, 128])
        features = torch.randn(8, 512)

        hazard = head(features)

        assert hazard.shape == (8, 1)

    def test_with_batch_norm(self):
        """Test with batch normalization."""
        head = CoxSurvivalHead(input_dim=512, use_batch_norm=True)
        features = torch.randn(8, 512)

        hazard = head(features)

        assert hazard.shape == (8, 1)

    def test_gradient_flow(self):
        """Test gradient flow."""
        head = CoxSurvivalHead(input_dim=128)
        features = torch.randn(4, 128, requires_grad=True)

        hazard = head(features)
        loss = hazard.sum()
        loss.backward()

        assert features.grad is not None

    def test_batch_size_one(self):
        """Test with batch size of 1."""
        head = CoxSurvivalHead(input_dim=512)
        features = torch.randn(1, 512)

        hazard = head(features)

        assert hazard.shape == (1, 1)


class TestDiscreteTimeSurvivalHead:
    """Test suite for DiscreteTimeSurvivalHead."""

    def test_basic_forward(self):
        """Test basic forward pass."""
        head = DiscreteTimeSurvivalHead(input_dim=512, num_time_bins=10)
        features = torch.randn(8, 512)

        hazards = head(features)

        assert hazards.shape == (8, 10)

    def test_predict_survival(self):
        """Test survival probability prediction."""
        head = DiscreteTimeSurvivalHead(input_dim=512, num_time_bins=10)
        features = torch.randn(8, 512)

        survival_probs = head.predict_survival(features)

        assert survival_probs.shape == (8, 10)
        # Survival probabilities should be in [0, 1]
        assert (survival_probs >= 0).all() and (survival_probs <= 1).all()
        # Survival probabilities should be non-increasing
        for i in range(8):
            for t in range(9):
                assert survival_probs[i, t] >= survival_probs[i, t + 1]

    def test_predict_risk_scores(self):
        """Test risk score prediction."""
        head = DiscreteTimeSurvivalHead(input_dim=512, num_time_bins=10)
        features = torch.randn(8, 512)

        risk_scores = head.predict_risk_scores(features)

        assert risk_scores.shape == (8,)
        assert (risk_scores >= 0).all()

    def test_different_num_bins(self):
        """Test with different number of time bins."""
        for num_bins in [5, 10, 20]:
            head = DiscreteTimeSurvivalHead(input_dim=512, num_time_bins=num_bins)
            features = torch.randn(4, 512)

            hazards = head(features)
            survival_probs = head.predict_survival(features)

            assert hazards.shape == (4, num_bins)
            assert survival_probs.shape == (4, num_bins)

    def test_gradient_flow(self):
        """Test gradient flow."""
        head = DiscreteTimeSurvivalHead(input_dim=128, num_time_bins=10)
        features = torch.randn(4, 128, requires_grad=True)

        hazards = head(features)
        loss = hazards.sum()
        loss.backward()

        assert features.grad is not None


class TestDeepSurvivalHead:
    """Test suite for DeepSurvivalHead."""

    def test_basic_forward(self):
        """Test basic forward pass."""
        head = DeepSurvivalHead(input_dim=512, num_time_bins=10)
        features = torch.randn(8, 512)

        hazards = head(features)

        assert hazards.shape == (8, 10)

    def test_predict_survival(self):
        """Test survival probability prediction."""
        head = DeepSurvivalHead(input_dim=512, num_time_bins=10)
        features = torch.randn(8, 512)

        survival_probs = head.predict_survival(features)

        assert survival_probs.shape == (8, 10)
        assert (survival_probs >= 0).all() and (survival_probs <= 1).all()

    def test_different_hidden_dim(self):
        """Test with different hidden dimensions."""
        for hidden_dim in [128, 256, 512]:
            head = DeepSurvivalHead(
                input_dim=512, num_time_bins=10, hidden_dim=hidden_dim
            )
            features = torch.randn(4, 512)

            hazards = head(features)

            assert hazards.shape == (4, 10)

    def test_different_num_layers(self):
        """Test with different number of LSTM layers."""
        for num_layers in [1, 2, 3]:
            head = DeepSurvivalHead(
                input_dim=512, num_time_bins=10, num_layers=num_layers
            )
            features = torch.randn(4, 512)

            hazards = head(features)

            assert hazards.shape == (4, 10)

    def test_gradient_flow(self):
        """Test gradient flow."""
        head = DeepSurvivalHead(input_dim=128, num_time_bins=10)
        features = torch.randn(4, 128, requires_grad=True)

        hazards = head(features)
        loss = hazards.sum()
        loss.backward()

        assert features.grad is not None


class TestMultiTaskSurvivalHead:
    """Test suite for MultiTaskSurvivalHead."""

    def test_basic_forward(self):
        """Test basic forward pass."""
        head = MultiTaskSurvivalHead(input_dim=512, num_classes=4, num_time_bins=10)
        features = torch.randn(8, 512)

        class_logits, survival_hazards = head(features)

        assert class_logits.shape == (8, 4)
        assert survival_hazards.shape == (8, 10)

    def test_predict_survival(self):
        """Test survival probability prediction."""
        head = MultiTaskSurvivalHead(input_dim=512, num_classes=4, num_time_bins=10)
        features = torch.randn(8, 512)

        survival_probs = head.predict_survival(features)

        assert survival_probs.shape == (8, 10)
        assert (survival_probs >= 0).all() and (survival_probs <= 1).all()

    def test_different_num_classes(self):
        """Test with different number of classes."""
        for num_classes in [2, 4, 8]:
            head = MultiTaskSurvivalHead(
                input_dim=512, num_classes=num_classes, num_time_bins=10
            )
            features = torch.randn(4, 512)

            class_logits, survival_hazards = head(features)

            assert class_logits.shape == (4, num_classes)
            assert survival_hazards.shape == (4, 10)

    def test_gradient_flow(self):
        """Test gradient flow."""
        head = MultiTaskSurvivalHead(input_dim=128, num_classes=4, num_time_bins=10)
        features = torch.randn(4, 128, requires_grad=True)

        class_logits, survival_hazards = head(features)
        loss = class_logits.sum() + survival_hazards.sum()
        loss.backward()

        assert features.grad is not None


class TestRankingSurvivalHead:
    """Test suite for RankingSurvivalHead."""

    def test_basic_forward(self):
        """Test basic forward pass."""
        head = RankingSurvivalHead(input_dim=512)
        features = torch.randn(8, 512)

        risk_scores = head(features)

        assert risk_scores.shape == (8, 1)

    def test_ranking_loss(self):
        """Test ranking loss computation."""
        head = RankingSurvivalHead(input_dim=512)
        features = torch.randn(8, 512)

        # Create dummy survival data
        survival_times = torch.tensor([10.0, 20.0, 15.0, 30.0, 5.0, 25.0, 12.0, 18.0])
        events = torch.tensor([1, 1, 0, 1, 1, 0, 1, 1])

        loss = head.compute_ranking_loss(features, survival_times, events)

        assert loss.ndim == 0  # Scalar
        assert loss >= 0  # Loss should be non-negative

    def test_ranking_loss_gradient(self):
        """Test gradient flow through ranking loss."""
        head = RankingSurvivalHead(input_dim=128)
        features = torch.randn(4, 128, requires_grad=True)

        survival_times = torch.tensor([10.0, 20.0, 15.0, 30.0])
        events = torch.tensor([1, 1, 1, 1])

        loss = head.compute_ranking_loss(features, survival_times, events)
        loss.backward()

        assert features.grad is not None

    def test_ranking_consistency(self):
        """Test that ranking is consistent with survival times."""
        head = RankingSurvivalHead(input_dim=512)

        # Create features that should produce consistent rankings
        features = torch.randn(5, 512)
        survival_times = torch.tensor([5.0, 10.0, 15.0, 20.0, 25.0])
        events = torch.ones(5)

        # Train for a few steps
        optimizer = torch.optim.Adam(head.parameters(), lr=0.01)

        for _ in range(10):
            optimizer.zero_grad()
            loss = head.compute_ranking_loss(features, survival_times, events)
            loss.backward()
            optimizer.step()

        # Check that risk scores are inversely related to survival times
        with torch.no_grad():
            risk_scores = head(features).squeeze()

        # Patients with shorter survival should have higher risk
        assert risk_scores[0] > risk_scores[-1]

    def test_different_hidden_dims(self):
        """Test with different hidden dimensions."""
        for hidden_dims in [[256], [256, 128], [512, 256, 128]]:
            head = RankingSurvivalHead(input_dim=512, hidden_dims=hidden_dims)
            features = torch.randn(4, 512)

            risk_scores = head(features)

            assert risk_scores.shape == (4, 1)

    def test_gradient_flow(self):
        """Test gradient flow."""
        head = RankingSurvivalHead(input_dim=128)
        features = torch.randn(4, 128, requires_grad=True)

        risk_scores = head(features)
        loss = risk_scores.sum()
        loss.backward()

        assert features.grad is not None


class TestIntegration:
    """Integration tests for survival heads."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda_forward(self):
        """Test forward pass on CUDA."""
        head = CoxSurvivalHead(input_dim=512).cuda()
        features = torch.randn(8, 512).cuda()

        hazard = head(features)

        assert hazard.shape == (8, 1)
        assert hazard.device.type == "cuda"

    def test_training_vs_eval_mode(self):
        """Test behavior in training vs eval mode."""
        head = DiscreteTimeSurvivalHead(input_dim=512, num_time_bins=10, dropout=0.5)
        features = torch.randn(8, 512)

        # Training mode
        head.train()
        _hazards_train = head(features)

        # Eval mode
        head.eval()
        with torch.no_grad():
            hazards_eval1 = head(features)
            hazards_eval2 = head(features)

        # In eval mode, outputs should be deterministic
        assert torch.allclose(hazards_eval1, hazards_eval2)

    def test_combined_classification_survival(self):
        """Test using multi-task head for combined prediction."""
        head = MultiTaskSurvivalHead(input_dim=512, num_classes=4, num_time_bins=10)
        features = torch.randn(8, 512)

        # Get both predictions
        class_logits, survival_hazards = head(features)

        # Classification loss
        class_labels = torch.randint(0, 4, (8,))
        class_loss = torch.nn.functional.cross_entropy(class_logits, class_labels)

        # Survival loss (negative log-likelihood)
        survival_probs = head.predict_survival(features)
        time_bins = torch.randint(0, 10, (8,))
        events = torch.randint(0, 2, (8,))

        # Simple survival loss
        survival_loss = 0.0
        for i in range(8):
            if events[i] == 1:
                # Event occurred
                t = time_bins[i]
                if t > 0:
                    survival_loss -= torch.log(survival_probs[i, t - 1] + 1e-8)
                    survival_loss += torch.log(survival_probs[i, t] + 1e-8)
            else:
                # Censored
                t = time_bins[i]
                survival_loss -= torch.log(survival_probs[i, t] + 1e-8)

        survival_loss = survival_loss / 8

        # Combined loss
        total_loss = class_loss + survival_loss

        assert total_loss.requires_grad

    def test_survival_probability_properties(self):
        """Test mathematical properties of survival probabilities."""
        head = DiscreteTimeSurvivalHead(input_dim=512, num_time_bins=10)
        features = torch.randn(8, 512)

        survival_probs = head.predict_survival(features)

        # Property 1: S(t) should be non-increasing
        for i in range(8):
            for t in range(9):
                assert survival_probs[i, t] >= survival_probs[i, t + 1] - 1e-6

        # Property 2: S(t) should be in [0, 1]
        assert (survival_probs >= 0).all()
        assert (survival_probs <= 1).all()

        # Property 3: S(0) >= S(T) (first time bin has higher survival than last)
        assert (survival_probs[:, 0] >= survival_probs[:, -1]).all()
