"""
Survival analysis heads for medical imaging tasks.

This module provides various survival analysis head implementations
for predicting patient survival outcomes, including Cox proportional
hazards models and discrete-time survival analysis.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class CoxSurvivalHead(nn.Module):
    """
    Cox proportional hazards model head for survival analysis.

    Predicts a single hazard score (log-hazard ratio) for each sample.
    The Cox model assumes proportional hazards over time.

    Args:
        input_dim: Input feature dimension
        hidden_dims: List of hidden layer dimensions
        dropout: Dropout rate
        use_batch_norm: Use batch normalization

    Example:
        >>> head = CoxSurvivalHead(input_dim=512, hidden_dims=[256])
        >>> features = torch.randn(8, 512)
        >>> hazard = head(features)  # [8, 1]
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: Optional[list[int]] = None,
        dropout: float = 0.5,
        use_batch_norm: bool = False,
    ):
        super().__init__()

        self.input_dim = input_dim

        if hidden_dims is None:
            hidden_dims = [256]

        # Build MLP layers
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))

            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))

            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim

        self.mlp = nn.Sequential(*layers) if layers else nn.Identity()

        # Output layer (single hazard score)
        self.hazard_predictor = nn.Linear(prev_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input features [B, input_dim]

        Returns:
            Hazard scores [B, 1]
        """
        x = self.mlp(x)
        hazard = self.hazard_predictor(x)
        return hazard


class DiscreteTimeSurvivalHead(nn.Module):
    """
    Discrete-time survival analysis head.

    Divides time into discrete intervals and predicts the probability
    of event occurrence in each interval.

    Args:
        input_dim: Input feature dimension
        num_time_bins: Number of discrete time intervals
        hidden_dims: List of hidden layer dimensions
        dropout: Dropout rate

    Example:
        >>> head = DiscreteTimeSurvivalHead(input_dim=512, num_time_bins=10)
        >>> features = torch.randn(8, 512)
        >>> hazards = head(features)  # [8, 10]
        >>> survival_probs = head.predict_survival(features)  # [8, 10]
    """

    def __init__(
        self,
        input_dim: int,
        num_time_bins: int,
        hidden_dims: Optional[list[int]] = None,
        dropout: float = 0.5,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.num_time_bins = num_time_bins

        if hidden_dims is None:
            hidden_dims = [256]

        # Shared feature extractor
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            ])
            prev_dim = hidden_dim

        self.feature_extractor = nn.Sequential(*layers)

        # Time-specific hazard predictors
        self.hazard_predictor = nn.Linear(prev_dim, num_time_bins)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input features [B, input_dim]

        Returns:
            Hazard logits for each time bin [B, num_time_bins]
        """
        x = self.feature_extractor(x)
        hazard_logits = self.hazard_predictor(x)
        return hazard_logits

    def predict_survival(self, x: torch.Tensor) -> torch.Tensor:
        """
        Predict survival probabilities for each time bin.

        Args:
            x: Input features [B, input_dim]

        Returns:
            Survival probabilities [B, num_time_bins]
        """
        hazard_logits = self.forward(x)
        hazards = torch.sigmoid(hazard_logits)  # P(event in interval t)

        # Survival probability: S(t) = prod(1 - h_i) for i <= t
        survival_probs = torch.cumprod(1 - hazards, dim=1)

        return survival_probs

    def predict_risk_scores(self, x: torch.Tensor) -> torch.Tensor:
        """
        Predict risk scores (sum of hazards).

        Args:
            x: Input features [B, input_dim]

        Returns:
            Risk scores [B]
        """
        hazard_logits = self.forward(x)
        hazards = torch.sigmoid(hazard_logits)
        risk_scores = hazards.sum(dim=1)
        return risk_scores


class DeepSurvivalHead(nn.Module):
    """
    Deep survival analysis head with time-dependent features.

    Uses a more sophisticated architecture that can model
    time-varying effects.

    Args:
        input_dim: Input feature dimension
        num_time_bins: Number of discrete time intervals
        hidden_dim: Hidden dimension
        num_layers: Number of recurrent layers
        dropout: Dropout rate

    Example:
        >>> head = DeepSurvivalHead(input_dim=512, num_time_bins=10)
        >>> features = torch.randn(8, 512)
        >>> hazards = head(features)  # [8, 10]
    """

    def __init__(
        self,
        input_dim: int,
        num_time_bins: int,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.num_time_bins = num_time_bins
        self.hidden_dim = hidden_dim

        # Feature projection
        self.feature_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

        # Time embedding
        self.time_embedding = nn.Embedding(num_time_bins, hidden_dim)

        # LSTM for temporal modeling
        self.lstm = nn.LSTM(
            input_size=hidden_dim * 2,  # feature + time embedding
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
        )

        # Hazard predictor
        self.hazard_predictor = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input features [B, input_dim]

        Returns:
            Hazard logits for each time bin [B, num_time_bins]
        """
        batch_size = x.size(0)

        # Project features
        features = self.feature_proj(x)  # [B, hidden_dim]

        # Expand features for all time bins
        features = features.unsqueeze(1).expand(-1, self.num_time_bins, -1)  # [B, T, hidden_dim]

        # Get time embeddings
        time_indices = torch.arange(self.num_time_bins, device=x.device)
        time_embeds = self.time_embedding(time_indices)  # [T, hidden_dim]
        time_embeds = time_embeds.unsqueeze(0).expand(batch_size, -1, -1)  # [B, T, hidden_dim]

        # Concatenate features and time embeddings
        combined = torch.cat([features, time_embeds], dim=2)  # [B, T, hidden_dim*2]

        # LSTM forward
        lstm_out, _ = self.lstm(combined)  # [B, T, hidden_dim]

        # Predict hazards
        hazards = self.hazard_predictor(lstm_out).squeeze(-1)  # [B, T]

        return hazards

    def predict_survival(self, x: torch.Tensor) -> torch.Tensor:
        """
        Predict survival probabilities.

        Args:
            x: Input features [B, input_dim]

        Returns:
            Survival probabilities [B, num_time_bins]
        """
        hazard_logits = self.forward(x)
        hazards = torch.sigmoid(hazard_logits)
        survival_probs = torch.cumprod(1 - hazards, dim=1)
        return survival_probs


class MultiTaskSurvivalHead(nn.Module):
    """
    Multi-task survival head that jointly predicts survival and classification.

    Useful for tasks where you want to predict both patient outcome
    (e.g., tumor grade) and survival time.

    Args:
        input_dim: Input feature dimension
        num_classes: Number of classification classes
        num_time_bins: Number of survival time bins
        hidden_dim: Hidden dimension
        dropout: Dropout rate

    Example:
        >>> head = MultiTaskSurvivalHead(
        ...     input_dim=512, num_classes=4, num_time_bins=10
        ... )
        >>> features = torch.randn(8, 512)
        >>> class_logits, survival_hazards = head(features)
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        num_time_bins: int,
        hidden_dim: int = 256,
        dropout: float = 0.5,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.num_classes = num_classes
        self.num_time_bins = num_time_bins

        # Shared feature extractor
        self.shared_features = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

        # Classification branch
        self.classification_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )

        # Survival branch
        self.survival_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_time_bins),
        )

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            x: Input features [B, input_dim]

        Returns:
            Tuple of (classification_logits, survival_hazards)
            - classification_logits: [B, num_classes]
            - survival_hazards: [B, num_time_bins]
        """
        # Shared features
        shared = self.shared_features(x)

        # Classification
        class_logits = self.classification_head(shared)

        # Survival
        survival_hazards = self.survival_head(shared)

        return class_logits, survival_hazards

    def predict_survival(self, x: torch.Tensor) -> torch.Tensor:
        """
        Predict survival probabilities.

        Args:
            x: Input features [B, input_dim]

        Returns:
            Survival probabilities [B, num_time_bins]
        """
        _, hazard_logits = self.forward(x)
        hazards = torch.sigmoid(hazard_logits)
        survival_probs = torch.cumprod(1 - hazards, dim=1)
        return survival_probs


class RankingSurvivalHead(nn.Module):
    """
    Ranking-based survival head using pairwise ranking loss.

    Learns to rank patients by their survival time without
    requiring exact time-to-event information.

    Args:
        input_dim: Input feature dimension
        hidden_dims: List of hidden layer dimensions
        dropout: Dropout rate

    Example:
        >>> head = RankingSurvivalHead(input_dim=512)
        >>> features = torch.randn(8, 512)
        >>> risk_scores = head(features)  # [8, 1]
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: Optional[list[int]] = None,
        dropout: float = 0.5,
    ):
        super().__init__()

        self.input_dim = input_dim

        if hidden_dims is None:
            hidden_dims = [256, 128]

        # Build MLP
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            ])
            prev_dim = hidden_dim

        self.feature_extractor = nn.Sequential(*layers)

        # Risk score predictor
        self.risk_predictor = nn.Linear(prev_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input features [B, input_dim]

        Returns:
            Risk scores [B, 1] (higher = higher risk)
        """
        x = self.feature_extractor(x)
        risk_scores = self.risk_predictor(x)
        return risk_scores

    def compute_ranking_loss(
        self,
        features: torch.Tensor,
        survival_times: torch.Tensor,
        events: torch.Tensor,
        margin: float = 1.0,
    ) -> torch.Tensor:
        """
        Compute pairwise ranking loss.

        Args:
            features: Input features [B, input_dim]
            survival_times: Survival times [B]
            events: Event indicators (1=event, 0=censored) [B]
            margin: Margin for ranking loss

        Returns:
            Ranking loss scalar
        """
        risk_scores = self.forward(features).squeeze(-1)  # [B]

        # Create pairwise comparisons
        # Patient i should have higher risk than patient j if:
        # - Both experienced event and t_i < t_j
        # - Patient i experienced event and patient j is censored with t_j > t_i
        batch_size = features.size(0)
        loss = 0.0
        num_pairs = 0

        for i in range(batch_size):
            for j in range(batch_size):
                if i == j:
                    continue

                # Check if we should rank i > j (i has higher risk)
                should_rank = False

                if events[i] == 1 and events[j] == 1:
                    # Both events: rank by time
                    if survival_times[i] < survival_times[j]:
                        should_rank = True
                elif events[i] == 1 and events[j] == 0:
                    # i event, j censored: rank if t_i < t_j
                    if survival_times[i] < survival_times[j]:
                        should_rank = True

                if should_rank:
                    # Hinge loss: max(0, margin - (risk_i - risk_j))
                    loss += F.relu(margin - (risk_scores[i] - risk_scores[j]))
                    num_pairs += 1

        if num_pairs > 0:
            loss = loss / num_pairs

        return loss
