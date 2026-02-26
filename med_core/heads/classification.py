"""
Classification heads for multimodal medical imaging tasks.

This module provides various classification head implementations
for different medical imaging tasks, including multi-class classification,
multi-label classification, and ordinal classification.
"""

import torch
import torch.nn.functional as F
from torch import nn


class ClassificationHead(nn.Module):
    """
    Standard classification head with optional class balancing.

    Args:
        input_dim: Input feature dimension
        num_classes: Number of output classes
        hidden_dims: List of hidden layer dimensions
        dropout: Dropout rate
        use_batch_norm: Use batch normalization
        activation: Activation function ('relu', 'gelu', 'leaky_relu')

    Example:
        >>> head = ClassificationHead(input_dim=512, num_classes=4)
        >>> features = torch.randn(8, 512)
        >>> logits = head(features)  # [8, 4]
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_dims: list[int] | None = None,
        dropout: float = 0.5,
        use_batch_norm: bool = False,
        activation: str = "relu",
    ) -> None:
        super().__init__()

        self.input_dim = input_dim
        self.num_classes = num_classes

        if hidden_dims is None:
            hidden_dims = []

        # Build MLP layers
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))

            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))

            # Activation
            if activation == "relu":
                layers.append(nn.ReLU(inplace=True))
            elif activation == "gelu":
                layers.append(nn.GELU())
            elif activation == "leaky_relu":
                layers.append(nn.LeakyReLU(0.2, inplace=True))

            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim

        self.mlp = nn.Sequential(*layers) if layers else nn.Identity()

        # Final classification layer
        self.classifier = nn.Linear(prev_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input features [B, input_dim]

        Returns:
            Logits [B, num_classes]
        """
        x = self.mlp(x)
        logits = self.classifier(x)
        return logits


class MultiLabelClassificationHead(nn.Module):
    """
    Multi-label classification head for tasks where multiple labels can be active.

    Args:
        input_dim: Input feature dimension
        num_labels: Number of labels
        hidden_dims: List of hidden layer dimensions
        dropout: Dropout rate
        use_independent_classifiers: Use separate classifier for each label

    Example:
        >>> head = MultiLabelClassificationHead(input_dim=512, num_labels=5)
        >>> features = torch.randn(8, 512)
        >>> logits = head(features)  # [8, 5]
    """

    def __init__(
        self,
        input_dim: int,
        num_labels: int,
        hidden_dims: list[int] | None = None,
        dropout: float = 0.5,
        use_independent_classifiers: bool = False,
    ) -> None:
        super().__init__()

        self.input_dim = input_dim
        self.num_labels = num_labels
        self.use_independent_classifiers = use_independent_classifiers

        if hidden_dims is None:
            hidden_dims = [256]

        if use_independent_classifiers:
            # Separate classifier for each label
            self.classifiers = nn.ModuleList()
            for _ in range(num_labels):
                layers = []
                prev_dim = input_dim

                for hidden_dim in hidden_dims:
                    layers.extend(
                        [
                            nn.Linear(prev_dim, hidden_dim),
                            nn.ReLU(inplace=True),
                            nn.Dropout(dropout),
                        ],
                    )
                    prev_dim = hidden_dim

                layers.append(nn.Linear(prev_dim, 1))
                self.classifiers.append(nn.Sequential(*layers))
        else:
            # Shared feature extractor
            layers = []
            prev_dim = input_dim

            for hidden_dim in hidden_dims:
                layers.extend(
                    [
                        nn.Linear(prev_dim, hidden_dim),
                        nn.ReLU(inplace=True),
                        nn.Dropout(dropout),
                    ],
                )
                prev_dim = hidden_dim

            self.feature_extractor = nn.Sequential(*layers)
            self.classifier = nn.Linear(prev_dim, num_labels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input features [B, input_dim]

        Returns:
            Logits [B, num_labels]
        """
        if self.use_independent_classifiers:
            # Concatenate outputs from independent classifiers
            outputs = [classifier(x) for classifier in self.classifiers]
            logits = torch.cat(outputs, dim=1)
        else:
            x = self.feature_extractor(x)
            logits = self.classifier(x)

        return logits


class OrdinalClassificationHead(nn.Module):
    """
    Ordinal classification head for ordered categories (e.g., tumor grades).

    Uses the ordinal regression approach where K classes are modeled
    as K-1 binary classification problems.

    Args:
        input_dim: Input feature dimension
        num_classes: Number of ordinal classes
        hidden_dims: List of hidden layer dimensions
        dropout: Dropout rate

    Example:
        >>> head = OrdinalClassificationHead(input_dim=512, num_classes=4)
        >>> features = torch.randn(8, 512)
        >>> logits = head(features)  # [8, 3] (K-1 thresholds)
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_dims: list[int] | None = None,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()

        self.input_dim = input_dim
        self.num_classes = num_classes
        self.num_thresholds = num_classes - 1

        if hidden_dims is None:
            hidden_dims = [256]

        # Shared feature extractor
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(prev_dim, hidden_dim),
                    nn.ReLU(inplace=True),
                    nn.Dropout(dropout),
                ],
            )
            prev_dim = hidden_dim

        self.feature_extractor = nn.Sequential(*layers)

        # Ordinal thresholds (K-1 binary classifiers)
        self.thresholds = nn.Linear(prev_dim, self.num_thresholds)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input features [B, input_dim]

        Returns:
            Threshold logits [B, num_thresholds]
        """
        x = self.feature_extractor(x)
        logits = self.thresholds(x)
        return logits

    def predict_probabilities(self, x: torch.Tensor) -> torch.Tensor:
        """
        Predict class probabilities from ordinal logits.

        Args:
            x: Input features [B, input_dim]

        Returns:
            Class probabilities [B, num_classes]
        """
        logits = self.forward(x)
        # Convert threshold probabilities to class probabilities
        threshold_probs = torch.sigmoid(logits)

        # P(y = k) = P(y > k-1) - P(y > k)
        # Add boundaries: P(y > -1) = 1, P(y > K-1) = 0
        batch_size = x.size(0)
        probs = torch.zeros(batch_size, self.num_classes, device=x.device)

        # Prepend 1 and append 0 to threshold probabilities
        # [B, K-1] -> [B, K+1]
        padded_probs = torch.cat(
            [
                torch.ones(batch_size, 1, device=x.device),
                threshold_probs,
                torch.zeros(batch_size, 1, device=x.device),
            ],
            dim=1,
        )

        # P(y = k) = P(y > k-1) - P(y > k)
        for k in range(self.num_classes):
            probs[:, k] = padded_probs[:, k] - padded_probs[:, k + 1]

        # Clamp to ensure non-negative probabilities
        probs = torch.clamp(probs, min=0.0)

        # Renormalize to ensure sum to 1
        probs = probs / probs.sum(dim=1, keepdim=True)

        return probs


class AttentionClassificationHead(nn.Module):
    """
    Classification head with attention mechanism for interpretability.

    Uses attention to weight different parts of the input features,
    providing interpretable attention weights.

    Args:
        input_dim: Input feature dimension
        num_classes: Number of output classes
        attention_dim: Attention hidden dimension
        dropout: Dropout rate

    Example:
        >>> head = AttentionClassificationHead(input_dim=512, num_classes=4)
        >>> features = torch.randn(8, 512)
        >>> logits, attention_weights = head(features, return_attention=True)
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        attention_dim: int = 128,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()

        self.input_dim = input_dim
        self.num_classes = num_classes

        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(input_dim, attention_dim),
            nn.Tanh(),
            nn.Linear(attention_dim, 1),
        )

        # Classification layers
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(
        self, x: torch.Tensor, return_attention: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            x: Input features [B, input_dim] or [B, N, input_dim]
            return_attention: Return attention weights

        Returns:
            Logits [B, num_classes]
            If return_attention=True, also returns attention weights
        """
        # Handle both 2D and 3D inputs
        if x.dim() == 2:
            # Single feature vector per sample
            x = x.unsqueeze(1)  # [B, 1, input_dim]
            squeeze_output = True
        else:
            squeeze_output = False

        # Compute attention weights
        attention_logits = self.attention(x)  # [B, N, 1]
        attention_weights = F.softmax(attention_logits, dim=1)  # [B, N, 1]

        # Weighted sum of features
        weighted_features = (x * attention_weights).sum(dim=1)  # [B, input_dim]

        # Classification
        logits = self.classifier(weighted_features)

        if return_attention:
            if squeeze_output:
                attention_weights = attention_weights.squeeze(1)
            return logits, attention_weights
        return logits


class EnsembleClassificationHead(nn.Module):
    """
    Ensemble of multiple classification heads for improved robustness.

    Args:
        input_dim: Input feature dimension
        num_classes: Number of output classes
        num_heads: Number of ensemble heads
        hidden_dims: List of hidden layer dimensions
        dropout: Dropout rate
        aggregation: How to aggregate predictions ('mean', 'max', 'vote')

    Example:
        >>> head = EnsembleClassificationHead(
        ...     input_dim=512, num_classes=4, num_heads=3
        ... )
        >>> features = torch.randn(8, 512)
        >>> logits = head(features)  # [8, 4]
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        num_heads: int = 3,
        hidden_dims: list[int] | None = None,
        dropout: float = 0.5,
        aggregation: str = "mean",
    ) -> None:
        super().__init__()

        self.input_dim = input_dim
        self.num_classes = num_classes
        self.num_heads = num_heads
        self.aggregation = aggregation

        # Create ensemble of heads
        self.heads = nn.ModuleList(
            [
                ClassificationHead(
                    input_dim=input_dim,
                    num_classes=num_classes,
                    hidden_dims=hidden_dims,
                    dropout=dropout,
                )
                for _ in range(num_heads)
            ],
        )

    def forward(
        self, x: torch.Tensor, return_individual: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            x: Input features [B, input_dim]
            return_individual: Return individual head predictions

        Returns:
            Aggregated logits [B, num_classes]
            If return_individual=True, also returns individual predictions [B, num_heads, num_classes]
        """
        # Get predictions from all heads
        predictions = torch.stack(
            [head(x) for head in self.heads], dim=1,
        )  # [B, num_heads, num_classes]

        # Aggregate predictions
        if self.aggregation == "mean":
            logits = predictions.mean(dim=1)
        elif self.aggregation == "max":
            logits = predictions.max(dim=1)[0]
        elif self.aggregation == "vote":
            # Majority voting on predicted classes
            pred_classes = predictions.argmax(dim=2)  # [B, num_heads]
            logits = torch.zeros(x.size(0), self.num_classes, device=x.device)
            for i in range(x.size(0)):
                votes = torch.bincount(pred_classes[i], minlength=self.num_classes)
                logits[i] = votes.float()
        else:
            raise ValueError(f"Unknown aggregation: {self.aggregation}")

        if return_individual:
            return logits, predictions
        return logits
