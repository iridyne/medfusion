"""
Adaptive MLP backbone for tabular (structured) data.

Provides a flexible MLP architecture that:
- Auto-adapts to input feature dimensions
- Supports variable hidden layer configurations
- Includes batch normalization and dropout for regularization
- Handles both numerical and categorical features
"""

from typing import Literal

import torch
import torch.nn as nn

from med_core.backbones.base import BaseTabularBackbone


class AdaptiveMLP(BaseTabularBackbone):
    """
    Adaptive Multi-Layer Perceptron for tabular data.

    Automatically adjusts to input dimensions and provides
    a flexible architecture for structured data processing.

    Features:
    - Variable number of hidden layers
    - Configurable activation functions
    - Optional batch normalization
    - Dropout for regularization

    Example:
        >>> mlp = AdaptiveMLP(input_dim=11, output_dim=32, hidden_dims=[64, 64])
        >>> features = mlp(tabular_data)  # (B, 32)
    """

    ACTIVATIONS = {
        "relu": nn.ReLU,
        "gelu": nn.GELU,
        "silu": nn.SiLU,
        "leaky_relu": nn.LeakyReLU,
        "elu": nn.ELU,
    }

    def __init__(
        self,
        input_dim: int,
        output_dim: int = 32,
        hidden_dims: list[int] | None = None,
        dropout: float = 0.2,
        use_batch_norm: bool = True,
        activation: Literal["relu", "gelu", "silu", "leaky_relu", "elu"] = "relu",
        input_dropout: float = 0.0,
    ):
        """
        Initialize AdaptiveMLP.

        Args:
            input_dim: Number of input features
            output_dim: Output feature dimension
            hidden_dims: List of hidden layer dimensions (default: [64, 64])
            dropout: Dropout rate between hidden layers
            use_batch_norm: Whether to use batch normalization
            activation: Activation function type
            input_dropout: Dropout rate applied to input (for regularization)
        """
        super().__init__(
            input_dim=input_dim, output_dim=output_dim, hidden_dims=hidden_dims
        )

        self.dropout_rate = dropout
        self.use_batch_norm = use_batch_norm
        self.activation_name = activation
        self.input_dropout = input_dropout

        # Get activation class
        if activation not in self.ACTIVATIONS:
            raise ValueError(
                f"Unknown activation: {activation}. Choose from {list(self.ACTIVATIONS.keys())}"
            )
        activation_cls = self.ACTIVATIONS[activation]

        # Build network
        layers: list[nn.Module] = []

        # Optional input dropout
        if input_dropout > 0:
            layers.append(nn.Dropout(input_dropout))

        # Hidden layers
        hidden_dims = hidden_dims or [64, 64]
        prev_dim = input_dim

        for _i, hidden_dim in enumerate(hidden_dims):
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(activation_cls())

            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))

            if dropout > 0:
                layers.append(nn.Dropout(dropout))

            prev_dim = hidden_dim

        # Output layer
        layers.append(nn.Linear(prev_dim, output_dim))
        layers.append(activation_cls())

        if use_batch_norm:
            layers.append(nn.BatchNorm1d(output_dim))

        self.mlp = nn.Sequential(*layers)
        self._output_dim = output_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the MLP.

        Args:
            x: Input tensor of shape (B, input_dim)

        Returns:
            Feature tensor of shape (B, output_dim)
        """
        return self.mlp(x)

    def get_config(self) -> dict:
        """Return configuration for serialization."""
        return {
            "input_dim": self.input_dim,
            "output_dim": self._output_dim,
            "hidden_dims": self.hidden_dims,
            "dropout": self.dropout_rate,
            "use_batch_norm": self.use_batch_norm,
            "activation": self.activation_name,
            "input_dropout": self.input_dropout,
        }


class ResidualMLP(BaseTabularBackbone):
    """
    MLP with residual connections for deeper networks.

    Useful when processing high-dimensional tabular data
    that benefits from deeper architectures.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int = 32,
        hidden_dim: int = 64,
        num_blocks: int = 3,
        dropout: float = 0.2,
        use_batch_norm: bool = True,
    ):
        """
        Initialize ResidualMLP.

        Args:
            input_dim: Number of input features
            output_dim: Output feature dimension
            hidden_dim: Dimension of hidden layers (same for all blocks)
            num_blocks: Number of residual blocks
            dropout: Dropout rate
            use_batch_norm: Whether to use batch normalization
        """
        super().__init__(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dims=[hidden_dim] * num_blocks,
        )

        self.hidden_dim = hidden_dim
        self.num_blocks = num_blocks

        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.input_norm = (
            nn.BatchNorm1d(hidden_dim) if use_batch_norm else nn.Identity()
        )

        # Residual blocks
        self.blocks = nn.ModuleList()
        for _ in range(num_blocks):
            block = self._make_residual_block(hidden_dim, dropout, use_batch_norm)
            self.blocks.append(block)

        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, output_dim),
            nn.ReLU(),
            nn.BatchNorm1d(output_dim) if use_batch_norm else nn.Identity(),
        )

        self._output_dim = output_dim

    def _make_residual_block(
        self, dim: int, dropout: float, use_batch_norm: bool
    ) -> nn.Module:
        """Create a single residual block."""
        layers = [
            nn.Linear(dim, dim),
            nn.ReLU(),
        ]

        if use_batch_norm:
            layers.append(nn.BatchNorm1d(dim))

        layers.extend(
            [
                nn.Dropout(dropout),
                nn.Linear(dim, dim),
            ]
        )

        if use_batch_norm:
            layers.append(nn.BatchNorm1d(dim))

        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with residual connections."""
        # Project input
        x = self.input_proj(x)
        x = self.input_norm(x)
        x = torch.relu(x)

        # Residual blocks
        for block in self.blocks:
            residual = x
            x = block(x)
            x = torch.relu(x + residual)  # Add & activate

        # Output projection
        x = self.output_proj(x)

        return x


class FeatureTokenizer(nn.Module):
    """
    Tokenize tabular features for transformer-style processing.

    Converts each feature into a learnable embedding,
    enabling attention-based feature interactions.
    """

    def __init__(
        self,
        num_features: int,
        embedding_dim: int = 64,
        numerical_features: int | None = None,
        categorical_cardinalities: list[int] | None = None,
    ):
        """
        Initialize FeatureTokenizer.

        Args:
            num_features: Total number of features
            embedding_dim: Dimension of feature embeddings
            numerical_features: Number of numerical features (first N)
            categorical_cardinalities: Cardinality of each categorical feature
        """
        super().__init__()

        self.num_features = num_features
        self.embedding_dim = embedding_dim

        # For numerical features: linear projection per feature
        num_numerical = numerical_features or num_features
        self.numerical_embeddings = nn.ModuleList(
            [nn.Linear(1, embedding_dim) for _ in range(num_numerical)]
        )

        # For categorical features: embedding tables
        self.categorical_embeddings = nn.ModuleList()
        if categorical_cardinalities:
            for cardinality in categorical_cardinalities:
                self.categorical_embeddings.append(
                    nn.Embedding(cardinality, embedding_dim)
                )

        # Learnable [CLS] token for aggregation
        self.cls_token = nn.Parameter(torch.randn(1, 1, embedding_dim))

    def forward(
        self,
        numerical: torch.Tensor | None = None,
        categorical: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Tokenize features into embeddings.

        Args:
            numerical: Numerical features (B, num_numerical)
            categorical: Categorical features as indices (B, num_categorical)

        Returns:
            Feature tokens (B, num_features + 1, embedding_dim) including CLS token
        """
        batch_size = numerical.size(0) if numerical is not None else categorical.size(0)
        tokens = []

        # Add CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        tokens.append(cls_tokens)

        # Embed numerical features
        if numerical is not None:
            for i, embed in enumerate(self.numerical_embeddings):
                feat = numerical[:, i : i + 1]  # (B, 1)
                tokens.append(embed(feat).unsqueeze(1))  # (B, 1, embed_dim)

        # Embed categorical features
        if categorical is not None:
            for i, embed in enumerate(self.categorical_embeddings):
                feat = categorical[:, i].long()  # (B,)
                tokens.append(embed(feat).unsqueeze(1))  # (B, 1, embed_dim)

        # Concatenate all tokens
        tokens = torch.cat(tokens, dim=1)  # (B, num_tokens, embed_dim)

        return tokens


def create_tabular_backbone(
    input_dim: int,
    output_dim: int = 32,
    hidden_dims: list[int] | None = None,
    backbone_type: Literal["mlp", "residual"] = "mlp",
    **kwargs,
) -> BaseTabularBackbone:
    """
    Factory function to create tabular backbones.

    Args:
        input_dim: Number of input features
        output_dim: Output feature dimension
        hidden_dims: List of hidden layer dimensions
        backbone_type: Type of backbone ("mlp" or "residual")
        **kwargs: Additional arguments passed to backbone constructor

    Returns:
        Tabular backbone instance
    """
    # Filter out 'config' from kwargs as it's not needed by backbone constructors
    filtered_kwargs = {k: v for k, v in kwargs.items() if k != "config"}

    if backbone_type == "mlp":
        return AdaptiveMLP(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dims=hidden_dims,
            **filtered_kwargs,
        )
    elif backbone_type == "residual":
        hidden_dim = hidden_dims[0] if hidden_dims else 64
        num_blocks = len(hidden_dims) if hidden_dims else 3
        return ResidualMLP(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dim=hidden_dim,
            num_blocks=num_blocks,
            **filtered_kwargs,
        )
    else:
        raise ValueError(f"Unknown backbone type: {backbone_type}")
