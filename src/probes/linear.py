"""Linear probe for uncertainty quantification.

A simple probe consisting of a single linear layer that maps hidden states
to confidence scores. Serves as the baseline for more complex probe architectures.
"""

import torch
import torch.nn as nn
from .base import BaseProbe


class LinearProbe(BaseProbe):
    """Linear probe for binary confidence prediction.

    Maps hidden states to a single confidence score using a linear transformation
    followed by sigmoid activation.

    Args:
        input_dim: Dimension of input hidden states
        dropout: Dropout rate (0.0 = no dropout)
        device: Device to run computations on ('cuda' or 'cpu')

    Example:
        >>> probe = LinearProbe(input_dim=4096, dropout=0.1)
        >>> probe.fit(X_train, y_train, X_val, y_val)
        >>> confidences = probe.predict(X_test)
    """

    def __init__(self, input_dim: int, dropout: float = 0.0, device: str = "cuda"):
        super().__init__(input_dim, device)
        self.dropout_rate = dropout

        # Network architecture
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(input_dim, 1)
        self.sigmoid = nn.Sigmoid()

        # Initialize weights
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

        self.to(self.device)

    def forward(self, hiddens: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            hiddens: Input hidden states of shape (batch_size, input_dim)

        Returns:
            Confidence scores of shape (batch_size, 1)
        """
        x = self.dropout(hiddens)
        x = self.linear(x)
        x = self.sigmoid(x)
        return x


class MLPProbe(BaseProbe):
    """Multi-layer perceptron probe for uncertainty quantification.

    A more expressive probe with one or more hidden layers.

    Args:
        input_dim: Dimension of input hidden states
        hidden_dim: Dimension of hidden layer(s)
        num_layers: Number of hidden layers (1 or 2)
        dropout: Dropout rate
        device: Device to run computations on

    Example:
        >>> probe = MLPProbe(input_dim=4096, hidden_dim=512, num_layers=2, dropout=0.2)
        >>> probe.fit(X_train, y_train, X_val, y_val)
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 512,
        num_layers: int = 1,
        dropout: float = 0.1,
        device: str = "cuda",
    ):
        super().__init__(input_dim, device)
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout_rate = dropout

        # Build network
        layers = []

        # Input layer
        layers.extend([
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        ])

        # Additional hidden layers
        for _ in range(num_layers - 1):
            layers.extend([
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])

        # Output layer
        layers.append(nn.Linear(hidden_dim, 1))
        layers.append(nn.Sigmoid())

        self.network = nn.Sequential(*layers)

        # Initialize weights
        for module in self.network.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

        self.to(self.device)

    def forward(self, hiddens: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            hiddens: Input hidden states of shape (batch_size, input_dim)

        Returns:
            Confidence scores of shape (batch_size, 1)
        """
        return self.network(hiddens)
