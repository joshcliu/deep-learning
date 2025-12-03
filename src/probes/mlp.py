"""Multi-layer perceptron (MLP) probe for uncertainty quantification.

This module implements a feedforward neural network probe with multiple
hidden layers. It serves as a stronger baseline compared to the linear probe.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
from torch import nn

from .base import BaseProbe


class MLPProbe(BaseProbe):
    """Multi-layer perceptron probe for confidence prediction.

    This probe uses a feedforward neural network with configurable depth
    and width. It provides a stronger baseline than the linear probe while
    still being relatively lightweight.

    Architecture:
        Input -> [Linear -> ReLU -> Dropout] x num_layers -> Linear -> Sigmoid

    Args:
        input_dim: Dimensionality of the hidden state features.
        hidden_dim: Width of hidden layers (default: 512).
        num_layers: Number of hidden layers (default: 2).
        dropout: Dropout probability applied after each hidden layer (default: 0.1).
        device: Device to run the probe on. If None, uses CUDA if available.

    Raises:
        ValueError: If input_dim <= 0, hidden_dim <= 0, num_layers < 1, or dropout not in [0, 1).

    Example:
        >>> probe = MLPProbe(input_dim=4096, hidden_dim=512, num_layers=2)
        >>> probe.fit(X_train, y_train, X_val, y_val)
        >>> confidences = probe.predict(X_test)
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 512,
        num_layers: int = 2,
        dropout: float = 0.1,
        device: Optional[Union[str, torch.device]] = None,
    ) -> None:
        super().__init__()

        # Input validation
        if input_dim <= 0:
            raise ValueError(f"input_dim must be positive, got {input_dim}")
        if hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {hidden_dim}")
        if num_layers < 1:
            raise ValueError(f"num_layers must be at least 1, got {num_layers}")
        if not 0 <= dropout < 1:
            raise ValueError(f"dropout must be in [0, 1), got {dropout}")

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device: torch.device = torch.device(device)
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout_p = dropout

        # Build MLP layers
        layers: List[nn.Module] = []

        # First layer: input_dim -> hidden_dim
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout))

        # Hidden layers: hidden_dim -> hidden_dim
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))

        # Output layer: hidden_dim -> 1
        layers.append(nn.Linear(hidden_dim, 1))
        layers.append(nn.Sigmoid())

        self.mlp = nn.Sequential(*layers)

        self.to(self.device)

    @classmethod
    def from_config(cls, config: Dict[str, Any], input_dim: int) -> "MLPProbe":
        """Create probe from configuration dictionary.

        Args:
            config: Configuration dictionary with probe hyperparameters.
            input_dim: Dimensionality of the hidden state features.

        Returns:
            Initialized MLPProbe instance.

        Example:
            >>> config = {"hidden_dim": 512, "num_layers": 2, "dropout": 0.1}
            >>> probe = MLPProbe.from_config(config, input_dim=4096)
        """
        return cls(
            input_dim=input_dim,
            hidden_dim=config.get("hidden_dim", 512),
            num_layers=config.get("num_layers", 2),
            dropout=config.get("dropout", 0.1),
            device=config.get("device", None),
        )

    def forward(self, hiddens: torch.Tensor) -> torch.Tensor:
        """Forward pass returning confidence scores.

        Args:
            hiddens: Hidden state tensor of shape (batch_size, input_dim).

        Returns:
            Tensor of shape (batch_size, 1) with confidence scores in [0, 1].
        """
        hiddens = hiddens.to(self.device)
        return self.mlp(hiddens)

    def get_num_parameters(self) -> int:
        """Get the total number of trainable parameters in the probe.

        Returns:
            Total number of parameters.
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_model_info(self) -> Dict[str, Any]:
        """Get probe architecture information.

        Returns:
            Dictionary containing:
                - input_dim: Input feature dimension
                - hidden_dim: Hidden layer dimension
                - num_layers: Number of hidden layers
                - dropout: Dropout probability
                - device: Device the probe is on
                - num_parameters: Total number of trainable parameters
        """
        return {
            "input_dim": self.input_dim,
            "hidden_dim": self.hidden_dim,
            "num_layers": self.num_layers,
            "dropout": self.dropout_p,
            "device": str(self.device),
            "num_parameters": self.get_num_parameters(),
        }


__all__ = ["MLPProbe"]
