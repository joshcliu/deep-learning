"""
Base probe class for uncertainty quantification.

All probe implementations should inherit from BaseProbe and implement
the required abstract methods.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Union

import numpy as np
import torch
from torch import nn


ArrayLike = Union[np.ndarray, torch.Tensor]


class BaseProbe(ABC, nn.Module):
    """Abstract base class for all uncertainty probes.

    Probes are lightweight models trained on frozen hidden states
    extracted from a language model. They map hidden representations
    to confidence scores for uncertainty quantification.

    All concrete probe implementations must inherit from this class
    and implement the abstract methods.

    Example:
        >>> class MyProbe(BaseProbe):
        ...     def forward(self, hiddens):
        ...         return self.linear(hiddens)
        ...
        ...     def fit(self, X_train, y_train, X_val, y_val):
        ...         # Training logic
        ...         pass
        ...
        ...     def predict(self, X):
        ...         # Prediction logic
        ...         pass
    """

    @abstractmethod
    def forward(self, hiddens: torch.Tensor) -> torch.Tensor:
        """Forward pass returning confidence scores.

        Args:
            hiddens: Hidden state tensor of shape (batch_size, input_dim).

        Returns:
            Tensor of confidence scores of shape (batch_size, 1) or
            (batch_size,) with values in [0, 1].

        Raises:
            NotImplementedError: If not implemented by subclass.
        """
        raise NotImplementedError("Subclasses must implement forward()")

    @abstractmethod
    def fit(
        self,
        X_train: ArrayLike,
        y_train: ArrayLike,
        X_val: Optional[ArrayLike] = None,
        y_val: Optional[ArrayLike] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Train the probe on precomputed hidden states.

        Args:
            X_train: Training hidden states of shape
                (num_examples, input_dim).
            y_train: Training labels of shape (num_examples,) with binary
                values {0, 1}.
            X_val: Optional validation hidden states.
            y_val: Optional validation labels.
            **kwargs: Additional training arguments (batch_size, epochs, etc.).

        Returns:
            Dictionary with training history and metadata. Should include
            at minimum:
            {
                "train_loss": List[float],  # Loss per epoch
                "val_loss": List[float],    # Validation loss per epoch
                "converged": bool,          # Whether training converged
            }

        Raises:
            NotImplementedError: If not implemented by subclass.
        """
        raise NotImplementedError("Subclasses must implement fit()")

    @abstractmethod
    def predict(self, X: ArrayLike, batch_size: int = 512) -> np.ndarray:
        """Predict confidence scores for a set of hidden states.

        Args:
            X: Hidden states of shape (num_examples, input_dim) as a
                NumPy array or torch tensor.
            batch_size: Batch size for prediction.

        Returns:
            NumPy array of shape (num_examples,) with confidence scores
            in [0, 1].

        Raises:
            NotImplementedError: If not implemented by subclass.
        """
        raise NotImplementedError("Subclasses must implement predict()")

    def get_model_info(self) -> Dict[str, Any]:
        """Get probe architecture and configuration information.

        This method can be overridden by subclasses to provide more
        detailed information.

        Returns:
            Dictionary with probe metadata including:
                - num_parameters: Total number of parameters
                - device: Device the probe is on
        """
        return {
            "num_parameters": sum(p.numel() for p in self.parameters()),
            "device": next(self.parameters()).device.type
            if any(self.parameters())
            else "cpu",
        }

    def save(self, path: str) -> None:
        """Save probe state to disk.

        Args:
            path: File path to save probe state (typically .pt or .pth).

        Example:
            >>> probe.save("checkpoints/probe_epoch10.pt")
        """
        torch.save(
            {
                "state_dict": self.state_dict(),
                "model_info": self.get_model_info(),
            },
            path,
        )

    def load(self, path: str, map_location: Optional[str] = None) -> None:
        """Load probe state from disk.

        Args:
            path: File path to load probe state from.
            map_location: Device to map tensors to (e.g., 'cpu', 'cuda:0').

        Example:
            >>> probe.load("checkpoints/probe_epoch10.pt", map_location="cpu")
        """
        checkpoint = torch.load(path, map_location=map_location)
        self.load_state_dict(checkpoint["state_dict"])

    @staticmethod
    def _to_tensor(x: ArrayLike) -> torch.Tensor:
        """Convert input array-like to a float32 torch tensor.

        Helper method for subclasses to convert NumPy arrays or
        existing tensors to the correct format.

        Args:
            x: Input array or tensor.

        Returns:
            Float32 torch tensor.
        """
        if isinstance(x, torch.Tensor):
            return x.float()
        return torch.from_numpy(np.asarray(x)).float()


__all__ = ["BaseProbe", "ArrayLike"]
