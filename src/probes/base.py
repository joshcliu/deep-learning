"""Base class for uncertainty probes.

This module defines the abstract interface that all probe implementations
should follow. It provides common functionality for training, prediction,
and model persistence.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm


ArrayLike = Union[np.ndarray, torch.Tensor]


class BaseProbe(ABC, nn.Module):
    """Abstract base class for uncertainty quantification probes.

    All probe implementations should inherit from this class and implement
    the abstract forward() method. The base class provides common training,
    prediction, and persistence utilities.

    Subclasses should define their own __init__() method to set up the
    architecture, device, and hyperparameters.

    Example:
        >>> class MyProbe(BaseProbe):
        ...     def __init__(self, input_dim: int):
        ...         super().__init__()
        ...         self.linear = nn.Linear(input_dim, 1)
        ...
        ...     def forward(self, hiddens: torch.Tensor) -> torch.Tensor:
        ...         return torch.sigmoid(self.linear(hiddens))
    """

    def __init__(self) -> None:
        """Initialize the base probe.

        Note: Subclasses should call super().__init__() and then set up
        their own architecture, device, and any other attributes.
        """
        super().__init__()

    @abstractmethod
    def forward(self, hiddens: torch.Tensor) -> torch.Tensor:
        """Forward pass returning confidence scores.

        Args:
            hiddens: Input hidden states. Shape depends on the probe type:
                - For simple probes: (batch_size, input_dim)
                - For hierarchical probes: (batch_size, seq_len, input_dim)

        Returns:
            Confidence scores. Shape depends on the probe type:
                - Typically (batch_size, 1) or (batch_size,)
        """
        raise NotImplementedError("Subclasses must implement forward()")

    @torch.no_grad()
    def predict(self, X: ArrayLike, batch_size: int = 512) -> np.ndarray:
        """Predict confidence scores for a set of hidden states.

        This is a convenience method that wraps forward() with batching
        and numpy conversion. Subclasses can override this if they need
        custom prediction logic.

        Args:
            X: Hidden states as a NumPy array or torch tensor.
               Shape depends on probe type (see forward()).
            batch_size: Batch size used during prediction.

        Returns:
            NumPy array of confidence scores, typically shape (num_examples,).
        """
        self.eval()

        # Convert to tensor and get device
        X_tensor = self._to_tensor(X)
        device = next(self.parameters()).device
        X_tensor = X_tensor.to(device)

        dataset = TensorDataset(X_tensor)
        dataloader = DataLoader(dataset, batch_size=batch_size)

        all_probs: List[np.ndarray] = []
        for (batch_x,) in dataloader:
            probs = self.forward(batch_x)  # Shape varies by probe
            # Squeeze to 1D if needed
            if probs.dim() > 1:
                probs = probs.squeeze(-1)
            all_probs.append(probs.cpu().numpy())

        return np.concatenate(all_probs, axis=0)

    def fit(
        self,
        X_train: ArrayLike,
        y_train: ArrayLike,
        X_val: Optional[ArrayLike] = None,
        y_val: Optional[ArrayLike] = None,
        batch_size: int = 128,
        num_epochs: int = 50,
        patience: int = 5,
        lr: Optional[float] = None,
        weight_decay: Optional[float] = None,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """Train the probe on precomputed hidden states.

        This provides a default training loop with early stopping. Subclasses
        can override this if they need custom training logic, but should
        maintain a similar interface.

        Args:
            X_train: Training hidden states.
            y_train: Training labels (binary values {0, 1}).
            X_val: Optional validation hidden states.
            y_val: Optional validation labels.
            batch_size: Batch size for training.
            num_epochs: Maximum number of epochs.
            patience: Early stopping patience (epochs without improvement).
            lr: Learning rate. If None, uses 1e-3.
            weight_decay: L2 weight decay. If None, uses 0.0.
            verbose: Whether to print training progress.

        Returns:
            Dictionary with training/validation loss histories and metadata:
            {
                "train_loss": List[float],
                "val_loss": List[float],  # Empty if X_val is None
                "best_epoch": int,  # Epoch with best validation loss
                "converged": bool,  # False if early stopped
            }
        """
        if lr is None:
            lr = 1e-3
        if weight_decay is None:
            weight_decay = 0.0

        device = next(self.parameters()).device

        X_train_tensor = self._to_tensor(X_train).to(device)
        y_train_tensor = self._to_tensor(y_train).float().to(device)
        y_train_tensor = y_train_tensor.view(-1, 1)

        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        if X_val is not None and y_val is not None:
            X_val_tensor = self._to_tensor(X_val).to(device)
            y_val_tensor = self._to_tensor(y_val).float().to(device)
            y_val_tensor = y_val_tensor.view(-1, 1)
        else:
            X_val_tensor = None
            y_val_tensor = None

        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(
            self.parameters(), lr=lr, weight_decay=weight_decay
        )

        history: Dict[str, List[float]] = {"train_loss": [], "val_loss": []}
        best_val_loss: float = float("inf")
        best_epoch: int = 0
        best_state: Optional[Dict[str, Any]] = None
        epochs_without_improvement = 0
        early_stopped = False

        epoch_iter = tqdm(
            range(num_epochs),
            desc="Training probe",
            disable=not verbose,
            leave=True
        )

        for epoch in epoch_iter:
            self.train()
            epoch_train_losses: List[float] = []

            for batch_x, batch_y in train_loader:
                optimizer.zero_grad()
                # Get raw outputs (may be logits or probabilities depending on probe)
                outputs = self.forward(batch_x)
                if outputs.dim() > 1:
                    outputs = outputs.squeeze(-1)

                # Note: Using BCELoss here assumes forward() returns probabilities.
                # If your probe returns logits, you should use BCEWithLogitsLoss
                # or override this method.
                # For compatibility with sigmoid outputs:
                loss = nn.functional.binary_cross_entropy(
                    outputs.view(-1, 1), batch_y, reduction='mean'
                )

                loss.backward()
                optimizer.step()

                epoch_train_losses.append(loss.item())

            mean_train_loss = float(np.mean(epoch_train_losses))
            history["train_loss"].append(mean_train_loss)

            if X_val_tensor is not None and y_val_tensor is not None:
                val_loss = self._compute_validation_loss(
                    X_val_tensor, y_val_tensor, batch_size
                )
                history["val_loss"].append(val_loss)

                # Update progress bar
                epoch_iter.set_postfix({
                    "train_loss": f"{mean_train_loss:.4f}",
                    "val_loss": f"{val_loss:.4f}"
                })

                if val_loss < best_val_loss - 1e-6:
                    best_val_loss = val_loss
                    best_epoch = epoch
                    best_state = self.state_dict()
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += 1
                    if epochs_without_improvement >= patience:
                        if verbose:
                            print(
                                f"\nEarly stopping: validation loss did not improve "
                                f"for {patience} epochs."
                            )
                        early_stopped = True
                        break
            else:
                # Update progress bar (training loss only)
                epoch_iter.set_postfix({"train_loss": f"{mean_train_loss:.4f}"})

        # Restore best model parameters if validation was used
        if best_state is not None:
            self.load_state_dict(best_state)

        # Return comprehensive history
        return {
            "train_loss": history["train_loss"],
            "val_loss": history["val_loss"],
            "best_epoch": best_epoch,
            "converged": not early_stopped,
        }

    @torch.no_grad()
    def _compute_validation_loss(
        self,
        X_val: torch.Tensor,
        y_val: torch.Tensor,
        batch_size: int,
    ) -> float:
        """Compute validation loss over the entire validation set.

        Args:
            X_val: Validation hidden states.
            y_val: Validation labels.
            batch_size: Batch size for validation.

        Returns:
            Mean validation loss.
        """
        self.eval()

        dataset = TensorDataset(X_val, y_val)
        dataloader = DataLoader(dataset, batch_size=batch_size)

        val_losses: List[float] = []
        for batch_x, batch_y in dataloader:
            outputs = self.forward(batch_x)
            if outputs.dim() > 1:
                outputs = outputs.squeeze(-1)

            loss = nn.functional.binary_cross_entropy(
                outputs.view(-1, 1), batch_y, reduction='mean'
            )
            val_losses.append(loss.item())

        return float(np.mean(val_losses))

    def save(self, path: Union[str, Path]) -> None:
        """Save probe weights to disk.

        Args:
            path: Path to save the probe weights.
        """
        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        torch.save(
            {
                "state_dict": self.state_dict(),
                "class_name": self.__class__.__name__,
            },
            save_path,
        )

    def load(self, path: Union[str, Path]) -> None:
        """Load probe weights from disk.

        Args:
            path: Path to load the probe weights from.
        """
        device = next(self.parameters()).device
        checkpoint = torch.load(path, map_location=device)

        self.load_state_dict(checkpoint["state_dict"])
        self.eval()

    @staticmethod
    def _to_tensor(x: ArrayLike) -> torch.Tensor:
        """Convert input array-like to a float32 torch tensor.

        Args:
            x: Input as NumPy array or torch tensor.

        Returns:
            Float32 torch tensor.
        """
        if isinstance(x, torch.Tensor):
            return x.float()
        return torch.from_numpy(np.asarray(x)).float()


__all__ = ["BaseProbe"]
