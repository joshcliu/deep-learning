"""Calibrated probe trained with Brier score loss.

This probe is trained to predict whether the model's OWN generated answer
is correct, using Brier score (squared error) loss which properly penalizes
overconfident wrong predictions.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from .base import BaseProbe


ArrayLike = Union[np.ndarray, torch.Tensor]


def build_default_network(
    input_dim: int,
    hidden_dim: Optional[int] = 256,
    dropout: float = 0.1,
) -> nn.Module:
    """Build the default MLP network for CalibratedProbe.

    Args:
        input_dim: Input feature dimension.
        hidden_dim: Hidden layer dimension. If None, uses linear probe.
        dropout: Dropout probability.

    Returns:
        nn.Module that maps (batch, input_dim) -> (batch, 1) with sigmoid output.
    """
    if hidden_dim is None:
        # Linear probe
        return nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(input_dim, 1),
            nn.Sigmoid(),
        )
    else:
        # MLP probe
        return nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )


class CalibratedProbe(BaseProbe):
    """Probe trained with Brier score loss for well-calibrated confidence.

    Unlike standard probes trained with BCE loss, this probe uses Brier score:
        loss = (confidence - correct)^2

    This properly penalizes:
    - High confidence + wrong answer → high loss (very bad)
    - Low confidence + wrong answer → low loss (acceptable)
    - High confidence + correct answer → low loss (good)
    - Low confidence + correct answer → medium loss (could be better)

    Args:
        input_dim: Dimensionality of hidden state features.
        hidden_dim: Hidden layer dimension (None for linear probe).
        dropout: Dropout probability.
        lr: Learning rate.
        weight_decay: L2 regularization.
        device: Device to run on.

    Example:
        >>> probe = CalibratedProbe(input_dim=4096, hidden_dim=256)
        >>> probe.fit(hidden_states, correctness_labels)
        >>> confidences = probe.predict(test_hidden_states)
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: Optional[int] = None,
        dropout: float = 0.1,
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
        device: Optional[str] = None,
    ) -> None:
        super().__init__()

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        # Build network
        if hidden_dim is None:
            # Linear probe
            self.network = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(input_dim, 1),
                nn.Sigmoid(),
            )
        else:
            # MLP probe
            self.network = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 1),
                nn.Sigmoid(),
            )

        self.lr = lr
        self.weight_decay = weight_decay
        self.to(self.device)

    def forward(self, hiddens: torch.Tensor) -> torch.Tensor:
        """Forward pass returning confidence scores in [0, 1]."""
        hiddens = hiddens.to(self.device)
        return self.network(hiddens)

    @torch.no_grad()
    def predict(self, X: ArrayLike, batch_size: int = 512) -> np.ndarray:
        """Predict confidence scores."""
        self.eval()
        X_tensor = self._to_tensor(X).to(self.device)
        dataset = TensorDataset(X_tensor)
        dataloader = DataLoader(dataset, batch_size=batch_size)

        all_confs = []
        for (batch_x,) in dataloader:
            confs = self.forward(batch_x)
            all_confs.append(confs.squeeze(-1).cpu().numpy())

        return np.concatenate(all_confs, axis=0)

    def fit(
        self,
        X_train: ArrayLike,
        y_train: ArrayLike,
        X_val: Optional[ArrayLike] = None,
        y_val: Optional[ArrayLike] = None,
        batch_size: int = 64,
        num_epochs: int = 100,
        patience: int = 10,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """Train probe with Brier score loss.

        Args:
            X_train: Hidden states (num_examples, input_dim)
            y_train: Binary correctness labels (num_examples,) with {0, 1}
            X_val: Optional validation hidden states
            y_val: Optional validation labels
            batch_size: Training batch size
            num_epochs: Maximum epochs
            patience: Early stopping patience
            verbose: Print progress

        Returns:
            Training history dict with losses and metrics
        """
        X_train_t = self._to_tensor(X_train).to(self.device)
        y_train_t = self._to_tensor(y_train).float().to(self.device).view(-1, 1)

        train_dataset = TensorDataset(X_train_t, y_train_t)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        if X_val is not None and y_val is not None:
            X_val_t = self._to_tensor(X_val).to(self.device)
            y_val_t = self._to_tensor(y_val).float().to(self.device).view(-1, 1)
        else:
            X_val_t, y_val_t = None, None

        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay
        )

        history = {"train_loss": [], "val_loss": [], "train_brier": [], "val_brier": []}
        best_val_loss = float("inf")
        best_state = None
        best_epoch = 0
        no_improve = 0

        epoch_iter = tqdm(range(num_epochs), desc="Training", disable=not verbose)

        for epoch in epoch_iter:
            # Training
            self.train()
            train_losses = []

            for batch_x, batch_y in train_loader:
                optimizer.zero_grad()
                confidence = self.forward(batch_x)

                # Brier score loss: (confidence - correct)^2
                loss = ((confidence - batch_y) ** 2).mean()

                loss.backward()
                optimizer.step()
                train_losses.append(loss.item())

            mean_train_loss = np.mean(train_losses)
            history["train_loss"].append(mean_train_loss)
            history["train_brier"].append(mean_train_loss)  # Brier = MSE for binary

            # Validation
            if X_val_t is not None:
                val_loss = self._compute_brier(X_val_t, y_val_t)
                history["val_loss"].append(val_loss)
                history["val_brier"].append(val_loss)

                epoch_iter.set_postfix({
                    "train_brier": f"{mean_train_loss:.4f}",
                    "val_brier": f"{val_loss:.4f}"
                })

                if val_loss < best_val_loss - 1e-6:
                    best_val_loss = val_loss
                    best_state = self.state_dict()
                    best_epoch = epoch
                    no_improve = 0
                else:
                    no_improve += 1
                    if no_improve >= patience:
                        if verbose:
                            print(f"\nEarly stopping at epoch {epoch}")
                        break
            else:
                epoch_iter.set_postfix({"train_brier": f"{mean_train_loss:.4f}"})

        # Restore best model
        if best_state is not None:
            self.load_state_dict(best_state)

        return {
            "train_loss": history["train_loss"],
            "val_loss": history["val_loss"],
            "train_brier": history["train_brier"],
            "val_brier": history["val_brier"],
            "best_epoch": best_epoch,
            "best_val_brier": best_val_loss,
        }

    @torch.no_grad()
    def _compute_brier(self, X: torch.Tensor, y: torch.Tensor) -> float:
        """Compute Brier score on a dataset."""
        self.eval()
        confidence = self.forward(X)
        brier = ((confidence - y) ** 2).mean().item()
        return brier

    @staticmethod
    def _to_tensor(x: ArrayLike) -> torch.Tensor:
        if isinstance(x, torch.Tensor):
            return x.float()
        return torch.from_numpy(np.asarray(x)).float()


__all__ = ["CalibratedProbe"]
