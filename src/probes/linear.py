from __future__ import annotations

import functools
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from .base import BaseProbe


ArrayLike = Union[np.ndarray, torch.Tensor]


class LinearProbe(BaseProbe):
    """Linear probe mapping hidden states to confidence scores.

    This probe is a single-layer logistic classifier trained with
    binary cross-entropy. It operates on frozen hidden states extracted
    from an upstream language model.

    The probe also supports optional post-hoc temperature scaling
    using a held-out validation set.

    Args:
        input_dim: Dimensionality of the hidden state features.
        dropout: Dropout probability applied before the linear layer.
        lr: Learning rate used during training.
        weight_decay: L2 weight decay for the optimizer.
        device: Device to run the probe on. If None, uses CUDA if available.
        use_bias: Whether to include a bias term in the linear layer.

    Raises:
        ValueError: If input_dim <= 0 or dropout not in [0, 1).

    Example:
        >>> probe = LinearProbe(input_dim=1024)
        >>> probe.fit(X_train, y_train, X_val, y_val)
        >>> confidences = probe.predict(X_test)
    """

    def __init__(
        self,
        input_dim: int,
        dropout: float = 0.0,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        device: Optional[Union[str, torch.device]] = None,
        use_bias: bool = True,
    ) -> None:
        super().__init__()

        # Input validation
        if input_dim <= 0:
            raise ValueError(f"input_dim must be positive, got {input_dim}")
        if not 0 <= dropout < 1:
            raise ValueError(f"dropout must be in [0, 1), got {dropout}")
        if lr <= 0:
            raise ValueError(f"lr must be positive, got {lr}")
        if weight_decay < 0:
            raise ValueError(f"weight_decay must be non-negative, got {weight_decay}")

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device: torch.device = torch.device(device)
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(input_dim, 1, bias=use_bias)
        # Temperature parameter for post-hoc calibration
        self.log_temperature = nn.Parameter(torch.zeros(1), requires_grad=True)

        self.lr = lr
        self.weight_decay = weight_decay

        self.to(self.device)

    @classmethod
    def from_config(cls, config: Dict[str, Any], input_dim: int) -> "LinearProbe":
        """Create probe from configuration dictionary.

        Args:
            config: Configuration dictionary with probe hyperparameters.
            input_dim: Dimensionality of the hidden state features.

        Returns:
            Initialized LinearProbe instance.

        Example:
            >>> config = {"dropout": 0.1, "lr": 1e-3, "weight_decay": 1e-5}
            >>> probe = LinearProbe.from_config(config, input_dim=4096)
        """
        return cls(
            input_dim=input_dim,
            dropout=config.get("dropout", 0.0),
            lr=config.get("lr", 1e-3),
            weight_decay=config.get("weight_decay", 0.0),
            device=config.get("device", None),
            use_bias=config.get("use_bias", True),
        )

    # -------------------------------------------------------------------------
    # Core forward / prediction API
    # -------------------------------------------------------------------------

    def _forward_logits(self, hiddens: torch.Tensor) -> torch.Tensor:
        """Compute raw logits before sigmoid.

        Args:
            hiddens: Hidden state tensor of shape (batch_size, input_dim).

        Returns:
            Logits tensor of shape (batch_size, 1).
        """
        hiddens = hiddens.to(self.device)
        dropped = self.dropout(hiddens)
        logits = self.linear(dropped)
        return logits

    def forward(self, hiddens: torch.Tensor) -> torch.Tensor:
        """Forward pass returning calibrated confidence scores.

        This applies the learned temperature scaling (if any) and a
        sigmoid to map logits to probabilities in [0, 1].

        Args:
            hiddens: Hidden state tensor of shape (batch_size, input_dim).

        Returns:
            Tensor of shape (batch_size, 1) with confidence scores.
        """
        logits = self._forward_logits(hiddens)
        temperature = self.log_temperature.exp().clamp(min=1e-6)
        scaled_logits = logits / temperature
        return torch.sigmoid(scaled_logits)

    @torch.no_grad()
    def predict(self, X: ArrayLike, batch_size: int = 512) -> np.ndarray:
        """Predict confidence scores for a set of hidden states.

        Args:
            X: Hidden states of shape (num_examples, input_dim) as a NumPy
                array or torch tensor.
            batch_size: Batch size used during prediction.

        Returns:
            NumPy array of shape (num_examples,) with confidence scores.
        """
        self.eval()

        X_tensor = self._to_tensor(X).to(self.device)
        dataset = TensorDataset(X_tensor)
        dataloader = DataLoader(dataset, batch_size=batch_size)

        all_probs: List[np.ndarray] = []
        for (batch_x,) in dataloader:
            probs = self.forward(batch_x)  # (batch, 1)
            all_probs.append(probs.squeeze(-1).cpu().numpy())

        return np.concatenate(all_probs, axis=0)

    # -------------------------------------------------------------------------
    # Training API
    # -------------------------------------------------------------------------

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
        """Train the linear probe on precomputed hidden states.

        Uses binary cross-entropy with logits and early stopping based on
        validation loss (if validation data is provided).

        Args:
            X_train: Training hidden states of shape
                (num_examples, input_dim).
            y_train: Training labels of shape (num_examples,) with binary
                values {0, 1}.
            X_val: Optional validation hidden states.
            y_val: Optional validation labels.
            batch_size: Batch size for training.
            num_epochs: Maximum number of epochs.
            patience: Early stopping patience (epochs without improvement).
            lr: Optional override for learning rate.
            weight_decay: Optional override for weight decay.
            verbose: Whether to print training progress.

        Returns:
            Dictionary with training/validation loss histories and metadata:
            {
                "train_loss": [...],
                "val_loss": [...],  # Empty if X_val is None
                "best_epoch": int,  # Epoch with best validation loss
                "converged": bool,  # False if early stopped
                "final_temperature": float
            }
        """
        lr = self.lr if lr is None else lr
        weight_decay = self.weight_decay if weight_decay is None else weight_decay

        X_train_tensor = self._to_tensor(X_train).to(self.device)
        y_train_tensor = self._to_tensor(y_train).float().to(self.device)
        y_train_tensor = y_train_tensor.view(-1, 1)

        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        if X_val is not None and y_val is not None:
            X_val_tensor = self._to_tensor(X_val).to(self.device)
            y_val_tensor = self._to_tensor(y_val).float().to(self.device)
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
                logits = self._forward_logits(batch_x)
                loss = criterion(logits, batch_y)
                loss.backward()
                optimizer.step()

                epoch_train_losses.append(loss.item())

            mean_train_loss = float(np.mean(epoch_train_losses))
            history["train_loss"].append(mean_train_loss)

            if X_val_tensor is not None and y_val_tensor is not None:
                val_loss = self._compute_validation_loss(
                    X_val_tensor, y_val_tensor, criterion, batch_size
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
                    best_state = {
                        "model": self.state_dict(),
                        "log_temperature": self.log_temperature.detach().clone(),
                    }
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
            self.load_state_dict(best_state["model"])
            with torch.no_grad():
                self.log_temperature.copy_(best_state["log_temperature"])

        # Return comprehensive history
        return {
            "train_loss": history["train_loss"],
            "val_loss": history["val_loss"],
            "best_epoch": best_epoch,
            "converged": not early_stopped,
            "final_temperature": self.log_temperature.exp().item(),
        }

    @torch.no_grad()
    def _compute_validation_loss(
        self,
        X_val: torch.Tensor,
        y_val: torch.Tensor,
        criterion: nn.Module,
        batch_size: int,
    ) -> float:
        """Compute validation loss over the entire validation set."""
        self.eval()

        dataset = TensorDataset(X_val, y_val)
        dataloader = DataLoader(dataset, batch_size=batch_size)

        val_losses: List[float] = []
        for batch_x, batch_y in dataloader:
            logits = self._forward_logits(batch_x)
            loss = criterion(logits, batch_y)
            val_losses.append(loss.item())

        return float(np.mean(val_losses))

    # -------------------------------------------------------------------------
    # Temperature scaling
    # -------------------------------------------------------------------------

    def fit_temperature(
        self,
        X_val: ArrayLike,
        y_val: ArrayLike,
        max_iters: int = 50,
        lr: float = 0.01,
        freeze_after: bool = True,
        verbose: bool = True,
    ) -> None:
        """Fit a temperature parameter using a held-out validation set.

        The probe weights are frozen and only the temperature parameter is
        optimized to minimize negative log-likelihood.

        Args:
            X_val: Validation hidden states of shape
                (num_examples, input_dim).
            y_val: Validation labels of shape (num_examples,) with {0, 1}.
            max_iters: Maximum number of optimization steps.
            lr: Learning rate for temperature optimization.
            freeze_after: Whether to freeze temperature after fitting.
            verbose: Whether to print optimization progress.
        """
        self.eval()
        for param in self.linear.parameters():
            param.requires_grad = False

        X_val_tensor = self._to_tensor(X_val).to(self.device)
        y_val_tensor = self._to_tensor(y_val).float().to(self.device)
        y_val_tensor = y_val_tensor.view(-1, 1)

        optimizer = torch.optim.LBFGS(
            [self.log_temperature],
            lr=lr,
            max_iter=max_iters,
            line_search_fn="strong_wolfe",
        )
        criterion = nn.BCEWithLogitsLoss()

        def closure() -> torch.Tensor:
            optimizer.zero_grad()
            logits = self._forward_logits(X_val_tensor)
            temperature = self.log_temperature.exp().clamp(min=1e-6)
            scaled_logits = logits / temperature
            loss = criterion(scaled_logits, y_val_tensor)
            loss.backward()
            return loss

        loss = optimizer.step(closure)

        if verbose:
            temperature = self.log_temperature.exp().item()
            print(
                f"Fitted temperature: {temperature:.4f} "
                f"(validation NLL={loss.item():.4f})"
            )

        # Optionally freeze temperature after fitting
        if freeze_after:
            self.log_temperature.requires_grad_(False)

    # -------------------------------------------------------------------------
    # Utilities and introspection
    # -------------------------------------------------------------------------

    def get_num_parameters(self) -> int:
        """Get the total number of trainable parameters in the probe.

        Returns:
            Total number of parameters.
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @functools.lru_cache(maxsize=1)
    def get_model_info(self) -> Dict[str, Any]:
        """Get probe architecture information.

        Returns:
            Dictionary containing:
                - input_dim: Input feature dimension
                - dropout: Dropout probability
                - device: Device the probe is on
                - num_parameters: Total number of trainable parameters
                - temperature: Current temperature value
        """
        return {
            "input_dim": self.linear.in_features,
            "dropout": self.dropout.p,
            "device": str(self.device),
            "num_parameters": sum(p.numel() for p in self.parameters()),
            "temperature": self.log_temperature.exp().item(),
        }

    @staticmethod
    def _to_tensor(x: ArrayLike) -> torch.Tensor:
        """Convert input array-like to a float32 torch tensor."""
        if isinstance(x, torch.Tensor):
            return x.float()
        return torch.from_numpy(np.asarray(x)).float()


__all__ = ["LinearProbe"]