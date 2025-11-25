"""Base class for uncertainty probes."""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from loguru import logger


class BaseProbe(ABC, nn.Module):
    """Abstract base class for uncertainty quantification probes.

    All probe implementations should inherit from this class and implement
    the abstract methods.

    Args:
        input_dim: Dimension of input hidden states
        device: Device to run computations on ('cuda' or 'cpu')
    """

    def __init__(self, input_dim: int, device: str = "cuda"):
        super().__init__()
        self.input_dim = input_dim
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.is_fitted = False

    @abstractmethod
    def forward(self, hiddens: torch.Tensor) -> torch.Tensor:
        """Forward pass returning confidence scores.

        Args:
            hiddens: Input hidden states of shape (batch_size, input_dim)

        Returns:
            Confidence scores of shape (batch_size,) or (batch_size, 1)
        """
        pass

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        epochs: int = 100,
        batch_size: int = 32,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        early_stopping_patience: int = 10,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """Train the probe on labeled data.

        Args:
            X_train: Training hidden states (n_samples, input_dim)
            y_train: Training labels (n_samples,) - binary correctness (0/1)
            X_val: Validation hidden states
            y_val: Validation labels
            epochs: Maximum number of training epochs
            batch_size: Batch size for training
            lr: Learning rate
            weight_decay: L2 regularization strength
            early_stopping_patience: Stop if validation loss doesn't improve for N epochs
            verbose: Whether to print training progress

        Returns:
            Dictionary with training history
        """
        self.to(self.device)
        self.train()

        # Convert to tensors
        X_train_tensor = torch.FloatTensor(X_train).to(self.device)
        y_train_tensor = torch.FloatTensor(y_train).to(self.device)

        # Create data loader
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, drop_last=False
        )

        # Setup optimizer and loss
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=lr, weight_decay=weight_decay
        )
        criterion = nn.BCELoss()

        # Training history
        history = {
            "train_loss": [],
            "val_loss": [],
            "best_epoch": 0,
            "best_val_loss": float("inf"),
        }

        # Validation setup
        has_validation = X_val is not None and y_val is not None
        if has_validation:
            X_val_tensor = torch.FloatTensor(X_val).to(self.device)
            y_val_tensor = torch.FloatTensor(y_val).to(self.device)

        # Early stopping
        best_state = None
        patience_counter = 0

        # Training loop
        for epoch in range(epochs):
            # Training
            train_loss = 0.0
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = self.forward(batch_X).squeeze()
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * len(batch_X)

            train_loss /= len(X_train)
            history["train_loss"].append(train_loss)

            # Validation
            if has_validation:
                self.eval()
                with torch.no_grad():
                    val_outputs = self.forward(X_val_tensor).squeeze()
                    val_loss = criterion(val_outputs, y_val_tensor).item()
                history["val_loss"].append(val_loss)
                self.train()

                # Early stopping check
                if val_loss < history["best_val_loss"]:
                    history["best_val_loss"] = val_loss
                    history["best_epoch"] = epoch
                    best_state = self.state_dict().copy()
                    patience_counter = 0
                else:
                    patience_counter += 1

                if patience_counter >= early_stopping_patience:
                    if verbose:
                        logger.info(
                            f"Early stopping at epoch {epoch}. "
                            f"Best epoch: {history['best_epoch']}"
                        )
                    break

                if verbose and (epoch + 1) % 10 == 0:
                    logger.info(
                        f"Epoch {epoch+1}/{epochs} - "
                        f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}"
                    )
            else:
                if verbose and (epoch + 1) % 10 == 0:
                    logger.info(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}")

        # Restore best model if using validation
        if has_validation and best_state is not None:
            self.load_state_dict(best_state)

        self.is_fitted = True
        self.eval()

        if verbose:
            logger.info("Training completed")

        return history

    def predict(self, X: np.ndarray, batch_size: int = 128) -> np.ndarray:
        """Predict confidence scores for input hidden states.

        Args:
            X: Hidden states (n_samples, input_dim)
            batch_size: Batch size for inference

        Returns:
            Confidence scores (n_samples,)
        """
        if not self.is_fitted:
            logger.warning("Probe has not been fitted. Results may be unreliable.")

        self.eval()
        self.to(self.device)

        X_tensor = torch.FloatTensor(X).to(self.device)
        predictions = []

        with torch.no_grad():
            for i in range(0, len(X), batch_size):
                batch = X_tensor[i : i + batch_size]
                outputs = self.forward(batch).squeeze()
                predictions.append(outputs.cpu().numpy())

        return np.concatenate(predictions)

    def save(self, path: str):
        """Save probe weights and configuration.

        Args:
            path: Path to save the probe
        """
        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        torch.save(
            {
                "state_dict": self.state_dict(),
                "input_dim": self.input_dim,
                "is_fitted": self.is_fitted,
                "class_name": self.__class__.__name__,
            },
            save_path,
        )
        logger.info(f"Saved probe to {save_path}")

    def load(self, path: str):
        """Load probe weights and configuration.

        Args:
            path: Path to load the probe from
        """
        checkpoint = torch.load(path, map_location=self.device)

        if checkpoint["input_dim"] != self.input_dim:
            raise ValueError(
                f"Saved probe has input_dim={checkpoint['input_dim']}, "
                f"but current probe has input_dim={self.input_dim}"
            )

        self.load_state_dict(checkpoint["state_dict"])
        self.is_fitted = checkpoint["is_fitted"]
        self.eval()

        logger.info(f"Loaded probe from {path}")

    def get_num_parameters(self) -> int:
        """Get total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
