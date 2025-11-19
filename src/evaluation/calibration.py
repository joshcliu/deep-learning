"""Calibration visualization and post-hoc calibration methods."""

import logging
from pathlib import Path
from typing import Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression

from .metrics import compute_ece

logger = logging.getLogger(__name__)

# Set style
sns.set_style("whitegrid")


def plot_reliability_diagram(
    confidences: np.ndarray,
    labels: np.ndarray,
    num_bins: int = 10,
    title: str = "Reliability Diagram",
    save_path: Optional[Union[str, Path]] = None,
    show: bool = False,
) -> plt.Figure:
    """Plot reliability (calibration) diagram.

    Shows the relationship between predicted confidence and empirical accuracy.
    Well-calibrated models should align with the diagonal.

    Args:
        confidences: Confidence scores in [0, 1], shape (n,)
        labels: Ground truth binary labels {0, 1}, shape (n,)
        num_bins: Number of bins for calibration curve
        title: Plot title
        save_path: Path to save figure (optional)
        show: Whether to display the plot

    Returns:
        Matplotlib figure object

    Example:
        >>> fig = plot_reliability_diagram(
        ...     confidences, labels,
        ...     save_path="outputs/reliability.png"
        ... )
    """
    # Compute calibration curve
    prob_true, prob_pred = calibration_curve(
        labels, confidences, n_bins=num_bins, strategy='uniform'
    )

    # Compute ECE
    predictions = (confidences >= 0.5).astype(int)
    ece, _ = compute_ece(confidences, predictions, labels, num_bins=num_bins)

    # Create figure
    fig, ax = plt.subplots(figsize=(8, 8))

    # Plot perfect calibration line
    ax.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Perfect calibration')

    # Plot calibration curve
    ax.plot(prob_pred, prob_true, 'o-', linewidth=2, markersize=8, label='Model')

    # Shaded region showing miscalibration
    ax.fill_between(
        prob_pred, prob_true, prob_pred,
        alpha=0.2, color='red', label='Calibration gap'
    )

    # Formatting
    ax.set_xlabel('Predicted Confidence', fontsize=12)
    ax.set_ylabel('Empirical Accuracy', fontsize=12)
    ax.set_title(f'{title}\nECE: {ece:.4f}', fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.set_aspect('equal')

    plt.tight_layout()

    # Save if requested
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved reliability diagram to {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig


def plot_confidence_histogram(
    confidences: np.ndarray,
    labels: np.ndarray,
    num_bins: int = 20,
    title: str = "Confidence Distribution",
    save_path: Optional[Union[str, Path]] = None,
    show: bool = False,
) -> plt.Figure:
    """Plot histogram of confidence scores, separated by correctness.

    Args:
        confidences: Confidence scores in [0, 1], shape (n,)
        labels: Ground truth binary labels {0, 1}, shape (n,)
        num_bins: Number of histogram bins
        title: Plot title
        save_path: Path to save figure (optional)
        show: Whether to display the plot

    Returns:
        Matplotlib figure object
    """
    predictions = (confidences >= 0.5).astype(int)
    correct = predictions == labels

    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot histograms
    ax.hist(
        confidences[correct],
        bins=num_bins,
        alpha=0.6,
        label=f'Correct ({np.sum(correct)})',
        color='green',
        edgecolor='black',
    )
    ax.hist(
        confidences[~correct],
        bins=num_bins,
        alpha=0.6,
        label=f'Incorrect ({np.sum(~correct)})',
        color='red',
        edgecolor='black',
    )

    # Add vertical line at threshold 0.5
    ax.axvline(0.5, color='black', linestyle='--', linewidth=2, label='Threshold')

    # Formatting
    ax.set_xlabel('Confidence', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved confidence histogram to {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig


def plot_roc_curve(
    confidences: np.ndarray,
    labels: np.ndarray,
    title: str = "ROC Curve",
    save_path: Optional[Union[str, Path]] = None,
    show: bool = False,
) -> plt.Figure:
    """Plot Receiver Operating Characteristic (ROC) curve.

    Args:
        confidences: Confidence scores in [0, 1], shape (n,)
        labels: Ground truth binary labels {0, 1}, shape (n,)
        title: Plot title
        save_path: Path to save figure (optional)
        show: Whether to display the plot

    Returns:
        Matplotlib figure object
    """
    from sklearn.metrics import roc_auc_score, roc_curve

    # Compute ROC curve
    fpr, tpr, thresholds = roc_curve(labels, confidences)
    auroc = roc_auc_score(labels, confidences)

    # Create figure
    fig, ax = plt.subplots(figsize=(8, 8))

    # Plot ROC curve
    ax.plot(fpr, tpr, linewidth=2, label=f'AUROC = {auroc:.4f}')

    # Plot random classifier line
    ax.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Random')

    # Formatting
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.set_aspect('equal')

    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved ROC curve to {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig


class TemperatureScaling:
    """Post-hoc calibration via temperature scaling.

    Temperature scaling rescales logits by a learned temperature parameter
    to improve calibration without changing discrimination.

    Reference:
        Guo et al. "On Calibration of Modern Neural Networks" (ICML 2017)

    Example:
        >>> calibrator = TemperatureScaling()
        >>> calibrator.fit(val_logits, val_labels)
        >>> calibrated_probs = calibrator.transform(test_logits)
    """

    def __init__(self):
        """Initialize temperature scaling calibrator."""
        self.temperature = 1.0

    def fit(
        self,
        logits: np.ndarray,
        labels: np.ndarray,
        max_iter: int = 100,
    ) -> float:
        """Learn optimal temperature parameter.

        Args:
            logits: Raw model logits (before softmax), shape (n,)
            labels: Ground truth binary labels {0, 1}, shape (n,)
            max_iter: Maximum optimization iterations

        Returns:
            Optimal temperature value
        """
        from scipy.optimize import minimize

        # Define negative log-likelihood objective
        def nll(temperature):
            scaled_logits = logits / temperature
            probs = 1 / (1 + np.exp(-scaled_logits))  # Sigmoid
            probs = np.clip(probs, 1e-7, 1 - 1e-7)  # Numerical stability
            log_probs = labels * np.log(probs) + (1 - labels) * np.log(1 - probs)
            return -np.mean(log_probs)

        # Optimize temperature
        result = minimize(
            nll,
            x0=1.0,
            bounds=[(0.01, 100.0)],
            options={'maxiter': max_iter},
        )

        self.temperature = result.x[0]
        logger.info(f"Learned temperature: {self.temperature:.4f}")

        return self.temperature

    def transform(self, logits: np.ndarray) -> np.ndarray:
        """Apply temperature scaling to logits.

        Args:
            logits: Raw model logits, shape (n,)

        Returns:
            Calibrated probabilities, shape (n,)
        """
        scaled_logits = logits / self.temperature
        probs = 1 / (1 + np.exp(-scaled_logits))  # Sigmoid
        return probs

    def fit_transform(
        self,
        train_logits: np.ndarray,
        train_labels: np.ndarray,
        test_logits: np.ndarray,
    ) -> np.ndarray:
        """Fit on training data and transform test data.

        Args:
            train_logits: Training logits for fitting temperature
            train_labels: Training labels
            test_logits: Test logits to calibrate

        Returns:
            Calibrated test probabilities
        """
        self.fit(train_logits, train_labels)
        return self.transform(test_logits)


class PlattScaling:
    """Post-hoc calibration via Platt scaling (logistic regression).

    Fits a logistic regression model on top of confidence scores.

    Example:
        >>> calibrator = PlattScaling()
        >>> calibrator.fit(val_confidences, val_labels)
        >>> calibrated_probs = calibrator.transform(test_confidences)
    """

    def __init__(self):
        """Initialize Platt scaling calibrator."""
        self.model = LogisticRegression()

    def fit(
        self,
        confidences: np.ndarray,
        labels: np.ndarray,
    ) -> "PlattScaling":
        """Fit Platt scaling model.

        Args:
            confidences: Confidence scores, shape (n,)
            labels: Ground truth binary labels, shape (n,)

        Returns:
            self
        """
        X = confidences.reshape(-1, 1)
        self.model.fit(X, labels)
        logger.info("Fitted Platt scaling model")
        return self

    def transform(self, confidences: np.ndarray) -> np.ndarray:
        """Apply Platt scaling.

        Args:
            confidences: Confidence scores, shape (n,)

        Returns:
            Calibrated probabilities, shape (n,)
        """
        X = confidences.reshape(-1, 1)
        return self.model.predict_proba(X)[:, 1]

    def fit_transform(
        self,
        train_confidences: np.ndarray,
        train_labels: np.ndarray,
        test_confidences: np.ndarray,
    ) -> np.ndarray:
        """Fit and transform in one step.

        Args:
            train_confidences: Training confidences
            train_labels: Training labels
            test_confidences: Test confidences to calibrate

        Returns:
            Calibrated test probabilities
        """
        self.fit(train_confidences, train_labels)
        return self.transform(test_confidences)
