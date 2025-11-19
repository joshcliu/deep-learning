"""Calibration and discrimination metrics for uncertainty quantification."""

import logging
from typing import Dict, Optional, Tuple

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    auc,
    average_precision_score,
    brier_score_loss,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

logger = logging.getLogger(__name__)


def compute_ece(
    confidences: np.ndarray,
    predictions: np.ndarray,
    labels: np.ndarray,
    num_bins: int = 10,
) -> Tuple[float, Dict]:
    """Compute Expected Calibration Error (ECE).

    ECE measures the gap between predicted confidence and empirical accuracy.
    Well-calibrated systems should have ECE < 0.05 (5%).

    Args:
        confidences: Confidence scores in [0, 1], shape (n,)
        predictions: Binary predictions {0, 1}, shape (n,)
        labels: Ground truth binary labels {0, 1}, shape (n,)
        num_bins: Number of bins for calibration (default: 10)

    Returns:
        Tuple of (ece_value, bin_statistics_dict)
        bin_statistics contains: bin_edges, bin_accuracies, bin_confidences, bin_counts

    Example:
        >>> confidences = np.array([0.9, 0.8, 0.6, 0.4])
        >>> predictions = np.array([1, 1, 1, 0])
        >>> labels = np.array([1, 0, 1, 0])
        >>> ece, stats = compute_ece(confidences, predictions, labels)
        >>> print(f"ECE: {ece:.4f}")
    """
    # Validate inputs
    assert len(confidences) == len(predictions) == len(labels)
    assert np.all((confidences >= 0) & (confidences <= 1))
    assert np.all((predictions == 0) | (predictions == 1))
    assert np.all((labels == 0) | (labels == 1))

    # Create bins
    bin_edges = np.linspace(0, 1, num_bins + 1)
    bin_indices = np.digitize(confidences, bin_edges[:-1]) - 1
    bin_indices = np.clip(bin_indices, 0, num_bins - 1)

    # Compute statistics per bin
    ece = 0.0
    bin_accuracies = []
    bin_confidences = []
    bin_counts = []

    for bin_idx in range(num_bins):
        in_bin = bin_indices == bin_idx
        bin_count = np.sum(in_bin)

        if bin_count > 0:
            bin_accuracy = np.mean(predictions[in_bin] == labels[in_bin])
            bin_confidence = np.mean(confidences[in_bin])
            bin_weight = bin_count / len(confidences)

            # ECE is weighted average of absolute differences
            ece += bin_weight * np.abs(bin_accuracy - bin_confidence)

            bin_accuracies.append(bin_accuracy)
            bin_confidences.append(bin_confidence)
        else:
            bin_accuracies.append(np.nan)
            bin_confidences.append(np.nan)

        bin_counts.append(bin_count)

    stats = {
        "bin_edges": bin_edges,
        "bin_accuracies": np.array(bin_accuracies),
        "bin_confidences": np.array(bin_confidences),
        "bin_counts": np.array(bin_counts),
    }

    return ece, stats


def compute_brier(
    confidences: np.ndarray,
    labels: np.ndarray,
) -> float:
    """Compute Brier score.

    Brier score combines calibration and discrimination into a proper scoring rule.
    Lower is better. Values < 0.10 are strong, > 0.30 indicate problems.

    Args:
        confidences: Confidence scores in [0, 1], shape (n,)
        labels: Ground truth binary labels {0, 1}, shape (n,)

    Returns:
        Brier score value

    Example:
        >>> confidences = np.array([0.9, 0.8, 0.6, 0.4])
        >>> labels = np.array([1, 0, 1, 0])
        >>> brier = compute_brier(confidences, labels)
        >>> print(f"Brier: {brier:.4f}")
    """
    assert len(confidences) == len(labels)
    assert np.all((confidences >= 0) & (confidences <= 1))
    assert np.all((labels == 0) | (labels == 1))

    return float(brier_score_loss(labels, confidences))


def compute_auroc(
    confidences: np.ndarray,
    labels: np.ndarray,
) -> float:
    """Compute Area Under ROC Curve (AUROC).

    AUROC measures discrimination quality independently of calibration.
    Values > 0.80 are excellent, > 0.90 outstanding.

    Args:
        confidences: Confidence scores in [0, 1], shape (n,)
        labels: Ground truth binary labels {0, 1}, shape (n,)

    Returns:
        AUROC value

    Example:
        >>> confidences = np.array([0.9, 0.8, 0.6, 0.4])
        >>> labels = np.array([1, 1, 1, 0])
        >>> auroc = compute_auroc(confidences, labels)
        >>> print(f"AUROC: {auroc:.4f}")
    """
    assert len(confidences) == len(labels)
    assert np.all((confidences >= 0) & (confidences <= 1))

    # Check if labels are all same class (AUROC undefined)
    if len(np.unique(labels)) < 2:
        logger.warning("AUROC undefined: all labels are the same class")
        return np.nan

    return float(roc_auc_score(labels, confidences))


def compute_aupr(
    confidences: np.ndarray,
    labels: np.ndarray,
) -> float:
    """Compute Area Under Precision-Recall Curve (AUPR).

    AUPR is particularly useful for imbalanced datasets where positive class is rare.

    Args:
        confidences: Confidence scores in [0, 1], shape (n,)
        labels: Ground truth binary labels {0, 1}, shape (n,)

    Returns:
        AUPR value (average precision)

    Example:
        >>> confidences = np.array([0.9, 0.8, 0.6, 0.4])
        >>> labels = np.array([1, 1, 0, 0])
        >>> aupr = compute_aupr(confidences, labels)
        >>> print(f"AUPR: {aupr:.4f}")
    """
    assert len(confidences) == len(labels)
    assert np.all((confidences >= 0) & (confidences <= 1))

    # Check if labels are all same class
    if len(np.unique(labels)) < 2:
        logger.warning("AUPR undefined: all labels are the same class")
        return np.nan

    return float(average_precision_score(labels, confidences))


def compute_accuracy_at_coverage(
    confidences: np.ndarray,
    predictions: np.ndarray,
    labels: np.ndarray,
    coverage: float = 0.8,
) -> Tuple[float, float]:
    """Compute accuracy at a specified coverage level.

    Useful for selective prediction: what accuracy can we achieve if we
    abstain from the most uncertain predictions?

    Args:
        confidences: Confidence scores in [0, 1], shape (n,)
        predictions: Binary predictions {0, 1}, shape (n,)
        labels: Ground truth binary labels {0, 1}, shape (n,)
        coverage: Fraction of examples to retain (default: 0.8 = 80%)

    Returns:
        Tuple of (accuracy_at_coverage, threshold_used)

    Example:
        >>> confidences = np.array([0.9, 0.8, 0.6, 0.4])
        >>> predictions = np.array([1, 1, 1, 0])
        >>> labels = np.array([1, 1, 0, 0])
        >>> acc, threshold = compute_accuracy_at_coverage(
        ...     confidences, predictions, labels, coverage=0.75
        ... )
        >>> print(f"Accuracy at 75% coverage: {acc:.2%}")
    """
    assert 0 < coverage <= 1
    assert len(confidences) == len(predictions) == len(labels)

    # Sort by confidence (descending)
    sorted_indices = np.argsort(confidences)[::-1]
    num_to_keep = int(len(confidences) * coverage)

    # Keep top-k most confident predictions
    kept_indices = sorted_indices[:num_to_keep]
    threshold = confidences[sorted_indices[num_to_keep - 1]]

    # Compute accuracy on kept examples
    accuracy = accuracy_score(labels[kept_indices], predictions[kept_indices])

    return float(accuracy), float(threshold)


class CalibrationMetrics:
    """Comprehensive calibration metrics calculator.

    Computes all standard calibration and discrimination metrics in one pass.

    Example:
        >>> metrics = CalibrationMetrics(predictions, confidences, labels)
        >>> print(f"ECE: {metrics.ece():.4f}")
        >>> print(f"Brier: {metrics.brier():.4f}")
        >>> print(f"AUROC: {metrics.auroc():.4f}")
        >>> summary = metrics.compute_all()
    """

    def __init__(
        self,
        predictions: np.ndarray,
        confidences: np.ndarray,
        labels: np.ndarray,
        num_bins: int = 10,
    ):
        """Initialize metrics calculator.

        Args:
            predictions: Binary predictions {0, 1}, shape (n,)
            confidences: Confidence scores in [0, 1], shape (n,)
            labels: Ground truth binary labels {0, 1}, shape (n,)
            num_bins: Number of bins for ECE calculation
        """
        self.predictions = np.array(predictions)
        self.confidences = np.array(confidences)
        self.labels = np.array(labels)
        self.num_bins = num_bins

        # Validate
        assert len(predictions) == len(confidences) == len(labels)
        assert np.all((predictions == 0) | (predictions == 1))
        assert np.all((confidences >= 0) & (confidences <= 1))
        assert np.all((labels == 0) | (labels == 1))

        # Cache computed metrics
        self._ece_value = None
        self._ece_stats = None

    def ece(self) -> float:
        """Compute Expected Calibration Error."""
        if self._ece_value is None:
            self._ece_value, self._ece_stats = compute_ece(
                self.confidences, self.predictions, self.labels, self.num_bins
            )
        return self._ece_value

    def brier(self) -> float:
        """Compute Brier score."""
        return compute_brier(self.confidences, self.labels)

    def auroc(self) -> float:
        """Compute AUROC."""
        return compute_auroc(self.confidences, self.labels)

    def aupr(self) -> float:
        """Compute AUPR."""
        return compute_aupr(self.confidences, self.labels)

    def accuracy(self) -> float:
        """Compute overall accuracy."""
        return float(accuracy_score(self.labels, self.predictions))

    def compute_all(self) -> Dict[str, float]:
        """Compute all metrics.

        Returns:
            Dictionary with all metric values
        """
        return {
            "accuracy": self.accuracy(),
            "ece": self.ece(),
            "brier": self.brier(),
            "auroc": self.auroc(),
            "aupr": self.aupr(),
        }

    def get_ece_stats(self) -> Dict:
        """Get detailed ECE bin statistics.

        Returns:
            Dictionary with bin edges, accuracies, confidences, counts
        """
        if self._ece_stats is None:
            self.ece()  # Compute if not cached
        return self._ece_stats

    def get_roc_curve(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Get ROC curve data.

        Returns:
            Tuple of (fpr, tpr, thresholds)
        """
        if len(np.unique(self.labels)) < 2:
            return np.array([]), np.array([]), np.array([])
        return roc_curve(self.labels, self.confidences)

    def get_pr_curve(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Get Precision-Recall curve data.

        Returns:
            Tuple of (precision, recall, thresholds)
        """
        if len(np.unique(self.labels)) < 2:
            return np.array([]), np.array([]), np.array([])
        return precision_recall_curve(self.labels, self.confidences)
