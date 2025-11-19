"""Selective prediction analysis and risk-coverage curves."""

import logging
from pathlib import Path
from typing import List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import accuracy_score

logger = logging.getLogger(__name__)

sns.set_style("whitegrid")


def compute_coverage_accuracy_curve(
    confidences: np.ndarray,
    predictions: np.ndarray,
    labels: np.ndarray,
    num_points: int = 100,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute coverage-accuracy curve for selective prediction.

    At each coverage level, compute accuracy on the most confident predictions.

    Args:
        confidences: Confidence scores in [0, 1], shape (n,)
        predictions: Binary predictions {0, 1}, shape (n,)
        labels: Ground truth binary labels {0, 1}, shape (n,)
        num_points: Number of points on the curve

    Returns:
        Tuple of (coverage_levels, accuracies, thresholds)

    Example:
        >>> coverage, accuracy, thresholds = compute_coverage_accuracy_curve(
        ...     confidences, predictions, labels
        ... )
    """
    # Sort by confidence (descending)
    sorted_indices = np.argsort(confidences)[::-1]
    sorted_predictions = predictions[sorted_indices]
    sorted_labels = labels[sorted_indices]

    # Compute accuracies at different coverage levels
    coverage_levels = np.linspace(0.01, 1.0, num_points)
    accuracies = []
    thresholds = []

    for coverage in coverage_levels:
        num_to_keep = max(1, int(len(confidences) * coverage))
        kept_predictions = sorted_predictions[:num_to_keep]
        kept_labels = sorted_labels[:num_to_keep]

        accuracy = accuracy_score(kept_labels, kept_predictions)
        accuracies.append(accuracy)

        # Record threshold
        threshold = confidences[sorted_indices[num_to_keep - 1]]
        thresholds.append(threshold)

    return (
        np.array(coverage_levels),
        np.array(accuracies),
        np.array(thresholds),
    )


def plot_coverage_accuracy_curve(
    confidences: np.ndarray,
    predictions: np.ndarray,
    labels: np.ndarray,
    title: str = "Coverage vs Accuracy",
    save_path: Optional[Union[str, Path]] = None,
    show: bool = False,
) -> plt.Figure:
    """Plot coverage vs accuracy curve.

    Shows the trade-off between coverage (% of predictions retained)
    and accuracy when abstaining from uncertain predictions.

    Args:
        confidences: Confidence scores in [0, 1], shape (n,)
        predictions: Binary predictions {0, 1}, shape (n,)
        labels: Ground truth binary labels {0, 1}, shape (n,)
        title: Plot title
        save_path: Path to save figure (optional)
        show: Whether to display the plot

    Returns:
        Matplotlib figure object
    """
    coverage, accuracy, _ = compute_coverage_accuracy_curve(
        confidences, predictions, labels
    )

    # Compute baseline accuracy (no abstention)
    baseline_accuracy = accuracy_score(labels, predictions)

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot curve
    ax.plot(coverage, accuracy, linewidth=2.5, color='blue', label='Selective prediction')

    # Plot baseline
    ax.axhline(
        baseline_accuracy,
        color='red',
        linestyle='--',
        linewidth=2,
        label=f'Baseline (no abstention): {baseline_accuracy:.3f}'
    )

    # Formatting
    ax.set_xlabel('Coverage (% of predictions retained)', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='lower left', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])

    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved coverage-accuracy curve to {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig


def find_coverage_for_accuracy(
    confidences: np.ndarray,
    predictions: np.ndarray,
    labels: np.ndarray,
    target_accuracy: float = 0.9,
) -> Tuple[float, float]:
    """Find maximum coverage that achieves target accuracy.

    Useful for production scenarios: "What % of queries can we answer
    while maintaining 90% accuracy?"

    Args:
        confidences: Confidence scores in [0, 1], shape (n,)
        predictions: Binary predictions {0, 1}, shape (n,)
        labels: Ground truth binary labels {0, 1}, shape (n,)
        target_accuracy: Desired accuracy level (default: 0.9)

    Returns:
        Tuple of (max_coverage, confidence_threshold)

    Example:
        >>> coverage, threshold = find_coverage_for_accuracy(
        ...     confidences, predictions, labels, target_accuracy=0.9
        ... )
        >>> print(f"Can achieve 90% accuracy at {coverage:.1%} coverage")
    """
    coverage, accuracy, thresholds = compute_coverage_accuracy_curve(
        confidences, predictions, labels, num_points=1000
    )

    # Find first point where accuracy >= target
    valid_indices = accuracy >= target_accuracy
    if not np.any(valid_indices):
        logger.warning(
            f"Cannot achieve target accuracy {target_accuracy:.2%}. "
            f"Maximum accuracy: {accuracy.max():.2%}"
        )
        return 0.0, 1.0

    # Get maximum coverage at target accuracy
    max_coverage_idx = np.where(valid_indices)[0][-1]
    max_coverage = coverage[max_coverage_idx]
    threshold = thresholds[max_coverage_idx]

    return float(max_coverage), float(threshold)


def compute_rejection_metrics(
    confidences: np.ndarray,
    predictions: np.ndarray,
    labels: np.ndarray,
    coverage_levels: Optional[List[float]] = None,
) -> dict:
    """Compute comprehensive rejection/selective prediction metrics.

    Args:
        confidences: Confidence scores in [0, 1], shape (n,)
        predictions: Binary predictions {0, 1}, shape (n,)
        labels: Ground truth binary labels {0, 1}, shape (n,)
        coverage_levels: Specific coverage levels to report (default: [0.7, 0.8, 0.9, 0.95])

    Returns:
        Dictionary with metrics at different coverage levels
    """
    if coverage_levels is None:
        coverage_levels = [0.7, 0.8, 0.9, 0.95]

    baseline_accuracy = accuracy_score(labels, predictions)

    results = {
        "baseline_accuracy": baseline_accuracy,
        "coverage_levels": {},
    }

    # Sort by confidence
    sorted_indices = np.argsort(confidences)[::-1]

    for coverage in coverage_levels:
        num_to_keep = max(1, int(len(confidences) * coverage))
        kept_indices = sorted_indices[:num_to_keep]

        accuracy = accuracy_score(labels[kept_indices], predictions[kept_indices])
        threshold = confidences[sorted_indices[num_to_keep - 1]]

        results["coverage_levels"][coverage] = {
            "accuracy": float(accuracy),
            "threshold": float(threshold),
            "num_predictions": num_to_keep,
            "accuracy_gain": float(accuracy - baseline_accuracy),
        }

    return results


def plot_prediction_rejection_ratio(
    confidences: np.ndarray,
    predictions: np.ndarray,
    labels: np.ndarray,
    title: str = "Prediction-Rejection Ratio",
    save_path: Optional[Union[str, Path]] = None,
    show: bool = False,
) -> plt.Figure:
    """Plot prediction-rejection ratio curve.

    Shows how accuracy improves as we reject (abstain from) predictions.

    Args:
        confidences: Confidence scores in [0, 1], shape (n,)
        predictions: Binary predictions {0, 1}, shape (n,)
        labels: Ground truth binary labels {0, 1}, shape (n,)
        title: Plot title
        save_path: Path to save figure (optional)
        show: Whether to display the plot

    Returns:
        Matplotlib figure object
    """
    coverage, accuracy, _ = compute_coverage_accuracy_curve(
        confidences, predictions, labels
    )

    # Convert coverage to rejection rate
    rejection_rate = 1 - coverage

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot curve
    ax.plot(rejection_rate, accuracy, linewidth=2.5, color='purple')

    # Formatting
    ax.set_xlabel('Rejection Rate (% of predictions abstained)', fontsize=12)
    ax.set_ylabel('Accuracy on Retained Predictions', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])

    # Invert x-axis so higher rejection is on the right
    ax.invert_xaxis()

    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved prediction-rejection curve to {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig
