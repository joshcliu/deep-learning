"""Probe implementations for uncertainty quantification.

This module provides probe architectures for predicting confidence
scores from frozen language model hidden states.

Available probes:
- LinearProbe: Single-layer logistic regression with temperature scaling
- MLPProbe: Multi-layer perceptron with configurable depth and width
- HierarchicalProbe: Multi-scale probe with token/span/semantic/global levels
- BaseProbe: Abstract base class for implementing custom probes

Usage:
    from src.probes import LinearProbe, MLPProbe, HierarchicalProbe

    # Create and train probe
    probe = LinearProbe(input_dim=4096, dropout=0.1)
    history = probe.fit(X_train, y_train, X_val, y_val)

    # Predict confidence scores
    confidences = probe.predict(X_test)

    # Or use MLP probe for stronger baseline
    mlp_probe = MLPProbe(input_dim=4096, hidden_dim=512, num_layers=2)
    mlp_probe.fit(X_train, y_train, X_val, y_val)
    mlp_confidences = mlp_probe.predict(X_test)

    # Or use hierarchical probe
    h_probe = HierarchicalProbe(input_dim=4096, hidden_dim=512)
    h_probe.fit(X_train, y_train, X_val, y_val)
    h_confidences = h_probe.predict(X_test)
"""

from .base import BaseProbe, ArrayLike
from .linear import LinearProbe
from .mlp import MLPProbe
from .hierarchical import HierarchicalProbe

__all__ = [
    "BaseProbe",
    "ArrayLike",
    "LinearProbe",
    "MLPProbe",
    "HierarchicalProbe",
]
