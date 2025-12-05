"""Probe implementations for uncertainty quantification.

This module provides probe architectures for predicting confidence
scores from frozen language model hidden states.

Available probes:
- LinearProbe: Single-layer logistic regression with temperature scaling
- CalibratedProbe: Probe trained with Brier score for calibrated confidence
- MLPProbe: Multi-layer perceptron with configurable depth and width
- HierarchicalProbe: Multi-scale probe with token/span/semantic/global levels
- BaseProbe: Abstract base class for implementing custom probes

Novel architectures (via CalibratedProbe + custom network):
- AttentionProbe: Self-attention over hidden state chunks
- ResidualProbe: Deep MLP with skip connections
- BottleneckProbe: Compression to low-rank representation
- MultiHeadProbe: Multiple expert heads with learned aggregation
- GatedProbe: GLU-style gating for feature selection
- SparseProbe: Top-k dimension selection (interpretable)
- HeteroscedasticProbe: Per-example uncertainty estimation
- BilinearProbe: Explicit feature interactions
- HierarchicalProbe: Multi-scale hierarchical processing (fine → coarse → global)

Usage:
    from src.probes import LinearProbe, CalibratedProbe, MLPProbe, HierarchicalProbe

    # Standard probe with BCE loss
    probe = LinearProbe(input_dim=4096, dropout=0.1)
    history = probe.fit(X_train, y_train, X_val, y_val)

    # Calibrated probe with Brier score loss (recommended)
    probe = CalibratedProbe(input_dim=4096, hidden_dim=256)
    history = probe.fit(X_train, y_train, X_val, y_val)
    confidences = probe.predict(X_test)

    # Novel architecture: Attention probe
    from src.probes.architectures import build_attention_network
    network = build_attention_network(input_dim=4096, num_chunks=16)
    probe = CalibratedProbe(network=network)

    # Novel architecture: Hierarchical multi-scale probe
    from src.probes.architectures import build_hierarchical_network
    network = build_hierarchical_network(input_dim=4096, num_chunks=16)
    probe = CalibratedProbe(network=network)

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
from .calibrated_probe import CalibratedProbe, build_default_network
from .mlp import MLPProbe
from .hierarchical import HierarchicalProbe
from .architectures import (
    build_attention_network,
    build_residual_network,
    build_bottleneck_network,
    build_multihead_network,
    build_gated_network,
    build_sparse_network,
    build_heteroscedastic_network,
    build_bilinear_network,
    build_hierarchical_network,
)

__all__ = [
    # Base
    "BaseProbe",
    "ArrayLike",
    # Probe classes
    "LinearProbe",
    "CalibratedProbe",
    "MLPProbe",
    "HierarchicalProbe",
    # Network builders
    "build_default_network",
    "build_attention_network",
    "build_residual_network",
    "build_bottleneck_network",
    "build_multihead_network",
    "build_gated_network",
    "build_sparse_network",
    "build_heteroscedastic_network",
    "build_bilinear_network",
    "build_hierarchical_network",
]
