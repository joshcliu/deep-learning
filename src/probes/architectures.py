"""Novel probe network architectures for confidence prediction.

These architectures are designed specifically for probing LLM hidden states
to predict answer correctness. Each addresses different hypotheses about
how uncertainty information is encoded in hidden states.

Usage:
    from src.probes import CalibratedProbe
    from src.probes.architectures import build_attention_network

    network = build_attention_network(input_dim=4096)
    probe = CalibratedProbe(network=network)
"""

from __future__ import annotations

import math
from typing import Optional

import torch
from torch import nn
import torch.nn.functional as F


# =============================================================================
# 1. ATTENTION PROBE
# =============================================================================
# Hypothesis: Different "regions" of the hidden state encode different information.
# Self-attention can learn which regions are most relevant for uncertainty.


class ChunkedSelfAttention(nn.Module):
    """Self-attention over chunks of the hidden state.

    Splits the hidden state into chunks (treating them like tokens),
    applies self-attention, then aggregates for prediction.
    """

    def __init__(
        self,
        input_dim: int,
        num_chunks: int = 16,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        assert input_dim % num_chunks == 0, "input_dim must be divisible by num_chunks"

        self.num_chunks = num_chunks
        self.chunk_dim = input_dim // num_chunks

        # Project chunks to attention dimension
        self.proj = nn.Linear(self.chunk_dim, self.chunk_dim)

        # Multi-head self-attention
        self.attention = nn.MultiheadAttention(
            embed_dim=self.chunk_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        # Output projection
        self.output = nn.Sequential(
            nn.LayerNorm(self.chunk_dim),
            nn.Linear(self.chunk_dim, 1),
            nn.Sigmoid(),
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, input_dim)
        batch_size = x.shape[0]

        # Reshape to chunks: (batch, num_chunks, chunk_dim)
        x = x.view(batch_size, self.num_chunks, self.chunk_dim)

        # Project
        x = self.proj(x)
        x = self.dropout(x)

        # Self-attention
        attn_out, _ = self.attention(x, x, x)

        # Mean pool over chunks
        pooled = attn_out.mean(dim=1)  # (batch, chunk_dim)

        # Output
        return self.output(pooled)


def build_attention_network(
    input_dim: int,
    num_chunks: int = 16,
    num_heads: int = 4,
    dropout: float = 0.1,
) -> nn.Module:
    """Build attention-based probe network.

    Treats the hidden state as a sequence of chunks and applies self-attention
    to learn which regions are most relevant for confidence prediction.

    Args:
        input_dim: Hidden state dimension (must be divisible by num_chunks).
        num_chunks: Number of chunks to split hidden state into.
        num_heads: Number of attention heads.
        dropout: Dropout probability.

    Returns:
        Network that maps (batch, input_dim) -> (batch, 1).
    """
    return ChunkedSelfAttention(input_dim, num_chunks, num_heads, dropout)


# =============================================================================
# 2. RESIDUAL MLP PROBE
# =============================================================================
# Hypothesis: The relationship between hidden state and correctness is complex
# and non-linear. Deeper networks with skip connections can capture this.


class ResidualBlock(nn.Module):
    """Residual block with pre-norm."""

    def __init__(self, dim: int, dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.ff(self.norm(x))


def build_residual_network(
    input_dim: int,
    hidden_dim: int = 512,
    num_blocks: int = 3,
    dropout: float = 0.1,
) -> nn.Module:
    """Build deep residual MLP probe.

    Uses pre-norm residual blocks for stable deep training.

    Args:
        input_dim: Hidden state dimension.
        hidden_dim: Dimension of residual blocks.
        num_blocks: Number of residual blocks.
        dropout: Dropout probability.

    Returns:
        Network that maps (batch, input_dim) -> (batch, 1).
    """
    layers = [
        nn.Dropout(dropout),
        nn.Linear(input_dim, hidden_dim),
        nn.GELU(),
    ]

    for _ in range(num_blocks):
        layers.append(ResidualBlock(hidden_dim, dropout))

    layers.extend([
        nn.LayerNorm(hidden_dim),
        nn.Linear(hidden_dim, 1),
        nn.Sigmoid(),
    ])

    return nn.Sequential(*layers)


# =============================================================================
# 3. BOTTLENECK PROBE
# =============================================================================
# Hypothesis: The uncertainty signal is low-rank and can be extracted by
# compressing the hidden state to a small bottleneck dimension.


def build_bottleneck_network(
    input_dim: int,
    bottleneck_dim: int = 64,
    hidden_dim: int = 256,
    dropout: float = 0.1,
) -> nn.Module:
    """Build bottleneck (autoencoder-style) probe.

    Compresses hidden state to a small bottleneck, forcing the network
    to extract the most salient features for confidence prediction.

    Args:
        input_dim: Hidden state dimension.
        bottleneck_dim: Compressed representation dimension.
        hidden_dim: Intermediate dimension before bottleneck.
        dropout: Dropout probability.

    Returns:
        Network that maps (batch, input_dim) -> (batch, 1).
    """
    return nn.Sequential(
        nn.Dropout(dropout),
        # Compress
        nn.Linear(input_dim, hidden_dim),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, bottleneck_dim),
        nn.GELU(),
        # Predict from bottleneck
        nn.Linear(bottleneck_dim, 1),
        nn.Sigmoid(),
    )


# =============================================================================
# 4. MULTI-HEAD PROBE
# =============================================================================
# Hypothesis: Different aspects of uncertainty might be captured by different
# "expert" heads. Aggregating multiple predictions improves calibration.


class MultiHeadProbeNetwork(nn.Module):
    """Multiple prediction heads with learned aggregation."""

    def __init__(
        self,
        input_dim: int,
        num_heads: int = 4,
        head_dim: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_heads = num_heads

        # Each head is a small MLP
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(input_dim, head_dim),
                nn.ReLU(),
                nn.Linear(head_dim, 1),
            )
            for _ in range(num_heads)
        ])

        # Learned aggregation weights
        self.gate = nn.Sequential(
            nn.Linear(input_dim, num_heads),
            nn.Softmax(dim=-1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Get predictions from each head: (batch, num_heads)
        head_preds = torch.cat([head(x) for head in self.heads], dim=-1)

        # Get aggregation weights: (batch, num_heads)
        weights = self.gate(x)

        # Weighted sum
        aggregated = (head_preds * weights).sum(dim=-1, keepdim=True)

        return torch.sigmoid(aggregated)


def build_multihead_network(
    input_dim: int,
    num_heads: int = 4,
    head_dim: int = 128,
    dropout: float = 0.1,
) -> nn.Module:
    """Build multi-head probe with learned aggregation.

    Multiple independent "expert" heads make predictions, then a gating
    mechanism learns to weight them based on the input.

    Args:
        input_dim: Hidden state dimension.
        num_heads: Number of prediction heads.
        head_dim: Hidden dimension within each head.
        dropout: Dropout probability.

    Returns:
        Network that maps (batch, input_dim) -> (batch, 1).
    """
    return MultiHeadProbeNetwork(input_dim, num_heads, head_dim, dropout)


# =============================================================================
# 5. GATED PROBE (GLU-style)
# =============================================================================
# Hypothesis: Gating mechanisms can learn to filter out irrelevant dimensions,
# focusing on the parts of the hidden state that encode uncertainty.


class GatedLinearUnit(nn.Module):
    """GLU activation: x * sigmoid(gate(x))."""

    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.gate = nn.Linear(input_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x) * torch.sigmoid(self.gate(x))


def build_gated_network(
    input_dim: int,
    hidden_dim: int = 256,
    num_layers: int = 2,
    dropout: float = 0.1,
) -> nn.Module:
    """Build gated (GLU-style) probe network.

    Uses Gated Linear Units to selectively filter information,
    potentially learning to focus on uncertainty-relevant dimensions.

    Args:
        input_dim: Hidden state dimension.
        hidden_dim: Hidden layer dimension.
        num_layers: Number of gated layers.
        dropout: Dropout probability.

    Returns:
        Network that maps (batch, input_dim) -> (batch, 1).
    """
    layers = [nn.Dropout(dropout)]

    current_dim = input_dim
    for i in range(num_layers):
        out_dim = hidden_dim if i < num_layers - 1 else hidden_dim
        layers.append(GatedLinearUnit(current_dim, out_dim))
        layers.append(nn.Dropout(dropout))
        current_dim = out_dim

    layers.extend([
        nn.Linear(hidden_dim, 1),
        nn.Sigmoid(),
    ])

    return nn.Sequential(*layers)


# =============================================================================
# 6. SPARSE TOP-K PROBE
# =============================================================================
# Hypothesis: The uncertainty signal is concentrated in a small subset of
# dimensions. Selecting top-k most activated dimensions could help.


class TopKSparseNetwork(nn.Module):
    """Network that learns to weight dimensions by importance.

    Uses a differentiable soft weighting mechanism during training,
    with optional hard top-k selection at inference for interpretability.

    Note: The original implementation using torch.topk had a bug where
    importance weights couldn't learn (indices are non-differentiable).
    This version uses soft attention weights that ARE differentiable.
    """

    def __init__(
        self,
        input_dim: int,
        k: int = 256,
        hidden_dim: int = 128,
        dropout: float = 0.1,
        temperature: float = 1.0,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.k = min(k, input_dim)
        self.temperature = temperature

        # Learnable importance scores for each dimension
        self.importance = nn.Parameter(torch.zeros(input_dim))

        # Process weighted dimensions (use full input_dim since we weight, not select)
        self.network = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Soft weighting: sigmoid gives importance in [0, 1] for each dimension
        # This is DIFFERENTIABLE, so importance weights can learn
        weights = torch.sigmoid(self.importance / self.temperature)

        # Weight the input dimensions
        x_weighted = x * weights.unsqueeze(0)

        return self.network(x_weighted)

    def get_top_k_indices(self) -> torch.Tensor:
        """Get indices of top-k most important dimensions (for interpretability)."""
        _, indices = torch.topk(self.importance, self.k)
        return indices

    def get_importance_scores(self) -> torch.Tensor:
        """Get importance scores for all dimensions."""
        return torch.sigmoid(self.importance).detach()


def build_sparse_network(
    input_dim: int,
    k: int = 256,
    hidden_dim: int = 128,
    dropout: float = 0.1,
    temperature: float = 1.0,
) -> nn.Module:
    """Build sparse dimension-weighted probe network.

    Learns importance weights for each dimension. Uses soft (differentiable)
    weighting during training, with methods to extract top-k for interpretability.

    Args:
        input_dim: Hidden state dimension.
        k: Number of top dimensions to report for interpretability.
        hidden_dim: Hidden layer dimension.
        dropout: Dropout probability.
        temperature: Temperature for importance sigmoid (lower = sharper selection).

    Returns:
        Network that maps (batch, input_dim) -> (batch, 1).
    """
    return TopKSparseNetwork(input_dim, k, hidden_dim, dropout, temperature)


# =============================================================================
# 7. HETEROSCEDASTIC PROBE
# =============================================================================
# Hypothesis: Different examples have different inherent uncertainty.
# Learning per-example "aleatoric" uncertainty could improve calibration.


class HeteroscedasticNetwork(nn.Module):
    """Network that predicts both mean and variance (heteroscedastic regression)."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()

        # Shared backbone
        self.backbone = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Mean head (logit)
        self.mean_head = nn.Linear(hidden_dim, 1)

        # Log-variance head (for numerical stability)
        self.logvar_head = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)

        mean = self.mean_head(features)
        logvar = self.logvar_head(features)

        # During inference, we sample or use the mean
        # For confidence, we want the probability, accounting for uncertainty
        # Using the "reparameterization" style but deterministically:
        # confidence = sigmoid(mean / sqrt(1 + exp(logvar) * pi/8))
        # This approximates the expected sigmoid under Gaussian noise

        var = torch.exp(logvar)
        # Probit approximation scaling
        scale = torch.sqrt(1 + var * (math.pi / 8))
        adjusted_logit = mean / scale

        return torch.sigmoid(adjusted_logit)


def build_heteroscedastic_network(
    input_dim: int,
    hidden_dim: int = 256,
    dropout: float = 0.1,
) -> nn.Module:
    """Build heteroscedastic probe network.

    Predicts both a confidence logit and per-example uncertainty,
    which can improve calibration for examples with varying difficulty.

    Args:
        input_dim: Hidden state dimension.
        hidden_dim: Hidden layer dimension.
        dropout: Dropout probability.

    Returns:
        Network that maps (batch, input_dim) -> (batch, 1).
    """
    return HeteroscedasticNetwork(input_dim, hidden_dim, dropout)


# =============================================================================
# 8. CONTRASTIVE-STYLE PROBE (Feature Interaction)
# =============================================================================
# Hypothesis: Uncertainty is encoded in the *relationships* between different
# parts of the hidden state, not just the values themselves.


class BilinearInteractionNetwork(nn.Module):
    """Network with explicit bilinear feature interactions."""

    def __init__(
        self,
        input_dim: int,
        num_factors: int = 32,
        hidden_dim: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()

        # Split input into two halves for interaction
        self.half_dim = input_dim // 2

        # Low-rank bilinear interaction: x1^T W x2 where W = U V^T
        self.U = nn.Linear(self.half_dim, num_factors, bias=False)
        self.V = nn.Linear(self.half_dim, num_factors, bias=False)

        # Also include direct features
        self.linear = nn.Linear(input_dim, hidden_dim)

        # Combine interaction and direct features
        self.output = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim + num_factors, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Split into halves
        x1 = x[:, :self.half_dim]
        x2 = x[:, self.half_dim:2*self.half_dim]

        # Bilinear interaction
        interaction = self.U(x1) * self.V(x2)  # (batch, num_factors)

        # Direct features
        direct = F.relu(self.linear(self.dropout(x)))

        # Combine
        combined = torch.cat([direct, interaction], dim=-1)

        return self.output(combined)


def build_bilinear_network(
    input_dim: int,
    num_factors: int = 32,
    hidden_dim: int = 128,
    dropout: float = 0.1,
) -> nn.Module:
    """Build bilinear feature interaction probe.

    Explicitly models interactions between different parts of the hidden state,
    capturing relationships that might encode uncertainty.

    Args:
        input_dim: Hidden state dimension.
        num_factors: Rank of bilinear interaction.
        hidden_dim: Hidden layer dimension.
        dropout: Dropout probability.

    Returns:
        Network that maps (batch, input_dim) -> (batch, 1).
    """
    return BilinearInteractionNetwork(input_dim, num_factors, hidden_dim, dropout)

class ContrastiveProbe(nn.Module):
    def __init__(self, input_dim, spurious_directions):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)
        self.spurious_directions = spurious_directions  # (num_spurious, input_dim)
    
    def forward(self, x):
        return torch.sigmoid(self.linear(x))
    
    def orthogonality_loss(self):
        """Penalize overlap with spurious directions"""
        w = self.linear.weight  # (1, input_dim)
        overlaps = torch.matmul(w, self.spurious_directions.T)  # (1, num_spurious)
        return torch.sum(overlaps ** 2)  # Want this to be zero
    
    def loss(self, predictions, labels):
        bce = F.binary_cross_entropy(predictions, labels)
        ortho = 0.1 * self.orthogonality_loss()
        return bce + ortho
    
def build_contrastive_network(
    input_dim: int,
    spurious_directions: torch.Tensor,
    dropout: float = 0.1,
) -> nn.Module:
    """
    Build a contrastive probe that avoids projecting onto spurious directions.

    Args:
        input_dim: Feature dimension of hidden states.
        spurious_directions: Tensor of shape (num_spurious, input_dim)
                             representing directions the probe should avoid.
        dropout: Dropout applied before the probe (optional).

    Returns:
        nn.Module implementing a contrastive probe with orthogonality regularization.
    """
    class ContrastiveProbeWrapper(nn.Module):
        def __init__(self):
            super().__init__()
            self.dropout = nn.Dropout(dropout)
            self.probe = ContrastiveProbe(input_dim, spurious_directions)

        def forward(self, x):
            x = self.dropout(x)
            return self.probe(x)

        def loss(self, predictions, labels):
            return self.probe.loss(predictions, labels)

    return ContrastiveProbeWrapper()


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Network builders
    "build_attention_network",
    "build_residual_network",
    "build_bottleneck_network",
    "build_multihead_network",
    "build_gated_network",
    "build_sparse_network",
    "build_heteroscedastic_network",
    "build_bilinear_network",
    "build_contrastive_network",
    # Network classes (for customization)
    "ChunkedSelfAttention",
    "ResidualBlock",
    "MultiHeadProbeNetwork",
    "GatedLinearUnit",
    "TopKSparseNetwork",
    "HeteroscedasticNetwork",
    "BilinearInteractionNetwork",
    "ContrastiveProbe",
]
