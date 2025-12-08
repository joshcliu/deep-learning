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
# 9. HIERARCHICAL PROBE
# =============================================================================
# Hypothesis: Uncertainty information exists at multiple levels of granularity.
# Process hidden states hierarchically: fine-grained → coarse-grained → global.


class HierarchicalNetwork(nn.Module):
    """Hierarchical multi-scale confidence network.

    Processes the hidden state at multiple levels of granularity:
    1. Fine-grained: Individual chunks (token-level analogy)
    2. Mid-level: Aggregated chunks (span-level analogy)
    3. Semantic: Broader aggregation (semantic-level analogy)
    4. Global: Final aggregation

    Each level builds on the previous, creating a hierarchy of representations.
    """

    def __init__(
        self,
        input_dim: int,
        num_chunks: int = 16,
        hidden_dim: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        assert input_dim % num_chunks == 0, "input_dim must be divisible by num_chunks"

        self.num_chunks = num_chunks
        self.chunk_dim = input_dim // num_chunks
        self.hidden_dim = hidden_dim

        # Fine-grained processing (per-chunk)
        self.fine_processor = nn.Sequential(
            nn.Linear(self.chunk_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
        )

        # Mid-level aggregation (attention over chunks)
        self.mid_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim // 2,
            num_heads=4,
            dropout=dropout,
            batch_first=True,
        )

        # Semantic-level processing
        self.semantic_processor = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
        )

        # Global aggregation
        self.global_processor = nn.Sequential(
            nn.Linear(hidden_dim // 2 * 2, hidden_dim),  # Concat mid + semantic
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, input_dim)
        batch_size = x.shape[0]

        # Split into chunks: (batch, num_chunks, chunk_dim)
        chunks = x.view(batch_size, self.num_chunks, self.chunk_dim)

        # Fine-grained: Process each chunk independently
        fine_features = self.fine_processor(chunks)  # (batch, num_chunks, hidden//2)

        # Mid-level: Aggregate chunks with attention
        mid_features, _ = self.mid_attention(
            fine_features, fine_features, fine_features
        )  # (batch, num_chunks, hidden//2)

        # Semantic-level: Mean pooling + processing
        semantic_pooled = mid_features.mean(dim=1)  # (batch, hidden//2)
        semantic_features = self.semantic_processor(
            semantic_pooled
        )  # (batch, hidden//2)

        # Also get mid-level summary
        mid_pooled = mid_features.mean(dim=1)  # (batch, hidden//2)

        # Global: Combine mid and semantic levels
        global_input = torch.cat([mid_pooled, semantic_features], dim=-1)
        confidence = self.global_processor(global_input)  # (batch, 1)

        return confidence


def build_hierarchical_network(
    input_dim: int,
    num_chunks: int = 16,
    hidden_dim: int = 256,
    dropout: float = 0.1,
) -> nn.Module:
    """Build hierarchical multi-scale probe network.

    Processes hidden states at multiple levels of granularity:
    fine-grained → mid-level → semantic → global.

    This captures the hypothesis that uncertainty is encoded at different scales,
    and that hierarchical processing can better extract these signals.

    Args:
        input_dim: Hidden state dimension (must be divisible by num_chunks).
        num_chunks: Number of chunks to split hidden state into (default: 16).
        hidden_dim: Hidden layer dimension for processing (default: 256).
        dropout: Dropout probability (default: 0.1).

    Returns:
        Network that maps (batch, input_dim) -> (batch, 1).

    Example:
        >>> from src.probes import CalibratedProbe
        >>> from src.probes.architectures import build_hierarchical_network
        >>> network = build_hierarchical_network(input_dim=4096, num_chunks=16)
        >>> probe = CalibratedProbe(network=network)
        >>> probe.fit(X_train, y_train, X_val, y_val)
    """
    return HierarchicalNetwork(input_dim, num_chunks, hidden_dim, dropout)


# =============================================================================
# LAYER ENSEMBLE PROBE
# =============================================================================


class LayerEnsembleNetwork(nn.Module):
    """Layer-wise ensemble probe.

    Processes representations from multiple layers independently and learns
    optimal weights to combine their predictions.

    Inductive bias: Different layers encode complementary uncertainty signals.
    Early layers may capture lexical/syntactic uncertainty, while deeper layers
    capture semantic/task-specific confidence.

    Args:
        input_dim: Total input dimension (num_layers * hidden_dim per layer)
        num_layers: Number of layers to ensemble
        hidden_dim_per_layer: Hidden dimension from each layer
        layer_probe_hidden: Hidden dimension for per-layer probes (None = linear)
        dropout: Dropout probability
    """

    def __init__(
        self,
        input_dim: int,
        num_layers: int = 4,
        hidden_dim_per_layer: int = None,
        layer_probe_hidden: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.num_layers = num_layers

        # Auto-detect hidden_dim_per_layer from input_dim if not provided
        if hidden_dim_per_layer is None:
            assert input_dim % num_layers == 0, \
                f"input_dim {input_dim} must be divisible by num_layers {num_layers}"
            hidden_dim_per_layer = input_dim // num_layers

        self.hidden_dim_per_layer = hidden_dim_per_layer

        # Per-layer lightweight probes
        self.layer_probes = nn.ModuleList()
        for _ in range(num_layers):
            if layer_probe_hidden is None:
                # Linear probe for this layer
                probe = nn.Sequential(
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim_per_layer, 1),
                )
            else:
                # MLP probe for this layer
                probe = nn.Sequential(
                    nn.Linear(hidden_dim_per_layer, layer_probe_hidden),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(layer_probe_hidden, 1),
                )
            self.layer_probes.append(probe)

        # Learnable ensemble weights (initialized uniformly)
        self.ensemble_weights = nn.Parameter(torch.ones(num_layers) / num_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (batch, num_layers * hidden_dim_per_layer)
               OR (batch, num_layers, hidden_dim_per_layer)

        Returns:
            Confidence scores of shape (batch, 1)
        """
        batch_size = x.shape[0]

        # Reshape if needed: (batch, num_layers * hidden) -> (batch, num_layers, hidden)
        if x.dim() == 2:
            x = x.view(batch_size, self.num_layers, self.hidden_dim_per_layer)

        # Get prediction from each layer
        layer_logits = []
        for i in range(self.num_layers):
            layer_input = x[:, i, :]  # (batch, hidden_dim_per_layer)
            logit = self.layer_probes[i](layer_input)  # (batch, 1)
            layer_logits.append(logit)

        # Stack: (batch, num_layers, 1)
        layer_logits = torch.stack(layer_logits, dim=1)

        # Normalize weights with softmax (ensure they sum to 1)
        weights = torch.softmax(self.ensemble_weights, dim=0)  # (num_layers,)

        # Weighted average: (batch, num_layers, 1) * (num_layers,) -> (batch, 1)
        weighted_logits = (layer_logits * weights.view(1, -1, 1)).sum(dim=1)

        # Apply sigmoid to get confidence
        confidence = torch.sigmoid(weighted_logits)

        return confidence

    def get_layer_weights(self) -> torch.Tensor:
        """Get learned ensemble weights (after softmax).

        Returns:
            Normalized weights of shape (num_layers,)
        """
        with torch.no_grad():
            return torch.softmax(self.ensemble_weights, dim=0).cpu()


def build_layer_ensemble_network(
    input_dim: int,
    num_layers: int = 4,
    hidden_dim_per_layer: int = None,
    layer_probe_hidden: int = 64,
    dropout: float = 0.1,
) -> nn.Module:
    """Build layer-ensemble probe network.

    Creates an ensemble of lightweight probes, one per layer, with learned
    combination weights. Useful when hidden states are extracted from multiple
    transformer layers.

    Args:
        input_dim: Total input dimension (num_layers * hidden_dim_per_layer)
        num_layers: Number of layers to ensemble (default: 4)
        hidden_dim_per_layer: Hidden dimension from each layer (auto-detected if None)
        layer_probe_hidden: Hidden dim for per-layer probes (None = linear, else MLP)
        dropout: Dropout probability

    Returns:
        LayerEnsembleNetwork instance

    Example:
        >>> # Extract from 4 layers: [7, 14, 21, 27]
        >>> # Each layer has 3584 dims, so input_dim = 4 * 3584 = 14336
        >>> network = build_layer_ensemble_network(
        ...     input_dim=14336,
        ...     num_layers=4,
        ...     layer_probe_hidden=64
        ... )
        >>> probe = CalibratedProbe(network=network)
        >>> probe.fit(X_train, y_train, X_val, y_val)
        >>>
        >>> # Check learned weights
        >>> weights = probe.network.get_layer_weights()
        >>> print(f"Layer 7: {weights[0]:.3f}")
        >>> print(f"Layer 14: {weights[1]:.3f}")
        >>> print(f"Layer 21: {weights[2]:.3f}")
        >>> print(f"Layer 27: {weights[3]:.3f}")
    """
    return LayerEnsembleNetwork(
        input_dim, num_layers, hidden_dim_per_layer, layer_probe_hidden, dropout
    )


# =============================================================================
# SPARSE-ATTENTION-MULTIHEAD HYBRID
# =============================================================================
# Combines three key ideas:
# 1. Sparse: Not all dimensions are equally informative - learn importance weights
# 2. Attention: Relationships between different regions of hidden state matter
# 3. MultiHead: Different experts capture different aspects of uncertainty


class SparseAttentionMultiHeadNetwork(nn.Module):
    """Hybrid probe combining sparse weighting, attention, and multi-head prediction.

    Architecture flow:
    1. Sparse weighting: Learn which dimensions are important (differentiable)
    2. Chunking: Split weighted input into chunks (like tokens)
    3. Self-attention: Capture relationships between chunks
    4. Multi-head prediction: Multiple expert heads make predictions
    5. Gated aggregation: Learn to combine expert predictions based on input

    This addresses multiple hypotheses about how uncertainty is encoded:
    - Not all dimensions matter equally (sparse)
    - Relationships between hidden state regions encode information (attention)
    - Different aspects of uncertainty need different detectors (multi-head)
    """

    def __init__(
        self,
        input_dim: int,
        num_chunks: int = 16,
        num_attention_heads: int = 4,
        num_expert_heads: int = 4,
        expert_hidden_dim: int = 64,
        temperature: float = 1.0,
        dropout: float = 0.1,
    ):
        super().__init__()
        assert input_dim % num_chunks == 0, "input_dim must be divisible by num_chunks"

        self.input_dim = input_dim
        self.num_chunks = num_chunks
        self.chunk_dim = input_dim // num_chunks
        self.num_expert_heads = num_expert_heads
        self.temperature = temperature

        # === SPARSE COMPONENT ===
        # Learnable importance weights for each dimension
        self.importance = nn.Parameter(torch.zeros(input_dim))

        # === ATTENTION COMPONENT ===
        # Project chunks to attention dimension
        self.chunk_proj = nn.Linear(self.chunk_dim, self.chunk_dim)

        # Multi-head self-attention over chunks
        self.attention = nn.MultiheadAttention(
            embed_dim=self.chunk_dim,
            num_heads=num_attention_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.attention_norm = nn.LayerNorm(self.chunk_dim)

        # === MULTI-HEAD EXPERT COMPONENT ===
        # Each expert head processes the attended features differently
        self.expert_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.chunk_dim, expert_hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(expert_hidden_dim, 1),
            )
            for _ in range(num_expert_heads)
        ])

        # === GATED AGGREGATION ===
        # Input-dependent gating to combine expert predictions
        self.gate = nn.Sequential(
            nn.Linear(self.chunk_dim, num_expert_heads),
            nn.Softmax(dim=-1),
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]

        # === SPARSE WEIGHTING ===
        # Apply learned dimension importance (differentiable soft selection)
        weights = torch.sigmoid(self.importance / self.temperature)
        x_weighted = x * weights.unsqueeze(0)

        # === CHUNKING + ATTENTION ===
        # Reshape to chunks: (batch, num_chunks, chunk_dim)
        chunks = x_weighted.view(batch_size, self.num_chunks, self.chunk_dim)

        # Project chunks
        chunks_proj = self.chunk_proj(chunks)
        chunks_proj = self.dropout(chunks_proj)

        # Self-attention over chunks
        attn_out, _ = self.attention(chunks_proj, chunks_proj, chunks_proj)
        attn_out = self.attention_norm(attn_out + chunks_proj)  # Residual connection

        # Global representation: mean pool attended chunks
        global_repr = attn_out.mean(dim=1)  # (batch, chunk_dim)

        # === MULTI-HEAD EXPERT PREDICTION ===
        # Each expert makes a prediction
        expert_logits = []
        for expert in self.expert_heads:
            logit = expert(global_repr)  # (batch, 1)
            expert_logits.append(logit)

        # Stack expert predictions: (batch, num_experts)
        expert_logits = torch.cat(expert_logits, dim=-1)

        # === GATED AGGREGATION ===
        # Compute input-dependent weights for experts
        gate_weights = self.gate(global_repr)  # (batch, num_experts)

        # Weighted combination of expert predictions
        aggregated_logit = (expert_logits * gate_weights).sum(dim=-1, keepdim=True)

        # Final confidence
        confidence = torch.sigmoid(aggregated_logit)

        return confidence

    def get_importance_scores(self) -> torch.Tensor:
        """Get learned dimension importance scores."""
        return torch.sigmoid(self.importance / self.temperature).detach()

    def get_top_k_dimensions(self, k: int = 50) -> torch.Tensor:
        """Get indices of top-k most important dimensions."""
        _, indices = torch.topk(self.importance, k)
        return indices


def build_sparse_attention_multihead_network(
    input_dim: int,
    num_chunks: int = 16,
    num_attention_heads: int = 4,
    num_expert_heads: int = 4,
    expert_hidden_dim: int = 64,
    temperature: float = 1.0,
    dropout: float = 0.1,
) -> nn.Module:
    """Build hybrid sparse-attention-multihead probe network.

    Combines three complementary inductive biases:
    1. Sparse: Learns which hidden dimensions are most informative
    2. Attention: Captures relationships between different regions
    3. MultiHead: Multiple experts specialize in different uncertainty aspects

    Args:
        input_dim: Hidden state dimension (must be divisible by num_chunks).
        num_chunks: Number of chunks to split hidden state into (default: 16).
        num_attention_heads: Number of attention heads (default: 4).
        num_expert_heads: Number of prediction expert heads (default: 4).
        expert_hidden_dim: Hidden dimension within each expert (default: 64).
        temperature: Temperature for sparse sigmoid (lower = sharper, default: 1.0).
        dropout: Dropout probability (default: 0.1).

    Returns:
        Network that maps (batch, input_dim) -> (batch, 1).

    Example:
        >>> from src.probes import CalibratedProbe
        >>> from src.probes.architectures import build_sparse_attention_multihead_network
        >>>
        >>> network = build_sparse_attention_multihead_network(
        ...     input_dim=4096,
        ...     num_chunks=16,
        ...     num_expert_heads=4
        ... )
        >>> probe = CalibratedProbe(network=network)
        >>> probe.fit(X_train, y_train, X_val, y_val)
        >>>
        >>> # Analyze learned importance
        >>> importance = probe.network.get_importance_scores()
        >>> top_dims = probe.network.get_top_k_dimensions(k=20)
    """
    return SparseAttentionMultiHeadNetwork(
        input_dim=input_dim,
        num_chunks=num_chunks,
        num_attention_heads=num_attention_heads,
        num_expert_heads=num_expert_heads,
        expert_hidden_dim=expert_hidden_dim,
        temperature=temperature,
        dropout=dropout,
    )


# =============================================================================
# MULTI-SOURCE CONFIDENCE PREDICTOR
# =============================================================================
# Combines hidden states from multiple layers WITH output logits
# to predict the model's true confidence level.


class MultiSourceConfidenceNetwork(nn.Module):
    """Multi-source confidence predictor combining hidden states and logits.

    This architecture addresses a key insight: the model's internal uncertainty
    (encoded in hidden states) may differ from its expressed confidence (logits).
    By combining both sources, the probe can detect miscalibration.

    Architecture:
    1. Process each layer's hidden state with a lightweight probe
    2. Extract features from output logits (prob, entropy, margin)
    3. Combine layer predictions with logit features
    4. Predict final confidence score

    Inputs:
        - hidden_states: (batch, num_layers, hidden_dim) from k quartile layers
        - logits: (batch, num_choices) raw logits for answer choices (optional)

    The probe learns to detect:
        - Overconfidence: high logits but uncertain hidden states
        - Underconfidence: low logits but confident hidden states
        - Well-calibrated: agreement between sources
    """

    def __init__(
        self,
        hidden_dim: int,
        num_layers: int = 4,
        num_choices: int = 4,
        layer_probe_dim: int = 64,
        fusion_dim: int = 128,
        use_logits: bool = True,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_choices = num_choices
        self.use_logits = use_logits

        # === PER-LAYER PROBES ===
        # Each layer gets its own lightweight probe
        self.layer_probes = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, layer_probe_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(layer_probe_dim, layer_probe_dim // 2),
            )
            for _ in range(num_layers)
        ])

        # === LOGIT FEATURE EXTRACTOR ===
        # Extract meaningful features from output logits
        if use_logits:
            # Input: raw logits (num_choices) -> features
            # Features: softmax probs, entropy, top-2 margin, max prob
            logit_feature_dim = num_choices + 3  # probs + entropy + margin + max_prob
            self.logit_processor = nn.Sequential(
                nn.Linear(logit_feature_dim, layer_probe_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(layer_probe_dim, layer_probe_dim // 2),
            )
        else:
            logit_feature_dim = 0

        # === CROSS-LAYER ATTENTION ===
        # Let layers attend to each other
        self.layer_attention = nn.MultiheadAttention(
            embed_dim=layer_probe_dim // 2,
            num_heads=2,
            dropout=dropout,
            batch_first=True,
        )

        # === FUSION NETWORK ===
        # Combine layer features with logit features
        layer_output_dim = layer_probe_dim // 2
        fusion_input_dim = layer_output_dim + (layer_probe_dim // 2 if use_logits else 0)

        self.fusion = nn.Sequential(
            nn.Linear(fusion_input_dim, fusion_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, fusion_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim // 2, 1),
            nn.Sigmoid(),
        )

        # === LEARNABLE LAYER WEIGHTS ===
        self.layer_weights = nn.Parameter(torch.ones(num_layers) / num_layers)

    def _extract_logit_features(self, logits: torch.Tensor) -> torch.Tensor:
        """Extract meaningful features from raw logits.

        Features:
        - Softmax probabilities (num_choices values)
        - Entropy of distribution (1 value) - higher = more uncertain
        - Margin between top-2 (1 value) - higher = more confident
        - Max probability (1 value) - the "expressed" confidence
        """
        probs = torch.softmax(logits, dim=-1)

        # Entropy: -sum(p * log(p))
        entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1, keepdim=True)

        # Normalize entropy by max possible (log(num_choices))
        max_entropy = torch.log(torch.tensor(self.num_choices, dtype=torch.float32, device=logits.device))
        entropy = entropy / max_entropy

        # Top-2 margin
        top2 = torch.topk(probs, k=2, dim=-1).values
        margin = (top2[:, 0] - top2[:, 1]).unsqueeze(-1)

        # Max probability
        max_prob = probs.max(dim=-1, keepdim=True).values

        # Concatenate all features
        features = torch.cat([probs, entropy, margin, max_prob], dim=-1)
        return features

    def forward(
        self,
        x: torch.Tensor,
        logits: torch.Tensor = None,
    ) -> torch.Tensor:
        """Forward pass.

        For use with CalibratedProbe, input should be a single concatenated tensor:
            x: (batch, num_layers * hidden_dim + num_choices) if use_logits=True
            x: (batch, num_layers * hidden_dim) if use_logits=False

        For direct use, can also pass hidden_states and logits separately:
            x: (batch, num_layers, hidden_dim) hidden states
            logits: (batch, num_choices) raw output logits

        Returns:
            Confidence scores of shape (batch, 1)
        """
        batch_size = x.shape[0]

        # Handle different input formats
        if logits is not None:
            # Separate hidden_states and logits provided
            hidden_states = x
            if hidden_states.dim() == 2:
                hidden_states = hidden_states.view(batch_size, self.num_layers, self.hidden_dim)
        elif x.dim() == 2 and self.use_logits:
            # Single concatenated tensor: split into hidden_states and logits
            expected_hidden_size = self.num_layers * self.hidden_dim
            hidden_states = x[:, :expected_hidden_size].view(batch_size, self.num_layers, self.hidden_dim)
            logits = x[:, expected_hidden_size:]
        elif x.dim() == 2:
            # Flattened hidden states only
            hidden_states = x.view(batch_size, self.num_layers, self.hidden_dim)
        else:
            # Already shaped as (batch, num_layers, hidden_dim)
            hidden_states = x

        # === PROCESS EACH LAYER ===
        layer_features = []
        for i in range(self.num_layers):
            layer_input = hidden_states[:, i, :]
            layer_feat = self.layer_probes[i](layer_input)
            layer_features.append(layer_feat)

        # Stack: (batch, num_layers, layer_probe_dim//2)
        layer_features = torch.stack(layer_features, dim=1)

        # === CROSS-LAYER ATTENTION ===
        attended_features, _ = self.layer_attention(
            layer_features, layer_features, layer_features
        )

        # Weighted combination of layers
        weights = torch.softmax(self.layer_weights, dim=0)
        hidden_summary = (attended_features * weights.view(1, -1, 1)).sum(dim=1)

        # === PROCESS LOGITS ===
        if self.use_logits and logits is not None:
            logit_features = self._extract_logit_features(logits)
            logit_processed = self.logit_processor(logit_features)

            # Concatenate hidden summary with logit features
            fusion_input = torch.cat([hidden_summary, logit_processed], dim=-1)
        else:
            fusion_input = hidden_summary

        # === FINAL PREDICTION ===
        confidence = self.fusion(fusion_input)

        return confidence

    def get_layer_weights(self) -> torch.Tensor:
        """Get learned layer importance weights."""
        with torch.no_grad():
            return torch.softmax(self.layer_weights, dim=0).cpu()


def build_multi_source_network(
    hidden_dim: int,
    num_layers: int = 4,
    num_choices: int = 4,
    layer_probe_dim: int = 64,
    fusion_dim: int = 128,
    use_logits: bool = True,
    dropout: float = 0.1,
) -> nn.Module:
    """Build multi-source confidence prediction network.

    Combines hidden states from multiple layers with output logits to predict
    the model's true confidence level. This can detect miscalibration by
    comparing internal uncertainty signals with expressed confidence.

    Args:
        hidden_dim: Dimension of hidden states from each layer.
        num_layers: Number of layers to use (e.g., 4 for quartiles).
        num_choices: Number of answer choices (e.g., 4 for MCQA).
        layer_probe_dim: Hidden dimension for per-layer probes.
        fusion_dim: Hidden dimension for fusion network.
        use_logits: Whether to include output logits as input.
        dropout: Dropout probability.

    Returns:
        MultiSourceConfidenceNetwork instance.

    Example:
        >>> # Extract from 4 quartile layers
        >>> layers = [7, 14, 21, 27]  # For 28-layer model
        >>> hidden_states = extractor.extract(prompts, layers=layers)
        >>> # hidden_states shape: (batch, 4, 3584)
        >>>
        >>> # Get logits during extraction
        >>> logits = model(inputs).logits[:, -1, answer_tokens]
        >>> # logits shape: (batch, 4)
        >>>
        >>> # Build and train probe
        >>> network = build_multi_source_network(
        ...     hidden_dim=3584,
        ...     num_layers=4,
        ...     num_choices=4,
        ...     use_logits=True
        ... )
        >>> probe = CalibratedProbe(network=network)
        >>>
        >>> # Forward pass needs both hidden states and logits
        >>> confidence = network(hidden_states, logits)
    """
    return MultiSourceConfidenceNetwork(
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_choices=num_choices,
        layer_probe_dim=layer_probe_dim,
        fusion_dim=fusion_dim,
        use_logits=use_logits,
        dropout=dropout,
    )


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
    "build_hierarchical_network",
    "build_layer_ensemble_network",
    "build_sparse_attention_multihead_network",
    # Network classes (for customization)
    "ChunkedSelfAttention",
    "ResidualBlock",
    "MultiHeadProbeNetwork",
    "GatedLinearUnit",
    "TopKSparseNetwork",
    "HeteroscedasticNetwork",
    "BilinearInteractionNetwork",
    "ContrastiveProbe",
    "HierarchicalNetwork",
    "LayerEnsembleNetwork",
    "SparseAttentionMultiHeadNetwork",
]
