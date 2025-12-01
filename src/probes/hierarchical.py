"""Hierarchical multi-scale probe for uncertainty quantification.

This module implements a novel hierarchical approach to uncertainty estimation
that operates at multiple granularities: token, span, semantic, and global levels.
Each level captures different aspects of model uncertainty.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Union

import torch
from torch import nn
import torch.nn.functional as F

from .base import BaseProbe


class AttentionPooling(nn.Module):
    """Attention-based pooling layer for aggregating token representations.

    Args:
        input_dim: Dimension of input features
        output_dim: Dimension of output features
    """

    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.attention = nn.Linear(input_dim, 1)
        self.projection = nn.Linear(input_dim, output_dim)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Apply attention pooling.

        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim)
            mask: Optional attention mask of shape (batch_size, seq_len)

        Returns:
            Pooled tensor of shape (batch_size, output_dim)
        """
        # Compute attention weights
        attn_weights = self.attention(x).squeeze(-1)  # (batch_size, seq_len)

        # Apply mask if provided
        if mask is not None:
            attn_weights = attn_weights.masked_fill(~mask.bool(), float("-inf"))

        # Softmax normalization
        attn_weights = F.softmax(attn_weights, dim=-1).unsqueeze(-1)  # (batch_size, seq_len, 1)

        # Weighted sum
        pooled = torch.sum(x * attn_weights, dim=1)  # (batch_size, input_dim)

        # Project to output dimension
        output = self.projection(pooled)  # (batch_size, output_dim)

        return output


class TokenLevelProbe(nn.Module):
    """Token-level uncertainty probe.

    Estimates confidence for individual tokens in the sequence.

    Args:
        input_dim: Dimension of token hidden states
        hidden_dim: Dimension of hidden layer
    """

    def __init__(self, input_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, token_hiddens: torch.Tensor) -> torch.Tensor:
        """Compute token-level confidence scores.

        Args:
            token_hiddens: Token representations (batch_size, seq_len, input_dim)

        Returns:
            Token confidences (batch_size, seq_len, 1)
        """
        return self.mlp(token_hiddens)


class SpanLevelProbe(nn.Module):
    """Span-level uncertainty probe.

    Aggregates token representations to estimate confidence for spans/phrases.

    Args:
        input_dim: Dimension of token hidden states
        hidden_dim: Dimension of hidden layer
        span_size: Size of spans to consider
    """

    def __init__(self, input_dim: int, hidden_dim: int = 256, span_size: int = 4):
        super().__init__()
        self.span_size = span_size
        self.attention_pool = AttentionPooling(input_dim, hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )

    def forward(
        self, token_hiddens: torch.Tensor, token_confidences: torch.Tensor
    ) -> torch.Tensor:
        """Compute span-level confidence scores.

        Args:
            token_hiddens: Token representations (batch_size, seq_len, input_dim)
            token_confidences: Token-level confidences (batch_size, seq_len, 1)

        Returns:
            Span-level confidence (batch_size, 1)
        """
        # Use attention pooling to aggregate tokens
        span_repr = self.attention_pool(token_hiddens)  # (batch_size, hidden_dim)

        # Compute span-level confidence
        span_confidence = self.mlp(span_repr)  # (batch_size, 1)

        return span_confidence


class SemanticLevelProbe(nn.Module):
    """Semantic-level uncertainty probe.

    Estimates confidence based on semantic coherence and meaning.

    Args:
        input_dim: Dimension of input features
        hidden_dim: Dimension of hidden layer
    """

    def __init__(self, input_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, semantic_repr: torch.Tensor) -> torch.Tensor:
        """Compute semantic-level confidence.

        Args:
            semantic_repr: Semantic representation (batch_size, input_dim)

        Returns:
            Semantic confidence (batch_size, 1)
        """
        return self.mlp(semantic_repr)


class HierarchicalProbe(BaseProbe):
    """Hierarchical multi-scale probe for uncertainty quantification.

    This probe operates at four levels of granularity:
    1. Token-level: Per-token confidence
    2. Span-level: Phrase/chunk confidence
    3. Semantic-level: Sentence/clause confidence
    4. Global-level: Overall answer confidence

    Each level builds upon the previous, creating a hierarchical representation
    of uncertainty. This allows the model to capture both fine-grained and
    coarse-grained uncertainty signals.

    Args:
        input_dim: Dimension of input hidden states
        hidden_dim: Dimension of hidden layers (default: 512)
        num_layers: Number of transformer layers to use for multi-layer input (default: 3)
        use_layer_fusion: Whether to fuse information from multiple layers (default: True)
        device: Device to run computations on. If None, uses CUDA if available.

    Raises:
        ValueError: If input_dim <= 0 or hidden_dim <= 0.

    Example:
        >>> probe = HierarchicalProbe(input_dim=4096, hidden_dim=512)
        >>> # For training, need to extract hidden states with sequence info
        >>> probe.fit(X_train, y_train, X_val, y_val)
        >>> confidences = probe.predict(X_test)
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 512,
        num_layers: int = 3,
        use_layer_fusion: bool = True,
        device: Optional[Union[str, torch.device]] = None,
    ) -> None:
        super().__init__()

        # Input validation
        if input_dim <= 0:
            raise ValueError(f"input_dim must be positive, got {input_dim}")
        if hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {hidden_dim}")

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device: torch.device = torch.device(device)
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.use_layer_fusion = use_layer_fusion

        # Layer-wise feature extraction (if using multiple layers)
        if use_layer_fusion:
            self.layer_projections = nn.ModuleList([
                nn.Linear(input_dim, hidden_dim) for _ in range(num_layers)
            ])
            self.layer_attention = nn.MultiheadAttention(
                embed_dim=hidden_dim, num_heads=8, batch_first=True
            )
            effective_dim = hidden_dim
        else:
            effective_dim = input_dim

        # Hierarchical probe components
        self.token_probe = TokenLevelProbe(effective_dim, hidden_dim=256)
        self.span_probe = SpanLevelProbe(effective_dim, hidden_dim=256)
        self.semantic_probe = SemanticLevelProbe(effective_dim, hidden_dim=256)

        # Global aggregation
        # Combines all levels: token mean/max + span + semantic
        global_input_dim = 256 + 1 + 1  # token features + span conf + semantic conf
        self.global_mlp = nn.Sequential(
            nn.Linear(global_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )

        # Attention pooling for token aggregation
        self.token_pooling = AttentionPooling(effective_dim, 256)

        self.to(self.device)

    @classmethod
    def from_config(cls, config: Dict[str, Any], input_dim: int) -> "HierarchicalProbe":
        """Create probe from configuration dictionary.

        Args:
            config: Configuration dictionary with probe hyperparameters.
            input_dim: Dimensionality of the hidden state features.

        Returns:
            Initialized HierarchicalProbe instance.

        Example:
            >>> config = {"hidden_dim": 512, "num_layers": 3, "use_layer_fusion": True}
            >>> probe = HierarchicalProbe.from_config(config, input_dim=4096)
        """
        return cls(
            input_dim=input_dim,
            hidden_dim=config.get("hidden_dim", 512),
            num_layers=config.get("num_layers", 3),
            use_layer_fusion=config.get("use_layer_fusion", True),
            device=config.get("device", None),
        )

    def _process_layers(self, hiddens: torch.Tensor) -> torch.Tensor:
        """Fuse information from multiple layers if using layer fusion.

        Args:
            hiddens: Hidden states (batch_size, num_layers * seq_len, input_dim)
                     or (batch_size, seq_len, input_dim) if single layer

        Returns:
            Fused representation (batch_size, seq_len, hidden_dim)
        """
        if not self.use_layer_fusion:
            return hiddens

        batch_size = hiddens.shape[0]
        seq_len = hiddens.shape[1] // self.num_layers

        # Reshape to (batch_size, num_layers, seq_len, input_dim)
        hiddens_reshaped = hiddens.view(batch_size, self.num_layers, seq_len, -1)

        # Project each layer
        projected_layers = []
        for i, proj in enumerate(self.layer_projections):
            layer_hidden = hiddens_reshaped[:, i, :, :]  # (batch_size, seq_len, input_dim)
            projected = proj(layer_hidden)  # (batch_size, seq_len, hidden_dim)
            projected_layers.append(projected)

        # Stack layers: (batch_size, seq_len, num_layers, hidden_dim)
        stacked = torch.stack(projected_layers, dim=2)

        # Reshape for attention: (batch_size * seq_len, num_layers, hidden_dim)
        stacked_flat = stacked.view(batch_size * seq_len, self.num_layers, self.hidden_dim)

        # Apply cross-layer attention
        fused, _ = self.layer_attention(stacked_flat, stacked_flat, stacked_flat)

        # Take mean across layers
        fused = fused.mean(dim=1)  # (batch_size * seq_len, hidden_dim)

        # Reshape back
        fused = fused.view(batch_size, seq_len, self.hidden_dim)

        return fused

    def forward(self, hiddens: torch.Tensor) -> torch.Tensor:
        """Forward pass through hierarchical probe.

        Args:
            hiddens: Hidden states. Shape depends on use_sequence_info:
                    - If False: (batch_size, input_dim) - single vector per example
                    - If True: (batch_size, seq_len, input_dim) - full sequence

        Returns:
            Global confidence scores (batch_size, 1)
        """
        # Handle both sequence and single-vector inputs
        if hiddens.dim() == 2:
            # Single vector per example: expand to sequence of length 1
            hiddens = hiddens.unsqueeze(1)  # (batch_size, 1, input_dim)

        # Process multiple layers if using layer fusion
        if self.use_layer_fusion and hiddens.shape[1] > 1:
            processed_hiddens = self._process_layers(hiddens)
        else:
            processed_hiddens = hiddens

        # Token-level processing
        token_confidences = self.token_probe(processed_hiddens)  # (batch_size, seq_len, 1)

        # Aggregate token features for higher levels
        token_aggregated = self.token_pooling(processed_hiddens)  # (batch_size, 256)

        # Span-level processing
        span_confidence = self.span_probe(
            processed_hiddens, token_confidences
        )  # (batch_size, 1)

        # Semantic-level processing (use aggregated tokens)
        semantic_repr = processed_hiddens.mean(dim=1)  # Simple mean pooling
        semantic_confidence = self.semantic_probe(semantic_repr)  # (batch_size, 1)

        # Global-level: combine all levels
        global_features = torch.cat(
            [token_aggregated, span_confidence, semantic_confidence], dim=-1
        )  # (batch_size, 258)

        global_confidence = self.global_mlp(global_features)  # (batch_size, 1)

        return global_confidence

    def forward_with_intermediates(
        self, hiddens: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Forward pass that returns intermediate confidence scores at each level.

        Useful for analysis and visualization.

        Args:
            hiddens: Hidden states (batch_size, seq_len, input_dim)

        Returns:
            Dictionary with confidence scores at each level:
            {
                "token": (batch_size, seq_len, 1),
                "span": (batch_size, 1),
                "semantic": (batch_size, 1),
                "global": (batch_size, 1),
            }
        """
        # Handle both sequence and single-vector inputs
        if hiddens.dim() == 2:
            hiddens = hiddens.unsqueeze(1)

        # Process multiple layers if using layer fusion
        if self.use_layer_fusion and hiddens.shape[1] > 1:
            processed_hiddens = self._process_layers(hiddens)
        else:
            processed_hiddens = hiddens

        # Token-level
        token_confidences = self.token_probe(processed_hiddens)

        # Aggregate tokens
        token_aggregated = self.token_pooling(processed_hiddens)

        # Span-level
        span_confidence = self.span_probe(processed_hiddens, token_confidences)

        # Semantic-level
        semantic_repr = processed_hiddens.mean(dim=1)
        semantic_confidence = self.semantic_probe(semantic_repr)

        # Global-level
        global_features = torch.cat(
            [token_aggregated, span_confidence, semantic_confidence], dim=-1
        )
        global_confidence = self.global_mlp(global_features)

        return {
            "token": token_confidences,  # (batch_size, seq_len, 1)
            "span": span_confidence,  # (batch_size, 1)
            "semantic": semantic_confidence,  # (batch_size, 1)
            "global": global_confidence,  # (batch_size, 1)
        }

    def get_model_info(self) -> Dict[str, Any]:
        """Get probe architecture information.

        Returns:
            Dictionary containing:
                - input_dim: Input feature dimension
                - hidden_dim: Hidden layer dimension
                - num_layers: Number of layers for fusion
                - use_layer_fusion: Whether layer fusion is enabled
                - device: Device the probe is on
                - num_parameters: Total number of trainable parameters
        """
        return {
            "input_dim": self.input_dim,
            "hidden_dim": self.hidden_dim,
            "num_layers": self.num_layers,
            "use_layer_fusion": self.use_layer_fusion,
            "device": str(self.device),
            "num_parameters": sum(p.numel() for p in self.parameters()),
        }


__all__ = ["HierarchicalProbe"]
