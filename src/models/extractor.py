"""Hidden state extraction utilities with efficient caching."""

import hashlib
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import torch
from tqdm.auto import tqdm
from transformers import PreTrainedModel, PreTrainedTokenizer

logger = logging.getLogger(__name__)


class HiddenStateExtractor:
    """Extract and cache hidden states from LLMs efficiently.

    Features:
    - Layer-specific extraction (single layer, multiple layers, or all layers)
    - Token position selection (last token, CLS token, or averaged)
    - Batch processing with progress tracking
    - Memory-mapped caching for large datasets
    - Support for variable-length sequences

    Example:
        >>> extractor = HiddenStateExtractor(model, tokenizer)
        >>> hiddens = extractor.extract(
        ...     texts=["What is the capital of France?"],
        ...     layers=[16, 24, 31],
        ...     cache_dir="cache/llama-3.1-8b"
        ... )
        >>> print(hiddens.shape)  # (num_texts, num_layers, hidden_dim)
    """

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        device: Optional[str] = None,
    ):
        """Initialize hidden state extractor.

        Args:
            model: Pre-trained language model
            tokenizer: Corresponding tokenizer
            device: Device to run extraction on (default: model's device)
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device or next(model.parameters()).device

        # Get model config
        self.num_layers = model.config.num_hidden_layers
        self.hidden_dim = model.config.hidden_size

        logger.info(
            f"Initialized extractor: {self.num_layers} layers, "
            f"hidden_dim={self.hidden_dim}, device={self.device}"
        )

    def extract(
        self,
        texts: List[str],
        layers: Optional[List[int]] = None,
        batch_size: int = 16,
        max_length: int = 512,
        token_position: str = "last",
        cache_dir: Optional[Union[str, Path]] = None,
        use_cache: bool = True,
        show_progress: bool = True,
    ) -> np.ndarray:
        """Extract hidden states from texts.

        Args:
            texts: List of input texts
            layers: Layer indices to extract (default: all layers)
            batch_size: Batch size for processing
            max_length: Maximum sequence length
            token_position: Token to extract ("last", "cls", "mean")
            cache_dir: Directory to cache extracted hiddens
            use_cache: Whether to use cached hiddens if available
            show_progress: Whether to show progress bar

        Returns:
            Hidden states array of shape (num_texts, num_layers, hidden_dim)

        Raises:
            ValueError: If layer indices are invalid or token_position is unknown
        """
        # Validate layers
        if layers is None:
            layers = list(range(self.num_layers))
        else:
            for layer_idx in layers:
                if not 0 <= layer_idx < self.num_layers:
                    raise ValueError(
                        f"Invalid layer index {layer_idx}. "
                        f"Model has {self.num_layers} layers (0-{self.num_layers-1})"
                    )

        # Validate token position
        if token_position not in ["last", "cls", "mean"]:
            raise ValueError(
                f"Invalid token_position: {token_position}. "
                "Expected 'last', 'cls', or 'mean'"
            )

        # Check cache
        if cache_dir and use_cache:
            cache_path = self._get_cache_path(
                texts, layers, max_length, token_position, cache_dir
            )
            if cache_path.exists():
                logger.info(f"Loading cached hiddens from {cache_path}")
                return np.load(cache_path, mmap_mode='r')

        # Extract hiddens
        logger.info(
            f"Extracting hiddens from {len(texts)} texts, "
            f"layers={layers}, batch_size={batch_size}"
        )

        all_hiddens = []
        num_batches = (len(texts) + batch_size - 1) // batch_size

        iterator = range(0, len(texts), batch_size)
        if show_progress:
            iterator = tqdm(iterator, total=num_batches, desc="Extracting hiddens")

        with torch.no_grad():
            for i in iterator:
                batch_texts = texts[i : i + batch_size]
                batch_hiddens = self._extract_batch(
                    batch_texts, layers, max_length, token_position
                )
                all_hiddens.append(batch_hiddens)

        # Concatenate all batches
        hiddens = np.concatenate(all_hiddens, axis=0)

        # Cache if requested
        if cache_dir:
            cache_path = self._get_cache_path(
                texts, layers, max_length, token_position, cache_dir
            )
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(cache_path, hiddens)
            logger.info(f"Cached hiddens to {cache_path}")

        return hiddens

    def _extract_batch(
        self,
        texts: List[str],
        layers: List[int],
        max_length: int,
        token_position: str,
    ) -> np.ndarray:
        """Extract hidden states for a batch of texts.

        Args:
            texts: List of input texts
            layers: Layer indices to extract
            max_length: Maximum sequence length
            token_position: Token to extract ("last", "cls", "mean")

        Returns:
            Hidden states array of shape (batch_size, num_layers, hidden_dim)
        """
        # Tokenize
        encodings = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        encodings = {k: v.to(self.device) for k, v in encodings.items()}

        # Forward pass with hidden states output
        outputs = self.model(
            **encodings,
            output_hidden_states=True,
            return_dict=True,
        )

        # Extract hidden states from specified layers
        # outputs.hidden_states is a tuple of (num_layers + 1) tensors
        # Index 0 is embeddings, indices 1-N are layer outputs
        batch_hiddens = []

        for layer_idx in layers:
            # Get layer output (add 1 because index 0 is embeddings)
            layer_hiddens = outputs.hidden_states[layer_idx + 1]  # (batch, seq_len, hidden)

            # Select token position
            if token_position == "last":
                # Get last non-padding token
                attention_mask = encodings["attention_mask"]
                sequence_lengths = attention_mask.sum(dim=1) - 1  # Last token index
                token_hiddens = layer_hiddens[
                    torch.arange(layer_hiddens.size(0), device=self.device),
                    sequence_lengths,
                ]
            elif token_position == "cls":
                # Get first token (CLS)
                token_hiddens = layer_hiddens[:, 0, :]
            elif token_position == "mean":
                # Average over sequence length (excluding padding)
                attention_mask = encodings["attention_mask"].unsqueeze(-1)
                masked_hiddens = layer_hiddens * attention_mask
                token_hiddens = masked_hiddens.sum(dim=1) / attention_mask.sum(dim=1)

            batch_hiddens.append(token_hiddens.float().cpu().numpy())

        # Stack layers: (batch, num_layers, hidden_dim)
        return np.stack(batch_hiddens, axis=1)

    def _get_cache_path(
        self,
        texts: List[str],
        layers: List[int],
        max_length: int,
        token_position: str,
        cache_dir: Union[str, Path],
    ) -> Path:
        """Generate cache file path based on extraction parameters.

        Args:
            texts: Input texts
            layers: Layer indices
            max_length: Maximum sequence length
            token_position: Token position mode
            cache_dir: Cache directory

        Returns:
            Path to cache file
        """
        # Create hash of texts and parameters
        content = "".join(texts) + str(layers) + str(max_length) + token_position
        content_hash = hashlib.md5(content.encode()).hexdigest()[:16]

        layers_str = "_".join(map(str, layers))
        filename = f"hiddens_{len(texts)}texts_{layers_str}_{token_position}_{content_hash}.npy"

        cache_path = Path(cache_dir) / filename
        return cache_path

    def extract_with_labels(
        self,
        texts: List[str],
        labels: List[int],
        **kwargs,
    ) -> Dict[str, np.ndarray]:
        """Extract hidden states along with labels.

        Convenience method that extracts hiddens and returns them with labels.

        Args:
            texts: List of input texts
            labels: Corresponding labels
            **kwargs: Additional arguments passed to extract()

        Returns:
            Dictionary with "hiddens" and "labels" keys
        """
        if len(texts) != len(labels):
            raise ValueError(
                f"Number of texts ({len(texts)}) doesn't match "
                f"number of labels ({len(labels)})"
            )

        hiddens = self.extract(texts, **kwargs)
        return {
            "hiddens": hiddens,
            "labels": np.array(labels),
        }

    def get_layer_statistics(
        self,
        texts: List[str],
        layers: Optional[List[int]] = None,
        batch_size: int = 16,
    ) -> Dict[int, Dict[str, float]]:
        """Compute statistics of hidden states across layers.

        Useful for understanding representation properties and selecting layers.

        Args:
            texts: Sample texts to analyze
            layers: Layer indices (default: all layers)
            batch_size: Batch size for extraction

        Returns:
            Dictionary mapping layer index to statistics (mean, std, norm)
        """
        hiddens = self.extract(
            texts,
            layers=layers,
            batch_size=batch_size,
            show_progress=False,
        )

        stats = {}
        for i, layer_idx in enumerate(layers or range(self.num_layers)):
            layer_hiddens = hiddens[:, i, :]  # (num_texts, hidden_dim)
            stats[layer_idx] = {
                "mean": float(np.mean(layer_hiddens)),
                "std": float(np.std(layer_hiddens)),
                "norm": float(np.linalg.norm(layer_hiddens, axis=-1).mean()),
                "min": float(np.min(layer_hiddens)),
                "max": float(np.max(layer_hiddens)),
            }

        return stats
