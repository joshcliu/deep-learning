"""Model registry with configurations for supported LLMs."""

from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class ModelConfig:
    """Configuration for a supported model.

    Attributes:
        name: HuggingFace model identifier
        num_layers: Total number of transformer layers
        hidden_dim: Dimension of hidden states
        optimal_layers: Recommended layers for probing (quartile positions)
        architecture: Model architecture family (llama, mistral, qwen, etc.)
        default_quantization: Recommended quantization for memory efficiency
        min_vram_gb: Minimum VRAM required without quantization
    """
    name: str
    num_layers: int
    hidden_dim: int
    optimal_layers: List[int]
    architecture: str
    default_quantization: Optional[str] = "8bit"
    min_vram_gb: int = 16


# Registry of supported models
MODEL_REGISTRY: Dict[str, ModelConfig] = {
    # Llama 3.1 family
    "meta-llama/Llama-3.1-8B": ModelConfig(
        name="meta-llama/Llama-3.1-8B",
        num_layers=32,
        hidden_dim=4096,
        optimal_layers=[8, 16, 24, 31],  # Quartiles
        architecture="llama",
        default_quantization="8bit",
        min_vram_gb=16,
    ),
    "meta-llama/Llama-3.1-70B": ModelConfig(
        name="meta-llama/Llama-3.1-70B",
        num_layers=80,
        hidden_dim=8192,
        optimal_layers=[20, 40, 60, 79],
        architecture="llama",
        default_quantization="4bit",
        min_vram_gb=140,
    ),

    # Llama 2 family (for compatibility)
    "meta-llama/Llama-2-7b-hf": ModelConfig(
        name="meta-llama/Llama-2-7b-hf",
        num_layers=32,
        hidden_dim=4096,
        optimal_layers=[8, 16, 24, 31],
        architecture="llama",
        default_quantization="8bit",
        min_vram_gb=14,
    ),
    "meta-llama/Llama-2-13b-hf": ModelConfig(
        name="meta-llama/Llama-2-13b-hf",
        num_layers=40,
        hidden_dim=5120,
        optimal_layers=[10, 20, 30, 39],
        architecture="llama",
        default_quantization="8bit",
        min_vram_gb=26,
    ),

    # Mistral family
    "mistralai/Mistral-7B-v0.1": ModelConfig(
        name="mistralai/Mistral-7B-v0.1",
        num_layers=32,
        hidden_dim=4096,
        optimal_layers=[8, 16, 24, 31],
        architecture="mistral",
        default_quantization="8bit",
        min_vram_gb=14,
    ),
    "mistralai/Mixtral-8x7B-v0.1": ModelConfig(
        name="mistralai/Mixtral-8x7B-v0.1",
        num_layers=32,
        hidden_dim=4096,
        optimal_layers=[8, 16, 24, 31],
        architecture="mixtral",
        default_quantization="4bit",
        min_vram_gb=90,
    ),

    # Qwen family
    "Qwen/Qwen2.5-7B": ModelConfig(
        name="Qwen/Qwen2.5-7B",
        num_layers=28,
        hidden_dim=3584,
        optimal_layers=[7, 14, 21, 27],
        architecture="qwen",
        default_quantization="8bit",
        min_vram_gb=14,
    ),
    "Qwen/Qwen2.5-14B": ModelConfig(
        name="Qwen/Qwen2.5-14B",
        num_layers=48,
        hidden_dim=5120,
        optimal_layers=[12, 24, 36, 47],
        architecture="qwen",
        default_quantization="8bit",
        min_vram_gb=28,
    ),
    "Qwen/Qwen2.5-72B": ModelConfig(
        name="Qwen/Qwen2.5-72B",
        num_layers=80,
        hidden_dim=8192,
        optimal_layers=[20, 40, 60, 79],
        architecture="qwen",
        default_quantization="4bit",
        min_vram_gb=144,
    ),
}


def get_model_config(model_name: str) -> ModelConfig:
    """Get configuration for a model.

    Args:
        model_name: HuggingFace model identifier or alias

    Returns:
        ModelConfig for the requested model

    Raises:
        ValueError: If model is not in registry
    """
    if model_name not in MODEL_REGISTRY:
        available = list(MODEL_REGISTRY.keys())
        raise ValueError(
            f"Model '{model_name}' not found in registry. "
            f"Available models: {available}"
        )
    return MODEL_REGISTRY[model_name]


def list_supported_models() -> List[str]:
    """List all supported model identifiers.

    Returns:
        List of HuggingFace model identifiers
    """
    return list(MODEL_REGISTRY.keys())


def get_quartile_layers(num_layers: int) -> List[int]:
    """Calculate quartile layer positions for a model.

    Args:
        num_layers: Total number of layers in the model

    Returns:
        List of 4 layer indices at quartile positions (25%, 50%, 75%, 100%)
    """
    return [
        num_layers // 4,              # 25%
        num_layers // 2,              # 50%
        (3 * num_layers) // 4,        # 75%
        num_layers - 1,               # 100% (last layer)
    ]
