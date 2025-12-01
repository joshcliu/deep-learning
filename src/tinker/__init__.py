"""
Tinker API Integration for LLM Confidence Probing.

This module provides integration with the Tinker API for efficient LoRA
fine-tuning and model training. Trained models can be downloaded and loaded
into HuggingFace/PEFT for hidden state extraction and probing.

Workflow:
    1. Fine-tune models using Tinker API (distributed, efficient)
    2. Download LoRA weights from Tinker
    3. Load weights into HuggingFace via PEFT
    4. Extract hidden states using existing infrastructure

Available modules:
- client: Tinker API client wrapper
- weights: Download and convert Tinker weights to PEFT format
- training: LoRA training interface for datasets
- sampling: Inference and logprob extraction

Usage:
    from src.tinker import TinkerClient, download_and_convert_weights

    # Setup client
    client = TinkerClient()  # Uses TINKER_API_KEY env var

    # Download weights and convert to PEFT format
    local_path = download_and_convert_weights(
        tinker_checkpoint="tinker://model_id/checkpoint",
        output_dir="weights/my_model"
    )

    # Load into HuggingFace for probing
    from src.models import ModelLoader
    loader = ModelLoader("meta-llama/Llama-3.1-8B", tinker_lora_path=local_path)
    model, tokenizer = loader.load()

API Key Setup:
    export TINKER_API_KEY=<your_key>
"""

from .client import TinkerClient
from .weights import download_and_convert_weights, download_checkpoint

__all__ = [
    "TinkerClient",
    "download_and_convert_weights",
    "download_checkpoint",
]

__version__ = "0.1.0"
