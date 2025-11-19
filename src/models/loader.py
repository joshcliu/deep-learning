"""Model loading utilities with quantization support."""

import logging
from typing import Optional, Tuple, Union

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    PreTrainedModel,
    PreTrainedTokenizer,
)

from .registry import ModelConfig, get_model_config

logger = logging.getLogger(__name__)


class ModelLoader:
    """Unified interface for loading LLMs with optional quantization.

    Supports loading models with:
    - No quantization (full precision)
    - 8-bit quantization (bitsandbytes)
    - 4-bit quantization (bitsandbytes)

    Example:
        >>> loader = ModelLoader("meta-llama/Llama-3.1-8B")
        >>> model, tokenizer = loader.load(quantization="8bit")
        >>> print(f"Model loaded with {model.num_parameters():,} parameters")
    """

    def __init__(
        self,
        model_name: str,
        device: Optional[str] = None,
        use_auth_token: Optional[str] = None,
    ):
        """Initialize model loader.

        Args:
            model_name: HuggingFace model identifier
            device: Device to load model on ("cuda", "cpu", "auto"). If None, auto-detects.
            use_auth_token: HuggingFace authentication token for gated models
        """
        self.model_name = model_name
        self.config = get_model_config(model_name)
        self.use_auth_token = use_auth_token

        # Auto-detect device if not specified
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        logger.info(f"Initialized loader for {model_name} on device: {self.device}")

    def load(
        self,
        quantization: Optional[str] = None,
        torch_dtype: Optional[torch.dtype] = None,
        device_map: Union[str, dict] = "auto",
        trust_remote_code: bool = False,
    ) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
        """Load model and tokenizer with optional quantization.

        Args:
            quantization: Quantization mode ("8bit", "4bit", or None for full precision)
            torch_dtype: Data type for model weights (default: auto-detect based on quantization)
            device_map: Device mapping strategy ("auto", "balanced", or custom dict)
            trust_remote_code: Whether to trust remote code (required for some models)

        Returns:
            Tuple of (model, tokenizer)

        Raises:
            ValueError: If quantization mode is invalid
            RuntimeError: If model loading fails
        """
        # Use default quantization from config if not specified
        if quantization is None:
            quantization = self.config.default_quantization

        # Prepare quantization config
        quantization_config = None
        if quantization == "8bit":
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False,
            )
            logger.info("Loading model with 8-bit quantization")
        elif quantization == "4bit":
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
            logger.info("Loading model with 4-bit quantization")
        elif quantization is not None:
            raise ValueError(
                f"Invalid quantization mode: {quantization}. "
                "Expected '8bit', '4bit', or None."
            )

        # Auto-detect dtype if not specified
        if torch_dtype is None:
            if quantization:
                torch_dtype = torch.bfloat16  # Use bfloat16 for quantized models
            else:
                torch_dtype = torch.float16 if self.device == "cuda" else torch.float32

        try:
            # Load tokenizer
            logger.info(f"Loading tokenizer for {self.model_name}")
            tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                use_auth_token=self.use_auth_token,
                trust_remote_code=trust_remote_code,
            )

            # Ensure tokenizer has pad token
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
                logger.info("Set pad_token to eos_token")

            # Load model
            logger.info(f"Loading model {self.model_name} with dtype {torch_dtype}")
            model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                quantization_config=quantization_config,
                torch_dtype=torch_dtype,
                device_map=device_map,
                use_auth_token=self.use_auth_token,
                trust_remote_code=trust_remote_code,
            )

            # Set model to eval mode
            model.eval()

            # Log memory usage if on CUDA
            if torch.cuda.is_available():
                memory_allocated = torch.cuda.memory_allocated() / 1e9  # GB
                logger.info(f"GPU memory allocated: {memory_allocated:.2f} GB")

            logger.info("Model and tokenizer loaded successfully")
            return model, tokenizer

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise RuntimeError(f"Model loading failed: {e}") from e

    def get_model_info(self) -> dict:
        """Get information about the model configuration.

        Returns:
            Dictionary with model metadata
        """
        return {
            "name": self.config.name,
            "architecture": self.config.architecture,
            "num_layers": self.config.num_layers,
            "hidden_dim": self.config.hidden_dim,
            "optimal_layers": self.config.optimal_layers,
            "default_quantization": self.config.default_quantization,
            "min_vram_gb": self.config.min_vram_gb,
            "device": self.device,
        }
