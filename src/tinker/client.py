"""
Tinker API Client Wrapper.

Provides a simplified interface to the Tinker API with automatic authentication
and error handling.
"""

import os
from typing import List, Optional

try:
    import tinker
except ImportError:
    tinker = None


class TinkerClient:
    """Wrapper for Tinker API ServiceClient with authentication.

    Handles API key management and provides convenient access to Tinker
    functionality for training, sampling, and weight management.

    Args:
        api_key: Tinker API key. If None, reads from TINKER_API_KEY environment variable.

    Raises:
        ImportError: If tinker package is not installed.
        ValueError: If API key is not provided or found in environment.
        RuntimeError: If Tinker API connection fails.

    Example:
        >>> client = TinkerClient()  # Uses TINKER_API_KEY env var
        >>> models = client.get_supported_models()
        >>> print(models)
        ['meta-llama/Llama-3.1-8B', 'Qwen/Qwen2.5-7B', ...]
    """

    def __init__(self, api_key: Optional[str] = None):
        if tinker is None:
            raise ImportError(
                "tinker package not installed. Install with: pip install tinker"
            )

        # Get API key
        self.api_key = api_key or os.getenv("TINKER_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Tinker API key not provided. Either pass api_key parameter or "
                "set TINKER_API_KEY environment variable."
            )

        # Set environment variable for tinker SDK
        os.environ["TINKER_API_KEY"] = self.api_key

        try:
            self._service_client = tinker.ServiceClient()
            self._rest_client = self._service_client.create_rest_client()
        except Exception as e:
            raise RuntimeError(f"Failed to connect to Tinker API: {e}") from e

    @property
    def service_client(self):
        """Get underlying Tinker ServiceClient."""
        return self._service_client

    @property
    def rest_client(self):
        """Get underlying Tinker RestClient."""
        return self._rest_client

    def get_supported_models(self) -> List[str]:
        """Get list of supported base models.

        Returns:
            List of model names/paths supported by Tinker API.

        Example:
            >>> client = TinkerClient()
            >>> models = client.get_supported_models()
            >>> print("meta-llama/Llama-3.1-8B" in models)
            True
        """
        try:
            capabilities = self._service_client.get_server_capabilities()
            return capabilities.supported_models
        except Exception as e:
            raise RuntimeError(
                f"Failed to get supported models from Tinker API: {e}"
            ) from e

    def create_training_client(self, model_name: str):
        """Create a TrainingClient for fine-tuning.

        Args:
            model_name: Name of base model to fine-tune
                (e.g., "meta-llama/Llama-3.1-8B")

        Returns:
            Tinker TrainingClient instance

        Raises:
            ValueError: If model_name is not supported
            RuntimeError: If client creation fails

        Example:
            >>> client = TinkerClient()
            >>> training_client = client.create_training_client(
            ...     "meta-llama/Llama-3.1-8B"
            ... )
        """
        supported = self.get_supported_models()
        if model_name not in supported:
            raise ValueError(
                f"Model '{model_name}' not supported by Tinker API. "
                f"Supported models: {supported}"
            )

        try:
            return self._service_client.create_training_client(model_name)
        except Exception as e:
            raise RuntimeError(
                f"Failed to create training client for {model_name}: {e}"
            ) from e

    def get_checkpoint_archive_url(self, tinker_path: str) -> str:
        """Get download URL for a checkpoint archive.

        Args:
            tinker_path: Tinker checkpoint path
                (e.g., "tinker://<id>/sampler_weights/final")

        Returns:
            Signed URL for downloading checkpoint archive (time-limited)

        Raises:
            RuntimeError: If URL generation fails

        Example:
            >>> client = TinkerClient()
            >>> url = client.get_checkpoint_archive_url(
            ...     "tinker://abc123/sampler_weights/final"
            ... )
            >>> print(url)
            https://...signed_url...
        """
        try:
            future = self._rest_client.get_checkpoint_archive_url_from_tinker_path(
                tinker_path
            )
            response = future.result()
            return response.url
        except Exception as e:
            raise RuntimeError(
                f"Failed to get checkpoint URL for {tinker_path}: {e}"
            ) from e

    def __repr__(self) -> str:
        return f"TinkerClient(api_key={'***' if self.api_key else None})"


__all__ = ["TinkerClient"]
