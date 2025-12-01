"""
Tinker Weight Download and Conversion.

Functions for downloading LoRA weights from Tinker API and converting them
to HuggingFace PEFT format for local inference and hidden state extraction.
"""

import json
import os
import tarfile
import urllib.request
from pathlib import Path
from typing import Optional, Dict, Any

from .client import TinkerClient


def download_checkpoint(
    tinker_path: str,
    output_dir: str,
    client: Optional[TinkerClient] = None,
) -> str:
    """Download checkpoint archive from Tinker API.

    Args:
        tinker_path: Tinker checkpoint path
            (e.g., "tinker://<id>/sampler_weights/final")
        output_dir: Directory to save downloaded archive
        client: TinkerClient instance. If None, creates a new one.

    Returns:
        Path to downloaded tar archive

    Raises:
        RuntimeError: If download fails

    Example:
        >>> archive_path = download_checkpoint(
        ...     "tinker://abc123/sampler_weights/final",
        ...     "weights/my_model"
        ... )
        >>> print(archive_path)
        weights/my_model/archive.tar
    """
    if client is None:
        client = TinkerClient()

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    archive_file = output_path / "archive.tar"

    try:
        # Get signed download URL
        url = client.get_checkpoint_archive_url(tinker_path)

        # Download archive
        print(f"Downloading checkpoint from {tinker_path}...")
        urllib.request.urlretrieve(url, archive_file)
        print(f"Downloaded to {archive_file}")

        return str(archive_file)

    except Exception as e:
        raise RuntimeError(
            f"Failed to download checkpoint from {tinker_path}: {e}"
        ) from e


def extract_archive(archive_path: str, extract_to: Optional[str] = None) -> str:
    """Extract Tinker checkpoint archive.

    Args:
        archive_path: Path to downloaded tar archive
        extract_to: Directory to extract to. If None, extracts to same
            directory as archive.

    Returns:
        Path to extraction directory

    Raises:
        RuntimeError: If extraction fails

    Example:
        >>> extract_dir = extract_archive("weights/my_model/archive.tar")
        >>> print(extract_dir)
        weights/my_model/
    """
    archive_path = Path(archive_path)

    if extract_to is None:
        extract_to = archive_path.parent
    else:
        extract_to = Path(extract_to)
        extract_to.mkdir(parents=True, exist_ok=True)

    try:
        print(f"Extracting archive to {extract_to}...")
        with tarfile.open(archive_path, "r") as tar:
            tar.extractall(path=extract_to)
        print(f"Extracted to {extract_to}")

        return str(extract_to)

    except Exception as e:
        raise RuntimeError(f"Failed to extract archive {archive_path}: {e}") from e


def convert_to_peft_format(
    extracted_dir: str,
    base_model_name: str,
    output_dir: Optional[str] = None,
) -> str:
    """Convert extracted Tinker LoRA weights to PEFT format.

    Tinker LoRA archives contain adapter weights and config. This function
    ensures they're in the format expected by HuggingFace PEFT library.

    Args:
        extracted_dir: Directory with extracted Tinker checkpoint
        base_model_name: Name of base model (e.g., "meta-llama/Llama-3.1-8B")
        output_dir: Directory to save PEFT-formatted weights. If None,
            uses extracted_dir.

    Returns:
        Path to PEFT-formatted adapter directory

    Raises:
        RuntimeError: If conversion fails
        FileNotFoundError: If required files are missing

    Example:
        >>> peft_dir = convert_to_peft_format(
        ...     "weights/my_model/",
        ...     "meta-llama/Llama-3.1-8B"
        ... )
        >>> print(peft_dir)
        weights/my_model/peft_adapter/
    """
    extracted_path = Path(extracted_dir)

    if output_dir is None:
        output_path = extracted_path / "peft_adapter"
    else:
        output_path = Path(output_dir)

    output_path.mkdir(parents=True, exist_ok=True)

    try:
        # Look for adapter files in extracted directory
        # Tinker LoRA format typically includes adapter weights and config
        adapter_files = list(extracted_path.glob("**/*.safetensors")) + list(
            extracted_path.glob("**/*.bin")
        )
        config_files = list(extracted_path.glob("**/adapter_config.json"))

        if not adapter_files:
            raise FileNotFoundError(
                f"No adapter weight files found in {extracted_path}. "
                "Expected .safetensors or .bin files."
            )

        print(f"Found {len(adapter_files)} adapter file(s)")

        # If adapter_config.json doesn't exist, create a basic one
        if not config_files:
            print("No adapter_config.json found, creating default config...")
            adapter_config = {
                "base_model_name_or_path": base_model_name,
                "peft_type": "LORA",
                "task_type": "CAUSAL_LM",
                # Default LoRA hyperparameters
                # These may need adjustment based on actual Tinker training config
                "r": 8,
                "lora_alpha": 16,
                "lora_dropout": 0.0,
                "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj"],
                "inference_mode": False,
            }

            config_path = output_path / "adapter_config.json"
            with open(config_path, "w") as f:
                json.dump(adapter_config, f, indent=2)
            print(f"Created {config_path}")
        else:
            # Copy existing config
            import shutil

            shutil.copy(config_files[0], output_path / "adapter_config.json")
            print(f"Copied adapter_config.json from {config_files[0]}")

        # Copy adapter weights
        for adapter_file in adapter_files:
            import shutil

            dest = output_path / adapter_file.name
            shutil.copy(adapter_file, dest)
            print(f"Copied {adapter_file.name}")

        print(f"PEFT adapter ready at {output_path}")
        return str(output_path)

    except Exception as e:
        raise RuntimeError(
            f"Failed to convert to PEFT format: {e}"
        ) from e


def download_and_convert_weights(
    tinker_checkpoint: str,
    base_model_name: str,
    output_dir: str,
    client: Optional[TinkerClient] = None,
    cleanup_archive: bool = False,
) -> str:
    """Complete workflow: download, extract, and convert Tinker weights to PEFT.

    This is a convenience function that combines download_checkpoint(),
    extract_archive(), and convert_to_peft_format() into a single call.

    Args:
        tinker_checkpoint: Tinker checkpoint path
            (e.g., "tinker://<id>/sampler_weights/final")
        base_model_name: Name of base model (e.g., "meta-llama/Llama-3.1-8B")
        output_dir: Directory to save PEFT adapter
        client: TinkerClient instance. If None, creates a new one.
        cleanup_archive: If True, delete tar archive after extraction

    Returns:
        Path to PEFT-formatted adapter directory (ready to load with PEFT)

    Raises:
        RuntimeError: If any step fails

    Example:
        >>> # Download and convert in one step
        >>> peft_path = download_and_convert_weights(
        ...     tinker_checkpoint="tinker://abc123/sampler_weights/final",
        ...     base_model_name="meta-llama/Llama-3.1-8B",
        ...     output_dir="weights/my_finetuned_model"
        ... )
        >>>
        >>> # Now load with HuggingFace
        >>> from peft import PeftModel
        >>> from transformers import AutoModelForCausalLM
        >>> model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B")
        >>> model = PeftModel.from_pretrained(model, peft_path)
    """
    print(f"\n{'=' * 70}")
    print("Downloading and Converting Tinker Weights to PEFT Format")
    print(f"{'=' * 70}")
    print(f"Checkpoint: {tinker_checkpoint}")
    print(f"Base Model: {base_model_name}")
    print(f"Output Dir: {output_dir}")
    print(f"{'=' * 70}\n")

    # Step 1: Download archive
    archive_path = download_checkpoint(tinker_checkpoint, output_dir, client)

    # Step 2: Extract archive
    extract_dir = extract_archive(archive_path)

    # Step 3: Convert to PEFT format
    peft_adapter_path = convert_to_peft_format(extract_dir, base_model_name)

    # Step 4: Cleanup if requested
    if cleanup_archive:
        import os

        os.remove(archive_path)
        print(f"Removed archive: {archive_path}")

    print(f"\n{'=' * 70}")
    print("✓ Download and conversion complete!")
    print(f"{'=' * 70}")
    print(f"PEFT adapter path: {peft_adapter_path}")
    print(f"\nYou can now load this adapter with:")
    print(f"  from peft import PeftModel")
    print(f"  from transformers import AutoModelForCausalLM")
    print(f"  model = AutoModelForCausalLM.from_pretrained('{base_model_name}')")
    print(f"  model = PeftModel.from_pretrained(model, '{peft_adapter_path}')")
    print(f"{'=' * 70}\n")

    return peft_adapter_path


__all__ = [
    "download_checkpoint",
    "extract_archive",
    "convert_to_peft_format",
    "download_and_convert_weights",
]
