"""Configuration management utilities."""

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Union

import yaml
from omegaconf import DictConfig, OmegaConf

logger = logging.getLogger(__name__)


def load_config(config_path: Union[str, Path]) -> DictConfig:
    """Load configuration from YAML file.

    Args:
        config_path: Path to YAML configuration file

    Returns:
        OmegaConf DictConfig object

    Example:
        >>> config = load_config("configs/linear_probe.yaml")
        >>> print(config.model.name)
        >>> print(config.training.learning_rate)
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)

    config = OmegaConf.create(config_dict)
    logger.info(f"Loaded configuration from {config_path}")

    return config


def save_config(config: Union[Dict, DictConfig], save_path: Union[str, Path]) -> None:
    """Save configuration to YAML file.

    Args:
        config: Configuration dictionary or DictConfig
        save_path: Path to save configuration

    Example:
        >>> save_config(config, "outputs/experiment_1/config.yaml")
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(config, DictConfig):
        config_dict = OmegaConf.to_container(config, resolve=True)
    else:
        config_dict = config

    with open(save_path, 'w') as f:
        yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)

    logger.info(f"Saved configuration to {save_path}")


def merge_configs(base_config: DictConfig, override_config: Dict[str, Any]) -> DictConfig:
    """Merge override configuration into base configuration.

    Args:
        base_config: Base configuration
        override_config: Override values

    Returns:
        Merged configuration

    Example:
        >>> base = load_config("configs/base.yaml")
        >>> override = {"training": {"learning_rate": 1e-4}}
        >>> config = merge_configs(base, override)
    """
    override_conf = OmegaConf.create(override_config)
    merged = OmegaConf.merge(base_config, override_conf)
    return merged


def get_default_config() -> DictConfig:
    """Get default configuration template.

    Returns:
        Default configuration with sensible defaults
    """
    default = {
        "experiment": {
            "name": "default_experiment",
            "seed": 42,
            "output_dir": "outputs",
        },
        "model": {
            "name": "meta-llama/Llama-3.1-8B",
            "quantization": "8bit",
            "device": "cuda",
        },
        "data": {
            "dataset": "mmlu",
            "split_ratio": [0.7, 0.15, 0.15],
            "batch_size": 16,
            "max_length": 512,
        },
        "extraction": {
            "layers": None,  # None means optimal layers from registry
            "token_position": "last",
            "cache_dir": "cache",
        },
        "probe": {
            "type": "linear",
            "hidden_dim": None,
            "dropout": 0.0,
        },
        "training": {
            "epochs": 50,
            "learning_rate": 1e-3,
            "weight_decay": 1e-5,
            "early_stopping_patience": 5,
        },
        "evaluation": {
            "metrics": ["ece", "brier", "auroc", "aupr"],
            "calibration_method": "temperature",
            "num_bins": 10,
            "save_plots": True,
        },
    }

    return OmegaConf.create(default)
