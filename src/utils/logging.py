"""Logging and experiment tracking utilities."""

import logging
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Union

from loguru import logger as loguru_logger


def setup_logging(
    log_level: str = "INFO",
    log_file: Optional[Union[str, Path]] = None,
    format_string: Optional[str] = None,
) -> None:
    """Setup logging configuration.

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Optional file to write logs to
        format_string: Custom format string for logs

    Example:
        >>> setup_logging(log_level="DEBUG", log_file="outputs/experiment.log")
    """
    # Remove default handler
    loguru_logger.remove()

    # Default format
    if format_string is None:
        format_string = (
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
            "<level>{message}</level>"
        )

    # Add console handler
    loguru_logger.add(
        sys.stderr,
        format=format_string,
        level=log_level,
        colorize=True,
    )

    # Add file handler if specified
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        loguru_logger.add(
            log_path,
            format=format_string,
            level=log_level,
            rotation="10 MB",
            retention="7 days",
        )
        loguru_logger.info(f"Logging to file: {log_path}")

    # Configure standard logging to use loguru
    class InterceptHandler(logging.Handler):
        def emit(self, record):
            # Get corresponding Loguru level if it exists
            try:
                level = loguru_logger.level(record.levelname).name
            except ValueError:
                level = record.levelno

            # Find caller from where originated the logged message
            frame, depth = logging.currentframe(), 2
            while frame.f_code.co_filename == logging.__file__:
                frame = frame.f_back
                depth += 1

            loguru_logger.opt(depth=depth, exception=record.exc_info).log(
                level, record.getMessage()
            )

    # Intercept standard logging
    logging.basicConfig(handlers=[InterceptHandler()], level=0, force=True)


def setup_wandb(
    project: str,
    config: Dict[str, Any],
    name: Optional[str] = None,
    tags: Optional[list] = None,
    notes: Optional[str] = None,
) -> Any:
    """Setup Weights & Biases experiment tracking.

    Args:
        project: W&B project name
        config: Configuration dictionary to log
        name: Run name (optional)
        tags: List of tags (optional)
        notes: Run notes/description (optional)

    Returns:
        W&B run object

    Example:
        >>> run = setup_wandb(
        ...     project="llm-confidence",
        ...     config=config_dict,
        ...     name="linear-probe-llama-8b",
        ...     tags=["baseline", "llama"]
        ... )
    """
    try:
        import wandb

        run = wandb.init(
            project=project,
            config=config,
            name=name,
            tags=tags,
            notes=notes,
        )

        loguru_logger.info(f"Initialized W&B run: {run.name} ({run.id})")
        return run

    except ImportError:
        loguru_logger.warning("wandb not installed. Skipping W&B logging.")
        return None


class ExperimentLogger:
    """Comprehensive experiment logger combining file logging and W&B.

    Example:
        >>> logger = ExperimentLogger(
        ...     experiment_name="linear_probe_llama",
        ...     output_dir="outputs",
        ...     config=config_dict,
        ...     use_wandb=True,
        ...     wandb_project="llm-confidence"
        ... )
        >>> logger.log_metrics({"ece": 0.05, "auroc": 0.82}, step=1)
        >>> logger.save_artifact("model.pt", "checkpoint")
    """

    def __init__(
        self,
        experiment_name: str,
        output_dir: Union[str, Path] = "outputs",
        config: Optional[Dict[str, Any]] = None,
        use_wandb: bool = False,
        wandb_project: Optional[str] = None,
        wandb_tags: Optional[list] = None,
    ):
        """Initialize experiment logger.

        Args:
            experiment_name: Name of the experiment
            output_dir: Base output directory
            config: Configuration dictionary
            use_wandb: Whether to use W&B tracking
            wandb_project: W&B project name
            wandb_tags: W&B tags
        """
        self.experiment_name = experiment_name
        self.output_dir = Path(output_dir) / experiment_name
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Setup file logging
        log_file = self.output_dir / "experiment.log"
        setup_logging(log_file=log_file)

        # Setup W&B if requested
        self.wandb_run = None
        if use_wandb and wandb_project:
            self.wandb_run = setup_wandb(
                project=wandb_project,
                config=config or {},
                name=experiment_name,
                tags=wandb_tags,
            )

        loguru_logger.info(f"Experiment: {experiment_name}")
        loguru_logger.info(f"Output directory: {self.output_dir}")

    def log_metrics(
        self,
        metrics: Dict[str, float],
        step: Optional[int] = None,
        prefix: Optional[str] = None,
    ) -> None:
        """Log metrics.

        Args:
            metrics: Dictionary of metric names and values
            step: Training step/epoch (optional)
            prefix: Prefix for metric names (e.g., "train/", "val/")
        """
        # Add prefix if specified
        if prefix:
            metrics = {f"{prefix}{k}": v for k, v in metrics.items()}

        # Log to console
        metrics_str = " | ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
        loguru_logger.info(f"Metrics [{step or 'N/A'}]: {metrics_str}")

        # Log to W&B
        if self.wandb_run:
            import wandb
            wandb.log(metrics, step=step)

    def save_artifact(
        self,
        artifact_path: Union[str, Path],
        artifact_type: str = "file",
    ) -> Path:
        """Save artifact to output directory.

        Args:
            artifact_path: Path to artifact file
            artifact_type: Type of artifact (for organization)

        Returns:
            Path where artifact was saved
        """
        artifact_path = Path(artifact_path)
        dest_dir = self.output_dir / artifact_type
        dest_dir.mkdir(exist_ok=True)

        dest_path = dest_dir / artifact_path.name

        # Copy or move artifact
        import shutil
        shutil.copy2(artifact_path, dest_path)

        loguru_logger.info(f"Saved {artifact_type}: {dest_path}")

        # Log to W&B
        if self.wandb_run:
            import wandb
            wandb.save(str(dest_path))

        return dest_path

    def finish(self) -> None:
        """Finish experiment and cleanup."""
        if self.wandb_run:
            import wandb
            wandb.finish()
            loguru_logger.info("Finished W&B run")

        loguru_logger.info(f"Experiment completed: {self.experiment_name}")
