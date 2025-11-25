"""Hierarchical Multi-Scale Probe Experiment.

This script implements the novel hierarchical probing approach for uncertainty
quantification in LLMs. It operates at four levels of granularity:
1. Token-level: Per-token confidence
2. Span-level: Phrase/chunk confidence
3. Semantic-level: Sentence/clause confidence
4. Global-level: Overall answer confidence

The experiment compares hierarchical probing against baseline approaches
(linear and MLP probes) and evaluates using standard calibration metrics.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import argparse
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from loguru import logger

from src.utils import load_config, ExperimentLogger, setup_logging
from src.models import ModelLoader, HiddenStateExtractor
from src.data import MMLUDataset
from src.probes import HierarchicalProbe, LinearProbe, MLPProbe
from src.evaluation import CalibrationMetrics


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run hierarchical probe experiment for uncertainty quantification"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/hierarchical_probe.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable caching of hidden states",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Run in debug mode with reduced dataset",
    )
    parser.add_argument(
        "--skip-baselines",
        action="store_true",
        help="Skip baseline experiments",
    )
    return parser.parse_args()


def generate_llm_predictions(model, tokenizer, prompts, batch_size=8):
    """Generate predictions from the LLM for evaluation.

    Args:
        model: Language model
        tokenizer: Tokenizer
        prompts: List of prompts
        batch_size: Batch size for generation

    Returns:
        List of generated responses and predicted answer indices
    """
    logger.info(f"Generating LLM predictions for {len(prompts)} prompts...")
    model.eval()

    predictions = []
    predicted_indices = []

    for i in tqdm(range(0, len(prompts), batch_size), desc="Generating predictions"):
        batch_prompts = prompts[i : i + batch_size]

        # Tokenize
        inputs = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(model.device)

        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=10,  # Short answer for multiple choice
                do_sample=False,  # Greedy decoding
                pad_token_id=tokenizer.pad_token_id,
            )

        # Decode responses
        responses = tokenizer.batch_decode(outputs, skip_special_tokens=True)

        for response in responses:
            # Extract answer (look for A, B, C, or D)
            answer_text = response.split("Answer:")[-1].strip()
            predicted_letter = answer_text[0] if answer_text else "A"

            # Convert to index (A=0, B=1, C=2, D=3)
            if predicted_letter in ["A", "B", "C", "D"]:
                pred_idx = ord(predicted_letter) - ord("A")
            else:
                pred_idx = 0  # Default to A if unclear

            predictions.append(response)
            predicted_indices.append(pred_idx)

    return predictions, predicted_indices


def prepare_training_data(hidden_states, predicted_indices, true_labels):
    """Prepare training data with correctness labels.

    Args:
        hidden_states: Extracted hidden states
        predicted_indices: Model's predicted answers
        true_labels: Ground truth labels

    Returns:
        X (hidden states), y (binary correctness labels)
    """
    # Binary labels: 1 if correct, 0 if incorrect
    correctness = (np.array(predicted_indices) == np.array(true_labels)).astype(float)

    logger.info(f"Accuracy: {correctness.mean():.2%}")
    logger.info(f"Positive samples: {correctness.sum()}/{len(correctness)}")

    return hidden_states, correctness


def train_and_evaluate_probe(
    probe,
    probe_name,
    X_train,
    y_train,
    X_val,
    y_val,
    X_test,
    y_test,
    config,
    exp_logger,
):
    """Train and evaluate a probe.

    Args:
        probe: Probe instance
        probe_name: Name for logging
        X_train, y_train: Training data
        X_val, y_val: Validation data
        X_test, y_test: Test data
        config: Configuration
        exp_logger: Experiment logger

    Returns:
        Dictionary with evaluation metrics
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"Training {probe_name} Probe")
    logger.info(f"{'='*60}")

    # Training
    history = probe.fit(
        X_train,
        y_train,
        X_val,
        y_val,
        epochs=config.training.epochs,
        batch_size=config.data.batch_size,
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
        early_stopping_patience=config.training.early_stopping_patience,
        verbose=True,
    )

    # Log training history
    exp_logger.log_metrics(
        {
            f"{probe_name}/train_loss": history["train_loss"][-1],
            f"{probe_name}/val_loss": history["val_loss"][-1] if history["val_loss"] else 0,
            f"{probe_name}/best_epoch": history["best_epoch"],
        }
    )

    # Evaluate on test set
    logger.info(f"\nEvaluating {probe_name} on test set...")
    confidences = probe.predict(X_test)

    # Compute binary predictions (threshold at 0.5)
    predictions = (confidences > 0.5).astype(int)

    # Compute metrics
    metrics = CalibrationMetrics(predictions, confidences, y_test)
    ece, _ = metrics.ece(num_bins=config.evaluation.num_bins)
    brier = metrics.brier_score()
    auroc = metrics.auroc()
    accuracy = metrics.accuracy()

    results = {
        "ece": ece,
        "brier": brier,
        "auroc": auroc,
        "accuracy": accuracy,
        "num_parameters": probe.get_num_parameters(),
    }

    logger.info(f"\n{probe_name} Results:")
    logger.info(f"  ECE: {ece:.4f}")
    logger.info(f"  Brier Score: {brier:.4f}")
    logger.info(f"  AUROC: {auroc:.4f}")
    logger.info(f"  Accuracy: {accuracy:.2%}")
    logger.info(f"  Num Parameters: {results['num_parameters']:,}")

    # Log to experiment tracker
    exp_logger.log_metrics(
        {
            f"{probe_name}/ece": ece,
            f"{probe_name}/brier": brier,
            f"{probe_name}/auroc": auroc,
            f"{probe_name}/accuracy": accuracy,
        }
    )

    # Save probe
    save_path = Path(config.experiment.output_dir) / probe_name.lower() / "probe.pt"
    probe.save(str(save_path))

    return results


def main():
    """Run hierarchical probe experiment."""
    args = parse_args()

    # Load configuration
    config = load_config(args.config)
    if args.no_cache:
        config.extraction.use_cache = False
    if args.debug:
        config.data.num_samples = 100
        config.training.epochs = 10

    # Setup logging
    setup_logging(log_level="INFO")
    logger.info(f"Starting experiment: {config.experiment.name}")
    logger.info(f"Output directory: {config.experiment.output_dir}")

    # Initialize experiment logger
    exp_logger = ExperimentLogger(
        experiment_name=config.experiment.name,
        output_dir=config.experiment.output_dir,
        config=config,
        use_wandb=config.logging.use_wandb,
        wandb_project=config.logging.get("wandb_project", "llm-confidence"),
    )

    # Set random seed
    torch.manual_seed(config.experiment.seed)
    np.random.seed(config.experiment.seed)

    # =========================================================================
    # 1. Load Dataset
    # =========================================================================
    logger.info("\n" + "="*60)
    logger.info("Loading Dataset")
    logger.info("="*60)

    dataset = MMLUDataset(
        split="test",
        subjects=config.data.get("subjects", None),
        max_samples=config.data.get("num_samples", None),
    )
    logger.info(f"Loaded {len(dataset)} examples from MMLU")

    # Get prompts and labels
    prompts = dataset.get_prompts_for_generation(
        format_type=config.data.get("format_type", "standard")
    )
    true_labels = np.array([ex["answer"] for ex in dataset.data])

    # =========================================================================
    # 2. Load Model
    # =========================================================================
    logger.info("\n" + "="*60)
    logger.info("Loading Model")
    logger.info("="*60)

    model_loader = ModelLoader(
        model_name=config.model.name,
        quantization=config.model.get("quantization", None),
        device_map=config.model.get("device", "auto"),
        use_auth_token=config.model.get("use_auth_token", None),
    )
    model, tokenizer = model_loader.load()
    model_info = model_loader.get_model_info()
    logger.info(f"Model: {model_info['name']}")
    logger.info(f"Num layers: {model_info['num_layers']}")
    logger.info(f"Hidden size: {model_info['hidden_size']}")

    # =========================================================================
    # 3. Generate Predictions
    # =========================================================================
    logger.info("\n" + "="*60)
    logger.info("Generating LLM Predictions")
    logger.info("="*60)

    predictions, predicted_indices = generate_llm_predictions(
        model, tokenizer, prompts, batch_size=config.data.batch_size
    )

    # Calculate accuracy
    accuracy = (np.array(predicted_indices) == true_labels).mean()
    logger.info(f"LLM Accuracy: {accuracy:.2%}")
    exp_logger.log_metrics({"llm/accuracy": accuracy})

    # =========================================================================
    # 4. Extract Hidden States
    # =========================================================================
    logger.info("\n" + "="*60)
    logger.info("Extracting Hidden States")
    logger.info("="*60)

    extractor = HiddenStateExtractor(model, tokenizer)

    # Determine layers to extract
    if config.extraction.layers is None:
        # Use optimal layers from registry
        layers = model_loader.config.optimal_layers
    else:
        layers = config.extraction.layers

    logger.info(f"Extracting from layers: {layers}")

    # Extract hidden states
    hidden_states = extractor.extract(
        texts=prompts,
        layers=layers,
        token_position=config.extraction.get("token_position", "last"),
        batch_size=config.data.batch_size,
        max_length=config.data.max_length,
        use_cache=config.extraction.get("use_cache", True),
        cache_dir=config.extraction.get("cache_dir", "cache"),
    )

    logger.info(f"Extracted hidden states shape: {hidden_states.shape}")

    # =========================================================================
    # 5. Prepare Training Data
    # =========================================================================
    logger.info("\n" + "="*60)
    logger.info("Preparing Training Data")
    logger.info("="*60)

    X, y = prepare_training_data(hidden_states, predicted_indices, true_labels)

    # Split into train/val/test
    split_ratio = config.data.split_ratio
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=split_ratio[2], random_state=config.experiment.seed, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val,
        y_train_val,
        test_size=split_ratio[1] / (split_ratio[0] + split_ratio[1]),
        random_state=config.experiment.seed,
        stratify=y_train_val,
    )

    logger.info(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    # =========================================================================
    # 6. Train Hierarchical Probe
    # =========================================================================
    input_dim = model_info["hidden_size"]

    hierarchical_probe = HierarchicalProbe(
        input_dim=input_dim,
        hidden_dim=config.probe.get("hidden_dim", 512),
        num_layers=config.probe.get("num_layers", len(layers)),
        use_layer_fusion=config.probe.get("use_layer_fusion", True),
        device=config.model.get("device", "cuda"),
    )

    hierarchical_results = train_and_evaluate_probe(
        hierarchical_probe,
        "Hierarchical",
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
        config,
        exp_logger,
    )

    # =========================================================================
    # 7. Train Baseline Probes (if not skipped)
    # =========================================================================
    baseline_results = {}

    if not args.skip_baselines:
        logger.info("\n" + "="*60)
        logger.info("Training Baseline Probes")
        logger.info("="*60)

        # Linear probe
        linear_probe = LinearProbe(input_dim=input_dim, dropout=0.0)
        baseline_results["Linear"] = train_and_evaluate_probe(
            linear_probe,
            "Linear",
            X_train,
            y_train,
            X_val,
            y_val,
            X_test,
            y_test,
            config,
            exp_logger,
        )

        # MLP probe
        mlp_probe = MLPProbe(
            input_dim=input_dim, hidden_dim=512, num_layers=2, dropout=0.1
        )
        baseline_results["MLP"] = train_and_evaluate_probe(
            mlp_probe,
            "MLP",
            X_train,
            y_train,
            X_val,
            y_val,
            X_test,
            y_test,
            config,
            exp_logger,
        )

    # =========================================================================
    # 8. Compare Results
    # =========================================================================
    logger.info("\n" + "="*60)
    logger.info("Final Results Comparison")
    logger.info("="*60)

    all_results = {"Hierarchical": hierarchical_results, **baseline_results}

    # Print comparison table
    logger.info("\nMethod           | ECE    | Brier  | AUROC  | Accuracy | Params")
    logger.info("-" * 70)
    for method, results in all_results.items():
        logger.info(
            f"{method:16} | {results['ece']:.4f} | {results['brier']:.4f} | "
            f"{results['auroc']:.4f} | {results['accuracy']:.2%} | "
            f"{results['num_parameters']:,}"
        )

    # Calculate improvements
    if baseline_results:
        linear_ece = baseline_results["Linear"]["ece"]
        hierarchical_ece = hierarchical_results["ece"]
        ece_improvement = (linear_ece - hierarchical_ece) / linear_ece * 100

        logger.info(f"\nHierarchical ECE improvement over Linear: {ece_improvement:.1f}%")
        exp_logger.log_metrics({"comparison/ece_improvement_pct": ece_improvement})

    # Finish logging
    exp_logger.finish()
    logger.info(f"\nExperiment complete! Results saved to {config.experiment.output_dir}")


if __name__ == "__main__":
    main()
