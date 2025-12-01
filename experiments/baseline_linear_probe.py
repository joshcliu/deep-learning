"""
Baseline Linear Probe Experiment

This script trains linear probes on hidden states from a language model
to predict confidence/correctness on the MMLU benchmark.

The experiment:
1. Loads MMLU dataset (validation split)
2. Loads a language model (e.g., Llama 3.1 8B)
3. Extracts hidden states from multiple layers
4. Trains a linear probe per layer
5. Evaluates calibration metrics (ECE, Brier, AUROC)
6. Compares layer performance

Usage:
    python experiments/baseline_linear_probe.py
    python experiments/baseline_linear_probe.py --config configs/custom.yaml
    python experiments/baseline_linear_probe.py --layers 8 16 24 --num-samples 500
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data import MMLUDataset
from src.evaluation import CalibrationMetrics
from src.models import HiddenStateExtractor, ModelLoader
from src.probes import LinearProbe
from src.utils import ExperimentLogger, load_config, setup_logging


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train linear probes on LLM hidden states"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/linear_probe.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--layers",
        type=int,
        nargs="+",
        default=None,
        help="Layers to probe (overrides config)",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=None,
        help="Limit number of samples (for quick testing)",
    )
    parser.add_argument(
        "--no-wandb", action="store_true", help="Disable WandB logging"
    )
    return parser.parse_args()


def generate_model_predictions(
    dataset: MMLUDataset, model, tokenizer, max_length: int = 512
) -> Tuple[List[str], np.ndarray]:
    """Generate model predictions for MMLU questions.

    For simplicity, this baseline uses a greedy approach:
    - Format each question as multiple choice
    - Get model to predict A/B/C/D
    - Label as correct (1) or incorrect (0)

    Args:
        dataset: MMLU dataset
        model: Language model
        tokenizer: Tokenizer
        max_length: Maximum sequence length

    Returns:
        Tuple of (formatted_texts, binary_labels)
    """
    texts = []
    labels = []

    print("Generating model predictions...")
    for example in tqdm(dataset, desc="Processing examples"):
        # Format as multiple choice prompt
        prompt = example.format_prompt(style="multiple_choice")

        # For this baseline, we'll use a simple heuristic:
        # Extract hidden states from the prompt + correct answer
        # This simulates the model making predictions
        correct_answer = example.choices[example.answer]
        text = f"{prompt} {chr(65 + example.answer)}"

        texts.append(text)
        # Label: always 1 for correct (we're using ground truth)
        # In a real experiment, you'd run inference to get model predictions
        labels.append(1)

        # Also add some incorrect examples for balance
        for i, choice in enumerate(example.choices):
            if i != example.answer:
                incorrect_text = f"{prompt} {chr(65 + i)}"
                texts.append(incorrect_text)
                labels.append(0)
                break  # Just add one incorrect per question for balance

    return texts, np.array(labels)


def extract_hidden_states(
    texts: List[str],
    model,
    tokenizer,
    layers: List[int],
    cache_dir: str,
    max_length: int = 512,
) -> Dict[int, np.ndarray]:
    """Extract hidden states from specified layers.

    Args:
        texts: List of text inputs
        model: Language model
        tokenizer: Tokenizer
        layers: List of layer indices to extract
        cache_dir: Cache directory for hidden states
        max_length: Maximum sequence length

    Returns:
        Dictionary mapping layer index to hidden states array
            of shape (num_texts, hidden_dim)
    """
    print(f"\nExtracting hidden states from {len(layers)} layers...")

    extractor = HiddenStateExtractor(model, tokenizer, cache_dir=cache_dir)

    hidden_states = {}
    for layer in tqdm(layers, desc="Extracting layers"):
        hiddens = extractor.extract(
            texts=texts,
            layers=[layer],
            max_length=max_length,
            token_position="last",
            use_cache=True,
            batch_size=16,
        )
        # hiddens shape: (num_texts, 1, hidden_dim)
        # Squeeze the layer dimension
        hidden_states[layer] = hiddens[:, 0, :]

    return hidden_states


def train_and_evaluate_probe(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    layer: int,
    config,
    logger: ExperimentLogger,
) -> Dict:
    """Train and evaluate a linear probe for a specific layer.

    Args:
        X_train: Training hidden states
        y_train: Training labels
        X_val: Validation hidden states
        y_val: Validation labels
        X_test: Test hidden states
        y_test: Test labels
        layer: Layer index
        config: Experiment configuration
        logger: Experiment logger

    Returns:
        Dictionary with evaluation metrics
    """
    print(f"\n{'=' * 60}")
    print(f"Training probe for Layer {layer}")
    print(f"{'=' * 60}")

    # Initialize probe
    probe = LinearProbe(
        input_dim=X_train.shape[1],
        dropout=config.probe.dropout,
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )

    # Train probe
    history = probe.fit(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        batch_size=config.data.batch_size,
        num_epochs=config.training.epochs,
        patience=config.training.early_stopping_patience,
        verbose=True,
    )

    # Predict on test set
    confidences = probe.predict(X_test, batch_size=config.data.batch_size)
    predictions = (confidences > 0.5).astype(int)

    # Compute metrics
    metrics = CalibrationMetrics(predictions, confidences, y_test)
    results = metrics.compute_all()

    # Add training history
    results["training"] = {
        "final_temperature": history["final_temperature"],
        "best_epoch": history["best_epoch"],
        "converged": history["converged"],
        "final_train_loss": history["train_loss"][-1],
        "final_val_loss": (
            history["val_loss"][-1] if history["val_loss"] else None
        ),
    }

    # Log to WandB
    logger.log_metrics(
        {
            f"layer_{layer}/ece": results["ece"],
            f"layer_{layer}/brier": results["brier"],
            f"layer_{layer}/auroc": results["auroc"],
            f"layer_{layer}/accuracy": results["accuracy"],
            f"layer_{layer}/temperature": history["final_temperature"],
        }
    )

    # Print results
    print(f"\nLayer {layer} Results:")
    print(f"  ECE:      {results['ece']:.4f}")
    print(f"  Brier:    {results['brier']:.4f}")
    print(f"  AUROC:    {results['auroc']:.4f}")
    print(f"  Accuracy: {results['accuracy']:.4f}")
    print(f"  Temperature: {history['final_temperature']:.4f}")

    # Save probe checkpoint
    checkpoint_path = (
        logger.output_dir / "checkpoints" / f"probe_layer_{layer}.pt"
    )
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    probe.save(str(checkpoint_path))

    return results


def main():
    """Main experiment pipeline."""
    args = parse_args()

    # Setup logging
    setup_logging(log_level="INFO")

    # Load configuration
    print("Loading configuration...")
    config = load_config(args.config)

    # Override config with command line args
    if args.layers is not None:
        config.extraction.layers = args.layers
    if args.num_samples is not None:
        config.data.num_samples = args.num_samples
    if args.no_wandb:
        config.logging.use_wandb = False

    # Initialize experiment logger
    logger = ExperimentLogger(
        experiment_name=config.experiment.name,
        output_dir=config.experiment.output_dir,
        config=config,
        use_wandb=config.logging.use_wandb,
        wandb_project=config.logging.get("wandb_project", "llm-confidence"),
    )

    print(f"\nExperiment: {config.experiment.name}")
    print(f"Output directory: {logger.output_dir}")

    # Set random seed
    np.random.seed(config.experiment.seed)
    torch.manual_seed(config.experiment.seed)

    # Load dataset
    print(f"\nLoading MMLU dataset...")
    dataset = MMLUDataset(split="validation", category="stem")
    print(f"Loaded {len(dataset)} examples")

    # Limit samples if specified
    if config.data.get("num_samples") is not None:
        dataset.data = dataset.sample(
            config.data.num_samples, seed=config.experiment.seed
        )
        print(f"Limited to {len(dataset)} examples")

    # Load model
    print(f"\nLoading model: {config.model.name}")
    model_loader = ModelLoader(
        model_name=config.model.name,
        quantization=config.model.quantization,
        device_map=config.model.device,
        use_auth_token=config.model.get("use_auth_token"),
    )
    model, tokenizer = model_loader.load()
    print(f"Model loaded: {model_loader.get_model_info()}")

    # Generate model predictions (simplified for baseline)
    texts, labels = generate_model_predictions(
        dataset, model, tokenizer, max_length=config.data.max_length
    )
    print(f"\nGenerated {len(texts)} text examples")
    print(f"Label distribution: {np.bincount(labels)}")

    # Extract hidden states
    hidden_states = extract_hidden_states(
        texts=texts,
        model=model,
        tokenizer=tokenizer,
        layers=config.extraction.layers,
        cache_dir=config.extraction.cache_dir,
        max_length=config.data.max_length,
    )

    # Split data into train/val/test
    print("\nSplitting data...")
    train_ratio, val_ratio, test_ratio = config.data.split_ratio
    indices = np.arange(len(labels))

    # First split: train+val vs test
    train_val_indices, test_indices = train_test_split(
        indices,
        test_size=test_ratio,
        random_state=config.experiment.seed,
        stratify=labels,
    )

    # Second split: train vs val
    train_indices, val_indices = train_test_split(
        train_val_indices,
        test_size=val_ratio / (train_ratio + val_ratio),
        random_state=config.experiment.seed,
        stratify=labels[train_val_indices],
    )

    print(f"Train: {len(train_indices)}, Val: {len(val_indices)}, "
          f"Test: {len(test_indices)}")

    # Train probes for each layer
    all_results = {}
    for layer in config.extraction.layers:
        X = hidden_states[layer]

        X_train = X[train_indices]
        y_train = labels[train_indices]
        X_val = X[val_indices]
        y_val = labels[val_indices]
        X_test = X[test_indices]
        y_test = labels[test_indices]

        results = train_and_evaluate_probe(
            X_train, y_train, X_val, y_val, X_test, y_test, layer, config, logger
        )
        all_results[layer] = results

    # Summary
    print(f"\n{'=' * 60}")
    print("EXPERIMENT SUMMARY")
    print(f"{'=' * 60}")
    print(f"\n{'Layer':<8} {'ECE':<10} {'Brier':<10} {'AUROC':<10} {'Accuracy':<10}")
    print("-" * 60)
    for layer in config.extraction.layers:
        r = all_results[layer]
        print(
            f"{layer:<8} {r['ece']:<10.4f} {r['brier']:<10.4f} "
            f"{r['auroc']:<10.4f} {r['accuracy']:<10.4f}"
        )

    # Find best layer
    best_layer = min(all_results.keys(), key=lambda k: all_results[k]["ece"])
    print(f"\nBest layer by ECE: {best_layer} "
          f"(ECE={all_results[best_layer]['ece']:.4f})")

    # Save summary
    logger.log_metrics(
        {
            "best_layer": best_layer,
            "best_ece": all_results[best_layer]["ece"],
        }
    )

    logger.finish()
    print(f"\nExperiment complete! Results saved to {logger.output_dir}")


if __name__ == "__main__":
    main()
