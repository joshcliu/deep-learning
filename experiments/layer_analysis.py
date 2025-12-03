"""
Systematic Layer Analysis Experiment

This script performs a comprehensive analysis of probe performance across
all layers of a language model to:
1. Validate the "middle layers optimal" hypothesis
2. Identify the best layer(s) for uncertainty quantification
3. Generate visualizations of layer-wise performance

The experiment trains linear probes on hidden states from ALL layers
and compares calibration metrics across layers.

Usage:
    python experiments/layer_analysis.py
    python experiments/layer_analysis.py --model meta-llama/Llama-2-7b-hf
    python experiments/layer_analysis.py --num-samples 200 --quick
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data import MMLUDataset
from src.evaluation import CalibrationMetrics
from src.models import HiddenStateExtractor, ModelLoader
from src.probes import LinearProbe
from src.utils import ExperimentLogger, setup_logging


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Systematic layer analysis for uncertainty probing"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="mistralai/Mistral-7B-v0.1",
        help="Model name or path",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="mmlu",
        choices=["mmlu", "triviaqa", "gsm8k"],
        help="Dataset to use",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=500,
        help="Number of dataset samples",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick mode: test quartile layers only",
    )
    parser.add_argument(
        "--quantization",
        type=str,
        default="8bit",
        choices=["4bit", "8bit", "none"],
        help="Model quantization",
    )
    parser.add_argument(
        "--output-dir", type=str, default="outputs/layer_analysis", help="Output directory"
    )
    return parser.parse_args()


def get_layer_indices(num_layers: int, quick: bool = False) -> List[int]:
    """Get layer indices to analyze.

    Args:
        num_layers: Total number of layers in model
        quick: If True, only analyze quartile layers

    Returns:
        List of layer indices to analyze
    """
    if quick:
        # Quartile positions: 0%, 25%, 50%, 75%, 100%
        quartiles = [0, num_layers // 4, num_layers // 2,
                     3 * num_layers // 4, num_layers - 1]
        print(f"Quick mode: Testing quartile layers {quartiles}")
        return quartiles
    else:
        # All layers
        layers = list(range(num_layers))
        print(f"Full mode: Testing all {num_layers} layers")
        return layers


def generate_dataset_examples(dataset, sample_size: int, seed: int = 42):
    """Generate balanced examples from dataset.

    Args:
        dataset: Dataset instance
        sample_size: Number of examples to generate
        seed: Random seed

    Returns:
        Tuple of (texts, labels)
    """
    texts = []
    labels = []

    # Sample from dataset
    sampled = dataset.sample(n=min(sample_size, len(dataset)), seed=seed)

    print(f"Generating examples from {len(sampled)} dataset items...")
    for example in tqdm(sampled):
        prompt = example.format_prompt(style="multiple_choice")

        # Add correct answer example
        correct_text = f"{prompt} {chr(65 + example.answer)}"
        texts.append(correct_text)
        labels.append(1)

        # Add one incorrect answer example for balance
        for i in range(len(example.choices)):
            if i != example.answer:
                incorrect_text = f"{prompt} {chr(65 + i)}"
                texts.append(incorrect_text)
                labels.append(0)
                break

    return texts, np.array(labels)


def train_probe_for_layer(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    layer: int,
    config: Dict,
) -> Dict:
    """Train and evaluate probe for a single layer.

    Args:
        X_train: Training hidden states
        y_train: Training labels
        X_val: Validation hidden states
        y_val: Validation labels
        layer: Layer index
        config: Configuration dictionary

    Returns:
        Dictionary with metrics and predictions
    """
    probe = LinearProbe(
        input_dim=X_train.shape[1],
        dropout=config.get("dropout", 0.0),
        lr=config.get("lr", 1e-3),
        weight_decay=config.get("weight_decay", 1e-5),
    )

    # Train
    history = probe.fit(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        batch_size=config.get("batch_size", 128),
        num_epochs=config.get("epochs", 50),
        patience=config.get("patience", 5),
        verbose=False,
    )

    # Predict on validation set
    confidences = probe.predict(X_val, batch_size=config.get("batch_size", 128))
    predictions = (confidences > 0.5).astype(int)

    # Compute metrics
    metrics = CalibrationMetrics(predictions, confidences, y_val)
    results = metrics.compute_all()

    # Add training info
    results["temperature"] = history["final_temperature"]
    results["converged"] = history["converged"]

    return results


def plot_layer_analysis(
    results: Dict[int, Dict],
    num_layers: int,
    output_dir: Path,
    model_name: str,
):
    """Generate visualization plots for layer analysis.

    Args:
        results: Dictionary mapping layer to metrics
        num_layers: Total number of layers
        output_dir: Directory to save plots
        model_name: Model name for plot titles
    """
    layers = sorted(results.keys())
    metrics_to_plot = ["ece", "brier", "auroc", "accuracy"]

    # Set style
    sns.set_style("whitegrid")
    plt.rcParams["figure.figsize"] = (14, 10)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        f"Layer-wise Probe Performance: {model_name}", fontsize=16, y=1.00
    )

    for idx, metric in enumerate(metrics_to_plot):
        ax = axes[idx // 2, idx % 2]

        values = [results[layer][metric] for layer in layers]

        # Plot line
        ax.plot(layers, values, marker="o", linewidth=2, markersize=6)

        # Highlight best layer
        if metric == "ece" or metric == "brier":
            best_idx = np.argmin(values)
            ylabel = f"{metric.upper()} (lower is better)"
        else:
            best_idx = np.argmax(values)
            ylabel = f"{metric.upper()} (higher is better)"

        best_layer = layers[best_idx]
        ax.axvline(
            best_layer, color="red", linestyle="--", alpha=0.5, label=f"Best: Layer {best_layer}"
        )
        ax.scatter([best_layer], [values[best_idx]], color="red", s=100, zorder=5)

        # Highlight middle layers (25%-75%)
        middle_start = num_layers // 4
        middle_end = 3 * num_layers // 4
        ax.axvspan(middle_start, middle_end, alpha=0.1, color="green",
                   label="Middle layers (25%-75%)")

        ax.set_xlabel("Layer Index", fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(f"{metric.upper()} by Layer", fontsize=12, fontweight="bold")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = output_dir / "layer_analysis.png"
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    print(f"Saved plot: {plot_path}")
    plt.close()

    # Create heatmap of all metrics
    fig, ax = plt.subplots(figsize=(12, 6))

    # Prepare data for heatmap
    metric_names = ["ECE", "Brier", "AUROC", "Accuracy"]
    data = np.array([
        [results[layer]["ece"] for layer in layers],
        [results[layer]["brier"] for layer in layers],
        [results[layer]["auroc"] for layer in layers],
        [results[layer]["accuracy"] for layer in layers],
    ])

    # Normalize each metric to [0, 1] for visualization
    data_normalized = np.zeros_like(data)
    for i in range(len(metric_names)):
        if metric_names[i] in ["ECE", "Brier"]:
            # Lower is better - invert for heatmap
            data_normalized[i] = 1 - (data[i] - data[i].min()) / (
                data[i].max() - data[i].min() + 1e-8
            )
        else:
            # Higher is better
            data_normalized[i] = (data[i] - data[i].min()) / (
                data[i].max() - data[i].min() + 1e-8
            )

    sns.heatmap(
        data_normalized,
        xticklabels=layers,
        yticklabels=metric_names,
        cmap="RdYlGn",
        center=0.5,
        annot=False,
        fmt=".2f",
        cbar_kws={"label": "Normalized Performance (green=better)"},
        ax=ax,
    )

    ax.set_xlabel("Layer Index", fontsize=12)
    ax.set_ylabel("Metric", fontsize=12)
    ax.set_title(
        f"Performance Heatmap Across Layers: {model_name}",
        fontsize=14,
        fontweight="bold",
        pad=20,
    )

    plt.tight_layout()
    heatmap_path = output_dir / "layer_heatmap.png"
    plt.savefig(heatmap_path, dpi=300, bbox_inches="tight")
    print(f"Saved heatmap: {heatmap_path}")
    plt.close()


def main():
    """Main experiment pipeline."""
    args = parse_args()

    # Setup
    setup_logging(log_level="INFO")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 70}")
    print("SYSTEMATIC LAYER ANALYSIS")
    print(f"{'=' * 70}")
    print(f"Model: {args.model}")
    print(f"Dataset: {args.dataset}")
    print(f"Samples: {args.num_samples}")
    print(f"Mode: {'Quick (quartiles only)' if args.quick else 'Full (all layers)'}")
    print(f"Output: {output_dir}")
    print(f"{'=' * 70}\n")

    # Load model
    print("Loading model...")
    model_loader = ModelLoader(model_name=args.model)
    quantization = None if args.quantization == "none" else args.quantization
    model, tokenizer = model_loader.load(quantization=quantization, device_map="auto")
    model_info = model_loader.get_model_info()
    num_layers = model_info["num_layers"]
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model loaded: {num_layers} layers, {num_params/1e9:.2f}B params")

    # Determine layers to analyze
    layers = get_layer_indices(num_layers, quick=args.quick)

    # Load dataset
    print(f"\nLoading {args.dataset} dataset...")
    if args.dataset == "mmlu":
        dataset = MMLUDataset(split="validation", category="stem")
    else:
        raise NotImplementedError(f"Dataset {args.dataset} not yet supported")

    print(f"Dataset loaded: {len(dataset)} examples")

    # Generate examples
    texts, labels = generate_dataset_examples(dataset, args.num_samples)
    print(f"Generated {len(texts)} examples (label dist: {np.bincount(labels)})")

    # Extract hidden states for all layers
    print(f"\nExtracting hidden states from {len(layers)} layers...")
    cache_dir = output_dir / "cache"
    extractor = HiddenStateExtractor(model, tokenizer)

    all_hiddens = {}
    for layer in tqdm(layers, desc="Extracting layers"):
        hiddens = extractor.extract(
            texts=texts,
            layers=[layer],
            max_length=512,
            token_position="last",
            cache_dir=str(cache_dir),
            use_cache=True,
            batch_size=16,
        )
        all_hiddens[layer] = hiddens[:, 0, :]  # Remove layer dimension

    # Split data
    print("\nSplitting data (70% train, 30% val)...")
    indices = np.arange(len(labels))
    train_indices, val_indices = train_test_split(
        indices, test_size=0.3, random_state=42, stratify=labels
    )
    print(f"Train: {len(train_indices)}, Val: {len(val_indices)}")

    # Train probes for each layer
    print(f"\nTraining probes for {len(layers)} layers...")
    config = {
        "dropout": 0.0,
        "lr": 1e-3,
        "weight_decay": 1e-5,
        "batch_size": 128,
        "epochs": 50,
        "patience": 5,
    }

    results = {}
    for layer in tqdm(layers, desc="Training probes"):
        X = all_hiddens[layer]
        X_train = X[train_indices]
        y_train = labels[train_indices]
        X_val = X[val_indices]
        y_val = labels[val_indices]

        layer_results = train_probe_for_layer(
            X_train, y_train, X_val, y_val, layer, config
        )
        results[layer] = layer_results

    # Analyze results
    print(f"\n{'=' * 70}")
    print("RESULTS")
    print(f"{'=' * 70}\n")
    print(f"{'Layer':<8} {'ECE':<10} {'Brier':<10} {'AUROC':<10} {'Accuracy':<10} {'Temp':<8}")
    print("-" * 70)

    for layer in layers:
        r = results[layer]
        print(
            f"{layer:<8} {r['ece']:<10.4f} {r['brier']:<10.4f} "
            f"{r['auroc']:<10.4f} {r['accuracy']:<10.4f} {r['temperature']:<8.2f}"
        )

    # Find best layers
    best_ece_layer = min(results.keys(), key=lambda k: results[k]["ece"])
    best_auroc_layer = max(results.keys(), key=lambda k: results[k]["auroc"])

    print(f"\n{'=' * 70}")
    print("ANALYSIS")
    print(f"{'=' * 70}")
    print(f"Best layer by ECE: {best_ece_layer} "
          f"(ECE={results[best_ece_layer]['ece']:.4f})")
    print(f"Best layer by AUROC: {best_auroc_layer} "
          f"(AUROC={results[best_auroc_layer]['auroc']:.4f})")

    # Check if best layer is in middle 50%
    middle_start = num_layers // 4
    middle_end = 3 * num_layers // 4
    in_middle = middle_start <= best_ece_layer <= middle_end
    print(f"\nMiddle layers hypothesis (25%-75%): {middle_start}-{middle_end}")
    print(f"Best ECE layer in middle range: {in_middle} ✓" if in_middle else
          f"Best ECE layer in middle range: {in_middle} ✗")

    # Generate plots
    print("\nGenerating visualizations...")
    plot_layer_analysis(results, num_layers, output_dir, args.model)

    # Save results
    results_file = output_dir / "results.txt"
    with open(results_file, "w") as f:
        f.write("Layer Analysis Results\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Model: {args.model}\n")
        f.write(f"Layers analyzed: {len(layers)}\n")
        f.write(f"Dataset: {args.dataset}\n")
        f.write(f"Samples: {len(texts)}\n\n")
        f.write(f"{'Layer':<8} {'ECE':<10} {'Brier':<10} {'AUROC':<10} {'Accuracy':<10}\n")
        f.write("-" * 70 + "\n")
        for layer in layers:
            r = results[layer]
            f.write(
                f"{layer:<8} {r['ece']:<10.4f} {r['brier']:<10.4f} "
                f"{r['auroc']:<10.4f} {r['accuracy']:<10.4f}\n"
            )
        f.write(f"\nBest layer by ECE: {best_ece_layer}\n")
        f.write(f"Best layer by AUROC: {best_auroc_layer}\n")

    print(f"\nResults saved to {results_file}")
    print(f"\nExperiment complete! All outputs in {output_dir}")


if __name__ == "__main__":
    main()
