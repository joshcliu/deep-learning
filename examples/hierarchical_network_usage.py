"""Example: Using the Hierarchical Network Architecture

This demonstrates how to use the hierarchical multi-scale network
with CalibratedProbe for uncertainty quantification.

The hierarchical network processes hidden states at multiple levels:
1. Fine-grained: Individual chunks
2. Mid-level: Aggregated chunks with attention
3. Semantic: Broader context processing
4. Global: Final confidence prediction
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
from src.probes import CalibratedProbe
from src.probes.architectures import build_hierarchical_network

# ============================================================================
# Example 1: Basic Usage
# ============================================================================

print("="*60)
print("EXAMPLE 1: Basic Hierarchical Network Usage")
print("="*60)

# Build the network
network = build_hierarchical_network(
    input_dim=4096,      # Llama hidden dim
    num_chunks=16,       # Split hidden state into 16 chunks
    hidden_dim=256,      # Hidden layer width
    dropout=0.1          # Dropout probability
)

# Create probe with the network
probe = CalibratedProbe(network=network)

# Train (assuming you have X_train, y_train, X_val, y_val)
# probe.fit(X_train, y_train, X_val, y_val, num_epochs=50)

print("✓ Hierarchical network created")
print(f"  Input: (batch, 4096)")
print(f"  Output: (batch, 1) with sigmoid")
print(f"  Chunks: {16}")
print(f"  Hidden dim: {256}")


# ============================================================================
# Example 2: Compare with Other Architectures
# ============================================================================

print("\n" + "="*60)
print("EXAMPLE 2: Architecture Comparison")
print("="*60)

from src.probes.architectures import (
    build_attention_network,
    build_residual_network,
    build_gated_network,
)

architectures = {
    "Linear (default)": None,  # CalibratedProbe builds default MLP
    "Attention": build_attention_network(input_dim=4096, num_chunks=16),
    "Residual": build_residual_network(input_dim=4096, hidden_dim=512),
    "Gated": build_gated_network(input_dim=4096, hidden_dim=512),
    "Hierarchical": build_hierarchical_network(input_dim=4096, num_chunks=16),
}

print("\nAvailable architectures:")
for name, net in architectures.items():
    if net is None:
        print(f"  • {name}")
    else:
        num_params = sum(p.numel() for p in net.parameters())
        print(f"  • {name:15s} ({num_params:,} parameters)")


# ============================================================================
# Example 3: Full Pipeline with Real Data
# ============================================================================

print("\n" + "="*60)
print("EXAMPLE 3: Full Pipeline (Commented - Uncomment to Run)")
print("="*60)

print("""
# Step 1: Load model and extract hidden states
loader = ModelLoader("meta-llama/Llama-3.1-8B")
model, tokenizer = loader.load(quantization="8bit")

extractor = HiddenStateExtractor(model, tokenizer)
dataset = MMLUDataset(split="test", max_samples=500)

# Get prompts and generate predictions
prompts = dataset.get_prompts_for_generation()
# ... (generate predictions, extract hiddens)

# Step 2: Build hierarchical probe
network = build_hierarchical_network(input_dim=4096, num_chunks=16)
probe = CalibratedProbe(network=network)

# Step 3: Train
history = probe.fit(
    X_train, y_train,
    X_val, y_val,
    num_epochs=50,
    batch_size=32,
    patience=10
)

# Step 4: Evaluate
confidences = probe.predict(X_test)
predictions = (confidences > 0.5).astype(int)

metrics = CalibrationMetrics(predictions, confidences, y_test)
print(f"ECE: {metrics.ece():.4f}")
print(f"AUROC: {metrics.auroc():.4f}")
print(f"Accuracy: {metrics.accuracy():.2%}")
""")


# ============================================================================
# Example 4: Using in Colab
# ============================================================================

print("\n" + "="*60)
print("EXAMPLE 4: Colab Usage")
print("="*60)

print("""
In Google Colab:

```python
# After cloning the repo and installing dependencies
from src.probes import CalibratedProbe
from src.probes.architectures import build_hierarchical_network

# Build network
network = build_hierarchical_network(input_dim=4096)

# Create and train probe
probe = CalibratedProbe(network=network)
probe.fit(X_train, y_train, X_val, y_val, num_epochs=50)

# Evaluate
confidences = probe.predict(X_test)
```
""")


# ============================================================================
# Example 5: Hyperparameter Variations
# ============================================================================

print("\n" + "="*60)
print("EXAMPLE 5: Hyperparameter Variations")
print("="*60)

variations = [
    {"num_chunks": 8, "hidden_dim": 128, "description": "Lightweight"},
    {"num_chunks": 16, "hidden_dim": 256, "description": "Default"},
    {"num_chunks": 32, "hidden_dim": 512, "description": "Heavy"},
]

print("\nHierarchical network variations:")
for var in variations:
    network = build_hierarchical_network(
        input_dim=4096,
        num_chunks=var["num_chunks"],
        hidden_dim=var["hidden_dim"]
    )
    num_params = sum(p.numel() for p in network.parameters())
    print(f"  • {var['description']:12s}: "
          f"chunks={var['num_chunks']:2d}, "
          f"hidden={var['hidden_dim']:3d}, "
          f"params={num_params:,}")

print("\n✅ Examples complete!")
print("\nFor more examples, see:")
print("  - notebooks/colab_architecture_comparison.ipynb")
print("  - experiments/hierarchical_probe.py")
