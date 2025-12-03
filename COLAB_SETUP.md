# Running Experiments on Google Colab

This guide shows how to run experiments on Google Colab with free GPU access.

## Quick Start (Layer Analysis)

1. **Open the pre-made notebook**:
   - Upload `notebooks/colab_layer_analysis.ipynb` to Google Colab
   - Or create a new notebook and copy the cells

2. **Enable GPU**:
   - Runtime → Change runtime type → Hardware accelerator → **GPU (T4)**
   - Click Save

3. **Run the cells in order**:
   - Cell 1: Check GPU availability
   - Cell 2: Install dependencies
   - Cell 3: Clone repository
   - Cells 4-17: Run layer analysis

4. **Results**: Download the generated `layer_analysis_results.png`

## Running Hierarchical Probe on Colab

Create a new notebook with these cells:

### Cell 1: Setup GPU
```python
# Check GPU
!nvidia-smi
```

### Cell 2: Install Dependencies
```python
!pip install -q torch transformers accelerate bitsandbytes datasets scikit-learn tqdm loguru matplotlib seaborn pyyaml wandb
```

### Cell 3: Clone Repository
```python
!git clone https://github.com/joshcliu/deep-learning.git
%cd deep-learning
```

### Cell 4: Configure Experiment
```python
# Create minimal config for Colab
import yaml

config = {
    "experiment": {
        "name": "hierarchical-colab",
        "seed": 42,
        "output_dir": "outputs/hierarchical"
    },
    "model": {
        "name": "mistralai/Mistral-7B-v0.1",  # Use Mistral (no auth needed)
        "quantization": "8bit",
        "device": "cuda"
    },
    "data": {
        "dataset": "mmlu",
        "num_samples": 200,  # Small for testing
        "batch_size": 16,
        "max_length": 512,
        "split_ratio": [0.7, 0.15, 0.15]
    },
    "extraction": {
        "layers": None,  # Use optimal layers
        "token_position": "last",
        "use_cache": True,
        "cache_dir": "cache"
    },
    "probe": {
        "hidden_dim": 512,
        "num_layers": 3,
        "use_layer_fusion": True
    },
    "training": {
        "epochs": 30,
        "learning_rate": 1e-3,
        "weight_decay": 0.01,
        "early_stopping_patience": 5
    },
    "evaluation": {
        "num_bins": 10,
        "metrics": ["ece", "brier", "auroc", "accuracy"]
    },
    "logging": {
        "use_wandb": False  # Set True if you want WandB tracking
    }
}

# Save config
import os
os.makedirs("configs", exist_ok=True)
with open("configs/hierarchical_colab.yaml", "w") as f:
    yaml.dump(config, f)

print("Config saved!")
```

### Cell 5: Run Experiment
```python
# Run hierarchical probe experiment
!python experiments/hierarchical_probe.py \
    --config configs/hierarchical_colab.yaml \
    --debug  # Remove --debug for full run
```

### Cell 6: View Results
```python
import json
from pathlib import Path

# Load results
output_dir = Path("outputs/hierarchical")
if output_dir.exists():
    print("Experiment completed!")

    # Show metrics (if saved)
    metrics_file = output_dir / "metrics.json"
    if metrics_file.exists():
        with open(metrics_file) as f:
            metrics = json.load(f)
        print("\nResults:")
        print(json.dumps(metrics, indent=2))
else:
    print("No results found. Check experiment logs above.")
```

## Tips for Colab

### Memory Management
- **Free T4 GPU**: ~15GB VRAM
- **Recommended**: Use 8-bit quantization for 7B-8B models
- **For 70B models**: Use 4-bit quantization or smaller models

```python
# In config
config["model"]["quantization"] = "8bit"  # or "4bit"
```

### Reduce Dataset Size for Testing
```python
config["data"]["num_samples"] = 100  # Start small
config["training"]["epochs"] = 20    # Fewer epochs
```

### Save Results Before Session Ends
```python
# Download results
from google.colab import files

# Download plots
files.download("outputs/hierarchical/results.png")

# Download probe weights
files.download("outputs/hierarchical/hierarchical/probe.pt")
```

### Using Llama Models (Gated)
If you have access to Llama models:

```python
# Add before running experiment
from huggingface_hub import login
login(token="hf_your_token_here")

# Then update config
config["model"]["name"] = "meta-llama/Llama-3.1-8B"
```

## Experiment Comparison

| Experiment | Time (Colab T4) | Dataset Size | What It Does |
|------------|----------------|--------------|--------------|
| `layer_analysis.py --quick` | ~5-10 min | 100 samples | Tests which layers encode uncertainty best |
| `baseline_linear_probe.py` | ~1-2 hours | 1000 samples | Trains simple linear probe baseline |
| `hierarchical_probe.py --debug` | ~20-30 min | 200 samples | Tests hierarchical probe (quick) |
| `hierarchical_probe.py` | ~2-4 hours | Full dataset | Full hierarchical probe experiment |

## WandB Integration (Optional)

To track experiments with Weights & Biases:

```python
# Install wandb
!pip install wandb

# Login
import wandb
wandb.login()

# Update config
config["logging"]["use_wandb"] = True
config["logging"]["wandb_project"] = "llm-confidence"
```

## Common Issues

### Out of Memory
```python
# Reduce batch size
config["data"]["batch_size"] = 8  # Default is 16-32

# Use more aggressive quantization
config["model"]["quantization"] = "4bit"
```

### Session Timeout
Colab free tier sessions timeout after ~12 hours of inactivity. For long experiments:
- Use Colab Pro (24-hour sessions)
- Or run in multiple shorter sessions with caching enabled

### Cache Persistence
Cached hidden states are saved in `/content/deep-learning/cache/` but **won't persist** after session ends. To reuse:

```python
# Before session ends, zip and download cache
!zip -r cache.zip cache/
from google.colab import files
files.download("cache.zip")

# In next session, upload and unzip
!unzip cache.zip
```

## Full Workflow Example

```python
# 1. Setup
!nvidia-smi
!pip install -q torch transformers accelerate bitsandbytes datasets scikit-learn tqdm loguru pyyaml
!git clone https://github.com/joshcliu/deep-learning.git
%cd deep-learning

# 2. Quick test first
!python experiments/layer_analysis.py --quick --num-samples 100

# 3. Run hierarchical probe (debug mode)
!python experiments/hierarchical_probe.py --debug

# 4. If successful, run full experiment
!python experiments/hierarchical_probe.py

# 5. Download results
from google.colab import files
files.download("outputs/hierarchical/results.png")
```

## Next Steps

After running experiments on Colab:
1. Download results and plots
2. Analyze metrics (ECE, AUROC, accuracy)
3. Compare hierarchical vs baseline probes
4. Iterate on architecture or try different models

---

**Pro Tip**: Start with `--debug` mode (200 samples, 20 epochs) to validate the pipeline works before running full experiments!
