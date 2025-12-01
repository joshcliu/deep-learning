# Experiments

This directory contains experiment scripts for the LLM Confidence Probing Framework.

## Available Experiments

### 1. Baseline Linear Probe (`baseline_linear_probe.py`)

Trains linear probes on hidden states from a language model to predict confidence/correctness.

**What it does:**
- Loads MMLU dataset (configurable)
- Extracts hidden states from specified layers
- Trains a linear probe per layer
- Evaluates calibration metrics (ECE, Brier, AUROC, Accuracy)
- Compares performance across layers
- Logs results to WandB (optional)

**Usage:**
```bash
# Basic usage (uses default config)
python experiments/baseline_linear_probe.py

# Custom config
python experiments/baseline_linear_probe.py --config configs/custom.yaml

# Override specific settings
python experiments/baseline_linear_probe.py --layers 8 16 24 --num-samples 500

# Disable WandB
python experiments/baseline_linear_probe.py --no-wandb
```

**Arguments:**
- `--config`: Path to config file (default: `configs/linear_probe.yaml`)
- `--layers`: Layer indices to probe (overrides config)
- `--num-samples`: Limit number of samples for quick testing
- `--no-wandb`: Disable WandB logging

**Output:**
- Trained probe checkpoints in `outputs/<experiment-name>/checkpoints/`
- Training logs and metrics
- WandB dashboard (if enabled)

**Example config:** See `configs/linear_probe.yaml`

---

### 2. Layer Analysis (`layer_analysis.py`)

Systematic analysis of probe performance across ALL layers to identify optimal layers for uncertainty quantification.

**What it does:**
- Tests probes on all layers (or quartile layers in quick mode)
- Generates performance visualizations
- Validates "middle layers optimal" hypothesis
- Creates heatmaps and line plots

**Usage:**
```bash
# Quick mode (quartile layers only, good for testing)
python experiments/layer_analysis.py --quick --num-samples 200

# Full analysis (all layers)
python experiments/layer_analysis.py --num-samples 500

# Custom model
python experiments/layer_analysis.py --model meta-llama/Llama-2-7b-hf

# With quantization
python experiments/layer_analysis.py --quantization 8bit
```

**Arguments:**
- `--model`: Model name or path (default: `meta-llama/Llama-3.1-8B`)
- `--dataset`: Dataset to use (`mmlu`, `triviaqa`, `gsm8k`)
- `--num-samples`: Number of dataset samples to use
- `--quick`: Quick mode - test quartile layers only (0%, 25%, 50%, 75%, 100%)
- `--quantization`: Model quantization (`4bit`, `8bit`, `none`)
- `--output-dir`: Output directory (default: `outputs/layer_analysis`)

**Output:**
- `layer_analysis.png`: Line plots of metrics by layer
- `layer_heatmap.png`: Heatmap visualization of all metrics
- `results.txt`: Numerical results table
- Hidden state cache in `<output-dir>/cache/`

---

## Quick Start Guide

### First Time Setup

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure experiment:**
   Edit `configs/linear_probe.yaml` to set:
   - Model name (ensure you have access/token if gated)
   - Quantization settings (based on your GPU memory)
   - Dataset and layer preferences

3. **Run quick test:**
   ```bash
   # Test with small sample (fast, good for debugging)
   python experiments/layer_analysis.py --quick --num-samples 100
   ```

### Running Full Experiments

**Baseline experiment:**
```bash
# Default: Llama 3.1 8B on MMLU, layers [8, 16, 24, 31]
python experiments/baseline_linear_probe.py

# With WandB logging
python experiments/baseline_linear_probe.py --config configs/linear_probe.yaml
```

**Layer analysis:**
```bash
# Full analysis on Llama 3.1 8B (32 layers)
python experiments/layer_analysis.py --num-samples 500

# Takes ~2-4 hours depending on GPU
# Generates visualizations and saves results
```

**Testing different models:**
```bash
# Llama 2 7B
python experiments/layer_analysis.py --model meta-llama/Llama-2-7b-hf --quick

# Mistral 7B
python experiments/layer_analysis.py --model mistralai/Mistral-7B-v0.1 --quick

# Qwen 2.5 7B
python experiments/layer_analysis.py --model Qwen/Qwen2.5-7B --quick
```

---

## Configuration

### Config File Structure (`configs/linear_probe.yaml`)

```yaml
experiment:
  name: "experiment-name"
  seed: 42
  output_dir: "outputs"

model:
  name: "meta-llama/Llama-3.1-8B"
  quantization: "8bit"  # 4bit, 8bit, or null
  device: "auto"

data:
  dataset: "mmlu"
  split_ratio: [0.7, 0.15, 0.15]  # train/val/test
  batch_size: 32
  max_length: 512
  num_samples: null  # null = use all

extraction:
  layers: [8, 16, 24, 31]  # null = use optimal from registry
  token_position: "last"
  use_cache: true

probe:
  dropout: 0.0

training:
  epochs: 50
  learning_rate: 1e-3
  weight_decay: 1e-5
  early_stopping_patience: 5

evaluation:
  metrics: ["ece", "brier", "auroc", "aupr", "accuracy"]
  num_bins: 10

logging:
  use_wandb: false
  wandb_project: "llm-confidence"
```

---

## Expected Results

### Layer Analysis

Based on research literature, you should observe:

1. **Middle layers (50-75% depth) perform best** for uncertainty quantification
2. **Early layers (0-25%)**: Poor performance (still learning basic features)
3. **Middle layers (25-75%)**: Best performance (semantic understanding)
4. **Final layers (75-100%)**: Moderate performance (task-specific specialization)

**Example for Llama 3.1 8B (32 layers):**
- Best layer: ~16-24 (middle layers)
- ECE: 0.05-0.10 (well-calibrated if < 0.05)
- AUROC: 0.75-0.85 (good discrimination)

### Baseline Linear Probe

**Expected metrics:**
- **ECE (Expected Calibration Error)**: < 0.10 is reasonable, < 0.05 is excellent
- **Brier Score**: Lower is better, ~0.15-0.25 typical
- **AUROC**: > 0.75 is good for hallucination detection
- **Accuracy**: Depends on model quality, 60-80% typical

---

## Tips & Best Practices

### Memory Management

1. **Use quantization for large models:**
   - 8-bit for 7B-13B models: ~8-16GB VRAM
   - 4-bit for 70B models: ~40GB VRAM

2. **Start with `--num-samples 200` for testing**
   - Quick iteration during development
   - Full run when ready (500-1000 samples recommended)

3. **Enable caching** (default: ON)
   - First run: slow (extracts hidden states)
   - Subsequent runs: fast (loads from cache)
   - Cache location: `<output-dir>/cache/`

### Debugging

If experiments fail:

1. **Test with minimal config:**
   ```bash
   python experiments/layer_analysis.py --quick --num-samples 50
   ```

2. **Check GPU memory:**
   ```python
   import torch
   print(torch.cuda.memory_allocated() / 1e9)  # GB
   ```

3. **Reduce batch size** in config if OOM errors

4. **Verify model access:**
   - Gated models (Llama) need HuggingFace token
   - Set `use_auth_token` in config

### Reproducibility

- Set `seed` in config for deterministic results
- Cache ensures same hidden states across runs
- Log all hyperparameters to WandB/config files

---

## Adding New Experiments

To create a new experiment:

1. **Copy a template:**
   ```bash
   cp experiments/baseline_linear_probe.py experiments/my_experiment.py
   ```

2. **Follow the pattern:**
   ```python
   from src.utils import load_config, ExperimentLogger, setup_logging
   from src.models import ModelLoader, HiddenStateExtractor
   from src.data import MMLUDataset
   from src.probes import LinearProbe
   from src.evaluation import CalibrationMetrics

   def main():
       setup_logging(log_level="INFO")
       config = load_config("configs/my_config.yaml")
       logger = ExperimentLogger(config.experiment.name, config=config)

       # Your experiment logic

       logger.finish()
   ```

3. **Document in this README**

---

## Next Steps

After running baseline experiments:

1. **Implement CCPS** (Conformal Calibration via Perturbation Sampling)
2. **Implement Semantic Entropy**
3. **Cross-model comparison** experiments
4. **Cross-dataset evaluation**

See `CLAUDE.md` for detailed research roadmap.
