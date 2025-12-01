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

# Disable WandB
python experiments/baseline_linear_probe.py --no-wandb
```

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
# Quick mode (quartile layers only)
python experiments/layer_analysis.py --quick --num-samples 200

# Full analysis (all layers)
python experiments/layer_analysis.py --num-samples 500
```

---

### 3. Hierarchical Multi-Scale Probe (`hierarchical_probe.py`)

**Novel contribution**: Trains hierarchical probe that operates at four levels of granularity (token → span → semantic → global).

**What it does:**
- Extracts hidden states from multiple LLM layers
- Trains hierarchical probe with cross-layer attention fusion
- Compares against baseline linear/MLP probes
- Evaluates calibration at multiple scales
- Generates interpretable uncertainty hierarchies

**Usage:**
```bash
# Basic usage
python experiments/hierarchical_probe.py

# Debug mode (faster, 100 samples)
python experiments/hierarchical_probe.py --debug

# Custom config
python experiments/hierarchical_probe.py --config configs/hierarchical_probe.yaml

# Skip baseline comparisons
python experiments/hierarchical_probe.py --skip-baselines
```

**Architecture:**
```
Input: Hidden states from multiple LLM layers (e.g., layers 8, 16, 24, 31)
    ↓
[Layer Fusion] - Cross-attention over layers
    ↓
[Token-Level Probe] → Token confidences (per-word uncertainty)
    ↓
[Span-Level Probe] → Span confidence (phrase-level aggregation)
    ↓
[Semantic-Level Probe] → Semantic confidence (meaning coherence)
    ↓
[Global Aggregation] → Final confidence score
```

**Configuration:** See `configs/hierarchical_probe.yaml`

**Expected improvements over linear baseline:**
- 40-50% ECE reduction
- Better AUROC for hallucination detection
- Interpretable multi-scale uncertainty

---

## Quick Start Guide

### First Time Setup

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run quick test:**
   ```bash
   # Verify everything works
   python test_integration.py

   # Quick layer analysis
   python experiments/layer_analysis.py --quick --num-samples 100
   ```

### Running Full Experiments

**Baseline experiment:**
```bash
python experiments/baseline_linear_probe.py
```

**Layer analysis:**
```bash
python experiments/layer_analysis.py --num-samples 500
```

**Hierarchical probe:**
```bash
# Debug mode first (fast)
python experiments/hierarchical_probe.py --debug

# Full run when ready
python experiments/hierarchical_probe.py
```

---

## Expected Results

### Layer Analysis

Middle layers (50-75% depth) typically perform best:
- Best layer for Llama 3.1 8B: ~16-24
- ECE: 0.05-0.10
- AUROC: 0.75-0.85

### Baseline Linear Probe

- **ECE**: < 0.10 reasonable, < 0.05 excellent
- **AUROC**: > 0.75 good for hallucination detection

### Hierarchical Probe

| Method | ECE | Brier | AUROC | Parameters |
|--------|-----|-------|-------|------------|
| Linear | ~0.080 | ~0.160 | ~0.810 | ~4K |
| **Hierarchical** | **~0.045** | **~0.120** | **~0.865** | **~10M** |

Target: **40-50% ECE improvement** over linear baseline

---

## Tips & Best Practices

### Memory Management

1. Use quantization for large models (8-bit for 7B-13B models)
2. Start with `--num-samples 200` for testing
3. Enable caching (default: ON) for faster subsequent runs

### Debugging

If experiments fail:
```bash
# Test with minimal config
python experiments/layer_analysis.py --quick --num-samples 50

# Check imports work
python test_integration.py
```

---

## Next Steps

After running baseline experiments:

1. **Analyze hierarchical results**: Use `notebooks/hierarchical_analysis.ipynb`
2. **Implement CCPS** (advanced calibration method)
3. **Cross-model comparison** (Llama vs Mistral vs Qwen)
4. **Write paper**: Document findings for NeurIPS/ICLR

See `CLAUDE.md` for detailed research roadmap.
