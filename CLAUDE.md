# Claude Code Agent Guide

This document provides context for Claude Code agents working on this project. It describes what has been implemented, architectural decisions, and how to continue development effectively.

## Project Overview

**Name**: LLM Confidence Probing Framework
**Goal**: Research framework for extracting and quantifying uncertainty in large language models
**Research Focus**: Probing hidden states, calibration metrics, mechanistic interpretability
**Target**: MIT deep learning project with publication potential (NeurIPS, ICLR, ACL/EMNLP)

## Current Status: Phase 1 Complete ✅ | Phase 2 In Progress

### What Has Been Implemented

#### 1. Project Infrastructure ✅
- Complete directory structure (`src/`, `configs/`, `experiments/`, `notebooks/`, etc.)
- Dependency management (`requirements.txt`, `pyproject.toml`)
- Git configuration (`.gitignore` with project-specific entries)
- Documentation (`README.md`, `IMPLEMENTATION.md`, `RESEARCH.md`)

#### 2. Model Management Module ✅ (`src/models/`)

**Files**:
- `registry.py`: Model configurations for 9 LLMs (Llama 3.1, Llama 2, Mistral, Mixtral, Qwen 2.5)
- `loader.py`: Unified model loading with quantization support + Tinker LoRA
- `extractor.py`: Hidden state extraction with caching (bfloat16 compatible)

**Key Features**:
- Automatic quantization (4-bit/8-bit via bitsandbytes)
- Layer-specific extraction (single, multiple, quartile positions)
- Token position handling (last, CLS, mean)
- Memory-mapped caching for large datasets
- Batch processing with progress bars
- **bfloat16 → float32 conversion** for numpy compatibility

**API Notes**:
```python
# ModelLoader - quantization goes in load(), not __init__()
loader = ModelLoader(model_name="mistralai/Mistral-7B-v0.1")
model, tokenizer = loader.load(quantization="8bit", device_map="auto")

# HiddenStateExtractor - cache_dir goes in extract(), not __init__()
extractor = HiddenStateExtractor(model, tokenizer)
hiddens = extractor.extract(texts, layers=[16], cache_dir="cache/")
```

#### 3. Evaluation Framework ✅ (`src/evaluation/`)

**Files**:
- `metrics.py`: Core calibration metrics (ECE, Brier, AUROC, AUPR, accuracy)
- `calibration.py`: Visualization and post-hoc calibration methods
- `selective.py`: Selective prediction analysis

**Key Features**:
- Expected Calibration Error (ECE) with binning statistics
- ROC/PR curve generation
- Reliability diagrams (matplotlib + seaborn)
- Temperature scaling and Platt scaling calibrators

#### 4. Dataset Loaders ✅ (`src/data/`)

**Files**:
- `base.py`: BaseDataset abstract class and DatasetExample dataclass
- `mmlu.py`: MMLU with 57 subjects, category filtering, 4-choice questions
- `triviaqa.py`: TriviaQA open-domain QA with fuzzy answer matching
- `gsm8k.py`: GSM8K math problems with numerical answer extraction

**Usage**:
```python
from src.data import MMLUDataset
dataset = MMLUDataset(split="test")
example = dataset[0]  # Returns DatasetExample with .question, .choices, .answer
```

#### 5. Probe Implementations ✅ (`src/probes/`)

**Files**:
- `base.py`: BaseProbe abstract class
- `linear.py`: LinearProbe with BCE loss and temperature scaling
- `calibrated_probe.py`: CalibratedProbe with Brier score loss ⭐ NEW
- `hierarchical.py`: HierarchicalProbe for multi-scale analysis

**LinearProbe** - Standard BCE loss:
```python
probe = LinearProbe(input_dim=4096, dropout=0.1)
history = probe.fit(X_train, y_train, X_val, y_val,
                    num_epochs=50, patience=5)  # Note: num_epochs, not epochs
```

**CalibratedProbe** - Brier score loss (recommended for calibration):
```python
from src.probes import CalibratedProbe

probe = CalibratedProbe(
    input_dim=4096,
    hidden_dim=256,  # None for linear, int for MLP
    dropout=0.1,
)
probe.fit(X_train, y_train, X_val, y_val)
confidences = probe.predict(X_test)
```

**Why Brier Score?** The loss is `(confidence - correct)²`:
- High confidence + wrong = **heavily penalized** (the dangerous case)
- Low confidence + wrong = acceptable (model knows it's uncertain)
- High confidence + correct = good
- Low confidence + correct = room for improvement

#### 6. Experiment Scripts ✅ (`experiments/`)

**Files**:
- `baseline_linear_probe.py`: End-to-end training pipeline
- `layer_analysis.py`: Systematic layer-wise performance analysis

**Default model**: `mistralai/Mistral-7B-v0.1` (ungated, no HuggingFace approval needed)

#### 7. Colab Notebooks ✅ (`notebooks/`)

**Files**:
- `colab_layer_analysis.ipynb`: Basic layer analysis with LinearProbe
- `colab_calibrated_probe.ipynb`: Model self-knowledge experiment ⭐ NEW

**Colab Setup Notes**:
- Requires bfloat16 patch for 8-bit quantized models
- Use `Runtime > Change runtime type > GPU (T4)`
- Clone repo fresh to get latest fixes

**bfloat16 Patch** (apply in Colab before extraction):
```python
import src.models.extractor as extractor_module
import numpy as np
import torch

def patched_extract_batch(self, texts, layers, max_length, token_position):
    # ... standard extraction code ...
    # Key fix: convert via tolist() for bfloat16 compatibility
    batch_hiddens.append(np.array(token_hiddens.detach().cpu().tolist(), dtype=np.float32))
    return np.stack(batch_hiddens, axis=1)

extractor_module.HiddenStateExtractor._extract_batch = patched_extract_batch
```

#### 8. Tinker API Integration ✅ (`src/tinker/`)

**Files**:
- `client.py`: Tinker API client wrapper with authentication
- `weights.py`: Download and convert Tinker LoRA weights to PEFT format

**Note**: Tinker is for **training only**, not inference. Hidden state extraction requires loading models locally via HuggingFace.

### Phase 2: In Progress

#### Completed:
- [x] CalibratedProbe with Brier score loss
- [x] Model-generated answer evaluation framework
- [x] Colab notebooks for GPU-accelerated experiments

#### Remaining:
- [ ] CCPS implementation (perturbation-based calibration)
- [ ] Semantic entropy (clustering + entropy over meanings)
- [ ] Cross-model comparison experiments
- [ ] Cross-dataset evaluation

### Phase 3: Planned
- [ ] Hierarchical multi-scale probing (novel contribution)
- [ ] Uncertainty-aware sparse autoencoders (SAE)
- [ ] Causal circuit tracing
- [ ] VLM extensions

## Experimental Setup: Model Self-Knowledge

The improved experimental setup tests whether the model "knows" when it's wrong:

### Pipeline
```
┌──────────────┐     ┌─────────────┐     ┌──────────────┐
│ MMLU Question│────▶│ Model       │────▶│ Generated    │
│              │     │ Generates   │     │ Answer       │
└──────────────┘     └─────────────┘     └──────┬───────┘
                                                │
        ┌───────────────────────────────────────┘
        ▼
┌──────────────┐     ┌─────────────┐     ┌──────────────┐
│ Check if     │────▶│ Label: 0/1  │────▶│ Train Probe  │
│ Answer Right │     │ (wrong/right)│    │ Brier Loss   │
└──────────────┘     └─────────────┘     └──────────────┘
```

### Key Insight
Instead of training on pre-defined correct/incorrect pairs, we:
1. Have the model **generate** its own answer
2. Check if that answer is correct (binary label)
3. Train probe to predict the model's confidence in its **own** answer
4. Use Brier loss to properly penalize overconfident wrong predictions

## Important API Notes & Gotchas

### Parameter Names (Common Mistakes)

```python
# LinearProbe.fit() parameters:
probe.fit(
    X_train, y_train, X_val, y_val,
    num_epochs=50,    # NOT 'epochs'
    patience=5,       # NOT 'early_stopping_patience'
    batch_size=32,
    lr=1e-3,
    verbose=True,
)
```

### ModelLoader API

```python
# CORRECT:
loader = ModelLoader(model_name="mistralai/Mistral-7B-v0.1")
model, tokenizer = loader.load(quantization="8bit", device_map="auto")

# WRONG - these params don't go in __init__():
# loader = ModelLoader(model_name="...", quantization="8bit")  # ERROR
```

### HiddenStateExtractor API

```python
# CORRECT:
extractor = HiddenStateExtractor(model, tokenizer)
hiddens = extractor.extract(texts, layers=[16], cache_dir="cache/")

# WRONG - cache_dir doesn't go in __init__():
# extractor = HiddenStateExtractor(model, tokenizer, cache_dir="cache/")  # ERROR
```

### bfloat16 Compatibility

When using 8-bit quantization, hidden states are in bfloat16 which numpy doesn't support. The fix in `extractor.py`:

```python
# Convert via tolist() to handle bfloat16
batch_hiddens.append(np.array(token_hiddens.detach().cpu().tolist(), dtype=np.float32))
```

### HuggingFace Authentication

Llama models are gated and require approval:
1. Request access at https://huggingface.co/meta-llama/Llama-3.1-8B
2. Login: `huggingface-cli login`

**Alternative**: Use ungated models like `mistralai/Mistral-7B-v0.1` or `Qwen/Qwen2.5-7B`

## Key Research Context

### Critical Findings (from literature)
1. **Middle layers (50-75% depth) outperform final layers** for uncertainty
2. **CCPS achieves 55% ECE reduction** - state-of-the-art as of 2025
3. **Well-calibrated systems: ECE < 0.05** - baseline LLMs often 0.10-0.15
4. **AUROC > 0.75-0.85** for hallucination detection is achievable

### Model Support

**Tier 1** (default, ungated):
- Mistral 7B v0.1 (primary for experiments)

**Tier 2** (requires HuggingFace approval):
- Llama 3.1 8B
- Llama 2 7B

**Tier 3** (supported):
- Qwen 2.5 7B/14B/72B

## Development Workflow

### Running Experiments

**Local (CPU - slow)**:
```bash
python experiments/layer_analysis.py --quick --num-samples 100
```

**Colab (GPU - recommended)**:
1. Push changes to GitHub
2. Open notebook in Colab
3. Set Runtime > GPU
4. Run all cells

### Adding a New Probe

1. Create `src/probes/my_probe.py` extending `BaseProbe`
2. Implement `forward()`, `fit()`, `predict()` methods
3. Export in `src/probes/__init__.py`
4. Create Colab notebook to test

## File Structure

```
deep-learning/
├── src/
│   ├── models/
│   │   ├── loader.py      # ModelLoader
│   │   ├── extractor.py   # HiddenStateExtractor
│   │   └── registry.py    # Model configs
│   ├── probes/
│   │   ├── base.py        # BaseProbe ABC
│   │   ├── linear.py      # LinearProbe (BCE)
│   │   ├── calibrated_probe.py  # CalibratedProbe (Brier) ⭐
│   │   └── hierarchical.py
│   ├── data/
│   │   ├── mmlu.py        # MMLUDataset
│   │   ├── triviaqa.py    # TriviaQADataset
│   │   └── gsm8k.py       # GSM8KDataset
│   ├── evaluation/
│   │   └── metrics.py     # compute_ece, compute_auroc, etc.
│   └── tinker/            # Tinker API integration
├── experiments/
│   ├── layer_analysis.py
│   └── baseline_linear_probe.py
├── notebooks/
│   ├── colab_layer_analysis.ipynb
│   └── colab_calibrated_probe.ipynb  ⭐
└── configs/
```

## Next Steps for Development

### Immediate
1. Run `colab_calibrated_probe.ipynb` with more samples (500+)
2. Compare probe performance across layers
3. Validate middle-layer hypothesis with Brier-trained probes

### Short-term
1. Cross-model comparison (Mistral vs Qwen)
2. Cross-dataset evaluation (MMLU vs TriviaQA vs GSM8K)
3. Implement CCPS for further calibration improvement

### Research Contributions
1. **Calibrated probes**: Show Brier loss improves probe calibration
2. **Layer analysis**: Confirm optimal layers for uncertainty
3. **Model self-knowledge**: Quantify when models know they're wrong

---
**Last Updated**: 2025-12-03
**Phase**: Phase 2 In Progress
**Primary Contact**: User (joshc)
