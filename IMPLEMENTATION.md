# Implementation Plan: LLM Confidence Probing Framework

**Last Updated**: 2025-11-30
**Status**: Phase 1 Complete ✅ | Phase 2 Ready

## Recent Updates (2025-11-30)

### Phase 1 Completion

All baseline infrastructure is now implemented and ready for use:

- ✅ **Created `src/probes/base.py`**: BaseProbe abstract class for consistent probe interface
- ✅ **Updated `src/probes/__init__.py`**: Proper module exports for LinearProbe and BaseProbe
- ✅ **Created `experiments/baseline_linear_probe.py`**: End-to-end baseline experiment with multi-layer training
- ✅ **Created `experiments/layer_analysis.py`**: Systematic layer performance analysis with visualizations
- ✅ **Created `experiments/README.md`**: Comprehensive documentation for running experiments
- ✅ **Verified all existing modules**: Data loaders, model management, evaluation metrics all functional

**Ready to run**:
```bash
# Quick test
python experiments/layer_analysis.py --quick --num-samples 50

# Full experiments
python experiments/baseline_linear_probe.py
python experiments/layer_analysis.py --num-samples 500
```

## Project Overview

This codebase implements a comprehensive framework for probing large language models (LLMs) to extract and quantify uncertainty/confidence. The implementation follows the research roadmap detailed in `RESEARCH.md` and provides modular, extensible infrastructure for rapid experimentation.

## Architecture

### Core Principles
- **Modularity**: Each component (models, probes, evaluation) is independently testable
- **Efficiency**: Memory-mapped caching, gradient checkpointing, mixed precision
- **Reproducibility**: Config-driven experiments with full logging
- **Extensibility**: Easy to add new models, probes, datasets, and metrics

### Directory Structure

```
deep-learning/
├── src/
│   ├── models/          # Model loading and hidden state extraction
│   ├── data/            # Dataset loaders and preprocessing
│   ├── probes/          # Probe implementations (linear, CCPS, hierarchical)
│   ├── evaluation/      # Metrics and evaluation utilities
│   ├── interpretability/# SAE, causal tracing, circuit analysis
│   └── utils/           # Shared utilities (caching, logging)
├── experiments/         # Experiment scripts
├── configs/             # YAML configuration files
├── notebooks/           # Jupyter notebooks for exploration
├── tests/               # Unit and integration tests
├── scripts/             # CLI utilities (extract_hiddens, train_probe)
├── data/                # Raw datasets (gitignored, downloaded at runtime)
├── cache/               # Cached hidden states and embeddings
├── outputs/             # Experiment outputs, checkpoints, logs
├── requirements.txt     # Python dependencies
├── pyproject.toml       # Project metadata and build config
└── README.md            # Getting started guide
```

## Module Specifications

### 1. Model Management (`src/models/`)

**Files**:
- `loader.py`: Unified model loading interface
- `extractor.py`: Hidden state extraction with caching
- `registry.py`: Supported model configurations

**Key Features**:
- Support for Llama 3.1, Mistral, Qwen 2.5 families
- Automatic quantization (4-bit/8-bit) for memory efficiency
- Layer-specific extraction (single layer, quartiles, all layers)
- Token position handling (last token, [CLS], averaged)
- Efficient batching with progress tracking

**API Example**:
```python
from src.models import ModelLoader, HiddenStateExtractor

loader = ModelLoader("meta-llama/Llama-3.1-8B")
model, tokenizer = loader.load(quantization="8bit")

extractor = HiddenStateExtractor(model, tokenizer)
hiddens = extractor.extract(
    texts=["What is the capital of France?"],
    layers=[8, 16, 24, 31],  # Quartile positions
    cache_dir="cache/llama-3.1-8b"
)
```

### 2. Data Pipeline (`src/data/`)

**Files**:
- `datasets.py`: Dataset loaders (MMLU, TriviaQA, GSM8K, TruthfulQA)
- `preprocessing.py`: Tokenization, formatting, label generation
- `samplers.py`: Balanced sampling, stratification

**Key Features**:
- Automatic downloading from HuggingFace datasets
- Consistent interface across benchmarks
- Train/val/test splits with configurable ratios
- Class imbalance handling (weighted sampling)
- Lazy loading for large datasets

**API Example**:
```python
from src.data import MMLUDataset

dataset = MMLUDataset(split="validation", subjects=["abstract_algebra", "anatomy"])
for item in dataset:
    print(item["question"], item["choices"], item["answer"])
```

### 3. Probe Implementations (`src/probes/`)

**Files**:
- `base.py`: BaseProbe abstract class
- `linear.py`: Linear probe with temperature scaling
- `ccps.py`: CCPS (perturbation-based) implementation
- `semantic.py`: Semantic entropy clustering
- `hierarchical.py`: Multi-scale hierarchical probe

**Key Features**:
- Unified training interface (fit/predict/evaluate)
- Automatic checkpointing and resumption
- Hyperparameter sweeps with Optuna
- Post-hoc calibration (temperature scaling, Platt scaling)

**API Example**:
```python
from src.probes import LinearProbe

probe = LinearProbe(input_dim=4096, hidden_dim=512)
probe.fit(hiddens_train, labels_train, hiddens_val, labels_val)
predictions = probe.predict(hiddens_test)
calibrated = probe.calibrate(hiddens_val, labels_val, method="temperature")
```

### 4. Evaluation Framework (`src/evaluation/`)

**Files**:
- `metrics.py`: ECE, Brier, AUROC, AUPR implementations
- `calibration.py`: Reliability diagrams, calibration curves
- `selective.py`: Prediction-rejection analysis, coverage plots
- `conformal.py`: Conformal prediction utilities

**Key Features**:
- Comprehensive metric suite with confidence intervals
- Automatic plot generation (reliability diagrams, ROC curves)
- Benchmark runner with multi-metric tracking
- Statistical significance testing

**API Example**:
```python
from src.evaluation import CalibrationMetrics, plot_reliability_diagram

metrics = CalibrationMetrics(predictions, confidences, labels)
print(f"ECE: {metrics.ece():.4f}")
print(f"Brier: {metrics.brier():.4f}")
print(f"AUROC: {metrics.auroc():.4f}")

plot_reliability_diagram(confidences, labels, save_path="outputs/reliability.png")
```

### 5. Interpretability Modules (`src/interpretability/`)

**Files**:
- `sae.py`: Sparse autoencoder training
- `circuits.py`: Activation patching, causal tracing
- `attribution.py`: Feature importance, attention analysis

**Key Features** (Phase 2):
- Standard SAE with L1 sparsity
- Uncertainty-aware SAE variant
- Circuit identification via ablation
- Interactive visualization exports

### 6. Utilities (`src/utils/`)

**Files**:
- `caching.py`: Memory-mapped array utilities, HDF5 helpers
- `logging.py`: Experiment logging, WandB integration
- `config.py`: YAML config loading and validation
- `device.py`: GPU/CPU management, memory monitoring

## Development Workflow

### Phase 1: Foundation ✅ COMPLETE (Updated: 2025-11-30)

**Status**: All baseline components implemented and tested

**Completed Components**:

1. ✅ **Model Management** (`src/models/`)
   - `loader.py` - Unified model loading with quantization
   - `extractor.py` - Hidden state extraction with caching
   - `registry.py` - 9 LLM configurations (Llama, Mistral, Qwen)

2. ✅ **Data Pipeline** (`src/data/`)
   - `base.py` - BaseDataset abstract class
   - `mmlu.py` - MMLU with 57 subjects, category filtering
   - `triviaqa.py` - TriviaQA open-domain QA
   - `gsm8k.py` - GSM8K math problems
   - `DATA_README.md` - Comprehensive documentation

3. ✅ **Probe Implementations** (`src/probes/`)
   - `base.py` - BaseProbe abstract class ✨ NEW (2025-11-30)
   - `linear.py` - Linear probe with temperature scaling

4. ✅ **Evaluation Framework** (`src/evaluation/`)
   - `metrics.py` - ECE, Brier, AUROC, AUPR, accuracy
   - `calibration.py` - Reliability diagrams, temperature/Platt scaling
   - `selective.py` - Selective prediction analysis

5. ✅ **Utilities** (`src/utils/`)
   - `config.py` - YAML configuration management
   - `logging.py` - Experiment tracking (loguru + WandB)

6. ✅ **Experiment Scripts** (`experiments/`) ✨ NEW (2025-11-30)
   - `baseline_linear_probe.py` - End-to-end baseline experiment
   - `layer_analysis.py` - Systematic layer performance analysis
   - `README.md` - Comprehensive usage documentation

**Quick Start**:
```bash
# Install dependencies
pip install -r requirements.txt

# Quick test (5-10 minutes)
python experiments/layer_analysis.py --quick --num-samples 50

# Full layer analysis
python experiments/layer_analysis.py --num-samples 500

# Baseline probe training
python experiments/baseline_linear_probe.py
```

### Phase 2: Advanced Methods (Next Steps)

**Planned Implementations**:
- [ ] CCPS implementation (perturbation-based calibration)
- [ ] Semantic entropy (clustering-based uncertainty)
- [ ] Cross-model comparison experiments
- [ ] Cross-dataset evaluation
- [ ] Probe architecture variants (MLP, attention-based)

**Research Goals**:
- Replicate CCPS 55% ECE reduction
- Validate "middle layers optimal" hypothesis
- Compare Llama/Mistral/Qwen performance
- Establish baseline metrics for future work

### Phase 3: Innovation (Future)

- Hierarchical multi-scale probing (novel contribution)
- Uncertainty-aware SAE training
- Causal circuit identification
- VLM extension experiments
- Temporal dynamics analysis

## Configuration Management

All experiments use YAML configs in `configs/`:

```yaml
# configs/linear_probe.yaml
experiment:
  name: "llama-3.1-8b-linear-probe"
  seed: 42

model:
  name: "meta-llama/Llama-3.1-8B"
  quantization: "8bit"
  device: "cuda"

data:
  dataset: "mmlu"
  split_ratio: [0.7, 0.15, 0.15]
  batch_size: 32

probe:
  type: "linear"
  layers: [8, 16, 24, 31]
  input_dim: 4096
  hidden_dim: null  # Linear probe
  dropout: 0.0

training:
  epochs: 50
  learning_rate: 1e-3
  weight_decay: 1e-5
  early_stopping_patience: 5

evaluation:
  metrics: ["ece", "brier", "auroc", "aupr"]
  calibration_method: "temperature"
  save_plots: true
```

## Testing Strategy

- **Unit tests**: Each module independently tested
- **Integration tests**: End-to-end experiment workflows
- **Regression tests**: Ensure metric calculations match published values
- **Performance tests**: Memory usage, runtime benchmarks

```bash
pytest tests/                    # Run all tests
pytest tests/test_metrics.py     # Specific module
pytest -k "calibration"          # Pattern matching
```

## Logging and Tracking

**WandB Integration**:
```python
import wandb
from src.utils import setup_logging

setup_logging(project="llm-confidence", config=experiment_config)
# Automatic logging of metrics, artifacts, configs
```

**Outputs Structure**:
```
outputs/
├── experiment_name/
│   ├── config.yaml          # Saved configuration
│   ├── checkpoints/         # Model checkpoints
│   ├── metrics.json         # Evaluation results
│   ├── plots/               # Generated visualizations
│   └── logs/                # Training logs
```

## Memory and Performance Optimization

### Hidden State Caching
- Extract once, reuse for multiple probes
- Memory-mapped NumPy arrays for datasets > RAM
- HDF5 for structured storage with compression

### Training Optimization
- Gradient checkpointing for large models
- Mixed precision training (bfloat16)
- Batch size tuning based on available VRAM
- Multi-GPU support via DistributedDataParallel

### Recommended Hardware
- **Minimum**: 16GB VRAM GPU (RTX 4080, A10) for 8B models
- **Recommended**: 24GB VRAM GPU (RTX 4090, A5000) for 8B + experiments
- **Optimal**: 40GB+ VRAM GPU (A100) for 70B models

## Dependency Management

**Core Dependencies**:
- `torch >= 2.0` (PyTorch for models and training)
- `transformers >= 4.36` (HuggingFace models)
- `datasets >= 2.14` (HuggingFace datasets)
- `bitsandbytes >= 0.41` (Quantization)
- `accelerate >= 0.25` (Multi-GPU, mixed precision)
- `scikit-learn >= 1.3` (Metrics, utilities)
- `numpy >= 1.24`
- `pandas >= 2.0`
- `matplotlib >= 3.7` (Plotting)
- `seaborn >= 0.12` (Statistical plots)
- `wandb >= 0.16` (Experiment tracking)
- `hydra-core >= 1.3` (Configuration)
- `tqdm >= 4.65` (Progress bars)

**Optional Dependencies**:
- `jupyter >= 1.0` (Notebooks)
- `optuna >= 3.3` (Hyperparameter optimization)
- `pytest >= 7.4` (Testing)
- `black >= 23.0` (Code formatting)
- `mypy >= 1.5` (Type checking)

## Next Steps

**Immediate (Phase 2)**:
1. Run baseline experiments to establish performance benchmarks
2. Implement CCPS (Conformal Calibration via Perturbation Sampling)
3. Implement semantic entropy clustering
4. Cross-model comparison (Llama vs Mistral vs Qwen)
5. Cross-dataset evaluation (MMLU vs TriviaQA vs GSM8K)

**Medium-term**:
1. Prototype hierarchical multi-scale probe
2. Probe architecture variants (MLP, attention-based)
3. Ablation studies on layer selection and training hyperparameters

**Long-term (Phase 3)**:
1. Uncertainty-aware SAE training
2. Causal circuit identification for confidence
3. VLM extension experiments

## Success Metrics

### Phase 1 Milestones ✅ COMPLETE

**Infrastructure**:
- [x] Complete data loading pipeline (3 datasets: MMLU, TriviaQA, GSM8K)
- [x] Model management with quantization support (9 LLMs)
- [x] Hidden state extraction with caching
- [x] Baseline probe implementation (LinearProbe with temperature scaling)
- [x] Comprehensive evaluation metrics (ECE, Brier, AUROC, AUPR)
- [x] End-to-end experiment scripts with visualization

**Code Quality**:
- [x] Modular, extensible architecture
- [x] Comprehensive docstrings and documentation
- [x] Configuration-driven experiments
- [x] WandB integration for tracking

### Phase 2 Milestones (In Progress)

**Technical Targets**:
- [ ] Baseline linear probe achieves AUROC > 0.75 on MMLU
- [ ] CCPS replication within 5% of published ECE reduction (target: 55%)
- [ ] Framework processes 10k examples in < 1 hour (extraction + training)
- [ ] Identify optimal layer range across model families

**Research Targets**:
- [ ] Validate "middle layers optimal" hypothesis (25-75% depth)
- [ ] Establish baseline metrics for all 3 datasets
- [ ] Cross-model probe transfer feasibility study
- [ ] Layer-wise performance analysis published

### Phase 3 Milestones (Future)

**Innovation Targets**:
- [ ] Hierarchical probe outperforms linear by 3+ AUROC points
- [ ] Train first uncertainty-aware SAE with interpretable features
- [ ] Map preliminary confidence circuit in Llama architecture
- [ ] Demonstrate cross-architecture generalization

## References

See `RESEARCH.md` for comprehensive literature review and research roadmap.
