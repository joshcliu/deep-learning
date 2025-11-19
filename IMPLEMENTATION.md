# Implementation Plan: LLM Confidence Probing Framework

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

### Phase 1: Foundation (Weeks 1-2)

1. **Setup Environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # or venv\Scripts\activate on Windows
   pip install -e .
   ```

2. **Download Initial Data**
   ```bash
   python scripts/download_datasets.py --datasets mmlu triviaqa
   ```

3. **Extract Hidden States**
   ```bash
   python scripts/extract_hiddens.py \
     --model meta-llama/Llama-3.1-8B \
     --dataset mmlu \
     --layers 8 16 24 31 \
     --output cache/llama-3.1-8b/mmlu
   ```

4. **Train Baseline Probe**
   ```bash
   python experiments/baseline_linear_probes.py \
     --config configs/linear_probe.yaml
   ```

### Phase 2: Experimentation (Weeks 3-6)

- Implement CCPS replication
- Add semantic entropy baseline
- Prototype hierarchical probe
- Run systematic layer analysis
- Compare across model families

### Phase 3: Innovation (Weeks 7+)

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

1. **Immediate**: Implement baseline linear probes on MMLU with Llama 3.1 8B
2. **Week 2**: Replicate CCPS results, compare with linear baseline
3. **Week 3**: Prototype hierarchical multi-scale probe
4. **Week 4**: Extend to TriviaQA and GSM8K benchmarks
5. **Month 2**: Begin SAE experiments and mechanistic interpretability

## Success Metrics

**Technical Milestones**:
- [ ] Baseline linear probe achieves AUROC > 0.75 on MMLU
- [ ] CCPS replication within 5% of published ECE reduction
- [ ] Hierarchical probe outperforms linear by 3+ AUROC points
- [ ] Framework processes 10k examples in < 1 hour (extraction + training)

**Research Milestones**:
- [ ] Identify optimal layers for uncertainty across 3+ model families
- [ ] Demonstrate cross-architecture probe transfer
- [ ] Train first uncertainty-aware SAE with interpretable features
- [ ] Map preliminary confidence circuit in Llama architecture

## References

See `RESEARCH.md` for comprehensive literature review and research roadmap.
