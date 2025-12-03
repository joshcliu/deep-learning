# Claude Code Agent Guide

This document provides context for Claude Code agents working on this project. It describes what has been implemented, architectural decisions, and how to continue development effectively.

## Project Overview

**Name**: LLM Confidence Probing Framework
**Goal**: Research framework for extracting and quantifying uncertainty in large language models
**Research Focus**: Probing hidden states, calibration metrics, mechanistic interpretability
**Target**: MIT deep learning project with publication potential (NeurIPS, ICLR, ACL/EMNLP)

## Current Status: Phase 1 Complete ✓

### What Has Been Implemented

#### 1. Project Infrastructure ✓
- Complete directory structure (`src/`, `configs/`, `experiments/`, `notebooks/`, etc.)
- Dependency management (`requirements.txt`, `pyproject.toml`)
- Git configuration (`.gitignore` with project-specific entries)
- Documentation (`README.md`, `IMPLEMENTATION.md`, `RESEARCH.md`)

#### 2. Model Management Module ✓ (`src/models/`)

**Files**:
- `registry.py`: Model configurations for 9 LLMs (Llama 3.1, Llama 2, Mistral, Mixtral, Qwen 2.5)
- `loader.py`: Unified model loading with quantization support
- `extractor.py`: Hidden state extraction with caching

**Key Features**:
- Automatic quantization (4-bit/8-bit via bitsandbytes)
- Layer-specific extraction (single, multiple, quartile positions)
- Token position handling (last, CLS, mean)
- Memory-mapped caching for large datasets
- Batch processing with progress bars

**Design Decisions**:
- Used HuggingFace Transformers as base (ecosystem compatibility)
- Caching keyed by content hash (deterministic, collision-resistant)
- Quartile layer selection based on research showing middle layers optimal for uncertainty
- Default quantization in registry per model (balances memory/performance)

#### 3. Evaluation Framework ✓ (`src/evaluation/`)

**Files**:
- `metrics.py`: Core calibration metrics (ECE, Brier, AUROC, AUPR, accuracy)
- `calibration.py`: Visualization and post-hoc calibration methods
- `selective.py`: Selective prediction analysis

**Key Features**:
- Expected Calibration Error (ECE) with binning statistics
- ROC/PR curve generation
- Reliability diagrams (matplotlib + seaborn)
- Temperature scaling and Platt scaling calibrators
- Coverage-accuracy curves for selective prediction
- Statistical utilities (confidence intervals, significance testing ready)

**Design Decisions**:
- `CalibrationMetrics` class caches computations (efficiency)
- Separate visualization functions (matplotlib figures can be saved/shown independently)
- All metrics return floats/arrays (easy to log to WandB)
- Bins default to 10 for ECE (standard in literature per Guo et al. 2017)

#### 4. Utilities Module ✓ (`src/utils/`)

**Files**:
- `config.py`: YAML configuration management (OmegaConf-based)
- `logging.py`: Experiment tracking (loguru + WandB)

**Key Features**:
- Default config template with sensible defaults
- Config merging for overrides
- `ExperimentLogger` class (unified file logging + WandB)
- Automatic artifact saving
- Structured output directories

**Design Decisions**:
- OmegaConf for configs (dot notation access, type safety, merging)
- loguru instead of standard logging (better DX, automatic interception)
- WandB optional (graceful degradation if not installed)

### What Is NOT Yet Implemented

#### Phase 1 Remaining:
- [ ] Dataset loaders (`src/data/`) - MMLU, TriviaQA, GSM8K, TruthfulQA
- [ ] Linear probe implementation (`src/probes/linear.py`)
- [ ] Base probe class (`src/probes/base.py`)

#### Phase 2:
- [ ] CCPS implementation (perturbation-based calibration)
- [ ] Semantic entropy (clustering + entropy over meanings)
- [ ] Consistency-based methods
- [ ] Full experiment scripts (`experiments/`)

#### Phase 3:
- [ ] Hierarchical multi-scale probing (novel contribution)
- [ ] Uncertainty-aware sparse autoencoders (SAE)
- [ ] Causal circuit tracing
- [ ] VLM extensions

## Architectural Patterns

### 1. Module Organization

```
src/<module>/
├── __init__.py      # Exports public API
├── <core>.py        # Main implementation
└── <utilities>.py   # Supporting code
```

**Convention**: `__init__.py` exports only user-facing classes/functions. Internal utilities stay private.

### 2. Class Design

**ModelLoader Pattern**:
```python
loader = ModelLoader(model_name)  # Initialize with config
model, tokenizer = loader.load()   # Load with options
info = loader.get_model_info()    # Introspection
```

**Extractor Pattern**:
```python
extractor = Extractor(model, tokenizer)  # Bind to model
results = extractor.extract(data, **options)  # Process data
stats = extractor.get_statistics(data)  # Analysis
```

**Metrics Pattern**:
```python
metrics = Metrics(predictions, confidences, labels)  # Initialize with data
value = metrics.ece()  # Compute specific metric
all_metrics = metrics.compute_all()  # Batch computation
```

### 3. Configuration-Driven Experiments

All experiments should follow this pattern:
```python
from src.utils import load_config, ExperimentLogger

config = load_config("configs/experiment.yaml")
logger = ExperimentLogger(config.experiment.name, config=config)

# Run experiment
for step in training:
    logger.log_metrics({"loss": loss}, step=step)

logger.finish()
```

### 4. Caching Strategy

**Hidden states**:
- Cache key = hash(texts + layers + max_length + token_position)
- Format: NumPy `.npy` files (memory-mappable)
- Location: `cache/<model-name>/<dataset>/`

**Models**:
- HuggingFace handles caching automatically
- Location: `~/.cache/huggingface/`

### 5. Error Handling

**Pattern used**:
```python
try:
    # Operation
except SpecificError as e:
    logger.error(f"Context: {e}")
    raise RuntimeError("User-friendly message") from e
```

**Validation**: All public functions validate inputs with assertions/raises at entry

## Code Conventions

### Style
- **Line length**: 100 characters (Black formatter)
- **Type hints**: All function signatures (mypy-compatible)
- **Docstrings**: Google style with Args/Returns/Example sections
- **Imports**: Grouped (stdlib, third-party, local) with absolute imports

### Naming
- **Classes**: PascalCase (`ModelLoader`, `CalibrationMetrics`)
- **Functions**: snake_case (`compute_ece`, `load_config`)
- **Private**: Leading underscore (`_extract_batch`, `_get_cache_path`)
- **Constants**: UPPER_CASE (`MODEL_REGISTRY`)

### Documentation
- **Public API**: Full docstrings with examples
- **Private functions**: Brief docstrings
- **Complex logic**: Inline comments explaining "why"

## Key Research Context

### From RESEARCH.md

**Critical findings**:
1. **Middle layers (50-75% depth) outperform final layers** for uncertainty - layer 16/32 for Llama-2-7B optimal
2. **CCPS achieves 55% ECE reduction** - state-of-the-art as of 2025
3. **Well-calibrated systems: ECE < 0.05** - baseline LLMs often 0.10-0.15
4. **AUROC > 0.75-0.85** for hallucination detection is achievable

**Novel research directions**:
- Hierarchical multi-scale probing (token → span → semantic → global)
- Uncertainty-aware SAEs (interpretable uncertainty features)
- Confidence circuits (mechanistic understanding via activation patching)
- Temporal dynamics (uncertainty evolution during generation)

### Model Support Priority

**Tier 1** (fully supported):
- Llama 3.1 8B (primary research model)
- Llama 2 7B (baseline comparisons)
- Mistral 7B (efficiency comparisons)

**Tier 2** (supported, less tested):
- Qwen 2.5 7B/14B/72B
- Llama 3.1 70B
- Mixtral 8x7B

**Future**: Vision-language models (LLaVA, CLIP-based)

## Development Workflow

### Adding a New Feature

1. **Plan**: Check if fits existing architecture or needs new module
2. **Implement**: Follow existing patterns in similar modules
3. **Document**: Docstrings + update README.md
4. **Test**: Create example in `notebooks/` or test script
5. **Config**: Add config options if needed in `src/utils/config.py` defaults

### Adding a Dataset Loader

**Template** (`src/data/mmlu.py`):
```python
from typing import List, Dict
from datasets import load_dataset

class MMLUDataset:
    def __init__(self, split: str = "validation", subjects: List[str] = None):
        self.dataset = load_dataset("cais/mmlu", split=split)
        # Filter subjects, format data

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return {
            "question": ...,
            "choices": ...,
            "answer": ...,
            "metadata": ...
        }
```

**Export** in `src/data/__init__.py`:
```python
from .mmlu import MMLUDataset
__all__ = ["MMLUDataset"]
```

### Adding a Probe

**Template** (`src/probes/base.py`):
```python
from abc import ABC, abstractmethod
import numpy as np
import torch.nn as nn

class BaseProbe(ABC, nn.Module):
    @abstractmethod
    def forward(self, hiddens: torch.Tensor) -> torch.Tensor:
        """Forward pass returning confidence scores."""
        pass

    def fit(self, X_train, y_train, X_val, y_val):
        """Training loop with validation."""
        pass

    def predict(self, X) -> np.ndarray:
        """Predict confidence scores."""
        pass
```

**Concrete implementation** (`src/probes/linear.py`):
```python
class LinearProbe(BaseProbe):
    def __init__(self, input_dim: int, dropout: float = 0.0):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, hiddens):
        return torch.sigmoid(self.linear(self.dropout(hiddens)))
```

### Creating an Experiment Script

**Template** (`experiments/baseline_linear_probe.py`):
```python
from src.utils import load_config, ExperimentLogger, setup_logging
from src.models import ModelLoader, HiddenStateExtractor
from src.data import MMLUDataset
from src.probes import LinearProbe
from src.evaluation import CalibrationMetrics

def main():
    setup_logging(log_level="INFO")
    config = load_config("configs/linear_probe.yaml")

    logger = ExperimentLogger(
        experiment_name=config.experiment.name,
        output_dir=config.experiment.output_dir,
        config=config,
        use_wandb=config.logging.use_wandb,
        wandb_project=config.logging.wandb_project
    )

    # Load model
    # Extract hiddens
    # Train probe
    # Evaluate
    # Log results

    logger.finish()

if __name__ == "__main__":
    main()
```

## Important Notes & Gotchas

### Memory Management

1. **Hidden states for Llama-2-70B with 32k context = ~20GB**
   - Always use caching for experiments
   - Process in batches (16-64 recommended)
   - Use memory-mapped arrays for datasets > RAM

2. **Quantization is essential for larger models**
   - 8-bit: ~2x memory reduction, minimal quality loss
   - 4-bit: ~4x reduction, acceptable for probing (not generation)

3. **GPU memory monitoring**:
   ```python
   import torch
   print(f"Allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
   ```

### Model Loading

1. **Gated models** (Llama) require HuggingFace token:
   ```python
   loader = ModelLoader(model_name, use_auth_token="hf_...")
   ```

2. **Device mapping**: `device_map="auto"` works for multi-GPU
   - Single GPU: Automatic
   - CPU fallback: Slower but works

3. **Trust remote code**: Some models (Qwen) need `trust_remote_code=True`

### Hidden State Extraction

1. **Layer indexing**:
   - `outputs.hidden_states[0]` = embeddings
   - `outputs.hidden_states[1]` = layer 0 output
   - `outputs.hidden_states[N+1]` = layer N output
   - Extractor handles this automatically (+1 offset)

2. **Token positions**:
   - Autoregressive (Llama, Mistral): Use "last" token
   - Encoder-only (BERT): Use "cls" token
   - Research suggests "last" token sufficient (attention aggregates context)

3. **Sequence length**:
   - Padding: Handled automatically
   - Truncation: Set `max_length` appropriately
   - Attention mask: Used for "mean" token position

### Evaluation Metrics

1. **ECE bins**: 10-20 bins standard, but need sufficient samples per bin
   - Rule of thumb: `num_samples / num_bins > 100`
   - Small datasets: Use 5 bins

2. **AUROC undefined** when all labels same class
   - Handled gracefully (returns np.nan)
   - Check label distribution before computing

3. **Calibration vs discrimination**:
   - ECE/Brier: Calibration quality
   - AUROC/AUPR: Discrimination quality
   - Both important; can have high AUROC but poor calibration

### Configuration

1. **Layer specification**:
   - `null` in config → use optimal layers from registry
   - List → use specified layers
   - Single int → wrap in list automatically

2. **Paths**:
   - All paths in configs are relative to project root
   - Use forward slashes (works on Windows too)

3. **Overrides**:
   ```python
   config = load_config("base.yaml")
   config = merge_configs(config, {"training": {"lr": 1e-4}})
   ```

## Testing & Validation

### Quick Smoke Test

```python
# Test model loading
from src.models import ModelLoader
loader = ModelLoader("meta-llama/Llama-3.1-8B")
assert loader.config.num_layers == 32

# Test metrics
from src.evaluation import compute_ece
import numpy as np
conf = np.array([0.9, 0.8, 0.6, 0.4])
pred = np.array([1, 1, 1, 0])
labels = np.array([1, 0, 1, 0])
ece, _ = compute_ece(conf, pred, labels)
assert 0 <= ece <= 1
```

### Integration Test

See `notebooks/01_getting_started.ipynb` for end-to-end example.

## Future Agent Tasks

### Immediate Next Steps (Priority Order)

1. **Implement dataset loaders** (`src/data/`):
   - Start with MMLU (most important benchmark)
   - Add TriviaQA, GSM8K
   - Create unified `DataLoader` interface

2. **Implement linear probe** (`src/probes/linear.py`):
   - Extend `BaseProbe` (create this first)
   - PyTorch nn.Module with single linear layer
   - Training loop with early stopping
   - Post-hoc temperature scaling

3. **Create baseline experiment** (`experiments/baseline_linear_probes.py`):
   - Load MMLU dataset
   - Extract hiddens from Llama 3.1 8B
   - Train linear probe on multiple layers
   - Evaluate and compare layer performance
   - Log to WandB

4. **Systematic layer analysis** (`experiments/layer_analysis.py`):
   - Test all 32 layers for Llama 3.1 8B
   - Generate heatmap of performance by layer
   - Validate "middle layers optimal" hypothesis

### Medium-term Tasks

1. **CCPS implementation**: Follow 2025 paper closely, verify 55% ECE reduction
2. **Semantic entropy**: Implement clustering-based uncertainty
3. **Probe architecture search**: MLP, attention-based variants
4. **Cross-model analysis**: Compare Llama/Mistral/Qwen on same dataset

### Research Contributions (Novel)

1. **Hierarchical multi-scale probing**: Most promising for publication
2. **Uncertainty-aware SAE**: Integration with Anthropic/DeepMind work
3. **Confidence circuits**: Mechanistic interpretability angle

## External Resources

### User Has Access To
- **Tinker API key**: For fine-tuning open-source LLMs (mentioned but not configured yet)
- **GPU resources**: Assumed sufficient for 8B models (16GB+ VRAM)

### Key Repositories (for reference)
- `factual-confidence-of-llms` (Amazon Science): Production baseline
- `llm-uncertainty` (ICLR 2024): Academic benchmarking
- `LLaMA-Factory`: Fine-tuning infrastructure

### Papers to Reference
- Guo et al. 2017: Temperature scaling (calibration)
- Kuhn et al. 2024 (Nature): Semantic entropy
- CCPS 2025: Perturbation-based calibration
- Research roadmap in `RESEARCH.md` has full bibliography

## Final Notes for Claude Agents

1. **Always check this file first** when resuming work on this project
2. **Update this file** when completing major features or making architectural decisions
3. **User expectation**: Clean, production-quality code with strong documentation
4. **Research context matters**: This is for publication, not just a demo
5. **Ask user about**: Hardware constraints, API keys, dataset preferences before heavy operations
6. **Testing philosophy**: Create notebooks for validation, not formal test suites (research code)

The foundation is solid. Next agent should focus on datasets → probes → experiments in that order.

---
**Last Updated**: 2025-11-19
**Phase**: 1 Complete, moving to Phase 2
**Primary Contact**: User (joshc)
