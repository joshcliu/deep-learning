# Claude Code Agent Guide

This document provides context for Claude Code agents working on this project. It describes what has been implemented, architectural decisions, and how to continue development effectively.

## Project Overview

**Name**: LLM Confidence Probing Framework
**Goal**: Research framework for extracting and quantifying uncertainty in large language models
**Research Focus**: Probing hidden states, calibration metrics, mechanistic interpretability
**Target**: MIT deep learning project with publication potential (NeurIPS, ICLR, ACL/EMNLP)

## Current Status: Phase 1 Complete ✅ | Phase 2 Ready

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

#### 5. Dataset Loaders Module ✓ (`src/data/`)

**Files**:
- `base.py`: BaseDataset abstract class and DatasetExample dataclass
- `mmlu.py`: MMLU with 57 subjects, category filtering, 4-choice questions
- `triviaqa.py`: TriviaQA open-domain QA with fuzzy answer matching
- `gsm8k.py`: GSM8K math problems with numerical answer extraction
- `DATA_README.md`: Comprehensive usage documentation

**Key Features**:
- Unified DatasetExample format across all datasets
- Multiple prompt formatting styles (QA, multiple choice, chain-of-thought)
- Answer correctness checking with fuzzy matching (TriviaQA/GSM8K)
- Dataset statistics and filtering utilities
- Helper functions for label generation

**Design Decisions**:
- Standardized interface via BaseDataset ABC
- HuggingFace datasets as backend (automatic caching)
- Flexible prompt formatting for different evaluation styles
- Metadata preservation for analysis and filtering

#### 6. Probe Module ✓ (`src/probes/`)

**Files**:
- `base.py`: BaseProbe abstract class with training/prediction interface
- `linear.py`: Linear probe with temperature scaling and early stopping

**Key Features (LinearProbe)**:
- Single-layer logistic classifier with dropout
- Built-in temperature scaling for calibration
- Early stopping with validation monitoring
- Batch prediction with progress bars
- Model checkpointing (save/load)
- Config-based initialization

**Design Decisions**:
- BaseProbe as ABC ensures consistent interface for future probes
- Temperature scaling integrated (not post-hoc addon)
- LBFGS optimizer for temperature fitting (convex optimization)
- Training returns comprehensive history dict

#### 7. Experiment Scripts ✓ (`experiments/`)

**Files**:
- `baseline_linear_probe.py`: End-to-end training pipeline for linear probes
- `layer_analysis.py`: Systematic layer-wise performance analysis

**Key Features**:
- Complete pipeline: data loading → extraction → training → evaluation
- Multi-layer probe training and comparison
- Automatic result visualization and logging
- WandB integration for experiment tracking
- Command-line interface with argparse

#### 8. Tinker API Integration ✓ (`src/tinker/`) ✨ NEW (2025-11-30)

**Files**:
- `client.py`: Tinker API client wrapper with authentication
- `weights.py`: Download and convert Tinker LoRA weights to PEFT format
- `__init__.py`: Public API exports

**Key Features**:
- Seamless integration with Tinker API for distributed LoRA fine-tuning
- Automatic download of trained weights from Tinker
- Conversion of Tinker LoRA adapters to PEFT format
- Load Tinker-trained models into HuggingFace for hidden state extraction
- Environment variable API key management

**Hybrid Workflow**:
1. Fine-tune models on Tinker (distributed, efficient)
2. Download LoRA weights as tar archive
3. Convert to PEFT format (compatible with HuggingFace)
4. Load into ModelLoader for hidden state extraction
5. Probe the fine-tuned model using existing infrastructure

**Integration with Existing Code**:
- `ModelLoader` updated to support `tinker_lora_path` parameter
- PEFT library added to requirements for LoRA adapter loading
- Documentation in `TINKER_INTEGRATION.md`

### What Is NOT Yet Implemented

#### Phase 1: ✅ COMPLETE

All Phase 1 components are now implemented.

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

**Phase 1 Complete! ✅** All baseline components implemented and ready to use.

**Next: Phase 2 - Advanced Calibration Methods**

1. **Run baseline experiments**:
   - Execute `experiments/baseline_linear_probe.py` on Llama 3.1 8B
   - Execute `experiments/layer_analysis.py` to validate layer hypothesis
   - Document baseline performance metrics

2. **Implement CCPS** (Conformal Calibration via Perturbation Sampling):
   - Create `src/calibration/ccps.py`
   - Follow 2025 paper methodology
   - Target: 55% ECE reduction over baseline
   - Compare with temperature scaling

3. **Implement Semantic Entropy**:
   - Create `src/calibration/semantic_entropy.py`
   - Clustering-based uncertainty quantification
   - Integration with existing probe framework

4. **Expand experiment suite**:
   - Cross-model comparison (Llama vs Mistral vs Qwen)
   - Cross-dataset evaluation (MMLU vs TriviaQA vs GSM8K)
   - Ablation studies on probe architecture

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

The foundation is solid. Phase 1 is complete with full baseline infrastructure.
Next work should focus on advanced calibration methods (CCPS, semantic entropy) and comprehensive experiments.

---
**Last Updated**: 2025-11-30
**Phase**: Phase 1 Complete ✅ | Phase 2 Ready
**Primary Contact**: User (joshc)

## Recent Updates (2025-11-30)

### Phase 1 Completion
- ✅ Created `src/probes/base.py` - BaseProbe abstract class
- ✅ Updated `src/probes/__init__.py` - Exports LinearProbe and BaseProbe
- ✅ Verified dataset loaders (MMLU, TriviaQA, GSM8K) - All functional
- ✅ Verified `src/probes/linear.py` - Comprehensive implementation with temperature scaling
- ✅ Created `experiments/baseline_linear_probe.py` - End-to-end baseline experiment
- ✅ Created `experiments/layer_analysis.py` - Systematic layer performance analysis
- ✅ Updated documentation to reflect Phase 1 completion

### Tinker API Integration
- ✅ Created `src/tinker/` module - Tinker API integration for distributed fine-tuning
  - `client.py` - API client wrapper with authentication
  - `weights.py` - Download and convert Tinker weights to PEFT format
- ✅ Updated `src/models/loader.py` - Support for loading Tinker LoRA adapters via PEFT
- ✅ Updated `requirements.txt` - Added PEFT library (required) and Tinker API (optional)
- ✅ Created `TINKER_INTEGRATION.md` - Comprehensive integration guide

**Hybrid Architecture**: Fine-tune on Tinker → Download weights → Load with PEFT → Extract hidden states for probing

### Ready to Run
All components are now implemented and ready for experiments:
```bash
# Run baseline linear probe experiment
python experiments/baseline_linear_probe.py

# Run layer analysis (quick mode for testing)
python experiments/layer_analysis.py --quick --num-samples 200

# Full layer analysis
python experiments/layer_analysis.py
```
