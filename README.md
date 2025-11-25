# LLM Confidence Probing Framework

A comprehensive framework for probing large language models to extract and quantify uncertainty. This codebase implements state-of-the-art methods for uncertainty quantification, including linear probes, CCPS, semantic entropy, and mechanistic interpretability approaches.

## Features

- **Model Management**: Unified interface for loading LLMs (Llama, Mistral, Qwen) with automatic quantization
- **Hidden State Extraction**: Efficient extraction and caching of hidden states from transformer layers
- **Calibration Metrics**: ECE, Brier score, AUROC, AUPR with comprehensive visualization
- **Selective Prediction**: Coverage-accuracy analysis for production deployment
- **Experiment Tracking**: Integration with Weights & Biases and structured logging
- **Modular Architecture**: Easy to extend with new models, probes, and evaluation methods

## Quick Start

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd deep-learning

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -e .
```

### Basic Usage

#### 1. Load a Model and Extract Hidden States

```python
from src.models import ModelLoader, HiddenStateExtractor

# Load model with 8-bit quantization
loader = ModelLoader("meta-llama/Llama-3.1-8B")
model, tokenizer = loader.load(quantization="8bit")

# Extract hidden states
extractor = HiddenStateExtractor(model, tokenizer)
hiddens = extractor.extract(
    texts=["What is the capital of France?", "Who wrote Hamlet?"],
    layers=[8, 16, 24, 31],  # Quartile layers
    cache_dir="cache/llama-3.1-8b"
)

print(f"Hidden states shape: {hiddens.shape}")
# Output: (2, 4, 4096) - 2 texts, 4 layers, 4096 hidden dim
```

#### 2. Evaluate Calibration Metrics

```python
from src.evaluation import CalibrationMetrics, plot_reliability_diagram
import numpy as np

# Example predictions and confidences
predictions = np.array([1, 1, 0, 1, 0])
confidences = np.array([0.9, 0.7, 0.4, 0.8, 0.3])
labels = np.array([1, 0, 0, 1, 0])

# Compute metrics
metrics = CalibrationMetrics(predictions, confidences, labels)
print(f"ECE: {metrics.ece():.4f}")
print(f"Brier: {metrics.brier():.4f}")
print(f"AUROC: {metrics.auroc():.4f}")

# Visualize calibration
plot_reliability_diagram(
    confidences, labels,
    save_path="outputs/reliability.png"
)
```

#### 3. Configuration-Driven Experiments

```python
from src.utils import load_config, ExperimentLogger

# Load experiment configuration
config = load_config("configs/linear_probe.yaml")

# Setup experiment logger
logger = ExperimentLogger(
    experiment_name=config.experiment.name,
    output_dir="outputs",
    config=config,
    use_wandb=True,
    wandb_project="llm-confidence"
)

# Log metrics during training
logger.log_metrics({"ece": 0.05, "auroc": 0.82}, step=1)
```

## Project Structure

```
deep-learning/
├── src/
│   ├── models/          # Model loading and hidden state extraction
│   │   ├── loader.py
│   │   ├── extractor.py
│   │   └── registry.py
│   ├── data/            # Dataset loaders (coming soon)
│   ├── probes/          # Probe implementations (coming soon)
│   ├── evaluation/      # Metrics and visualization
│   │   ├── metrics.py
│   │   ├── calibration.py
│   │   └── selective.py
│   ├── interpretability/# SAE, circuits (coming soon)
│   └── utils/           # Configuration and logging
├── experiments/         # Experiment scripts
├── configs/             # YAML configurations
├── notebooks/           # Jupyter notebooks
├── cache/               # Cached hidden states
└── outputs/             # Experiment results
```

## Supported Models

| Model | Size | Layers | Hidden Dim | Default Quant | Min VRAM |
|-------|------|--------|------------|---------------|----------|
| Llama 3.1 | 8B | 32 | 4096 | 8-bit | 16GB |
| Llama 3.1 | 70B | 80 | 8192 | 4-bit | 140GB |
| Llama 2 | 7B | 32 | 4096 | 8-bit | 14GB |
| Mistral | 7B | 32 | 4096 | 8-bit | 14GB |
| Mixtral | 8x7B | 32 | 4096 | 4-bit | 90GB |
| Qwen 2.5 | 7B | 28 | 3584 | 8-bit | 14GB |
| Qwen 2.5 | 72B | 80 | 8192 | 4-bit | 144GB |

## Example: End-to-End Workflow

```python
import numpy as np
from src.models import ModelLoader, HiddenStateExtractor
from src.evaluation import CalibrationMetrics, plot_reliability_diagram
from src.utils import setup_logging

# Setup logging
setup_logging(log_level="INFO")

# 1. Load model
loader = ModelLoader("meta-llama/Llama-3.1-8B")
model, tokenizer = loader.load(quantization="8bit")

# 2. Prepare data
texts = [
    "The capital of France is Paris.",
    "The Earth orbits the Moon.",
    "Python is a programming language.",
]
labels = np.array([1, 0, 1])  # Correct/incorrect

# 3. Extract hidden states
extractor = HiddenStateExtractor(model, tokenizer)
hiddens = extractor.extract(
    texts=texts,
    layers=[16],  # Middle layer
    cache_dir="cache/demo"
)

# 4. Train a simple linear probe (placeholder)
# In practice, use src.probes.LinearProbe
from sklearn.linear_model import LogisticRegression
X = hiddens[:, 0, :]  # Use first (only) layer
clf = LogisticRegression()
clf.fit(X, labels)

# 5. Get predictions and confidences
predictions = clf.predict(X)
confidences = clf.predict_proba(X)[:, 1]

# 6. Evaluate
metrics = CalibrationMetrics(predictions, confidences, labels)
print(f"ECE: {metrics.ece():.4f}")
print(f"AUROC: {metrics.auroc():.4f}")

# 7. Visualize
plot_reliability_diagram(confidences, labels, save_path="outputs/demo.png")
```

## Configuration Files

Experiments are configured via YAML files in `configs/`. Example:

```yaml
# configs/linear_probe.yaml
experiment:
  name: "llama-3.1-8b-linear-probe"
  seed: 42

model:
  name: "meta-llama/Llama-3.1-8B"
  quantization: "8bit"

extraction:
  layers: [8, 16, 24, 31]  # Quartile layers
  token_position: "last"
  cache_dir: "cache/llama-3.1-8b"

evaluation:
  metrics: ["ece", "brier", "auroc"]
  num_bins: 10
  save_plots: true
```

## Development Roadmap

### Phase 1: Foundation (Current)
- [x] Project structure and scaffolding
- [x] Model loading with quantization
- [x] Hidden state extraction
- [x] Core evaluation metrics
- [ ] Dataset loaders (MMLU, TriviaQA, GSM8K)
- [ ] Linear probe implementation

### Phase 2: Baselines (Next)
- [ ] CCPS implementation
- [ ] Semantic entropy
- [ ] Consistency-based methods
- [ ] Systematic layer analysis

### Phase 3: Innovation
- [ ] Hierarchical multi-scale probing
- [ ] Uncertainty-aware sparse autoencoders
- [ ] Causal circuit tracing
- [ ] VLM extensions

## Citation

If you use this code in your research, please cite:

```bibtex
@software{llm_confidence_probing,
  title = {LLM Confidence Probing Framework},
  author = {MIT Deep Learning Team},
  year = {2025},
  url = {https://github.com/...}
}
```

## License

MIT License - see LICENSE file for details

## Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## Resources

- [Research Roadmap](RESEARCH.md): Comprehensive literature review and research directions
- [Implementation Guide](IMPLEMENTATION.md): Detailed technical specifications
- [Documentation](docs/): API reference and tutorials (coming soon)

## Support

For issues and questions:
- GitHub Issues: Report bugs and feature requests
- Discussions: Ask questions and share ideas
