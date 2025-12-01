# LLM Confidence Probing Framework

**A research toolkit for extracting and quantifying uncertainty in large language models**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This framework implements state-of-the-art methods for uncertainty quantification in LLMs, including linear probes, calibration metrics, and mechanistic interpretability approaches. Designed for researchers working on model confidence, hallucination detection, and safe AI deployment.

---

## 🎯 What This Does

Train lightweight **probes** on LLM hidden states to:
- **Predict model confidence** before generation
- **Detect hallucinations** and incorrect answers
- **Understand uncertainty** across model layers
- **Calibrate predictions** for production deployment

**Key Insight**: Middle layers (25-75% depth) often encode more uncertainty information than final layers!

---

## ✨ Features

- ✅ **Model Management**: Unified interface for Llama, Mistral, Qwen (9 models supported)
- ✅ **Hidden State Extraction**: Efficient layer-wise extraction with automatic caching
- ✅ **Dataset Loaders**: MMLU, TriviaQA, GSM8K with standardized interfaces
- ✅ **Linear Probes**: Temperature-scaled classifiers with early stopping
- ✅ **Calibration Metrics**: ECE, Brier, AUROC, AUPR with visualizations
- ✅ **Experiment Scripts**: Ready-to-run baseline experiments
- ✅ **Tinker API Integration**: Distributed fine-tuning → local probing workflow
- ✅ **WandB Logging**: Track experiments with automatic artifact saving

---

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd deep-learning

# Install dependencies
pip install -r requirements.txt

# Optional: Install Tinker API for distributed fine-tuning
pip install tinker
export TINKER_API_KEY=your_key
```

### Run Your First Experiment (5 minutes)

```bash
# Quick test - validates full pipeline
python experiments/layer_analysis.py --quick --num-samples 100
```

This will:
1. Load MMLU dataset
2. Extract hidden states from a small model
3. Train probes on multiple layers
4. Generate performance visualizations
5. Save results to `outputs/layer_analysis/`

### Run Full Baseline (2-4 hours)

```bash
# Train linear probes on multiple layers
python experiments/baseline_linear_probe.py

# Or customize via config
python experiments/baseline_linear_probe.py --config configs/linear_probe.yaml
```

Expected results:
- ECE: 0.05-0.10 (well-calibrated if < 0.05)
- AUROC: 0.75-0.85 (good discrimination)
- Best layer: typically 50-75% depth (e.g., layer 16-24 for 32-layer models)

---

## 📚 Usage Examples

### 1. Extract Hidden States

```python
from src.models import ModelLoader, HiddenStateExtractor

# Load model with quantization
loader = ModelLoader("meta-llama/Llama-3.1-8B")
model, tokenizer = loader.load(quantization="8bit")

# Extract hidden states from quartile layers
extractor = HiddenStateExtractor(model, tokenizer)
hiddens = extractor.extract(
    texts=["What is the capital of France?"],
    layers=[8, 16, 24, 31],
    cache_dir="cache/llama-3.1-8b",
    use_cache=True  # Reuse if already extracted
)

print(f"Shape: {hiddens.shape}")  # (1, 4, 4096) - texts, layers, hidden_dim
```

### 2. Train a Probe

```python
from src.probes import LinearProbe
import numpy as np

# Prepare data
hiddens_train = np.random.randn(1000, 4096)  # From your extracted states
labels_train = np.random.randint(0, 2, 1000)  # 0=incorrect, 1=correct

# Train probe with temperature scaling
probe = LinearProbe(input_dim=4096, dropout=0.1)
history = probe.fit(
    X_train=hiddens_train,
    y_train=labels_train,
    X_val=hiddens_val,
    y_val=labels_val,
    batch_size=128,
    num_epochs=50,
    patience=5
)

# Predict confidence scores
confidences = probe.predict(hiddens_test)
```

### 3. Evaluate Calibration

```python
from src.evaluation import CalibrationMetrics

predictions = (confidences > 0.5).astype(int)
metrics = CalibrationMetrics(predictions, confidences, labels_test)

# Compute all metrics
results = metrics.compute_all()
print(f"ECE:      {results['ece']:.4f}")
print(f"Brier:    {results['brier']:.4f}")
print(f"AUROC:    {results['auroc']:.4f}")
print(f"Accuracy: {results['accuracy']:.4f}")

# Generate visualizations
from src.evaluation import plot_reliability_diagram
plot_reliability_diagram(
    confidences, labels_test,
    save_path="outputs/reliability.png"
)
```

### 4. Load Datasets

```python
from src.data import MMLUDataset, TriviaQADataset, GSM8KDataset

# MMLU: 57 subjects, multiple choice
mmlu = MMLUDataset(split="validation", category="stem")
print(f"Loaded {len(mmlu)} STEM questions")

# TriviaQA: Open-domain QA
trivia = TriviaQADataset(split="validation", max_examples=500)

# GSM8K: Math word problems
gsm8k = GSM8KDataset(split="test")

# Iterate
for example in mmlu:
    print(example.question)
    print(example.choices)
    print(f"Answer: {example.answer}")
```

### 5. Tinker API Integration (Optional)

```python
from src.tinker import download_and_convert_weights
from src.models import ModelLoader

# Download Tinker-trained model
peft_path = download_and_convert_weights(
    tinker_checkpoint="tinker://abc123/sampler_weights/final",
    base_model_name="meta-llama/Llama-3.1-8B",
    output_dir="weights/my_finetuned_model"
)

# Load with LoRA adapter
loader = ModelLoader(
    "meta-llama/Llama-3.1-8B",
    tinker_lora_path=peft_path,
    quantization="8bit"
)
model, tokenizer = loader.load()

# Extract hidden states as usual!
```

See [`TINKER_INTEGRATION.md`](TINKER_INTEGRATION.md) for complete guide.

---

## 📁 Project Structure

```
deep-learning/
├── src/
│   ├── models/          # ✅ Model loading + hidden state extraction
│   ├── data/            # ✅ Dataset loaders (MMLU, TriviaQA, GSM8K)
│   ├── probes/          # ✅ Linear probes with temperature scaling
│   ├── evaluation/      # ✅ Calibration metrics + visualizations
│   ├── tinker/          # ✅ Tinker API integration
│   └── utils/           # ✅ Configuration, logging, caching
├── experiments/         # ✅ Ready-to-run experiment scripts
│   ├── baseline_linear_probe.py
│   ├── layer_analysis.py
│   └── README.md
├── configs/             # ✅ YAML configuration files
├── notebooks/           # Jupyter notebooks for exploration
├── cache/               # Cached hidden states (auto-generated)
└── outputs/             # Experiment results (auto-generated)
```

**Status**: Phase 1 Complete ✅ | All baseline infrastructure implemented

---

## 🔬 Supported Models

| Model | Size | Layers | Hidden Dim | Default Quant | Min VRAM |
|-------|------|--------|------------|---------------|----------|
| **Llama 3.1** | 8B | 32 | 4096 | 8-bit | 16GB |
| **Llama 3.1** | 70B | 80 | 8192 | 4-bit | 140GB |
| Llama 2 | 7B | 32 | 4096 | 8-bit | 14GB |
| Mistral | 7B | 32 | 4096 | 8-bit | 14GB |
| Mixtral | 8x7B | 32 | 4096 | 4-bit | 90GB |
| Qwen 2.5 | 7B | 28 | 3584 | 8-bit | 14GB |
| Qwen 2.5 | 14B | 48 | 5120 | 8-bit | 28GB |
| Qwen 2.5 | 72B | 80 | 8192 | 4-bit | 144GB |

**Recommended**: Start with Llama 3.1 8B (well-documented, good performance)

---

## 📊 Benchmarks & Datasets

| Dataset | Type | Size | Task |
|---------|------|------|------|
| **MMLU** | Multiple choice | 14k (val) | 57 subjects across STEM, humanities, etc. |
| **TriviaQA** | Open QA | 11k (val) | Factual knowledge with multiple answers |
| **GSM8K** | Math | 1.3k (test) | Grade school math word problems |

All datasets include:
- Standardized `DatasetExample` format
- Multiple prompt styles (QA, multiple choice, CoT)
- Correctness checking utilities
- Metadata for filtering and analysis

See [`src/data/DATA_README.md`](src/data/DATA_README.md) for detailed usage.

---

## 🛠️ Configuration

Experiments are configured via YAML files. Example (`configs/linear_probe.yaml`):

```yaml
experiment:
  name: "llama-3.1-8b-linear-probe"
  seed: 42
  output_dir: "outputs"

model:
  name: "meta-llama/Llama-3.1-8B"
  quantization: "8bit"
  device: "auto"

data:
  dataset: "mmlu"
  split_ratio: [0.7, 0.15, 0.15]  # train/val/test
  batch_size: 32
  num_samples: null  # null = use all

extraction:
  layers: [8, 16, 24, 31]  # Quartile layers
  token_position: "last"
  use_cache: true

probe:
  dropout: 0.0

training:
  epochs: 50
  learning_rate: 1e-3
  early_stopping_patience: 5

evaluation:
  metrics: ["ece", "brier", "auroc", "accuracy"]
  num_bins: 10

logging:
  use_wandb: false  # Set to true for experiment tracking
  wandb_project: "llm-confidence"
```

---

## 🎓 Research Context

### Why Probe Hidden States?

Recent research shows that **model internals contain rich uncertainty information** that's often lost by the time we see the final output. By training lightweight probes on hidden states, we can:

1. **Detect uncertainty earlier** in the forward pass
2. **Understand which layers** encode confidence vs task-specific features
3. **Build better calibrated systems** for production deployment
4. **Gain mechanistic insights** into how models represent uncertainty

### Key Findings (from literature)

- **Middle layers optimal**: Layers at 50-75% depth outperform final layers for uncertainty (Mielke et al., Kadavath et al.)
- **CCPS achieves 55% ECE reduction**: State-of-the-art calibration method (Burns et al. 2025)
- **Well-calibrated systems**: ECE < 0.05 is excellent, < 0.10 is good
- **AUROC > 0.75-0.85**: Achievable for hallucination detection with probes

### Research Roadmap

**Phase 1** ✅ (Complete):
- Baseline infrastructure
- Linear probes on hidden states
- Calibration metrics
- Layer-wise analysis

**Phase 2** (Next):
- CCPS implementation
- Semantic entropy
- Cross-model/dataset experiments

**Phase 3** (Future):
- Hierarchical multi-scale probing
- Uncertainty-aware sparse autoencoders
- Causal circuit tracing

See [`RESEARCH.md`](RESEARCH.md) for comprehensive literature review.

---

## 📖 Documentation

- **[QUICK_START.md](QUICK_START.md)** - Get started in 3 steps
- **[IMPLEMENTATION.md](IMPLEMENTATION.md)** - Technical specifications and architecture
- **[CLAUDE.md](CLAUDE.md)** - Complete project context for AI agents
- **[TINKER_INTEGRATION.md](TINKER_INTEGRATION.md)** - Tinker API integration guide
- **[experiments/README.md](experiments/README.md)** - Detailed experiment documentation
- **[src/data/DATA_README.md](src/data/DATA_README.md)** - Dataset usage guide

---

## 🔍 Troubleshooting

### Common Issues

**Out of Memory (OOM)**
```yaml
# Use quantization in config
model:
  quantization: "8bit"  # or "4bit" for larger models

# Or reduce batch size
data:
  batch_size: 16  # default is 32
```

**Slow First Run**
- Expected! Hidden state extraction is cached
- Subsequent runs are much faster
- Cache location: `cache/<model>/<dataset>/`

**HuggingFace Token Required**
```bash
# For gated models (Llama)
export HUGGING_FACE_HUB_TOKEN=your_token

# Or in config
model:
  use_auth_token: "hf_..."
```

**Import Errors**
```bash
# Make sure all dependencies installed
pip install -r requirements.txt

# For Tinker (optional)
pip install tinker peft
```

---

## 🤝 Contributing

Contributions welcome! Areas where help is needed:

- **Phase 2 methods**: CCPS, semantic entropy implementations
- **New datasets**: TruthfulQA, HotpotQA, etc.
- **Probe architectures**: MLP, attention-based variants
- **Visualizations**: Interactive dashboards, layer analysis plots
- **Documentation**: Tutorials, examples, API docs

**Process**:
1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Follow code conventions (see `CLAUDE.md`)
4. Add tests and documentation
5. Submit pull request

---

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

Built for MIT Deep Learning research project with publication potential (NeurIPS, ICLR, ACL/EMNLP).

Key inspirations:
- **Burns et al. (2025)**: CCPS calibration method
- **Kadavath et al. (2022)**: LLM confidence via hidden states
- **Kuhn et al. (2024)**: Semantic entropy (Nature)
- **Mielke et al. (2022)**: Layer-wise uncertainty analysis

Special thanks to:
- Thinking Machines Lab for Tinker API integration
- HuggingFace for transformers ecosystem
- The open-source LLM community

---

## 📬 Contact & Support

- **Issues**: [GitHub Issues](https://github.com/.../issues) for bug reports
- **Discussions**: [GitHub Discussions](https://github.com/.../discussions) for questions
- **Primary Contact**: joshc (MIT Deep Learning Team)

---

## 🚦 Getting Started Checklist

- [ ] Install dependencies: `pip install -r requirements.txt`
- [ ] Run quick test: `python experiments/layer_analysis.py --quick --num-samples 100`
- [ ] Review outputs in `outputs/layer_analysis/`
- [ ] Customize config: Edit `configs/linear_probe.yaml`
- [ ] Run full experiment: `python experiments/baseline_linear_probe.py`
- [ ] Explore documentation: Read `CLAUDE.md` and `IMPLEMENTATION.md`
- [ ] (Optional) Setup Tinker: See `TINKER_INTEGRATION.md`

**Ready to probe?** Run your first experiment now! 🚀

```bash
python experiments/layer_analysis.py --quick --num-samples 100
```
