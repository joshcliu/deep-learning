# Tinker API Integration Guide

**Last Updated**: 2025-11-30
**Status**: Integrated and Ready to Use

This document explains how to use the Tinker API with our LLM Confidence Probing Framework.

---

## Overview

### What is Tinker API?

Tinker is a training API by Thinking Machines Lab for efficient distributed fine-tuning of large language models using LoRA (Low-Rank Adaptation). It provides:

- **Distributed training** on powerful GPU clusters
- **LoRA fine-tuning** for memory-efficient adaptation
- **Low-level control** over training loops
- **Seamless evaluation** via Inspect AI integration

### Integration Architecture

Our framework uses a **hybrid approach**:

1. **Tinker API** for fine-tuning (distributed, efficient)
2. **Download weights** from Tinker as LoRA adapters
3. **Load into HuggingFace** locally via PEFT library
4. **Extract hidden states** for probing (core framework feature)

```
┌─────────────┐
│ Tinker API  │ Fine-tune model with LoRA
└──────┬──────┘
       │
       ▼
┌─────────────┐
│ Download    │ Get LoRA weights as tar archive
│ Weights     │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│ Convert to  │ Transform to PEFT format
│ PEFT        │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│ Load into   │ Use with HuggingFace transformers
│ HuggingFace │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│ Extract     │ Get hidden states for probing
│ Hiddens     │
└─────────────┘
```

---

## Setup

### 1. Install Dependencies

```bash
# Install PEFT for LoRA adapter loading (required)
pip install peft>=0.7.0

# Install Tinker API (optional - only if training/downloading from Tinker)
pip install tinker
```

### 2. Set API Key

```bash
# Set your Tinker API key as environment variable
export TINKER_API_KEY=<your_api_key_here>

# Or add to your .bashrc / .zshrc for persistence
echo 'export TINKER_API_KEY=<your_key>' >> ~/.bashrc
```

### 3. Verify Setup

```python
from src.tinker import TinkerClient

# This will check for API key and test connection
client = TinkerClient()
print("Supported models:", client.get_supported_models())
```

---

## Usage Workflows

### Workflow 1: Download Pre-trained Tinker Model

If you already have a Tinker checkpoint (e.g., from a colleague or previous training):

```python
from src.tinker import download_and_convert_weights
from src.models import ModelLoader, HiddenStateExtractor

# Step 1: Download and convert Tinker weights to PEFT format
peft_adapter_path = download_and_convert_weights(
    tinker_checkpoint="tinker://abc123xyz/sampler_weights/final",
    base_model_name="meta-llama/Llama-3.1-8B",
    output_dir="weights/my_finetuned_model",
    cleanup_archive=True  # Remove tar after extraction
)

# Step 2: Load model with Tinker LoRA adapter
loader = ModelLoader(
    model_name="meta-llama/Llama-3.1-8B",
    tinker_lora_path=peft_adapter_path,
    quantization="8bit"
)
model, tokenizer = loader.load()

# Step 3: Extract hidden states (works exactly like standard models!)
extractor = HiddenStateExtractor(model, tokenizer)
hiddens = extractor.extract(
    texts=["What is the capital of France?"],
    layers=[8, 16, 24, 31],
    cache_dir="cache/tinker_model"
)

print(f"Hidden states shape: {hiddens.shape}")
# Output: (1, 4, 4096) - (texts, layers, hidden_dim)
```

### Workflow 2: Fine-tune with Tinker, Then Probe

Full workflow from fine-tuning to probing:

```python
# === Part 1: Fine-tune with Tinker (run once) ===
import tinker
from src.data import MMLUDataset

# Load dataset
dataset = MMLUDataset(split="train", category="stem")

# Create training client
client = tinker.ServiceClient()
training_client = client.create_training_client("meta-llama/Llama-3.1-8B")

# ... fine-tuning loop (see Tinker documentation) ...

# Save weights
sampler_weights_path = training_client.save_weights_for_sampler("final")
print(f"Weights saved to: {sampler_weights_path}")

# === Part 2: Download and probe ===
from src.tinker import download_and_convert_weights
from src.models import ModelLoader
from src.probes import LinearProbe

# Download weights
peft_path = download_and_convert_weights(
    tinker_checkpoint=sampler_weights_path,
    base_model_name="meta-llama/Llama-3.1-8B",
    output_dir="weights/mmlu_finetuned"
)

# Load and extract hidden states
loader = ModelLoader("meta-llama/Llama-3.1-8B", tinker_lora_path=peft_path)
model, tokenizer = loader.load(quantization="8bit")

from src.models import HiddenStateExtractor
extractor = HiddenStateExtractor(model, tokenizer)

# ... extract hidden states from test set ...

# Train probe on fine-tuned model's hidden states
probe = LinearProbe(input_dim=4096)
probe.fit(hiddens_train, labels_train, hiddens_val, labels_val)
```

### Workflow 3: Compare Base vs Fine-tuned

Compare probing performance on base vs Tinker-fine-tuned models:

```python
from src.models import ModelLoader, HiddenStateExtractor
from src.probes import LinearProbe
import numpy as np

# Extract hidden states from BOTH models
texts = ["Sample question 1", "Sample question 2", ...]
labels = np.array([1, 0, ...])

# === Base model ===
loader_base = ModelLoader("meta-llama/Llama-3.1-8B")
model_base, tokenizer_base = loader_base.load(quantization="8bit")
extractor_base = HiddenStateExtractor(model_base, tokenizer_base)
hiddens_base = extractor_base.extract(texts, layers=[16])

# === Tinker fine-tuned model ===
loader_tuned = ModelLoader(
    "meta-llama/Llama-3.1-8B",
    tinker_lora_path="weights/mmlu_finetuned/peft_adapter"
)
model_tuned, tokenizer_tuned = loader_tuned.load(quantization="8bit")
extractor_tuned = HiddenStateExtractor(model_tuned, tokenizer_tuned)
hiddens_tuned = extractor_tuned.extract(texts, layers=[16])

# Train probes on both
probe_base = LinearProbe(input_dim=4096)
probe_base.fit(hiddens_base[:800], labels[:800], hiddens_base[800:], labels[800:])

probe_tuned = LinearProbe(input_dim=4096)
probe_tuned.fit(hiddens_tuned[:800], labels[:800], hiddens_tuned[800:], labels[800:])

# Compare performance
from src.evaluation import CalibrationMetrics

metrics_base = CalibrationMetrics(
    probe_base.predict(hiddens_base[800:]) > 0.5,
    probe_base.predict(hiddens_base[800:]),
    labels[800:]
)
metrics_tuned = CalibrationMetrics(
    probe_tuned.predict(hiddens_tuned[800:]) > 0.5,
    probe_tuned.predict(hiddens_tuned[800:]),
    labels[800:]
)

print(f"Base Model ECE:        {metrics_base.ece():.4f}")
print(f"Fine-tuned Model ECE:  {metrics_tuned.ece():.4f}")
```

---

## API Reference

### `src.tinker.TinkerClient`

Wrapper for Tinker API with authentication.

```python
from src.tinker import TinkerClient

client = TinkerClient()  # Uses TINKER_API_KEY env var
# or
client = TinkerClient(api_key="your_key")

# Get supported models
models = client.get_supported_models()

# Create training client
training_client = client.create_training_client("meta-llama/Llama-3.1-8B")

# Get checkpoint download URL
url = client.get_checkpoint_archive_url("tinker://path/to/checkpoint")
```

### `src.tinker.download_and_convert_weights`

Download Tinker checkpoint and convert to PEFT format (one-step function).

```python
from src.tinker import download_and_convert_weights

peft_path = download_and_convert_weights(
    tinker_checkpoint="tinker://abc123/sampler_weights/final",
    base_model_name="meta-llama/Llama-3.1-8B",
    output_dir="weights/my_model",
    cleanup_archive=True  # Optional: delete tar after extraction
)
# Returns: "weights/my_model/peft_adapter"
```

### `ModelLoader` with Tinker Support

Load models with Tinker LoRA adapters.

```python
from src.models import ModelLoader

# Standard model
loader = ModelLoader("meta-llama/Llama-3.1-8B")

# Model with Tinker LoRA adapter
loader = ModelLoader(
    model_name="meta-llama/Llama-3.1-8B",
    tinker_lora_path="weights/my_model/peft_adapter"
)

model, tokenizer = loader.load(quantization="8bit")
```

---

## Configuration

### Tinker Settings in Experiment Configs

You can add Tinker-specific settings to your YAML configs:

```yaml
# configs/tinker_probe.yaml
experiment:
  name: "tinker-finetuned-probe"
  seed: 42

model:
  name: "meta-llama/Llama-3.1-8B"
  quantization: "8bit"
  tinker_lora_path: "weights/mmlu_finetuned/peft_adapter"  # Optional

tinker:
  enabled: true
  checkpoint_path: "tinker://abc123/sampler_weights/final"
  download_weights: true
  output_dir: "weights/mmlu_finetuned"

extraction:
  layers: [8, 16, 24, 31]
  cache_dir: "cache/tinker_model"

# ... rest of config ...
```

---

## Supported Models

Tinker currently supports:

**Meta-Llama**:
- `meta-llama/Llama-3.1-8B`
- `meta-llama/Llama-3.1-70B`

**Qwen**:
- `Qwen/Qwen2.5-7B`
- `Qwen/Qwen2.5-14B`
- `Qwen/Qwen2.5-72B`
- `Qwen/Qwen2.5-235B-A22B-Instruct-2507` (MoE)
- `Qwen/Qwen2.5-30B-A3B-Base` (MoE)

Check current list:
```python
from src.tinker import TinkerClient
print(TinkerClient().get_supported_models())
```

---

## Troubleshooting

### Issue: "TINKER_API_KEY not found"

**Solution**: Set environment variable

```bash
export TINKER_API_KEY=your_key
# Or in Python
import os
os.environ["TINKER_API_KEY"] = "your_key"
```

### Issue: "PEFT library not installed"

**Solution**: Install PEFT

```bash
pip install peft>=0.7.0
```

### Issue: "adapter_config.json not found"

**Cause**: Tinker archive doesn't include adapter config
**Solution**: Our code auto-generates a default config. If you need custom LoRA parameters, manually edit `weights/your_model/peft_adapter/adapter_config.json`.

### Issue: Downloaded weights don't work with base model

**Cause**: Base model mismatch
**Solution**: Ensure `base_model_name` in `download_and_convert_weights()` matches the model used for training in Tinker.

---

## Best Practices

### 1. Cache Tinker-Downloaded Weights

Once downloaded and converted, Tinker weights are local PEFT adapters. Cache them:

```python
# Download once
peft_path = download_and_convert_weights(...)

# Reuse multiple times
loader1 = ModelLoader("meta-llama/Llama-3.1-8B", tinker_lora_path=peft_path)
loader2 = ModelLoader("meta-llama/Llama-3.1-8B", tinker_lora_path=peft_path)
```

### 2. Use Quantization with Tinker Models

Tinker-trained models can be loaded with quantization for memory efficiency:

```python
loader = ModelLoader(
    "meta-llama/Llama-3.1-8B",
    tinker_lora_path=peft_path,
    quantization="8bit"  # Reduces memory by ~2x
)
```

### 3. Organize Weights by Experiment

```
weights/
├── baseline/              # Base models
├── tinker_mmlu_stem/      # Tinker model trained on MMLU STEM
├── tinker_math_tuned/     # Tinker model trained on GSM8K
└── tinker_combined/       # Tinker model trained on mixed data
```

### 4. Version Control Checkpoints

Keep track of which Tinker checkpoint corresponds to which experiment:

```python
# Save checkpoint info
import json

checkpoint_info = {
    "tinker_path": "tinker://abc123/sampler_weights/final",
    "base_model": "meta-llama/Llama-3.1-8B",
    "training_dataset": "MMLU-STEM",
    "training_date": "2025-11-30",
    "local_path": "weights/mmlu_stem_tuned/peft_adapter"
}

with open("weights/mmlu_stem_tuned/checkpoint_info.json", "w") as f:
    json.dump(checkpoint_info, f, indent=2)
```

---

## Limitations

### What Tinker CAN do:
✅ Efficient distributed LoRA fine-tuning
✅ Logprob extraction for sequences
✅ Evaluation via Inspect AI
✅ Download trained weights

### What Tinker CANNOT do (but our framework can):
❌ **Hidden state extraction** (use HuggingFace after downloading)
❌ Direct access to activations/internals
❌ Arbitrary hooks during forward pass

**Solution**: Our hybrid approach handles this by downloading weights and loading them locally!

---

## Examples

See `experiments/` for complete examples:

- `experiments/tinker_download_example.py` - Download and convert weights
- `experiments/tinker_probe_comparison.py` - Compare base vs fine-tuned models

---

## Additional Resources

- **Tinker Documentation**: https://tinker-docs.thinkingmachines.ai/
- **Tinker Cookbook**: https://github.com/thinking-machines-lab/tinker-cookbook
- **PEFT Documentation**: https://huggingface.co/docs/peft/
- **Our Integration**: `src/tinker/` module

---

## FAQ

**Q: Do I need Tinker API to use this framework?**
A: No! Tinker integration is optional. The framework works perfectly with standard HuggingFace models.

**Q: Can I probe Tinker-trained models?**
A: Yes! Download the weights, convert to PEFT format, and use exactly like any other model.

**Q: Does hidden state extraction work with Tinker models?**
A: Yes, once you download and load the LoRA adapter locally via PEFT.

**Q: Can I use Tinker for inference only (no training)?**
A: If you have a Tinker checkpoint, you can download it and use for inference + probing without training.

**Q: What's the cost of using Tinker API?**
A: Contact Thinking Machines Lab for pricing. Currently in private beta with free early access.

---

**Questions or issues?** Check the main documentation in `CLAUDE.md` or open an issue on GitHub.
