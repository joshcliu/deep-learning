# Quick Start Guide: Hierarchical Uncertainty Probing

## What Does This Code Do?

This project helps you **measure how confident a large language model (LLM) is when it gives an answer** - and more importantly, **whether that confidence is accurate**.

### The Big Picture

When an LLM like GPT-4 or Llama answers a question, it might be:
- **Right and confident** ✓ (good!)
- **Right but uncertain** (okay, but could be better)
- **Wrong but confident** ✗ (dangerous! This is hallucination)
- **Wrong and uncertain** (at least it knows it doesn't know)

Your code helps **detect when models are overconfident** so you can trust them more safely.

## How Does It Work?

### Step-by-Step Process

```
1. Ask LLM a question → "What is the capital of France?"
2. LLM answers → "Paris" (correct!)
3. Extract hidden states → Get the internal "thoughts" (4096-dimensional vectors)
4. Train a probe → Learn to predict: "Will this answer be right or wrong?"
5. Evaluate calibration → Check if confidence scores match actual correctness
```

### Your Novel Contribution: Hierarchical Probing

Instead of just looking at the final answer, your **hierarchical probe** examines uncertainty at FOUR levels:

1. **Token-level**: Individual word confidence ("Is 'Paris' the right word?")
2. **Span-level**: Phrase confidence ("Is 'capital of France' understood correctly?")
3. **Semantic-level**: Meaning confidence ("Does the model grasp the question?")
4. **Global-level**: Overall answer confidence ("Is the final answer correct?")

This multi-scale approach gives you **richer uncertainty estimates** than existing methods.

## What Code Have You Written?

### Core Files

```
src/probes/
├── base.py           # Base class all probes inherit from
├── linear.py         # Simple baseline probe (from main branch)
└── hierarchical.py   # YOUR NOVEL CONTRIBUTION ⭐
```

### Your Hierarchical Probe (`hierarchical.py`)

**8+ million trainable parameters** organized into:
- `TokenLevelProbe`: MLP for per-token uncertainty
- `SpanLevelProbe`: Attention pooling + MLP for phrase uncertainty
- `SemanticLevelProbe`: MLP for meaning-level uncertainty
- `AttentionPooling`: Learns which tokens to focus on
- Cross-layer fusion: Combines info from multiple LLM layers

## How to Actually Run This

### Option 1: Quick Test (Verifies Code Works)

```bash
# Just run the integration test
python test_integration.py
```

**What this does:**
- Creates fake data (random numbers)
- Trains probes for 5 epochs
- Verifies everything compiles and runs
- **Takes ~30 seconds**

**What you get:**
```
✓ ALL TESTS PASSED!
✓ LinearProbe: 4,098 parameters
✓ HierarchicalProbe: 10,330,886 parameters
```

### Option 2: Real Experiment (Actual Research)

This is what you'd run to get **publishable results**:

```bash
# 1. Make sure you have the requirements
pip install transformers datasets torch scikit-learn tqdm

# 2. Run the hierarchical probe experiment
python experiments/hierarchical_probe.py --config configs/hierarchical_probe.yaml --debug
```

**What this does (if you had a GPU):**
1. Downloads Llama-3.1-8B model (~16 GB)
2. Loads MMLU dataset (57 subjects, 14k questions)
3. Generates model answers
4. Extracts hidden states from layers 8, 16, 24, 31
5. Trains hierarchical probe to predict correctness
6. Evaluates calibration metrics (ECE, Brier, AUROC)
7. Compares against baseline linear probe

**Time estimate:** 4-8 hours on a good GPU

## What Results Do You Get?

### Calibration Metrics (What You're Measuring)

**1. ECE (Expected Calibration Error)** - Lower is better
- Measures: Does 70% confidence mean 70% accurate?
- Good: < 0.05
- Your goal: Beat baseline linear probe

**2. Brier Score** - Lower is better
- Measures: Overall prediction quality
- Range: 0.0 (perfect) to 1.0 (worst)

**3. AUROC (Area Under ROC Curve)** - Higher is better
- Measures: Can you distinguish correct from incorrect answers?
- Range: 0.5 (random) to 1.0 (perfect)
- Target: > 0.75

### Example Output

```
Method           | ECE    | Brier  | AUROC  | Accuracy | Params
-----------------------------------------------------------------
Linear           | 0.0823 | 0.1245 | 0.7234 | 68.3%    | 4,098
Hierarchical     | 0.0612 | 0.1104 | 0.7891 | 68.3%    | 10,330,886

Hierarchical ECE improvement over Linear: 25.6%  ← THIS IS YOUR CONTRIBUTION!
```

## What This Means for Your Research

### Why This Matters

Current LLMs hallucinate **15-30% of the time** but express high confidence. Your work helps:
1. **Detect hallucinations** before they cause harm
2. **Build safer AI systems** (medical, legal, financial applications)
3. **Understand uncertainty** at multiple levels of abstraction

### Your Novel Contribution

**Multi-scale hierarchical probing** is new. Existing work uses:
- Single-layer probes (your baseline)
- Final-token-only approaches

You're exploring **how uncertainty propagates** through linguistic structure.

### Publication Potential

If hierarchical beats linear by **>20% ECE reduction**, that's:
- **ICLR/NeurIPS** workshop material (definitely)
- **Main conference** material (if combined with ablations + analysis)

You'd want to add:
- Ablation studies (which level matters most?)
- Visualization (heatmaps of token-level uncertainty)
- Cross-model generalization (test on Mistral, Qwen)

## Troubleshooting

### "I don't have a GPU"

You have 3 options:
1. **Use Google Colab** (free GPU for limited time)
2. **Use smaller model** (Llama-3.1-8B → Llama-2-7B with 4-bit quantization)
3. **Use pre-extracted hidden states** (if your team has them)

### "ModuleNotFoundError: No module named 'transformers'"

```bash
pip install -r requirements.txt
```

### "CUDA out of memory"

In `configs/hierarchical_probe.yaml`:
```yaml
model:
  quantization: "4bit"  # Change from 8bit to 4bit
data:
  batch_size: 4  # Reduce from 16
  num_samples: 1000  # Limit dataset size for testing
```

## Next Steps

### Immediate (To Verify Everything Works)
1. ✅ Run `python test_integration.py` ← You can do this RIGHT NOW
2. Make sure all tests pass

### Short-term (If You Have GPU Access)
1. Run experiment with `--debug` flag (uses only 100 samples)
2. Check that you can train and get metrics
3. Verify hierarchical improves over linear

### Research (For Your Paper)
1. Full MMLU run (14k examples)
2. Ablation studies (test each hierarchy level)
3. Visualization (plot token-level uncertainty)
4. Cross-model testing (Llama vs Mistral vs Qwen)
5. Write up results for NeurIPS workshop

## Summary

✅ **Your code is committed** to the `maureen` branch on GitHub
✅ **It compiles and runs** (verified by integration test)
✅ **It's mergeable** with main (compatible architecture)

**What you built:**
- Novel hierarchical uncertainty probe
- 10M+ parameter neural network
- Multi-scale calibration system

**What it does:**
- Predicts when LLMs will be wrong
- Measures confidence calibration
- Operates at 4 levels of granularity

**How to use it:**
- Quick test: `python test_integration.py`
- Real experiment: `python experiments/hierarchical_probe.py --debug`

**Why it matters:**
- Safer AI systems
- Better hallucination detection
- Publication-worthy research

---

Questions? Check `IMPLEMENTATION.md` for architecture details or `RESEARCH.md` for related work.
