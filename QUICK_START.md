# Quick Start Guide

Get started with the LLM Confidence Probing Framework in 3 steps.

## Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

## Step 2: Quick Test (5-10 minutes)

Verify everything works:

```bash
python experiments/layer_analysis.py --quick --num-samples 50
```

This will:
- Load a small sample of MMLU data
- Test quartile layers (5 layers instead of all 32)
- Generate visualizations
- Validate the full pipeline

**Expected output**: Plots in `outputs/layer_analysis/`

## Step 3: Run Full Experiments

### Option A: Layer Analysis (Recommended First)

```bash
python experiments/layer_analysis.py --num-samples 500
```

**What it does**: Tests all layers to find optimal ones for uncertainty
**Time**: 2-4 hours (depends on GPU)
**Output**: Performance plots and heatmaps

### Option B: Baseline Linear Probe

```bash
python experiments/baseline_linear_probe.py
```

**What it does**: Trains probes on specified layers
**Time**: 1-2 hours
**Output**: Trained probe checkpoints + metrics

---

## Common Issues

### Issue: Out of Memory

**Solution**: Use quantization

```bash
# Edit configs/linear_probe.yaml
model:
  quantization: "8bit"  # or "4bit" for very large models
```

### Issue: HuggingFace Token Required

**Solution**: Get token from https://huggingface.co/settings/tokens

```bash
# Edit configs/linear_probe.yaml
model:
  use_auth_token: "hf_YOUR_TOKEN_HERE"
```

### Issue: Slow First Run

**Expected behavior**: First run extracts hidden states (slow)
**Subsequent runs**: Fast (uses cache)

---

## What's Next?

After running baseline experiments:

1. Check `outputs/` for results
2. Review visualizations
3. Read `CLAUDE.md` for Phase 2 roadmap (CCPS, semantic entropy)
4. Modify configs for your use case

---

## File Guide

- `experiments/README.md` - Detailed experiment documentation
- `CLAUDE.md` - Complete project context for AI agents
- `IMPLEMENTATION_COMPLETE.md` - Phase 1 completion summary
- `src/data/DATA_README.md` - Dataset usage guide

---

## Need Help?

- Check experiment docstrings: `--help` flag
- Review example configs in `configs/`
- Read inline documentation in source files
