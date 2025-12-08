# Research Report Outline: LLM Confidence Probing

## Title Options
- "Probing the Mind of LLMs: Extracting Confidence from Hidden States"
- "Does Your LLM Know What It Doesn't Know? A Probing Approach"
- "Multi-Source Confidence Estimation: Combining Internal Uncertainty with Expressed Confidence in LLMs"

---

## 1. Introduction & Motivation (1-2 pages)

### 1.1 The Problem: LLMs Are Confidently Wrong
- LLMs generate plausible-sounding but incorrect answers (hallucinations)
- Output probabilities don't reflect true confidence
- Critical for deployment in high-stakes domains (medical, legal, finance)

### 1.2 The Key Insight
- **Hidden states contain rich uncertainty information** that's lost by final output
- Middle layers (50-75% depth) encode more uncertainty than final layers
- The model may "know" it's uncertain internally, even when it sounds confident

### 1.3 Our Contributions
1. Comprehensive probe architecture comparison (13+ architectures)
2. Novel **MultiSourceConfidenceNetwork** combining hidden states + logits
3. Key insight: Internal uncertainty vs. expressed confidence can diverge
4. Open-source framework for LLM confidence probing

---

## 2. Background & Related Work (1-2 pages)

### 2.1 Uncertainty Quantification in LLMs
- Three paradigms: probing, consistency-based, verbalized confidence
- CCPS achieves 55% ECE reduction (Burns et al. 2025)
- Semantic entropy for meaning-level uncertainty (Kuhn et al., Nature 2024)

### 2.2 Linear Probing for Representation Analysis
- Origins in Hewitt & Manning (2019) structural probes
- Why linear probes: interpretability, low memorization risk, fast training

### 2.3 Calibration Metrics
- **ECE** (Expected Calibration Error): Gap between confidence and accuracy
- **Brier Score**: Proper scoring rule combining calibration + discrimination
- **AUROC**: Discrimination quality (separating correct from incorrect)

---

## 3. Methodology (2-3 pages)

### 3.1 Experimental Setup
- **Model**: Qwen2.5-7B (28 layers, 3584 hidden dim)
- **Dataset**: MMLU validation set (multiple-choice QA, 57 subjects)
- **Task**: Predict whether model's answer is correct from hidden states

### 3.2 Hidden State Extraction
- Extract from quartile layers: [L/4, L/2, 3L/4, L-1]
- Last token position (captures full context via attention)
- 8-bit quantization for efficiency

### 3.3 Training Protocol
- **Loss**: Brier score (properly penalizes overconfident mistakes)
- **Training**: 200 epochs, cosine annealing LR, no early stopping
- **Split**: 60% train, 20% val, 20% test

### 3.4 Probe Architectures Tested

| Architecture | Key Idea | Parameters |
|--------------|----------|------------|
| Linear | Simplest baseline | ~4K |
| MLP | Non-linear features | ~1M |
| Attention | Self-attention over hidden chunks | ~500K |
| Sparse | Learn dimension importance | ~500K |
| MultiHead | Multiple expert predictions | ~200K |
| Hierarchical | Multi-scale (token→span→semantic→global) | ~2M |
| LayerEnsemble | Ensemble across layers | ~300K |
| **MultiSource (Ours)** | Hidden states + logits fusion | ~100K |

---

## 4. Novel Architecture: MultiSourceConfidenceNetwork (1-2 pages)

### 4.1 Motivation
- **Hypothesis**: Internal uncertainty (hidden states) may differ from expressed confidence (logits)
- A model might be internally uncertain but output high-confidence logits
- Combining both sources can detect miscalibration

### 4.2 Architecture Design
```
Hidden States (4 layers) ──→ Per-layer probes ──→ Cross-layer attention ──┐
                                                                          │──→ Fusion ──→ Confidence
Output Logits (A,B,C,D) ──→ Feature extraction ──→ Processed features ────┘
                            (entropy, margin, max_prob)
```

### 4.3 Key Components
1. **Per-layer probes**: Lightweight MLP per quartile layer
2. **Cross-layer attention**: Layers attend to each other
3. **Logit features**: Entropy, margin, max probability, softmax probs
4. **Learnable layer weights**: Which layers matter most?

### 4.4 What It Can Detect
- **Overconfidence**: High logits + uncertain hidden states
- **Underconfidence**: Low logits + confident hidden states
- **Well-calibrated**: Agreement between sources

---

## 5. Experimental Results (2-3 pages)

### 5.1 Architecture Comparison
*(Include experimental results table here)*

| Architecture | AUROC | Brier | ECE |
|--------------|-------|-------|-----|
| Linear | 0.XXX | 0.XXXX | 0.XXXX |
| LayerEnsemble | 0.XXX | 0.XXXX | 0.XXXX |
| MultiSource | 0.XXX | 0.XXXX | 0.XXXX |

### 5.2 Key Finding: Middle Layers Are Optimal
- Learned layer weights from MultiSource/LayerEnsemble
- Validation of prior research (Mielke et al., Kadavath et al.)

### 5.3 Reliability Diagrams
- Visual comparison of calibration across architectures
- Does MultiSource improve calibration?

### 5.4 Ablation: Do Logits Help?
- Compare MultiSource (with logits) vs LayerEnsemble (without)
- When does combining sources help?

---

## 6. Analysis & Discussion (1-2 pages)

### 6.1 Why Simple Probes Often Suffice
- Linear separability of correct/incorrect in hidden space
- Low intrinsic dimensionality of uncertainty signal
- PCA/t-SNE visualizations

### 6.2 When Complex Architectures Help
- High intrinsic dimensionality tasks
- Multi-scale uncertainty (different aspects at different scales)
- Detecting subtle miscalibration

### 6.3 The Logits Question
- Does the model's expressed confidence (logits) add information?
- Or is everything already in the hidden states?
- Implications for deployment

### 6.4 Limitations
- Single model (Qwen2.5-7B) - generalization to other models?
- MMLU only - different domains may behave differently
- Argmax evaluation vs. generation-based

---

## 7. Broader Impact & Applications (0.5-1 page)

### 7.1 Practical Applications
- **Selective prediction**: Abstain when uncertain
- **Human-in-the-loop**: Flag low-confidence responses for review
- **Retrieval augmentation**: Trigger RAG when confidence is low

### 7.2 Safety Implications
- Detecting hallucinations before deployment
- Calibrated confidence for high-stakes domains
- Understanding when models "know they don't know"

---

## 8. Future Work (0.5 page)

1. **Cross-model transfer**: Do probes generalize across architectures?
2. **CCPS implementation**: Perturbation-based calibration
3. **Causal circuit tracing**: Where does uncertainty computation happen?
4. **VLM extension**: Multi-modal uncertainty quantification
5. **Real-time confidence steering**: Intervene during generation

---

## 9. Conclusion (0.5 page)

- Summary of key findings
- Contribution: Open-source framework + novel MultiSource architecture
- The broader insight: Internal and expressed confidence can diverge
- Call to action: Better uncertainty quantification for safer AI

---

## Appendix

### A. Implementation Details
- Training hyperparameters
- Architecture configurations
- Computational requirements

### B. Additional Results
- Per-subject MMLU performance
- Training curves
- Extended ablations

### C. Code Availability
- GitHub repository link
- Colab notebooks for reproduction

---

## Suggested Figures

1. **Figure 1**: Overview diagram of probe pipeline (input → hidden states → probe → confidence)
2. **Figure 2**: MultiSourceConfidenceNetwork architecture diagram
3. **Figure 3**: Bar chart comparing AUROC/Brier across architectures
4. **Figure 4**: Reliability diagrams (3-panel: Linear, LayerEnsemble, MultiSource)
5. **Figure 5**: Learned layer weights visualization
6. **Figure 6**: PCA/t-SNE of hidden states (correct vs incorrect)
7. **Figure 7**: Training curves showing convergence

---

## Key References

1. Burns et al. (2025) - CCPS: 55% ECE reduction via perturbation stability
2. Kuhn et al. (2024) - Semantic entropy for hallucination detection (Nature)
3. Kadavath et al. (2022) - LLM confidence via hidden states
4. Hewitt & Manning (2019) - Structural probes for linguistic knowledge
5. Mielke et al. (2022) - Layer-wise uncertainty analysis

---

## Writing Checklist

- [ ] Run experiments and fill in results tables
- [ ] Create all figures
- [ ] Write abstract (last)
- [ ] Get feedback from group members
- [ ] Proofread and format
