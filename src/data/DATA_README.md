# Dataset Loaders Documentation

This module provides standardized loaders for common LLM benchmarks used in confidence probing research.

## Quick Start

```python
from src.data import MMLUDataset, TriviaQADataset, GSM8KDataset

# Load MMLU (multiple choice)
mmlu = MMLUDataset(split="validation", category="stem")

# Load TriviaQA (open-ended QA)
trivia = TriviaQADataset(split="validation", max_examples=1000)

# Load GSM8K (math problems)
gsm8k = GSM8KDataset(split="test")
```

## Datasets

### MMLU (Massive Multitask Language Understanding)

**What it is:** 57 subjects spanning STEM, humanities, social sciences. Each question has 4 multiple choice options.

**Usage:**
```python
# Load all subjects
dataset = MMLUDataset(split="validation")

# Load specific category
dataset = MMLUDataset(split="validation", category="stem")

# Load specific subjects
dataset = MMLUDataset(
    split="validation", 
    subjects=["anatomy", "philosophy", "astronomy"]
)

# Get available subjects
from src.data import MMLU_SUBJECTS
print(MMLU_SUBJECTS["stem"])  # List of STEM subjects
```

**Splits:** `auxiliary_train`, `validation`, `test`, `dev`

**Categories:** `stem`, `humanities`, `social_sciences`, `other`

**Example workflow:**
```python
dataset = MMLUDataset(split="validation", category="stem")

for example in dataset:
    # Format as prompt
    prompt = example.format_prompt(style="multiple_choice")
    
    # Get model to choose A, B, C, or D
    model_choice = your_model.generate(prompt)  # e.g., "B"
    predicted_idx = ord(model_choice) - ord('A')  # Convert to 0,1,2,3
    
    # Check correctness
    is_correct = (predicted_idx == example.answer)
    
    # For confidence probing, format with answer
    text_with_answer = example.format_with_answer(
        example.choices[predicted_idx]
    )
    # Extract hidden states from text_with_answer
```

### TriviaQA

**What it is:** Open-domain question answering with multiple acceptable answers per question.

**Usage:**
```python
dataset = TriviaQADataset(
    split="validation",
    subset="unfiltered",  # or "rc" for reading comprehension
    max_examples=1000  # Optional: limit dataset size
)

# Check if answer is correct
example = dataset[0]
model_answer = "Paris"
is_correct = dataset.is_correct(example, model_answer, fuzzy_match=True)
```

**Splits:** `train`, `validation`, `test`

**Subsets:** 
- `unfiltered`: Full dataset (95K examples)
- `rc`: Reading comprehension subset with evidence documents

**Example workflow:**
```python
from src.data import TriviaQADataset, generate_trivia_labels

dataset = TriviaQADataset(split="validation", max_examples=100)

# Generate answers with your model
model_answers = []
for example in dataset:
    prompt = f"Q: {example.question}\nA:"
    answer = your_model.generate(prompt)
    model_answers.append(answer)

# Generate labels (1=correct, 0=incorrect)
labels = generate_trivia_labels(dataset, model_answers, fuzzy_match=True)

# Now extract hidden states for correct vs incorrect answers
for example, answer, label in zip(dataset, model_answers, labels):
    text = f"Q: {example.question} A: {answer}"
    # Extract hidden states...
```

### GSM8K (Grade School Math 8K)

**What it is:** Grade school math word problems requiring multi-step reasoning. Each has a numerical answer.

**Usage:**
```python
dataset = GSM8KDataset(
    split="test",
    max_examples=100  # Optional: limit dataset size
)

# Check if answer is correct
example = dataset[0]
model_answer = "The answer is 42"
is_correct = dataset.is_correct(example, model_answer, tolerance=1e-4)
```

**Splits:** `train`, `test`

**Example workflow:**
```python
from src.data import GSM8KDataset, generate_gsm8k_labels

dataset = GSM8KDataset(split="test", max_examples=50)

# Generate answers with your model
model_answers = []
for example in dataset:
    prompt = example.question
    answer = your_model.generate(prompt)  # Should include reasoning + final number
    model_answers.append(answer)

# Generate labels
labels = generate_gsm8k_labels(dataset, model_answers, tolerance=1e-4)

# Extract hidden states
for example, answer, label in zip(dataset, model_answers, labels):
    text = f"{example.question} {answer}"
    # Extract hidden states...
```

## Common Patterns

### Standard Format

All datasets return `DatasetExample` objects with:
```python
example.question      # Question text
example.choices       # List of answer choices
example.answer        # Correct answer index (for MMLU) or 0 (for others)
example.metadata      # Dataset-specific metadata
```

### Formatting Prompts

```python
# Multiple choice format (for MMLU)
prompt = example.format_prompt(style="multiple_choice")
# Output: "Q: What is...?\n(A) Option 1\n(B) Option 2\n..."

# Simple Q&A format
prompt = example.format_prompt(style="qa")
# Output: "Q: What is...?\nA:"

# Chain-of-thought format
prompt = example.format_prompt(style="cot")
# Output: "Q: What is...?\n...\nLet's think step by step.\nAnswer:"
```

### Creating Training Data for Probes

```python
# Step 1: Generate model answers
examples = []
labels = []

for example in dataset:
    # Get model prediction
    model_answer = your_model.generate(example.question)
    
    # Determine correctness
    if isinstance(dataset, MMLUDataset):
        predicted_idx = parse_choice(model_answer)  # Parse "A", "B", etc.
        is_correct = (predicted_idx == example.answer)
    elif isinstance(dataset, TriviaQADataset):
        is_correct = dataset.is_correct(example, model_answer)
    elif isinstance(dataset, GSM8KDataset):
        is_correct = dataset.is_correct(example, model_answer)
    
    # Format for hidden state extraction
    text = f"Q: {example.question} A: {model_answer}"
    examples.append(text)
    labels.append(1 if is_correct else 0)

# Step 2: Extract hidden states
from src.models import HiddenStateExtractor
extractor = HiddenStateExtractor(model, tokenizer)
hiddens = extractor.extract(examples, layers=[16])

# Step 3: Train probe
from src.probes import LinearProbe
probe = LinearProbe(input_dim=hiddens.shape[-1])
probe.fit(hiddens, labels)
```

## Dataset Statistics

Get information about loaded dataset:

```python
stats = dataset.get_statistics()

# Common to all datasets
print(stats["size"])          # Number of examples
print(stats["split"])         # Which split was loaded

# MMLU-specific
print(stats["num_subjects"])
print(stats["examples_per_category"])

# TriviaQA-specific
print(stats["avg_gold_answers"])  # Average acceptable answers per question

# GSM8K-specific
print(stats["avg_answer"])        # Average numerical answer
print(stats["negative_answers"])  # Count of negative numbers
```

## Advanced Usage

### Filtering Examples

```python
# Filter by metadata
stem_physics = dataset.filter(
    lambda ex: ex.metadata.get("subject") == "high_school_physics"
)

# Filter by question length
short_questions = dataset.filter(
    lambda ex: len(ex.question.split()) < 20
)
```

### Sampling

```python
# Random sample
sample = dataset.sample(n=100, seed=42)

# Get specific indices
subset = [dataset[i] for i in [0, 10, 20, 30]]
```

### Batch Processing

```python
from torch.utils.data import DataLoader

# Wrap in PyTorch DataLoader
def collate_fn(batch):
    questions = [ex.question for ex in batch]
    labels = [ex.answer for ex in batch]
    return questions, labels

loader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
    collate_fn=collate_fn
)

for questions, labels in loader:
    # Process batch...
    pass
```

## Implementation Notes

### Answer Matching for Open-Ended QA

**TriviaQA:** Uses flexible matching with normalization
- Case insensitive
- Removes articles (a, an, the)
- Removes punctuation
- Checks substring matches

**GSM8K:** Extracts numbers with multiple strategies
- Looks for "#### NUMBER" format
- Looks for "The answer is X" patterns
- Falls back to last number in text

### Caching

Datasets are loaded from HuggingFace's `datasets` library, which automatically caches downloads in `~/.cache/huggingface/datasets/`

### Memory Considerations

- **MMLU:** ~5MB for all subjects
- **TriviaQA:** ~50MB for unfiltered subset
- **GSM8K:** ~3MB for full dataset

Use `max_examples` parameter to limit memory for quick testing.

## Extending with New Datasets

To add a new dataset:

1. Create `src/data/your_dataset.py`
2. Inherit from `BaseDataset`
3. Implement `_load_data()` method
4. Convert to `DatasetExample` format
5. Export in `src/data/__init__.py`

Example:
```python
from .base import BaseDataset, DatasetExample

class YourDataset(BaseDataset):
    def _load_data(self) -> List[DatasetExample]:
        # Load from source
        raw_data = load_dataset("your_dataset")
        
        # Convert to standard format
        examples = []
        for item in raw_data:
            example = DatasetExample(
                question=item["question"],
                choices=item["options"],
                answer=item["correct_idx"],
                metadata={"source": "your_dataset"}
            )
            examples.append(example)
        
        return examples
```

## Testing

Run the example script to verify all datasets load correctly:

```bash
python src/data/dataset_usage_examples.py
```

This will test:
- Loading each dataset
- Formatting prompts
- Checking answer correctness
- Generating statistics