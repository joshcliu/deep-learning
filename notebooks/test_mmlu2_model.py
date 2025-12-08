#!/usr/bin/env python3
import sys
import torch
import numpy as np

# =============================================================================
# 0. PATH SETUP — UPDATE THIS TO WHERE YOU CLONED THE REPO
# =============================================================================
import sys, os
sys.path.insert(0, "/orcd/home/002/crli9772/deep-learning")


# =============================================================================
# 1. PATCH FOR BFLOAT16 / 8-BIT COMPATIBILITY
# =============================================================================
from src.models import extractor as extractor_module

_original_extract_batch = extractor_module.HiddenStateExtractor._extract_batch

def _patched_extract_batch(self, texts, layers, max_length, token_position):
    encodings = self.tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    encodings = {k: v.to(self.device) for k, v in encodings.items()}

    with torch.no_grad():
        outputs = self.model(
            **encodings,
            output_hidden_states=True,
            return_dict=True,
        )

    hidden_states = outputs.hidden_states
    batch_hiddens = []

    for layer_idx in layers:
        layer_h = hidden_states[layer_idx + 1]

        if token_position == "cls":
            token_h = layer_h[:, 0, :]
        elif token_position == "last":
            attention_mask = encodings["attention_mask"]
            seq_lengths = attention_mask.sum(dim=1) - 1
            token_h = layer_h[
                torch.arange(layer_h.size(0), device=self.device),
                seq_lengths
            ]
        elif token_position == "mean":
            attention_mask = encodings["attention_mask"].unsqueeze(-1)
            masked = layer_h * attention_mask
            token_h = masked.sum(dim=1) / attention_mask.sum(dim=1)
        else:
            raise ValueError(f"Unsupported token_position: {token_position}")

        token_h = token_h.detach().cpu().to(torch.float32).numpy()
        batch_hiddens.append(token_h)

    return np.stack(batch_hiddens, axis=1)

extractor_module.HiddenStateExtractor._extract_batch = _patched_extract_batch
print("Patched HiddenStateExtractor for bfloat16 / 8-bit compatibility.\n")


# =============================================================================
# 2. LOAD MODEL
# =============================================================================
from src.models import ModelLoader, HiddenStateExtractor

print("Loading model...")
model_name = "Qwen/Qwen2.5-7B"
loader = ModelLoader(model_name)

model, tokenizer = loader.load(
    quantization="8bit",
    device_map="auto"
)

print(f"Loaded {model_name}")
print(f"Layers: {loader.config.num_layers}, Hidden dim: {loader.config.hidden_dim}\n")


# =============================================================================
# 3. LOAD MMLU-PRO DATASET
# =============================================================================
from datasets import load_dataset

print("Loading MMLU-Pro from HuggingFace...")
dataset = load_dataset("TIGER-Lab/MMLU-Pro", split="test")

print("Dataset columns:", dataset.column_names)
print("Example row:", dataset[0])

class Example:
    def __init__(self, row):
        self.question = row["question"]
        self.options = row["options"]
        self.answer_letter = row["answer"]
        self.answer_index = int(row["answer_index"])
        self.category = row.get("category", None)

    def get_correct_letter(self):
        return self.answer_letter.upper()

    def get_correct_option(self):
        return self.options[self.answer_index]

    def format_prompt(self, style="multiple_choice"):
        options_str = "\n".join(
            f"{chr(65+i)}. {opt}" for i, opt in enumerate(self.options)
        )
        return f"{self.question}\n{options_str}\n\nAnswer:"

examples = [Example(row) for row in dataset]

NUM_SAMPLES = 5
examples = examples[:NUM_SAMPLES]
print(f"Using {len(examples)} examples.\n")


# =============================================================================
# 4. GENERATE MODEL ANSWERS
# =============================================================================
from tqdm import tqdm

def generate_answer(model, tokenizer, prompt, max_new_tokens=64):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=0,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(
        outputs[0][inputs.input_ids.shape[1]:],
        skip_special_tokens=True
    ).strip()

def check_correctness(answer, ex: Example):
    answer_upper = answer.upper()
    if ex.answer_letter in answer_upper:
        return True
    if ex.get_correct_option().lower() in answer.lower():
        return True
    return False

print("Generating answers...")
prompts = []
generated = []
correctness = []

for ex in tqdm(examples):
    prompt = ex.format_prompt()
    ans = generate_answer(model, tokenizer, prompt)
    prompts.append(prompt)
    generated.append(ans)
    correctness.append(check_correctness(ans, ex))

correctness = np.array(correctness)
print(f"\nAccuracy: {correctness.mean():.3f}  ({correctness.sum()}/{len(correctness)})\n")


# =============================================================================
# 5. EXTRACT HIDDEN STATES
# =============================================================================
num_layers = loader.config.num_layers
LAYERS = [
    num_layers // 4,
    num_layers // 2,
    3 * num_layers // 4,
    num_layers - 1,
]

print(f"Extracting from layers: {LAYERS} (out of {num_layers})")

extractor = HiddenStateExtractor(model, tokenizer)
hidden_states = extractor.extract(
    texts=prompts,
    layers=LAYERS,
    batch_size=8,
    show_progress=True,
)

X = hidden_states.reshape(hidden_states.shape[0], -1)
y = correctness

print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}\n")


# =============================================================================
# 6. SAVE OUTPUTS
# =============================================================================
import json

save_dir = "outputs/mmlu_pro_qwen"
os.makedirs(save_dir, exist_ok=True)

np.save(os.path.join(save_dir, "X_hidden_states.npy"), X)
np.save(os.path.join(save_dir, "correctness.npy"), y)

with open(os.path.join(save_dir, "prompts.jsonl"), "w") as f:
    for p in prompts:
        f.write(json.dumps({"prompt": p}) + "\n")

with open(os.path.join(save_dir, "generated_answers.jsonl"), "w") as f:
    for example, gen_ans in zip(examples, generated):
        f.write(json.dumps({
            "prompt": example.format_prompt(),
            "generated_answer": gen_ans,
            "correct_answer": example.get_correct_option(),
            "correct_letter": example.get_correct_letter(),
            "options": example.options,
            "category": example.category,
        }) + "\n")

print("\nAll outputs saved successfully.")
