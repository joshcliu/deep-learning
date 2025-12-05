from src.data import MMLUDataset, TriviaQADataset, GSM8KDataset

# Load MMLU (multiple choice)
mmlu = MMLUDataset(split="validation", category="stem")

# Load TriviaQA (open-ended QA)
trivia = TriviaQADataset(split="validation", max_examples=1000)

# Load GSM8K (math problems)
gsm8k = GSM8KDataset(split="test")