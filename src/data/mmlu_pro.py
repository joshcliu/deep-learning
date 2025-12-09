"""
MMLU-Pro dataset loader.

MMLU-Pro extends MMLU with:
- 10 answer choices instead of 4 (A-J)
- More challenging, reasoning-focused questions
- 12,000 questions across 14 disciplines
"""

import random
from typing import List, Optional
from datasets import load_dataset
from .base import BaseDataset, DatasetExample


class MMLUProDataset(BaseDataset):
    """
    MMLU-Pro: A More Robust and Challenging Multi-Task Language Understanding Benchmark

    Paper: https://arxiv.org/abs/2406.01574
    HuggingFace: https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro

    Key differences from MMLU:
    - 10 answer choices (A-J) instead of 4
    - Harder questions requiring more reasoning
    - 14 major categories vs 57 fine-grained subjects
    """

    def __init__(self, split: str = "test", category: Optional[str] = None):
        """
        Initialize MMLU-Pro dataset.

        Args:
            split: Dataset split ("test" or "validation")
            category: Optional category filter. Available categories:
                     - business
                     - law
                     - psychology
                     - biology
                     - chemistry
                     - history
                     - math
                     - physics
                     - economics
                     - engineering
                     - philosophy
                     - other
                     - health
                     - computer science
        """
        self.split = split
        self.category = category

        # Load from HuggingFace
        # MMLU-Pro only has 'test' and 'validation' splits
        dataset = load_dataset("TIGER-Lab/MMLU-Pro", split=split)

        # Filter by category if specified
        if category:
            dataset = dataset.filter(lambda x: x["category"] == category)

        self.examples = []
        for item in dataset:
            example = DatasetExample(
                question=item["question"],
                choices=item["options"],  # List of 10 choices
                answer=item["answer_index"],  # 0-9 for A-J
                metadata={
                    "category": item["category"],
                    "subject": item.get("subject", item["category"]),
                    "source": "mmlu_pro"
                }
            )
            self.examples.append(example)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> DatasetExample:
        return self.examples[idx]

    def sample(self, n: int, seed: Optional[int] = None) -> List[DatasetExample]:
        """Sample n examples randomly."""
        if seed is not None:
            random.seed(seed)
        return random.sample(self.examples, min(n, len(self.examples)))

    def get_categories(self) -> List[str]:
        """Get list of unique categories."""
        return sorted(set(ex.metadata["category"] for ex in self.examples))

    def filter_by_category(self, category: str) -> List[DatasetExample]:
        """Get all examples from a specific category."""
        return [ex for ex in self.examples if ex.metadata["category"] == category]
