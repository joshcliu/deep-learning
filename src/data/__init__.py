"""
Dataset loaders for LLM confidence probing.

Available datasets:
- MMLU: Massive Multitask Language Understanding (57 subjects)
- TriviaQA: Open-domain question answering
- GSM8K: Grade school math word problems

Usage:
    from src.data import MMLUDataset, TriviaQADataset, GSM8KDataset
    
    # Load MMLU
    mmlu = MMLUDataset(split="validation", category="stem")
    
    # Load TriviaQA
    trivia = TriviaQADataset(split="validation")
    
    # Load GSM8K
    gsm8k = GSM8KDataset(split="test")
"""

from .base import BaseDataset, DatasetExample
from .mmlu import MMLUDataset, MMLU_SUBJECTS
from .triviaqa import TriviaQADataset, generate_trivia_labels
from .gsm8k import GSM8KDataset, generate_gsm8k_labels

__all__ = [
    # Base classes
    "BaseDataset",
    "DatasetExample",
    
    # MMLU
    "MMLUDataset",
    "MMLU_SUBJECTS",
    
    # TriviaQA
    "TriviaQADataset",
    "generate_trivia_labels",
    
    # GSM8K
    "GSM8KDataset",
    "generate_gsm8k_labels",
]