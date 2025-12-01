"""
Base dataset class for LLM confidence probing.

All dataset loaders should inherit from BaseDataset and implement the required methods.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass


@dataclass
class DatasetExample:
    """Standard format for all dataset examples."""
    
    question: str
    choices: List[str]
    answer: int  # Index of correct choice
    metadata: Dict[str, Any]
    
    def format_prompt(self, style: str = "qa") -> str:
        """
        Format the example as a prompt for the LLM.
        
        Args:
            style: Prompt style ("qa", "multiple_choice", "cot")
            
        Returns:
            Formatted prompt string
        """
        if style == "qa":
            return f"Q: {self.question}\nA:"
        
        elif style == "multiple_choice":
            choices_text = "\n".join([
                f"({chr(65 + i)}) {choice}" 
                for i, choice in enumerate(self.choices)
            ])
            return f"Q: {self.question}\n{choices_text}\nAnswer:"
        
        elif style == "cot":
            choices_text = "\n".join([
                f"({chr(65 + i)}) {choice}" 
                for i, choice in enumerate(self.choices)
            ])
            return (
                f"Q: {self.question}\n{choices_text}\n"
                "Let's think step by step.\nAnswer:"
            )
        
        else:
            raise ValueError(f"Unknown prompt style: {style}")
    
    def format_with_answer(self, answer_text: str) -> str:
        """
        Format question with a specific answer for correctness labeling.
        
        Args:
            answer_text: The answer text to append
            
        Returns:
            "Q: ... A: ..." format
        """
        return f"Q: {self.question} A: {answer_text}"
    
    def get_correct_answer(self) -> str:
        """
        Get the correct answer text.

        NOTE: For datasets that use a dummy `answer` index (e.g., GSM8K/TriviaQA),
        you should not call this method and instead rely on `metadata`.

        Raises:
            IndexError: If the stored `answer` index is out of range for `choices`.
        """
        # CHANGE: Added explicit bounds check so bad upstream data raises a clear error
        # instead of a generic IndexError from list indexing.
        if not 0 <= self.answer < len(self.choices):
            raise IndexError(
                f"Answer index {self.answer} is out of range for choices "
                f"of length {len(self.choices)}."
            )
        return self.choices[self.answer]


class BaseDataset(ABC):
    """
    Abstract base class for all dataset loaders.
    
    All datasets should:
    1. Load data from HuggingFace datasets or local files
    2. Return examples in standardized DatasetExample format
    3. Support train/validation/test splits
    4. Provide metadata for filtering/analysis
    """
    
    def __init__(self, split: str = "validation"):
        """
        Initialize dataset.
        
        Args:
            split: Dataset split ("train", "validation", "test")
        """
        self.split = split
        self.data = self._load_data()
    
    @abstractmethod
    def _load_data(self) -> List[DatasetExample]:
        """
        Load and process the dataset.
        
        Returns:
            List of DatasetExample objects
        """
        pass
    
    def __len__(self) -> int:
        """Return number of examples."""
        return len(self.data)
    
    def __getitem__(self, idx: int) -> DatasetExample:
        """Get example by index."""
        return self.data[idx]
    
    def __iter__(self):
        """Iterate over examples."""
        return iter(self.data)
    
    def sample(self, n: int, seed: Optional[int] = None) -> List[DatasetExample]:
        """
        Sample n examples randomly.
        
        Args:
            n: Number of examples to sample
            seed: Random seed for reproducibility
            
        Returns:
            List of sampled examples
        """
        import random
        if seed is not None:
            random.seed(seed)
        return random.sample(self.data, min(n, len(self.data)))
    
    def filter(self, condition: Callable[[DatasetExample], bool]) -> List[DatasetExample]:
        """
        Filter examples by condition.
        
        Args:
            condition: Callable that takes DatasetExample and returns bool
            
        Returns:
            Filtered list of examples
        """
        return [ex for ex in self.data if condition(ex)]
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get dataset statistics.
        
        Returns:
            Dictionary with statistics (size, choice distribution, etc.)
        """
        num_choices = [len(ex.choices) for ex in self.data]
        return {
            "size": len(self.data),
            "split": self.split,
            "avg_choices": sum(num_choices) / len(num_choices) if num_choices else 0,
            "min_choices": min(num_choices) if num_choices else 0,
            "max_choices": max(num_choices) if num_choices else 0,
        }
