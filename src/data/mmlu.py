"""
MMLU (Massive Multitask Language Understanding) dataset loader.

MMLU is a benchmark with 57 subjects spanning STEM, humanities, social sciences.
Each question is multiple choice with 4 options.

Reference: Hendrycks et al. 2021 - "Measuring Massive Multitask Language Understanding"
"""

from typing import List, Optional, Dict, Any
from datasets import load_dataset
from .base import BaseDataset, DatasetExample


# All 57 MMLU subjects organized by category
MMLU_SUBJECTS = {
    "stem": [
        "abstract_algebra", "anatomy", "astronomy", "college_biology",
        "college_chemistry", "college_computer_science", "college_mathematics",
        "college_physics", "computer_security", "conceptual_physics",
        "electrical_engineering", "elementary_mathematics", "high_school_biology",
        "high_school_chemistry", "high_school_computer_science",
        "high_school_mathematics", "high_school_physics", "high_school_statistics",
        "machine_learning"
    ],
    "humanities": [
        "formal_logic", "high_school_european_history", "high_school_us_history",
        "high_school_world_history", "international_law", "jurisprudence",
        "logical_fallacies", "moral_disputes", "moral_scenarios", "philosophy",
        "prehistory", "professional_law", "world_religions"
    ],
    "social_sciences": [
        "econometrics", "high_school_geography", "high_school_government_and_politics",
        "high_school_macroeconomics", "high_school_microeconomics",
        "high_school_psychology", "human_sexuality", "professional_psychology",
        "public_relations", "security_studies", "sociology", "us_foreign_policy"
    ],
    "other": [
        "business_ethics", "clinical_knowledge", "college_medicine", "global_facts",
        "human_aging", "management", "marketing", "medical_genetics",
        "miscellaneous", "nutrition", "professional_accounting",
        "professional_medicine", "virology"
    ]
}


class MMLUDataset(BaseDataset):
    """
    MMLU dataset loader with subject filtering.
    
    Usage:
        # Load all subjects
        dataset = MMLUDataset(split="validation")
        
        # Load specific subjects
        dataset = MMLUDataset(split="test", subjects=["anatomy", "philosophy"])
        
        # Load by category
        dataset = MMLUDataset(split="validation", category="stem")
    """
    
    def __init__(
        self,
        split: str = "validation",
        subjects: Optional[List[str]] = None,
        category: Optional[str] = None
    ):
        """
        Initialize MMLU dataset.
        
        Args:
            split: Dataset split ("auxiliary_train", "validation", "test", "dev")
            subjects: List of specific subjects to load (None = all subjects)
            category: Category to load ("stem", "humanities", "social_sciences", "other")
        """
        self.subjects = self._resolve_subjects(subjects, category)
        self.category = category
        super().__init__(split)
    
    def _resolve_subjects(
        self, 
        subjects: Optional[List[str]], 
        category: Optional[str]
    ) -> List[str]:
        """Determine which subjects to load based on inputs."""
        if subjects is not None:
            return subjects
        elif category is not None:
            if category not in MMLU_SUBJECTS:
                raise ValueError(
                    f"Unknown category: {category}. "
                    f"Must be one of {list(MMLU_SUBJECTS.keys())}"
                )
            return MMLU_SUBJECTS[category]
        else:
            # Load all subjects
            all_subjects = []
            for subj_list in MMLU_SUBJECTS.values():
                all_subjects.extend(subj_list)
            return all_subjects
    
    def _load_data(self) -> List[DatasetExample]:
        """Load MMLU data from HuggingFace datasets."""
        examples = []
        
        for subject in self.subjects:
            try:
                # Load subject-specific dataset
                dataset = load_dataset(
                    "cais/mmlu",
                    subject,
                    split=self.split,
                    trust_remote_code=True
                )
                
                # Convert to standard format
                for item in dataset:
                    example = DatasetExample(
                        question=item["question"].strip(),
                        choices=[
                            item["choices"][0].strip(),
                            item["choices"][1].strip(),
                            item["choices"][2].strip(),
                            item["choices"][3].strip()
                        ],
                        answer=item["answer"],  # 0, 1, 2, or 3
                        metadata={
                            "subject": subject,
                            "category": self._get_category(subject)
                        }
                    )
                    examples.append(example)
                    
            except Exception as e:
                print(f"Warning: Failed to load subject '{subject}': {e}")
                continue
        
        return examples
    
    def _get_category(self, subject: str) -> str:
        """Get category for a subject."""
        for category, subjects in MMLU_SUBJECTS.items():
            if subject in subjects:
                return category
        return "other"
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get MMLU-specific statistics."""
        base_stats = super().get_statistics()
        
        # Count examples per subject
        subject_counts = {}
        for ex in self.data:
            subj = ex.metadata["subject"]
            subject_counts[subj] = subject_counts.get(subj, 0) + 1
        
        # Count examples per category
        category_counts = {}
        for ex in self.data:
            cat = ex.metadata["category"]
            category_counts[cat] = category_counts.get(cat, 0) + 1
        
        return {
            **base_stats,
            "num_subjects": len(self.subjects),
            "subjects": self.subjects,
            "examples_per_subject": subject_counts,
            "examples_per_category": category_counts
        }
    
    @staticmethod
    def get_all_subjects() -> Dict[str, List[str]]:
        """Get dictionary of all subjects organized by category."""
        return MMLU_SUBJECTS.copy()
    
    @staticmethod
    def get_category_subjects(category: str) -> List[str]:
        """Get list of subjects in a specific category."""
        if category not in MMLU_SUBJECTS:
            raise ValueError(
                f"Unknown category: {category}. "
                f"Must be one of {list(MMLU_SUBJECTS.keys())}"
            )
        return MMLU_SUBJECTS[category].copy()