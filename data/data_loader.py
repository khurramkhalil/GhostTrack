"""Data loading module for hallucination detection."""

import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

import numpy as np
from datasets import load_dataset
from sklearn.model_selection import train_test_split


@dataclass
class HallucinationExample:
    """Single example with factual and hallucinated answers."""
    id: str
    prompt: str
    factual_answer: str
    hallucinated_answer: str
    category: str
    metadata: Dict


class HallucinationDataset:
    """
    Dataset for hallucination detection.
    Loads TruthfulQA and creates factual/hallucinated pairs.
    """

    def __init__(
        self,
        dataset_name: str = 'truthful_qa',
        cache_dir: Optional[str] = None,
        seed: int = 42
    ):
        """
        Initialize dataset.

        Args:
            dataset_name: Name of dataset to load ('truthful_qa').
            cache_dir: Directory to cache processed data.
            seed: Random seed for reproducibility.
        """
        self.dataset_name = dataset_name
        self.cache_dir = Path(cache_dir) if cache_dir else Path('./data/cache')
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.seed = seed
        self.examples: List[HallucinationExample] = []

        # Set random seeds
        random.seed(seed)
        np.random.seed(seed)

    def load_truthfulqa(self) -> List[HallucinationExample]:
        """
        Load TruthfulQA dataset from HuggingFace.

        Returns:
            List of HallucinationExample objects.
        """
        print("Loading TruthfulQA dataset...")

        # Load dataset
        dataset = load_dataset('truthful_qa', 'generation')

        examples = []

        for idx, item in enumerate(dataset['validation']):
            # Extract fields
            question = item['question']
            best_answer = item['best_answer']
            incorrect_answers = item['incorrect_answers']
            category = item['category']

            # Create example for each incorrect answer
            for inc_idx, incorrect_answer in enumerate(incorrect_answers):
                example = HallucinationExample(
                    id=f"truthfulqa_{idx}_{inc_idx}",
                    prompt=question,
                    factual_answer=best_answer,
                    hallucinated_answer=incorrect_answer,
                    category=category,
                    metadata={
                        'source': 'truthful_qa',
                        'source_idx': idx,
                        'incorrect_idx': inc_idx
                    }
                )
                examples.append(example)

        print(f"Loaded {len(examples)} question-answer pairs")
        return examples

    def load(self) -> 'HallucinationDataset':
        """Load the dataset."""
        if self.dataset_name == 'truthful_qa':
            self.examples = self.load_truthfulqa()
        else:
            raise ValueError(f"Unknown dataset: {self.dataset_name}")

        return self

    def split(
        self,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        stratify: bool = True
    ) -> Tuple['HallucinationDataset', 'HallucinationDataset', 'HallucinationDataset']:
        """
        Split dataset into train/val/test sets.

        Args:
            train_ratio: Proportion for training set.
            val_ratio: Proportion for validation set.
            test_ratio: Proportion for test set.
            stratify: Whether to stratify by category.

        Returns:
            Tuple of (train_dataset, val_dataset, test_dataset).
        """
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
            "Ratios must sum to 1.0"

        # Prepare stratification labels
        stratify_labels = None
        if stratify:
            stratify_labels = [ex.category for ex in self.examples]

        # First split: train vs (val + test)
        train_examples, temp_examples, train_labels, temp_labels = train_test_split(
            self.examples,
            stratify_labels,
            train_size=train_ratio,
            random_state=self.seed,
            stratify=stratify_labels
        )

        # Second split: val vs test
        val_size = val_ratio / (val_ratio + test_ratio)
        val_examples, test_examples = train_test_split(
            temp_examples,
            train_size=val_size,
            random_state=self.seed,
            stratify=temp_labels if stratify else None
        )

        # Create new dataset instances
        train_dataset = HallucinationDataset(
            dataset_name=self.dataset_name,
            cache_dir=str(self.cache_dir),
            seed=self.seed
        )
        train_dataset.examples = train_examples

        val_dataset = HallucinationDataset(
            dataset_name=self.dataset_name,
            cache_dir=str(self.cache_dir),
            seed=self.seed
        )
        val_dataset.examples = val_examples

        test_dataset = HallucinationDataset(
            dataset_name=self.dataset_name,
            cache_dir=str(self.cache_dir),
            seed=self.seed
        )
        test_dataset.examples = test_examples

        print(f"Split sizes - Train: {len(train_examples)}, "
              f"Val: {len(val_examples)}, Test: {len(test_examples)}")

        return train_dataset, val_dataset, test_dataset

    def __len__(self) -> int:
        """Return number of examples."""
        return len(self.examples)

    def __getitem__(self, idx: int) -> HallucinationExample:
        """Get example by index."""
        return self.examples[idx]

    def get_categories(self) -> List[str]:
        """Get unique categories in dataset."""
        return list(set(ex.category for ex in self.examples))

    def get_category_counts(self) -> Dict[str, int]:
        """Get count of examples per category."""
        counts = {}
        for ex in self.examples:
            counts[ex.category] = counts.get(ex.category, 0) + 1
        return counts

    def save(self, path: str):
        """Save dataset to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            'dataset_name': self.dataset_name,
            'seed': self.seed,
            'examples': [
                {
                    'id': ex.id,
                    'prompt': ex.prompt,
                    'factual_answer': ex.factual_answer,
                    'hallucinated_answer': ex.hallucinated_answer,
                    'category': ex.category,
                    'metadata': ex.metadata
                }
                for ex in self.examples
            ]
        }

        with open(path, 'w') as f:
            json.dump(data, f, indent=2)

        print(f"Saved dataset to {path}")

    @classmethod
    def load_from_file(cls, path: str) -> 'HallucinationDataset':
        """Load dataset from disk."""
        with open(path, 'r') as f:
            data = json.load(f)

        dataset = cls(
            dataset_name=data['dataset_name'],
            seed=data['seed']
        )

        dataset.examples = [
            HallucinationExample(**ex)
            for ex in data['examples']
        ]

        print(f"Loaded dataset from {path} with {len(dataset)} examples")
        return dataset


def load_truthfulqa(
    cache_dir: Optional[str] = None,
    split_ratios: Tuple[float, float, float] = (0.7, 0.15, 0.15),
    seed: int = 42
) -> Tuple[HallucinationDataset, HallucinationDataset, HallucinationDataset]:
    """
    Convenience function to load and split TruthfulQA.

    Args:
        cache_dir: Cache directory for processed data.
        split_ratios: (train, val, test) ratios.
        seed: Random seed.

    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset).
    """
    dataset = HallucinationDataset(
        dataset_name='truthful_qa',
        cache_dir=cache_dir,
        seed=seed
    ).load()

    return dataset.split(
        train_ratio=split_ratios[0],
        val_ratio=split_ratios[1],
        test_ratio=split_ratios[2],
        stratify=True
    )
