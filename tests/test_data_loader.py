"""
Genuine tests for data loading.

These tests validate:
1. TruthfulQA can be loaded
2. Examples have correct format
3. Train/val/test splits work correctly
4. Stratification works
5. Dataset can be saved and loaded
6. Category analysis works
"""

import unittest
import tempfile
from pathlib import Path

from data.data_loader import (
    HallucinationDataset, HallucinationExample, load_truthfulqa
)


class TestHallucinationExample(unittest.TestCase):
    """Test HallucinationExample dataclass."""

    def test_example_creation(self):
        """Test creating a hallucination example."""
        example = HallucinationExample(
            id='test_1',
            prompt='What is 2+2?',
            factual_answer='4',
            hallucinated_answer='5',
            category='math',
            metadata={'source': 'test'}
        )

        self.assertEqual(example.id, 'test_1')
        self.assertEqual(example.prompt, 'What is 2+2?')
        self.assertEqual(example.factual_answer, '4')
        self.assertEqual(example.hallucinated_answer, '5')
        self.assertEqual(example.category, 'math')


class TestHallucinationDataset(unittest.TestCase):
    """Test HallucinationDataset class."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.cache_dir = self.temp_dir.name

    def tearDown(self):
        """Clean up test fixtures."""
        self.temp_dir.cleanup()

    def test_dataset_initialization(self):
        """Test dataset can be initialized."""
        dataset = HallucinationDataset(
            dataset_name='truthful_qa',
            cache_dir=self.cache_dir,
            seed=42
        )

        self.assertEqual(dataset.dataset_name, 'truthful_qa')
        self.assertEqual(dataset.seed, 42)
        self.assertEqual(len(dataset.examples), 0)  # Not loaded yet

    def test_load_truthfulqa(self):
        """Test loading TruthfulQA dataset."""
        dataset = HallucinationDataset(
            dataset_name='truthful_qa',
            cache_dir=self.cache_dir,
            seed=42
        )

        # Load dataset
        dataset.load()

        # Check that examples were loaded
        self.assertGreater(len(dataset), 0)

        # Check first example has correct format
        example = dataset[0]
        self.assertIsInstance(example, HallucinationExample)
        self.assertIsInstance(example.id, str)
        self.assertIsInstance(example.prompt, str)
        self.assertIsInstance(example.factual_answer, str)
        self.assertIsInstance(example.hallucinated_answer, str)
        self.assertIsInstance(example.category, str)
        self.assertIsInstance(example.metadata, dict)

        # Check that prompt and answers are non-empty
        self.assertGreater(len(example.prompt), 0)
        self.assertGreater(len(example.factual_answer), 0)
        self.assertGreater(len(example.hallucinated_answer), 0)

    def test_dataset_length(self):
        """Test __len__ method."""
        dataset = HallucinationDataset(cache_dir=self.cache_dir, seed=42)
        dataset.load()

        length = len(dataset)
        self.assertGreater(length, 0)
        self.assertEqual(length, len(dataset.examples))

    def test_dataset_getitem(self):
        """Test __getitem__ method."""
        dataset = HallucinationDataset(cache_dir=self.cache_dir, seed=42)
        dataset.load()

        # Get first item
        item = dataset[0]
        self.assertIsInstance(item, HallucinationExample)

        # Get last item
        item = dataset[-1]
        self.assertIsInstance(item, HallucinationExample)

    def test_dataset_split(self):
        """Test splitting dataset into train/val/test."""
        dataset = HallucinationDataset(cache_dir=self.cache_dir, seed=42)
        dataset.load()

        total_size = len(dataset)

        # Split with default ratios (0.7, 0.15, 0.15)
        train, val, test = dataset.split()

        # Check sizes are reasonable
        self.assertGreater(len(train), 0)
        self.assertGreater(len(val), 0)
        self.assertGreater(len(test), 0)

        # Check total size is preserved
        self.assertEqual(len(train) + len(val) + len(test), total_size)

        # Check approximate ratios (allow some variance due to stratification)
        train_ratio = len(train) / total_size
        val_ratio = len(val) / total_size
        test_ratio = len(test) / total_size

        self.assertAlmostEqual(train_ratio, 0.7, delta=0.05)
        self.assertAlmostEqual(val_ratio, 0.15, delta=0.05)
        self.assertAlmostEqual(test_ratio, 0.15, delta=0.05)

    def test_split_custom_ratios(self):
        """Test splitting with custom ratios."""
        dataset = HallucinationDataset(cache_dir=self.cache_dir, seed=42)
        dataset.load()

        total_size = len(dataset)

        # Custom split (0.8, 0.1, 0.1)
        train, val, test = dataset.split(
            train_ratio=0.8,
            val_ratio=0.1,
            test_ratio=0.1
        )

        # Check approximate ratios
        train_ratio = len(train) / total_size
        self.assertAlmostEqual(train_ratio, 0.8, delta=0.05)

    def test_split_reproducibility(self):
        """Test that split is reproducible with same seed."""
        dataset1 = HallucinationDataset(cache_dir=self.cache_dir, seed=42)
        dataset1.load()
        train1, val1, test1 = dataset1.split()

        dataset2 = HallucinationDataset(cache_dir=self.cache_dir, seed=42)
        dataset2.load()
        train2, val2, test2 = dataset2.split()

        # Check that splits are identical
        self.assertEqual(len(train1), len(train2))
        self.assertEqual(len(val1), len(val2))
        self.assertEqual(len(test1), len(test2))

        # Check first examples are the same
        self.assertEqual(train1[0].id, train2[0].id)
        self.assertEqual(val1[0].id, val2[0].id)
        self.assertEqual(test1[0].id, test2[0].id)

    def test_get_categories(self):
        """Test getting unique categories."""
        dataset = HallucinationDataset(cache_dir=self.cache_dir, seed=42)
        dataset.load()

        categories = dataset.get_categories()

        # Should return a list of strings
        self.assertIsInstance(categories, list)
        self.assertGreater(len(categories), 0)

        # All should be strings
        for cat in categories:
            self.assertIsInstance(cat, str)

        # Should be unique
        self.assertEqual(len(categories), len(set(categories)))

    def test_get_category_counts(self):
        """Test getting category counts."""
        dataset = HallucinationDataset(cache_dir=self.cache_dir, seed=42)
        dataset.load()

        counts = dataset.get_category_counts()

        # Should return a dict
        self.assertIsInstance(counts, dict)

        # All values should be positive integers
        for category, count in counts.items():
            self.assertIsInstance(category, str)
            self.assertIsInstance(count, int)
            self.assertGreater(count, 0)

        # Sum should equal total size
        self.assertEqual(sum(counts.values()), len(dataset))

    def test_save_and_load_dataset(self):
        """Test saving dataset to disk and loading it back."""
        # Create and load dataset
        dataset = HallucinationDataset(cache_dir=self.cache_dir, seed=42)
        dataset.load()

        original_size = len(dataset)
        original_first_id = dataset[0].id

        # Save to file
        save_path = Path(self.cache_dir) / 'test_dataset.json'
        dataset.save(str(save_path))

        # Verify file exists
        self.assertTrue(save_path.exists())

        # Load from file
        loaded_dataset = HallucinationDataset.load_from_file(str(save_path))

        # Verify size matches
        self.assertEqual(len(loaded_dataset), original_size)

        # Verify first example matches
        self.assertEqual(loaded_dataset[0].id, original_first_id)

        # Verify metadata is preserved
        self.assertEqual(loaded_dataset.dataset_name, dataset.dataset_name)
        self.assertEqual(loaded_dataset.seed, dataset.seed)

    def test_stratified_split(self):
        """Test that stratification maintains category proportions."""
        dataset = HallucinationDataset(cache_dir=self.cache_dir, seed=42)
        dataset.load()

        # Get original category proportions
        original_counts = dataset.get_category_counts()
        total = len(dataset)
        original_proportions = {
            cat: count / total
            for cat, count in original_counts.items()
        }

        # Split with stratification
        train, val, test = dataset.split(stratify=True)

        # Check train set proportions are similar
        train_counts = train.get_category_counts()
        train_total = len(train)

        for category in original_proportions:
            if category in train_counts:  # May not have all categories in small splits
                train_prop = train_counts[category] / train_total
                original_prop = original_proportions[category]

                # Allow 10% deviation due to small sample sizes
                self.assertAlmostEqual(
                    train_prop,
                    original_prop,
                    delta=0.1
                )


class TestLoadTruthfulQA(unittest.TestCase):
    """Test convenience function for loading TruthfulQA."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.cache_dir = self.temp_dir.name

    def tearDown(self):
        """Clean up test fixtures."""
        self.temp_dir.cleanup()

    def test_load_truthfulqa_convenience(self):
        """Test convenience function for loading and splitting."""
        train, val, test = load_truthfulqa(
            cache_dir=self.cache_dir,
            seed=42
        )

        # All should be HallucinationDataset instances
        self.assertIsInstance(train, HallucinationDataset)
        self.assertIsInstance(val, HallucinationDataset)
        self.assertIsInstance(test, HallucinationDataset)

        # All should have examples
        self.assertGreater(len(train), 0)
        self.assertGreater(len(val), 0)
        self.assertGreater(len(test), 0)


if __name__ == '__main__':
    unittest.main()
