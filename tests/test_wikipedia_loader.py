"""
Tests for Wikipedia corpus loading and hidden state extraction.
"""

import unittest
import torch
import numpy as np
from pathlib import Path
import tempfile
import shutil

from data.wikipedia_loader import WikipediaCorpus, HiddenStateExtractor
from models import GPT2WithResidualHooks


class TestWikipediaCorpus(unittest.TestCase):
    """Test WikipediaCorpus loader."""

    def setUp(self):
        """Set up test fixtures."""
        self.corpus = WikipediaCorpus(
            language='en'
        )

    def test_initialization(self):
        """Test corpus initialization."""
        self.assertIsNotNone(self.corpus.dataset)
        self.assertTrue(self.corpus.streaming)

    def test_get_batch(self):
        """Test getting a batch of texts."""
        batch = self.corpus.get_batch(batch_size=5)

        self.assertEqual(len(batch), 5)
        for text in batch:
            self.assertIsInstance(text, str)
            self.assertGreater(len(text), 0)

    def test_get_batch_with_min_length(self):
        """Test getting batch with minimum length filter."""
        batch = self.corpus.get_batch(batch_size=5, min_length=100)

        self.assertEqual(len(batch), 5)
        for text in batch:
            self.assertGreaterEqual(len(text), 100)

    def test_get_batch_empty(self):
        """Test getting empty batch."""
        batch = self.corpus.get_batch(batch_size=0)
        self.assertEqual(len(batch), 0)

    def test_iteration(self):
        """Test iterating through corpus."""
        count = 0
        for text in self.corpus.get_batch(batch_size=10):
            count += 1
            self.assertIsInstance(text, str)

        self.assertEqual(count, 10)


class TestHiddenStateExtractor(unittest.TestCase):
    """Test HiddenStateExtractor."""

    def setUp(self):
        """Set up test fixtures."""
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = GPT2WithResidualHooks('gpt2', device=self.device)

        # Create temporary directory for saving states
        self.temp_dir = tempfile.mkdtemp()

        # Small corpus for testing
        self.corpus = WikipediaCorpus(language='en')

        self.extractor = HiddenStateExtractor(
            model_wrapper=self.model,
            corpus=self.corpus,
            device=self.device
        )

    def tearDown(self):
        """Clean up temporary files."""
        shutil.rmtree(self.temp_dir)

    def test_initialization(self):
        """Test extractor initialization."""
        self.assertEqual(self.extractor.device, self.device)
        self.assertIsNotNone(self.extractor.model)
        self.assertIsNotNone(self.extractor.corpus)

    def test_extract_from_text(self):
        """Test extracting hidden states from text."""
        text = "The capital of France is Paris."
        layer_idx = 0

        hidden_states = self.extractor.extract_from_text(text, layer_idx)

        # Check shape: [seq_len, d_model]
        self.assertEqual(len(hidden_states.shape), 2)
        self.assertEqual(hidden_states.shape[1], 768)  # GPT2 d_model
        self.assertIsInstance(hidden_states, np.ndarray)

    def test_extract_from_batch(self):
        """Test extracting from batch of texts."""
        texts = [
            "The capital of France is Paris.",
            "Machine learning is fascinating.",
            "Python is a programming language."
        ]
        layer_idx = 0

        all_states = self.extractor.extract_from_batch(texts, layer_idx)

        # Should have states from all texts
        self.assertGreater(len(all_states), 0)
        self.assertEqual(all_states.shape[1], 768)

    def test_extract_for_layer(self):
        """Test extracting and saving states for a layer."""
        layer_idx = 0
        num_tokens = 100
        batch_size = 2

        save_path = self.extractor.extract_for_layer(
            layer_idx=layer_idx,
            num_tokens=num_tokens,
            batch_size=batch_size,
            save_dir=self.temp_dir
        )

        # Check file was created
        self.assertTrue(Path(save_path).exists())

        # Load and check shape
        data = torch.load(save_path, map_location='cpu')
        states = data['hidden_states']
        self.assertEqual(states.shape[1], 768)
        # Should have approximately num_tokens (may vary due to tokenization)
        self.assertGreater(states.shape[0], 0)

    def test_extract_different_layers(self):
        """Test extracting from different layers."""
        text = "The capital of France is Paris."

        for layer_idx in [0, 5, 11]:
            hidden_states = self.extractor.extract_from_text(text, layer_idx)
            self.assertEqual(hidden_states.shape[1], 768)

    def test_max_length_handling(self):
        """Test that long texts are truncated properly."""
        # Create a very long text
        long_text = "word " * 1000
        layer_idx = 0

        hidden_states = self.extractor.extract_from_text(
            long_text,
            layer_idx,
            max_length=128
        )

        # Should be truncated to max_length tokens
        self.assertLessEqual(hidden_states.shape[0], 128)

    def test_extract_with_different_batch_sizes(self):
        """Test extraction with different batch sizes."""
        texts = ["Text " + str(i) for i in range(10)]
        layer_idx = 0

        for batch_size in [1, 2, 5]:
            states = self.extractor.extract_from_batch(
                texts[:batch_size],
                layer_idx
            )
            self.assertGreater(states.shape[0], 0)


class TestHiddenStateExtractionIntegration(unittest.TestCase):
    """Integration tests for hidden state extraction."""

    def setUp(self):
        """Set up test fixtures."""
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = GPT2WithResidualHooks('gpt2', device=self.device)
        self.corpus = WikipediaCorpus(language='en')
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up."""
        shutil.rmtree(self.temp_dir)

    def test_extract_multiple_layers(self):
        """Test extracting states for multiple layers."""
        extractor = HiddenStateExtractor(
            model_wrapper=self.model,
            corpus=self.corpus,
            device=self.device
        )

        num_tokens = 50
        layers_to_extract = [0, 6, 11]

        saved_paths = []
        for layer_idx in layers_to_extract:
            path = extractor.extract_for_layer(
                layer_idx=layer_idx,
                num_tokens=num_tokens,
                batch_size=2,
                save_dir=self.temp_dir
            )
            saved_paths.append(path)
            self.assertTrue(Path(path).exists())

        # All layers should have same number of tokens (approximately)
        shapes = [torch.load(p, map_location='cpu')['hidden_states'].shape for p in saved_paths]
        num_tokens_per_layer = [s[0] for s in shapes]

        # All should be positive
        for n in num_tokens_per_layer:
            self.assertGreater(n, 0)

    def test_deterministic_extraction(self):
        """Test that extracting same text gives same result."""
        extractor = HiddenStateExtractor(
            model_wrapper=self.model,
            corpus=self.corpus,
            device=self.device
        )

        text = "The capital of France is Paris."
        layer_idx = 0

        states1 = extractor.extract_from_text(text, layer_idx)
        states2 = extractor.extract_from_text(text, layer_idx)

        np.testing.assert_array_almost_equal(states1, states2)


if __name__ == '__main__':
    unittest.main()
