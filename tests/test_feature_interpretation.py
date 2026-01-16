"""
Tests for feature interpretation tools.
"""

import unittest
import torch
import numpy as np
from pathlib import Path
import tempfile
import shutil

from models import JumpReLUSAE, GPT2WithResidualHooks
from evaluation.interpret_features import FeatureInterpreter


class TestFeatureInterpreter(unittest.TestCase):
    """Test FeatureInterpreter."""

    def setUp(self):
        """Set up test fixtures."""
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Create model and SAE
        self.model = GPT2WithResidualHooks('gpt2', device=self.device)
        self.d_model = 768
        self.d_hidden = 2048

        self.sae = JumpReLUSAE(d_model=self.d_model, d_hidden=self.d_hidden)
        self.sae = self.sae.to(self.device).eval()

        self.layer_idx = 0

        # Create interpreter
        self.interpreter = FeatureInterpreter(
            sae=self.sae,
            model_wrapper=self.model,
            layer_idx=self.layer_idx,
            device=self.device
        )

        # Test texts
        self.test_texts = [
            "The capital of France is Paris.",
            "Machine learning is fascinating.",
            "Python is a programming language.",
            "The quick brown fox jumps over the lazy dog.",
            "Artificial intelligence will change the world."
        ]

    def test_initialization(self):
        """Test interpreter initialization."""
        self.assertEqual(self.interpreter.layer_idx, self.layer_idx)
        self.assertEqual(self.interpreter.device, self.device)
        self.assertIsNotNone(self.interpreter.sae)
        self.assertIsNotNone(self.interpreter.model)

    def test_get_feature_activations(self):
        """Test getting feature activations for text."""
        text = "The capital of France is Paris."
        feature_id = 0

        activations = self.interpreter.get_feature_activations(text, feature_id)

        # Should return array of activations (one per token)
        self.assertIsInstance(activations, np.ndarray)
        self.assertGreater(len(activations), 0)

        # All activations should be >= 0 (JumpReLU)
        self.assertTrue(np.all(activations >= 0))

    def test_find_top_activating_examples(self):
        """Test finding top activating examples for a feature."""
        feature_id = 0
        top_k = 3

        results = self.interpreter.find_top_activating_examples(
            texts=self.test_texts,
            feature_id=feature_id,
            top_k=top_k
        )

        # Should return top_k results
        self.assertEqual(len(results), min(top_k, len(self.test_texts)))

        # Each result should have text, max_activation, token_idx
        for result in results:
            self.assertIn('text', result)
            self.assertIn('max_activation', result)
            self.assertIn('token_idx', result)
            self.assertIn('activations', result)

            self.assertIsInstance(result['text'], str)
            self.assertGreaterEqual(result['max_activation'], 0)
            self.assertIsInstance(result['token_idx'], int)

        # Results should be sorted by max_activation (descending)
        max_acts = [r['max_activation'] for r in results]
        self.assertEqual(max_acts, sorted(max_acts, reverse=True))

    def test_get_common_tokens(self):
        """Test extracting common tokens for a feature."""
        feature_id = 0
        top_k = 5

        common_tokens = self.interpreter.get_common_tokens(
            texts=self.test_texts,
            feature_id=feature_id,
            top_k=top_k
        )

        # Should return list of (token, count) tuples
        self.assertIsInstance(common_tokens, list)
        self.assertLessEqual(len(common_tokens), top_k)

        for token, count in common_tokens:
            self.assertIsInstance(token, str)
            self.assertIsInstance(count, int)
            self.assertGreater(count, 0)

        # Should be sorted by count (descending)
        counts = [c for _, c in common_tokens]
        self.assertEqual(counts, sorted(counts, reverse=True))

    def test_identify_dead_features(self):
        """Test identifying dead features."""
        threshold = 1e-6

        dead_features = self.interpreter.identify_dead_features(
            texts=self.test_texts,
            threshold=threshold
        )

        # Should return list of feature IDs
        self.assertIsInstance(dead_features, list)

        # All should be valid feature IDs
        for feat_id in dead_features:
            self.assertGreaterEqual(feat_id, 0)
            self.assertLess(feat_id, self.d_hidden)

    def test_get_feature_statistics(self):
        """Test computing feature statistics."""
        feature_id = 0

        stats = self.interpreter.get_feature_statistics(
            texts=self.test_texts,
            feature_id=feature_id
        )

        # Check all expected statistics
        expected_keys = [
            'mean_activation',
            'max_activation',
            'activation_frequency',
            'num_samples'
        ]

        for key in expected_keys:
            self.assertIn(key, stats)

        # Verify values are reasonable
        self.assertGreaterEqual(stats['mean_activation'], 0)
        self.assertGreaterEqual(stats['max_activation'], stats['mean_activation'])
        self.assertGreaterEqual(stats['activation_frequency'], 0)
        self.assertLessEqual(stats['activation_frequency'], 1.0)
        self.assertEqual(stats['num_samples'], len(self.test_texts))

    def test_activation_frequency(self):
        """Test activation frequency calculation."""
        feature_id = 0

        stats = self.interpreter.get_feature_statistics(
            texts=self.test_texts,
            feature_id=feature_id
        )

        freq = stats['activation_frequency']

        # Frequency should be between 0 and 1
        self.assertGreaterEqual(freq, 0.0)
        self.assertLessEqual(freq, 1.0)

    def test_different_features(self):
        """Test interpreter works for different features."""
        text = "The capital of France is Paris."

        # Test several random features
        for feature_id in [0, 100, 500, 1000]:
            activations = self.interpreter.get_feature_activations(text, feature_id)
            self.assertIsInstance(activations, np.ndarray)

    def test_empty_text_list(self):
        """Test handling empty text list."""
        feature_id = 0

        results = self.interpreter.find_top_activating_examples(
            texts=[],
            feature_id=feature_id,
            top_k=5
        )

        self.assertEqual(len(results), 0)

    def test_single_text(self):
        """Test with single text."""
        feature_id = 0

        results = self.interpreter.find_top_activating_examples(
            texts=["Hello world."],
            feature_id=feature_id,
            top_k=5
        )

        self.assertEqual(len(results), 1)


class TestFeatureInterpretationIntegration(unittest.TestCase):
    """Integration tests for feature interpretation."""

    def setUp(self):
        """Set up test fixtures."""
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = GPT2WithResidualHooks('gpt2', device=self.device)

        self.d_model = 768
        self.d_hidden = 2048
        self.sae = JumpReLUSAE(d_model=self.d_model, d_hidden=self.d_hidden)
        self.sae = self.sae.to(self.device).eval()

    def test_interpret_multiple_features(self):
        """Test interpreting multiple features."""
        interpreter = FeatureInterpreter(
            sae=self.sae,
            model_wrapper=self.model,
            layer_idx=0,
            device=self.device
        )

        texts = [
            "The capital of France is Paris.",
            "Machine learning is a subset of artificial intelligence.",
            "Python is widely used for data science.",
        ]

        # Interpret multiple features
        features_to_interpret = [0, 10, 50, 100]

        for feature_id in features_to_interpret:
            results = interpreter.find_top_activating_examples(
                texts=texts,
                feature_id=feature_id,
                top_k=2
            )

            self.assertGreater(len(results), 0)

    def test_dead_feature_detection(self):
        """Test dead feature detection across corpus."""
        interpreter = FeatureInterpreter(
            sae=self.sae,
            model_wrapper=self.model,
            layer_idx=0,
            device=self.device
        )

        # Use larger corpus
        texts = [
            "Text sample " + str(i) for i in range(20)
        ]

        dead_features = interpreter.identify_dead_features(
            texts=texts,
            threshold=1e-6
        )

        # Should be a list (may be empty or have dead features)
        self.assertIsInstance(dead_features, list)

        # Verify they are actually dead
        for feat_id in dead_features[:5]:  # Check first 5
            activations = interpreter.get_feature_activations(texts[0], feat_id)
            self.assertLess(np.max(activations), 1e-6)

    def test_feature_statistics_consistency(self):
        """Test that statistics are consistent."""
        interpreter = FeatureInterpreter(
            sae=self.sae,
            model_wrapper=self.model,
            layer_idx=0,
            device=self.device
        )

        texts = ["Sample text " + str(i) for i in range(10)]
        feature_id = 0

        # Get statistics twice
        stats1 = interpreter.get_feature_statistics(texts, feature_id)
        stats2 = interpreter.get_feature_statistics(texts, feature_id)

        # Should be identical
        self.assertAlmostEqual(
            stats1['mean_activation'],
            stats2['mean_activation']
        )
        self.assertAlmostEqual(
            stats1['max_activation'],
            stats2['max_activation']
        )

    def test_common_tokens_extraction(self):
        """Test common tokens extraction."""
        interpreter = FeatureInterpreter(
            sae=self.sae,
            model_wrapper=self.model,
            layer_idx=0,
            device=self.device
        )

        # Use texts with repeated words
        texts = [
            "The cat sat on the mat.",
            "The dog sat on the rug.",
            "The bird sat on the branch.",
        ]

        feature_id = 0
        common_tokens = interpreter.get_common_tokens(
            texts=texts,
            feature_id=feature_id,
            top_k=10
        )

        # Should find some common tokens
        self.assertGreater(len(common_tokens), 0)

        # "the" or "The" might appear frequently
        tokens = [t for t, c in common_tokens]
        # Just verify we got some tokens
        self.assertGreater(len(tokens), 0)


if __name__ == '__main__':
    unittest.main()
