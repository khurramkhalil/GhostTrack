"""
Tests for LayerwiseFeatureExtractor.
"""

import unittest
import torch
import numpy as np
from pathlib import Path
import tempfile
import shutil

from models import GPT2WithResidualHooks, JumpReLUSAE
from tracking.feature_extractor import LayerwiseFeatureExtractor


class TestLayerwiseFeatureExtractor(unittest.TestCase):
    """Test LayerwiseFeatureExtractor."""

    def setUp(self):
        """Set up test fixtures."""
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Create model
        self.model = GPT2WithResidualHooks('gpt2', device=self.device)

        # Create SAEs for each layer
        self.d_model = 768
        self.d_hidden = 2048
        self.n_layers = 12

        self.saes = []
        for _ in range(self.n_layers):
            sae = JumpReLUSAE(d_model=self.d_model, d_hidden=self.d_hidden)
            sae = sae.to(self.device).eval()
            self.saes.append(sae)

        # Create extractor
        self.extractor = LayerwiseFeatureExtractor(
            model_wrapper=self.model,
            saes=self.saes,
            device=self.device
        )

    def test_initialization(self):
        """Test extractor initialization."""
        self.assertEqual(self.extractor.n_layers, self.n_layers)
        self.assertEqual(self.extractor.device, self.device)
        self.assertEqual(len(self.extractor.saes), self.n_layers)

    def test_extract_features(self):
        """Test extracting features from text."""
        text = "The capital of France is Paris."

        layer_features = self.extractor.extract_features(text)

        # Should have features for all layers
        self.assertEqual(len(layer_features), self.n_layers)

        # Check each layer's output
        for i, layer_data in enumerate(layer_features):
            self.assertEqual(layer_data['layer'], i)
            self.assertIn('features', layer_data)
            self.assertIn('activations', layer_data)
            self.assertIn('error', layer_data)
            self.assertIn('hidden_state', layer_data)
            self.assertIn('sparsity', layer_data)

            # Check shapes
            seq_len = layer_data['features'].shape[0]
            self.assertEqual(layer_data['features'].shape[1], self.d_hidden)
            self.assertEqual(layer_data['error'].shape[1], self.d_model)

    def test_get_top_k_features(self):
        """Test getting top-k features from a layer."""
        text = "The capital of France is Paris."
        layer_features = self.extractor.extract_features(text)

        # Get top-50 features from layer 0
        k = 50
        top_features = self.extractor.get_top_k_features(
            layer_features[0],
            k=k
        )

        # Should return k features
        self.assertEqual(len(top_features), k)

        # Each should be (feat_id, activation, embedding) tuple
        for feat_id, activation, embedding in top_features:
            self.assertIsInstance(feat_id, int)
            self.assertIsInstance(activation, float)
            self.assertIsInstance(embedding, np.ndarray)

            self.assertGreaterEqual(feat_id, 0)
            self.assertLess(feat_id, self.d_hidden)
            self.assertGreaterEqual(activation, 0)  # JumpReLU
            self.assertEqual(embedding.shape[0], self.d_model)

        # Should be sorted by activation (descending)
        activations = [act for _, act, _ in top_features]
        self.assertEqual(activations, sorted(activations, reverse=True))

    def test_get_top_k_with_token_pos(self):
        """Test getting top-k features at specific token position."""
        text = "The capital of France is Paris."
        layer_features = self.extractor.extract_features(text)

        k = 20
        token_pos = 0

        top_features = self.extractor.get_top_k_features(
            layer_features[0],
            k=k,
            token_pos=token_pos
        )

        self.assertEqual(len(top_features), k)

    def test_extract_at_position(self):
        """Test extracting features at specific token position."""
        text = "The capital of France is Paris."
        position = 0
        k = 30

        results = self.extractor.extract_at_position(text, position, k)

        # Should have results for all layers
        self.assertEqual(len(results), self.n_layers)

        for layer_result in results:
            self.assertIn('layer', layer_result)
            self.assertIn('position', layer_result)
            self.assertIn('top_features', layer_result)
            self.assertIn('num_active', layer_result)

            self.assertEqual(layer_result['position'], position)
            self.assertLessEqual(len(layer_result['top_features']), k)

    def test_feature_embeddings_shape(self):
        """Test that feature embeddings have correct shape."""
        text = "The capital of France is Paris."
        layer_features = self.extractor.extract_features(text)

        top_features = self.extractor.get_top_k_features(layer_features[0], k=10)

        for feat_id, activation, embedding in top_features:
            self.assertEqual(embedding.shape, (self.d_model,))

    def test_different_texts(self):
        """Test extractor with different texts."""
        texts = [
            "Hello world.",
            "Machine learning is fascinating.",
            "The quick brown fox jumps over the lazy dog."
        ]

        for text in texts:
            layer_features = self.extractor.extract_features(text)
            self.assertEqual(len(layer_features), self.n_layers)

    def test_max_length_handling(self):
        """Test that max_length parameter works."""
        long_text = "word " * 1000
        max_length = 128

        layer_features = self.extractor.extract_features(
            long_text,
            max_length=max_length
        )

        # Check that sequence length is limited
        for layer_data in layer_features:
            seq_len = layer_data['features'].shape[0]
            self.assertLessEqual(seq_len, max_length)

    def test_sparsity_metric(self):
        """Test that sparsity is computed."""
        text = "The capital of France is Paris."
        layer_features = self.extractor.extract_features(text)

        for layer_data in layer_features:
            sparsity = layer_data['sparsity']
            self.assertIsInstance(sparsity, float)
            self.assertGreaterEqual(sparsity, 0.0)
            self.assertLessEqual(sparsity, 1.0)


class TestLayerwiseFeatureExtractorCheckpointLoading(unittest.TestCase):
    """Test loading SAEs from checkpoints."""

    def setUp(self):
        """Set up test fixtures."""
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = GPT2WithResidualHooks('gpt2', device=self.device)

        self.d_model = 768
        self.d_hidden = 2048
        self.n_layers = 12

        # Create temporary directory for checkpoints
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up."""
        shutil.rmtree(self.temp_dir)

    def test_load_from_checkpoints(self):
        """Test loading SAEs from checkpoint directory."""
        # Create and save SAE checkpoints
        for layer_idx in range(self.n_layers):
            sae = JumpReLUSAE(d_model=self.d_model, d_hidden=self.d_hidden)

            checkpoint = {
                'model_state_dict': sae.state_dict(),
                'best_loss': 0.5
            }

            checkpoint_path = Path(self.temp_dir) / f'sae_layer_{layer_idx}_best.pt'
            torch.save(checkpoint, checkpoint_path)

        # Load extractor
        extractor = LayerwiseFeatureExtractor.load_from_checkpoints(
            model_wrapper=self.model,
            checkpoint_dir=self.temp_dir,
            device=self.device
        )

        # Verify
        self.assertEqual(extractor.n_layers, self.n_layers)
        self.assertEqual(len(extractor.saes), self.n_layers)

        # Test extraction works
        text = "Test sentence."
        layer_features = extractor.extract_features(text)
        self.assertEqual(len(layer_features), self.n_layers)

    def test_missing_checkpoint_raises_error(self):
        """Test that missing checkpoint raises error."""
        # Create checkpoints for only some layers
        for layer_idx in range(5):  # Only first 5 layers
            sae = JumpReLUSAE(d_model=self.d_model, d_hidden=self.d_hidden)

            checkpoint = {
                'model_state_dict': sae.state_dict(),
                'best_loss': 0.5
            }

            checkpoint_path = Path(self.temp_dir) / f'sae_layer_{layer_idx}_best.pt'
            torch.save(checkpoint, checkpoint_path)

        # Should raise error for missing checkpoints
        with self.assertRaises(FileNotFoundError):
            LayerwiseFeatureExtractor.load_from_checkpoints(
                model_wrapper=self.model,
                checkpoint_dir=self.temp_dir,
                device=self.device
            )


class TestFeatureExtractionIntegration(unittest.TestCase):
    """Integration tests for feature extraction."""

    def setUp(self):
        """Set up test fixtures."""
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = GPT2WithResidualHooks('gpt2', device=self.device)

        self.d_model = 768
        self.d_hidden = 2048
        self.n_layers = 12

        # Create SAEs
        self.saes = []
        for _ in range(self.n_layers):
            sae = JumpReLUSAE(d_model=self.d_model, d_hidden=self.d_hidden)
            sae = sae.to(self.device).eval()
            self.saes.append(sae)

        self.extractor = LayerwiseFeatureExtractor(
            model_wrapper=self.model,
            saes=self.saes,
            device=self.device
        )

    def test_extract_and_track_workflow(self):
        """Test typical workflow of extracting features for tracking."""
        text = "The capital of France is Paris."

        # Step 1: Extract features for all layers
        layer_features = self.extractor.extract_features(text)

        # Step 2: Get top-k features from each layer
        k = 50
        top_features_per_layer = []

        for layer_data in layer_features:
            top_features = self.extractor.get_top_k_features(layer_data, k=k)
            top_features_per_layer.append(top_features)

        # Verify we have features for all layers
        self.assertEqual(len(top_features_per_layer), self.n_layers)

        for top_features in top_features_per_layer:
            self.assertEqual(len(top_features), k)

    def test_consistent_extraction(self):
        """Test that extraction is deterministic."""
        text = "The capital of France is Paris."

        # Extract twice
        features1 = self.extractor.extract_features(text)
        features2 = self.extractor.extract_features(text)

        # Should be identical
        for layer_data1, layer_data2 in zip(features1, features2):
            np.testing.assert_array_almost_equal(
                layer_data1['features'].cpu().numpy(),
                layer_data2['features'].cpu().numpy()
            )

    def test_different_layers_different_features(self):
        """Test that different layers produce different features."""
        text = "The capital of France is Paris."
        layer_features = self.extractor.extract_features(text)

        # Get top features from different layers
        top_0 = self.extractor.get_top_k_features(layer_features[0], k=20)
        top_6 = self.extractor.get_top_k_features(layer_features[6], k=20)
        top_11 = self.extractor.get_top_k_features(layer_features[11], k=20)

        # Feature IDs should be different (generally)
        ids_0 = set(feat_id for feat_id, _, _ in top_0)
        ids_6 = set(feat_id for feat_id, _, _ in top_6)
        ids_11 = set(feat_id for feat_id, _, _ in top_11)

        # While some overlap is possible, they shouldn't be identical
        self.assertFalse(ids_0 == ids_6 == ids_11)


if __name__ == '__main__':
    unittest.main()
