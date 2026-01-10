"""
Genuine tests for GPT-2 model wrapper.

These tests validate:
1. Model can be loaded
2. Hooks are registered correctly
3. Forward pass captures all activations
4. Activation shapes are correct
5. Cache clearing works
6. Text encoding works
7. Batch processing works
"""

import unittest
import torch

from models.model_wrapper import GPT2WithResidualHooks


class TestGPT2WithResidualHooks(unittest.TestCase):
    """Test GPT-2 model wrapper with hooks."""

    @classmethod
    def setUpClass(cls):
        """Load model once for all tests (expensive operation)."""
        cls.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        cls.model = GPT2WithResidualHooks(
            model_name='gpt2',
            device=cls.device
        )

    def test_model_initialization(self):
        """Test that model initializes correctly."""
        self.assertIsNotNone(self.model.model)
        self.assertIsNotNone(self.model.tokenizer)

        # Check config
        self.assertEqual(self.model.n_layers, 12)
        self.assertEqual(self.model.d_model, 768)

        # Check device
        self.assertIn(self.model.device, ['cuda', 'cpu'])

    def test_hook_registration(self):
        """Test that hooks are registered correctly."""
        # Clear any existing hooks
        self.model.remove_hooks()
        self.assertEqual(len(self.model.hooks), 0)

        # Register hooks
        self.model.register_hooks()

        # Should have 3 hooks per layer (residual, MLP, attention)
        expected_hooks = 3 * self.model.n_layers
        self.assertEqual(len(self.model.hooks), expected_hooks)

    def test_hook_removal(self):
        """Test that hooks can be removed."""
        self.model.register_hooks()
        self.assertGreater(len(self.model.hooks), 0)

        self.model.remove_hooks()
        self.assertEqual(len(self.model.hooks), 0)

    def test_cache_clearing(self):
        """Test that cache can be cleared."""
        # Add some dummy data to cache
        self.model.cache['residual_stream'] = [torch.randn(1, 10, 768)]

        # Clear cache
        self.model.clear_cache()

        # Check cache is empty
        self.assertEqual(len(self.model.cache['residual_stream']), 0)
        self.assertEqual(len(self.model.cache['mlp_outputs']), 0)
        self.assertEqual(len(self.model.cache['attn_outputs']), 0)

    def test_text_encoding(self):
        """Test encoding text to token IDs."""
        text = "Hello, world!"

        encoded = self.model.encode_text(text)

        # Check output format
        self.assertIn('input_ids', encoded)
        self.assertIn('attention_mask', encoded)

        # Check shapes
        input_ids = encoded['input_ids']
        attention_mask = encoded['attention_mask']

        self.assertEqual(input_ids.dim(), 2)  # [batch, seq_len]
        self.assertEqual(attention_mask.dim(), 2)

        # Check device
        self.assertEqual(str(input_ids.device), self.model.device)

        # Check that tokens are valid
        self.assertGreater(input_ids.shape[1], 0)  # Has some tokens
        self.assertTrue(torch.all(input_ids >= 0))  # No negative token IDs

    def test_forward_with_cache(self):
        """Test forward pass with activation caching."""
        # Encode some text
        text = "The quick brown fox jumps over the lazy dog."
        encoded = self.model.encode_text(text)

        # Forward pass with cache
        outputs = self.model.forward_with_cache(
            encoded['input_ids'],
            encoded['attention_mask']
        )

        # Check output keys
        self.assertIn('logits', outputs)
        self.assertIn('residual_stream', outputs)
        self.assertIn('mlp_outputs', outputs)
        self.assertIn('attn_outputs', outputs)

        # Check logits shape
        logits = outputs['logits']
        batch_size = encoded['input_ids'].shape[0]
        seq_len = encoded['input_ids'].shape[1]
        vocab_size = self.model.model.config.vocab_size

        self.assertEqual(logits.shape[0], batch_size)
        self.assertEqual(logits.shape[1], seq_len)
        self.assertEqual(logits.shape[2], vocab_size)

        # Check activation lists
        self.assertEqual(len(outputs['residual_stream']), self.model.n_layers)
        self.assertEqual(len(outputs['mlp_outputs']), self.model.n_layers)
        self.assertEqual(len(outputs['attn_outputs']), self.model.n_layers)

        # Check shapes of activations
        for i in range(self.model.n_layers):
            residual = outputs['residual_stream'][i]
            mlp = outputs['mlp_outputs'][i]
            attn = outputs['attn_outputs'][i]

            # All should have shape [batch, seq_len, d_model]
            self.assertEqual(residual.shape[0], batch_size)
            self.assertEqual(residual.shape[1], seq_len)
            self.assertEqual(residual.shape[2], self.model.d_model)

            self.assertEqual(mlp.shape[0], batch_size)
            self.assertEqual(mlp.shape[1], seq_len)
            self.assertEqual(mlp.shape[2], self.model.d_model)

            self.assertEqual(attn.shape[0], batch_size)
            self.assertEqual(attn.shape[1], seq_len)
            self.assertEqual(attn.shape[2], self.model.d_model)

    def test_activation_values_are_different(self):
        """Test that activations from different layers are actually different."""
        text = "This is a test sentence."
        encoded = self.model.encode_text(text)
        outputs = self.model.forward_with_cache(
            encoded['input_ids'],
            encoded['attention_mask']
        )

        # Get activations from first and last layer
        first_layer = outputs['residual_stream'][0]
        last_layer = outputs['residual_stream'][-1]

        # They should not be identical
        self.assertFalse(torch.allclose(first_layer, last_layer))

        # They should have non-zero values
        self.assertTrue(torch.any(first_layer != 0))
        self.assertTrue(torch.any(last_layer != 0))

    def test_process_text_convenience(self):
        """Test convenience method for processing text."""
        text = "Machine learning is fascinating."

        outputs = self.model.process_text(text)

        # Should have all required outputs
        self.assertIn('logits', outputs)
        self.assertIn('residual_stream', outputs)

        # Should have correct number of layers
        self.assertEqual(len(outputs['residual_stream']), self.model.n_layers)

    def test_multiple_forward_passes(self):
        """Test that multiple forward passes work correctly."""
        text1 = "First sentence."
        text2 = "Second sentence."

        # First forward pass
        outputs1 = self.model.process_text(text1)
        n_activations_1 = len(outputs1['residual_stream'])

        # Second forward pass
        outputs2 = self.model.process_text(text2)
        n_activations_2 = len(outputs2['residual_stream'])

        # Both should have same number of layers
        self.assertEqual(n_activations_1, n_activations_2)
        self.assertEqual(n_activations_1, self.model.n_layers)

        # Activations should be different
        self.assertFalse(
            torch.allclose(
                outputs1['residual_stream'][0],
                outputs2['residual_stream'][0]
            )
        )

    def test_batch_processing(self):
        """Test processing multiple texts in a batch."""
        texts = [
            "First text.",
            "Second text.",
            "Third text."
        ]

        # Encode texts
        encoded = self.model.tokenizer(
            texts,
            return_tensors='pt',
            padding=True,
            truncation=True
        )

        input_ids = encoded['input_ids'].to(self.model.device)
        attention_mask = encoded['attention_mask'].to(self.model.device)

        # Forward pass
        outputs = self.model.forward_with_cache(input_ids, attention_mask)

        # Check batch dimension
        batch_size = len(texts)
        self.assertEqual(outputs['logits'].shape[0], batch_size)

        for i in range(self.model.n_layers):
            self.assertEqual(
                outputs['residual_stream'][i].shape[0],
                batch_size
            )

    def test_get_activation_shape(self):
        """Test getting expected activation shape."""
        n_layers, d_model = self.model.get_activation_shape()

        self.assertEqual(n_layers, 12)
        self.assertEqual(d_model, 768)

    def test_no_gradient_computation(self):
        """Test that forward pass doesn't compute gradients."""
        text = "Test gradient computation."
        encoded = self.model.encode_text(text)

        # Forward pass
        outputs = self.model.forward_with_cache(
            encoded['input_ids'],
            encoded['attention_mask']
        )

        # Check that activations don't require gradients
        self.assertFalse(outputs['logits'].requires_grad)
        self.assertFalse(outputs['residual_stream'][0].requires_grad)
        self.assertFalse(outputs['mlp_outputs'][0].requires_grad)
        self.assertFalse(outputs['attn_outputs'][0].requires_grad)

    def test_model_is_in_eval_mode(self):
        """Test that model is in evaluation mode."""
        self.assertFalse(self.model.model.training)

    def test_long_sequence_handling(self):
        """Test handling of longer sequences."""
        # Create a long text
        long_text = " ".join(["word"] * 100)

        # Process with truncation
        outputs = self.model.process_text(long_text, max_length=128)

        # Should handle without errors
        self.assertIn('logits', outputs)
        self.assertEqual(len(outputs['residual_stream']), self.model.n_layers)

        # Sequence length should be at most max_length
        seq_len = outputs['logits'].shape[1]
        self.assertLessEqual(seq_len, 128)

    def test_empty_cache_after_clear(self):
        """Test that cache is properly cleared between runs."""
        # First forward pass
        text1 = "First pass."
        self.model.process_text(text1)

        # Clear cache
        self.model.clear_cache()

        # Check cache is empty
        self.assertEqual(len(self.model.cache['residual_stream']), 0)

        # Second forward pass
        text2 = "Second pass."
        outputs = self.model.process_text(text2)

        # Should have fresh activations
        self.assertEqual(len(outputs['residual_stream']), self.model.n_layers)


if __name__ == '__main__':
    unittest.main()
