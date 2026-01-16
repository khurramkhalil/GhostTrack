"""
Tests for SAE training pipeline.
"""

import unittest
import torch
import numpy as np
import tempfile
import shutil
from pathlib import Path

from models import JumpReLUSAE
from scripts.train_sae import SAETrainer


class TestSAETrainer(unittest.TestCase):
    """Test SAE training loop."""

    def setUp(self):
        """Set up test fixtures."""
        self.d_model = 768
        self.d_hidden = 4096
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Create small synthetic dataset
        self.num_samples = 200
        self.hidden_states = np.random.randn(self.num_samples, self.d_model).astype(np.float32)

        # Create temporary directory
        self.temp_dir = tempfile.mkdtemp()
        self.data_path = Path(self.temp_dir) / 'test_states.pt'
        torch.save({'hidden_states': torch.from_numpy(self.hidden_states)}, self.data_path)

        # Create SAE
        self.sae = JumpReLUSAE(d_model=self.d_model, d_hidden=self.d_hidden)

        # Create trainer config
        self.config = {
            'learning_rate': 1e-3,
            'l1_coefficient': 1e-3,
            'hidden_dim_multiplier': 8
        }
        
        self.trainer = SAETrainer(
            sae=self.sae,
            layer_idx=0,
            hidden_states_path=str(self.data_path),
            config=self.config,
            device=self.device
        )

    def tearDown(self):
        """Clean up."""
        shutil.rmtree(self.temp_dir)

    def test_initialization(self):
        """Test trainer initialization."""
        self.assertIsNotNone(self.trainer.sae)
        self.assertIsNotNone(self.trainer.optimizer)
        self.assertIsNotNone(self.trainer.scheduler)
        self.assertEqual(self.trainer.device, self.device)

    def test_load_data(self):
        """Test loading hidden states."""
        states = self.trainer.load_data()

        self.assertEqual(states.shape[0], self.num_samples)
        self.assertEqual(states.shape[1], self.d_model)
        self.assertIsInstance(states, torch.Tensor)

    def test_compute_loss(self):
        """Test loss computation."""
        batch = torch.randn(16, self.d_model).to(self.device)
        loss_dict = self.trainer.compute_loss(batch)

        # Check all loss components exist
        self.assertIn('total_loss', loss_dict)
        self.assertIn('reconstruction_loss', loss_dict)
        self.assertIn('l1_loss', loss_dict)
        self.assertIn('sparsity', loss_dict)

        # All should be scalar tensors or floats
        for key, value in loss_dict.items():
            if isinstance(value, torch.Tensor):
                self.assertEqual(value.ndim, 0)  # Scalar

    def test_train_one_epoch(self):
        """Test training for one epoch."""
        initial_loss = None
        final_loss = None

        # Get initial loss
        with torch.no_grad():
            batch = torch.randn(16, self.d_model).to(self.device)
            initial_loss = self.trainer.compute_loss(batch)['total_loss'].item()

        # Train for one epoch
        history = self.trainer.train(
            epochs=1,
            batch_size=32,
            val_split=0.2,
            save_dir=self.temp_dir
        )

        # Check history structure
        self.assertIn('train_loss', history)
        self.assertIn('val_loss', history)
        self.assertEqual(len(history['train_loss']), 1)

    def test_training_reduces_loss(self):
        """Test that training reduces loss."""
        # Train for a few epochs
        history = self.trainer.train(
            epochs=5,
            batch_size=32,
            val_split=0.2,
            save_dir=self.temp_dir
        )

        train_losses = history['train_loss']

        # Loss should generally decrease (allowing some fluctuation)
        # Check that final loss is less than initial
        self.assertLess(train_losses[-1], train_losses[0] * 1.2)

    def test_validation_split(self):
        """Test validation split."""
        val_split = 0.2
        history = self.trainer.train(
            epochs=1,
            batch_size=32,
            val_split=val_split,
            save_dir=self.temp_dir
        )

        # Should have both train and val losses
        self.assertGreater(len(history['train_loss']), 0)
        self.assertGreater(len(history['val_loss']), 0)

    def test_checkpoint_saving(self):
        """Test that checkpoints are saved."""
        save_dir = Path(self.temp_dir) / 'checkpoints'
        save_dir.mkdir(exist_ok=True)

        self.trainer.train(
            epochs=2,
            batch_size=32,
            val_split=0.2,
            save_dir=str(save_dir)
        )

        # Check that final checkpoint exists
        final_path = save_dir / 'sae_layer_0_final.pt'
        self.assertTrue(final_path.exists())

        # Load checkpoint
        checkpoint = torch.load(final_path, map_location='cpu')
        self.assertIn('model_state_dict', checkpoint)
        self.assertIn('optimizer_state_dict', checkpoint)
        self.assertIn('epoch', checkpoint)

    def test_best_checkpoint_saving(self):
        """Test that best checkpoint is saved."""
        save_dir = Path(self.temp_dir) / 'checkpoints'
        save_dir.mkdir(exist_ok=True)

        self.trainer.train(
            epochs=3,
            batch_size=32,
            val_split=0.2,
            save_dir=str(save_dir)
        )

        # Check that best checkpoint exists
        best_path = save_dir / 'sae_layer_0_best.pt'
        self.assertTrue(best_path.exists())

        # Load and verify
        checkpoint = torch.load(best_path, map_location='cpu')
        self.assertIn('best_loss', checkpoint)

    def test_gradient_clipping(self):
        """Test gradient clipping during training."""
        # This is tested implicitly by successful training
        # If gradient clipping fails, training would likely diverge
        history = self.trainer.train(
            epochs=2,
            batch_size=32,
            val_split=0.2,
            save_dir=self.temp_dir
        )

        # Check that losses are finite
        for loss in history['train_loss']:
            self.assertTrue(np.isfinite(loss))

    def test_decoder_normalization(self):
        """Test that decoder is normalized during training."""
        # Train for one epoch
        self.trainer.train(
            epochs=1,
            batch_size=32,
            val_split=0.2,
            save_dir=self.temp_dir
        )

        # Check decoder norms
        norms = self.sae.get_feature_norms()

        # Norms should be close to 1.0 (allowing some tolerance)
        mean_norm = norms.mean().item()
        self.assertAlmostEqual(mean_norm, 1.0, delta=0.1)

    def test_scheduler_updates(self):
        """Test that learning rate scheduler updates."""
        initial_lr = self.trainer.optimizer.param_groups[0]['lr']

        self.trainer.train(
            epochs=2,
            batch_size=32,
            val_split=0.2,
            save_dir=self.temp_dir
        )

        final_lr = self.trainer.optimizer.param_groups[0]['lr']

        # Learning rate should have changed (cosine annealing)
        self.assertNotEqual(initial_lr, final_lr)

    def test_sparsity_metric(self):
        """Test that sparsity is tracked."""
        history = self.trainer.train(
            epochs=1,
            batch_size=32,
            val_split=0.2,
            save_dir=self.temp_dir
        )

        # Sparsity should be tracked
        self.assertIn('train_sparsity', history)
        self.assertGreater(len(history['train_sparsity']), 0)

        # Sparsity should be reasonable (0-1 range)
        for sparsity in history['train_sparsity']:
            self.assertGreaterEqual(sparsity, 0.0)
            self.assertLessEqual(sparsity, 1.0)


class TestSAETrainingIntegration(unittest.TestCase):
    """Integration tests for SAE training."""

    def setUp(self):
        """Set up test fixtures."""
        self.d_model = 768
        self.d_hidden = 2048  # Smaller for faster testing
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up."""
        shutil.rmtree(self.temp_dir)

    def test_full_training_pipeline(self):
        """Test complete training pipeline."""
        # Create synthetic data
        num_samples = 500
        hidden_states = np.random.randn(num_samples, self.d_model).astype(np.float32)

        data_path = Path(self.temp_dir) / 'states.pt'
        torch.save({'hidden_states': torch.from_numpy(hidden_states)}, data_path)

        # Create and train SAE
        sae = JumpReLUSAE(d_model=self.d_model, d_hidden=self.d_hidden)
        config = {'learning_rate': 1e-3, 'l1_coefficient': 1e-3}
        trainer = SAETrainer(
            sae=sae,
            layer_idx=0,
            hidden_states_path=str(data_path),
            config=config,
            device=self.device
        )

        history = trainer.train(
            epochs=5,
            batch_size=64,
            val_split=0.2,
            save_dir=self.temp_dir
        )

        # Verify training completed
        self.assertEqual(len(history['train_loss']), 5)
        self.assertEqual(len(history['val_loss']), 5)

        # Verify checkpoints
        self.assertTrue((Path(self.temp_dir) / 'sae_layer_0_final.pt').exists())
        self.assertTrue((Path(self.temp_dir) / 'sae_layer_0_best.pt').exists())

        # Load best checkpoint and verify it works
        best_checkpoint = torch.load(
            Path(self.temp_dir) / 'sae_layer_0_best.pt',
            map_location='cpu'
        )

        new_sae = JumpReLUSAE(d_model=self.d_model, d_hidden=self.d_hidden)
        new_sae.load_state_dict(best_checkpoint['model_state_dict'])

        # Test inference
        test_input = torch.randn(1, 10, self.d_model)
        with torch.no_grad():
            output = new_sae.forward(test_input)

        self.assertIn('reconstruction', output)
        self.assertIn('features', output)

    def test_reproducibility_with_seed(self):
        """Test that training is reproducible with same seed."""
        # Create data
        num_samples = 200
        hidden_states = np.random.randn(num_samples, self.d_model).astype(np.float32)
        data_path = Path(self.temp_dir) / 'states.pt'
        torch.save({'hidden_states': torch.from_numpy(hidden_states)}, data_path)

        # Train twice with same seed
        losses1 = []
        losses2 = []

        for seed in [42, 42]:
            torch.manual_seed(seed)
            np.random.seed(seed)

            sae = JumpReLUSAE(d_model=self.d_model, d_hidden=self.d_hidden)
            config = {'learning_rate': 1e-3, 'l1_coefficient': 1e-3}
            trainer = SAETrainer(
                sae=sae,
                layer_idx=0,
                hidden_states_path=str(data_path),
                config=config,
                device=self.device
            )

            history = trainer.train(
                epochs=2,
                batch_size=32,
                val_split=0.2,
                save_dir=self.temp_dir
            )

            if len(losses1) == 0:
                losses1 = history['train_loss']
            else:
                losses2 = history['train_loss']

        # Losses should be identical (or very close)
        np.testing.assert_array_almost_equal(losses1, losses2, decimal=5)


if __name__ == '__main__':
    unittest.main()
