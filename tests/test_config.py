"""
Genuine tests for configuration loading.

These tests validate:
1. Config can be loaded from YAML
2. Default values are correct
3. Config can be serialized and deserialized
4. Paths are created properly
5. Individual config sections work correctly
"""

import unittest
import tempfile
import yaml
from pathlib import Path

from config.config_loader import (
    Config, ModelConfig, SAEConfig, TrackingConfig,
    DetectionConfig, load_config, save_config
)


class TestConfigLoader(unittest.TestCase):
    """Test configuration loading functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)

    def tearDown(self):
        """Clean up test fixtures."""
        self.temp_dir.cleanup()

    def test_default_config_creation(self):
        """Test that default config can be created."""
        config = Config()

        # Check model config defaults
        self.assertEqual(config.model.base_model, "gpt2")
        self.assertEqual(config.model.d_model, 768)
        self.assertEqual(config.model.n_layers, 12)

        # Check SAE config defaults
        self.assertEqual(config.sae.d_model, 768)
        self.assertEqual(config.sae.d_hidden, 4096)
        self.assertAlmostEqual(config.sae.threshold, 0.1)

        # Check tracking config defaults
        self.assertEqual(config.tracking.top_k_features, 50)
        self.assertAlmostEqual(config.tracking.semantic_weight, 0.6)

    def test_config_from_dict(self):
        """Test creating config from dictionary."""
        config_dict = {
            'model': {
                'base_model': 'gpt2-medium',
                'd_model': 1024,
                'n_layers': 24
            },
            'sae': {
                'd_hidden': 8192,
                'threshold': 0.2
            }
        }

        config = Config.from_dict(config_dict)

        # Verify custom values
        self.assertEqual(config.model.base_model, 'gpt2-medium')
        self.assertEqual(config.model.d_model, 1024)
        self.assertEqual(config.model.n_layers, 24)
        self.assertEqual(config.sae.d_hidden, 8192)
        self.assertAlmostEqual(config.sae.threshold, 0.2)

        # Verify defaults for non-specified values
        self.assertEqual(config.sae.d_model, 768)  # default

    def test_config_to_dict(self):
        """Test converting config to dictionary."""
        config = Config()
        config_dict = config.to_dict()

        # Check that all sections are present
        self.assertIn('model', config_dict)
        self.assertIn('sae', config_dict)
        self.assertIn('tracking', config_dict)
        self.assertIn('detection', config_dict)

        # Check that values are correct
        self.assertEqual(config_dict['model']['base_model'], 'gpt2')
        self.assertEqual(config_dict['sae']['d_hidden'], 4096)

    def test_save_and_load_config(self):
        """Test saving config to file and loading it back."""
        # Create config with custom values
        config = Config()
        config.model.base_model = 'gpt2-large'
        config.sae.d_hidden = 8192
        config.tracking.top_k_features = 100

        # Save to file
        config_path = self.temp_path / 'test_config.yaml'
        save_config(config, str(config_path))

        # Verify file exists
        self.assertTrue(config_path.exists())

        # Load config back
        loaded_config = load_config(str(config_path))

        # Verify values match
        self.assertEqual(loaded_config.model.base_model, 'gpt2-large')
        self.assertEqual(loaded_config.sae.d_hidden, 8192)
        self.assertEqual(loaded_config.tracking.top_k_features, 100)

    def test_load_config_with_missing_file(self):
        """Test loading config when file doesn't exist (should use defaults)."""
        non_existent_path = self.temp_path / 'does_not_exist.yaml'

        config = load_config(str(non_existent_path))

        # Should return default config
        self.assertEqual(config.model.base_model, 'gpt2')
        self.assertEqual(config.sae.d_hidden, 4096)

    def test_paths_creation(self):
        """Test that PathsConfig creates directories."""
        from config.config_loader import PathsConfig

        paths = PathsConfig(
            data_dir=str(self.temp_path / 'data'),
            cache_dir=str(self.temp_path / 'cache'),
            models_dir=str(self.temp_path / 'models'),
            results_dir=str(self.temp_path / 'results'),
            logs_dir=str(self.temp_path / 'logs')
        )

        # Check all directories were created
        self.assertTrue(Path(paths.data_dir).exists())
        self.assertTrue(Path(paths.cache_dir).exists())
        self.assertTrue(Path(paths.models_dir).exists())
        self.assertTrue(Path(paths.results_dir).exists())
        self.assertTrue(Path(paths.logs_dir).exists())

    def test_config_serialization_roundtrip(self):
        """Test that config survives serialization roundtrip."""
        original_config = Config()
        original_config.detection.entropy_threshold = 2.5
        original_config.tracking.association_threshold = 0.7

        # Convert to dict and back
        config_dict = original_config.to_dict()
        restored_config = Config.from_dict(config_dict)

        # Verify critical values
        self.assertAlmostEqual(
            restored_config.detection.entropy_threshold,
            2.5
        )
        self.assertAlmostEqual(
            restored_config.tracking.association_threshold,
            0.7
        )

    def test_model_config_validation(self):
        """Test that model config has reasonable values."""
        model_config = ModelConfig()

        # Check d_model is positive
        self.assertGreater(model_config.d_model, 0)

        # Check n_layers is positive
        self.assertGreater(model_config.n_layers, 0)

        # Check device is valid
        self.assertIn(model_config.device, ['cuda', 'cpu'])

    def test_tracking_weights_sum_to_one(self):
        """Test that tracking weights approximately sum to 1.0."""
        tracking_config = TrackingConfig()

        total_weight = (
            tracking_config.semantic_weight +
            tracking_config.activation_weight +
            tracking_config.position_weight
        )

        # Should sum to 1.0 (allow small floating point error)
        self.assertAlmostEqual(total_weight, 1.0, places=5)

    def test_detection_weights_sum_to_one(self):
        """Test that detection weights approximately sum to 1.0."""
        detection_config = DetectionConfig()

        total_weight = (
            detection_config.entropy_weight +
            detection_config.churn_weight +
            detection_config.ml_weight
        )

        # Should sum to 1.0 (allow small floating point error)
        self.assertAlmostEqual(total_weight, 1.0, places=5)


if __name__ == '__main__':
    unittest.main()
