"""
Tests for Track dataclass.
"""

import unittest
import numpy as np

from tracking.track import Track


class TestTrack(unittest.TestCase):
    """Test Track dataclass and methods."""

    def setUp(self):
        """Set up test fixtures."""
        self.track_id = 0
        self.birth_layer = 2
        self.token_pos = 5
        self.feature_embedding = np.random.randn(768)

        self.track = Track(
            track_id=self.track_id,
            feature_embedding=self.feature_embedding.copy(),
            birth_layer=self.birth_layer,
            token_pos=self.token_pos
        )

    def test_initialization(self):
        """Test track initialization."""
        self.assertEqual(self.track.track_id, self.track_id)
        self.assertEqual(self.track.birth_layer, self.birth_layer)
        self.assertEqual(self.track.token_pos, self.token_pos)
        self.assertIsNone(self.track.death_layer)
        self.assertEqual(len(self.track.trajectory), 0)
        self.assertTrue(self.track.is_alive())

    def test_update_trajectory(self):
        """Test updating track trajectory."""
        layer = 2
        activation = 0.8
        embedding = np.random.randn(768)

        self.track.update(layer, activation, embedding.copy())

        self.assertEqual(len(self.track.trajectory), 1)
        self.assertEqual(self.track.trajectory[0][0], layer)
        self.assertEqual(self.track.trajectory[0][1], activation)
        np.testing.assert_array_equal(self.track.trajectory[0][2], embedding)

    def test_multiple_updates(self):
        """Test multiple trajectory updates."""
        for layer in range(2, 6):
            activation = 0.5 + layer * 0.1
            embedding = np.random.randn(768)
            self.track.update(layer, activation, embedding.copy())

        self.assertEqual(len(self.track.trajectory), 4)
        self.assertEqual(self.track.length(), 4)

    def test_is_alive(self):
        """Test is_alive method."""
        self.assertTrue(self.track.is_alive())

        # Mark as dead
        self.track.death_layer = 5
        self.assertFalse(self.track.is_alive())

    def test_get_activation_at(self):
        """Test get_activation_at method."""
        # Add some trajectory
        self.track.update(2, 0.5, np.random.randn(768))
        self.track.update(3, 0.8, np.random.randn(768))
        self.track.update(4, 0.6, np.random.randn(768))

        # Test retrieval
        self.assertEqual(self.track.get_activation_at(2), 0.5)
        self.assertEqual(self.track.get_activation_at(3), 0.8)
        self.assertEqual(self.track.get_activation_at(4), 0.6)

        # Non-existent layer
        self.assertIsNone(self.track.get_activation_at(10))

    def test_get_embedding_at(self):
        """Test get_embedding_at method."""
        emb2 = np.random.randn(768)
        emb3 = np.random.randn(768)

        self.track.update(2, 0.5, emb2.copy())
        self.track.update(3, 0.8, emb3.copy())

        # Test retrieval
        np.testing.assert_array_equal(
            self.track.get_embedding_at(2),
            emb2
        )
        np.testing.assert_array_equal(
            self.track.get_embedding_at(3),
            emb3
        )

        # Non-existent layer
        self.assertIsNone(self.track.get_embedding_at(10))

    def test_layer_range(self):
        """Test layer_range method."""
        # Alive track
        layer_range = self.track.layer_range()
        self.assertEqual(layer_range.start, self.birth_layer)

        # Dead track
        self.track.death_layer = 5
        layer_range = self.track.layer_range()
        self.assertEqual(layer_range.start, 2)
        self.assertEqual(layer_range.stop, 6)

    def test_max_activation(self):
        """Test max_activation method."""
        self.track.update(2, 0.5, np.random.randn(768))
        self.track.update(3, 0.9, np.random.randn(768))
        self.track.update(4, 0.6, np.random.randn(768))

        self.assertEqual(self.track.max_activation(), 0.9)

    def test_mean_activation(self):
        """Test mean_activation method."""
        self.track.update(2, 0.4, np.random.randn(768))
        self.track.update(3, 0.6, np.random.randn(768))
        self.track.update(4, 0.8, np.random.randn(768))

        expected_mean = (0.4 + 0.6 + 0.8) / 3
        self.assertAlmostEqual(self.track.mean_activation(), expected_mean)

    def test_final_activation(self):
        """Test final_activation method."""
        self.track.update(2, 0.5, np.random.randn(768))
        self.track.update(3, 0.9, np.random.randn(768))
        self.track.update(4, 0.7, np.random.randn(768))

        self.assertEqual(self.track.final_activation(), 0.7)

    def test_length(self):
        """Test length method."""
        self.assertEqual(self.track.length(), 0)

        self.track.update(2, 0.5, np.random.randn(768))
        self.assertEqual(self.track.length(), 1)

        self.track.update(3, 0.6, np.random.randn(768))
        self.assertEqual(self.track.length(), 2)

    def test_activation_variance(self):
        """Test activation_variance method."""
        # Empty trajectory
        self.assertEqual(self.track.activation_variance(), 0.0)

        # Single point
        self.track.update(2, 0.5, np.random.randn(768))
        self.assertEqual(self.track.activation_variance(), 0.0)

        # Multiple points
        activations = [0.5, 0.7, 0.6]
        for i, act in enumerate(activations):
            self.track.update(2 + i, act, np.random.randn(768))

        # Clear previous single point
        self.track.trajectory = []
        for i, act in enumerate(activations):
            self.track.update(2 + i, act, np.random.randn(768))

        expected_var = np.var(activations)
        self.assertAlmostEqual(
            self.track.activation_variance(),
            expected_var,
            places=6
        )

    def test_is_stable(self):
        """Test is_stable method."""
        # Stable track
        for i in range(5):
            self.track.update(2 + i, 0.5 + 0.01 * i, np.random.randn(768))

        self.assertTrue(self.track.is_stable(threshold=0.1))

        # Unstable track
        self.track.trajectory = []
        for i in range(5):
            self.track.update(2 + i, 0.5 + 0.2 * i, np.random.randn(768))

        self.assertFalse(self.track.is_stable(threshold=0.01))

    def test_dominates_at_layer(self):
        """Test dominates_at_layer method."""
        self.track.update(2, 0.9, np.random.randn(768))

        # Dominates
        other_activations = [0.5, 0.6, 0.7]
        self.assertTrue(self.track.dominates_at_layer(2, other_activations))

        # Does not dominate
        other_activations = [0.95, 0.6, 0.7]
        self.assertFalse(self.track.dominates_at_layer(2, other_activations))

        # Empty others
        self.assertTrue(self.track.dominates_at_layer(2, []))

        # Non-existent layer
        self.assertFalse(self.track.dominates_at_layer(10, [0.5]))

    def test_to_dict(self):
        """Test serialization to dictionary."""
        self.track.update(2, 0.5, np.random.randn(768))
        self.track.update(3, 0.7, np.random.randn(768))

        track_dict = self.track.to_dict()

        # Check keys
        expected_keys = [
            'track_id', 'birth_layer', 'death_layer', 'token_pos',
            'trajectory', 'max_activation', 'mean_activation',
            'final_activation', 'length', 'is_alive', 'metadata'
        ]

        for key in expected_keys:
            self.assertIn(key, track_dict)

        # Check values
        self.assertEqual(track_dict['track_id'], self.track_id)
        self.assertEqual(track_dict['birth_layer'], self.birth_layer)
        self.assertEqual(track_dict['length'], 2)
        self.assertTrue(track_dict['is_alive'])

    def test_from_dict(self):
        """Test deserialization from dictionary."""
        # Create track with trajectory
        self.track.update(2, 0.5, np.random.randn(768))
        self.track.update(3, 0.7, np.random.randn(768))
        self.track.death_layer = 5

        # Serialize
        track_dict = self.track.to_dict()

        # Deserialize
        reconstructed = Track.from_dict(track_dict)

        # Verify
        self.assertEqual(reconstructed.track_id, self.track.track_id)
        self.assertEqual(reconstructed.birth_layer, self.track.birth_layer)
        self.assertEqual(reconstructed.death_layer, self.track.death_layer)
        self.assertEqual(reconstructed.token_pos, self.track.token_pos)
        self.assertEqual(len(reconstructed.trajectory), len(self.track.trajectory))

    def test_repr(self):
        """Test string representation."""
        self.track.update(2, 0.9, np.random.randn(768))
        self.track.update(3, 0.7, np.random.randn(768))

        repr_str = repr(self.track)

        # Should contain key information
        self.assertIn('Track', repr_str)
        self.assertIn(f'id={self.track_id}', repr_str)
        self.assertIn(f'birth=L{self.birth_layer}', repr_str)
        self.assertIn('ALIVE', repr_str)

        # Dead track
        self.track.death_layer = 5
        repr_str = repr(self.track)
        self.assertIn('DEAD@L5', repr_str)

    def test_empty_trajectory_methods(self):
        """Test methods with empty trajectory."""
        self.assertEqual(self.track.max_activation(), 0.0)
        self.assertEqual(self.track.mean_activation(), 0.0)
        self.assertEqual(self.track.final_activation(), 0.0)
        self.assertEqual(self.track.length(), 0)
        self.assertEqual(self.track.activation_variance(), 0.0)


class TestTrackEdgeCases(unittest.TestCase):
    """Test edge cases for Track."""

    def test_single_update(self):
        """Test track with single update."""
        track = Track(
            track_id=0,
            feature_embedding=np.random.randn(768),
            birth_layer=0,
            token_pos=0
        )

        track.update(0, 0.8, np.random.randn(768))

        self.assertEqual(track.length(), 1)
        self.assertEqual(track.max_activation(), 0.8)
        self.assertEqual(track.mean_activation(), 0.8)
        self.assertEqual(track.final_activation(), 0.8)
        self.assertEqual(track.activation_variance(), 0.0)

    def test_very_long_trajectory(self):
        """Test track with many updates."""
        track = Track(
            track_id=0,
            feature_embedding=np.random.randn(768),
            birth_layer=0,
            token_pos=0
        )

        # Add 100 updates
        for layer in range(100):
            track.update(layer, 0.5 + 0.001 * layer, np.random.randn(768))

        self.assertEqual(track.length(), 100)
        self.assertGreater(track.max_activation(), 0.5)

    def test_zero_activations(self):
        """Test track with all zero activations."""
        track = Track(
            track_id=0,
            feature_embedding=np.random.randn(768),
            birth_layer=0,
            token_pos=0
        )

        for layer in range(5):
            track.update(layer, 0.0, np.random.randn(768))

        self.assertEqual(track.max_activation(), 0.0)
        self.assertEqual(track.mean_activation(), 0.0)
        self.assertEqual(track.activation_variance(), 0.0)

    def test_embedding_update(self):
        """Test that feature_embedding is updated with each update."""
        track = Track(
            track_id=0,
            feature_embedding=np.random.randn(768),
            birth_layer=0,
            token_pos=0
        )

        emb1 = np.random.randn(768)
        emb2 = np.random.randn(768)

        track.update(0, 0.5, emb1.copy())
        np.testing.assert_array_equal(track.feature_embedding, emb1)

        track.update(1, 0.6, emb2.copy())
        np.testing.assert_array_equal(track.feature_embedding, emb2)


if __name__ == '__main__':
    unittest.main()
