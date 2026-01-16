"""
Tests for HypothesisTracker.
"""

import unittest
import numpy as np

from tracking.hypothesis_tracker import HypothesisTracker
from tracking.track import Track


class TestHypothesisTracker(unittest.TestCase):
    """Test HypothesisTracker."""

    def setUp(self):
        """Set up test fixtures."""
        self.d_model = 768

        self.config = {
            'birth_threshold': 0.5,
            'death_threshold': 0.1,
            'association_threshold': 0.5,
            'semantic_weight': 0.6,
            'activation_weight': 0.2,
            'position_weight': 0.2,
            'top_k_features': 50,
            'use_greedy': False
        }

        self.tracker = HypothesisTracker(config=self.config)

    def test_initialization(self):
        """Test tracker initialization."""
        self.assertEqual(len(self.tracker.tracks), 0)
        self.assertEqual(self.tracker.track_id_counter, 0)
        self.assertIsNotNone(self.tracker.config)
        self.assertIsNotNone(self.tracker.stats)

    def test_default_config(self):
        """Test default configuration."""
        tracker = HypothesisTracker()
        config = tracker.config

        self.assertIn('birth_threshold', config)
        self.assertIn('death_threshold', config)
        self.assertIn('association_threshold', config)

    def test_initialize_tracks(self):
        """Test initializing tracks from layer 0."""
        # Create features above threshold
        layer_0_features = []
        for i in range(5):
            feat_id = i
            activation = 0.6 + i * 0.05  # All above 0.5 threshold
            embedding = np.random.randn(self.d_model)
            layer_0_features.append((feat_id, activation, embedding))

        self.tracker.initialize_tracks(layer_0_features, token_pos=0)

        # Should create 5 tracks
        self.assertEqual(len(self.tracker.tracks), 5)
        self.assertEqual(self.tracker.stats['total_births'], 5)

        # All tracks should be alive
        alive = self.tracker.get_alive_tracks()
        self.assertEqual(len(alive), 5)

    def test_birth_threshold_filtering(self):
        """Test that birth threshold filters weak features."""
        # Mix of features above and below threshold
        layer_0_features = [
            (0, 0.7, np.random.randn(self.d_model)),  # Above
            (1, 0.3, np.random.randn(self.d_model)),  # Below
            (2, 0.6, np.random.randn(self.d_model)),  # Above
            (3, 0.2, np.random.randn(self.d_model)),  # Below
        ]

        self.tracker.initialize_tracks(layer_0_features, token_pos=0)

        # Should only create 2 tracks
        self.assertEqual(len(self.tracker.tracks), 2)

    def test_update_tracks_association(self):
        """Test updating tracks with new features."""
        # Initialize with features
        layer_0_features = [
            (0, 0.7, np.random.randn(self.d_model)),
            (1, 0.6, np.random.randn(self.d_model)),
        ]
        self.tracker.initialize_tracks(layer_0_features, token_pos=0)

        # Get embeddings from tracks for similar features
        emb_0 = self.tracker.tracks[0].feature_embedding
        emb_1 = self.tracker.tracks[1].feature_embedding

        # Create layer 1 features (similar to layer 0)
        layer_1_features = [
            (0, 0.75, emb_0 + np.random.randn(self.d_model) * 0.01),  # Similar to track 0
            (1, 0.65, emb_1 + np.random.randn(self.d_model) * 0.01),  # Similar to track 1
        ]

        self.tracker.update_tracks(1, layer_1_features)

        # Tracks should be updated, not dead
        alive = self.tracker.get_alive_tracks()
        self.assertEqual(len(alive), 2)

        # Tracks should have length 2 (layer 0 + layer 1)
        for track in self.tracker.tracks:
            self.assertEqual(track.length(), 2)

    def test_death_detection(self):
        """Test that tracks die when not matched."""
        # Initialize
        layer_0_features = [
            (0, 0.7, np.random.randn(self.d_model)),
        ]
        self.tracker.initialize_tracks(layer_0_features, token_pos=0)

        # Update with completely different features (no match)
        layer_1_features = [
            (1, 0.8, np.random.randn(self.d_model)),
        ]
        self.tracker.update_tracks(1, layer_1_features)

        # Original track should die (unless lucky random match)
        # New track should be born

        # Total deaths should increase
        self.assertGreater(self.tracker.stats['total_deaths'], 0)

    def test_birth_during_update(self):
        """Test that new tracks can be born during update."""
        # Initialize with one feature
        layer_0_features = [
            (0, 0.7, np.random.randn(self.d_model)),
        ]
        self.tracker.initialize_tracks(layer_0_features, token_pos=0)

        # Update with additional strong feature
        emb_0 = self.tracker.tracks[0].feature_embedding
        layer_1_features = [
            (0, 0.75, emb_0 + np.random.randn(self.d_model) * 0.01),  # Match
            (1, 0.8, np.random.randn(self.d_model)),  # New strong feature
        ]
        self.tracker.update_tracks(1, layer_1_features)

        # Should have 2 total tracks (1 original + 1 new birth)
        self.assertGreaterEqual(len(self.tracker.tracks), 2)
        self.assertGreaterEqual(self.tracker.stats['total_births'], 2)

    def test_get_alive_tracks(self):
        """Test getting alive tracks."""
        # Initialize
        layer_0_features = [
            (i, 0.6 + i * 0.05, np.random.randn(self.d_model))
            for i in range(3)
        ]
        self.tracker.initialize_tracks(layer_0_features, token_pos=0)

        # All should be alive
        alive = self.tracker.get_alive_tracks()
        self.assertEqual(len(alive), 3)

        # Mark one as dead
        self.tracker.tracks[0].death_layer = 1

        alive = self.tracker.get_alive_tracks()
        self.assertEqual(len(alive), 2)

    def test_get_alive_tracks_at_layer(self):
        """Test getting tracks alive at specific layer."""
        # Create track that exists from layer 2-5
        track = Track(
            track_id=0,
            feature_embedding=np.random.randn(self.d_model),
            birth_layer=2,
            token_pos=0,
            death_layer=5
        )
        self.tracker.tracks.append(track)

        # Should be alive at layer 3
        alive_at_3 = self.tracker.get_alive_tracks(layer_idx=3)
        self.assertEqual(len(alive_at_3), 1)

        # Should not be alive at layer 1 (before birth)
        alive_at_1 = self.tracker.get_alive_tracks(layer_idx=1)
        self.assertEqual(len(alive_at_1), 0)

        # Should not be alive at layer 6 (after death)
        alive_at_6 = self.tracker.get_alive_tracks(layer_idx=6)
        self.assertEqual(len(alive_at_6), 0)

    def test_get_dead_tracks(self):
        """Test getting dead tracks."""
        # Initialize
        layer_0_features = [
            (i, 0.6, np.random.randn(self.d_model))
            for i in range(3)
        ]
        self.tracker.initialize_tracks(layer_0_features, token_pos=0)

        # Initially no dead tracks
        dead = self.tracker.get_dead_tracks()
        self.assertEqual(len(dead), 0)

        # Mark some as dead
        self.tracker.tracks[0].death_layer = 1
        self.tracker.tracks[1].death_layer = 2

        dead = self.tracker.get_dead_tracks()
        self.assertEqual(len(dead), 2)

    def test_get_dominant_track(self):
        """Test getting dominant track at layer."""
        # Create tracks with different activations
        for i in range(3):
            track = Track(
                track_id=i,
                feature_embedding=np.random.randn(self.d_model),
                birth_layer=0,
                token_pos=i
            )
            # Different activations at layer 1
            track.update(0, 0.5, np.random.randn(self.d_model))
            track.update(1, 0.5 + i * 0.2, np.random.randn(self.d_model))
            self.tracker.tracks.append(track)

        # Get dominant at layer 1
        dominant = self.tracker.get_dominant_track(layer_idx=1)

        # Should be track with highest activation
        self.assertEqual(dominant.track_id, 2)  # Highest activation
        self.assertAlmostEqual(dominant.get_activation_at(1), 0.9)

    def test_get_competing_tracks(self):
        """Test getting competing tracks."""
        # Create tracks with varying activations
        for i in range(5):
            track = Track(
                track_id=i,
                feature_embedding=np.random.randn(self.d_model),
                birth_layer=0,
                token_pos=i
            )
            track.update(0, 0.3 + i * 0.15, np.random.randn(self.d_model))
            self.tracker.tracks.append(track)

        # Get tracks with activation > 0.5 at layer 0
        competing = self.tracker.get_competing_tracks(layer_idx=0, threshold=0.5)

        # Should get tracks 2, 3, 4 (activations 0.6, 0.75, 0.9)
        self.assertEqual(len(competing), 3)

        # All should have activation > 0.5
        for track in competing:
            self.assertGreater(track.get_activation_at(0), 0.5)

    def test_reset(self):
        """Test resetting tracker."""
        # Initialize and create some tracks
        layer_0_features = [
            (i, 0.6, np.random.randn(self.d_model))
            for i in range(3)
        ]
        self.tracker.initialize_tracks(layer_0_features, token_pos=0)

        # Reset
        self.tracker.reset()

        # Should be empty
        self.assertEqual(len(self.tracker.tracks), 0)
        self.assertEqual(self.tracker.track_id_counter, 0)
        self.assertEqual(self.tracker.stats['total_births'], 0)
        self.assertEqual(self.tracker.stats['total_deaths'], 0)

    def test_get_statistics(self):
        """Test getting statistics."""
        # Initialize
        layer_0_features = [
            (i, 0.6, np.random.randn(self.d_model))
            for i in range(5)
        ]
        self.tracker.initialize_tracks(layer_0_features, token_pos=0)

        # Mark some as dead
        self.tracker.tracks[0].death_layer = 1
        self.tracker.tracks[1].death_layer = 1
        self.tracker.stats['total_deaths'] = 2

        stats = self.tracker.get_statistics()

        # Check keys
        expected_keys = [
            'total_tracks', 'alive_tracks', 'dead_tracks',
            'total_births', 'total_deaths', 'max_concurrent_tracks',
            'survival_rate'
        ]
        for key in expected_keys:
            self.assertIn(key, stats)

        # Check values
        self.assertEqual(stats['total_tracks'], 5)
        self.assertEqual(stats['alive_tracks'], 3)
        self.assertEqual(stats['dead_tracks'], 2)
        self.assertEqual(stats['total_births'], 5)
        self.assertAlmostEqual(stats['survival_rate'], 0.6)

    def test_summarize(self):
        """Test summary generation."""
        # Initialize
        layer_0_features = [
            (i, 0.6, np.random.randn(self.d_model))
            for i in range(3)
        ]
        self.tracker.initialize_tracks(layer_0_features, token_pos=0)

        summary = self.tracker.summarize()

        # Should be a string
        self.assertIsInstance(summary, str)

        # Should contain key information
        self.assertIn('Total tracks', summary)
        self.assertIn('Alive', summary)
        self.assertIn('Dead', summary)

    def test_get_track_by_id(self):
        """Test getting track by ID."""
        # Initialize
        layer_0_features = [
            (i, 0.6, np.random.randn(self.d_model))
            for i in range(3)
        ]
        self.tracker.initialize_tracks(layer_0_features, token_pos=0)

        # Get track 1
        track = self.tracker.get_track_by_id(1)
        self.assertIsNotNone(track)
        self.assertEqual(track.track_id, 1)

        # Non-existent track
        track = self.tracker.get_track_by_id(999)
        self.assertIsNone(track)

    def test_to_dict(self):
        """Test serialization to dict."""
        # Initialize
        layer_0_features = [
            (i, 0.6, np.random.randn(self.d_model))
            for i in range(2)
        ]
        self.tracker.initialize_tracks(layer_0_features, token_pos=0)

        tracker_dict = self.tracker.to_dict()

        # Check keys
        self.assertIn('config', tracker_dict)
        self.assertIn('tracks', tracker_dict)
        self.assertIn('track_id_counter', tracker_dict)
        self.assertIn('stats', tracker_dict)

        # Check values
        self.assertEqual(len(tracker_dict['tracks']), 2)
        self.assertEqual(tracker_dict['track_id_counter'], 2)

    def test_from_dict(self):
        """Test deserialization from dict."""
        # Initialize
        layer_0_features = [
            (i, 0.6, np.random.randn(self.d_model))
            for i in range(2)
        ]
        self.tracker.initialize_tracks(layer_0_features, token_pos=0)

        # Serialize
        tracker_dict = self.tracker.to_dict()

        # Deserialize
        new_tracker = HypothesisTracker.from_dict(tracker_dict)

        # Verify
        self.assertEqual(len(new_tracker.tracks), len(self.tracker.tracks))
        self.assertEqual(
            new_tracker.track_id_counter,
            self.tracker.track_id_counter
        )

    def test_use_greedy_option(self):
        """Test using greedy association algorithm."""
        greedy_config = self.config.copy()
        greedy_config['use_greedy'] = True

        tracker = HypothesisTracker(config=greedy_config)

        # Initialize
        layer_0_features = [
            (i, 0.6, np.random.randn(self.d_model))
            for i in range(3)
        ]
        tracker.initialize_tracks(layer_0_features, token_pos=0)

        # Update (should use greedy algorithm)
        layer_1_features = [
            (i, 0.65, np.random.randn(self.d_model))
            for i in range(3)
        ]
        tracker.update_tracks(1, layer_1_features)

        # Should complete without error
        self.assertGreater(len(tracker.tracks), 0)


class TestHypothesisTrackerIntegration(unittest.TestCase):
    """Integration tests for hypothesis tracking."""

    def setUp(self):
        """Set up test fixtures."""
        self.d_model = 768
        self.config = {
            'birth_threshold': 0.5,
            'association_threshold': 0.6,
            'semantic_weight': 0.7,
            'activation_weight': 0.2,
            'position_weight': 0.1
        }

    def test_full_tracking_pipeline(self):
        """Test complete tracking through multiple layers."""
        tracker = HypothesisTracker(config=self.config)

        # Layer 0: Initialize
        layer_0_features = [
            (i, 0.6 + i * 0.05, np.random.randn(self.d_model))
            for i in range(5)
        ]
        tracker.initialize_tracks(layer_0_features, token_pos=0)

        initial_tracks = len(tracker.tracks)
        self.assertEqual(initial_tracks, 5)

        # Layers 1-5: Update with similar features
        for layer in range(1, 6):
            # Create features similar to existing tracks
            features = []
            for track in tracker.get_alive_tracks():
                emb = track.feature_embedding + np.random.randn(self.d_model) * 0.05
                act = track.trajectory[-1][1] + np.random.randn() * 0.05
                features.append((track.track_id, max(0.0, act), emb))

            tracker.update_tracks(layer, features)

        # Check final state
        stats = tracker.get_statistics()

        self.assertGreater(stats['total_tracks'], 0)
        self.assertGreater(stats['total_births'], 0)

    def test_track_lifecycle(self):
        """Test complete track lifecycle: birth, update, death."""
        tracker = HypothesisTracker(config=self.config)

        # Birth: Layer 0
        embedding = np.random.randn(self.d_model)
        layer_0_features = [(0, 0.8, embedding.copy())]
        tracker.initialize_tracks(layer_0_features, token_pos=0)

        self.assertEqual(len(tracker.tracks), 1)
        self.assertEqual(tracker.stats['total_births'], 1)

        # Update: Layer 1 (similar feature)
        layer_1_features = [(0, 0.85, embedding + np.random.randn(self.d_model) * 0.01)]
        tracker.update_tracks(1, layer_1_features)

        self.assertEqual(tracker.tracks[0].length(), 2)
        self.assertTrue(tracker.tracks[0].is_alive())

        # Death: Layer 2 (no matching feature)
        layer_2_features = [(1, 0.9, np.random.randn(self.d_model))]
        tracker.update_tracks(2, layer_2_features)

        # Original track might die (unless random match)
        # New track should be born
        self.assertGreater(len(tracker.tracks), 0)

    def test_max_concurrent_tracking(self):
        """Test max concurrent tracks statistic."""
        tracker = HypothesisTracker(config=self.config)

        # Start with 3 tracks
        layer_0_features = [
            (i, 0.6, np.random.randn(self.d_model))
            for i in range(3)
        ]
        tracker.initialize_tracks(layer_0_features, token_pos=0)

        self.assertEqual(tracker.stats['max_concurrent_tracks'], 3)

        # Add more tracks
        layer_1_features = [
            (i, 0.6, np.random.randn(self.d_model))
            for i in range(5)
        ]
        tracker.update_tracks(1, layer_1_features)

        # Max concurrent should increase
        self.assertGreaterEqual(tracker.stats['max_concurrent_tracks'], 3)

    def test_serialization_roundtrip(self):
        """Test serialization and deserialization."""
        tracker = HypothesisTracker(config=self.config)

        # Create some tracks
        layer_0_features = [
            (i, 0.6, np.random.randn(self.d_model))
            for i in range(3)
        ]
        tracker.initialize_tracks(layer_0_features, token_pos=0)

        # Update
        layer_1_features = [
            (i, 0.65, np.random.randn(self.d_model))
            for i in range(3)
        ]
        tracker.update_tracks(1, layer_1_features)

        # Serialize and deserialize
        data = tracker.to_dict()
        new_tracker = HypothesisTracker.from_dict(data)

        # Compare
        self.assertEqual(len(new_tracker.tracks), len(tracker.tracks))
        self.assertEqual(
            new_tracker.stats['total_births'],
            tracker.stats['total_births']
        )

        # Compare individual tracks
        for orig, new in zip(tracker.tracks, new_tracker.tracks):
            self.assertEqual(orig.track_id, new.track_id)
            self.assertEqual(orig.length(), new.length())


if __name__ == '__main__':
    unittest.main()
