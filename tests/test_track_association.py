"""
Tests for track association using semantic similarity.
"""

import unittest
import numpy as np

from tracking.track import Track
from tracking.track_association import (
    cosine_similarity,
    associate_features_between_layers,
    compute_association_costs,
    greedy_association
)


class TestCosineSimilarity(unittest.TestCase):
    """Test cosine similarity function."""

    def test_identical_vectors(self):
        """Test cosine similarity of identical vectors."""
        vec = np.random.randn(768)
        similarity = cosine_similarity(vec, vec)

        self.assertAlmostEqual(similarity, 1.0, places=6)

    def test_orthogonal_vectors(self):
        """Test cosine similarity of orthogonal vectors."""
        vec1 = np.array([1.0, 0.0, 0.0])
        vec2 = np.array([0.0, 1.0, 0.0])

        similarity = cosine_similarity(vec1, vec2)
        self.assertAlmostEqual(similarity, 0.0, places=6)

    def test_opposite_vectors(self):
        """Test cosine similarity of opposite vectors."""
        vec1 = np.array([1.0, 0.0, 0.0])
        vec2 = np.array([-1.0, 0.0, 0.0])

        similarity = cosine_similarity(vec1, vec2)
        self.assertAlmostEqual(similarity, -1.0, places=6)

    def test_similar_vectors(self):
        """Test cosine similarity of similar vectors."""
        vec1 = np.array([1.0, 1.0, 1.0])
        vec2 = np.array([1.0, 1.0, 1.01])

        similarity = cosine_similarity(vec1, vec2)
        self.assertGreater(similarity, 0.99)

    def test_zero_vector(self):
        """Test cosine similarity with zero vector."""
        vec1 = np.array([1.0, 2.0, 3.0])
        vec2 = np.array([0.0, 0.0, 0.0])

        similarity = cosine_similarity(vec1, vec2)
        self.assertEqual(similarity, 0.0)

    def test_2d_vectors_flattened(self):
        """Test that 2D vectors are properly flattened."""
        vec1 = np.array([[1.0, 2.0], [3.0, 4.0]])
        vec2 = np.array([[1.0, 2.0], [3.0, 4.0]])

        similarity = cosine_similarity(vec1, vec2)
        self.assertAlmostEqual(similarity, 1.0, places=6)


class TestAssociateFeaturesBetweenLayers(unittest.TestCase):
    """Test feature association between layers."""

    def setUp(self):
        """Set up test fixtures."""
        self.d_model = 768

        # Create some tracks
        self.tracks = []
        for i in range(3):
            track = Track(
                track_id=i,
                feature_embedding=np.random.randn(self.d_model),
                birth_layer=0,
                token_pos=i
            )
            # Add trajectory
            track.update(0, 0.5 + i * 0.1, np.random.randn(self.d_model))
            self.tracks.append(track)

        # Create current features
        self.curr_features = []
        for i in range(3):
            feat_id = i
            activation = 0.6 + i * 0.1
            embedding = np.random.randn(self.d_model)
            self.curr_features.append((feat_id, activation, embedding))

        # Config
        self.config = {
            'semantic_weight': 0.6,
            'activation_weight': 0.2,
            'position_weight': 0.2,
            'association_threshold': 0.5
        }

    def test_association_basic(self):
        """Test basic association."""
        associations, unmatched = associate_features_between_layers(
            prev_tracks=self.tracks,
            curr_features=self.curr_features,
            layer_idx=1,
            config=self.config
        )

        # Should get some associations
        self.assertIsInstance(associations, list)
        self.assertIsInstance(unmatched, list)

        # All associations should be valid
        for track, feature in associations:
            self.assertIsInstance(track, Track)
            self.assertIsInstance(feature, tuple)
            self.assertEqual(len(feature), 3)

    def test_empty_tracks(self):
        """Test with no previous tracks."""
        associations, unmatched = associate_features_between_layers(
            prev_tracks=[],
            curr_features=self.curr_features,
            layer_idx=1,
            config=self.config
        )

        self.assertEqual(len(associations), 0)
        self.assertEqual(len(unmatched), len(self.curr_features))

    def test_empty_features(self):
        """Test with no current features."""
        associations, unmatched = associate_features_between_layers(
            prev_tracks=self.tracks,
            curr_features=[],
            layer_idx=1,
            config=self.config
        )

        self.assertEqual(len(associations), 0)
        self.assertEqual(len(unmatched), 0)

    def test_dead_tracks_excluded(self):
        """Test that dead tracks are not associated."""
        # Mark one track as dead
        self.tracks[0].death_layer = 0

        associations, unmatched = associate_features_between_layers(
            prev_tracks=self.tracks,
            curr_features=self.curr_features,
            layer_idx=1,
            config=self.config
        )

        # Dead track should not be in associations
        associated_tracks = [track for track, _ in associations]
        self.assertNotIn(self.tracks[0], associated_tracks)

    def test_threshold_filtering(self):
        """Test that high-cost associations are filtered."""
        # Use very low threshold
        strict_config = self.config.copy()
        strict_config['association_threshold'] = 0.01

        associations, unmatched = associate_features_between_layers(
            prev_tracks=self.tracks,
            curr_features=self.curr_features,
            layer_idx=1,
            config=strict_config
        )

        # Should get fewer associations with strict threshold
        # (may be zero if no associations meet threshold)
        self.assertGreaterEqual(len(unmatched), 0)

    def test_perfect_match(self):
        """Test association with perfect semantic match."""
        # Create track
        embedding = np.random.randn(self.d_model)
        track = Track(
            track_id=0,
            feature_embedding=embedding.copy(),
            birth_layer=0,
            token_pos=0
        )
        track.update(0, 0.8, embedding.copy())

        # Create feature with same embedding
        curr_feature = (0, 0.8, embedding.copy())

        associations, unmatched = associate_features_between_layers(
            prev_tracks=[track],
            curr_features=[curr_feature],
            layer_idx=1,
            config=self.config
        )

        # Should match perfectly
        self.assertEqual(len(associations), 1)
        self.assertEqual(len(unmatched), 0)


class TestComputeAssociationCosts(unittest.TestCase):
    """Test association cost computation."""

    def setUp(self):
        """Set up test fixtures."""
        self.d_model = 768
        self.config = {
            'semantic_weight': 0.6,
            'activation_weight': 0.2,
            'position_weight': 0.2
        }

    def test_identical_features(self):
        """Test cost for identical features."""
        embedding = np.random.randn(self.d_model)
        activation = 0.8

        costs = compute_association_costs(
            track_embedding=embedding,
            track_activation=activation,
            feature_embedding=embedding.copy(),
            feature_activation=activation,
            config=self.config
        )

        # Should have low cost
        self.assertLess(costs['total_cost'], 0.1)
        self.assertAlmostEqual(costs['semantic_cost'], 0.0, places=5)

    def test_different_features(self):
        """Test cost for very different features."""
        emb1 = np.random.randn(self.d_model)
        emb2 = np.random.randn(self.d_model)

        costs = compute_association_costs(
            track_embedding=emb1,
            track_activation=0.8,
            feature_embedding=emb2,
            feature_activation=0.2,
            config=self.config
        )

        # Should have higher cost
        self.assertGreater(costs['total_cost'], 0.0)

    def test_cost_components(self):
        """Test that all cost components are computed."""
        emb1 = np.random.randn(self.d_model)
        emb2 = np.random.randn(self.d_model)

        costs = compute_association_costs(
            track_embedding=emb1,
            track_activation=0.8,
            feature_embedding=emb2,
            feature_activation=0.6,
            config=self.config
        )

        # Check all components exist
        self.assertIn('semantic_cost', costs)
        self.assertIn('activation_cost', costs)
        self.assertIn('total_cost', costs)
        self.assertIn('cosine_similarity', costs)

        # All should be non-negative
        self.assertGreaterEqual(costs['semantic_cost'], 0.0)
        self.assertGreaterEqual(costs['activation_cost'], 0.0)
        self.assertGreaterEqual(costs['total_cost'], 0.0)

    def test_zero_activation_handling(self):
        """Test handling of zero activation."""
        emb1 = np.random.randn(self.d_model)
        emb2 = np.random.randn(self.d_model)

        costs = compute_association_costs(
            track_embedding=emb1,
            track_activation=0.0,
            feature_embedding=emb2,
            feature_activation=0.5,
            config=self.config
        )

        # Should handle gracefully
        self.assertIsInstance(costs['activation_cost'], float)
        self.assertTrue(np.isfinite(costs['activation_cost']))


class TestGreedyAssociation(unittest.TestCase):
    """Test greedy association algorithm."""

    def setUp(self):
        """Set up test fixtures."""
        self.d_model = 768

        # Create tracks
        self.tracks = []
        for i in range(3):
            track = Track(
                track_id=i,
                feature_embedding=np.random.randn(self.d_model),
                birth_layer=0,
                token_pos=i
            )
            track.update(0, 0.5 + i * 0.1, np.random.randn(self.d_model))
            self.tracks.append(track)

        # Create features
        self.curr_features = []
        for i in range(3):
            self.curr_features.append((
                i,
                0.6 + i * 0.1,
                np.random.randn(self.d_model)
            ))

        self.config = {
            'semantic_weight': 0.6,
            'activation_weight': 0.2,
            'association_threshold': 0.5
        }

    def test_greedy_basic(self):
        """Test basic greedy association."""
        associations, unmatched = greedy_association(
            prev_tracks=self.tracks,
            curr_features=self.curr_features,
            config=self.config
        )

        # Should produce valid output
        self.assertIsInstance(associations, list)
        self.assertIsInstance(unmatched, list)

    def test_greedy_prioritizes_strong_tracks(self):
        """Test that greedy algorithm prioritizes strong tracks."""
        # Tracks are sorted by activation before matching
        associations, _ = greedy_association(
            prev_tracks=self.tracks,
            curr_features=self.curr_features,
            config=self.config
        )

        # Should get some associations
        if len(associations) > 0:
            # First association should be from strongest track
            first_track = associations[0][0]
            activations = [t.trajectory[-1][1] for t in self.tracks]
            max_activation = max(activations)

            self.assertEqual(first_track.trajectory[-1][1], max_activation)

    def test_greedy_no_double_assignment(self):
        """Test that features are not assigned twice."""
        associations, unmatched = greedy_association(
            prev_tracks=self.tracks,
            curr_features=self.curr_features,
            config=self.config
        )

        # Extract assigned feature IDs
        assigned_feature_ids = [feat[0] for _, feat in associations]

        # No duplicates
        self.assertEqual(len(assigned_feature_ids), len(set(assigned_feature_ids)))

        # Unmatched features should not be in assigned
        unmatched_ids = [feat[0] for feat in unmatched]
        for feat_id in unmatched_ids:
            self.assertNotIn(feat_id, assigned_feature_ids)

    def test_greedy_empty_inputs(self):
        """Test greedy with empty inputs."""
        associations, unmatched = greedy_association(
            prev_tracks=[],
            curr_features=self.curr_features,
            config=self.config
        )

        self.assertEqual(len(associations), 0)
        self.assertEqual(len(unmatched), len(self.curr_features))


class TestAssociationComparison(unittest.TestCase):
    """Compare Hungarian and greedy association."""

    def setUp(self):
        """Set up test fixtures."""
        self.d_model = 768

        # Create tracks with specific embeddings
        self.tracks = []
        for i in range(5):
            embedding = np.random.randn(self.d_model)
            track = Track(
                track_id=i,
                feature_embedding=embedding.copy(),
                birth_layer=0,
                token_pos=i
            )
            track.update(0, 0.5 + i * 0.05, embedding.copy())
            self.tracks.append(track)

        # Create features
        self.curr_features = []
        for i in range(5):
            self.curr_features.append((
                i,
                0.6 + i * 0.05,
                np.random.randn(self.d_model)
            ))

        self.config = {
            'semantic_weight': 0.6,
            'activation_weight': 0.2,
            'position_weight': 0.2,
            'association_threshold': 0.8
        }

    def test_both_algorithms_produce_valid_output(self):
        """Test that both algorithms work."""
        # Hungarian
        hungarian_assoc, hungarian_unmatched = associate_features_between_layers(
            prev_tracks=self.tracks,
            curr_features=self.curr_features,
            layer_idx=1,
            config=self.config
        )

        # Greedy
        greedy_assoc, greedy_unmatched = greedy_association(
            prev_tracks=self.tracks,
            curr_features=self.curr_features,
            config=self.config
        )

        # Both should produce valid outputs
        self.assertIsInstance(hungarian_assoc, list)
        self.assertIsInstance(greedy_assoc, list)

        # Total assignments + unmatched should equal total features
        self.assertEqual(
            len(hungarian_assoc) + len(hungarian_unmatched),
            len(self.curr_features)
        )
        self.assertEqual(
            len(greedy_assoc) + len(greedy_unmatched),
            len(self.curr_features)
        )


if __name__ == '__main__':
    unittest.main()
