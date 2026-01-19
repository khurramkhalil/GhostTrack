"""
Hypothesis Tracker - manages track lifecycle across all layers.
"""

from typing import List, Optional, Dict
import numpy as np

from .track import Track
from .track_association import associate_features_between_layers, greedy_association, associate_by_feature_id


class HypothesisTracker:
    """
    Manages hypothesis tracks across transformer layers.

    Tracks semantic hypotheses as they emerge (birth),
    evolve (update), and disappear (death) through the network.
    """

    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize hypothesis tracker.

        Args:
            config: Configuration dict with tracking parameters.
        """
        self.config = config or self._default_config()
        self.tracks: List[Track] = []
        self.track_id_counter = 0

        # Statistics
        self.stats = {
            'total_births': 0,
            'total_deaths': 0,
            'max_concurrent_tracks': 0
        }

    @staticmethod
    def _default_config() -> Dict:
        """Get default configuration."""
        return {
            'birth_threshold': 0.5,
            'death_threshold': 0.1,
            'association_threshold': 0.5,
            'semantic_weight': 0.6,
            'activation_weight': 0.2,
            'position_weight': 0.2,
            'top_k_features': 50,
            'use_greedy': False  # Use Hungarian by default
        }

    def initialize_tracks(
        self,
        layer_0_features: List[tuple],  # List[(feat_id, activation, embedding)]
        token_pos: int = 0
    ):
        """
        Create initial tracks from layer 0 features.

        Args:
            layer_0_features: List of (feature_id, activation, embedding) tuples.
            token_pos: Token position these features correspond to.
        """
        birth_threshold = self.config['birth_threshold']

        for feat_id, activation, embedding in layer_0_features:
            if activation > birth_threshold:
                # Create new track
                track = Track(
                    track_id=self.track_id_counter,
                    feature_embedding=embedding.copy(),
                    birth_layer=0,
                    token_pos=token_pos
                )
                track.metadata['last_feat_id'] = feat_id

                # Add initial observation
                track.update(0, activation, embedding.copy())

                self.tracks.append(track)
                self.track_id_counter += 1
                self.stats['total_births'] += 1

        # Update max concurrent
        self.stats['max_concurrent_tracks'] = max(
            self.stats['max_concurrent_tracks'],
            len(self.tracks)
        )

    def update_tracks(
        self,
        layer_idx: int,
        current_features: List[tuple]  # List[(feat_id, activation, embedding)]
    ):
        """
        Update tracks with features from current layer.

        This performs:
        1. Association: Match current features to existing tracks
        2. Update: Update matched tracks
        3. Death: Mark unmatched tracks as dead
        4. Birth: Create new tracks for unmatched features

        Args:
            layer_idx: Current layer index.
            current_features: List of (feature_id, activation, embedding) tuples.
        """
        # Choose association algorithm
        # Choose association algorithm
        if self.config.get('disable_association', False):
            # No association: all features start new tracks
            associations = []
            unmatched_features = current_features
        elif self.config.get('use_feature_id_matching', False):
            associations, unmatched_features = associate_by_feature_id(
                self.tracks, current_features, self.config
            )
        elif self.config.get('use_greedy', False):
            associations, unmatched_features = greedy_association(
                self.tracks, current_features, self.config
            )
        else:
            associations, unmatched_features = associate_features_between_layers(
                self.tracks, current_features, layer_idx, self.config
            )

        # Track which tracks were matched
        matched_track_ids = set()

        # Update matched tracks
        for track, (feat_id, activation, embedding) in associations:
            track.update(layer_idx, activation, embedding.copy())
            track.metadata['last_feat_id'] = feat_id
            matched_track_ids.add(track.track_id)

        # Mark unmatched alive tracks as dead
        for track in self.tracks:
            if track.is_alive() and track.track_id not in matched_track_ids:
                # Check if track was active in previous layer
                if layer_idx - 1 >= track.birth_layer:
                    # Track died at previous layer
                    track.death_layer = layer_idx - 1
                    self.stats['total_deaths'] += 1

        # Create new tracks for unmatched features
        birth_threshold = self.config['birth_threshold']

        for feat_id, activation, embedding in unmatched_features:
            if activation > birth_threshold:
                # Create new track (birth event)
                track = Track(
                    track_id=self.track_id_counter,
                    feature_embedding=embedding.copy(),
                    birth_layer=layer_idx,
                    token_pos=feat_id  # Using feat_id as proxy for position
                )
                track.metadata['last_feat_id'] = feat_id

                track.update(layer_idx, activation, embedding.copy())

                self.tracks.append(track)
                self.track_id_counter += 1
                self.stats['total_births'] += 1

        # Update statistics
        alive_count = len(self.get_alive_tracks())
        self.stats['max_concurrent_tracks'] = max(
            self.stats['max_concurrent_tracks'],
            alive_count
        )

    def get_alive_tracks(self, layer_idx: Optional[int] = None) -> List[Track]:
        """
        Get tracks that are alive (not dead).

        Args:
            layer_idx: If specified, get tracks alive at this layer.
                      If None, get tracks currently alive.

        Returns:
            List of alive Track objects.
        """
        if layer_idx is None:
            return [t for t in self.tracks if t.is_alive()]

        # Get tracks alive at specific layer
        return [
            t for t in self.tracks
            if t.birth_layer <= layer_idx and
               (t.death_layer is None or t.death_layer >= layer_idx)
        ]

    def get_dead_tracks(self) -> List[Track]:
        """
        Get tracks that have died.

        Returns:
            List of dead Track objects.
        """
        return [t for t in self.tracks if not t.is_alive()]

    def get_tracks_by_layer(self, layer_idx: int) -> List[Track]:
        """
        Get all tracks that existed at a specific layer.

        Args:
            layer_idx: Layer index.

        Returns:
            List of Track objects active at that layer.
        """
        return self.get_alive_tracks(layer_idx)

    def get_dominant_track(self, layer_idx: int) -> Optional[Track]:
        """
        Get the dominant (highest activation) track at a layer.

        Args:
            layer_idx: Layer index.

        Returns:
            Track with highest activation, or None if no tracks.
        """
        tracks_at_layer = self.get_tracks_by_layer(layer_idx)

        if not tracks_at_layer:
            return None

        return max(
            tracks_at_layer,
            key=lambda t: t.get_activation_at(layer_idx) or 0.0
        )

    def get_competing_tracks(
        self,
        layer_idx: int,
        threshold: float = 0.5
    ) -> List[Track]:
        """
        Get tracks with high activation (competing) at a layer.

        Args:
            layer_idx: Layer index.
            threshold: Minimum activation to be considered competing.

        Returns:
            List of competing Track objects.
        """
        tracks_at_layer = self.get_tracks_by_layer(layer_idx)

        return [
            t for t in tracks_at_layer
            if (t.get_activation_at(layer_idx) or 0.0) > threshold
        ]

    def reset(self):
        """Reset tracker to initial state."""
        self.tracks = []
        self.track_id_counter = 0
        self.stats = {
            'total_births': 0,
            'total_deaths': 0,
            'max_concurrent_tracks': 0
        }

    def get_statistics(self) -> Dict:
        """
        Get tracking statistics.

        Returns:
            Dict with statistics.
        """
        alive = len(self.get_alive_tracks())
        dead = len(self.get_dead_tracks())

        return {
            **self.stats,
            'total_tracks': len(self.tracks),
            'alive_tracks': alive,
            'dead_tracks': dead,
            'survival_rate': alive / len(self.tracks) if self.tracks else 0.0
        }

    def summarize(self) -> str:
        """
        Get human-readable summary.

        Returns:
            Summary string.
        """
        stats = self.get_statistics()

        lines = [
            "Hypothesis Tracker Summary",
            "=" * 50,
            f"Total tracks: {stats['total_tracks']}",
            f"  Alive: {stats['alive_tracks']}",
            f"  Dead: {stats['dead_tracks']}",
            f"  Survival rate: {stats['survival_rate']:.2%}",
            f"",
            f"Events:",
            f"  Births: {stats['total_births']}",
            f"  Deaths: {stats['total_deaths']}",
            f"  Max concurrent: {stats['max_concurrent_tracks']}",
        ]

        return "\n".join(lines)

    def get_track_by_id(self, track_id: int) -> Optional[Track]:
        """
        Get track by ID.

        Args:
            track_id: Track ID to find.

        Returns:
            Track object or None if not found.
        """
        for track in self.tracks:
            if track.track_id == track_id:
                return track
        return None

    def to_dict(self) -> Dict:
        """
        Serialize tracker to dictionary.

        Returns:
            Dictionary representation.
        """
        return {
            'config': self.config,
            'tracks': [t.to_dict() for t in self.tracks],
            'track_id_counter': self.track_id_counter,
            'stats': self.stats
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'HypothesisTracker':
        """
        Deserialize tracker from dictionary.

        Args:
            data: Dictionary representation.

        Returns:
            HypothesisTracker instance.
        """
        tracker = cls(config=data.get('config'))
        tracker.track_id_counter = data.get('track_id_counter', 0)
        tracker.stats = data.get('stats', tracker.stats)

        # Reconstruct tracks
        for track_data in data.get('tracks', []):
            track = Track.from_dict(track_data)
            tracker.tracks.append(track)

        return tracker
