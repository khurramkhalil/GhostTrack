"""
Track dataclass representing a semantic hypothesis across layers.
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Optional
import numpy as np


@dataclass
class Track:
    """
    Represents a semantic hypothesis tracked across transformer layers.

    A track captures how a particular semantic concept evolves through
    the network, with birth and potential death events.
    """

    track_id: int
    """Unique identifier for this track."""

    feature_embedding: np.ndarray
    """Semantic representation of this hypothesis (SAE feature vector)."""

    birth_layer: int
    """Layer where this track was first detected."""

    token_pos: int
    """Token position in the sequence."""

    death_layer: Optional[int] = None
    """Layer where this track died (stopped being tracked). None if still alive."""

    trajectory: List[Tuple[int, float, np.ndarray]] = field(default_factory=list)
    """
    History of track evolution across layers.
    Each tuple is (layer_idx, activation_value, feature_embedding).
    """

    confidence_history: List[float] = field(default_factory=list)
    """Optional confidence scores per layer."""

    metadata: dict = field(default_factory=dict)
    """Additional metadata (e.g., feature labels, semantic tags)."""

    def update(self, layer: int, activation: float, embedding: np.ndarray):
        """
        Add new observation to track trajectory.

        Args:
            layer: Layer index.
            activation: Feature activation value at this layer.
            embedding: Feature embedding at this layer.
        """
        self.trajectory.append((layer, activation, embedding))
        self.feature_embedding = embedding  # Update to latest embedding

    def is_alive(self) -> bool:
        """
        Check if track is still active.

        Returns:
            True if track has not been marked as dead.
        """
        return self.death_layer is None

    def get_activation_at(self, layer: int) -> Optional[float]:
        """
        Get activation value at a specific layer.

        Args:
            layer: Layer index to query.

        Returns:
            Activation value at that layer, or None if not present.
        """
        for l, act, _ in self.trajectory:
            if l == layer:
                return act
        return None

    def get_embedding_at(self, layer: int) -> Optional[np.ndarray]:
        """
        Get feature embedding at a specific layer.

        Args:
            layer: Layer index to query.

        Returns:
            Feature embedding at that layer, or None if not present.
        """
        for l, _, emb in self.trajectory:
            if l == layer:
                return emb
        return None

    def layer_range(self) -> range:
        """
        Get range of layers this track exists in.

        Returns:
            Range from birth_layer to death_layer (or max if alive).
        """
        start = self.birth_layer
        end = self.death_layer + 1 if self.death_layer is not None else 100  # Large number
        return range(start, end)

    def max_activation(self) -> float:
        """
        Get maximum activation across entire trajectory.

        Returns:
            Maximum activation value.
        """
        if not self.trajectory:
            return 0.0
        return max(act for _, act, _ in self.trajectory)

    def mean_activation(self) -> float:
        """
        Get mean activation across trajectory.

        Returns:
            Mean activation value.
        """
        if not self.trajectory:
            return 0.0
        return np.mean([act for _, act, _ in self.trajectory])

    def final_activation(self) -> float:
        """
        Get final activation value.

        Returns:
            Activation at last layer in trajectory.
        """
        if not self.trajectory:
            return 0.0
        return self.trajectory[-1][1]

    def length(self) -> int:
        """
        Get trajectory length (number of layers tracked).

        Returns:
            Number of layers in trajectory.
        """
        return len(self.trajectory)

    def activation_variance(self) -> float:
        """
        Get variance of activations across trajectory.

        Returns:
            Variance of activation values.
        """
        if len(self.trajectory) < 2:
            return 0.0
        activations = [act for _, act, _ in self.trajectory]
        return float(np.var(activations))

    def is_stable(self, threshold: float = 0.1) -> bool:
        """
        Check if track has stable activations.

        Args:
            threshold: Maximum allowed variance for stability.

        Returns:
            True if variance is below threshold.
        """
        return self.activation_variance() < threshold

    def dominates_at_layer(self, layer: int, other_activations: List[float]) -> bool:
        """
        Check if this track dominates other tracks at a given layer.

        Args:
            layer: Layer to check.
            other_activations: Activations of other tracks at this layer.

        Returns:
            True if this track has highest activation.
        """
        my_activation = self.get_activation_at(layer)
        if my_activation is None:
            return False

        if not other_activations:
            return True

        return my_activation > max(other_activations)

    def __repr__(self) -> str:
        """String representation of track."""
        status = "ALIVE" if self.is_alive() else f"DEAD@L{self.death_layer}"
        return (
            f"Track(id={self.track_id}, "
            f"birth=L{self.birth_layer}, "
            f"status={status}, "
            f"length={self.length()}, "
            f"max_act={self.max_activation():.3f})"
        )

    def to_dict(self) -> dict:
        """
        Convert track to dictionary for serialization.

        Returns:
            Dictionary representation.
        """
        return {
            'track_id': self.track_id,
            'birth_layer': self.birth_layer,
            'death_layer': self.death_layer,
            'token_pos': self.token_pos,
            'trajectory': [
                {
                    'layer': layer,
                    'activation': float(activation),
                    'embedding': embedding.tolist()
                }
                for layer, activation, embedding in self.trajectory
            ],
            'max_activation': self.max_activation(),
            'mean_activation': self.mean_activation(),
            'final_activation': self.final_activation(),
            'length': self.length(),
            'is_alive': self.is_alive(),
            'metadata': self.metadata
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'Track':
        """
        Create track from dictionary.

        Args:
            data: Dictionary representation.

        Returns:
            Track instance.
        """
        track = cls(
            track_id=data['track_id'],
            feature_embedding=np.array(data['trajectory'][-1]['embedding'])  # Latest
            if data['trajectory'] else np.array([]),
            birth_layer=data['birth_layer'],
            token_pos=data['token_pos'],
            death_layer=data.get('death_layer'),
            metadata=data.get('metadata', {})
        )

        # Reconstruct trajectory
        for traj_point in data['trajectory']:
            track.trajectory.append((
                traj_point['layer'],
                traj_point['activation'],
                np.array(traj_point['embedding'])
            ))

        return track
