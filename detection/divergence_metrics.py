"""
Divergence metrics for hallucination detection.

Computes metrics based on track patterns to distinguish
factual from hallucinated text.
"""

import numpy as np
from typing import List, Dict
from scipy.stats import entropy as scipy_entropy

from tracking.hypothesis_tracker import HypothesisTracker
from tracking.track import Track


class DivergenceMetrics:
    """
    Compute divergence metrics from hypothesis tracks.

    Key metrics:
    1. Entropy - Shannon entropy of activation distribution
    2. Churn - Rate of track births and deaths
    3. Competition - Number of competing strong tracks
    4. Stability - Variance in track activations
    5. Dominance - Strength of dominant track
    6. Track density - Average number of alive tracks per layer
    """

    def __init__(self):
        """Initialize metrics computer."""
        pass

    def compute_all_metrics(
        self,
        tracker: HypothesisTracker,
        num_layers: int = 12
    ) -> Dict[str, float]:
        """
        Compute all divergence metrics from tracker.

        Args:
            tracker: HypothesisTracker with completed tracking.
            num_layers: Total number of layers in model.

        Returns:
            Dictionary of metric name -> value.
        """
        metrics = {}

        # 1. Entropy metrics
        metrics.update(self.compute_entropy_metrics(tracker, num_layers))

        # 2. Churn metrics
        metrics.update(self.compute_churn_metrics(tracker, num_layers))

        # 3. Competition metrics
        metrics.update(self.compute_competition_metrics(tracker, num_layers))

        # 4. Stability metrics
        metrics.update(self.compute_stability_metrics(tracker))

        # 5. Dominance metrics
        metrics.update(self.compute_dominance_metrics(tracker, num_layers))

        # 6. Track density metrics
        metrics.update(self.compute_density_metrics(tracker, num_layers))

        return metrics

    def compute_entropy_metrics(
        self,
        tracker: HypothesisTracker,
        num_layers: int
    ) -> Dict[str, float]:
        """
        Compute entropy-based metrics.

        High entropy indicates many competing hypotheses.

        Args:
            tracker: HypothesisTracker instance.
            num_layers: Total number of layers.

        Returns:
            Dict with entropy metrics.
        """
        metrics = {}

        # Compute entropy at each layer
        layer_entropies = []

        for layer_idx in range(num_layers):
            tracks_at_layer = tracker.get_alive_tracks(layer_idx)

            if not tracks_at_layer:
                layer_entropies.append(0.0)
                continue

            # Get activations
            activations = [
                t.get_activation_at(layer_idx) or 0.0
                for t in tracks_at_layer
            ]

            # Normalize to probability distribution
            total = sum(activations)
            if total > 0:
                probs = [a / total for a in activations]
                ent = scipy_entropy(probs, base=2)
                layer_entropies.append(ent)
            else:
                layer_entropies.append(0.0)

        # Aggregate statistics
        metrics['entropy_mean'] = np.mean(layer_entropies)
        metrics['entropy_max'] = np.max(layer_entropies)
        metrics['entropy_std'] = np.std(layer_entropies)
        metrics['entropy_final'] = layer_entropies[-1] if layer_entropies else 0.0

        return metrics

    def compute_churn_metrics(
        self,
        tracker: HypothesisTracker,
        num_layers: int
    ) -> Dict[str, float]:
        """
        Compute track churn (birth/death) metrics.

        High churn indicates unstable hypothesis formation.

        Args:
            tracker: HypothesisTracker instance.
            num_layers: Total number of layers.

        Returns:
            Dict with churn metrics.
        """
        stats = tracker.get_statistics()

        total_tracks = stats['total_tracks']
        total_births = stats['total_births']
        total_deaths = stats['total_deaths']

        metrics = {
            'total_births': total_births,
            'total_deaths': total_deaths,
            'birth_rate': total_births / num_layers if num_layers > 0 else 0.0,
            'death_rate': total_deaths / num_layers if num_layers > 0 else 0.0,
            'churn_ratio': (total_births + total_deaths) / (2 * num_layers) if num_layers > 0 else 0.0,
            'survival_rate': stats['survival_rate']
        }

        return metrics

    def compute_competition_metrics(
        self,
        tracker: HypothesisTracker,
        num_layers: int
    ) -> Dict[str, float]:
        """
        Compute competition metrics.

        High competition indicates multiple strong hypotheses.

        Args:
            tracker: HypothesisTracker instance.
            num_layers: Total number of layers.

        Returns:
            Dict with competition metrics.
        """
        # Count competing tracks at each layer
        competition_counts = []
        competition_threshold = 0.5  # Tracks with activation > 0.5

        for layer_idx in range(num_layers):
            competing = tracker.get_competing_tracks(
                layer_idx=layer_idx,
                threshold=competition_threshold
            )
            competition_counts.append(len(competing))

        metrics = {
            'competition_mean': np.mean(competition_counts),
            'competition_max': np.max(competition_counts),
            'competition_std': np.std(competition_counts),
            'competition_final': competition_counts[-1] if competition_counts else 0
        }

        # Compute layers with high competition (> 2 competing tracks)
        high_competition_layers = sum(1 for c in competition_counts if c > 2)
        metrics['high_competition_ratio'] = high_competition_layers / num_layers if num_layers > 0 else 0.0

        return metrics

    def compute_stability_metrics(
        self,
        tracker: HypothesisTracker
    ) -> Dict[str, float]:
        """
        Compute stability metrics from track trajectories.

        Stable tracks have low activation variance.

        Args:
            tracker: HypothesisTracker instance.

        Returns:
            Dict with stability metrics.
        """
        if not tracker.tracks:
            return {
                'stability_mean': 0.0,
                'stability_min': 0.0,
                'unstable_track_ratio': 0.0
            }

        # Get variance for each track
        variances = []
        for track in tracker.tracks:
            if track.length() >= 2:
                variances.append(track.activation_variance())

        if not variances:
            return {
                'stability_mean': 0.0,
                'stability_min': 0.0,
                'unstable_track_ratio': 0.0
            }

        # Higher variance = less stable
        # Define stability as 1 / (1 + variance)
        stabilities = [1.0 / (1.0 + v) for v in variances]

        metrics = {
            'stability_mean': np.mean(stabilities),
            'stability_min': np.min(stabilities),
            'unstable_track_ratio': sum(1 for s in stabilities if s < 0.5) / len(stabilities)
        }

        return metrics

    def compute_dominance_metrics(
        self,
        tracker: HypothesisTracker,
        num_layers: int
    ) -> Dict[str, float]:
        """
        Compute dominance metrics.

        Strong dominance indicates one clear hypothesis.

        Args:
            tracker: HypothesisTracker instance.
            num_layers: Total number of layers.

        Returns:
            Dict with dominance metrics.
        """
        dominance_scores = []

        for layer_idx in range(num_layers):
            tracks_at_layer = tracker.get_alive_tracks(layer_idx)

            if not tracks_at_layer:
                dominance_scores.append(0.0)
                continue

            # Get all activations
            activations = [
                t.get_activation_at(layer_idx) or 0.0
                for t in tracks_at_layer
            ]

            if not activations or max(activations) == 0:
                dominance_scores.append(0.0)
                continue

            # Dominance = max / sum (how much of total activation is in top track)
            max_act = max(activations)
            total_act = sum(activations)

            dominance = max_act / total_act if total_act > 0 else 0.0
            dominance_scores.append(dominance)

        metrics = {
            'dominance_mean': np.mean(dominance_scores),
            'dominance_min': np.min(dominance_scores),
            'dominance_final': dominance_scores[-1] if dominance_scores else 0.0
        }

        # Weak dominance ratio (dominance < 0.5)
        weak_dominance_count = sum(1 for d in dominance_scores if d < 0.5)
        metrics['weak_dominance_ratio'] = weak_dominance_count / num_layers if num_layers > 0 else 0.0

        return metrics

    def compute_density_metrics(
        self,
        tracker: HypothesisTracker,
        num_layers: int
    ) -> Dict[str, float]:
        """
        Compute track density metrics.

        High density indicates many simultaneous hypotheses.

        Args:
            tracker: HypothesisTracker instance.
            num_layers: Total number of layers.

        Returns:
            Dict with density metrics.
        """
        alive_counts = []

        for layer_idx in range(num_layers):
            alive = tracker.get_alive_tracks(layer_idx)
            alive_counts.append(len(alive))

        stats = tracker.get_statistics()

        metrics = {
            'density_mean': np.mean(alive_counts),
            'density_max': np.max(alive_counts),
            'density_std': np.std(alive_counts),
            'max_concurrent_tracks': stats['max_concurrent_tracks']
        }

        return metrics

    def get_feature_vector(
        self,
        tracker: HypothesisTracker,
        num_layers: int = 12
    ) -> np.ndarray:
        """
        Get complete feature vector for classification.

        Args:
            tracker: HypothesisTracker instance.
            num_layers: Total number of layers.

        Returns:
            Feature vector as numpy array.
        """
        metrics = self.compute_all_metrics(tracker, num_layers)

        # Define feature order (for consistency)
        feature_names = [
            # Entropy (4 features)
            'entropy_mean', 'entropy_max', 'entropy_std', 'entropy_final',

            # Churn (6 features)
            'total_births', 'total_deaths', 'birth_rate', 'death_rate',
            'churn_ratio', 'survival_rate',

            # Competition (5 features)
            'competition_mean', 'competition_max', 'competition_std',
            'competition_final', 'high_competition_ratio',

            # Stability (3 features)
            'stability_mean', 'stability_min', 'unstable_track_ratio',

            # Dominance (4 features)
            'dominance_mean', 'dominance_min', 'dominance_final',
            'weak_dominance_ratio',

            # Density (4 features)
            'density_mean', 'density_max', 'density_std',
            'max_concurrent_tracks'
        ]

        # Extract features in order
        features = [metrics.get(name, 0.0) for name in feature_names]

        return np.array(features, dtype=np.float32)

    def get_feature_names(self) -> List[str]:
        """
        Get names of features in feature vector.

        Returns:
            List of feature names.
        """
        return [
            # Entropy
            'entropy_mean', 'entropy_max', 'entropy_std', 'entropy_final',

            # Churn
            'total_births', 'total_deaths', 'birth_rate', 'death_rate',
            'churn_ratio', 'survival_rate',

            # Competition
            'competition_mean', 'competition_max', 'competition_std',
            'competition_final', 'high_competition_ratio',

            # Stability
            'stability_mean', 'stability_min', 'unstable_track_ratio',

            # Dominance
            'dominance_mean', 'dominance_min', 'dominance_final',
            'weak_dominance_ratio',

            # Density
            'density_mean', 'density_max', 'density_std',
            'max_concurrent_tracks'
        ]


def compute_divergence_score(
    factual_tracker: HypothesisTracker,
    hallucinated_tracker: HypothesisTracker,
    num_layers: int = 12
) -> float:
    """
    Compute simple divergence score between factual and hallucinated.

    Higher score indicates more divergence (more likely hallucination).

    Args:
        factual_tracker: Tracker for factual completion.
        hallucinated_tracker: Tracker for hallucinated completion.
        num_layers: Total number of layers.

    Returns:
        Divergence score (0-1 range, higher = more hallucinated).
    """
    metrics_computer = DivergenceMetrics()

    factual_metrics = metrics_computer.compute_all_metrics(
        factual_tracker, num_layers
    )
    halluc_metrics = metrics_computer.compute_all_metrics(
        hallucinated_tracker, num_layers
    )

    # Simple heuristic: hallucinations have higher entropy, churn, competition
    # and lower dominance, stability

    score = 0.0

    # Entropy (higher = more hallucinated)
    score += halluc_metrics['entropy_mean'] - factual_metrics['entropy_mean']

    # Churn (higher = more hallucinated)
    score += halluc_metrics['churn_ratio'] - factual_metrics['churn_ratio']

    # Competition (higher = more hallucinated)
    score += halluc_metrics['competition_mean'] - factual_metrics['competition_mean']

    # Dominance (lower = more hallucinated)
    score += factual_metrics['dominance_mean'] - halluc_metrics['dominance_mean']

    # Stability (lower = more hallucinated)
    score += factual_metrics['stability_mean'] - halluc_metrics['stability_mean']

    # Normalize to [0, 1] range (rough approximation)
    score = np.clip(score / 5.0, 0.0, 1.0)

    return score
