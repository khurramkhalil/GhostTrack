"""
Track association using semantic similarity.

Associates features across layers using cosine similarity
(NOT feature IDs, as features from different SAEs are independent).
"""

import numpy as np
from typing import List, Tuple, Dict
from scipy.optimize import linear_sum_assignment


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Compute cosine similarity between two vectors.

    Args:
        a: First vector.
        b: Second vector.

    Returns:
        Cosine similarity in [0, 1] (after handling negatives).
    """
    # Ensure vectors are 1D
    a = a.flatten()
    b = b.flatten()

    # Compute cosine similarity
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)

    if norm_a == 0 or norm_b == 0:
        return 0.0

    similarity = dot_product / (norm_a * norm_b)

    # Clamp to [-1, 1] to handle numerical errors
    similarity = np.clip(similarity, -1.0, 1.0)

    return float(similarity)


def associate_features_between_layers(
    prev_tracks: List,  # List[Track]
    curr_features: List[Tuple[int, float, np.ndarray]],
    layer_idx: int,
    config: Dict
) -> Tuple[List[Tuple], List[Tuple]]:
    """
    Associate features between layers using semantic similarity.

    Uses Hungarian algorithm for optimal bipartite matching based on:
    1. Semantic similarity (cosine distance)
    2. Activation change (how much activation changed)
    3. Spatial proximity (token position)

    Args:
        prev_tracks: List of Track objects from previous layer.
        curr_features: List of (feature_id, activation, embedding) tuples.
        layer_idx: Current layer index.
        config: Configuration dict with weights and thresholds.

    Returns:
        Tuple of (associations, unmatched_features):
        - associations: List of (Track, (feat_id, activation, embedding)) pairs
        - unmatched_features: List of unmatched (feat_id, activation, embedding)
    """
    # Filter to only alive tracks
    alive_tracks = [t for t in prev_tracks if t.is_alive()]

    if len(alive_tracks) == 0 or len(curr_features) == 0:
        return [], curr_features

    # Build cost matrix
    n_tracks = len(alive_tracks)
    n_features = len(curr_features)
    cost_matrix = np.zeros((n_tracks, n_features))

    for i, track in enumerate(alive_tracks):
        # Get last embedding and activation from track
        last_embedding = track.trajectory[-1][2]  # np.ndarray
        last_activation = track.trajectory[-1][1]  # float

        for j, (feat_id, feat_act, feat_emb) in enumerate(curr_features):
            # Cost component 1: Semantic distance (1 - cosine similarity)
            cos_sim = cosine_similarity(last_embedding, feat_emb)
            semantic_cost = 1.0 - cos_sim

            # Cost component 2: Activation change (normalized)
            if last_activation > 1e-6:
                activation_cost = abs(last_activation - feat_act) / (last_activation + feat_act)
            else:
                activation_cost = 1.0 if feat_act > 0.1 else 0.0

            # Cost component 3: Spatial proximity (token position)
            # Note: Features don't have inherent position, but we track token_pos
            # For simplicity, assume same position (could enhance later)
            position_cost = 0.0  # Placeholder

            # Total weighted cost
            cost_matrix[i, j] = (
                config.get('semantic_weight', 0.6) * semantic_cost +
                config.get('activation_weight', 0.2) * activation_cost +
                config.get('position_weight', 0.2) * position_cost
            )

    # Use Hungarian algorithm for optimal assignment
    track_indices, feature_indices = linear_sum_assignment(cost_matrix)

    # Filter by association threshold
    associations = []
    matched_feature_indices = set()

    association_threshold = config.get('association_threshold', 0.5)

    for i, j in zip(track_indices, feature_indices):
        if cost_matrix[i, j] < association_threshold:
            track = alive_tracks[i]
            feature = curr_features[j]
            associations.append((track, feature))
            matched_feature_indices.add(j)

    # Find unmatched features
    unmatched_features = [
        curr_features[j]
        for j in range(n_features)
        if j not in matched_feature_indices
    ]

    return associations, unmatched_features


def compute_association_costs(
    track_embedding: np.ndarray,
    track_activation: float,
    feature_embedding: np.ndarray,
    feature_activation: float,
    config: Dict
) -> Dict[str, float]:
    """
    Compute individual cost components for association.

    Args:
        track_embedding: Track's feature embedding.
        track_activation: Track's last activation.
        feature_embedding: Current feature embedding.
        feature_activation: Current feature activation.
        config: Configuration with weights.

    Returns:
        Dict with cost components.
    """
    # Semantic cost
    cos_sim = cosine_similarity(track_embedding, feature_embedding)
    semantic_cost = 1.0 - cos_sim

    # Activation cost
    if track_activation > 1e-6:
        activation_cost = abs(track_activation - feature_activation) / \
                         (track_activation + feature_activation)
    else:
        activation_cost = 1.0 if feature_activation > 0.1 else 0.0

    # Total cost
    total_cost = (
        config.get('semantic_weight', 0.6) * semantic_cost +
        config.get('activation_weight', 0.2) * activation_cost
    )

    return {
        'semantic_cost': semantic_cost,
        'activation_cost': activation_cost,
        'total_cost': total_cost,
        'cosine_similarity': cos_sim
    }


def greedy_association(
    prev_tracks: List,
    curr_features: List[Tuple[int, float, np.ndarray]],
    config: Dict
) -> Tuple[List[Tuple], List[Tuple]]:
    """
    Greedy alternative to Hungarian algorithm (faster but sub-optimal).

    Args:
        prev_tracks: List of Track objects.
        curr_features: List of current features.
        config: Configuration dict.

    Returns:
        Tuple of (associations, unmatched_features).
    """
    alive_tracks = [t for t in prev_tracks if t.is_alive()]

    if not alive_tracks or not curr_features:
        return [], curr_features

    associations = []
    remaining_features = list(curr_features)

    # Sort tracks by last activation (prioritize strong tracks)
    sorted_tracks = sorted(
        alive_tracks,
        key=lambda t: t.trajectory[-1][1],
        reverse=True
    )

    for track in sorted_tracks:
        if not remaining_features:
            break

        last_embedding = track.trajectory[-1][2]
        last_activation = track.trajectory[-1][1]

        # Find best matching feature
        best_cost = float('inf')
        best_feature = None
        best_idx = -1

        for idx, (feat_id, feat_act, feat_emb) in enumerate(remaining_features):
            costs = compute_association_costs(
                last_embedding, last_activation,
                feat_emb, feat_act,
                config
            )

            if costs['total_cost'] < best_cost:
                best_cost = costs['total_cost']
                best_feature = (feat_id, feat_act, feat_emb)
                best_idx = idx

        # Associate if cost is below threshold
        if best_cost < config.get('association_threshold', 0.5):
            associations.append((track, best_feature))
            remaining_features.pop(best_idx)

    return associations, remaining_features


def associate_by_feature_id(
    prev_tracks: List,
    curr_features: List[Tuple[int, float, np.ndarray]],
    config: Dict
) -> Tuple[List[Tuple], List[Tuple]]:
    """
    Associate features purely by matching Feature IDs (indices).
    This serves as a random/null baseline for ablation studies.
    
    Args:
        prev_tracks: List of Track objects (must have 'last_feat_id' in metadata).
        curr_features: List of (feat_id, activation, embedding) tuples.
        config: Configuration dict (unused but kept for API consistency).
        
    Returns:
        Tuple of (associations, unmatched_features).
    """
    alive_tracks = [t for t in prev_tracks if t.is_alive()]
    
    if not alive_tracks or not curr_features:
        return [], curr_features

    associations = []
    
    # Map feat_id -> (feat_id, act, emb) for O(1) lookup
    # Note: If multiple features have same ID (unlikely in ONE layer), this overwrites.
    # But current_features comes from one layer of one SAE, so IDs are unique.
    feature_map = {f[0]: f for f in curr_features}
    matched_feat_ids = set()
    
    for track in alive_tracks:
        # We rely on Track metadata to store the feature ID it represents
        last_feat_id = track.metadata.get('last_feat_id', -1)
        
        if last_feat_id in feature_map:
            # Match found!
            associations.append((track, feature_map[last_feat_id]))
            matched_feat_ids.add(last_feat_id)
            
    # Identify unmatched features
    unmatched_features = [
        f for f in curr_features 
        if f[0] not in matched_feat_ids
    ]
    
    return associations, unmatched_features
