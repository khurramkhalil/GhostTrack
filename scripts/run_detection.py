#!/usr/bin/env python3
"""
Phase 4: Hallucination Detection Runner

Trains and evaluates hallucination detector on tracking results.
Supports full divergence metrics (26 features) with fallback for legacy results.
"""

import argparse
import sys
import json
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from detection.detector import HallucinationDetector
from detection.divergence_metrics import DivergenceMetrics
from config import load_config


def extract_features_from_result(result: dict, n_layers: int = 12, feature_names: list = None) -> np.ndarray:
    """
    Extract feature vector from tracking result.
    
    Uses ALL available metrics from the result for maximum expressiveness.
    Falls back to basic stats extraction for legacy results.
    
    Args:
        result: Tracking result dict with 'stats' and optionally 'metrics'
        n_layers: Number of model layers
        feature_names: Optional list of feature names to use (for consistency)
        
    Returns:
        Feature vector as numpy array
    """
    # Prefer precomputed metrics (supports dynamic feature sets)
    if 'metrics' in result and result['metrics']:
        if feature_names is not None:
            # Use provided feature names for consistency
            return np.array([result['metrics'].get(name, 0.0) for name in feature_names], dtype=np.float32)
        else:
            # Use all available metrics dynamically
            return np.array(list(result['metrics'].values()), dtype=np.float32)
    
    # Fallback: extract from basic stats (legacy format)
    return _extract_features_from_stats(result.get('stats', {}), n_layers)


def _extract_features_from_stats(stats: dict, n_layers: int = 12) -> np.ndarray:
    """
    Legacy feature extraction from basic tracking statistics.
    
    Used for backward compatibility with old tracking results.
    Note: This produces fewer/less informative features than divergence metrics.
    
    Args:
        stats: Basic tracking statistics dict
        n_layers: Number of model layers
        
    Returns:
        Feature vector as numpy array
    """
    # Compute synthetic features that approximate divergence metrics
    total_births = stats.get('total_births', 0)
    total_deaths = stats.get('total_deaths', 0)
    total_tracks = stats.get('total_tracks', 0)
    alive_tracks = stats.get('alive_tracks', 0)
    survival_rate = stats.get('survival_rate', 0)
    max_concurrent = stats.get('max_concurrent_tracks', 50)
    
    # Build feature vector matching DivergenceMetrics structure
    # Fill in approximate values for required 26 features
    features = [
        # Entropy (4 features) - approximate from track counts
        np.log2(max(alive_tracks, 1)),  # entropy_mean (approx)
        np.log2(max(max_concurrent, 1)),  # entropy_max (approx)
        abs(alive_tracks - max_concurrent/2) / max(max_concurrent, 1),  # entropy_std (approx)
        np.log2(max(alive_tracks, 1)),  # entropy_final (approx)
        
        # Churn (6 features)
        total_births,
        total_deaths,
        total_births / max(n_layers, 1),  # birth_rate
        total_deaths / max(n_layers, 1),  # death_rate
        (total_births + total_deaths) / max(2 * n_layers, 1),  # churn_ratio
        survival_rate,
        
        # Competition (5 features) - approximate
        max_concurrent * 0.5,  # competition_mean (approx)
        max_concurrent,  # competition_max
        max_concurrent * 0.1,  # competition_std (approx)
        alive_tracks,  # competition_final
        0.5,  # high_competition_ratio (default)
        
        # Stability (3 features) - approximate
        survival_rate,  # stability_mean (approx)
        survival_rate * 0.5,  # stability_min (approx)
        1.0 - survival_rate,  # unstable_track_ratio (approx)
        
        # Dominance (4 features) - approximate
        1.0 / max(alive_tracks, 1),  # dominance_mean (approx)
        0.5 / max(alive_tracks, 1),  # dominance_min (approx)
        1.0 / max(alive_tracks, 1),  # dominance_final (approx)
        0.5,  # weak_dominance_ratio (default)
        
        # Density (4 features)
        (total_tracks / max(n_layers, 1)),  # density_mean
        max_concurrent,  # density_max
        max_concurrent * 0.2,  # density_std (approx)
        max_concurrent,  # max_concurrent_tracks
    ]
    
    return np.array(features, dtype=np.float32)


def main():
    parser = argparse.ArgumentParser(description='Run hallucination detection')
    parser.add_argument('--config', type=str, required=True, help='Config file path')
    parser.add_argument('--model-dir', type=str, required=True, help='SAE checkpoints directory')
    parser.add_argument('--tracking-dir', type=str, required=True, help='Tracking results directory')
    parser.add_argument('--output-dir', type=str, required=True, help='Output directory')
    parser.add_argument('--device', type=str, default='cuda', help='Device')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load config
    print("Loading configuration...")
    config = load_config(args.config)
    
    # Load tracking results
    tracking_file = Path(args.tracking_dir) / 'tracking_results.json'
    print(f"\nLoading tracking results from {tracking_file}...")
    
    with open(tracking_file, 'r') as f:
        tracking_results = json.load(f)
    
    print(f"Loaded {len(tracking_results)} tracking results")
    
    # Always use the authoritative feature names from DivergenceMetrics
    # This prevents mismatch between training (JSON keys) and inference (get_feature_vector)
    metrics_computer = DivergenceMetrics()
    feature_names = metrics_computer.get_feature_names()
    print(f"✓ Using authorized feature set ({len(feature_names)} features)")
    # print(f"  Features: {feature_names}")
    
    # Extract features and labels
    print("\nExtracting features...")
    features = []
    labels = []
    
    for result in tracking_results:
        feat_vector = extract_features_from_result(result, config.model.n_layers, feature_names)
        features.append(feat_vector)
        labels.append(0 if result['is_factual'] else 1)

    
    X = np.array(features)
    y = np.array(labels)
    
    print(f"Feature matrix shape: {X.shape}")
    print(f"Labels: {sum(y == 0)} factual, {sum(y == 1)} hallucinated")
    
    # Split into train/test
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\nTrain set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    
    # Train detector
    print("\nTraining detector...")
    detector = HallucinationDetector(
        model_type='random_forest',
        num_layers=config.model.n_layers
    )
    
    detector.fit_features(X_train, y_train)
    print("✓ Detector trained")
    
    # Evaluate
    print("\nEvaluating...")
    metrics = detector.evaluate_features(X_test, y_test)
    
    print("\n=== Detection Results ===")
    print(f"AUROC:     {metrics['auroc']:.4f}")
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1:        {metrics['f1']:.4f}")
    
    # Save detector
    detector_path = output_dir / 'detector.pkl'
    detector.save(str(detector_path))
    print(f"\n✓ Saved detector to {detector_path}")
    
    # Save metrics
    metrics_file = output_dir / 'detection_metrics.json'
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"✓ Saved metrics to {metrics_file}")
    
    # Generate predictions for all samples
    print("\nGenerating predictions for all samples...")
    predictions = detector.predict_features(X)
    probabilities = detector.predict_proba_features(X)
    
    results = []
    for i, result in enumerate(tracking_results):
        results.append({
            'sample_id': result['sample_id'],
            'question': result['question'],
            'is_factual': result['is_factual'],
            'predicted': int(predictions[i]),
            'probability': float(probabilities[i, 1]),
            'correct': bool((predictions[i] == 0) == result['is_factual'])
        })
    
    predictions_file = output_dir / 'predictions.json'
    with open(predictions_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"✓ Saved predictions to {predictions_file}")
    
    # Summary
    correct = sum(1 for r in results if r['correct'])
    print(f"\n=== Summary ===")
    print(f"Total samples: {len(results)}")
    print(f"Correct predictions: {correct} ({correct/len(results)*100:.1f}%)")
    
    print("\nDetection complete!")


if __name__ == '__main__':
    main()
