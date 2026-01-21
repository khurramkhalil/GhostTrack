#!/usr/bin/env python3
"""
Phase 4: Hallucination Detection Runner

Trains and evaluates hallucination detector on tracking results.
"""

import argparse
import sys
import json
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from detection.detector import HallucinationDetector
from config import load_config


def extract_features_from_stats(stats: dict, n_layers: int = 24) -> np.ndarray:
    """Extract feature vector from tracking statistics."""
    features = [
        stats.get('total_tracks', 0) / 100.0,
        stats.get('alive_tracks', 0) / 50.0,
        stats.get('dead_tracks', 0) / 50.0,
        stats.get('birth_rate', 0),
        stats.get('death_rate', 0),
        stats.get('avg_lifespan', 0) / n_layers,
        stats.get('max_lifespan', 0) / n_layers,
        stats.get('avg_activation', 0),
        stats.get('max_activation', 0),
    ]
    # Add layer-specific features if available
    layer_activations = stats.get('layer_activations', [0] * n_layers)
    features.extend([a / 10.0 for a in layer_activations[:n_layers]])
    
    return np.array(features)


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
    
    # Extract features and labels
    print("\nExtracting features...")
    features = []
    labels = []
    
    for result in tracking_results:
        stats = result.get('stats', {})
        feat_vector = extract_features_from_stats(stats, config.model.n_layers)
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
