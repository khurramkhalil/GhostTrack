#!/usr/bin/env python3
"""
GhostTrack Demo Script

Demonstrates the complete pipeline:
1. Load model and SAEs
2. Process sample text
3. Track hypotheses
4. Compute metrics
5. Detect hallucination
"""

import torch
import sys
sys.path.insert(0, '.')

from models import GPT2WithResidualHooks
from tracking import LayerwiseFeatureExtractor, HypothesisTracker
from detection import HallucinationDetector
from detection.divergence_metrics import DivergenceMetrics

def main():
    print("=" * 60)
    print("GhostTrack Demo: Hallucination Detection")
    print("=" * 60)
    
    # Configuration
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    checkpoint_dir = './models/checkpoints'
    detector_path = './results/detection/detector.pkl'
    
    # Sample texts
    factual_text = "Q: What is the capital of France? A: The capital of France is Paris."
    hallucinated_text = "Q: What is the capital of France? A: The capital of France is London."
    
    print(f"\nDevice: {device}")
    print(f"Checkpoints: {checkpoint_dir}")
    
    # 1. Load Model and SAEs
    print("\n[1/5] Loading Model and SAEs...")
    model = GPT2WithResidualHooks('gpt2', device=device)
    extractor = LayerwiseFeatureExtractor.load_from_checkpoints(
        model_wrapper=model,
        checkpoint_dir=checkpoint_dir,
        device=device
    )
    
    # 2. Load Detector
    print("[2/5] Loading Detector...")
    detector = HallucinationDetector.load(detector_path)
    metrics_computer = DivergenceMetrics()
    
    # Tracking config
    config = {
        'birth_threshold': 0.5,
        'association_threshold': 0.5,
        'semantic_weight': 0.6,
        'activation_weight': 0.2,
        'position_weight': 0.2,
        'top_k_features': 50
    }
    
    # 3. Process Factual Text
    print("\n[3/5] Processing Factual Text...")
    print(f"  Input: {factual_text}")
    
    layer_features = extractor.extract_features(factual_text)
    tracker_factual = HypothesisTracker(config=config)
    top_l0 = extractor.get_top_k_features(layer_features[0], k=50)
    tracker_factual.initialize_tracks(top_l0, token_pos=0)
    for layer_idx in range(1, 12):
        top_feats = extractor.get_top_k_features(layer_features[layer_idx], k=50)
        tracker_factual.update_tracks(layer_idx, top_feats)
    
    metrics_factual = metrics_computer.compute_all_metrics(tracker_factual)
    print(f"  Entropy: {metrics_factual['entropy_mean']:.3f}")
    print(f"  Stability: {metrics_factual['stability_mean']:.3f}")
    print(f"  Dominance: {metrics_factual['dominance_mean']:.3f}")
    
    # 4. Process Hallucinated Text
    print("\n[4/5] Processing Hallucinated Text...")
    print(f"  Input: {hallucinated_text}")
    
    layer_features = extractor.extract_features(hallucinated_text)
    tracker_halluc = HypothesisTracker(config=config)
    top_l0 = extractor.get_top_k_features(layer_features[0], k=50)
    tracker_halluc.initialize_tracks(top_l0, token_pos=0)
    for layer_idx in range(1, 12):
        top_feats = extractor.get_top_k_features(layer_features[layer_idx], k=50)
        tracker_halluc.update_tracks(layer_idx, top_feats)
    
    metrics_halluc = metrics_computer.compute_all_metrics(tracker_halluc)
    print(f"  Entropy: {metrics_halluc['entropy_mean']:.3f}")
    print(f"  Stability: {metrics_halluc['stability_mean']:.3f}")
    print(f"  Dominance: {metrics_halluc['dominance_mean']:.3f}")
    
    # 5. Predict
    print("\n[5/5] Predictions...")
    proba = detector.predict_proba([tracker_factual, tracker_halluc])
    
    print(f"\n{'='*60}")
    print("RESULTS")
    print(f"{'='*60}")
    print(f"Factual Answer:      {proba[0, 1]*100:5.1f}% hallucination probability")
    print(f"Hallucinated Answer: {proba[1, 1]*100:5.1f}% hallucination probability")
    print(f"{'='*60}")
    
    # Compare metrics
    print("\nMetric Comparison:")
    print(f"{'Metric':<20} {'Factual':>10} {'Halluc':>10} {'Diff':>10}")
    print("-" * 50)
    for key in ['entropy_mean', 'stability_mean', 'dominance_mean']:
        f_val = metrics_factual[key]
        h_val = metrics_halluc[key]
        diff = h_val - f_val
        print(f"{key:<20} {f_val:>10.3f} {h_val:>10.3f} {diff:>+10.3f}")

if __name__ == '__main__':
    main()
