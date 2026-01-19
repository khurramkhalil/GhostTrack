
import os
import argparse
import json
import torch
import numpy as np
from pathlib import Path

from models import GPT2WithResidualHooks
from tracking import LayerwiseFeatureExtractor
from data import load_truthfulqa
from evaluation.pipeline import process_dataset, train_and_evaluate

def main():
    parser = argparse.ArgumentParser(description='Run GhostTrack Detection Pipeline (Phase 3 & 4)')
    parser.add_argument('--checkpoint-dir', type=str, default='./models/checkpoints', help='Directory with trained SAEs')
    parser.add_argument('--output-dir', type=str, default='./results/detection', help='Output directory')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    parser.add_argument('--train-ratio', type=float, default=0.7, help='Training set ratio')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    # Tracking config
    parser.add_argument('--birth-threshold', type=float, default=0.5)
    parser.add_argument('--association-threshold', type=float, default=0.5)
    parser.add_argument('--semantic-weight', type=float, default=0.6)
    parser.add_argument('--top-k', type=int, default=50)
    
    args = parser.parse_args()
    
    # Setup
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("================================================================")
    print("GhostTrack Detection Pipeline (Phase 3 & 4)")
    print(f"Device: {args.device}")
    print(f"Checkpoints: {args.checkpoint_dir}")
    print("================================================================")

    # 1. Load Data
    print("\n[1/5] Loading TruthfulQA Dataset...")
    try:
        train_data, val_data, test_data = load_truthfulqa(
            split_ratios=(args.train_ratio, 0.15, 0.15),
            seed=args.seed
        )
        print(f"Data split: Train={len(train_data)}, Val={len(val_data)}, Test={len(test_data)}")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        # Try finding cache
        print("Attempting to load standard dataset...")
        from datasets import load_dataset
        ds = load_dataset('truthful_qa', 'generation', trust_remote_code=True)
        print("Note: Custom loader failed, please check cluster internet access or HuggingFace cache.")
        raise e

    # 2. Load Model & SAEs
    print("\n[2/5] Loading Model and SAEs...")
    model = GPT2WithResidualHooks('gpt2', device=args.device)
    
    extractor = LayerwiseFeatureExtractor.load_from_checkpoints(
        model_wrapper=model,
        checkpoint_dir=args.checkpoint_dir,
        device=args.device
    )
    print("SAEs loaded for 12 layers")

    # Tracking Config
    tracking_config = {
        'birth_threshold': args.birth_threshold,
        'association_threshold': args.association_threshold,
        'semantic_weight': args.semantic_weight,
        'activation_weight': 0.2,
        'position_weight': 0.2,
        'top_k_features': args.top_k
    }

    # 3. Process Data
    print("\n[3/5] Running Hypothesis Tracking (Phase 3)...")
    train_trackers, train_labels = process_dataset(
        train_data, extractor, tracking_config, args.top_k, "Train"
    )
    test_trackers, test_labels = process_dataset(
        test_data, extractor, tracking_config, args.top_k, "Test"
    )

    # 4 & 5. Train & Evaluate
    print("\n[4 & 5] Training & Evaluating Detector...")
    detector, metrics, importance = train_and_evaluate(
        train_trackers, train_labels, test_trackers, test_labels
    )
    
    print("\n=== FINAL RESULTS ===")
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"AUROC:     {metrics['auroc']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1 Score:  {metrics['f1']:.4f}")
    
    # Feature Importance
    print("\nTop Predictors of Hallucination:")
    for feat, score in list(importance.items())[:10]:
        print(f"  {feat}: {score:.4f}")

    # Save Results
    results = {
        'config': vars(args),
        'metrics': metrics,
        'feature_importance': importance
    }
    
    with open(output_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)
        
    detector.save(output_dir / 'detector.pkl')
    print(f"\nResults saved to {output_dir}")

if __name__ == '__main__':
    main()
