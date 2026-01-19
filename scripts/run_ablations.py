
import argparse
import json
import torch
import numpy as np
from pathlib import Path
from copy import deepcopy

from models import GPT2WithResidualHooks
from tracking import LayerwiseFeatureExtractor
from data import load_truthfulqa
from evaluation.pipeline import process_dataset, train_and_evaluate

ABLATIONS = {
    'baseline': {
        'description': 'Full GhostTrack Pipeline',
        'config_update': {}
    },
    'single_hypothesis': {
        'description': 'Single Hypothesis (Top-1)',
        'config_update': {'top_k_features': 1},
        'top_k_override': 1
    },
    'no_association': {
        'description': 'No Track Association (Independent Layers)',
        'config_update': {'disable_association': True}
    },
    'feature_id_association': {
        'description': 'Feature ID Association (Random Baseline)',
        'config_update': {'use_feature_id_matching': True}
    }
}

def main():
    parser = argparse.ArgumentParser(description='Run GhostTrack Ablation Studies (Phase 6)')
    parser.add_argument('--checkpoint-dir', type=str, default='./models/checkpoints', help='Directory with trained SAEs')
    parser.add_argument('--output-dir', type=str, default='./results/ablations', help='Output directory')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    # Setup
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("================================================================")
    print("GhostTrack Phase 6: Ablation Studies")
    print(f"Device: {args.device}")
    print("================================================================")

    # 1. Load Data
    print("\n[1/4] Loading Dataset...")
    train_data, val_data, test_data = load_truthfulqa(
        split_ratios=(0.7, 0.15, 0.15),
        seed=args.seed
    )
    print(f"Data split: Train={len(train_data)}, Test={len(test_data)}")

    # 2. Load Model & SAEs
    print("\n[2/4] Loading Model and SAEs...")
    model = GPT2WithResidualHooks('gpt2', device=args.device)
    extractor = LayerwiseFeatureExtractor.load_from_checkpoints(
        model_wrapper=model,
        checkpoint_dir=args.checkpoint_dir,
        device=args.device
    )

    # Default Config
    base_tracking_config = {
        'birth_threshold': 0.5,
        'association_threshold': 0.5,
        'semantic_weight': 0.6,
        'activation_weight': 0.2,
        'position_weight': 0.2,
        'top_k_features': 50
    }

    # 3. Run Ablations
    # 3. Run Ablations
    print("\n[3/4] Running Ablation Studies...")
    
    results_file = output_dir / 'ablation_results.json'
    if results_file.exists():
        print(f"Loading existing results from {results_file}")
        with open(results_file, 'r') as f:
            all_results = json.load(f)
    else:
        all_results = {}

    for name, info in ABLATIONS.items():
        if name in all_results:
            print(f"\n>>> Skipping Ablation: {name} (Already completed)")
            continue

        print(f"\n>>> Running Ablation: {name} ({info['description']})")
        
        # Prepare Config
        current_config = deepcopy(base_tracking_config)
        current_config.update(info['config_update'])
        
        top_k = info.get('top_k_override', base_tracking_config['top_k_features'])
        
        # Run Pipeline
        print(f"  Generating tracks (config: {info['config_update']})...")
        train_trackers, train_labels = process_dataset(
            train_data, extractor, current_config, top_k, "Train"
        )
        test_trackers, test_labels = process_dataset(
            test_data, extractor, current_config, top_k, "Test"
        )
        
        # Train & Evaluate
        detector, metrics, importance = train_and_evaluate(
            train_trackers, train_labels, test_trackers, test_labels
        )
        
        print(f"  RESULT: AUROC = {metrics['auroc']:.4f}")
        
        all_results[name] = {
            'description': info['description'],
            'metrics': metrics,
            'top_features': {k: importance[k] for k in list(importance.keys())[:5]}
        }
        
        # Incremental Save
        with open(results_file, 'w') as f:
            json.dump(all_results, f, indent=2)

    # 4. Comparative Analysis
    print("\n[4/4] Comparative Analysis (AUROC)")
    baseline_auroc = all_results['baseline']['metrics']['auroc']
    print(f"Baseline: {baseline_auroc:.4f}")
    
    for name, data in all_results.items():
        if name == 'baseline': continue
        auroc = data['metrics']['auroc']
        delta = auroc - baseline_auroc
        print(f"{name:<25}: {auroc:.4f} (Delta: {delta:+.4f})")

    # Save
    with open(output_dir / 'ablation_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)
        
    print(f"\nDetailed results saved to {output_dir}/ablation_results.json")

if __name__ == '__main__':
    main()
