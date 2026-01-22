#!/usr/bin/env python3
"""
Phase 3: Hypothesis Tracking Runner

Runs hypothesis tracking on trained SAEs and evaluation dataset.
Supports multiple model architectures (GPT-2, Phi, Qwen, Llama, etc.)
"""

import argparse
import sys
import json
import torch
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from models import get_model_wrapper, JumpReLUSAE
from tracking.hypothesis_tracker import HypothesisTracker
from detection.divergence_metrics import DivergenceMetrics
from config import load_config
from datasets import load_dataset


class NumpyEncoder(json.JSONEncoder):
    """Custom encoder for numpy data types"""
    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
            np.int16, np.int32, np.int64, np.uint8,
            np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32, 
            np.float64)):
            return float(obj)
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def load_saes(model_dir: Path, n_layers: int, device: str):
    """Load all trained SAEs."""
    saes = {}
    for layer_idx in range(n_layers):
        sae_path = model_dir / f"sae_layer_{layer_idx}_best.pt"
        if sae_path.exists():
            checkpoint = torch.load(sae_path, map_location=device, weights_only=False)
            config = checkpoint.get('config', {})
            
            # Get model dimensions from checkpoint
            state_dict = checkpoint['model_state_dict']
            # Keys are: 'threshold', 'W_enc.weight', 'W_enc.bias', 'W_dec.weight', 'W_dec.bias'
            d_hidden = state_dict['W_enc.weight'].shape[0]
            d_model = state_dict['W_enc.weight'].shape[1]
            
            sae = JumpReLUSAE(
                d_model=d_model,
                d_hidden=d_hidden,
                threshold=float(state_dict.get('threshold', 0.1)),
                lambda_sparse=config.get('lambda_sparse', 0.01)
            )
            sae.load_state_dict(state_dict)
            sae.to(device)
            sae.eval()
            saes[layer_idx] = sae
            print(f"  ✓ Loaded SAE for layer {layer_idx}")
        else:
            print(f"  ⚠ SAE not found for layer {layer_idx}")
    return saes


def run_tracking_on_sample(model, saes, text: str, config, device: str):
    """Run hypothesis tracking on a single text sample."""
    tracker = HypothesisTracker(config)
    
    # Tokenize
    tokens = model.tokenizer(text, return_tensors='pt', truncation=True, max_length=512)
    input_ids = tokens['input_ids'].to(device)
    
    # Get hidden states from model
    with torch.no_grad():
        outputs = model.model(input_ids, output_hidden_states=True)
        hidden_states = outputs.hidden_states[1:]  # Skip embedding layer
    
    # Process each layer
    for layer_idx, sae in saes.items():
        hidden = hidden_states[layer_idx][:, -1, :]  # Last token position
        
        # Get SAE features
        with torch.no_grad():
            encoded = sae.encode(hidden)
            activations = encoded[0].cpu().numpy()
            
            # Get indices sorted by activation magnitude (descending)
            # argsort gives ascending, so we reverse
            top_indices = np.argsort(activations)[::-1]
            
            features = []
            top_k = config.get('top_k_features', 50)
            count = 0
            
            for feat_id in top_indices:
                activation = float(activations[feat_id])
                if activation <= 0:
                    break  # Sorted, so we can stop once we hit zero
                    
                # Use actual decoder weight as semantic embedding
                # W_dec.weight is [d_model, d_hidden]
                embedding = sae.W_dec.weight[:, feat_id].detach().cpu().numpy()
                features.append((feat_id, activation, embedding))
                
                count += 1
                if count >= top_k:
                    break
        
        if layer_idx == 0:
            tracker.initialize_tracks(features, token_pos=0)
        else:
            tracker.update_tracks(layer_idx, features)
    
    return tracker


def main():
    parser = argparse.ArgumentParser(description='Run hypothesis tracking')
    parser.add_argument('--config', type=str, required=True, help='Config file path')
    parser.add_argument('--model-dir', type=str, required=True, help='SAE checkpoints directory')
    parser.add_argument('--output-dir', type=str, required=True, help='Output directory')
    parser.add_argument('--device', type=str, default='cuda', help='Device')
    parser.add_argument('--num-samples', type=int, default=500, help='Number of samples to process')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load config  
    print("Loading configuration...")
    config = load_config(args.config)
    
    # Load model (using factory for multi-model support)
    print(f"\nLoading {config.model.base_model}...")
    model = get_model_wrapper(
        model_name=config.model.base_model,
        device=args.device
    )
    
    # Load SAEs
    print(f"\nLoading SAEs from {args.model_dir}...")
    model_dir = Path(args.model_dir)
    saes = load_saes(model_dir, config.model.n_layers, args.device)
    print(f"Loaded {len(saes)} SAEs")
    
    # Load evaluation dataset (TruthfulQA)
    print("\nLoading TruthfulQA dataset...")
    dataset = load_dataset("truthful_qa", "generation", split="validation")
    
    # Initialize metrics computer for full feature extraction
    metrics_computer = DivergenceMetrics()
    n_layers = config.model.n_layers
    
    # Process samples
    print(f"\nProcessing {args.num_samples} factual samples...")
    results = []
    
    for i, sample in enumerate(dataset):
        if i >= args.num_samples:
            break
            
        question = sample['question']
        best_answer = sample['best_answer']
        
        # Run tracking on question + answer
        text = f"Question: {question}\nAnswer: {best_answer}"
        
        try:
            tracker = run_tracking_on_sample(model, saes, text, {
                'top_k_features': config.tracking.top_k_features,
                'semantic_weight': config.tracking.semantic_weight,
                'activation_weight': config.tracking.activation_weight,
                'position_weight': config.tracking.position_weight,
                'association_threshold': config.tracking.association_threshold,
                'birth_threshold': config.tracking.birth_threshold,
                'death_threshold': config.tracking.death_threshold,
                'use_feature_id_matching': getattr(config.tracking, 'use_feature_id_matching', False),
            }, args.device)
            
            stats = tracker.get_statistics()
            divergence_metrics = metrics_computer.compute_all_metrics(tracker, n_layers)
            
            results.append({
                'sample_id': i,
                'question': question,
                'is_factual': True,  # Best answers are factual
                'stats': stats,
                'metrics': divergence_metrics  # Full 26 features for detection
            })
            
            if (i + 1) % 50 == 0:
                print(f"  Processed {i + 1}/{args.num_samples} samples")
                
        except Exception as e:
            print(f"  Error processing sample {i}: {e}")
            continue
    
    # Process incorrect answers (hallucinated)
    print("\nProcessing incorrect answers...")
    for i, sample in enumerate(dataset):
        if i >= args.num_samples // 2:  # Half as many
            break
            
        question = sample['question']
        # Use incorrect answers
        if 'incorrect_answers' in sample and sample['incorrect_answers']:
            incorrect = sample['incorrect_answers'][0]
            text = f"Question: {question}\nAnswer: {incorrect}"
            
            try:
                tracker = run_tracking_on_sample(model, saes, text, {
                    'top_k_features': config.tracking.top_k_features,
                    'semantic_weight': config.tracking.semantic_weight,
                    'activation_weight': config.tracking.activation_weight,
                    'position_weight': config.tracking.position_weight,
                    'association_threshold': config.tracking.association_threshold,
                    'birth_threshold': config.tracking.birth_threshold,
                    'death_threshold': config.tracking.death_threshold,
                    'use_feature_id_matching': getattr(config.tracking, 'use_feature_id_matching', False),
                }, args.device)
                
                stats = tracker.get_statistics()
                divergence_metrics = metrics_computer.compute_all_metrics(tracker, n_layers)
                
                results.append({
                    'sample_id': args.num_samples + i,
                    'question': question,
                    'is_factual': False,  # Incorrect answers are hallucinated
                    'stats': stats,
                    'metrics': divergence_metrics  # Full 26 features for detection
                })
            except Exception as e:
                continue
    
    # Save results
    output_file = output_dir / 'tracking_results.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, cls=NumpyEncoder)
    
    print(f"\n✓ Saved {len(results)} tracking results to {output_file}")
    
    # Generate summary
    factual_count = sum(1 for r in results if r['is_factual'])
    hallucinated_count = len(results) - factual_count
    
    summary = {
        'total_samples': len(results),
        'factual_samples': factual_count,
        'hallucinated_samples': hallucinated_count,
        'config': args.config,
        'model_dir': args.model_dir
    }
    
    summary_file = output_dir / 'tracking_summary.json'
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2, cls=NumpyEncoder)
    
    print(f"✓ Saved summary to {summary_file}")
    print("\nTracking complete!")


if __name__ == '__main__':
    main()
