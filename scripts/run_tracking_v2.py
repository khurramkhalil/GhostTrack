#!/usr/bin/env python3
"""
Phase 3 Enhanced: Tracking + Reconstruction Error Features

Computes both divergence metrics AND reconstruction error patterns
for improved hallucination detection.
"""

import argparse
import sys
import json
import torch
import torch.nn.functional as F
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
            state_dict = checkpoint['model_state_dict']
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


def compute_reconstruction_errors(model, saes, text: str, device: str):
    """
    Compute per-layer reconstruction errors.
    
    Returns dict with:
    - per_layer_errors: list of MSE errors per layer
    - error_mean, error_std, error_max, error_growth
    - max_error_layer: layer with highest error
    """
    tokens = model.tokenizer(text, return_tensors='pt', truncation=True, max_length=512)
    input_ids = tokens['input_ids'].to(device)
    
    with torch.no_grad():
        outputs = model.model(input_ids, output_hidden_states=True)
        hidden_states = outputs.hidden_states[1:]  # Skip embedding layer
    
    errors = []
    for layer_idx, sae in saes.items():
        hidden = hidden_states[layer_idx][:, -1, :]  # Last token
        
        with torch.no_grad():
            # Encode and decode
            encoded = sae.encode(hidden)
            decoded = sae.decode(encoded)
            
            # Compute MSE
            mse = F.mse_loss(decoded, hidden).item()
            errors.append(mse)
    
    errors = np.array(errors)
    
    return {
        'error_mean': float(np.mean(errors)),
        'error_std': float(np.std(errors)),
        'error_max': float(np.max(errors)),
        'error_min': float(np.min(errors)),
        'error_final': float(errors[-1]) if len(errors) > 0 else 0.0,
        'error_growth': float(errors[-1] - errors[0]) if len(errors) > 1 else 0.0,
        'max_error_layer': int(np.argmax(errors)),
        'error_late_ratio': float(np.mean(errors[-6:])) / (float(np.mean(errors[:6])) + 1e-8) if len(errors) >= 12 else 1.0,
    }


def run_tracking_on_sample(model, saes, text: str, config, device: str):
    """Run hypothesis tracking on a single text sample."""
    tracker = HypothesisTracker(config)
    
    tokens = model.tokenizer(text, return_tensors='pt', truncation=True, max_length=512)
    input_ids = tokens['input_ids'].to(device)
    
    with torch.no_grad():
        outputs = model.model(input_ids, output_hidden_states=True)
        hidden_states = outputs.hidden_states[1:]
    
    for layer_idx, sae in saes.items():
        hidden = hidden_states[layer_idx][:, -1, :]
        
        with torch.no_grad():
            encoded = sae.encode(hidden)
            activations = encoded[0].cpu().numpy()
            
            top_indices = np.argsort(activations)[::-1]
            
            features = []
            top_k = config.get('top_k_features', 50)
            count = 0
            
            for feat_id in top_indices:
                activation = float(activations[feat_id])
                if activation <= 0:
                    break
                    
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
    parser = argparse.ArgumentParser(description='Run enhanced tracking with reconstruction error')
    parser.add_argument('--config', type=str, required=True, help='Config file path')
    parser.add_argument('--model-dir', type=str, required=True, help='SAE checkpoints directory')
    parser.add_argument('--output-dir', type=str, required=True, help='Output directory')
    parser.add_argument('--device', type=str, default='cuda', help='Device')
    parser.add_argument('--num-samples', type=int, default=500, help='Number of samples to process')
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Loading configuration...")
    config = load_config(args.config)
    
    print(f"\nLoading {config.model.base_model}...")
    model = get_model_wrapper(
        model_name=config.model.base_model,
        device=args.device
    )
    
    print(f"\nLoading SAEs from {args.model_dir}...")
    model_dir = Path(args.model_dir)
    saes = load_saes(model_dir, config.model.n_layers, args.device)
    print(f"Loaded {len(saes)} SAEs")
    
    print("\nLoading TruthfulQA dataset...")
    dataset = load_dataset("truthful_qa", "generation", split="validation")
    
    metrics_computer = DivergenceMetrics()
    n_layers = config.model.n_layers
    
    print(f"\nProcessing {args.num_samples} factual samples...")
    results = []
    
    for i, sample in enumerate(dataset):
        if i >= args.num_samples:
            break
            
        question = sample['question']
        best_answer = sample['best_answer']
        text = f"Question: {question}\nAnswer: {best_answer}"
        
        try:
            # Tracking metrics
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
            
            # Reconstruction error metrics
            recon_metrics = compute_reconstruction_errors(model, saes, text, args.device)
            
            # Combine all metrics
            all_metrics = {**divergence_metrics, **recon_metrics}
            
            results.append({
                'sample_id': i,
                'question': question,
                'is_factual': True,
                'stats': stats,
                'metrics': all_metrics
            })
            
            if (i + 1) % 50 == 0:
                print(f"  Processed {i + 1}/{args.num_samples} samples")
                
        except Exception as e:
            print(f"  Error processing sample {i}: {e}")
            continue
    
    # Process incorrect answers (hallucinated)
    print("\nProcessing incorrect answers...")
    for i, sample in enumerate(dataset):
        if i >= args.num_samples // 2:
            break
            
        question = sample['question']
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
                recon_metrics = compute_reconstruction_errors(model, saes, text, args.device)
                all_metrics = {**divergence_metrics, **recon_metrics}
                
                results.append({
                    'sample_id': args.num_samples + i,
                    'question': question,
                    'is_factual': False,
                    'stats': stats,
                    'metrics': all_metrics
                })
            except Exception as e:
                continue
    
    # Save results
    output_file = output_dir / 'tracking_results.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, cls=NumpyEncoder)
    
    print(f"\n✓ Saved {len(results)} tracking results to {output_file}")
    
    # Show feature count
    if results:
        n_features = len(results[0]['metrics'])
        print(f"  Total features per sample: {n_features}")
    
    # Generate summary
    factual_count = sum(1 for r in results if r['is_factual'])
    hallucinated_count = len(results) - factual_count
    
    summary = {
        'total_samples': len(results),
        'factual_samples': factual_count,
        'hallucinated_samples': hallucinated_count,
        'config': args.config,
        'model_dir': args.model_dir,
        'includes_recon_error': True
    }
    
    summary_file = output_dir / 'tracking_summary.json'
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2, cls=NumpyEncoder)
    
    print(f"✓ Saved summary to {summary_file}")
    print("\nEnhanced tracking complete!")


if __name__ == '__main__':
    main()
