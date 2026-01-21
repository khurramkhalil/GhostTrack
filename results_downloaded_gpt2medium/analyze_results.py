#!/usr/bin/env python3
"""
Analyze GPT-2 Medium SAE Training Results
"""

import json
import glob
from pathlib import Path
import numpy as np

# Load all training histories
history_dir = Path("results_downloaded_gpt2medium/training_history")
history_files = sorted(glob.glob(str(history_dir / "sae_layer_*_history.json")))

print(f"Found {len(history_files)} training history files\n")

# Collect metrics
all_metrics = []

for hist_file in history_files:
    layer_num = int(Path(hist_file).stem.split("_")[2])
    
    with open(hist_file, 'r') as f:
        data = json.load(f)
    
    metrics = {
        'layer': layer_num,
        'final_train_loss': data['train_loss'][-1],
        'final_val_loss': data['val_loss'][-1],
        'final_recon_loss': data['recon_loss'][-1],
        'final_sparsity': data['sparsity'][-1],
        'best_val_loss': min(data['val_loss']),
        'epochs': len(data['train_loss'])
    }
    
    all_metrics.append(metrics)

# Sort by layer
all_metrics.sort(key=lambda x: x['layer'])

# Summary statistics
recon_losses = [m['final_recon_loss'] for m in all_metrics]
sparsities = [m['final_sparsity'] for m in all_metrics]
best_losses = [m['best_val_loss'] for m in all_metrics]

print("=" * 80)
print("GPT-2 MEDIUM - SAE TRAINING SUMMARY")
print("=" * 80)
print(f"\nTotal Layers Trained: {len(all_metrics)}")
print(f"Epochs per Layer: {all_metrics[0]['epochs']}")
print("\n" + "=" * 80)
print("RECONSTRUCTION LOSS")
print("=" * 80)
print(f"Mean:   {np.mean(recon_losses):.6f}")
print(f"Median: {np.median(recon_losses):.6f}")
print(f"Min:    {np.min(recon_losses):.6f} (Layer {all_metrics[np.argmin(recon_losses)]['layer']})")
print(f"Max:    {np.max(recon_losses):.6f} (Layer {all_metrics[np.argmax(recon_losses)]['layer']})")
print(f"Std:    {np.std(recon_losses):.6f}")

print("\n" + "=" * 80)
print("SPARSITY (Active Features)")
print("=" * 80)
print(f"Mean:   {np.mean(sparsities):.2f}")
print(f"Median: {np.median(sparsities):.2f}")
print(f"Min:    {np.min(sparsities):.2f} (Layer {all_metrics[np.argmin(sparsities)]['layer']})")
print(f"Max:    {np.max(sparsities):.2f} (Layer {all_metrics[np.argmax(sparsities)]['layer']})")
print(f"Std:    {np.std(sparsities):.2f}")

print("\n" + "=" * 80)
print("BEST VALIDATION LOSS")
print("=" * 80)
print(f"Mean:   {np.mean(best_losses):.6f}")
print(f"Median: {np.median(best_losses):.6f}")
print(f"Min:    {np.min(best_losses):.6f} (Layer {all_metrics[np.argmin(best_losses)]['layer']})")
print(f"Max:    {np.max(best_losses):.6f} (Layer {all_metrics[np.argmax(best_losses)]['layer']})")

print("\n" + "=" * 80)
print("PER-LAYER RESULTS")
print("=" * 80)
print(f"{'Layer':<8} {'Recon Loss':<14} {'Sparsity':<12} {'Best Val':<14} {'Status'}")
print("-" * 80)

for m in all_metrics:
    status = "✓ Good" if m['final_recon_loss'] < 0.01 else "⚠ High"
    print(f"{m['layer']:<8} {m['final_recon_loss']:<14.6f} "
          f"{m['final_sparsity']:<12.2f} {m['best_val_loss']:<14.6f} {status}")

print("\n" + "=" * 80)
print("QUALITY ASSESSMENT")
print("=" * 80)

excellent = sum(1 for m in all_metrics if m['final_recon_loss'] < 0.002)
good = sum(1 for m in all_metrics if 0.002 <= m['final_recon_loss'] < 0.005)
fair = sum(1 for m in all_metrics if 0.005 <= m['final_recon_loss'] < 0.01)
poor = sum(1 for m in all_metrics if m['final_recon_loss'] >= 0.01)

print(f"Excellent (< 0.002):  {excellent} layers ({excellent/len(all_metrics)*100:.1f}%)")
print(f"Good (0.002-0.005):   {good} layers ({good/len(all_metrics)*100:.1f}%)")
print(f"Fair (0.005-0.01):    {fair} layers ({fair/len(all_metrics)*100:.1f}%)")
print(f"Poor (>= 0.01):       {poor} layers ({poor/len(all_metrics)*100:.1f}%)")

print("\n" + "=" * 80)
print("SPARSITY ASSESSMENT")
print("=" * 80)

too_sparse = sum(1 for m in all_metrics if m['final_sparsity'] < 50)
good_sparse = sum(1 for m in all_metrics if 50 <= m['final_sparsity'] <= 100)
too_dense = sum(1 for m in all_metrics if m['final_sparsity'] > 100)

print(f"Too Sparse (< 50):    {too_sparse} layers ({too_sparse/len(all_metrics)*100:.1f}%)")
print(f"Optimal (50-100):     {good_sparse} layers ({good_sparse/len(all_metrics)*100:.1f}%)")
print(f"Too Dense (> 100):    {too_dense} layers ({too_dense/len(all_metrics)*100:.1f}%)")

print("\n" + "=" * 80)
print("CONCLUSION")
print("=" * 80)

if np.mean(recon_losses) < 0.005 and 50 <= np.mean(sparsities) <= 100:
    print("✓ EXCELLENT: Training was highly successful!")
    print("  All SAEs show good reconstruction with appropriate sparsity.")
elif np.mean(recon_losses) < 0.01:
    print("✓ GOOD: Training was successful.")
    print("  SAEs show acceptable reconstruction quality.")
else:
    print("⚠ NEEDS REVIEW: Some layers may need retraining.")
    print("  Check layers with high reconstruction loss.")

print("\n" + "=" * 80)
