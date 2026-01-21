#!/usr/bin/env python3
"""
Phase 5: Evaluation & Metrics Runner

Generates final evaluation metrics and reports.
"""

import argparse
import sys
import json
import numpy as np
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import load_config


def main():
    parser = argparse.ArgumentParser(description='Run evaluation pipeline')
    parser.add_argument('--config', type=str, required=True, help='Config file path')
    parser.add_argument('--results-dir', type=str, required=True, help='Results directory')
    parser.add_argument('--output-dir', type=str, required=True, help='Output directory')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results_dir = Path(args.results_dir)
    
    # Load config
    print("Loading configuration...")
    config = load_config(args.config)
    
    # Load detection metrics
    detection_metrics_file = results_dir / 'detection' / 'detection_metrics.json'
    print(f"\nLoading detection metrics from {detection_metrics_file}...")
    
    with open(detection_metrics_file, 'r') as f:
        detection_metrics = json.load(f)
    
    # Load predictions
    predictions_file = results_dir / 'detection' / 'predictions.json'
    with open(predictions_file, 'r') as f:
        predictions = json.load(f)
    
    # Load tracking summary
    tracking_summary_file = results_dir / 'tracking' / 'tracking_summary.json'
    with open(tracking_summary_file, 'r') as f:
        tracking_summary = json.load(f)
    
    # Calculate additional metrics
    print("\nCalculating additional metrics...")
    
    factual_correct = sum(1 for p in predictions if p['is_factual'] and p['predicted'] == 0)
    factual_total = sum(1 for p in predictions if p['is_factual'])
    
    hallucinated_correct = sum(1 for p in predictions if not p['is_factual'] and p['predicted'] == 1)
    hallucinated_total = sum(1 for p in predictions if not p['is_factual'])
    
    # Generate final report
    report = {
        'model': config.model.base_model,
        'n_layers': config.model.n_layers,
        'd_model': config.model.d_model,
        'timestamp': datetime.now().isoformat(),
        
        'tracking': {
            'total_samples': tracking_summary['total_samples'],
            'factual_samples': tracking_summary['factual_samples'],
            'hallucinated_samples': tracking_summary['hallucinated_samples']
        },
        
        'detection': {
            'auroc': detection_metrics['auroc'],
            'accuracy': detection_metrics['accuracy'],
            'precision': detection_metrics['precision'],
            'recall': detection_metrics['recall'],
            'f1': detection_metrics['f1']
        },
        
        'class_accuracy': {
            'factual': factual_correct / factual_total if factual_total > 0 else 0,
            'hallucinated': hallucinated_correct / hallucinated_total if hallucinated_total > 0 else 0
        },
        
        'quality_assessment': {
            'meets_auroc_min': detection_metrics['auroc'] >= config.evaluation.target_auroc_min,
            'meets_auroc_strong': detection_metrics['auroc'] >= config.evaluation.target_auroc_strong,
            'meets_auroc_exceptional': detection_metrics['auroc'] >= config.evaluation.target_auroc_exceptional
        },
        
        'targets': {
            'auroc_min': config.evaluation.target_auroc_min,
            'auroc_strong': config.evaluation.target_auroc_strong,
            'auroc_exceptional': config.evaluation.target_auroc_exceptional
        }
    }
    
    # Print report
    print("\n" + "=" * 70)
    print("GHOSTTRACK EVALUATION REPORT")
    print("=" * 70)
    print(f"\nModel: {report['model']}")
    print(f"Layers: {report['n_layers']}")
    print(f"Hidden Dimension: {report['d_model']}")
    
    print("\n--- Tracking Summary ---")
    print(f"Total Samples: {report['tracking']['total_samples']}")
    print(f"Factual: {report['tracking']['factual_samples']}")
    print(f"Hallucinated: {report['tracking']['hallucinated_samples']}")
    
    print("\n--- Detection Performance ---")
    print(f"AUROC:     {report['detection']['auroc']:.4f}")
    print(f"Accuracy:  {report['detection']['accuracy']:.4f}")
    print(f"Precision: {report['detection']['precision']:.4f}")
    print(f"Recall:    {report['detection']['recall']:.4f}")
    print(f"F1 Score:  {report['detection']['f1']:.4f}")
    
    print("\n--- Class-wise Accuracy ---")
    print(f"Factual Accuracy:      {report['class_accuracy']['factual']:.4f}")
    print(f"Hallucinated Accuracy: {report['class_accuracy']['hallucinated']:.4f}")
    
    print("\n--- Quality Assessment ---")
    if report['quality_assessment']['meets_auroc_exceptional']:
        print("✓✓✓ EXCEPTIONAL: Exceeds all targets!")
    elif report['quality_assessment']['meets_auroc_strong']:
        print("✓✓ STRONG: Meets strong performance target")
    elif report['quality_assessment']['meets_auroc_min']:
        print("✓ GOOD: Meets minimum target")
    else:
        print("⚠ BELOW TARGET: Does not meet minimum AUROC target")
    
    print("\n--- Targets ---")
    print(f"Minimum AUROC:     {report['targets']['auroc_min']:.2f}")
    print(f"Strong AUROC:      {report['targets']['auroc_strong']:.2f}")
    print(f"Exceptional AUROC: {report['targets']['auroc_exceptional']:.2f}")
    
    print("=" * 70)
    
    # Save report
    report_file = output_dir / 'final_report.json'
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"\n✓ Saved final report to {report_file}")
    
    # Generate markdown report
    md_report = f"""# GhostTrack Evaluation Report

**Model**: {report['model']}  
**Layers**: {report['n_layers']}  
**Timestamp**: {report['timestamp']}

## Detection Performance

| Metric | Value | Target |
|--------|-------|--------|
| AUROC | {report['detection']['auroc']:.4f} | ≥ {report['targets']['auroc_min']:.2f} |
| Accuracy | {report['detection']['accuracy']:.4f} | - |
| Precision | {report['detection']['precision']:.4f} | - |
| Recall | {report['detection']['recall']:.4f} | - |
| F1 Score | {report['detection']['f1']:.4f} | - |

## Class-wise Performance

| Class | Accuracy |
|-------|----------|
| Factual | {report['class_accuracy']['factual']:.4f} |
| Hallucinated | {report['class_accuracy']['hallucinated']:.4f} |

## Samples Processed

- **Total**: {report['tracking']['total_samples']}
- **Factual**: {report['tracking']['factual_samples']}  
- **Hallucinated**: {report['tracking']['hallucinated_samples']}

## Quality Assessment

"""
    
    if report['quality_assessment']['meets_auroc_exceptional']:
        md_report += "✅ **EXCEPTIONAL** - Exceeds all targets!"
    elif report['quality_assessment']['meets_auroc_strong']:
        md_report += "✅ **STRONG** - Meets strong performance target"
    elif report['quality_assessment']['meets_auroc_min']:
        md_report += "✅ **GOOD** - Meets minimum target"
    else:
        md_report += "⚠️ **BELOW TARGET** - Does not meet minimum AUROC"
    
    md_file = output_dir / 'EVALUATION_REPORT.md'
    with open(md_file, 'w') as f:
        f.write(md_report)
    print(f"✓ Saved markdown report to {md_file}")
    
    print("\nEvaluation complete!")


if __name__ == '__main__':
    main()
