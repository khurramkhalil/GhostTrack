
import os
import argparse
import random
import torch
import numpy as np
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from models import get_model_wrapper, GPT2WithResidualHooks
from tracking import LayerwiseFeatureExtractor, HypothesisTracker
from detection import HallucinationDetector
from visualization import create_interactive_dashboard, CaseStudyGenerator
from data import load_truthfulqa

def main():
    parser = argparse.ArgumentParser(description='Run GhostTrack Visualization Pipeline (Phase 5)')
    parser.add_argument('--checkpoint-dir', type=str, default='./models/checkpoints', help='Directory with trained SAEs')
    parser.add_argument('--detector-path', type=str, default='./results/detection/detector.pkl', help='Path to trained detector')
    parser.add_argument('--output-dir', type=str, default='./results/visualization', help='Output directory')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    parser.add_argument('--model-name', type=str, default='gpt2', help='Model architecture (gpt2, gpt2-medium)')
    parser.add_argument('--num-examples', type=int, default=10, help='Number of examples to visualize')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    # Setup
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("================================================================")
    print("GhostTrack Visualization Pipeline (Phase 5)")
    print(f"Model: {args.model_name}")
    print(f"Device: {args.device}")
    print(f"Output: {args.output_dir}")
    print("================================================================")

    # 1. Load Data
    print("\n[1/4] Loading Data...")
    _, _, test_data = load_truthfulqa()
    
    # Select random examples
    indices = random.sample(range(len(test_data)), min(args.num_examples, len(test_data)))
    examples = [test_data[i] for i in indices]
    print(f"Selected {len(examples)} examples for visualization.")

    # 2. Load Model & SAEs
    print("\n[2/4] Loading Model and SAEs...")
    model = get_model_wrapper(args.model_name, device=args.device)
    
    extractor = LayerwiseFeatureExtractor.load_from_checkpoints(
        model_wrapper=model,
        checkpoint_dir=args.checkpoint_dir,
        device=args.device
    )

    # 3. Load Detector
    print("\n[3/4] Loading Trained Detector...")
    if not os.path.exists(args.detector_path):
        print(f"Error: Detector not found at {args.detector_path}")
        print("Please run detection pipeline first (Phase 4).")
        return

    detector = HallucinationDetector.load(args.detector_path)
    
    # Tracking Config (match training)
    config = {
        'birth_threshold': 0.5,
        'association_threshold': 0.5,
        'semantic_weight': 0.6,
        'top_k_features': 50
    }

    # 4. Generate Visualizations
    print("\n[4/4] Generating Visualizations...")
    
    # Case Study Generator
    cs_generator = CaseStudyGenerator(detector, num_layers=model.n_layers)
    
    trackers_factual = []
    trackers_hallucinated = []
    
    for i, example in enumerate(examples):
        print(f"Processing Example {i+1}/{len(examples)}...")
        
        # Factual
        text_fact = f"{example.prompt} {example.factual_answer}"
        feats_fact = extractor.extract_features(text_fact)
        tracker_fact = HypothesisTracker(config=config)
        tracker_fact.initialize_tracks(extractor.get_top_k_features(feats_fact[0]), 0)
        for l in range(1, 12):
            tracker_fact.update_tracks(l, extractor.get_top_k_features(feats_fact[l]))
        
        # Hallucinated
        text_halluc = f"{example.prompt} {example.hallucinated_answer}"
        feats_halluc = extractor.extract_features(text_halluc)
        tracker_halluc = HypothesisTracker(config=config)
        tracker_halluc.initialize_tracks(extractor.get_top_k_features(feats_halluc[0]), 0)
        for l in range(1, 12):
            tracker_halluc.update_tracks(l, extractor.get_top_k_features(feats_halluc[l]))
            
        trackers_factual.append(tracker_fact)
        trackers_hallucinated.append(tracker_halluc)
        
        # Predict
        pred_fact = detector.predict_proba([tracker_fact])[0, 1]
        pred_halluc = detector.predict_proba([tracker_halluc])[0, 1]
        
        # Dashboard: Factual
        create_interactive_dashboard(
            tracker=tracker_fact,
            text=text_fact,
            prediction=pred_fact,
            is_hallucination=False,
            output_dir=output_dir / f"example_{i}_factual"
        )
        
        # Dashboard: Hallucinated
        create_interactive_dashboard(
            tracker=tracker_halluc,
            text=text_halluc,
            prediction=pred_halluc,
            is_hallucination=True,
            output_dir=output_dir / f"example_{i}_hallucinated"
        )
        
    # Generate Case Studies
    print("\nGenerating Detailed Case Studies...")
    cs_generator.generate_batch_studies(
        examples=examples,
        trackers_factual=trackers_factual,
        trackers_hallucinated=trackers_hallucinated,
        output_dir=output_dir / "case_studies",
        num_studies=len(examples)
    )
    
    print("\n================================================================")
    print("Visualization Pipeline Complete!")
    print(f"Results saved to {output_dir}")
    print("================================================================")

if __name__ == '__main__':
    main()
