"""
Evaluation pipeline for hallucination detection.
"""

import numpy as np
from typing import List, Dict, Tuple
from tqdm import tqdm

from data import load_truthfulqa, HallucinationExample
from models import GPT2WithResidualHooks
from tracking import LayerwiseFeatureExtractor, HypothesisTracker
from detection import HallucinationDetector


def evaluate_detector(
    detector: HallucinationDetector,
    test_examples: List[HallucinationExample],
    model: GPT2WithResidualHooks,
    extractor: LayerwiseFeatureExtractor,
    config: Dict,
    verbose: bool = True
) -> Dict[str, float]:
    """
    Evaluate detector on test examples.

    Args:
        detector: Trained HallucinationDetector.
        test_examples: List of test examples.
        model: GPT2 model wrapper.
        extractor: Feature extractor.
        config: Tracking configuration.
        verbose: Whether to show progress.

    Returns:
        Dictionary of evaluation metrics.
    """
    # Create trackers for test examples
    test_trackers = []
    test_labels = []

    iterator = tqdm(test_examples) if verbose else test_examples

    for example in iterator:
        # Process factual answer
        factual_tracker = process_text(
            text=example.prompt + " " + example.factual_answer,
            model=model,
            extractor=extractor,
            config=config
        )
        test_trackers.append(factual_tracker)
        test_labels.append(0)  # Factual

        # Process hallucinated answer
        halluc_tracker = process_text(
            text=example.prompt + " " + example.hallucinated_answer,
            model=model,
            extractor=extractor,
            config=config
        )
        test_trackers.append(halluc_tracker)
        test_labels.append(1)  # Hallucinated

    test_labels = np.array(test_labels)

    # Evaluate
    metrics = detector.evaluate(test_trackers, test_labels)

    if verbose:
        print("\nEvaluation Results:")
        print(f"  Accuracy:  {metrics['accuracy']:.4f}")
        print(f"  AUROC:     {metrics['auroc']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall:    {metrics['recall']:.4f}")
        print(f"  F1:        {metrics['f1']:.4f}")

    return metrics


def process_text(
    text: str,
    model: GPT2WithResidualHooks,
    extractor: LayerwiseFeatureExtractor,
    config: Dict
) -> HypothesisTracker:
    """
    Process text and create hypothesis tracker.

    Args:
        text: Input text.
        model: GPT2 model.
        extractor: Feature extractor.
        config: Tracking config.

    Returns:
        HypothesisTracker with completed tracking.
    """
    # Extract features for all layers
    layer_features = extractor.extract_features(text)

    # Create tracker
    tracker = HypothesisTracker(config=config)

    # Initialize from layer 0
    top_k = config.get('top_k_features', 50)
    layer_0_features = extractor.get_top_k_features(
        layer_features[0],
        k=top_k
    )
    tracker.initialize_tracks(layer_0_features, token_pos=0)

    # Update through remaining layers
    for layer_idx in range(1, len(layer_features)):
        top_features = extractor.get_top_k_features(
            layer_features[layer_idx],
            k=top_k
        )
        tracker.update_tracks(layer_idx, top_features)

    return tracker


def run_full_evaluation(
    model_type: str = 'random_forest',
    checkpoint_dir: str = './models/checkpoints',
    device: str = 'cuda',
    test_size: int = None,
    verbose: bool = True
) -> Dict[str, float]:
    """
    Run complete evaluation pipeline.

    Args:
        model_type: Type of detector to use.
        checkpoint_dir: Directory with SAE checkpoints.
        device: Device to use.
        test_size: Number of test examples (None = use all).
        verbose: Show progress.

    Returns:
        Evaluation metrics.
    """
    if verbose:
        print("Loading TruthfulQA dataset...")

    # Load data
    train_data, val_data, test_data = load_truthfulqa()

    if test_size is not None:
        test_data = test_data[:test_size]

    if verbose:
        print(f"Loaded {len(train_data)} train, {len(val_data)} val, {len(test_data)} test examples")

    # Load model
    if verbose:
        print(f"\nLoading GPT-2 model on {device}...")

    model = GPT2WithResidualHooks('gpt2', device=device)

    # Load SAE extractor
    if verbose:
        print(f"Loading SAE checkpoints from {checkpoint_dir}...")

    extractor = LayerwiseFeatureExtractor.load_from_checkpoints(
        model_wrapper=model,
        checkpoint_dir=checkpoint_dir,
        device=device
    )

    # Tracking configuration
    config = {
        'birth_threshold': 0.5,
        'death_threshold': 0.1,
        'association_threshold': 0.5,
        'semantic_weight': 0.6,
        'activation_weight': 0.2,
        'position_weight': 0.2,
        'top_k_features': 50,
        'use_greedy': False
    }

    # Process training data
    if verbose:
        print("\nProcessing training data...")

    train_trackers = []
    train_labels = []

    for example in tqdm(train_data) if verbose else train_data:
        # Factual
        factual_tracker = process_text(
            text=example.prompt + " " + example.factual_answer,
            model=model,
            extractor=extractor,
            config=config
        )
        train_trackers.append(factual_tracker)
        train_labels.append(0)

        # Hallucinated
        halluc_tracker = process_text(
            text=example.prompt + " " + example.hallucinated_answer,
            model=model,
            extractor=extractor,
            config=config
        )
        train_trackers.append(halluc_tracker)
        train_labels.append(1)

    train_labels = np.array(train_labels)

    # Train detector
    if verbose:
        print(f"\nTraining {model_type} detector...")

    detector = HallucinationDetector(
        model_type=model_type,
        num_layers=model.n_layers
    )
    detector.fit(train_trackers, train_labels)

    # Evaluate on test set
    if verbose:
        print("\nEvaluating on test set...")

    metrics = evaluate_detector(
        detector=detector,
        test_examples=test_data,
        model=model,
        extractor=extractor,
        config=config,
        verbose=verbose
    )

    return metrics
