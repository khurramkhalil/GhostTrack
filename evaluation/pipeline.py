
import time
import numpy as np
from tqdm import tqdm
from tracking import HypothesisTracker
from detection import HallucinationDetector
from detection.divergence_metrics import DivergenceMetrics

def process_dataset(dataset, extractor, tracking_config, top_k=50, name="Data", batch_size=8):
    """
    Process a dataset to generate hypothesis tracks and extract features immediately.
    Optimized for memory by discarding trackers after feature extraction.
    
    Args:
        dataset: List of truthfulQA examples.
        extractor: Initialized LayerwiseFeatureExtractor.
        tracking_config: Configuration dictionary for HypothesisTracker.
        top_k: Number of features to keep per layer (default 50).
        name: Name of the dataset split (for printing).
        batch_size: Batch size for feature extraction (default 8).
        
    Returns:
        X: Feature matrix [n_samples, n_features].
        y: Labels [n_samples].
    """
    X_list = []
    labels = []
    metrics_computer = DivergenceMetrics()
    
    print(f"Processing {name} ({len(dataset)} examples*2) with batch_size={batch_size}...")
    start_time = time.time()
    
    # Process in batches
    for i in tqdm(range(0, len(dataset), batch_size)):
        batch_examples = dataset[i : i + batch_size]
        current_batch_size = len(batch_examples)
        
        # --- Process Factual Answers (Label 0) ---
        texts_factual = [f"{ex.prompt} {ex.factual_answer}" for ex in batch_examples]
        features_list = extractor.extract_batch_features(texts_factual)
        
        for b in range(current_batch_size):
            tracker = HypothesisTracker(config=tracking_config)
            mask = features_list[0]['attention_mask'][b]
            
            # Layer 0
            l0_data = features_list[0]
            l0_input = {'activations': l0_data['activations'][b], 'layer': 0}
            top_l0 = extractor.get_top_k_features(l0_input, k=top_k, attention_mask=mask)
            tracker.initialize_tracks(top_l0, token_pos=0)
            
            # Layers 1..11
            for layer_idx in range(1, 12):
                l_data = features_list[layer_idx]
                l_input = {'activations': l_data['activations'][b], 'layer': layer_idx}
                top_feats = extractor.get_top_k_features(l_input, k=top_k, attention_mask=mask)
                tracker.update_tracks(layer_idx, top_feats)
            
            # Extract features and discard tracker
            feat_vec = metrics_computer.get_feature_vector(tracker, num_layers=12)
            X_list.append(feat_vec)
            labels.append(0)
            del tracker

        # --- Process Hallucinated Answers (Label 1) ---
        texts_halluc = [f"{ex.prompt} {ex.hallucinated_answer}" for ex in batch_examples]
        features_list = extractor.extract_batch_features(texts_halluc)
        
        for b in range(current_batch_size):
            tracker = HypothesisTracker(config=tracking_config)
            mask = features_list[0]['attention_mask'][b]
            
            # Layer 0
            l0_data = features_list[0]
            l0_input = {'activations': l0_data['activations'][b], 'layer': 0}
            top_l0 = extractor.get_top_k_features(l0_input, k=top_k, attention_mask=mask)
            tracker.initialize_tracks(top_l0, token_pos=0)
            
            # Layers 1..11
            for layer_idx in range(1, 12):
                l_data = features_list[layer_idx]
                l_input = {'activations': l_data['activations'][b], 'layer': layer_idx}
                top_feats = extractor.get_top_k_features(l_input, k=top_k, attention_mask=mask)
                tracker.update_tracks(layer_idx, top_feats)
            
            # Extract features and discard tracker
            feat_vec = metrics_computer.get_feature_vector(tracker, num_layers=12)
            X_list.append(feat_vec)
            labels.append(1)
            del tracker
        
    elapsed = time.time() - start_time
    print(f"Finished {name} in {elapsed:.1f}s")
    return np.array(X_list), np.array(labels)


def train_and_evaluate(X_train, y_train, X_test, y_test):
    """
    Train a hallucination detector and evaluate it using precomputed features.
    
    Args:
        X_train: Training features.
        y_train: Training labels.
        X_test: Test features.
        y_test: Test labels.
        
    Returns:
        detector: Trained HallucinationDetector.
        metrics: Dictionary of evaluation metrics.
        feature_importance: Dictionary of feature importances.
    """
    print("Training Hallucination Detector (on precomputed features)...")
    detector = HallucinationDetector(model_type='random_forest', n_estimators=100)
    detector.fit_features(X_train, y_train)
    print("Detector trained.")
    
    metrics = detector.evaluate_features(X_test, y_test)
    feat_names = detector.get_feature_names()
    importance = detector.get_feature_importance()
    
    # Sort importances
    sorted_importance = {}
    if importance is not None:
        sorted_indices = np.argsort(importance)[::-1]
        sorted_importance = {feat_names[i]: float(importance[i]) for i in sorted_indices}
    
    return detector, metrics, sorted_importance
