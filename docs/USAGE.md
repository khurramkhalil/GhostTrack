# Usage Guide

## Basic Usage

### 1. Single Text Analysis

```python
from models import GPT2WithResidualHooks
from tracking import LayerwiseFeatureExtractor, HypothesisTracker
from detection import HallucinationDetector

# Load components
model = GPT2WithResidualHooks('gpt2', device='cuda')
extractor = LayerwiseFeatureExtractor.load_from_checkpoints(model, './models/checkpoints')
detector = HallucinationDetector.load('./results/detection/detector.pkl')

# Analyze text
text = "The Eiffel Tower is located in Berlin."
layer_features = extractor.extract_features(text)

# Track hypotheses
tracker = HypothesisTracker()
tracker.initialize_tracks(extractor.get_top_k_features(layer_features[0]))
for i in range(1, 12):
    tracker.update_tracks(i, extractor.get_top_k_features(layer_features[i]))

# Get prediction
prob = detector.predict_proba([tracker])[0, 1]
print(f"Hallucination probability: {prob:.2%}")
```

### 2. Batch Processing

```python
from evaluation.pipeline import process_dataset, train_and_evaluate

# Process dataset
X_train, y_train = process_dataset(train_data, extractor, config, top_k=50)
X_test, y_test = process_dataset(test_data, extractor, config, top_k=50)

# Train detector
detector, metrics, importance = train_and_evaluate(X_train, y_train, X_test, y_test)
print(f"AUROC: {metrics['auroc']:.4f}")
```

### 3. Visualizations

```python
from visualization import create_interactive_dashboard

# Generate dashboard
create_interactive_dashboard(tracker, output_dir='./viz/')
```

## Command Line Interface

```bash
# Run full detection pipeline
python scripts/run_detection_pipeline.py \
    --checkpoint-dir ./models/checkpoints \
    --output-dir ./results/detection \
    --device cuda

# Run ablation studies
python scripts/run_ablations.py \
    --output-dir ./results/ablations

# Generate visualizations
python scripts/run_visualization.py \
    --output-dir ./results/visualization
```

## Configuration

Key parameters in tracking config:

| Parameter | Default | Description |
|:----------|--------:|:------------|
| `top_k` | 50 | Features tracked per layer |
| `birth_threshold` | 0.5 | Minimum activation to create track |
| `association_threshold` | 0.5 | Minimum similarity to associate |
| `semantic_weight` | 0.6 | Weight for embedding similarity |

## API Reference

See [API.md](API.md) for complete API documentation.
