# 🔮 GhostTrack

[![AUROC](https://img.shields.io/badge/AUROC-98.5%25-brightgreen?style=for-the-badge)](results_downloaded/detection/results.json)
[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

**Multi-Hypothesis Tracking for Interpretable Hallucination Detection in Large Language Models**

---

## 📋 Table of Contents

- [The Challenge](#-the-challenge)
- [The GhostTrack Solution](#-the-ghosttrack-solution)
- [Key Results](#-key-results)
- [How It Works](#-how-it-works)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Project Structure](#-project-structure)
- [Ablation Studies](#-ablation-studies)
- [Citation](#-citation)

---

## � The Challenge

Large Language Models (LLMs) suffer from a critical reliability issue: **hallucination**. They frequently generate plausible-sounding but factually incorrect statements with high confidence.

Existing detection methods (like `SelfCheckGPT` or linear probes) face two major limitations:
1.  **Black Box Nature**: They provide a probability score (e.g., "85% hallucinated") but offer **no explanation** for *why* the model is hallucinating.
2.  **Static Analysis**: Most methods look at frozen snapshots of activations, ignoring the **dynamic competition** between ideas that occurs during generation.

We need a system that not only detects errors but explains the *internal mechanism* of the failure.

---

## 💡 The GhostTrack Solution

**GhostTrack** reimagines hallucination detection as a **multi-object tracking** problem. 

Instead of treating the model's internal state as an opaque vector, we use **Sparse Autoencoders (SAEs)** to decompose activations into interpretable "semantic concepts." We then track these concepts as they evolve layer-by-layer.

### The "Ghost Track" Phenomenon
Our framework reveals a distinctive pattern:
- **Factual Generation**: A single, dominant semantic track emerges early and remains stable through all layers.
- **Hallucination**: Multiple competing "ghost tracks" appear, flicker, and fight for dominance, creating high entropy and low stability.

By measuring the physics of these tracks—their stability, competition, and survival rates—we can detect hallucinations with unprecedented accuracy *and* explainability.

---

## 🎯 Key Results

Testing on the **TruthfulQA** benchmark with GPT-2 (124M), GhostTrack achieves state-of-the-art detection performance.

| Metric | Value |
|:-------|------:|
| **AUROC** | 98.5% |
| **Accuracy** | 93.0% |
| **Precision** | 96.9% |
| **Recall** | 88.8% |

### Critical Discovery
> **Removing multi-hypothesis tracking drops AUROC from 98.5% → 57%**.

This single result proves that **tracking the competition** between hypotheses is the key to detecting hallucinations. Simple feature analysis fails; you *must* track the dynamic conflict.

### Top Predictive Features

| Feature | Importance | Interpretation |
|:--------|:----------:|:---------------|
| `entropy_std` | 11.0% | **High Variance**: Confusion spikes at specific layers |
| `stability_mean` | 10.6% | **Flickering**: Tracks appearing/disappearing rapidly |
| `dominance_mean` | 10.2% | **Weak Consensus**: No single strong idea |

---

## 🔬 How It Works

```
Input Text ("The capital of France is...")
    │
    ▼
┌─────────────────┐
│  GPT-2 Layers   │  Extract residual stream activations
│  (1-12)         │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  JumpReLU SAEs  │  Decompose dense vectors into
│  (Interpretable)│  sparse concepts (e.g., "Paris", "London")
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Hypothesis    │  Link concepts across layers:
│    Tracker      │  "Paris (L4)" → "Paris (L5)"
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Divergence    │  Calculate: Is the track stable?
│    Metrics      │  Are there competing tracks?
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Hallucination  │  Classify based on track physics
│   Detector      │  (Factual vs. Hallucinated)
└─────────────────┘
```

---

## 🚀 Installation

### Prerequisites
- Python 3.8+
- PyTorch 2.0+ with CUDA (recommended)
- 8GB+ GPU memory

### Setup

```bash
# Clone repository
git clone https://github.com/your-org/ghosttrack.git
cd ghosttrack

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Download Pretrained Models

```bash
# Download SAE checkpoints (required for inference)
# Models will be placed in ./models/checkpoints/
python scripts/download_checkpoints.py
```

---

## ⚡ Quick Start

### Python API

```python
from models import GPT2WithResidualHooks
from tracking import LayerwiseFeatureExtractor, HypothesisTracker
from detection import HallucinationDetector

# 1. Load model and SAEs
model = GPT2WithResidualHooks('gpt2', device='cuda')
extractor = LayerwiseFeatureExtractor.load_from_checkpoints(
    model, './models/checkpoints'
)

# 2. Process text
text = "The capital of France is Paris."
layer_features = extractor.extract_features(text)

# 3. Track hypotheses across layers
tracker = HypothesisTracker()
tracker.initialize_tracks(extractor.get_top_k_features(layer_features[0]))
for layer in range(1, 12):
    tracker.update_tracks(layer, extractor.get_top_k_features(layer_features[layer]))

# 4. Detect hallucination
detector = HallucinationDetector.load('./results/detection/detector.pkl')
probability = detector.predict_proba([tracker])[0, 1]
print(f"Hallucination probability: {probability:.1%}")
```

### Command Line

```bash
# Run demo
python examples/demo.py

# Run full pipeline
python scripts/run_detection_pipeline.py --device cuda
```

---

## 📁 Project Structure

```
ghosttrack/
├── models/                 # Model wrappers and SAE implementations
│   ├── model_wrapper.py    # GPT-2 with activation hooks
│   └── sae.py              # JumpReLU Sparse Autoencoder
├── tracking/               # Hypothesis tracking system
│   ├── hypothesis_tracker.py
│   ├── track_association.py
│   └── feature_extractor.py
├── detection/              # Hallucination detection
│   ├── detector.py         # Random Forest classifier
│   └── divergence_metrics.py
├── visualization/          # Dashboards and plots
├── scripts/                # Pipeline scripts
├── paper/                  # LaTeX paper source
└── examples/               # Demo scripts
```

---

## 🔍 Ablation Studies

We validate each component's contribution:

| Configuration | AUROC | Δ Baseline | Interpretation |
|:--------------|------:|-----------:|:---------------|
| **Baseline** | 98.5% | — | Full tracking system |
| **Single Hypothesis** | 57.0% | **-41.5%** | Fails without tracking competition |
| **No Association** | 97.9% | -0.6% | Temporal continuity matters |
| **Random Association** | 97.6% | -0.9% | Semantic meaning matters |

---

## 📝 Citation

```bibtex
@article{ghosttrack2026,
  title={GhostTrack: Multi-Hypothesis Tracking for Interpretable Hallucination Detection},
  author={Anonymous},
  journal={arXiv preprint},
  year={2026}
}
```

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
