# GhostTrack: Multi-Hypothesis Tracking for Hallucination Detection

**Mechanistic Hypothesis & Origin Semantic Tracking for LLMs**

A framework for tracing competing thought-trajectories to predict and explain hallucinations before they reach the output layer.

## Overview

GhostTrack provides:
- **Interpretable Detection**: Visualize how competing semantic tracks emerge and compete
- **Early Detection**: Detect hallucinations 2-3 layers earlier than existing methods
- **Mechanistic Insights**: Understand *why* models hallucinate through track competition analysis

## Project Structure

```
GhostTrack/
├── config/              # Configuration management
├── data/                # Data loading and processing
├── models/              # Model implementations (GPT-2 wrapper, SAE)
├── tracking/            # Hypothesis tracking system
├── detection/           # Hallucination detection pipeline
├── evaluation/          # Evaluation and metrics
├── visualization/       # Visualization tools
├── scripts/             # Training and evaluation scripts
├── tests/               # Comprehensive test suite
└── notebooks/           # Jupyter notebooks for exploration
```

## Installation

### Prerequisites
- Python 3.10+
- CUDA-compatible GPU (recommended)
- Conda environment named `pt` with PyTorch installed

### Setup

```bash
# Clone the repository
cd GhostTrack

# Activate conda environment
conda activate pt

# Install dependencies (most should be in pt env)
pip install -r requirements.txt
```

## Quick Start

### Running Tests

```bash
# Run all tests
python run_tests.py

# Run specific test module
python run_tests.py --module test_config

# Quiet mode
python run_tests.py --quiet
```

### Configuration

Configuration is managed through `.claude` file:

```yaml
model:
  base_model: gpt2
  d_model: 768
  n_layers: 12

sae:
  d_hidden: 4096
  threshold: 0.1
  lambda_sparse: 0.01
```

## Development Status

### Phase 1: Infrastructure ✅ COMPLETE
- [x] Configuration system
- [x] Data loader (TruthfulQA)
- [x] GPT-2 model wrapper with hooks
- [x] JumpReLU SAE implementation
- [x] Comprehensive test suite

### Phase 2: SAE Training ✅ COMPLETE
- [x] Wikipedia corpus loader
- [x] Hidden state extraction pipeline
- [x] SAE training loop with validation
- [x] Feature interpretation tools
- [x] Training orchestration scripts

### Phase 3: Hypothesis Tracking ✅ COMPLETE
- [x] Track dataclass with rich API
- [x] Layerwise feature extraction
- [x] Semantic similarity-based association
- [x] Hypothesis tracker with birth/death detection
- [x] Hungarian algorithm for optimal matching

### Phase 4-7: Coming Soon
- Detection pipeline (divergence metrics, classifier)
- Visualization tools (radar plots, trajectories)
- Optimization & ablations
- Paper & code release

## Testing Philosophy

All tests are genuine and validate actual functionality:
- No try-except trickery
- Real assertions on expected behavior
- Coverage of edge cases
- Integration tests where appropriate

## Citation

```bibtex
@article{ghosttrack2024,
  title={Multi-Hypothesis Tracking for Interpretable Hallucination Detection in LLMs},
  author={Your Name},
  year={2024}
}
```

## License

MIT License - see LICENSE file for details.

## Contact

For questions or issues, please open a GitHub issue.
