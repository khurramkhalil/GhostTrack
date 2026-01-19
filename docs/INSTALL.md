# Installation Guide

## Requirements

- Python 3.8+
- PyTorch 2.0+ with CUDA support (recommended)
- 8GB+ GPU memory for inference
- 16GB+ GPU memory for SAE training

## Quick Install

```bash
# Clone repository
git clone https://github.com/your-org/ghosttrack.git
cd ghosttrack

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

## Dependencies

Core packages:
- `torch>=2.0.0` - PyTorch
- `transformers>=4.30.0` - Hugging Face Transformers
- `datasets>=2.14.0` - Dataset loading
- `scikit-learn>=1.0.0` - Classification
- `numpy`, `scipy` - Scientific computing
- `tqdm` - Progress bars

## Pretrained Models

Download pretrained SAEs (required for inference):

```bash
# From Hugging Face Hub
python scripts/download_checkpoints.py

# Or manually
wget https://huggingface.co/ghosttrack/sae-gpt2/resolve/main/checkpoints.zip
unzip checkpoints.zip -d models/
```

## Verification

```bash
# Run demo to verify installation
python examples/demo.py
```

Expected output:
```
GhostTrack Demo: Hallucination Detection
...
Factual Answer:       5.2% hallucination probability
Hallucinated Answer: 94.8% hallucination probability
```

## Troubleshooting

**CUDA not available**: Ensure PyTorch is installed with CUDA support:
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

**Out of Memory**: Reduce batch size in scripts or use CPU:
```bash
python scripts/run_detection_pipeline.py --device cpu
```
