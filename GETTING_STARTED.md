# Getting Started with GhostTrack

## Quick Setup

### 1. Verify Environment

GhostTrack uses your existing `pt` conda environment. Ensure it has:

```bash
conda activate pt

# Check Python version (3.10+)
python --version

# Check PyTorch
python -c "import torch; print(torch.__version__)"

# Check Transformers
python -c "import transformers; print(transformers.__version__)"
```

### 2. Install Missing Dependencies

If any are missing, install them:

```bash
pip install pyyaml datasets scikit-learn
```

### 3. Run Tests

Verify Phase 1 implementation:

```bash
# Run all tests
python3 run_tests.py

# Or individual modules
python3 run_tests.py --module test_config
python3 run_tests.py --module test_data_loader
python3 run_tests.py --module test_model_wrapper  # Requires GPU/CPU
python3 run_tests.py --module test_sae_model
```

**Note**: Some tests require downloading GPT-2 (~500MB) and TruthfulQA dataset.

---

## Usage Examples

### Example 1: Load Configuration

```python
from config import load_config

config = load_config()
print(f"Model: {config.model.base_model}")
print(f"SAE hidden dim: {config.sae.d_hidden}")
```

### Example 2: Load TruthfulQA Data

```python
from data import load_truthfulqa

# Load and split dataset
train, val, test = load_truthfulqa(cache_dir='./data/cache', seed=42)

# Examine an example
example = train[0]
print(f"Question: {example.prompt}")
print(f"Factual: {example.factual_answer}")
print(f"Hallucinated: {example.hallucinated_answer}")
```

### Example 3: Extract Model Activations

```python
from models import GPT2WithResidualHooks

# Load model
model = GPT2WithResidualHooks('gpt2')

# Process text
outputs = model.process_text("The quick brown fox")

# Access activations
print(f"Layers: {len(outputs['residual_stream'])}")
print(f"Shape: {outputs['residual_stream'][0].shape}")
# Output: [batch, seq_len, 768]
```

### Example 4: Use Sparse Autoencoder

```python
import torch
from models import JumpReLUSAE

# Create SAE
sae = JumpReLUSAE(d_model=768, d_hidden=4096, threshold=0.1)

# Encode some activations
x = torch.randn(1, 10, 768)  # [batch, seq, d_model]
output = sae.forward(x)

print(f"Sparsity: {output['sparsity'].item():.3f}")
print(f"Active features: {sae.count_active_features(x):.1f}")
```

---

## Interactive Exploration

Launch Jupyter notebook for interactive demos:

```bash
jupyter notebook notebooks/01_phase1_quickstart.ipynb
```

This notebook includes:
- Configuration examples
- Data loading and analysis
- Model activation extraction
- SAE feature visualization
- Full pipeline demonstration

---

## Project Structure

```
GhostTrack/
├── config/           # Configuration management
│   └── config_loader.py
├── data/             # Data loading
│   └── data_loader.py
├── models/           # Model implementations
│   ├── model_wrapper.py  # GPT-2 + hooks
│   └── sae_model.py      # JumpReLU SAE
├── tests/            # Test suite (66 tests)
└── notebooks/        # Interactive demos
```

---

## Understanding the Tests

All tests are **genuine** and validate real functionality:

### Config Tests (`test_config.py`)
- Loading/saving YAML configs
- Default values
- Path creation
- Weight validation

### Data Tests (`test_data_loader.py`)
- TruthfulQA loading
- Train/val/test splitting
- Stratification
- Category analysis

### Model Tests (`test_model_wrapper.py`)
- Hook registration
- Activation extraction
- Shape validation
- Batch processing

### SAE Tests (`test_sae_model.py`)
- JumpReLU activation
- Encoder/decoder
- Sparsity enforcement
- Loss computation

---

## Common Issues

### Issue: Module not found errors
**Solution**: Ensure you're in the project root and `pt` env is activated:
```bash
conda activate pt
cd /path/to/GhostTrack
python run_tests.py
```

### Issue: GPU out of memory
**Solution**: Use smaller models or CPU:
```python
model = GPT2WithResidualHooks('gpt2', device='cpu')
```

### Issue: Dataset download fails
**Solution**: Check internet connection or use cached data:
```python
from datasets import load_dataset
dataset = load_dataset('truthful_qa', 'generation', cache_dir='./data/cache')
```

---

## Next Steps

### Phase 1 Complete ✅
You now have:
- Configuration system
- Data pipeline
- Model instrumentation
- SAE architecture

### Start Phase 2: SAE Training

See `IMPLEMENTATION_PLAN.md` for Phase 2 details:
1. Extract hidden states from Wikipedia
2. Train 12 SAEs (one per layer)
3. Validate reconstruction < 0.01
4. Interpret learned features

Estimated time: 2 weeks

---

## Resources

- **Full Plan**: `IMPLEMENTATION_PLAN.md` (8-week roadmap)
- **Phase 1 Summary**: `PHASE1_SUMMARY.md` (detailed docs)
- **PRD**: `prd.md` (original requirements)
- **Interactive Demo**: `notebooks/01_phase1_quickstart.ipynb`

---

## Questions?

1. Check `PHASE1_SUMMARY.md` for component details
2. Review test files for usage examples
3. Run the Jupyter notebook for interactive exploration

Happy tracking! 👻
