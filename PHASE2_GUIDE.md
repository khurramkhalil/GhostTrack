# Phase 2: SAE Training Guide

## Overview

Phase 2 implements the complete SAE training pipeline:
1. **Wikipedia corpus loading** - Stream Wikipedia text
2. **Hidden state extraction** - Extract activations from GPT-2
3. **SAE training** - Train JumpReLU SAEs per layer
4. **Feature interpretation** - Analyze what features represent

---

## Components Implemented

### 1. Wikipedia Corpus Loader
**File**: `data/wikipedia_loader.py`

**Features**:
- Streaming Wikipedia dataset (avoids loading everything to memory)
- Batch text processing
- Configurable language and date

**Usage**:
```python
from data import WikipediaCorpus

corpus = WikipediaCorpus(cache_dir='./data/cache')
corpus.load()

# Get text batches
for texts in corpus.get_text_batches(batch_size=32, max_batches=100):
    # Process batch
    pass
```

### 2. Hidden State Extractor
**File**: `data/wikipedia_loader.py`

**Features**:
- Extract residual stream activations from GPT-2
- Save to disk for efficient SAE training
- Progress tracking with tqdm
- Checkpoint saving

**Usage**:
```python
from models import GPT2WithResidualHooks
from data import WikipediaCorpus, HiddenStateExtractor

model = GPT2WithResidualHooks('gpt2')
corpus = WikipediaCorpus().load()

extractor = HiddenStateExtractor(model, corpus)

# Extract for one layer
state_path = extractor.extract_for_layer(
    layer_idx=6,
    num_tokens=1_000_000,  # Start small for testing
    batch_size=8
)

# Or extract all layers
paths = extractor.extract_all_layers(num_tokens=10_000_000)
```

### 3. SAE Trainer
**File**: `scripts/train_sae.py`

**Features**:
- Train-validation split
- Cosine annealing LR schedule
- Gradient clipping
- Periodic decoder normalization
- Checkpoint saving (best + final)
- Training history logging

**Usage**:
```python
from scripts.train_sae import train_sae_for_layer

history = train_sae_for_layer(
    layer_idx=6,
    hidden_states_path='./data/cache/hidden_states/layer_6_states.pt',
    save_dir='./models/checkpoints'
)
```

### 4. Feature Interpreter
**File**: `evaluation/interpret_features.py`

**Features**:
- Find top-activating examples per feature
- Extract common tokens around activations
- Identify dead features
- Save interpretations as JSON

**Usage**:
```python
from evaluation import analyze_sae
from models import GPT2WithResidualHooks

model = GPT2WithResidualHooks('gpt2')

# Prepare some texts for analysis
texts = [...]  # List of strings

interpretations = analyze_sae(
    sae_checkpoint_path='./models/checkpoints/sae_layer_6_best.pt',
    model_wrapper=model,
    layer_idx=6,
    texts=texts,
    save_dir='./results/interpretations'
)
```

---

## Complete Pipeline

### Option 1: All-in-One Script

```bash
# Extract hidden states and train SAEs for all layers
python scripts/extract_and_train.py \
    --layers all \
    --num-tokens 10000000 \
    --device cuda \
    --cache-dir ./data/cache \
    --save-dir ./models/checkpoints
```

### Option 2: Step-by-Step

#### Step 1: Extract Hidden States Only

```bash
# Extract for all layers (don't train yet)
python scripts/extract_and_train.py \
    --extract-only \
    --layers all \
    --num-tokens 10000000 \
    --batch-size-extract 8 \
    --device cuda
```

#### Step 2: Train SAEs

```bash
# Train on pre-extracted states
python scripts/extract_and_train.py \
    --train-only \
    --layers all \
    --device cuda
```

#### Step 3: Train Specific Layers

```bash
# Train only layers 0, 6, and 11
python scripts/extract_and_train.py \
    --train-only \
    --layers 0,6,11
```

#### Step 4: Interpret Features

```python
from evaluation import analyze_sae, print_feature_summary
from models import GPT2WithResidualHooks
from data import load_truthfulqa

# Load model
model = GPT2WithResidualHooks('gpt2')

# Get some texts for interpretation
train, _, _ = load_truthfulqa()
texts = [ex.prompt for ex in train[:1000]]

# Analyze each layer
for layer_idx in range(12):
    checkpoint = f'./models/checkpoints/sae_layer_{layer_idx}_best.pt'

    interpretations = analyze_sae(
        sae_checkpoint_path=checkpoint,
        model_wrapper=model,
        layer_idx=layer_idx,
        texts=texts
    )

    # Print summary of first 5 features
    for feat_id in range(5):
        print_feature_summary(interpretations[feat_id])
```

---

## Configuration

SAE training parameters are in `.claude`:

```yaml
sae:
  d_model: 768
  d_hidden: 4096
  threshold: 0.1
  lambda_sparse: 0.01

sae_training:
  epochs: 20
  batch_size: 256
  learning_rate: 0.0001
  weight_decay: 0.0
  gradient_clip: 1.0
  target_recon_loss: 0.01
  target_sparsity_min: 50
  target_sparsity_max: 100
```

---

## Expected Results

### Target Metrics

For each layer's SAE:
- **Reconstruction loss**: < 0.01
- **Active features per token**: 50-100
- **Dead features**: < 10% of total

### Training Time Estimates

On A100 GPU:
- Hidden state extraction: ~30 min per layer (10M tokens)
- SAE training: ~2-4 hours per layer (20 epochs)
- Feature interpretation: ~15 min per layer (1000 texts)

**Total for 12 layers**: ~40-60 hours

On CPU or slower GPU: 3-5x longer

---

## Troubleshooting

### Issue: Out of Memory

**Solution 1**: Reduce batch size
```bash
python scripts/extract_and_train.py --batch-size-extract 4
```

**Solution 2**: Extract fewer tokens for testing
```bash
python scripts/extract_and_train.py --num-tokens 1000000  # 1M instead of 10M
```

**Solution 3**: Use CPU
```bash
python scripts/extract_and_train.py --device cpu
```

### Issue: Reconstruction loss not improving

**Symptoms**: Loss plateaus above 0.01

**Solutions**:
1. Increase hidden dimension: `d_hidden: 8192`
2. Decrease sparsity penalty: `lambda_sparse: 0.001`
3. Train for more epochs: `epochs: 40`
4. Reduce learning rate: `learning_rate: 0.00005`

### Issue: Too many dead features

**Symptoms**: >50% features never activate

**Solutions**:
1. Decrease threshold: `threshold: 0.05`
2. Decrease sparsity penalty: `lambda_sparse: 0.005`
3. Use more diverse training data

### Issue: Training too slow

**Solutions**:
1. Increase batch size (if memory allows): `batch_size: 512`
2. Use mixed precision training (add to trainer)
3. Extract states to SSD instead of HDD
4. Use faster GPU

---

## File Structure After Phase 2

```
GhostTrack/
├── data/
│   └── cache/
│       ├── hidden_states/
│       │   ├── layer_0_states.pt
│       │   ├── layer_1_states.pt
│       │   └── ... (layer_11_states.pt)
│       └── datasets/  # HuggingFace cache
│
├── models/
│   └── checkpoints/
│       ├── sae_layer_0_best.pt
│       ├── sae_layer_0_final.pt
│       ├── sae_layer_0_history.json
│       └── ... (for all 12 layers)
│
└── results/
    └── interpretations/
        ├── layer_0_interpretations.json
        ├── layer_1_interpretations.json
        └── ... (layer_11_interpretations.json)
```

---

## Validation Checklist

Before proceeding to Phase 3, verify:

- [ ] All 12 SAEs trained successfully
- [ ] Reconstruction loss < 0.01 for all layers
- [ ] Active features per token: 50-100
- [ ] Feature interpretations saved for all layers
- [ ] At least 80% features are active (not dead)
- [ ] Training history shows convergence

---

## Quick Start Example

### Test on One Layer First

```python
# 1. Extract states for layer 6 (middle layer)
from models import GPT2WithResidualHooks
from data import WikipediaCorpus, HiddenStateExtractor

model = GPT2WithResidualHooks('gpt2', device='cuda')
corpus = WikipediaCorpus().load()
extractor = HiddenStateExtractor(model, corpus)

state_path = extractor.extract_for_layer(
    layer_idx=6,
    num_tokens=1_000_000,  # 1M for quick test
    batch_size=4
)

# 2. Train SAE
from scripts.train_sae import train_sae_for_layer

history = train_sae_for_layer(
    layer_idx=6,
    hidden_states_path=state_path,
    save_dir='./models/checkpoints',
    device='cuda'
)

# 3. Check results
print(f"Best recon loss: {min(history['recon_loss']):.6f}")
print(f"Final sparsity: {history['sparsity'][-1]:.4f}")

# 4. Interpret features
from evaluation import analyze_sae
from data import load_truthfulqa

train_data, _, _ = load_truthfulqa()
texts = [ex.prompt for ex in train_data[:500]]

interpretations = analyze_sae(
    sae_checkpoint_path='./models/checkpoints/sae_layer_6_best.pt',
    model_wrapper=model,
    layer_idx=6,
    texts=texts
)

# 5. Print summary
from evaluation.interpret_features import print_feature_summary

for feat_id in range(10):  # First 10 features
    print_feature_summary(interpretations[feat_id])
```

If this works well, scale to all 12 layers!

---

## Next Steps: Phase 3

Once Phase 2 is complete:

1. ✅ Hidden states extracted
2. ✅ SAEs trained and validated
3. ✅ Features interpreted

**Ready for Phase 3**: Hypothesis Tracking System
- Build Track dataclass
- Implement feature extraction per layer
- Create semantic similarity-based association
- Build hypothesis tracker with birth/death

See `IMPLEMENTATION_PLAN.md` for Phase 3 details.

---

## Questions?

- **Training issues**: Check troubleshooting section
- **Performance**: See expected metrics
- **Next steps**: See Phase 3 in implementation plan
