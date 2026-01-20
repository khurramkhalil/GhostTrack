# GhostTrack Model Configurations

## Directory Organization

We now have organized directory structures for each model variant:

```
GhostTrack/
├── config/model_configs/
│   ├── gpt2-small.yaml      # GPT-2 Small (12 layers, 768 dim)
│   └── gpt2-medium.yaml     # GPT-2 Medium (24 layers, 1024 dim)
├── data/cache/
│   ├── gpt2-small/
│   │   └── hidden_states/   # Extracted activations for GPT-2 small
│   └── gpt2-medium/
│       └── hidden_states/   # Extracted activations for GPT-2 medium
├── models/checkpoints/
│   ├── gpt2-small/          # Trained SAEs for GPT-2 small
│   └── gpt2-medium/         # Trained SAEs for GPT-2 medium
├── results/
│   ├── gpt2-small/          # Analysis results for GPT-2 small
│   └── gpt2-medium/         # Analysis results for GPT-2 medium
└── jobs/
    ├── full_pipeline.sbatch              # GPT-2 small full pipeline
    └── gpt2_medium_full_pipeline.sbatch  # GPT-2 medium full pipeline
```

## Model Comparison

| Aspect | GPT-2 Small | GPT-2 Medium |
|--------|-------------|--------------|
| **Model Specs** |
| HuggingFace ID | `gpt2` | `gpt2-medium` |
| Layers | 12 | 24 |
| Hidden Dimension | 768 | 1024 |
| Parameters | 117M | 345M |
| **SAE Specs** |
| SAE Hidden Dim | 4096 | 5120 |
| Expansion Ratio | 5.33x | 5.0x |
| Params per SAE | ~6.3M | ~10.5M |
| Total SAE Params | ~75M (12 SAEs) | ~252M (24 SAEs) |
| **Batch Sizes** |
| Extraction | 32 | 16 |
| Training | 256 | 128 |
| **Memory Estimates** |
| Model Size | ~500 MB | ~1.4 GB |
| Per-Layer States | ~300 MB | ~400 MB |
| Training Peak | ~8 GB | ~14 GB |
| **GPU Utilization** |
| A100 Utilization | ~60-70% | ~75-85% |
| Layers per GPU | 3 | 6 |
| **Time Estimates** |
| Extraction (10M tokens/layer) | ~30 min | ~60 min |
| Training (20 epochs) | ~45 min | ~90 min |
| Total per Layer | ~75 min | ~150 min |
| **Total Pipeline** | ~15 hours | ~60 hours |

## Batch Size Rationale

### Extraction Batch Sizes

**GPT-2 Small (32):**
- Model memory: ~500MB
- Batch of 32 sequences (512 tokens each): ~4GB activations
- Forward pass peak: ~6GB
- Leaves plenty of headroom on A100 (40GB)
- Good throughput without OOM risk

**GPT-2 Medium (16):**
- Model memory: ~1.4GB
- Larger hidden dimension (1024 vs 768)
- Batch of 16 keeps memory under 8GB
- Conservative to avoid OOM with larger model
- Still achieves good GPU utilization

### Training Batch Sizes

**GPT-2 Small (256):**
- SAE size: ~6.3M params (~25MB)
- Batch of 256: ~200MB activations
- Gradients + optimizer states: ~2GB
- Total: ~4GB, excellent A100 utilization
- Fast convergence with large batches

**GPT-2 Medium (128):**
- SAE size: ~10.5M params (~42MB)
- Larger hidden dimension increases memory per sample
- Batch of 128: ~200MB activations
- Gradients + optimizer states: ~3.5GB
- Total: ~6GB, good utilization without OOM
- Halved from small but still efficient

### Why These Choices?

1. **No Under-utilization**: Both configs keep GPUs busy
   - Memory usage: 60-80% of A100 capacity
   - Compute: High arithmetic intensity

2. **No OOM**: Conservative enough to avoid crashes
   - ~30-40% headroom for memory spikes
   - Accounts for PyTorch overhead and fragmentation

3. **Optimal Throughput**:
   - Large enough for good data parallelism
   - Small enough to fit in memory
   - Near-optimal batch sizes for Adam optimizer

## Configuration Files

### GPT-2 Small (`config/model_configs/gpt2-small.yaml`)

```yaml
model:
  base_model: gpt2
  d_model: 768
  n_layers: 12

sae:
  d_model: 768
  d_hidden: 4096

sae_training:
  batch_size: 256

paths:
  cache_dir: ./data/cache/gpt2-small
  models_dir: ./models/checkpoints/gpt2-small
  results_dir: ./results/gpt2-small
```

### GPT-2 Medium (`config/model_configs/gpt2-medium.yaml`)

```yaml
model:
  base_model: gpt2-medium
  d_model: 1024
  n_layers: 24

sae:
  d_model: 1024
  d_hidden: 5120

sae_training:
  batch_size: 128  # Reduced for larger model

paths:
  cache_dir: ./data/cache/gpt2-medium
  models_dir: ./models/checkpoints/gpt2-medium
  results_dir: ./results/gpt2-medium
```

## Pipeline Scripts

### GPT-2 Small
**Job Script**: `jobs/full_pipeline.sbatch`
**Launch**: `sbatch jobs/full_pipeline.sbatch`

Features:
- 4 GPUs × 3 layers = 12 layers total
- Extraction batch size: 32
- Training batch size: 256
- Estimated time: ~15 hours

### GPT-2 Medium
**Job Script**: `jobs/gpt2_medium_full_pipeline.sbatch`
**Launch**: `sbatch jobs/gpt2_medium_full_pipeline.sbatch`

Features:
- 4 GPUs × 6 layers = 24 layers total
- Extraction batch size: 16
- Training batch size: 128
- Estimated time: ~60 hours
- Includes all 5 phases:
  1. Hidden State Extraction
  2. SAE Training
  3. Hypothesis Tracking
  4. Hallucination Detection
  5. Evaluation & Metrics

## Running the Pipelines

### Quick Launch

```bash
# For GPT-2 Small (existing results will be organized)
sbatch jobs/full_pipeline.sbatch

# For GPT-2 Medium (new analysis)
sbatch jobs/gpt2_medium_full_pipeline.sbatch
```

### Monitor Progress

```bash
# Check job status
squeue -u $USER

# Watch output
tail -f gpt2s-full_*.out  # For small
tail -f gpt2m-full_*.out  # For medium

# Check GPU usage
ssh <compute-node>
nvidia-smi
```

### Results Structure

After completion:
```
results/
├── gpt2-small/
│   ├── tracking/
│   ├── detection/
│   └── evaluation/
└── gpt2-medium/
    ├── tracking/
    ├── detection/
    └── evaluation/
```

## Next Steps

1. **Launch GPT-2 Small**: Re-run with organized structure
2. **Launch GPT-2 Medium**: Start full pipeline
3. **Compare Results**: Analyze differences between model sizes
4. **Paper Results**: Use medium results for publication

## Troubleshooting

### Out of Memory (OOM)

If you encounter OOM errors:

**GPT-2 Small:**
- Reduce extraction batch size: 32 → 24 → 16
- Reduce training batch size: 256 → 192 → 128

**GPT-2 Medium:**
- Reduce extraction batch size: 16 → 12 → 8
- Reduce training batch size: 128 → 96 → 64

Edit the variables at the top of the .sbatch file:
```bash
EXTRACT_BATCH_SIZE=16  # Adjust this
TRAIN_BATCH_SIZE=128   # And this
```

### Under-utilization

If GPUs are idle (check with `nvidia-smi`):

- Increase batch sizes (opposite direction)
- Check if data loading is bottleneck
- Verify num_workers in DataLoader (currently 8)

### Time Limits

If jobs exceed time limits:
- Adjust `#SBATCH --time=168:00:00`
- Or split into separate extraction/training jobs
