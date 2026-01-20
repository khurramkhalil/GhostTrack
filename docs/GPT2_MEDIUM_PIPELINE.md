# GPT-2 Medium Analysis Pipeline

This directory contains the complete pipeline for analyzing GPT-2 Medium with GhostTrack's multi-hypothesis tracking system.

## Quick Start

```bash
# Launch the full pipeline
sbatch jobs/gpt2_medium_full_pipeline.sbatch

# Or use the helper script
./launch_model_analysis.sh gpt2-medium
```

## Pipeline Overview

The pipeline runs 5 complete phases on GPT-2 Medium (24 layers, 1024 hidden dim):

### Phase 1: Hidden State Extraction
- **Parallelization**: 4 GPUs × 6 layers = 24 layers
- **Batch Size**: 16 (optimized for 1024 dim)
- **Output**: ~9.6M tokens × 24 layers = 230M activation vectors
- **Time**: ~60 minutes
- **Memory**: ~10 GB per GPU

### Phase 2: SAE Training
- **Parallelization**: 4 GPUs × 6 SAEs = 24 SAEs
- **Batch Size**: 128 (optimized for A100)
- **Architecture**: JumpReLU with 5120 hidden units
- **Epochs**: 20
- **Time**: ~90 minutes per SAE
- **Output**: 24 trained SAE checkpoints (~250 MB each)

### Phase 3: Hypothesis Tracking
- **Input**: Trained SAEs from Phase 2
- **Process**: Track feature hypotheses across generation
- **Output**: Hypothesis trajectories and semantic clusters
- **Time**: ~30 minutes

### Phase 4: Hallucination Detection
- **Input**: Hypothesis tracks from Phase 3
- **Process**: Detect hallucinations using entropy and churn
- **Output**: Detection scores and classifications
- **Time**: ~20 minutes

### Phase 5: Evaluation & Metrics
- **Input**: Detection results from Phase 4
- **Metrics**: AUROC, Precision, Recall, F1
- **Output**: Final evaluation report
- **Time**: ~10 minutes

## Configuration

Configuration file: `config/model_configs/gpt2-medium.yaml`

Key settings:
```yaml
model:
  base_model: gpt2-medium
  d_model: 1024
  n_layers: 24

sae:
  d_hidden: 5120  # 5x expansion

sae_training:
  batch_size: 128  # Optimized for A100
  epochs: 20
```

## Resource Requirements

- **GPUs**: 4× NVIDIA A100 (40GB)
- **CPU**: 16 cores per task × 4 tasks
- **RAM**: 480 GB total
- **Storage**: ~200 GB
  - Hidden states: ~60 GB
  - Checkpoints: ~6 GB
  - Results: ~5 GB
- **Time**: ~60-72 hours total

## Output Structure

```
results/gpt2-medium/
├── tracking/
│   ├── hypothesis_trajectories.json
│   ├── semantic_clusters.pkl
│   └── feature_activations.pt
├── detection/
│   ├── hallucination_scores.json
│   ├── classifications.csv
│   └── detection_metrics.json
└── evaluation/
    ├── final_report.json
    ├── confusion_matrix.png
    └── roc_curve.png

models/checkpoints/gpt2-medium/
├── sae_layer_0_best.pt
├── sae_layer_1_best.pt
...
└── sae_layer_23_best.pt

data/cache/gpt2-medium/
└── hidden_states/
    ├── layer_0_states.pt
    ├── layer_1_states.pt
    ...
    └── layer_23_states.pt
```

## Monitoring

### Check Job Status
```bash
squeue -u $USER
```

### Watch Output
```bash
tail -f gpt2m-full_<JOBID>.out
```

### Check GPU Usage
```bash
# SSH to compute node (get from squeue)
ssh cn-<node>
nvidia-smi
```

### Monitor Disk Usage
```bash
watch -n 60 'du -sh data/cache/gpt2-medium/ models/checkpoints/gpt2-medium/'
```

## Batch Size Tuning

If you encounter OOM errors or want to optimize:

### Reduce for OOM
Edit the sbatch file variables:
```bash
EXTRACT_BATCH_SIZE=12  # Down from 16
TRAIN_BATCH_SIZE=96    # Down from 128
```

### Increase for Better Utilization
```bash
EXTRACT_BATCH_SIZE=20  # Up from 16
TRAIN_BATCH_SIZE=160   # Up from 128
```

Monitor GPU memory with `nvidia-smi` to find optimal settings.

## Expected Performance

### GPU Utilization
- **Extraction**: 75-85%
- **Training**: 80-90%
- **Tracking**: 60-70%
- **Detection**: 50-60%

### Throughput
- **Extraction**: ~167K tokens/sec
- **Training**: ~33K samples/sec
- **Total**: ~2.5 hours per layer (extraction + training)

### Quality Metrics (Expected)
- **Reconstruction Loss**: < 0.01
- **Sparsity**: 50-100 active features
- **AUROC**: > 0.90
- **F1 Score**: > 0.85

## Troubleshooting

### Job Stuck in Queue
```bash
# Check partition availability
sinfo -p hoquek-lab-gpu

# Check your quota
sacctmgr show assoc user=$USER format=User,Account,Partition,QOS
```

### Extraction Running Slowly
- Check batch size (may be too small)
- Verify data loader num_workers (should be 8)
- Check if data is cached properly

### Training Not Converging
- Check learning rate (0.0001 is default)
- Verify sparsity coefficient (0.01)
- May need more epochs (increase from 20)

### Out of Memory
- Reduce batch sizes (see Batch Size Tuning)
- Reduce max_length from 512 to 256
- Process fewer layers per GPU

## Comparison with GPT-2 Small

| Metric | Small | Medium | Ratio |
|--------|-------|--------|-------|
| Layers | 12 | 24 | 2× |
| Hidden Dim | 768 | 1024 | 1.33× |
| Extract Batch | 32 | 16 | 0.5× |
| Train Batch | 256 | 128 | 0.5× |
| Time (Total) | ~15h | ~60h | 4× |
| Storage | ~50 GB | ~200 GB | 4× |

## Next Steps After Completion

1. **Analyze Results**
   ```bash
   python analysis/compare_models.py \
       --small ./results/gpt2-small \
       --medium ./results/gpt2-medium
   ```

2. **Generate Visualizations**
   ```bash
   python visualization/dashboard.py \
       --results ./results/gpt2-medium
   ```

3. **Export for Paper**
   ```bash
   python scripts/export_paper_results.py \
       --model gpt2-medium \
       --output ./paper/results/
   ```

## Citation

If you use this pipeline, please cite:

```bibtex
@article{ghosttrack2024,
  title={GhostTrack: Multi-Hypothesis Tracking for Interpretable Hallucination Detection},
  author={Your Name},
  journal={arXiv preprint},
  year={2024}
}
```

## Support

For issues or questions:
- Check `docs/MODEL_CONFIGS.md` for detailed configuration info
- Review SLURM logs in `*.out` and `*.err` files
- Open an issue on GitHub
