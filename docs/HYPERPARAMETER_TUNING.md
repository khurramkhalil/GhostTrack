# GhostTrack Hyperparameter & Optimization Guide

## 1. Successful Baseline: GPT-2 Small (12 Layers)

These parameters achieved **AUROC 0.99** and **Accuracy 93.7%**.

### Model & SAE Config
- **Model**: `gpt2` (124M parameters)
- **SAE Expansion**: 5x (`d_model=768` → `d_hidden=3840`)
- **Sparsity**: L1 Coeff = 0.01

### Tracking Parameters
| Parameter | Value | Description |
|-----------|-------|-------------|
| `top_k_features` | **50** | Number of active SAE features tracked per layer |
| `semantic_weight` | **0.6** | Weight of cosine similarity in association |
| `activation_weight` | **0.2** | Weight of feature activation magnitude |
| `position_weight` | **0.2** | Weight of token position (locality) |
| `association_threshold` | **0.5** | Minimum score to link features between layers |
| `birth_threshold` | **0.5** | Minimum activation to start a new track |
| `death_threshold` | **0.1** | Activation below which a track dies |

### Detection Parameters
| Parameter | Value | Role |
|-----------|-------|------|
| `entropy_weight` | 0.4 | Importance of track entropy |
| `churn_weight` | 0.3 | Importance of track birth/death rate |
| `ml_weight` | 0.3 | Importance of detector probability |

---

## 2. Current State: GPT-2 Medium (24 Layers)

These parameters achieved **AUROC 0.61** (after pipeline fix).

### Model & SAE Config
- **Model**: `gpt2-medium` (355M parameters)
- **SAE Expansion**: 5x (`d_model=1024` → `d_hidden=5120`)
- **Sparsity**: L1 Coeff = 0.01

### Tracking Parameters (Identical to Small)
| Parameter | Value | Status |
|-----------|-------|--------|
| `top_k_features` | **50** | ⚠️ Likely too low for 24 layers |
| `semantic_weight` | 0.6 | Optimal for Small |
| `association_threshold` | 0.5 | Potential bottleneck |
| `birth_threshold` | 0.5 | Standard |

---

## 3. Optimization Strategy for Medium+ Models

To recover GPT-2 Small performance (>0.90 AUROC), we need to adapt parameters for deeper models.

### Why the Gap?
1. **Depth Scaling**: Metric entropy scales with $\log(\text{tracks})$. With 24 layers (vs 12), tracks have more opportunities to die or diverge.
2. **Feature Density**: Larger models represent more concepts. `top_k=50` captures a smaller fraction of the total information in a 1024-dim space vs 768-dim.
3. **Association Difficulty**: As feature spaces grow, random cosine similarities decrease. The fixed `0.5` threshold might be too aggressive.

### Proposed Tuning Grid

Run `scripts/run_tracking.py` with these variations on a 100-sample subset:

| Experiment | `top_k` | `assoc_thresh` | Rationale |
|------------|---------|----------------|-----------|
| **Baseline** | 50 | 0.5 | Current setting (AUROC 0.61) |
| **High Recall** | **100** | **0.4** | Capture more features, looser matching. |
| **High Precision** | 30 | 0.6 | Focus only on strongest, most stable features. |
| **Semantic Lean** | 50 | 0.5 | Increase `semantic_weight` to 0.8 to rely more on meaning. |

### Recommended "Best Guess" Configuration

For GPT-2 Medium and larger (Phi/Qwen), I recommend this starting point:

```yaml
tracking:
  # Increase capability to track more concurrent ideas
  top_k_features: 100
  
  # Slightly relax matching to account for higher dimensionality noise
  association_threshold: 0.45
  
  # Rely more on semantic similarity than position
  semantic_weight: 0.7
  activation_weight: 0.2
  position_weight: 0.1
```

### How to Tune Efficiently
1. Use `scripts/run_tracking.py --num-samples 100` (fast).
2. Look at `tracking_summary.json` → `avg_track_length`.
   - If < 3 layers: Decrease `association_threshold`.
   - If > 20 layers: Increase `association_threshold` (tracks are too sticky).
3. Check `detection_metrics.json` → `auroc`.
   - Goal: > 0.90.
