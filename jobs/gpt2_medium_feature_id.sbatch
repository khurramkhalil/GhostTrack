#!/bin/bash
#SBATCH --job-name=gpt2m-fid
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --account=hoquek-lab
#SBATCH --partition=hoquek-lab-gpu
#SBATCH --gres=gpu:A100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=4:00:00

# ==========================================
# GhostTrack Feature ID Tracking Experiment
# ==========================================
# Tests Feature ID matching vs semantic matching

echo "================================================================"
echo "GhostTrack GPT-2 Medium - Feature ID Tracking Experiment"
echo "Job started at $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "================================================================"

# Load environment
module load miniconda3/4.10.3_gcc_9.5.0
source activate deepseek
cd /cluster/VAST/hoquek-lab/GhostTrack

# Check GPU
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

# Base config
BASE_CONFIG="./config/model_configs/gpt2-medium.yaml"
RESULTS_BASE="./results/gpt2-medium/feature_id_exp"
NUM_SAMPLES=200

mkdir -p $RESULTS_BASE
mkdir -p ./config/feature_id_exp

# ==========================================
# EXPERIMENT 1: Feature ID Matching
# ==========================================
echo ""
echo "--- Running Experiment 1: Feature ID Matching ---"

# Create config with feature ID matching enabled
cat > ./config/feature_id_exp/feature_id.yaml << EOF
# GhostTrack Configuration for GPT-2 Medium - Feature ID Matching

project:
  name: GhostTrack
  version: 1.0.0
  description: Multi-Hypothesis Tracking for GPT-2 Medium

model:
  base_model: gpt2-medium
  d_model: 1024
  n_layers: 24
  device: cuda

sae:
  architecture: JumpReLU
  d_model: 1024
  d_hidden: 5120
  threshold: 0.1
  lambda_sparse: 0.01

paths:
  data_dir: ./data
  cache_dir: ./data/cache/gpt2-medium
  models_dir: ./models/checkpoints/gpt2-medium
  results_dir: $RESULTS_BASE/feature_id
  logs_dir: ./logs/gpt2-medium

tracking:
  top_k_features: 50
  semantic_weight: 0.6
  activation_weight: 0.2
  position_weight: 0.2
  association_threshold: 0.5
  birth_threshold: 0.5
  death_threshold: 0.1
  use_feature_id_matching: true

detection:
  entropy_threshold: 1.5
  churn_threshold: 0.3
  entropy_weight: 0.4
  churn_weight: 0.3
  ml_weight: 0.3
  threshold: 0.5

dataset:
  primary: truthful_qa
  split_train: 0.7
  split_val: 0.15
  split_test: 0.15
  stratify_by: category
EOF

python scripts/run_tracking.py \
    --config ./config/feature_id_exp/feature_id.yaml \
    --model-dir ./models/checkpoints/gpt2-medium \
    --output-dir $RESULTS_BASE/feature_id/tracking \
    --num-samples $NUM_SAMPLES

python scripts/run_detection.py \
    --config ./config/feature_id_exp/feature_id.yaml \
    --model-dir ./models/checkpoints/gpt2-medium \
    --tracking-dir $RESULTS_BASE/feature_id/tracking \
    --output-dir $RESULTS_BASE/feature_id/detection

# ==========================================
# EXPERIMENT 2: Semantic Matching (Baseline for comparison)
# ==========================================
echo ""
echo "--- Running Experiment 2: Semantic Matching (Baseline) ---"

cat > ./config/feature_id_exp/semantic.yaml << EOF
# GhostTrack Configuration for GPT-2 Medium - Semantic Matching

project:
  name: GhostTrack
  version: 1.0.0
  description: Multi-Hypothesis Tracking for GPT-2 Medium

model:
  base_model: gpt2-medium
  d_model: 1024
  n_layers: 24
  device: cuda

sae:
  architecture: JumpReLU
  d_model: 1024
  d_hidden: 5120
  threshold: 0.1
  lambda_sparse: 0.01

paths:
  data_dir: ./data
  cache_dir: ./data/cache/gpt2-medium
  models_dir: ./models/checkpoints/gpt2-medium
  results_dir: $RESULTS_BASE/semantic
  logs_dir: ./logs/gpt2-medium

tracking:
  top_k_features: 50
  semantic_weight: 0.6
  activation_weight: 0.2
  position_weight: 0.2
  association_threshold: 0.5
  birth_threshold: 0.5
  death_threshold: 0.1
  use_feature_id_matching: false

detection:
  entropy_threshold: 1.5
  churn_threshold: 0.3
  entropy_weight: 0.4
  churn_weight: 0.3
  ml_weight: 0.3
  threshold: 0.5

dataset:
  primary: truthful_qa
  split_train: 0.7
  split_val: 0.15
  split_test: 0.15
  stratify_by: category
EOF

python scripts/run_tracking.py \
    --config ./config/feature_id_exp/semantic.yaml \
    --model-dir ./models/checkpoints/gpt2-medium \
    --output-dir $RESULTS_BASE/semantic/tracking \
    --num-samples $NUM_SAMPLES

python scripts/run_detection.py \
    --config ./config/feature_id_exp/semantic.yaml \
    --model-dir ./models/checkpoints/gpt2-medium \
    --tracking-dir $RESULTS_BASE/semantic/tracking \
    --output-dir $RESULTS_BASE/semantic/detection

# ==========================================
# SUMMARY
# ==========================================
echo ""
echo "================================================================"
echo "EXPERIMENT COMPLETE"
echo "================================================================"
echo "Results summary:"

for exp in feature_id semantic; do
    echo ""
    echo "Experiment: $exp"
    if [ -f "$RESULTS_BASE/$exp/detection/detection_metrics.json" ]; then
        cat $RESULTS_BASE/$exp/detection/detection_metrics.json
    else
        echo "Metrics not found"
    fi
done

echo ""
echo "Job completed at $(date)"
