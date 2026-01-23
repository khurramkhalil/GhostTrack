#!/bin/bash
#SBATCH --job-name=phi-sweep
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --account=hoquek-lab
#SBATCH --partition=hoquek-lab-gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=4:00:00

# Load modules
module load miniconda3/4.10.3_gcc_9.5.0
module load cuda/11.8

# Activate environment
source activate deepseek

# Change to project directory
cd /cluster/VAST/hoquek-lab/GhostTrack

# Result directories
RESULTS_DIR="./results/phi-2/tuning"
MODEL_DIR="./models/checkpoints/phi-2"

mkdir -p $RESULTS_DIR

# --- Experiment 1: Baseline (Reference) ---
echo "--- Running Experiment 1: Baseline ---"
# Check if tracking exists
if [ ! -f "$RESULTS_DIR/baseline/tracking/tracking_results.json" ]; then
    python3 scripts/run_tracking.py \
        --config config/model_configs/phi-2.yaml \
        --model-dir $MODEL_DIR \
        --output-dir $RESULTS_DIR/baseline/tracking \
        --device cuda
fi

python3 scripts/run_detection.py \
    --config config/model_configs/phi-2.yaml \
    --model-dir $MODEL_DIR \
    --tracking-dir $RESULTS_DIR/baseline/tracking \
    --output-dir $RESULTS_DIR/baseline/detection \
    --device cuda


# --- Experiment 2: High Sensitivity (More Tracks) ---
echo "--- Running Experiment 2: High Sensitivity ---"
# Custom config override via args is not supported by script yet, 
# so we create temporary config files or assume the script handles manual overrides.
# Since run_tracking.py loads from yaml, we'll create temp yamls.

cp config/model_configs/phi-2.yaml config/model_configs/phi-2-sensitive.yaml
# Use sed to modify (quick hack for cluster tuning)
# top_k=100, thresh=0.3, sem=0.4
sed -i 's/top_k_features: 50/top_k_features: 100/' config/model_configs/phi-2-sensitive.yaml
sed -i 's/birth_threshold: 0.5/birth_threshold: 0.3/' config/model_configs/phi-2-sensitive.yaml
sed -i 's/semantic_weight: 0.6/semantic_weight: 0.4/' config/model_configs/phi-2-sensitive.yaml

python3 scripts/run_tracking.py \
    --config config/model_configs/phi-2-sensitive.yaml \
    --model-dir $MODEL_DIR \
    --output-dir $RESULTS_DIR/sensitive/tracking \
    --device cuda

python3 scripts/run_detection.py \
    --config config/model_configs/phi-2-sensitive.yaml \
    --model-dir $MODEL_DIR \
    --tracking-dir $RESULTS_DIR/sensitive/tracking \
    --output-dir $RESULTS_DIR/sensitive/detection \
    --device cuda


# --- Experiment 3: High Specificity (Stricter) ---
echo "--- Running Experiment 3: High Specificity ---"
cp config/model_configs/phi-2.yaml config/model_configs/phi-2-specific.yaml
# top_k=30, thresh=0.7, sem=0.8
sed -i 's/top_k_features: 50/top_k_features: 30/' config/model_configs/phi-2-specific.yaml
sed -i 's/birth_threshold: 0.5/birth_threshold: 0.7/' config/model_configs/phi-2-specific.yaml
sed -i 's/semantic_weight: 0.6/semantic_weight: 0.8/' config/model_configs/phi-2-specific.yaml

python3 scripts/run_tracking.py \
    --config config/model_configs/phi-2-specific.yaml \
    --model-dir $MODEL_DIR \
    --output-dir $RESULTS_DIR/specific/tracking \
    --device cuda

python3 scripts/run_detection.py \
    --config config/model_configs/phi-2-specific.yaml \
    --model-dir $MODEL_DIR \
    --tracking-dir $RESULTS_DIR/specific/tracking \
    --output-dir $RESULTS_DIR/specific/detection \
    --device cuda
    
echo "Tuning Sweep Complete"
