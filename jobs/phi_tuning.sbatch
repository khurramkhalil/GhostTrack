#!/bin/bash
#SBATCH --job-name=phi-tuning
#SBATCH --output=logs/phi_tuning_%j.out
#SBATCH --error=logs/phi_tuning_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:A100:3
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --account=hoquek-lab
#SBATCH --partition=hoquek-lab-gpu

# Load modules
module load miniconda3/4.10.3_gcc_9.5.0
module load cuda/11.8

# Activate environment
source activate deepseek

# Change to project directory
cd /cluster/VAST/hoquek-lab/GhostTrack

# Create directories
mkdir -p logs/phi-2
mkdir -p data/cache/phi-2
mkdir -p models/checkpoints/phi-2

echo "Starting Phi-2 Hypothesis Tracking Pipeline..."
echo "Job ID: $SLURM_JOB_ID"
echo "Date: $(date)"

# Result directories
RESULTS_DIR="./results/phi-2"
CHECKPOINT_DIR="./models/checkpoints/phi-2"
CACHE_DIR="./data/cache/phi-2"

mkdir -p $RESULTS_DIR/tracking
mkdir -p $RESULTS_DIR/detection
mkdir -p $CHECKPOINT_DIR
mkdir -p $CACHE_DIR

# echo "--- Phase 1 & 2: Extraction and SAE Training ---"
# python3 scripts/extract_and_train.py \
#     --config config/model_configs/phi-2.yaml \
#     --save-dir $CHECKPOINT_DIR \
#     --cache-dir $CACHE_DIR \
#     --num-tokens 1000000 \
#     --device cuda

echo "--- Phase 3: Hypothesis Tracking ---"
python3 scripts/run_tracking.py \
    --config config/model_configs/phi-2.yaml \
    --model-dir $CHECKPOINT_DIR \
    --output-dir $RESULTS_DIR/tracking \
    --device cuda

echo "--- Phase 4: Hallucination Detection ---"
python3 scripts/run_detection.py \
    --config config/model_configs/phi-2.yaml \
    --model-dir $CHECKPOINT_DIR \
    --tracking-dir $RESULTS_DIR/tracking \
    --output-dir $RESULTS_DIR/detection \
    --device cuda
