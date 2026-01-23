#!/bin/bash
#SBATCH --job-name=phi-viz
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --account=hoquek-lab
#SBATCH --partition=hoquek-lab-gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=1:00:00

# Load modules
module load miniconda3/4.10.3_gcc_9.5.0
module load cuda/11.8

# Activate environment
source activate deepseek

# Change to project directory
cd /cluster/VAST/hoquek-lab/GhostTrack

# Configuration: High Sensitivity (Winner)
# Top-k=100, Thresh=0.3, Sem=0.4

MODEL_NAME="microsoft/phi-2"
RESULT_DIR="./results/phi-2/tuning/sensitive"
CHECKPOINT_DIR="./models/checkpoints/phi-2"
OUTPUT_DIR="$RESULT_DIR/visualization"

echo "Running Visualization for Phi-2 (High Sensitivity)..."

python3 scripts/run_visualization.py \
    --model-name "$MODEL_NAME" \
    --checkpoint-dir "$CHECKPOINT_DIR" \
    --detector-path "$RESULT_DIR/detection/detector.pkl" \
    --output-dir "$OUTPUT_DIR" \
    --top-k 100 \
    --birth-threshold 0.3 \
    --semantic-weight 0.4 \
    --assoc-threshold 0.5 \
    --num-examples 10 \
    --device cuda

echo "Visualization Complete!"
