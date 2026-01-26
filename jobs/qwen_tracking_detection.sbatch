#!/bin/bash
#SBATCH --job-name=qwen-track
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00

# Load modules
module load miniconda3/4.10.3_gcc_9.5.0
module load cuda/11.8

# Activate environment
source activate deepseek

# Change to project directory
cd /cluster/VAST/hoquek-lab/GhostTrack

CONFIG="config/model_configs/qwen-1.5b.yaml"

echo "================================================================"
echo "Starting Qwen Tracking & Detection Phase"
echo "Config: $CONFIG"
echo "================================================================"

# 2. Run Tracking (Phase 3)
echo "[1/2] Running Tracking..."
python3 scripts/run_tracking.py \
    --config $CONFIG \
    --model-dir ./models/checkpoints/qwen \
    --output-dir ./results/qwen/tracking \
    --device cuda

# 3. Run Detection (Phase 4)
echo "[2/2] Running Detection..."
python3 scripts/run_detection.py \
    --config $CONFIG \
    --model-dir ./models/checkpoints/qwen \
    --tracking-dir ./results/qwen/tracking \
    --output-dir ./results/qwen/detection \
    --device cuda

echo "================================================================"
echo "Job Complete!"
echo "================================================================"
