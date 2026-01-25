#!/bin/bash
#SBATCH --job-name=qwen-pipe
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --account=hoquek-lab
#SBATCH --partition=hoquek-lab-gpu
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
echo "Starting GhostTrack Pipeline for Qwen2.5-1.5B"
echo "Config: $CONFIG"
echo "================================================================"

# 1. Train SAEs (Phase 1 & 2)
# Using fewer epochs (10) for speed, as we just want to validate the pipeline
echo "[1/3] Training SAEs..."
python3 scripts/train_sae.py \
    --config $CONFIG \
    --device cuda \
    --layer-start 0 \
    --layer-end 28 \
    --epochs 10

# 2. Run Tracking (Phase 3)
echo "[2/3] Running Tracking..."
python3 scripts/run_tracking.py \
    --config $CONFIG \
    --model-dir ./models/checkpoints/qwen \
    --output-dir ./results/qwen/tracking \
    --device cuda

# 3. Run Detection (Phase 4)
echo "[3/3] Running Detection..."
python3 scripts/run_detection.py \
    --config $CONFIG \
    --model-dir ./models/checkpoints/qwen \
    --tracking-dir ./results/qwen/tracking \
    --output-dir ./results/qwen/detection \
    --device cuda

echo "================================================================"
echo "Pipeline Complete!"
echo "================================================================"
